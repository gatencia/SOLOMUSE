import json
import csv
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import matplotlib.pyplot as plt

from solomuse_data.config import PipelineConfig

logger = logging.getLogger(__name__)

class ExperimentTracker:
    """
    A unified wrapper for experiment tracking (W&B) and local artifact generation.
    Fails safely if wandb is not installed unless explicitly enabled.
    """
    def __init__(self, cfg: PipelineConfig, run_dir: Path, job_type: str = "train"):
        self.cfg = cfg
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_history = []
        
        self._wandb_run = None
        self._setup_wandb(job_type)

    def _setup_wandb(self, job_type: str):
        if not self.cfg.wandb_enabled:
            return
            
        try:
            import wandb
        except ImportError:
            raise ImportError(
                "wandb_enabled=True in config, but 'wandb' package is not installed. "
                "Please run `pip install wandb` or set wandb_enabled=False."
            )

        run_name = self.cfg.wandb_run_name or f"{job_type}_{self.run_dir.name}"
        
        try:
            self._wandb_run = wandb.init(
                project=self.cfg.wandb_project,
                entity=self.cfg.wandb_entity,
                group=self.cfg.wandb_group,
                name=run_name,
                job_type=job_type,
                tags=self.cfg.wandb_tags,
                mode=self.cfg.wandb_mode,
                config=self.cfg.model_dump(),
                dir=str(self.run_dir)
            )
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            raise

    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Logs metrics locally to history and pushes to W&B if enabled."""
        if step is not None:
            metrics["epoch"] = step
            
        self.metrics_history.append(metrics)
        
        if self._wandb_run is not None:
            if step is not None and step % self.cfg.wandb_log_every_n_steps == 0:
                self._wandb_run.log(metrics, step=step)
            elif step is None:
                self._wandb_run.log(metrics)

    def log_summary(self, summary_dict: Dict[str, Any]):
        """Logs the final JSON summary and updates W&B run summary."""
        summary_path = self.run_dir / "metrics_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary_dict, f, indent=2)
            
        if self._wandb_run is not None:
            for k, v in summary_dict.items():
                self._wandb_run.summary[k] = v

    def finish(self, best_ckpt_path: Optional[Path] = None):
        """Saves final local artifacts (CSVs, plots) and closes W&B."""
        # Save CSV History
        if self.metrics_history:
            csv_path = self.run_dir / "metrics_history.csv"
            keys = self.metrics_history[-1].keys()
            with open(csv_path, "w", newline='') as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(self.metrics_history)
                
            # Generate local plots
            self._generate_training_curves()

        # Log best checkpoint as artifact to W&B
        if self._wandb_run is not None and best_ckpt_path and best_ckpt_path.exists():
            if self.cfg.wandb_save_checkpoints_as_artifacts:
                try:
                    import wandb
                    artifact = wandb.Artifact(
                        name=f"model-{self._wandb_run.id}",
                        type="model",
                        description="Best intent checkpoint"
                    )
                    artifact.add_file(str(best_ckpt_path))
                    self._wandb_run.log_artifact(artifact)
                except Exception as e:
                    logger.warning(f"Failed to log W&B artifact: {e}")

        if self._wandb_run is not None:
            self._wandb_run.finish()

    def _generate_training_curves(self):
        """Generates matplotlib curves for MSE across epochs."""
        if not self.metrics_history:
            return
            
        epochs = [m.get("epoch", i) for i, m in enumerate(self.metrics_history)]
        
        # Loss Curve
        if "train/loss" in self.metrics_history[0]:
            train_loss = [m.get("train/loss") for m in self.metrics_history]
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_loss, label="Train Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.legend()
            loss_path = self.run_dir / "train_loss_curve.png"
            plt.savefig(loss_path)
            plt.close()
            
        # MSE Curve
        if "train/mse" in self.metrics_history[0]:
            train_mse = [m.get("train/mse") for m in self.metrics_history]
            val_mse = [m.get("val/mse") for m in self.metrics_history if "val/mse" in m]
            
            plt.figure(figsize=(10, 5))
            plt.plot(epochs, train_mse, label="Train MSE")
            if len(val_mse) == len(epochs): # Ensure alignment
                plt.plot(epochs, val_mse, label="Val MSE")
            plt.xlabel("Epoch")
            plt.ylabel("MSE")
            plt.title("MSE Progression")
            plt.legend()
            mse_path = self.run_dir / "mse_curve.png"
            plt.savefig(mse_path)
            plt.close()
            
            # Optionally log image to W&B
            if self._wandb_run is not None:
                try:
                    import wandb
                    self._wandb_run.log({"plots/mse_curve": wandb.Image(str(mse_path))})
                except Exception:
                    pass

    def log_prediction_plot(self, target: np.ndarray, pred: np.ndarray, title: str, filename: str):
        """Plots a [T, D] sequence prediction vs target and saves to disk."""
        # Grab up to 3 dimensions to plot for clarity
        dims_to_plot = min(3, target.shape[1])
        
        fig, axes = plt.subplots(dims_to_plot, 1, figsize=(12, 3 * dims_to_plot), sharex=True)
        if dims_to_plot == 1:
            axes = [axes]
            
        for d in range(dims_to_plot):
            axes[d].plot(target[:, d], label="Target", alpha=0.7, color="blue")
            axes[d].plot(pred[:, d], label="Prediction", alpha=0.7, color="orange", linestyle="--")
            axes[d].set_ylabel(f"Dim {d}")
            axes[d].legend()
            
        axes[-1].set_xlabel("Time Step")
        plt.suptitle(title)
        plt.tight_layout()
        
        plot_dir = self.run_dir / "sample_predictions"
        plot_dir.mkdir(exist_ok=True)
        
        out_path = plot_dir / f"{filename}.png"
        plt.savefig(out_path)
        plt.close()
        
        # Save raw arrays
        np.save(plot_dir / f"{filename}_target.npy", target)
        np.save(plot_dir / f"{filename}_pred.npy", pred)
        
        if self._wandb_run is not None:
            try:
                import wandb
                self._wandb_run.log({f"predictions/{filename}": wandb.Image(str(out_path))})
            except Exception:
                pass
