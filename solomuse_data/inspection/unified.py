import os
import json
import csv
import logging
import numpy as np
import soundfile as sf
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm

from solomuse_data.config import PipelineConfig

logger = logging.getLogger(__name__)

class UnifiedArtifactExporter:
    """
    Produces a "Truth Table" CSV/JSON report summarizing all segment artifacts and outputs.
    """
    def __init__(self, cfg: PipelineConfig, dataset: str):
        self.cfg = cfg
        self.dataset = dataset
        self.output_root = Path(cfg.output_root)
        self.segment_root = self.output_root / "segments" / dataset
        
    def _array_metrics(self, path: Path) -> Dict[str, Any]:
        """Load npy and compute lightweight stats."""
        if not path.exists():
            return {"exists": 0}
            
        try:
            arr = np.load(path)
            is_fin = bool(np.isfinite(arr).all())
            stats = {
                "exists": 1,
                "shape": str(arr.shape),
                "is_finite": is_fin
            }
            if is_fin and arr.size > 0:
                stats.update({
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr))
                })
                # Special check for discrete tokens
                if "renderer_tokens" in path.name:
                    stats["token_min_id"] = int(np.min(arr))
                    stats["token_max_id"] = int(np.max(arr))
            return stats
        except Exception as e:
            logger.warning(f"Error reading {path}: {e}")
            return {"exists": 1, "error": str(e)}

    def _audio_metrics(self, path: Path) -> Dict[str, Any]:
        """Load wav and compute energy metrics."""
        if not path.exists():
            return {"exists": 0}
        try:
            data, sr = sf.read(path)
            if data.ndim > 1:
                # Merge channels for simple stats
                data_mono = np.mean(data, axis=1)
            else:
                data_mono = data
            
            return {
                "exists": 1,
                "rms": float(np.sqrt(np.mean(data_mono**2))),
                "peak": float(np.max(np.abs(data_mono))),
                "duration_actual_s": float(len(data) / sr)
            }
        except Exception as e:
            logger.warning(f"Error reading audio {path}: {e}")
            return {"exists": 1, "error": str(e)}

    def _intent_error_metrics(self, target_path: Path, pred_path: Path) -> Dict[str, Any]:
        """Compute MSE and Cosine Similarity between intent targets and predictions."""
        if not target_path.exists() or not pred_path.exists():
            return {}
        try:
            target = np.load(target_path)
            pred = np.load(pred_path)
            
            # Ensure same shape for metrics
            if target.shape != pred.shape:
                min_t = min(target.shape[0], pred.shape[0])
                target = target[:min_t]
                pred = pred[:min_t]
                
            mse = float(np.mean((target - pred)**2))
            
            # Cosine similarity (flattened)
            t_flat = target.flatten()
            p_flat = pred.flatten()
            denom = (np.linalg.norm(t_flat) * np.linalg.norm(p_flat))
            cos_sim = float(np.dot(t_flat, p_flat) / denom) if denom > 1e-8 else 0.0
            
            return {"intent_mse": mse, "intent_cosine_sim": cos_sim}
        except Exception as e:
            return {"intent_error": str(e)}

    def _renderer_quality_metrics(self, ground_truth_path: Path, pred_path: Path) -> Dict[str, Any]:
        """Compute SNR between ground truth and predicted solo."""
        if not ground_truth_path.exists() or not pred_path.exists():
            return {}
        try:
            y, sr_y = sf.read(ground_truth_path)
            y_hat, sr_h = sf.read(pred_path)
            
            # Simple alignment/truncation
            min_len = min(len(y), len(y_hat))
            y = y[:min_len]
            y_hat = y_hat[:min_len]
            
            noise = y - y_hat
            signal_power = np.mean(y**2)
            noise_power = np.mean(noise**2)
            
            snr = 10 * np.log10(signal_power / noise_power) if noise_power > 1e-10 else 100.0
            return {"y_hat_snr": float(snr)}
        except Exception as e:
            return {"renderer_quality_error": str(e)}

    def load_splits(self) -> Dict[str, str]:
        """Merge intent and renderer splits into a single lookup: segment_id -> split_name."""
        split_map = {}
        # We check both manifest_intent_splits.csv and manifest_renderer_splits.csv
        # Usually they should match or one might be a subset.
        for suffix in ["_intent_splits.csv", "_renderer_splits.csv"]:
            split_file = self.segment_root / f"manifest{suffix}"
            if split_file.exists():
                with open(split_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        sid = row.get("segment_id")
                        split = row.get("split", "UNKNOWN")
                        if sid:
                            split_map[sid] = split
        return split_map

    def run_export(self, 
                   action: str = "report",
                   limit: Optional[int] = None, 
                   seed: int = 42, 
                   include_missing: bool = True, 
                   output_path: Optional[str] = None,
                   rms_threshold: float = 1e-4,
                   num_samples: int = 10,
                   balanced_by_split: bool = False,
                   segment_ids_file: Optional[str] = None):
        """Main execution loop to generate report or specific diagnostics."""
        manifest_path = self.segment_root / "manifest.csv"
        if not manifest_path.exists():
            logger.error(f"Segment manifest not found at {manifest_path}")
            return
            
        # 1. Load Manifest & Splits
        with open(manifest_path, "r") as f:
            reader = csv.DictReader(f)
            all_rows = list(reader)
            
        split_map = self.load_splits()
        
        # 2. Filter Rows based on Action
        target_ids = None
        if segment_ids_file:
            with open(segment_ids_file, "r") as f:
                target_ids = {line.strip() for line in f if line.strip()}

        if action in ["coverage", "silence-audit"]:
            # Only include segments in splits (train/val/test)
            rows = [r for r in all_rows if r["segment_id"] in split_map]
        elif action == "sanity-triples":
            if target_ids:
                rows = [r for r in all_rows if r["segment_id"] in target_ids]
            else:
                # Sample balanced or random
                if balanced_by_split:
                    # Group by split
                    by_split = {}
                    for r in all_rows:
                        sp = split_map.get(r["segment_id"], "UNKNOWN")
                        if sp not in by_split: by_split[sp] = []
                        by_split[sp].append(r)
                    
                    rows = []
                    np.random.seed(seed)
                    per_split = num_samples // max(1, len(by_split))
                    for sp, split_rows in by_split.items():
                        if not split_rows: continue
                        indices = np.random.choice(len(split_rows), min(per_split, len(split_rows)), replace=False)
                        rows.extend([split_rows[i] for i in indices])
                else:
                    np.random.seed(seed)
                    indices = np.random.choice(len(all_rows), min(num_samples, len(all_rows)), replace=False)
                    rows = [all_rows[indices[i]] for i in range(len(indices))]
        else:
            # Default 'report' action
            if target_ids:
                rows = [r for r in all_rows if r["segment_id"] in target_ids]
            elif limit and limit < len(all_rows):
                np.random.seed(seed)
                indices = np.random.choice(len(all_rows), limit, replace=False)
                rows = [all_rows[i] for i in indices]
            else:
                rows = all_rows
            
        report_data = []
        logger.info(f"Running action '{action}' on {len(rows)} segments for {self.dataset}...")
        
        for row in tqdm(rows, desc=f"Unified Report [{action}]"):
            sid = row["segment_id"]
            tid = row["track_id"]
            seg_dir = self.segment_root / tid / sid
            
            # Identity
            item = {
                "dataset": self.dataset,
                "track_id": tid,
                "segment_id": sid,
                "split": split_map.get(sid, "UNKNOWN"),
                "start_s": row.get("start_s", ""),
                "end_s": row.get("end_s", ""),
                "duration_s": row.get("duration_s", "")
            }
            
            # Pre-populate keys for CSV consistency
            item.update({
                "x_rms": "", "x_peak": "", "y_rms": "", "y_peak": "",
                "intent_mse": "", "intent_cosine_sim": "",
                "has_y_hat": 0, "y_hat_path": "", "y_hat_rms": "", "y_hat_peak": "", "y_hat_snr": "",
                "has_y_hat_tokens": 0, "y_hat_tokens_path": "",
                "intent_error": "", "renderer_quality_error": "",
                "y_is_silent": 0, "renderer_target_all_zero": 0
            })

            # Deep Audio Metrics (x, y)
            x_full_path = self.output_root / row.get("x_path", "")
            y_full_path = self.output_root / row.get("y_path", "")
            
            x_stats = self._audio_metrics(x_full_path)
            y_stats = self._audio_metrics(y_full_path)
            
            if x_stats.get("exists"):
                item["x_rms"] = x_stats["rms"]
                item["x_peak"] = x_stats["peak"]
            if y_stats.get("exists"):
                item["y_rms"] = y_stats["rms"]
                item["y_peak"] = y_stats["peak"]
                item["y_is_silent"] = 1 if y_stats["rms"] < rms_threshold else 0

            # Artifact Stats
            artifacts_paths = {
                "situation": seg_dir / "situation.npy",
                "intent_targets": seg_dir / "intent_targets.npy",
                "intent_pred": seg_dir / "intent_pred.npy",
                "renderer_target": seg_dir / "renderer_target.npy",
                "renderer_tokens": seg_dir / "renderer_tokens.npy",
                "y_hat_tokens": seg_dir / "y_hat_tokens.npy"
            }
            
            for name, path in artifacts_paths.items():
                stats = self._array_metrics(path)
                item[f"has_{name}"] = stats["exists"]
                item[f"{name}_path"] = str(path.relative_to(self.output_root)) if stats["exists"] else ""
                item[f"{name}_shape"] = stats.get("shape", "")
                item[f"{name}_is_finite"] = stats.get("is_finite", "")
                item[f"{name}_min"] = stats.get("min", "")
                item[f"{name}_max"] = stats.get("max", "")
                item[f"{name}_mean"] = stats.get("mean", "")
                item[f"{name}_std"] = stats.get("std", "")
                if "token" in name:
                    item["token_min_id"] = stats.get("token_min_id", "")
                    item["token_max_id"] = stats.get("token_max_id", "")

            # Intent Error Metrics
            intent_errs = self._intent_error_metrics(artifacts_paths["intent_targets"], artifacts_paths["intent_pred"])
            item.update(intent_errs)

            # Audio output & Quality
            y_hat_path = seg_dir / "y_hat.wav"
            if y_hat_path.exists():
                item["y_hat_path"] = str(y_hat_path.relative_to(self.output_root))
                item["has_y_hat"] = 1
                y_hat_stats = self._audio_metrics(y_hat_path)
                item["y_hat_rms"] = y_hat_stats.get("rms", "")
                item["y_hat_peak"] = y_hat_stats.get("peak", "")
                
                qual = self._renderer_quality_metrics(y_full_path, y_hat_path)
                item.update(qual)
            
            # Alignment Metrics
            expected_intent = int(float(item["duration_s"]) * self.cfg.intent_hz) if item["duration_s"] else 0
            actual_intent = 0
            if item["has_intent_targets"]:
                shape_str = item["intent_targets_shape"].strip("()").split(", ")
                actual_intent = int(shape_str[0]) if len(shape_str) > 0 else 0
            
            item["expected_intent_frames"] = expected_intent
            item["actual_intent_frames"] = actual_intent
            
            alignment_ok = 1
            notes = []
            if abs(actual_intent - expected_intent) > 1 and item["has_intent_targets"]:
                alignment_ok = 0
                notes.append(f"Intent frame mismatch: {actual_intent} vs {expected_intent}")
            
            # Strict artifact existence checks
            if not item["has_situation"]:
                alignment_ok = 0
                notes.append("Missing situation.npy")
            if not item["has_intent_targets"]:
                alignment_ok = 0
                notes.append("Missing intent_targets.npy")
            if not item["has_renderer_target"]:
                alignment_ok = 0
                notes.append("Missing renderer_target.npy")
                
            if item.get("renderer_target_std", 1.0) == 0 and item["has_renderer_target"]:
                item["renderer_target_all_zero"] = 1
                notes.append("All-zero renderer target")
            else:
                item["renderer_target_all_zero"] = 0
                
            item["alignment_ok"] = alignment_ok
            item["notes"] = "; ".join(notes)
            
            report_data.append(item)

        # 3. Post-Process action specific outputs
        if action == "coverage":
            self._write_coverage_results(report_data, output_path)
        elif action == "silence-audit":
            self._write_silence_results(report_data, output_path)
        elif action == "sanity-triples":
            self._write_sanity_triples(report_data, artifacts_paths, output_path)
        else:
            self._write_base_report(report_data, output_path)
            
        return report_data

    def _write_base_report(self, data: List[Dict], output_path: Optional[str]):
        if not output_path:
            output_dir = self.output_root / "experiments" / "inspection"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_base = output_dir / "unified_pipeline_report"
        else:
            output_path = Path(output_path)
            output_base = output_path.with_suffix("")
            
        csv_path = output_base.with_suffix(".csv")
        if data:
            keys = data[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(data)
            logger.info(f"CSV Report written to: {csv_path}")
            
            json_path = output_base.with_suffix(".json")
            with open(json_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"JSON Report written to: {json_path}")

    def _write_coverage_results(self, data: List[Dict], output_path: Optional[str]):
        # data is already restricted to splits
        summary = {}
        missing_examples = {} # key -> list of ids
        
        artifact_keys = ["has_situation", "has_intent_targets", "has_intent_pred", 
                         "has_renderer_target", "has_renderer_tokens", "has_y_hat"]
        
        for row in data:
            split = row["split"]
            if split not in summary:
                summary[split] = { "total": 0 }
                for k in artifact_keys: summary[split][k] = 0
            
            s = summary[split]
            s["total"] += 1
            for k in artifact_keys:
                if row.get(k):
                    s[k] += 1
                else:
                    if k not in missing_examples: missing_examples[k] = []
                    if len(missing_examples[k]) < 5:
                        missing_examples[k].append(row["segment_id"])
        
        # Compute Percentages
        for sp, s in summary.items():
            s["percentages"] = {k: round(100 * v / s["total"], 2) for k, v in s.items() if k != "total"}
            
        out_dir = self.output_root / "experiments" / "inspection"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_base = Path(output_path).with_suffix("") if output_path else out_dir / "coverage_report"
        
        # Write JSON
        with open(output_base.with_suffix(".json"), "w") as f:
            json.dump({"summary": summary, "missing_examples": missing_examples}, f, indent=2)
            
        # Write CSV (flattened summary)
        csv_path = output_base.with_suffix(".csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["split", "total_segments"] + [f"{k}_count" for k in artifact_keys] + [f"{k}_pct" for k in artifact_keys]
            writer.writerow(header)
            for sp, s in summary.items():
                row = [sp, s["total"]] + [s[k] for k in artifact_keys] + [s["percentages"][k] for k in artifact_keys]
                writer.writerow(row)

        print("\n" + "="*60)
        print("SPLIT-ONLY COVERAGE SUMMARY")
        print("="*60)
        for sp, s in summary.items():
            print(f"Split: {sp:<8} | Segments: {s['total']}")
            for k in artifact_keys:
                pct = s["percentages"][k]
                print(f"  - {k:<20}: {pct:>6}% ({s[k]}/{s['total']})")
        
        print("\nMISSING EXAMPLES (First 5):")
        for k, ids in missing_examples.items():
            if ids:
                print(f"  - {k:<20}: {', '.join(ids)}")
        print("="*60 + "\n")
        logger.info(f"Coverage Reports written to: {output_base}.csv/json")

    def _write_silence_results(self, data: List[Dict], output_path: Optional[str]):
        out_dir = self.output_root / "experiments" / "inspection"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_base = Path(output_path).with_suffix("") if output_path else out_dir / "silence_audit"
        
        # Save full CSV/JSON
        self._write_base_report(data, str(output_base))
        
        # Aggregate by Split
        split_stats = {}
        for r in data:
            sp = r["split"]
            if sp not in split_stats: split_stats[sp] = {"total": 0, "silent": 0, "zero_renderer": 0}
            ss = split_stats[sp]
            ss["total"] += 1
            if r["y_is_silent"]: ss["silent"] += 1
            if r["renderer_target_all_zero"]: ss["zero_renderer"] += 1

        # Top-N Offenders
        silent_segments = sorted([r for r in data if r["y_is_silent"]], key=lambda x: x.get("y_rms", 0))
        zero_renderer = [r for r in data if r["renderer_target_all_zero"]]
        
        print("\n" + "="*60)
        print(f"SILENCE AUDIT SUMMARY (N={len(data)})")
        print("="*60)
        for sp, ss in split_stats.items():
            sil_pct = round(100 * ss["silent"] / ss["total"], 2)
            zero_pct = round(100 * ss["zero_renderer"] / ss["total"], 2)
            print(f"Split: {sp:<8} | Silent Y: {sil_pct:>6}% | Zero Renderer: {zero_pct:>6}%")

        print("\nTOP 20 SILENT OFFENDERS (Low RMS):")
        for r in silent_segments[:20]:
            print(f"  - {r['segment_id']} (RMS: {r['y_rms']:.6f}, split: {r['split']})")
            
        print(f"\nALL-ZERO RENDERER TARGETS (First 20):")
        for r in zero_renderer[:20]:
            print(f"  - {r['segment_id']} (split: {r['split']})")
        print("="*60 + "\n")

    def _write_sanity_triples(self, data: List[Dict], artifacts_template: Dict[str, Path], output_path: Optional[str]):
        import shutil
        out_root = Path(output_path) if output_path else self.output_root / "experiments" / "inspection" / "sanity_triples"
        if out_root.suffix == ".csv" or out_root.suffix == ".json":
            out_root = out_root.parent / "sanity_triples"
            
        out_root.mkdir(parents=True, exist_ok=True)
        
        for row in data:
            sid = row["segment_id"]
            tid = row["track_id"]
            target_dir = out_root / sid
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # 1. Gather all files
            files_to_copy = []
            # Audio
            for k in ["x_path", "y_path", "x_audio_path", "y_audio_path"]:
                if row.get(k):
                    p = self.output_root / row[k]
                    if p.exists() and p.is_file(): 
                        files_to_copy.append(p)
            
            y_hat = self.segment_root / tid / sid / "y_hat.wav"
            if y_hat.exists() and y_hat.is_file(): 
                files_to_copy.append(y_hat)
            
            # Artifacts (npy)
            seg_dir = self.segment_root / tid / sid
            for f in ["situation.npy", "intent_targets.npy", "intent_pred.npy", "renderer_target.npy"]:
                p = seg_dir / f
                if p.exists() and p.is_file(): 
                    files_to_copy.append(p)
                
            # Copy
            for f in files_to_copy:
                shutil.copy2(f, target_dir / f.name)
        
        # Write report for triples
        report_base = out_root.parent / "sanity_triples_report"
        self._write_base_report(data, str(report_base))
        
        print("\n" + "="*50)
        print(f"SANITY TRIPLES EXPORTED TO: {out_root}")
        print("="*50)
        print(f"{'SEGMENT_ID':<25} | {'SPLIT':<8} | {'I_MSE':<8} | {'SNR':<6}")
        print("-" * 55)
        for r in data[:10]:
            imse = f"{r['intent_mse']:.4f}" if r["intent_mse"] != "" else "N/A"
            snr = f"{r['y_hat_snr']:.2f}" if r['y_hat_snr'] != "" else "N/A"
            print(f"{r['segment_id']:<25} | {r['split']:<8} | {imse:<8} | {snr:<6}")
        if len(data) > 10: print("...")
        print("="*50 + "\n")
