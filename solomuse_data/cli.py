import argparse
import sys
import logging
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from typing import Optional
from solomuse_data.config import load_config, PipelineConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("solomuse")

def canonicalize(cfg: PipelineConfig, dataset: str):
    logger.info(f"Canonicalizing dataset: {dataset}")
    # Canonicalization is often implicit in build_pairs (reading via IO) or explicit.
    # We didn't implement a standalone 'canonicalize' module in the plan,
    # relying on IO/AudioOps during build_pairs.
    # We can leave this as a no-op or point to a utility if needed.
    # For now, let's log that it's handled during processing.
    logger.info("Canonicalization is handled on-the-fly during build-pairs.")

def build_pairs(cfg: PipelineConfig, dataset: str, clean: bool = False, mono: bool = False, subtype: str = "PCM_16"):
    from solomuse_data.build_pairs import build_pairs_for_dataset
    # Update config with CLI overrides
    cfg.build_mono = mono
    cfg.build_subtype = subtype
    build_pairs_for_dataset(dataset, cfg, clean=clean)

def segment(cfg: PipelineConfig, dataset: str):
    from solomuse_data.segment import segment_dataset
    segment_dataset(dataset, cfg)

def validate(cfg: PipelineConfig, dataset: str):
    from solomuse_data.validate import validate_pairs, validate_segments
    validate_pairs(cfg, dataset)
    validate_segments(cfg, dataset)

def run_all(cfg: PipelineConfig, dataset: str):
    logger.info(f"Running full pipeline for dataset: {dataset}")
    # canonicalize(cfg, dataset) # Implicit
    build_pairs(cfg, dataset)
    validate(cfg, dataset)
    segment(cfg, dataset)

def list_tracks(cfg: PipelineConfig, dataset: str):
    from solomuse_data.dataset_adapters.base import DatasetAdapter
    from solomuse_data.dataset_adapters.slakh import SlakhAdapter
    from solomuse_data.dataset_adapters.musdb import MusDBAdapter
    
    logger.info(f"Listing tracks for dataset: {dataset}")
    
    adapter = None
    if dataset == "slakh":
        adapter = SlakhAdapter(cfg)
    elif dataset == "musdb":
        adapter = MusDBAdapter(cfg)
    else:
        logger.error(f"Unknown adapter for dataset: {dataset}")
        return
        
    try:
        tracks = adapter.list_tracks()
        print(f"Found {len(tracks)} tracks in {dataset}:")
        for t in tracks[:10]:
            print(f" - {t.id} ({t.path})")
        if len(tracks) > 10:
            print(f" ... and {len(tracks)-10} more.")
    except Exception as e:
        logger.error(f"Failed to list tracks: {e}")

def main():
    parser = argparse.ArgumentParser(description="Solomuse Data Preparation Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common arguments
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parent_parser.add_argument("--dataset", type=str, required=True, help="Target dataset name (e.g., slakh, musdb)")
    parent_parser.add_argument("--num-workers", type=int, help="Number of workers override")

    # Subcommands
    subparsers.add_parser("canonicalize", parents=[parent_parser], help="Canonicalize audio files")
    build_pairs_parser = subparsers.add_parser("build-pairs", parents=[parent_parser], help="Build supervised (backing, solo) pairs")
    build_pairs_parser.add_argument("--clean", action="store_true", help="Clean output directory before starting")
    build_pairs_parser.add_argument("--mono", action="store_true", help="Downmix to mono to save space")
    build_pairs_parser.add_argument("--subtype", type=str, default="PCM_16", choices=["PCM_16", "FLOAT"], help="Audio subtype (PCM_16 is half the size of FLOAT)")
    subparsers.add_parser("segment", parents=[parent_parser], help="Segment pairs into fixed windows")
    subparsers.add_parser("validate", parents=[parent_parser], help="Validate dataset integrity")
    subparsers.add_parser("run-all", parents=[parent_parser], help="Run full pipeline")
    subparsers.add_parser("list-tracks", parents=[parent_parser], help="List tracks in dataset")
    subparsers.add_parser("generate-weak", parents=[parent_parser], help="Generate weak pairs using Demucs (requires 'weak_inputs' folder)")
    subparsers.add_parser("model-status", parents=[parent_parser], help="Print 3-layer architecture status")
    
    situation_parser = subparsers.add_parser("situation", parents=[parent_parser], help="Run Layer 1: Situation Extraction")
    situation_parser.add_argument("--limit", type=int, help="Limit number of segments to process")
    situation_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")

    intent_parser = subparsers.add_parser("intent-targets", parents=[parent_parser], help="Run Layer 2: Intent Dataset Builder")
    intent_parser.add_argument("--limit", type=int, help="Limit number of segments to process")
    intent_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")
    
    train_intent_parser = subparsers.add_parser("train-intent", parents=[parent_parser], help="Train Layer 2: Baseline Intent Planner")
    train_intent_parser.add_argument("--wandb", action="store_true", help="Enable W&B for this run")
    train_intent_parser.add_argument("--force-regenerate-splits", action="store_true", help="Force rebuild of persistent splits mapping instead of using existing one.")
    train_intent_parser.add_argument("--intent-model-type", type=str, choices=["gru", "transformer"], help="Model architecture override")
    train_intent_parser.add_argument("--intent-overfit-one-batch", action="store_true", help="Overfit on a single batch for testing")
    train_intent_parser.add_argument("--intent-epochs", type=int, help="Epochs override")
    train_intent_parser.add_argument("--intent-batch-size", type=int, help="Batch size override")
    train_intent_parser.add_argument("--intent-lr", type=float, help="Learning rate override")
    train_intent_parser.add_argument("--intent-d-model", type=int, help="Model dimension override")
    train_intent_parser.add_argument("--intent-num-layers", type=int, help="Number of layers override")
    train_intent_parser.add_argument("--intent-num-heads", type=int, help="Number of heads override")
    train_intent_parser.add_argument("--intent-ffn-dim", type=int, help="FFN dimension override")
    train_intent_parser.add_argument("--intent-dropout", type=float, help="Dropout override")

    eval_intent_parser = subparsers.add_parser("eval-intent", parents=[parent_parser], help="Evaluate Layer 2: Baseline Intent Planner")
    eval_intent_parser.add_argument("--split", type=str, default="test", help="Dataset split to evaluate (train, val, test, all)")
    eval_intent_parser.add_argument("--wandb", action="store_true", help="Enable W&B for this run")

    inspect_parser = subparsers.add_parser("inspect-intent-crashes", parents=[parent_parser], help="Inspect Layer 2 Intent bad batches payload")

    infer_intent_parser = subparsers.add_parser("infer-intent", parents=[parent_parser], help="Run inference with Intent Planner")
    infer_intent_parser.add_argument("--segment-dir", type=str, required=True, help="Path to segment directory containing situation.npy")
    infer_intent_parser.add_argument("--intent-model-type", type=str, choices=["gru", "transformer"], help="Model architecture override")

    infer_pipeline_parser = subparsers.add_parser("infer-pipeline", parents=[parent_parser], help="Unified E2E Pipeline Inference")
    infer_pipeline_parser.add_argument("--segment-dir", type=str, required=True, help="Path to segment directory containing x.wav")
    infer_pipeline_parser.add_argument("--intent-model-type", type=str, choices=["gru", "transformer"], help="Intent model override")
    infer_pipeline_parser.add_argument("--renderer-model-type", type=str, choices=["conv1d", "token_transformer"], help="Renderer model override")

    renderer_parser = subparsers.add_parser("renderer-targets", parents=[parent_parser], help="Run Layer 3: Renderer Target Builder")
    renderer_parser.add_argument("--limit", type=int, help="Limit number of segments to process")
    renderer_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")

    renderer_tokens_parser = subparsers.add_parser("renderer-token-targets", parents=[parent_parser], help="Run Layer 3: Renderer Token Dataset Builder (V2)")
    renderer_tokens_parser.add_argument("--limit", type=int, help="Limit number of segments to process")
    renderer_tokens_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")

    train_ren_parser = subparsers.add_parser("train-renderer", parents=[parent_parser], help="Train Layer 3: Baseline Renderer")
    train_ren_parser.add_argument("--force-regenerate-splits", action="store_true", help="Force rebuild of persistent splits mapping instead of using existing one.")
    
    train_ren_v2_parser = subparsers.add_parser("train-renderer-v2", parents=[parent_parser], help="Train Layer 3: Transformer Token LM Renderer (V2)")
    train_ren_v2_parser.add_argument("--renderer-batch-size", type=int, help="Batch size override")
    train_ren_v2_parser.add_argument("--renderer-epochs", type=int, help="Epochs override")
    train_ren_v2_parser.add_argument("--renderer-lr", type=float, help="Learning rate override")
    train_ren_v2_parser.add_argument("--renderer-d-model", type=int, help="Model dimension override")
    train_ren_v2_parser.add_argument("--renderer-num-layers", type=int, help="Number of layers override")
    train_ren_v2_parser.add_argument("--renderer-num-heads", type=int, help="Number of heads override")
    train_ren_v2_parser.add_argument("--renderer-ffn-dim", type=int, help="FFN dimension override")
    train_ren_v2_parser.add_argument("--renderer-dropout", type=float, help="Dropout override")
    
    render_seg_parser = subparsers.add_parser("render-segment", parents=[parent_parser], help="Render offline segment")
    render_seg_parser.add_argument("--segment-dir", type=str, required=True, help="Path to segment")
    
    # Simulate Renderer V2 (Transformer)
    parser_sim_r2 = subparsers.add_parser("infer-renderer-v2", parents=[parent_parser], help="Generate codec tokens via Transformer and write wavs")
    parser_sim_r2.add_argument("--segment-id", type=str, required=True, help="Target segment to infer over")
    
    live_sim_parser = subparsers.add_parser("live-sim", parents=[parent_parser], help="Run streaming live simulation")
    live_sim_parser.add_argument("--wav", type=str, required=True, help="Input backing track path")
    live_sim_parser.add_argument("--out", type=str, required=True, help="Output solo audio path")
    live_sim_parser.add_argument("--intent-model-type", type=str, choices=["gru", "transformer"], help="Intent model override")
    live_sim_parser.add_argument("--renderer-model-type", type=str, choices=["conv1d", "token_transformer"], help="Renderer model override")
    live_sim_parser.add_argument("--hop-s", type=float, default=0.5, help="Simulation hop size in seconds")
    live_sim_parser.add_argument("--wandb", action="store_true", help="Enable W&B tracking explicitly for this run")
    live_sim_parser.add_argument("--run-name", type=str, help="Custom W&B run name override")
    
    # Artifact inspection arguments
    inspect_artifacts_parser = subparsers.add_parser("inspect-artifacts", help="Inspect generated artifacts (e.g., segments, situations, intents)")
    inspect_subparsers = inspect_artifacts_parser.add_subparsers(dest="inspect_command")

    # Action-based inspection (old style)
    action_parser = inspect_subparsers.add_parser("action", help="Run legacy action-based inspection")
    action_parser.add_argument("--action", type=str, choices=["sample", "stats", "splits", "decode"], required=True, help="Action for inspect-artifacts command.")
    action_parser.add_argument("--sample-size", type=int, default=10, help="Number of items to sample in inspection commands.")
    action_parser.add_argument("--json-report", type=str, nargs="?", const="auto", help="Path to optionally dump json inspection report. Pass flag without value for auto-path.")
    action_parser.add_argument("--limit", type=int, help="Limit number of segments to process")
    action_parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    action_parser.add_argument("--dataset", type=str, required=True, help="Target dataset name (e.g., slakh, musdb)")

    # Summarize inspection (new style)
    summarize_parser = inspect_subparsers.add_parser("summarize", help="Summarize an existing unified report")
    summarize_parser.add_argument("--report-path", type=str, required=True, help="Path to the CSV or JSON report file to summarize")
    summarize_parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    summarize_parser.add_argument("--dataset", type=str, required=True, help="Target dataset name")
    summarize_parser.add_argument("--only-generated", action="store_true", help="Restricts rows to those with at least one existing artifact")
    summarize_parser.add_argument("--only-clean", action="store_true", help="Restricts rows to those that are 'clean' (all artifacts exist and alignment ok)")
    
    # Unified Report Export
    unified_report_parser = subparsers.add_parser("export-unified-report", parents=[parent_parser], help="Export unified truth table report for a dataset")
    unified_report_parser.add_argument("--action", type=str, choices=["report", "coverage", "silence-audit", "sanity-triples"], default="report", help="Specific diagnostic action to perform")
    unified_report_parser.add_argument("--limit", type=int, default=200, help="Limit number of segments to process")
    unified_report_parser.add_argument("--seed", type=int, default=42, help="Seed for deterministic sampling")
    unified_report_parser.add_argument("--include-missing", action="store_true", default=True, help="Include segments with missing artifacts")
    unified_report_parser.add_argument("--output", type=str, help="Output CSV/JSON path (relative to output_root or absolute)")
    
    # Silence Audit
    unified_report_parser.add_argument("--rms-threshold", type=float, default=1e-4, help="Threshold for silence detection in RMS")
    
    # Sanity Triples
    unified_report_parser.add_argument("--num-samples", type=int, default=10, help="Number of samples for sanity triples if no IDs provided")
    unified_report_parser.add_argument("--balanced-by-split", action="store_true", help="Try to sample equally from train/val/test splits")
    unified_report_parser.add_argument("--segment-ids-file", type=str, help="Path to text file with specific segment_ids to export")
    
    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        sys.exit(1)
    
    # Overrides from CLI
    if getattr(args, "num_workers", None) is not None:
        cfg.num_workers = args.num_workers

    # Dispatch command
    if args.command == "canonicalize":
        canonicalize(cfg, args.dataset)
    elif args.command == "build-pairs":
        build_pairs(
            cfg, 
            args.dataset, 
            clean=getattr(args, "clean", False),
            mono=getattr(args, "mono", False),
            subtype=getattr(args, "subtype", "PCM_16")
        )
    elif args.command == "segment":
        segment(cfg, args.dataset)
    elif args.command == "validate":
        validate(cfg, args.dataset)
    elif args.command == "run-all":
        run_all(cfg, args.dataset)
    elif args.command == "list-tracks":
        list_tracks(cfg, args.dataset)
    elif args.command == "generate-weak":
        from solomuse_data.weak_separation import generate_weak_data
        generate_weak_data(cfg)
    elif args.command == "model-status":
        model_status(cfg)
    elif args.command == "situation":
        from solomuse_model.situation.run import run_situation_extraction
        # We need a --limit and --overwrite if we want to exposed them
        # Let's check if they are in args
        limit = getattr(args, "limit", None)
        overwrite = getattr(args, "overwrite", False)
        run_situation_extraction(cfg, args.dataset, limit=limit, overwrite=overwrite)
    elif args.command == "intent-targets":
        from solomuse_model.intent.run import run_intent_target_build
        limit = getattr(args, "limit", None)
        overwrite = getattr(args, "overwrite", False)
        run_intent_target_build(cfg, args.dataset, limit=limit, overwrite=overwrite)
    elif args.command == "train-intent":
        if getattr(args, "wandb", False):
            cfg.wandb_enabled = True
        if getattr(args, "force_regenerate_splits", False):
            cfg.force_regenerate_splits = True
        if getattr(args, "intent_model_type", None):
            cfg.intent_model_type = args.intent_model_type
        if getattr(args, "intent_overfit_one_batch", False):
            cfg.intent_overfit_one_batch = True
        if getattr(args, "intent_epochs", None):
            cfg.intent_epochs = args.intent_epochs
        if getattr(args, "intent_batch_size", None):
            cfg.intent_batch_size = args.intent_batch_size
        if getattr(args, "intent_lr", None) is not None:
            cfg.intent_lr = args.intent_lr
        if getattr(args, "intent_d_model", None) is not None:
            cfg.intent_d_model = args.intent_d_model
        if getattr(args, "intent_num_layers", None) is not None:
            cfg.intent_num_layers = args.intent_num_layers
        if getattr(args, "intent_num_heads", None) is not None:
            cfg.intent_num_heads = args.intent_num_heads
        if getattr(args, "intent_ffn_dim", None) is not None:
            cfg.intent_ffn_dim = args.intent_ffn_dim
        if getattr(args, "intent_dropout", None) is not None:
            cfg.intent_dropout = args.intent_dropout
            
        from solomuse_model.intent.train import run_train_intent
        run_train_intent(cfg, args.dataset)
    elif args.command == "eval-intent":
        if getattr(args, "wandb", False):
            cfg.wandb_enabled = True
        from solomuse_model.intent.eval import run_eval_intent
        run_eval_intent(cfg, args.dataset, split=args.split)
    elif args.command == "inspect-intent-crashes":
        from solomuse_model.intent.inspect import inspect_intent_crashes
        inspect_intent_crashes(cfg, args.dataset)
    elif args.command == "inspect-artifacts":
        if args.inspect_command == "action":
            from solomuse_data.inspect_artifacts import inspect_artifacts
            limit = getattr(args, "limit", None)
            inspect_artifacts(
                cfg=cfg, 
                dataset_name=args.dataset, 
                action=args.action, 
                sample_size=args.sample_size, 
                limit=limit, 
                json_report=args.json_report
            )
        elif args.inspect_command == "summarize":
            from solomuse_data.inspection.summary import UnifiedSummaryExporter
            exporter = UnifiedSummaryExporter(cfg.output_root)
            exporter.run_summary(args.report_path)
        else:
            logger.error("Must specify sub-command for inspect-artifacts: action | summarize")
    elif args.command == "infer-intent":
        from solomuse_model.intent.infer import IntentInferencer
        import numpy as np
        from pathlib import Path
        
        if getattr(args, "intent_model_type", None):
            cfg.intent_model_type = args.intent_model_type
            
        seg_dir = Path(getattr(args, "segment_dir"))
        sit_path = seg_dir / "situation.npy"
        if not sit_path.exists():
            logger.error(f"situation.npy not found in {seg_dir}")
            return
            
        sit_vec = np.load(sit_path)
        inferencer = IntentInferencer(cfg)
        
        # Here we hardcode duration to 6.0s for the test command, 
        # normally duration_s is extracted from manifest/metadata.
        num_frames = int(6.0 * cfg.intent_hz) 
        
        preds = inferencer.predict_sequence(sit_vec, num_frames=num_frames)
        out_path = seg_dir / "intent_pred.npy"
        np.save(out_path, preds)
        logger.info(f"Saved inference prediction to {out_path} with shape {preds.shape}")
    elif args.command == "renderer-targets":
        from solomuse_model.renderer.run import run_renderer_target_build
        limit = getattr(args, "limit", None)
        overwrite = getattr(args, "overwrite", False)
        run_renderer_target_build(cfg, args.dataset, limit=limit, overwrite=overwrite)
    elif args.command == "renderer-token-targets":
        from solomuse_model.renderer.run_tokens import run_renderer_token_build
        limit = getattr(args, "limit", None)
        overwrite = getattr(args, "overwrite", False)
        run_renderer_token_build(cfg, args.dataset, limit=limit, overwrite=overwrite)
    elif args.command == "train-renderer":
        if getattr(args, "force_regenerate_splits", False):
            cfg.force_regenerate_splits = True
        from solomuse_model.renderer.train import run_train_renderer
        run_train_renderer(cfg, args.dataset)
        
    elif args.command == "train-renderer-v2":
        if getattr(args, "renderer_batch_size", None) is not None:
            cfg.renderer_batch_size = args.renderer_batch_size
        if getattr(args, "renderer_epochs", None) is not None:
            cfg.renderer_epochs = args.renderer_epochs
        if getattr(args, "renderer_lr", None) is not None:
            cfg.renderer_lr = args.renderer_lr
        if getattr(args, "renderer_d_model", None) is not None:
            cfg.renderer_d_model = args.renderer_d_model
        if getattr(args, "renderer_num_layers", None) is not None:
            cfg.renderer_num_layers = args.renderer_num_layers
        if getattr(args, "renderer_num_heads", None) is not None:
            cfg.renderer_num_heads = args.renderer_num_heads
        if getattr(args, "renderer_ffn_dim", None) is not None:
            cfg.renderer_ffn_dim = args.renderer_ffn_dim
        if getattr(args, "renderer_dropout", None) is not None:
            cfg.renderer_dropout = args.renderer_dropout
            
        from solomuse_model.renderer.train_v2 import run_train_renderer_v2
        run_train_renderer_v2(cfg, args.dataset)
        
    elif args.command == "infer-renderer-v2":
        from solomuse_model.renderer.infer_v2 import run_inference_v2
        segment_id = getattr(args, "segment_id", None)
        if not segment_id:
            logger.error("Must provide --segment-id for infer-renderer-v2 target")
            return
        run_inference_v2(cfg, args.dataset, segment_id)
        
    elif args.command == "render-segment":
        from solomuse_data.io import read_audio
        import soundfile as sf
        import numpy as np
        from pathlib import Path
        
        seg_dir = Path(getattr(args, "segment_dir"))
        x_path = seg_dir / "x.wav"
        sit_path = seg_dir / "situation.npy"
        intent_path = seg_dir / "intent_pred.npy"
        
        # fallback to intent_targets if no preds
        if not intent_path.exists():
            intent_path = seg_dir / "intent_targets.npy"
            
        x_audio, sr = read_audio(str(x_path))
        sit = np.load(sit_path) if sit_path.exists() else np.zeros(32, dtype=np.float32)
        intent = np.load(intent_path)
        
        from solomuse_model.pipeline import SoloMusePipeline
        pipeline = SoloMusePipeline(cfg)
        y_hat = pipeline.render_audio(x_audio, intent, sit)
        
        out_path = seg_dir / "y_hat.wav"
        sf.write(str(out_path), y_hat, sr)
        logger.info(f"Rendered offline segment solo to {out_path}")
    elif args.command == "infer-pipeline":
        from solomuse_model.pipeline import SoloMusePipeline
        import soundfile as sf
        from pathlib import Path
        
        if getattr(args, "intent_model_type", None):
            cfg.intent_model_type = args.intent_model_type
        if getattr(args, "renderer_model_type", None):
            cfg.renderer_model_type = args.renderer_model_type
            
        seg_dir = Path(getattr(args, "segment_dir"))
        x_path = seg_dir / "x.wav"
        if not x_path.exists():
            logger.error(f"Missing x.wav in {seg_dir}")
            return
            
        x_audio, _ = sf.read(x_path)
        pipeline = SoloMusePipeline(cfg)
        pipeline.run_pipeline_infer(x_audio, segment_id=seg_dir.name, output_dir=seg_dir)

    elif args.command == "live-sim":
        from solomuse_model.pipeline import SoloMusePipeline
        import soundfile as sf
        from pathlib import Path
        
        if getattr(args, "intent_model_type", None):
            cfg.intent_model_type = args.intent_model_type
        if getattr(args, "renderer_model_type", None):
            cfg.renderer_model_type = args.renderer_model_type
            
        wav_path = Path(getattr(args, "wav"))
        if not wav_path.exists():
            logger.error(f"Input wav {wav_path} not found")
            return
            
        x_full, sr = sf.read(wav_path)
        if sr != cfg.canonical_sample_rate:
            logger.warning(f"Resampling input from {sr} to {cfg.canonical_sample_rate}")
            # Simplified resample for sim
            import librosa
            x_full = librosa.resample(x_full, orig_sr=sr, target_sr=cfg.canonical_sample_rate)
            
        pipeline = SoloMusePipeline(cfg)
        y_out = pipeline.run_live_simulation(x_full, chunk_size_s=getattr(args, "hop_s", 0.5))
        
        out_path = Path(getattr(args, "out"))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(out_path, y_out, cfg.canonical_sample_rate)
        logger.info(f"Saved live-sim solo to {out_path}")
        
    elif args.command == "export-unified-report":
        from solomuse_data.inspection.unified import UnifiedArtifactExporter
        exporter = UnifiedArtifactExporter(cfg, args.dataset)
        exporter.run_export(
            action=getattr(args, "action", "report"),
            limit=args.limit,
            seed=args.seed,
            include_missing=args.include_missing,
            output_path=args.output,
            rms_threshold=getattr(args, "rms_threshold", 1e-4),
            num_samples=getattr(args, "num_samples", 10),
            balanced_by_split=getattr(args, "balanced_by_split", False),
            segment_ids_file=getattr(args, "segment_ids_file", None)
        )

def model_status(cfg: PipelineConfig):
    """
    Print the status of the 3-layer architecture and configuration.
    """
    print("\n" + "="*50)
    print("SOLOMUSE 3-LAYER ARCHITECTURE STATUS")
    print("="*50)
    print(f"Enabled:          {cfg.model_enable}")
    print(f"Situation Model:  {cfg.situation_model_version}")
    print(f"Intent Model:     {cfg.intent_model_version}")
    print(f"Renderer Model:   {cfg.renderer_model_version}")
    print("-" * 50)
    print("Real-time Settings:")
    print(f"  Chunk Size:     {cfg.live_chunk_ms} ms")
    print(f"  Hop Size:       {cfg.live_hop_ms} ms")
    print("-" * 50)
    print("Wiring Status:")
    
    try:
        from solomuse_model.pipeline import SoloMusePipeline
        pipeline = SoloMusePipeline(cfg)
        print("  [x] solomuse_model.pipeline.SoloMusePipeline: WIRED")
    except ImportError as e:
        print(f"  [ ] solomuse_model.pipeline: MISSING ({e})")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
