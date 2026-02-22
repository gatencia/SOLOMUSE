import argparse
import sys
import logging
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

def build_pairs(cfg: PipelineConfig, dataset: str):
    from solomuse_data.build_pairs import build_pairs_for_dataset
    build_pairs_for_dataset(dataset, cfg)

def segment(cfg: PipelineConfig, dataset: str):
    from solomuse_data.segment import segment_dataset
    segment_dataset(dataset, cfg)

def validate(cfg: PipelineConfig, dataset: str):
    from solomuse_data.validate import validate_pairs, validate_segments
    validate_pairs(cfg)
    validate_segments(cfg)

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

    # Subcommands
    subparsers.add_parser("canonicalize", parents=[parent_parser], help="Canonicalize audio files")
    subparsers.add_parser("build-pairs", parents=[parent_parser], help="Build supervised (backing, solo) pairs")
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

    infer_intent_parser = subparsers.add_parser("infer-intent", parents=[parent_parser], help="Run inference with Intent Planner")
    infer_intent_parser.add_argument("--segment-dir", type=str, required=True, help="Path to segment directory containing situation.npy")

    renderer_parser = subparsers.add_parser("renderer-targets", parents=[parent_parser], help="Run Layer 3: Renderer Target Builder")
    renderer_parser.add_argument("--limit", type=int, help="Limit number of segments to process")
    renderer_parser.add_argument("--overwrite", action="store_true", help="Overwrite existing artifacts")

    train_ren_parser = subparsers.add_parser("train-renderer", parents=[parent_parser], help="Train Layer 3: Baseline Renderer")
    
    render_seg_parser = subparsers.add_parser("render-segment", parents=[parent_parser], help="Render offline segment")
    render_seg_parser.add_argument("--segment-dir", type=str, required=True, help="Path to segment")
    
    live_sim_parser = subparsers.add_parser("live-sim", parents=[parent_parser], help="Run streaming live simulation")
    live_sim_parser.add_argument("--wav", type=str, required=True, help="Input backing track path")
    live_sim_parser.add_argument("--out", type=str, required=True, help="Output solo audio path")

    args = parser.parse_args()

    try:
        cfg = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        sys.exit(1)

    # Dispatch command
    if args.command == "canonicalize":
        canonicalize(cfg, args.dataset)
    elif args.command == "build-pairs":
        build_pairs(cfg, args.dataset)
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
        from solomuse_model.intent.train import run_train_intent
        run_train_intent(cfg, args.dataset)
    elif args.command == "infer-intent":
        from solomuse_model.intent.infer import IntentInferencer
        import numpy as np
        from pathlib import Path
        
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
    elif args.command == "train-renderer":
        from solomuse_model.renderer.train import run_train_renderer
        run_train_renderer(cfg, args.dataset)
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
    elif args.command == "live-sim":
        from solomuse_data.io import read_audio
        import soundfile as sf
        
        x_path = getattr(args, "wav")
        out_path = getattr(args, "out")
        
        x_audio, sr = read_audio(x_path)
        
        from solomuse_model.pipeline import SoloMusePipeline
        pipeline = SoloMusePipeline(cfg)
        y_out = pipeline.run_live_simulation(x_audio, chunk_size_s=1.0)
        
        sf.write(out_path, y_out, sr)
        logger.info(f"Live Simulation computed! Out: {out_path}")

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
