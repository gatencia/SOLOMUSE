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
