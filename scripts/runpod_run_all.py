#!/usr/bin/env python3
"""
RunPod "One-Click" Runner Script for SoloMuse Pipeline (V2 Architecture)
Downloads data, builds artifacts, tokenizes, trains Intent + Renderer models, 
runs E2E inference, and generates the final verification report.
Resumable via .done sentinel files.
"""

import os
import sys
import argparse
import subprocess
import shutil
import time
from pathlib import Path
import json
import csv
import random

def get_logger():
    import logging
    logger = logging.getLogger("runpod_runner")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(sh)
    return logger

logger = get_logger()

def banner(msg):
    print("\n" + "="*70)
    print(f" {msg}")
    print("="*70)

def run_cmd(cmd: str, env: dict = None, dry_run: bool = False):
    logger.info(f"Running: {cmd}")
    if dry_run:
        return
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    # We use subprocess.run with check=True to fail fast on errors.
    result = subprocess.run(cmd, shell=True, env=full_env)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}:\n{cmd}")
        sys.exit(1)

def is_done(marker_dir: Path, step_name: str) -> bool:
    return (marker_dir / f".{step_name}.done").exists()

def mark_done(marker_dir: Path, step_name: str):
    marker_dir.mkdir(parents=True, exist_ok=True)
    with open(marker_dir / f".{step_name}.done", "w") as f:
        f.write(str(time.time()))

def get_disk_usage(path: Path) -> str:
    if not path.exists():
        return "0 MB"
    try:
        total_size = sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
        return f"{total_size / (1024**2):.2f} MB"
    except Exception:
        return "? MB"
        
def bootstrap_system(args):
    """STAGE 0 - Bootstrap environment"""
    banner("STAGE 0: RunPod Bootstrap")
    if not is_done(args.output_root, "stage0_bootstrap") and not args.dry_run:
        # Check torch CUDA
        try:
            import torch
            has_cuda = torch.cuda.is_available()
            logger.info(f"PyTorch CUDA Available: {has_cuda} (v{torch.version.cuda})")
            if not has_cuda:
                logger.warning("CUDA is NOT available. Training will be extremely slow!")
        except ImportError:
            logger.error("PyTorch not installed! Please run: pip install torch")
            sys.exit(1)
            
        try:
            import solomuse_data
            logger.info("SoloMuse modules successfully imported.")
        except ImportError:
            logger.info("Installing SoloMuse incrementally (pip install -e .)")
            run_cmd("pip install -e .")
            
        if os.environ.get("WANDB_API_KEY"):
            logger.info("WANDB_API_KEY detected. W&B will be enabled for training commands.")
        
        mark_done(args.output_root, "stage0_bootstrap")
    else:
        logger.info("Stage 0 already completed.")

def write_runpod_config(args) -> Path:
    """STAGE 1 - Generate Config YAML"""
    banner("STAGE 1: Generating Solomuse Config")
    config_path = args.output_root / "runpod_pipeline.yaml"
    
    if not is_done(args.output_root, "stage1_config") and not args.dry_run:
        config_yaml = f"""# Auto-generated RunPod Config
output_root: {args.output_root}
dataset_roots:
  {args.dataset}: {args.slakh_root}

canonical_sample_rate: 44100
intent_hz: 10
renderer_sample_rate: 24000
renderer_model_type: token_transformer
intent_model_type: transformer

renderer_batch_size: 4
intent_batch_size: 16

renderer_epochs: 200
intent_epochs: 100
"""
        with open(config_path, "w") as f:
            f.write(config_yaml)
        logger.info(f"Wrote config to {config_path}")
        mark_done(args.output_root, "stage1_config")
    else:
        logger.info(f"Stage 1 already completed. Config at {config_path}")
        
    return config_path

def acquire_dataset(args):
    """STAGE 2 - Acquire Slakh2100 (or specified dataset)"""
    banner(f"STAGE 2: Acquire Dataset ({args.dataset})")
    
    if args.dataset == "mock":
        logger.info("Using mock dataset, skipping acquisition.")
        return
        
    if not is_done(args.output_root, "stage2_acquire") and not args.dry_run:
        args.slakh_root.mkdir(parents=True, exist_ok=True)
        
        has_tracks = any(args.slakh_root.glob("Track*"))
        if has_tracks:
            logger.info(f"Found existing tracks in {args.slakh_root}, skipping download/extract.")
        else:
            if args.slakh_archive and Path(args.slakh_archive).exists():
                logger.info(f"Extracting user archive {args.slakh_archive} to {args.slakh_root}")
                run_cmd(f"unzip -q {args.slakh_archive} -d {args.slakh_root}")
            elif args.slakh_url:
                zip_path = args.workspace_root / "dataset_download.zip"
                logger.info(f"Downloading from {args.slakh_url} to {zip_path}")
                run_cmd(f"wget -qO- \"{args.slakh_url}\" > {zip_path}")
                logger.info("Extracting...")
                run_cmd(f"unzip -q {zip_path} -d {args.slakh_root}")
            else:
                logger.error(f"Slakh directory {args.slakh_root} is empty!")
                logger.error("Please provide --slakh-archive or --slakh-url, or upload manually.")
                sys.exit(1)
                
        num_tracks = len(list(args.slakh_root.glob("Track*")))
        logger.info(f"Acquisition complete. Found {num_tracks} tracks. Directory size: {get_disk_usage(args.slakh_root)}")
        mark_done(args.output_root, "stage2_acquire")
    else:
        logger.info("Stage 2 already completed.")

def build_dataset_artifacts(args, config_path):
    """STAGE 3 - Dataset Preparation"""
    banner("STAGE 3: Build Dataset Artifacts")
    base_cmd = f"python -m solomuse_data.cli"
    if args.dataset == "mock":
        cmd_env = {"SOLOMUSE_FORCE_CPU": "1"} # Mock is often CPU bound safely
    else:
        cmd_env = {}
    
    steps = [
        ("build-pairs", f"{base_cmd} build-pairs --config {config_path} --dataset {args.dataset}"),
        ("segment", f"{base_cmd} segment --config {config_path} --dataset {args.dataset}"),
        ("situation", f"{base_cmd} situation --config {config_path} --dataset {args.dataset}" + (f" --limit {args.limit}" if args.limit else "")),
        ("intent-targets", f"{base_cmd} intent-targets --config {config_path} --dataset {args.dataset}" + (f" --limit {args.limit}" if args.limit else ""))
    ]
    
    for step_name, cmd in steps:
        if not is_done(args.output_root, f"stage3_{step_name}"):
            logger.info(f"Running '{step_name}' pipeline step...")
            run_cmd(cmd, env=cmd_env, dry_run=args.dry_run)
            if not args.dry_run:
                mark_done(args.output_root, f"stage3_{step_name}")
        else:
            logger.info(f"Step '{step_name}' already completed.")

def tokenize_targets(args, config_path):
    """STAGE 4 - Extract EnCodec Tokens"""
    banner("STAGE 4: Tokenize Renderer Targets (EnCodec)")
    if not is_done(args.output_root, "stage4_tokenize"):
        cmd = f"python -m solomuse_data.cli renderer-token-targets --config {config_path} --dataset {args.dataset}"
        if args.limit:
            cmd += f" --limit {args.limit}"
        
        logger.info("Running renderer token extraction (requires GPU ideally)...")
        run_cmd(cmd, dry_run=args.dry_run)
        
        if not args.dry_run:
            token_dir = args.output_root / "segments" / args.dataset
            token_files = list(token_dir.rglob("x_tokens.npy"))
            logger.info(f"Tokenization complete. Found {len(token_files)} x_tokens.npy files. Size: {get_disk_usage(token_dir)}")
            mark_done(args.output_root, "stage4_tokenize")
    else:
        logger.info("Stage 4 already completed.")

def train_models(args, config_path):
    """STAGE 5 - Deep Learning Training"""
    banner("STAGE 5: Train Models (Intent + Token Renderer)")
    
    wandb_flag = "--wandb" if (args.use_wandb or "WANDB_API_KEY" in os.environ) else ""
    
    # 1. Train Intent Model
    if not is_done(args.output_root, "stage5_train_intent"):
        logger.info("Training Layer 2: Transformer Intent Planner...")
        cmd_intent = f"python -m solomuse_data.cli train-intent --config {config_path} --dataset {args.dataset} --intent-model-type transformer {wandb_flag}"
        run_cmd(cmd_intent, dry_run=args.dry_run)
        if not args.dry_run:
            mark_done(args.output_root, "stage5_train_intent")
    else:
        logger.info("Intent Training already completed.")
        
    # 2. Train Renderer Model
    if not is_done(args.output_root, "stage5_train_renderer"):
        logger.info("Training Layer 3: Transformer Token Renderer...")
        cmd_ren = f"python -m solomuse_data.cli train-renderer-v2 --config {config_path} --dataset {args.dataset} {wandb_flag}"
        run_cmd(cmd_ren, dry_run=args.dry_run)
        if not args.dry_run:
            mark_done(args.output_root, "stage5_train_renderer")
    else:
        logger.info("Renderer Training already completed.")

def verify_pipeline(args, config_path):
    """STAGE 6 - Verification Suite"""
    banner(f"STAGE 6: End-to-End Verification (N={args.verify_n})")
    
    if args.dry_run:
        logger.info("Dry-run verification skipped.")
        return
        
    verification_dir = args.output_root / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Choose segments
    manifest_path = args.output_root / "segments" / args.dataset / "manifest_intent_splits.csv"
    if not manifest_path.exists():
        logger.error(f"Cannot find manifest at {manifest_path} for verification.")
        sys.exit(1)
        
    test_segments = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
             test_segments.append((row['segment_id'], row['track_id']))
             
    if not test_segments:
        logger.error("No segments found in manifest for verification.")
        sys.exit(1)
        
    random.seed(42)
    sample_segs = random.sample(test_segments, min(args.verify_n, len(test_segments)))
    
    ids_file = verification_dir / "sanity_segment_ids.txt"
    with open(ids_file, "w") as f:
        for seg_id, _ in sample_segs:
            f.write(f"{seg_id}\\n")
            
    # 2. Run Inference
    seg_root = args.output_root / "segments" / args.dataset
    for i, (seg_id, track_id) in enumerate(sample_segs, 1):
        seg_dir = seg_root / track_id / seg_id
        cmd = f"python -m solomuse_data.cli infer-pipeline --config {config_path} --dataset {args.dataset} --segment-dir {seg_dir}"
        logger.info(f"Verifying ({i}/{len(sample_segs)}): {seg_id}")
        run_cmd(cmd)
        
    # 3. Generate Report restricted to these IDs
    report_csv = verification_dir / "v2_slakh_infer_sanity.csv"
    cmd_report = f"python -m solomuse_data.cli export-unified-report --config {config_path} --dataset {args.dataset} --action sanity-triples --segment-ids-file {ids_file} --output {report_csv}"
    run_cmd(cmd_report)
    
    # 4. Summarize Report
    cmd_sum = f"python -m solomuse_data.cli inspect-artifacts summarize --config {config_path} --dataset {args.dataset} --report-path {report_csv} --only-generated"
    run_cmd(cmd_sum)
    
    # 5. Assert Pass Criteria
    with open(report_csv, 'r') as f:
        reader = list(csv.DictReader(f))
        
    clean_count = 0
    y_hat_count = 0
    silent_count = 0
    
    for row in reader:
        if row.get("has_y_hat") == "1":
            y_hat_count += 1
        if row.get("y_is_silent") == "1":
            silent_count += 1
            
        # Basic clean check: has y_hat, has intent seq
        if row.get("has_y_hat") == "1" and row.get("has_intent_pred") == "1" and row.get("alignment_ok") == "1":
            clean_count += 1
            
    logger.info(f"Verification Results: Generated {y_hat_count}/{len(sample_segs)} audio files. {silent_count} silent tracks. {clean_count} perfectly clean generations.")
    
    if y_hat_count < len(sample_segs):
        logger.error(f"VERIFICATION FAILED: Expected {len(sample_segs)} generated audios, got {y_hat_count}.")
        sys.exit(1)
    
    mark_done(args.output_root, "stage6_verification")

def package_outputs(args, config_path):
    """STAGE 7 - Packaging"""
    banner("STAGE 7: Final Packaging")
    logger.info("RUN COMPLETE!")
    logger.info("="*70)
    logger.info(f"Persistent Output Root : {args.output_root}")
    logger.info(f"Intent Checkpoints     : {args.output_root}/models/{args.dataset}/intent_transformer/")
    logger.info(f"Renderer Checkpoints   : {args.output_root}/models/{args.dataset}/renderer_token_transformer/")
    logger.info(f"Verification Report    : {args.output_root}/verification/")
    logger.info("="*70)
    logger.info("To run inference on a new segment, use the following command:")
    logger.info(f"python -m solomuse_data.cli infer-pipeline --config {config_path} --dataset {args.dataset} --segment-dir </path/to/segment>")

def main():
    # Environment Detection: Avoid hardcoded /workspace on local machines
    workspace_exists = Path("/workspace").exists()
    default_workspace = "/workspace" if workspace_exists else str(Path.cwd() / "workspace")
    
    persist_volume = Path("/runpod-volume")
    if persist_volume.exists():
        default_persist = "/runpod-volume"
    elif workspace_exists:
        default_persist = "/workspace"
    else:
        # Local development fallback
        default_persist = str(Path.cwd() / "runpod_results")

    parser = argparse.ArgumentParser(description="SoloMuse RunPod Single-Click Pipeline Runner")
    parser.add_argument("--run-name", type=str, default=f"solomuse_run_{int(time.time())}", help="Unique run identifier")
    parser.add_argument("--dataset", type=str, default="slakh", help="Dataset slug (slakh, mock)")
    parser.add_argument("--workspace-root", type=str, default=default_workspace, help="Ephemeral workspace dir")
    parser.add_argument("--persistent-root", type=str, default=default_persist, help="Persistent storage volume")
    
    parser.add_argument("--slakh-archive", type=str, help="Path to local slakh zip/tar file")
    parser.add_argument("--slakh-url", type=str, help="Direct download URL for Slakh archive")
    
    parser.add_argument("--limit", type=int, help="Limit number of segments processed (for fast testing)")
    parser.add_argument("--verify-n", type=int, default=10, help="Number of segments for E2E verification")
    parser.add_argument("--use-wandb", action="store_true", help="Explicitly enable Weights & Biases")
    parser.add_argument("--dry-run", action="store_true", help="Print commands instead of running them")
    
    args = parser.parse_args()
    
    # Setup Paths
    args.persistent_root = Path(args.persistent_root)
    args.workspace_root = Path(args.workspace_root)
    
    args.output_root = args.persistent_root / "SOLOMUSE_RUNS" / args.run_name
    
    try:
        args.output_root.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.error(f"Failed to create output directory at {args.output_root}")
        logger.error(f"Error: {e}")
        logger.error("If running locally, ensure the path is writable or provide --persistent-root.")
        sys.exit(1)
    
    args.slakh_root = args.persistent_root / "datasets" / "slakh2100"
    
    # Execute Pipeline
    bootstrap_system(args)
    config_path = write_runpod_config(args)
    acquire_dataset(args)
    build_dataset_artifacts(args, config_path)
    tokenize_targets(args, config_path)
    train_models(args, config_path)
    verify_pipeline(args, config_path)
    package_outputs(args, config_path)

if __name__ == "__main__":
    main()
