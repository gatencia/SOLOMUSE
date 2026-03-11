#!/usr/bin/env python3
"""
SoloMuse "One-Click" Autonomous RunPod Runner (V2 Architecture)
Handles everything from a blank pod:
1. System dependency install (apt)
2. Repository cloning & environment setup (venv)
3. Resumable dataset acquisition (aria2c/wget)
4. End-to-end SoloMuse V2 Pipeline (Artifacts -> Training -> Verification)
5. Final report bundling and packaging.

Resumable via .done sentinel files.
Logs to both stdout and persistent log files.
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
import logging

def get_logger(log_file: Path = None):
    logger = logging.getLogger("runpod_runner")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Console handler
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(sh)
        # File handler (if provided)
        if log_file:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s"))
            logger.addHandler(fh)
    return logger

# Initial logger, will be updated with file handler after path resolution
logger = get_logger()

def banner(msg):
    print("\n" + "="*70)
    print(f" {msg}")
    print("="*70)

def run_cmd(cmd: str, env: dict = None, dry_run: bool = False, cwd: Path = None, shell: bool = True):
    logger.info(f"Running: {cmd}")
    if dry_run:
        return
    full_env = os.environ.copy()
    if env:
        full_env.update(env)
    
    result = subprocess.run(cmd, shell=shell, env=full_env, cwd=cwd)
    if result.returncode != 0:
        logger.error(f"Command failed with exit code {result.returncode}:\n{cmd}")
        sys.exit(1)

def is_done(marker_dir: Path, step_name: str) -> bool:
    return (marker_dir / f".{step_name}.done").exists()

def mark_done(marker_dir: Path, step_name: str):
    marker_dir.mkdir(parents=True, exist_ok=True)
    with open(marker_dir / f".{step_name}.done", "w") as f:
        f.write(str(time.time()))

def get_disk_free(path: Path) -> float:
    """Returns free space in GB"""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024**3)
    except Exception:
        return 0.0

def retry_wrapper(func, retries=3, backoff=2):
    """Exponential backoff retry decorator"""
    def wrapper(*args, **kwargs):
        attempts = 0
        while attempts < retries:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                attempts += 1
                if attempts == retries:
                    raise e
                wait = backoff ** attempts
                logger.warning(f"Operation failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
    return wrapper

def robust_download(url: str, dest_path: Path, dry_run: bool = False):
    """Resumable download using aria2c if available, else wget -c"""
    if dry_run: return
    
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try aria2c first (much faster, better resuming)
    if shutil.which("aria2c"):
        cmd = f"aria2c -x 16 -s 16 -c --dir='{dest_path.parent}' --out='{dest_path.name}' '{url}'"
    else:
        cmd = f"wget -c -O '{dest_path}' '{url}'"
    
    run_cmd(cmd)

@retry_wrapper
def bootstrap_system(args):
    """STAGE -1: Install system dependencies"""
    banner("STAGE -1: System Bootstrap (Apt)")
    if not is_done(args.output_root, "stage_neg1_sys_bootstrap") and not args.dry_run:
        # Check if we are on a debian-based system
        if shutil.which("apt-get"):
            logger.info("Installing system packages: git, ffmpeg, sox, aria2, unzip, libsndfile1...")
            run_cmd("apt-get update && apt-get install -y git ffmpeg sox aria2 unzip wget curl libsndfile1-dev")
        else:
            logger.warning("apt-get not found. Skipping system package install. Assuming pre-installed.")
        
        # Verify NVIDIA
        if shutil.which("nvidia-smi"):
            run_cmd("nvidia-smi")
        else:
            logger.warning("nvidia-smi not found. Training will be extremely slow (CPU only).")
            
        mark_done(args.output_root, "stage_neg1_sys_bootstrap")
    else:
        logger.info("Stage -1 already completed.")

def clone_repo(args):
    """STAGE -0: Clone Repository"""
    banner(f"STAGE -0: Clone Repository ({args.repo_branch})")
    
    if not is_done(args.output_root, "stage_neg0_clone") and not args.dry_run:
        repo_dir = args.persistent_root / "repos" / "SOLOMUSE"
        if repo_dir.exists():
            logger.info(f"Repository already exists at {repo_dir}. Updating...")
            run_cmd(f"git fetch origin && git checkout {args.repo_branch} && git pull origin {args.repo_branch}", cwd=repo_dir)
        else:
            repo_dir.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Cloning {args.repo_url} into {repo_dir}")
            run_cmd(f"git clone -b {args.repo_branch} {args.repo_url} {repo_dir}")
            
        mark_done(args.output_root, "stage_neg0_clone")
    else:
        logger.info("Stage -0 already completed.")
    
    # Return the absolute path to the repo root
    return args.persistent_root / "repos" / "SOLOMUSE"

def setup_python(args, repo_root: Path):
    """STAGE 0: Environment Setup (venv)"""
    banner("STAGE 0: Python Environment Setup")
    venv_dir = args.persistent_root / "venvs" / "solomuse"
    python_bin = venv_dir / "bin" / "python"
    
    if not is_done(args.output_root, "stage0_env_setup") and not args.dry_run:
        # 0.1 Create Venv
        if not is_done(args.output_root, "stage0_1_venv") and not venv_dir.exists():
            logger.info(f"Creating virtual environment at {venv_dir}")
            run_cmd(f"python3 -m venv {venv_dir}")
            mark_done(args.output_root, "stage0_1_venv")
        
        # 0.2 Pip Upgrade
        if not is_done(args.output_root, "stage0_2_pip_upgrade"):
            logger.info("Upgrading pip...")
            run_cmd(f"{python_bin} -m pip install --upgrade pip")
            mark_done(args.output_root, "stage0_2_pip_upgrade")
            
        # 0.3 Install Torch
        if not is_done(args.output_root, "stage0_3_torch"):
            # Install Torch + CUDA explicitly (RunPod optimized)
            # We use cu121 as a safe modern default for RunPod environments
            torch_install_cmd = f"{python_bin} -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121"
            logger.info(f"Installing PyTorch with CUDA: {torch_install_cmd}")
            run_cmd(torch_install_cmd)
            mark_done(args.output_root, "stage0_3_torch")
            
        # 0.4 Install Common Modeling Deps
        if not is_done(args.output_root, "stage0_4_modeling_deps"):
            logger.info("Installing modeling dependencies (wandb, matplotlib, tqdm)...")
            run_cmd(f"{python_bin} -m pip install wandb matplotlib tqdm")
            mark_done(args.output_root, "stage0_4_modeling_deps")
            
        # 0.5 Install Project Editable
        if not is_done(args.output_root, "stage0_5_editable"):
            logger.info("Installing project in editable mode (grabbing remaining deps)...")
            run_cmd(f"{python_bin} -m pip install -e .", cwd=repo_root)
            mark_done(args.output_root, "stage0_5_editable")
        
        # 0.6 Verification
        if not is_done(args.output_root, "stage0_6_verification"):
            logger.info("Verifying environment...")
            verify_script = f"""
import torch
import sys
print(f'Python: {{sys.version}}')
print(f'Torch: {{torch.__version__}}')
print(f'CUDA Available: {{torch.cuda.is_available()}}')
if torch.cuda.is_available():
    print(f'GPU: {{torch.cuda.get_device_name(0)}}')
else:
    print('WARNING: CUDA NOT DETECTED')
"""
            run_cmd(f"{python_bin} -c \"{verify_script}\"")
            
            # Safety check for vendored torch
            safety_script = """
import sys
import torch
path = torch.__file__
if 'vendored_deps' in path:
    print(f'CRITICAL ERROR: Torch is importing from vendored location: {path}')
    print('Please remove it and ensure pip-installed torch is used.')
    sys.exit(1)
"""
            run_cmd(f"{python_bin} -c \"{safety_script}\"")
            mark_done(args.output_root, "stage0_6_verification")
            
        mark_done(args.output_root, "stage0_env_setup")
    else:
        logger.info("Stage 0 already completed.")
        
    return python_bin

def write_pipeline_config(args, config_path: Path):
    """STAGE 1 - Generate Config YAML"""
    banner("STAGE 1: Generating Solomuse Config")
    
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

# --- Intent V2 Hyperparameters ---
intent_d_model: 256
intent_num_layers: 6
intent_num_heads: 8
intent_ffn_dim: 1024
intent_lr: 3e-4
intent_weight_decay: 1e-2
intent_batch_size: {args.intent_batch_size}
intent_epochs: {args.intent_epochs}
intent_warmup_steps: 2000
intent_lr_schedule: cosine

# --- Renderer V2 Hyperparameters ---
renderer_d_model: 768
renderer_num_layers: 12
renderer_num_heads: 12
renderer_ffn_dim: 3072
renderer_lr: 2e-4
renderer_weight_decay: 1e-2
renderer_batch_size: {args.renderer_batch_size}
renderer_epochs: {args.renderer_epochs}
renderer_warmup_steps: 5000
renderer_lr_schedule: cosine

# --- Sampling Defaults ---
inference_temperature: 0.9
inference_top_k: 50
inference_top_p: 0.95
"""
        with open(config_path, "w") as f:
            f.write(config_yaml)
        logger.info(f"Wrote config to {config_path}")
        mark_done(args.output_root, "stage1_config")
    else:
        logger.info(f"Stage 1 already completed. Config at {config_path}")

def acquire_dataset(args):
    """STAGE 2 - Acquire Slakh2100"""
    banner(f"STAGE 2: Acquire Dataset ({args.dataset})")
    
    if args.dataset == "mock":
        logger.info("Using mock dataset, skipping acquisition.")
        return
        
    if not is_done(args.output_root, "stage2_acquire") and not args.dry_run:
        args.slakh_root.mkdir(parents=True, exist_ok=True)
        
        has_tracks = any(args.slakh_root.glob("Track*"))
        if has_tracks:
            logger.info(f"Found existing tracks in {args.slakh_root}, skipping download.")
        else:
            # Check Disk Space (Slakh2100 is ~100GB extracted)
            free_gb = get_disk_free(args.slakh_root)
            logger.info(f"Free space on {args.slakh_root.parent}: {free_gb:.2f} GB")
            if free_gb < 150: # Safety margin
                logger.error(f"Insufficient disk space! Need ~150GB, have {free_gb:.2f}GB.")
                sys.exit(1)

            if args.slakh_archive and Path(args.slakh_archive).exists():
                logger.info(f"Using local archive: {args.slakh_archive}")
                archive_path = Path(args.slakh_archive)
                if archive_path.suffix == ".zip":
                    run_cmd(f"unzip -q {archive_path} -d {args.slakh_root}")
                else:
                    run_cmd(f"tar --no-same-owner -xf {archive_path} -C {args.slakh_root} --strip-components=1")
            elif args.slakh_url:
                archive_name = "slakh_dataset.tar.gz" if ".tar.gz" in args.slakh_url else "slakh_dataset.zip"
                # Use persistent_root for the archive as well to leverage network volumes
                archive_path = args.persistent_root / archive_name
                logger.info(f"Downloading dataset from {args.slakh_url} to {archive_path}")
                robust_download(args.slakh_url, archive_path)
                logger.info("Extracting dataset...")
                try:
                    if ".tar.gz" in args.slakh_url:
                        run_cmd(f"tar --no-same-owner -xf {archive_path} -C {args.slakh_root} --strip-components=1")
                    else:
                        run_cmd(f"unzip -q {archive_path} -d {args.slakh_root}")
                except Exception as e:
                    logger.error(f"Extraction failed. Your disk might be full. Note: The 100GB dataset needs ~200GB of total free space during extraction (100GB archive + 100GB extracted files).")
                    logger.info("If you want to free up space and try again, you can manually delete the archive and run with the Tiny BabySlakh dataset.")
                    raise e
                    
                logger.info("Cleaning up archive to save space...")
                if archive_path.exists():
                    archive_path.unlink()
            else:
                logger.error("=" * 60)
                logger.error(" DATASET ACQUISITION FAILED")
                logger.error("-" * 60)
                logger.error(" No dataset found and no download source specified.")
                logger.error(" If you hit a 'Disk quota exceeded' error, it means your pod's disk is too small.")
                logger.error(" To fix this, you have two options:")
                logger.error(" 1. Use the 'Tiny' BabySlakh (0.3GB - recommended for verification):")
                logger.error("    python scripts/runpod_run_all.py --slakh-url https://zenodo.org/records/4603844/files/babyslakh_16k.zip?download=1")
                logger.error(" 2. Run with the official Slakh2100 Redux URL (100GB - requires large disk/volume):")
                logger.error("    python scripts/runpod_run_all.py --slakh-url https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1")
                logger.error(" 3. Run a smoke test with mock data (fastest):")
                logger.error("    python scripts/runpod_run_all.py --smoke-test")
                logger.error("=" * 60)
                sys.exit(1)
                
        num_tracks = len(list(args.slakh_root.glob("Track*")))
        logger.info(f"Acquisition complete. Tracks: {num_tracks}")
        mark_done(args.output_root, "stage2_acquire")
    else:
        logger.info("Stage 2 already completed.")

def run_pipeline_step(args, python_bin: Path, config_path: Path, step_name: str, cmd: str):
    """Helper to run a pipeline command using the venv python"""
    if not is_done(args.output_root, step_name):
        logger.info(f"Running step: {step_name}")
        # Explicitly set PYTHONPATH to the repository root so solomuse_model can be imported
        repo_root = args.persistent_root / "repos" / "SOLOMUSE"
        full_cmd = f"PYTHONPATH={repo_root} {python_bin} -m {cmd}"
        run_cmd(full_cmd, dry_run=args.dry_run)
        if not args.dry_run:
            mark_done(args.output_root, step_name)
    else:
        logger.info(f"Step {step_name} already completed.")

def build_artifacts(args, python_bin: Path, config_path: Path):
    """STAGE 3 - Pipeline Artifact Build"""
    banner("STAGE 3: Build Dataset Artifacts")
    base_cmd = f"solomuse_data.cli"
    limit_suffix = f" --limit {args.limit}" if args.limit else ""
    
    steps = [
        ("stage3_build_pairs", f"{base_cmd} build-pairs --config {config_path} --dataset {args.dataset}"),
        ("stage3_segment", f"{base_cmd} segment --config {config_path} --dataset {args.dataset}"),
        ("stage3_situation", f"{base_cmd} situation --config {config_path} --dataset {args.dataset}{limit_suffix}"),
        ("stage3_intent_targets", f"{base_cmd} intent-targets --config {config_path} --dataset {args.dataset}{limit_suffix}"),
    ]
    
    for marker, cmd in steps:
        run_pipeline_step(args, python_bin, config_path, marker, cmd)

def tokenize(args, python_bin: Path, config_path: Path):
    """STAGE 4 - EnCodec Tokenization"""
    banner("STAGE 4: Tokenize Renderer Targets")
    limit_suffix = f" --limit {args.limit}" if args.limit else ""
    cmd = f"solomuse_data.cli renderer-token-targets --config {config_path} --dataset {args.dataset}{limit_suffix}"
    run_pipeline_step(args, python_bin, config_path, "stage4_tokenize", cmd)

def train(args, python_bin: Path, config_path: Path):
    """STAGE 5 - Training"""
    banner("STAGE 5: Train Models (Intent + Renderer)")
    wandb_flag = "--wandb" if (args.use_wandb or "WANDB_API_KEY" in os.environ) else ""
    
    # Intent
    cmd_intent = f"solomuse_data.cli train-intent --config {config_path} --dataset {args.dataset} {wandb_flag}"
    run_pipeline_step(args, python_bin, config_path, "stage5_train_intent", cmd_intent)
    
    # Renderer
    cmd_ren = f"solomuse_data.cli train-renderer-v2 --config {config_path} --dataset {args.dataset}"
    run_pipeline_step(args, python_bin, config_path, "stage5_train_renderer", cmd_ren)

def verify(args, python_bin: Path, config_path: Path):
    """STAGE 6 - Verification"""
    banner("STAGE 6: Verification & End-to-End Test")
    if args.dry_run: return
    
    # 1. Random Sample Verification
    verification_dir = args.output_root / "verification"
    verification_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_path = args.output_root / "segments" / args.dataset / "manifest_intent_splits.csv"
    if not manifest_path.exists():
        logger.error(f"Manifest missing at {manifest_path}")
        return

    # Choose N segments
    with open(manifest_path, 'r') as f:
        rows = list(csv.DictReader(f))
    
    if not rows:
        logger.error("Empty manifest!")
        return

    random.seed(42)
    sample = random.sample(rows, min(args.verify_n, len(rows)))
    
    for i, row in enumerate(sample):
        seg_id = row['segment_id']
        track_id = row['track_id']
        seg_dir = args.output_root / "segments" / args.dataset / track_id / seg_id
        logger.info(f"[{i+1}/{len(sample)}] Inferring: {seg_id}")
        cmd = f"solomuse_data.cli infer-pipeline --config {config_path} --dataset {args.dataset} --segment-dir {seg_dir}"
        run_cmd(f"{python_bin} -m {cmd}")

    # 2. Unified Report
    report_csv = verification_dir / "autonomous_verify_report.csv"
    cmd_report = f"solomuse_data.cli export-unified-report --config {config_path} --dataset {args.dataset} --action report --output {report_csv}"
    run_cmd(f"{python_bin} -m {cmd_report}")
    
    # 3. Summarize
    cmd_sum = f"solomuse_data.cli inspect-artifacts summarize --config {config_path} --dataset {args.dataset} --report-path {report_csv}"
    run_cmd(f"{python_bin} -m {cmd_sum}")

def package(args, config_path: Path):
    """STAGE 7 - Packaging"""
    banner("STAGE 7: Final Packaging & Bundle")
    if args.dry_run:
        logger.info("Dry-run: Skipping final bundling.")
        return
        
    bundle_dir = args.output_root / "final_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    
    # Copy Config
    shutil.copy2(config_path, bundle_dir / "config_runpod.yaml")
    
    # Find Checkpoints
    logger.info("Collecting best checkpoints...")
    checkpoint_roots = [
        args.output_root / "checkpoints" / "intent",
        args.output_root / "checkpoints" / "renderer"
    ]
    
    for root in checkpoint_roots:
        if root.exists():
            for pt in root.glob("*.pt"):
                shutil.copy2(pt, bundle_dir / pt.name)
    
    # Copy Verification Report
    report_csv = args.output_root / "verification" / "autonomous_verify_report.csv"
    if report_csv.exists():
        shutil.copy2(report_csv, bundle_dir / "verification_report.csv")
    
    # Create Readme
    with open(bundle_dir / "README_INFERENCE.txt", "w") as f:
        f.write("SoloMuse V2 Inference Guide\n" + "="*30 + "\n")
        f.write(f"Run Name: {args.run_name}\n")
        f.write(f"Date: {time.ctime()}\n\n")
        f.write("To run inference on a new file locally:\n")
        f.write(f"python -m solomuse_data.cli infer-pipeline --config config_runpod.yaml --dataset {args.dataset} --segment-dir <path>\n")

    logger.info(f"Bundle created at: {bundle_dir}")
    
    if args.tar_final_bundle:
        archive_path = args.output_root / f"{args.run_name}_bundle.tar.gz"
        logger.info(f"Creating archive: {archive_path}")
        run_cmd(f"tar -czf {archive_path} -C {args.output_root} final_bundle")

    banner("RUN COMPLETE!")
    logger.info(f"Logs: {args.output_root}/runpod_run.log")

def main():
    global logger
    parser = argparse.ArgumentParser(description="SoloMuse Autonomous RunPod Runner")
    # Global / Run Identity
    parser.add_argument("--run-name", type=str, default="solomuse_v2_run")
    parser.add_argument("--dataset", type=str, default="slakh", help="slakh, mock")
    
    # Paths & Persistent Storage
    parser.add_argument("--workspace-root", type=str, default="/workspace")
    default_persist = "/runpod-volume" if Path("/runpod-volume").exists() else "/workspace"
    parser.add_argument("--persistent-root", type=str, default=default_persist)
    
    # Repo Clone Settings
    parser.add_argument("--repo-url", type=str, default="https://github.com/gatencia/SOLOMUSE.git")
    parser.add_argument("--repo-branch", type=str, default="main")
    
    # Dataset Acquisition
    parser.add_argument("--slakh-url", type=str, 
                        default="https://zenodo.org/records/4599666/files/slakh2100_flac_redux.tar.gz?download=1",
                        help="Direct download URL for Slakh archive (Default: Zenodo full release)")
    parser.add_argument("--slakh-archive", type=str, help="Local path to uploaded slakh zip or tar.gz")
    
    # Hyper-parameters
    parser.add_argument("--limit", type=int, help="Limit number of segments for fast pass")
    parser.add_argument("--intent-epochs", type=int, default=100)
    parser.add_argument("--renderer-epochs", type=int, default=200)
    parser.add_argument("--intent-batch-size", type=int, default=16)
    parser.add_argument("--renderer-batch-size", type=int, default=4)
    
    # Flags
    parser.add_argument("--verify-n", type=int, default=10)
    parser.add_argument("--tar-final-bundle", action="store_true")
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true", help="Ultra-fast pass (limit 10, epochs 1)")
    
    args = parser.parse_args()
    
    if args.smoke_test:
        args.limit = 10
        args.intent_epochs = 1
        args.renderer_epochs = 1
        args.verify_n = 2
        args.dataset = "mock" if not args.slakh_url else args.dataset
        logger.info("SMOKE TEST enabled. Running ultra-light pass.")

    # 1. Resolve Global Paths
    args.persistent_root = Path(args.persistent_root)
    args.workspace_root = Path(args.workspace_root)
    args.output_root = args.persistent_root / "SOLOMUSE_RUNS" / args.run_name
    args.slakh_root = args.persistent_root / "datasets" / "slakh2100"
    
    # 2. Update Logger with File Handler
    log_file = args.output_root / "runpod_run.log"
    logger = get_logger(log_file)
    
    logger.info(f"Starting SoloMuse Autonomous Runner v2. RunName: {args.run_name}")
    logger.info(f"Persistent Root: {args.persistent_root}")
    logger.info(f"Output Root:     {args.output_root}")
    
    # 3. Execution Pipeline
    bootstrap_system(args)
    repo_root = clone_repo(args)
    python_bin = setup_python(args, repo_root)
    
    config_path = args.output_root / "runpod_pipeline.yaml"
    write_pipeline_config(args, config_path)
    
    acquire_dataset(args)
    build_artifacts(args, python_bin, config_path)
    tokenize(args, python_bin, config_path)
    train(args, python_bin, config_path)
    verify(args, python_bin, config_path)
    package(args, config_path)

if __name__ == "__main__":
    main()
