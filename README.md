# Solomuse Data Preparation Pipeline

This pipeline prepares audio data for training the Solomuse model by creating aligned `(Backing, Solo)` pairs from multitrack datasets (Slakh2100, MUSDB18) and segmenting them into fixed-length windows.

## Prerequisites

- Python 3.11+
- `ffmpeg` installed and on your system PATH.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/solomuse-data.git
cd solomuse-data

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies in editable mode
pip install -e .
```

## 1. Get the Data

You need to download the source datasets yourself. The pipeline expects the raw uncompressed (or flac) data.

### Slakh2100
Download the full dataset (or a subset) from [slakh.com](http://www.slakh.com/).
- Recommended: `slakh2100-v2.1` structure.
- You should have a folder containing subfolders like `Train`, `Validation`, `Test`, each containing track folders (e.g., `Track00001`).

### MUSDB18
Download MUSDB18 (HQ or regular).
- You need the decoded stems version (folders with `vocals.wav`, `drums.wav`, etc.) or the Native Instruments STEMS format (mp4). 
- **Note**: The current adapter expects uncompressed WAV/FLAC stems in folders. If you have the `musdb18.zip` or `musdb18hq.zip`, extracting it usually gives you the right structure (folders per song).

### Optional: Weak Data (Raw MP3s)
Gather a folder of raw audio files (mp3, wav, flac) for weak supervision.

## 2. Configuration

Create a `config.yaml` file in the project root.

```yaml
# storage/config.yaml

output_root: "./data/processed"  # Where processed data will go

dataset_roots:
  slakh: "/path/to/slakh2100"    # Path to Slakh root
  musdb: "/path/to/musdb18"      # Path to MUSDB root
  weak_songs: "/path/to/my_mp3s" # Path to weak input folder (optional)

# Pipeline Settings
canonical_sample_rate: 44100
canonical_channels: 2
segment_seconds: 6.0
segment_hop_seconds: 3.0
min_segment_energy: 0.0001
enable_weak_demucs: true         # Set to true if using weak data
demucs_model: "htdemucs"         # Demucs model (requires 'pip install demucs')
```

## 3. Usage

Run the pipeline steps in order using the `solomuse-data` CLI command (or `python -m solomuse_data.cli`).

### Step 1: Build Pairs (Supervised)
Converts raw datasets into `(backing, solo)` pairs.

```bash
# Process Slakh
solomuse-data build-pairs --config config.yaml --dataset slakh

# Process MUSDB
solomuse-data build-pairs --config config.yaml --dataset musdb
```
Outputs: `data/processed/pairs/{dataset}/...`

### Step 2: Generate Weak Data (Optional)
Uses Demucs to separate raw songs.

```bash
# Requires enable_weak_demucs: true in config
solomuse-data generate-weak --config config.yaml --dataset weak_songs
```
Outputs: `data/processed/weak_pairs/...`

### Step 3: Validate
Checks the generated pairs for issues (clipping, silence, format).

```bash
solomuse-data validate --config config.yaml --dataset slakh
solomuse-data validate --config config.yaml --dataset musdb
```
Check reports in `data/processed/reports/`.

### Step 4: Segment
Chops the pairs into fixed-length training windows (e.g., 6s).

```bash
solomuse-data segment --config config.yaml --dataset slakh
solomuse-data segment --config config.yaml --dataset musdb
```
temp
Outputs: `data/processed/segments/{dataset}/...` and the final manifest `data/processed/segments/{dataset}/manifest.csv`.

## 4. Final Dataset
The final training data is located in the segmentation manifests.
- `data/processed/segments/slakh/manifest.csv`
- `data/processed/segments/musdb/manifest.csv`


## 5. Docker Support

You can run the entire pipeline in a container without installing dependencies manually.

### Build the Image
```bash
docker-compose build
```

### Configuration for Docker
Ensure your `config.yaml` points to paths inside `/app/data`.
Example if your local data is in `./data/raw/slakh`:
```yaml
dataset_roots:
  slakh: "/app/data/raw/slakh"
output_root: "/app/data/processed"
```

### Run Commands
Use `docker-compose run` to execute commands.

```bash
# Build Pairs
docker-compose run --rm solomuse build-pairs --config config.yaml --dataset slakh

# Validate
docker-compose run --rm solomuse validate --config config.yaml --dataset slakh

# Segment
docker-compose run --rm solomuse segment --config config.yaml --dataset slakh

# Or simply run ALL steps at once:
docker-compose run --rm solomuse run-all --config config.yaml --dataset slakh
```


The processed data will appear in your local `./data` folder (mounted as volume).

