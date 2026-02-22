# SoloMuse Baseline Validation Runbook

This runbook provides the exact terminal commands required to validate the end-to-end baseline architecture of the SoloMuse pipeline before upgrading to the final-quality neural codec architecture.

**Prerequisites:**
- Ensure your `config.yaml` is fully populated.
- Ensure the `slakh` source dataset (or another target) has been canonicalized, paired, and segmented (using `build-pairs` and `segment`).

---

## 1. Small-Subset Artifact Generation Check

Extract the necessary representations for a small subset of segments to verify the data abstraction pipelines.

### Layer 1: Situation Extraction
```bash
python -m solomuse_data.cli situation --config config.yaml --dataset slakh --limit 5
```
**What it does:** Computes the Layer 1 `situation.npy` vectors for the first 5 segments.
**Success & Artifacts:** Completes without errors. Creates `situation.npy` inside the respective segment directories (e.g., `data/processed/segments/slakh/track_0/seg_0/`).
**Failure signs:** Librosa caching or `Numba` JIT errors. 

### Layer 2: Intent Target Extraction
```bash
python -m solomuse_data.cli intent-targets --config config.yaml --dataset slakh --limit 5
```
**What it does:** Computes the ground-truth `intent_targets.npy` sequences for those segments.
**Success & Artifacts:** Creates `intent_targets.npy` in the segment directories and updates `manifest_intent.csv` in the dataset root.
**Failure signs:** Missing `situation.npy` files from the previous step.

### Layer 3: Renderer Target Extraction
```bash
python -m solomuse_data.cli renderer-targets --config config.yaml --dataset slakh --limit 5
```
**What it does:** Extracts the ground-truth audio target representations (`renderer_target.npy`). For the baseline `WaveChunkCodec`, these are simply chopped waveforms.
**Success & Artifacts:** Creates `renderer_target.npy` in the segment directories.
**Failure signs:** Mismatched sample rates causing indexing bounds errors.

---

## 2. Shape & Timing Sanity Checks

Before training, verify the dimensions of the generated artifacts using a quick Python inspection.

**Command (Interactive Python/IPython):**
```python
import numpy as np
sit = np.load("data/processed/segments/slakh/track_0/seg_0/situation.npy")
int_targ = np.load("data/processed/segments/slakh/track_0/seg_0/intent_targets.npy")
ren_targ = np.load("data/processed/segments/slakh/track_0/seg_0/renderer_target.npy")
print(f"Situation: {sit.shape}, Intent: {int_targ.shape}, Renderer: {ren_targ.shape}")
```

**Expected Results (Assuming 6.0s segments, 10Hz intent, 44100Hz sr, 20ms/10ms hop):**
- **Situation:** `(32,)`
- **Intent Target:** `(60, 7)` (F_intent = 60 frames, D = 7 parameters)
- **Renderer Target:** `(599, 882)` (F_renderer ≈ 600 frames, 882 samples per 20ms continuous chunk)

*(Note: exact frame counts may fluctuate by `±1` depending on framing stride truncations)*

---

## 3. Planner Baseline Smoke Training + Inference

### Training
```bash
python -m solomuse_data.cli train-intent --config config.yaml --dataset slakh
```
**What it does:** Trains the `IntentGRU_V1` network on the `intent_targets.npy` dataset.
**Success & Artifacts:** Output logs showing declining MSE. Saves model to `data/processed/models/intent_v1/best.pt`.
**Failure signs:** DataLoader collation errors due to sequence length mismatches (use padding if this occurs).

### Inference
*Select a valid segment directory from the limits used above.*
```bash
python -m solomuse_data.cli infer-intent --config config.yaml --dataset slakh --segment-dir data/processed/segments/slakh/track_0/seg_0
```
**What it does:** Generates `intent_pred.npy` from the `situation.npy` vector.
**Success & Artifacts:** Creates `intent_pred.npy` in the segment directory.

---

## 4. Renderer Baseline Smoke Training

```bash
python -m solomuse_data.cli train-renderer --config config.yaml --dataset slakh
```
**What it does:** Trains the `RendererConv1D_V1` model utilizing the continuous `WaveChunkCodec`.
**Success & Artifacts:** Runs instantly (tiny ConvNet). Logs declining MSE. Saves checkpoint to `data/processed/models/renderer_v1/best.pt`.
**Failure signs:** Getting a `NotImplementedError` regarding "Discrete token training". This means your config is erroneously pointing to the new `EnCodecAdapter` before you have actually implemented the Autoregressive model! Ensure `config.yaml` points to `wavechunk`.

---

## 5. End-to-End One-Segment Pipeline Run

```bash
python -m solomuse_data.cli render-segment --config config.yaml --dataset slakh --segment-dir data/processed/segments/slakh/track_0/seg_0
```
**What it does:** Loads `x.wav`, `situation.npy`, and `intent_pred.npy` (if it exists, otherwise falls back to `intent_targets.npy`). It passes them through the loaded `RendererConv1D_V1` checkpoint to statically generate `y_hat.wav`.
**Success & Artifacts:** Creates `y_hat.wav` in the segment directory. 
**Failure signs:** Missing checkpoint files from the training steps. 

---

## 6. Live Simulation Smoke Test

Test the overlap-add continuous frame-chunking logic handling a lengthy backing track.
```bash
python -m solomuse_data.cli live-sim --config config.yaml --dataset slakh --wav data/processed/segments/slakh/track_0/seg_0/x.wav --out data/processed/live_sim_out.wav
```
**What it does:** Spins up `LiveSimulationRunner`, chunks the input `wav` at `live_chunk_ms` intervals, extracts the situation, plans the intent, runs the renderer, and overlap-adds the output.
**Success & Artifacts:** Runs progressively and outputs `data/processed/live_sim_out.wav` perfectly matching the length of `x.wav`.
**Failure signs:** Audio artifacts/clicking (overlap mismatch). 

---

## 7. Inspection Pack Export

To package the smoke-test results for offline review or sharing:
```bash
tar -czvf baseline_smoke_test.tar.gz \
  data/processed/models/ \
  data/processed/live_sim_out.wav \
  data/processed/segments/slakh/track_0/seg_0/y_hat.wav \
  data/processed/segments/slakh/track_0/seg_0/intent_pred.npy \
  data/processed/segments/slakh/track_0/seg_0/situation.npy
```

---

## 8. Promotion Gate Checklist

**Before upgrading to the final-quality neural codec (EnCodec+MusicGen), ALL of the following must be true:**

- [ ] **Artifact Alignment:** `situation.npy`, `intent_targets.npy`, and `renderer_target.npy` can be generated successfully without batch dimension crashing.
- [ ] **Planner Training Convergence:** The `train-intent` command executes and successfully writes `best.pt` after reducing its synthetic loss.
- [ ] **Renderer Baseline Convergence:** The `train-renderer` command successfully processes the continuous `WaveChunk` representations over MSE Loss.
- [ ] **Offline Generation:** `render-segment` produces a `y_hat.wav` file. (Audio quality will sound poor/robotic because wavechunks are a simple proxy, but it MUST sound like *something* correlated to the input, demonstrating the pipeline tensors are mathematically bound).
- [ ] **Live Simulation Stability:** The `live-sim` overlap-add engine does not throw dimension mismatch errors or crash midway through a longer `.wav` file. 

**If you can check all 5 boxes, the underlying infrastructure is perfectly sound. Proceed to implement the Autoregressive Token architecture!**
