# Renderer Neural Codec Upgrade Plan

This document explains how to upgrade Layer 3 of the **SoloMuse** architecture from the 1D Convolutional continuous baseline (`WaveChunkCodec`) to a high-quality autoregressive token model built on top of **Meta's EnCodec**.

## 1. Why Upgrade?
The `WaveChunkCodec` is a fast proxy codec passing `L2` continuous loss perfectly to prove that End-to-End data piping (`[Situation, Intent] -> Audio`) functions flawlessly. However, raw wavechunks do not produce musically cohesive high-fidelity music synthesis.

By switching to **EnCodec**, we represent audio as a sequence of highly compressed, discrete acoustic tokens `[F, Q]` across `Q` codebooks (quantizers), enabling LLM-style Next-Token generation for music (like MusicGen).

## 2. Setting up the Adapter
The codebase contains a placeholder adapter designed specifically to support this jump.
- Ensure the runtime dependency is available: `pip install encodec`
- Swap out the instantiation of the codec inside `solomuse_data.config` or manually in `solomuse_model.pipeline` to use:
  ```python
  from solomuse_model.renderer.encodec_adapter import EnCodecAdapter
  codec = EnCodecAdapter(target_bandwidth=6.0, target_sr=24000)
  ```

## 3. The New Renderer Model
While the baseline used `RendererConv1D_V1` optimizing `MSELoss(continuous_chunks)`, discrete tokens are fundamentally categorical (e.g., Vocab = `1024`).

**You must implement an Autoregressive Transformer:**
1. Your model should receive `[B, F, C_context]` (your backing and intent sequences) concatenated or cross-attended with the target token stream.
2. Because EnCodec uses `Q` distinct token streams concurrently in every frame (e.g., 4 codebooks), standard autoregressive approaches cannot predict all 4 simultaneously without violating causality or assuming independence. 
3. **The Delay Pattern**: Implement the standard MusicGen interleaving pattern. Shift the sequence of Codebook 2 by `1` step, Codebook 3 by `2` steps, etc. This flattens `[F, Q]` into a single causally predictable stream `[F+Q-1, 1]`, allowing standard `CrossEntropyLoss` per token.

## 4. Bypassing Safeguards
In `solomuse_model/renderer/train.py`, you will find an explicit `NotImplementedError` raised if `codec.code_type == "discrete"`.
- Remove this safeguard.
- Replace the legacy `MSELoss` path with your categorical Next-Token prediction loop utilizing the new Transformer.
- Make sure `prepare_targets.py` accurately caches `[F, Q]` integers instead of Floats, adhering strictly to the assertions created in `solomuse_model/renderer/token_contracts.py`.

Once the transformer trains to low Cross-Entropy loss on the tokens, `EnCodecAdapter.decode(codes)` natively handles turning predicted tokens `[F, Q]` directly back into high-fidelity solo audio waves.
