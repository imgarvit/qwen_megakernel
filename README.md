# Qwen3-TTS Megakernel — RTX 5090 Decode Backend for Pipecat

A CUDA megakernel implementation of the Qwen3-TTS talker decoder, adapted from [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel), wired into a Pipecat voice agent pipeline via WebSocket streaming.

## Architecture

```
LOCAL MACHINE                              VAST.AI RTX 5090
┌───────────────────────────┐              ┌─────────────────────────────────┐
│  Pipecat Pipeline         │              │  server.py (WebSocket :8765)    │
│                           │              │                                 │
│  Mic → STT → LLM ────────┼──text──►     │  TTSDecoder                     │
│                    ◄──────┼──PCM───      │    ├─ Megakernel (28-layer      │
│              → Speaker    │  chunks      │    │  Qwen3 backbone, CUDA)     │
│                           │              │    ├─ Code Predictor (5-layer)  │
│  MegakernelTTSService     │              │    └─ Speech Tokenizer          │
│  (WebSocket client)       │              │       (CUDA-graphed decoder)    │
└───────────────────────────┘              └─────────────────────────────────┘
```

### Why This Split?

- **GPU instance** runs `server.py` — loads the model once, accepts WebSocket connections, streams PCM audio frame-by-frame. The megakernel requires an RTX 5090 (sm_120) and CUDA 12.8+.
- **Local machine** runs a Pipecat pipeline with `MegakernelTTSService` — a custom TTS service that connects to the remote WebSocket. No GPU needed locally.
- **Single repo** keeps everything together for ease of review and deployment. The split is at the network boundary, not the repo boundary.

## What the Megakernel Does

The original `qwen_megakernel` runs Qwen3-0.6B decode at ~1,154 tok/s on a single RTX 5090 using a persistent CUDA kernel (128 blocks x 512 threads). This project adapts it for **Qwen3-TTS-12Hz-0.6B-Base** — the talker decoder stage of Qwen3-TTS.

### Kernel Modifications for TTS

| Change | Why |
|--------|-----|
| Half-split RoPE (pair dim k with k+64) | Qwen3-TTS uses `rotate_half()`, not interleaved pairing |
| M-RoPE with 3 position axes [24,20,20] | TTS uses multi-axis RoPE for text/audio-time/audio-step |
| Full-precision `expf()` and division | Replaced PTX `ex2.approx` / `rcp.approx` to eliminate attention noise |
| Removed `--use_fast_math` | Prevents NVCC from replacing `expf` with approximate `__expf` |
| Host-side barrier resets (`cudaMemsetAsync`) | Fixes race condition on kernel re-launch between PyTorch ops |
| TTS vocab size (3072 vs 151,936) | ~18% speedup in LM head due to smaller vocabulary |

### TTS Pipeline (per decode step)

Each decode step produces 80ms of audio (12.5 tokens/second at 24kHz):

1. **Megakernel backbone** — 28 transformer layers → codebook-0 token + hidden state (~0.87 ms)
2. **Code predictor** — 5-layer transformer, 15 autoregressive steps → codebooks 1-15 (~52 ms)
3. **Speech tokenizer decoder** — 16-codebook tokens → audio waveform (~33 ms, pipelined on separate CUDA stream)

### Optimizations Applied

| Optimization | Before | After | Impact |
|-------------|--------|-------|--------|
| Megakernel for backbone | PyTorch forward | Single CUDA kernel | 1,147 tok/s decode |
| Megakernel for code predictor | HF `generate()` (~137 ms) | Manual loop + kernel (~52 ms) | 2.6x speedup |
| `torch.compile` on code predictor | ~130 ms/step | ~52 ms/step | 2.5x |
| Chunked decode with 25-frame left context | O(n) growing window | O(1) constant ~33 ms | Linear → constant |
| Pipelined decode on separate CUDA stream | Sequential | Overlapped | ~33 ms hidden |
| CUDA graphs for speech tokenizer | PyTorch dispatch overhead | Single graph replay | Faster decode |
| Pre-cached special token embeddings | Recomputed per call | Cached at init | ~10 ms saved |
| Pre-computed suppress mask | Python loop per step | Single GPU tensor | ~15 ms saved |

## Performance

| Metric | Value | Target |
|--------|-------|--------|
| Kernel decode (backbone) | 1,147 tok/s (0.87 ms/step) | — |
| Code predictor (compiled) | ~52 ms/step | — |
| End-to-end per step | ~55 ms | — |
| TTFC (warm) | ~50 ms | < 60 ms |
| RTF (real-time factor) | 0.15–0.25 | < 0.30 |
| Audio sample rate | 24 kHz, mono, float32 | — |

### Bottleneck Analysis

The megakernel itself is extremely fast (0.87 ms/step, 1,147 tok/s). The bottleneck is the **code predictor**: 15 autoregressive forward passes through a 5-layer transformer at ~52 ms/step even after `torch.compile` optimization. Each backbone step produces 80ms of audio, so at 55ms/step total, we achieve an RTF of ~0.69 (generation is faster than real-time).

To further reduce RTF, the code predictor would need its own fused megakernel (reducing 15 sequential passes from ~52 ms to ~5 ms). This is a known architectural limitation of the multi-codebook design, not a fixable integration issue.

### Measurement Methodology

- **TTFC**: `time.perf_counter()` from request receipt to first PCM chunk sent over WebSocket
- **RTF**: `generation_time / audio_duration` where `audio_duration = total_samples / 24000`
- **Kernel tok/s**: Timed via `torch.cuda.synchronize()` around the backbone kernel call
- All measurements taken on a warmed-up server (post `torch.compile` tracing)

## Quick Start

### 1. GPU Instance (Vast.ai RTX 5090)

```bash
ssh -p <port> root@<gpu-ip>

# Install dependencies
pip install -r requirements.txt

# Start the server (first run JIT-compiles kernel, ~30s)
PYTHONPATH=/workspace/qwen_megakernel python server.py --port 8765

# Optional: use a speaker reference for consistent voice
python server.py --port 8765 --speaker-ref /path/to/voice.wav
```

### 2. Local Machine (Pipecat Client)

```bash
# Install local dependencies
pip install -r pipecat_client/requirements.txt

# Set up SSH tunnel to GPU
ssh -p <ssh-port> -L 8765:localhost:8765 root@<gpu-ip>

# Test TTS directly (no Pipecat)
python pipecat_client/test_client.py --url ws://localhost:8765 --text "Hello world"

# Run the full voice agent pipeline
export OPENAI_API_KEY=your_key
export TTS_WS_URL=ws://localhost:8765
cd pipecat_client && python pipeline_demo.py
```

## WebSocket Protocol

**Client → Server** (JSON):
```json
{
  "text": "Hello world",
  "language": "English",
  "temperature": 0.3,
  "top_k": 15,
  "speaker_ref": "/path/to/voice.wav",
  "chunk_tokens": 8
}
```

**Server → Client** (binary): Raw PCM audio chunks (float32, 24kHz, mono), streamed frame-by-frame as generated.

**Server → Client** (JSON, final):
```json
{"done": true, "duration_s": 3.04, "generation_s": 0.47, "ttfc_ms": 48.2, "rtf": 0.155, "samples": 72960}
```

**Cancellation**: Client sends `{"cancel": true}` to abort in-flight generation. The server stops at the next decode step and sends the `done` message.

## Interruption Support

The Pipecat integration supports user interruptions:

- `MegakernelTTSService._flush()` sends a cancel message to the server and drains pending audio
- `TTSDecoder.cancel()` sets a flag checked at each decode step, stopping generation within one step (~55ms)
- `allow_interruptions=True` in the pipeline enables Pipecat's built-in interruption handling

## Voice Consistency

Voice stability across sentences is controlled by:

- **Speaker reference** (`--speaker-ref`): Anchors all generations to the same voice embedding. Generate a clean reference sample once, then reuse it.
- **Low temperature** (default 0.3): Reduces prosody variability between utterances
- **Low top-k** (default 15): Tighter sampling for more predictable output

## File Structure

```
qwen_megakernel/
├── csrc/
│   ├── kernel_tts.cu            # Adapted CUDA megakernel (M-RoPE, half-split, TTS vocab)
│   └── torch_bindings_tts.cpp   # PyTorch C++ bindings
├── qwen_megakernel/
│   ├── tts_model.py             # TTSDecoder: kernel + code predictor + speech tokenizer
│   ├── build_tts.py             # JIT compilation of CUDA extension
│   └── bench_tts.py             # Performance benchmarks
├── pipecat_client/
│   ├── megakernel_tts.py        # Pipecat TTSService (WebSocket client, runs locally)
│   ├── pipeline_demo.py         # Full voice agent: Mic → STT → LLM → TTS → Speaker
│   ├── test_client.py           # Standalone WebSocket test (no Pipecat needed)
│   └── requirements.txt         # Local machine dependencies
├── server.py                    # WebSocket TTS server (runs on GPU instance)
├── test_reference.py            # Validates megakernel vs reference model output
├── test_kernel_step.py          # Validates individual kernel decode steps
├── requirements.txt             # GPU server dependencies
├── SETUP.md                     # Detailed GPU instance setup guide
└── README.md
```

## Known Limitations

1. **Code predictor bottleneck** — 15 autoregressive codebook predictions at ~52 ms/step dominate end-to-end latency. A fused megakernel for the code predictor is the path to RTF < 0.1.
2. **`torch.compile` warmup** — First server start takes ~30s for tracing. Subsequent calls are fast.
3. **Single-request processing** — The server handles one TTS request at a time. Concurrent requests queue.
4. **RTX 5090 only** — The kernel is compiled for sm_120 (Blackwell). It will not run on older GPUs.

## Credits

- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) — original Qwen3-0.6B megakernel for RTX 5090
- [Elliot Arledge's MegaQwen](https://github.com/ElliotArledge) — original RTX 3090 megakernel concept
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) — the TTS model architecture
- [Pipecat](https://docs.pipecat.ai) — voice agent pipeline framework
