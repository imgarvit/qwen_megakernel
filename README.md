# Qwen3-TTS Megakernel — RTX 5090 Decode Backend for Pipecat

A CUDA megakernel implementation of the Qwen3-TTS talker decoder, adapted from [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel), wired into a real-time Pipecat voice agent pipeline.

## Architecture

Two deployment modes are supported:

### GPU-Only Pipeline (Recommended)

Everything runs on the GPU. The local machine only handles mic/speaker I/O.

```
LOCAL MACHINE                              VAST.AI RTX 5090
┌───────────────────┐                      ┌──────────────────────────────────────┐
│  audio_client.py  │                      │  gpu_pipeline.py (Pipecat + TTS)    │
│                   │                      │                                      │
│  Mic ─────────────┼──PCM 16kHz──►        │  VAD → STT → LLM → DirectTTS       │
│                   │                      │                     │                │
│  Speaker ◄────────┼──PCM 24kHz──         │    ┌────────────────┘                │
│                   │                      │    │ TTSDecoder (direct, no WS hop)  │
│  (pyaudio + WS)   │                      │    │  ├─ Megakernel (28-layer        │
│                   │                      │    │  │  Qwen3 backbone, CUDA)       │
└───────────────────┘                      │    │  ├─ Code Predictor (5-layer)    │
         ▲                                 │    │  └─ Speech Tokenizer (CUDA)     │
         │  SSH tunnel :8766               │    └─ Speaker Embedding (cached)     │
         └─────────────────────────────────┘                                      │
                                           └──────────────────────────────────────┘
```

### Split Mode (Alternative)

Pipecat runs locally with STT/LLM; the GPU runs only the TTS server.

```
LOCAL MACHINE                              VAST.AI RTX 5090
┌───────────────────────────┐              ┌─────────────────────────────────┐
│  Pipecat Pipeline         │              │  server.py (WebSocket :8765)    │
│                           │              │                                 │
│  Mic → STT → LLM ────────┼──text──►     │  TTSDecoder                     │
│                    ◄──────┼──PCM───      │    ├─ Megakernel backbone       │
│              → Speaker    │  chunks      │    ├─ Code Predictor            │
│                           │              │    └─ Speech Tokenizer          │
│  MegakernelTTSService     │              │                                 │
│  (WebSocket client)       │              │  Speaker embedding cached once  │
└───────────────────────────┘              └─────────────────────────────────┘
```

## What the Megakernel Does

The original `qwen_megakernel` runs Qwen3-0.6B decode at ~1,154 tok/s on a single RTX 5090 using a persistent CUDA kernel (128 blocks × 512 threads). This project adapts it for **Qwen3-TTS-12Hz-0.6B-Base** — the talker decoder stage of Qwen3-TTS.

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

Each decode step produces 80 ms of audio (12.5 tokens/second at 24 kHz):

1. **Megakernel backbone** — 28 transformer layers → codebook-0 token + hidden state (~0.87 ms)
2. **Code predictor** — 5-layer transformer, 3 autoregressive steps → codebooks 1–3 (~2.5 ms)
3. **Speech tokenizer decoder** — 4-codebook tokens → audio waveform (pipelined on separate CUDA stream)

## Performance

Measured on a warmed-up RTX 5090 instance (post `torch.compile` tracing).

| Metric | Value | Target |
|--------|-------|--------|
| **TTS TTFC** (time to first audio chunk) | **58–64 ms** | < 90 ms |
| **RTF** (real-time factor) | **0.09–0.12** | < 0.30 |
| Backbone decode | 1,147 tok/s (0.87 ms/step) | — |
| Code predictor | ~2.5 ms/step (compiled) | — |
| End-to-end per decode step | ~5 ms | — |
| Audio output | 24 kHz, mono, int16 | — |

### Real Benchmark Data (from server logs)

```
[127.0.0.1] done: 2.88s audio in 0.35s (RTF=0.121, TTFC=59ms)
[127.0.0.1] done: 5.12s audio in 0.51s (RTF=0.101, TTFC=61ms)
[127.0.0.1] done: 3.92s audio in 0.45s (RTF=0.114, TTFC=63ms)
[127.0.0.1] done: 18.72s audio in 1.67s (RTF=0.089, TTFC=58ms)
```

### Bottleneck Analysis

The megakernel backbone is extremely fast (0.87 ms/step, 1,147 tok/s). The end-to-end decode step time is ~5 ms, giving an RTF well below 0.15. The dominant latency in the full pipeline comes from:

1. **LLM response time** (~500 ms TTFB from OpenAI API) — not under our control
2. **STT transcription** (~1 s round-trip to Whisper) — not under our control
3. **Network round-trip** (SSH tunnel) — <5 ms for local tunnel

The TTS engine itself streams audio faster than real-time, meaning the user hears audio starting within ~60 ms of TTS receiving text.

### Measurement Methodology

- **TTFC**: `time.perf_counter()` from text receipt to first PCM chunk available
- **RTF**: `generation_time / audio_duration` where `audio_duration = total_samples / 24000`
- **Kernel tok/s**: Timed via `torch.cuda.synchronize()` around the backbone kernel call
- All numbers from a warmed-up server (post `torch.compile` tracing), measured across 20+ utterances

## Quick Start

### Option A: GPU-Only Pipeline (Recommended)

**1. GPU Instance (Vast.ai RTX 5090)**

```bash
ssh -p <port> root@<gpu-ip>

# Install dependencies
pip install -r requirements.txt
pip install "pipecat-ai[openai,silero,websocket]"

# Start the pipeline (first run compiles kernel, ~30s)
export OPENAI_API_KEY=your_key
PYTHONPATH=/workspace/qwen_megakernel python gpu_pipeline.py \
  --port 8766 \
  --speaker-ref data/speaker_ref.wav
```

**2. Local Machine (Audio Client)**

```bash
pip install pyaudio websockets

# Open SSH tunnel
ssh -p <ssh-port> root@<gpu-ip> -L 8766:localhost:8766

# Stream audio
python audio_client.py --url ws://localhost:8766
```

### Option B: Split Mode (Local Pipecat + Remote TTS)

**1. GPU Instance**

```bash
PYTHONPATH=/workspace/qwen_megakernel python server.py \
  --port 8765 \
  --speaker-ref data/speaker_ref.wav
```

**2. Local Machine**

```bash
pip install -r pipecat_client/requirements.txt

# SSH tunnel
ssh -p <ssh-port> root@<gpu-ip> -L 8765:localhost:8765

# Run voice agent
export OPENAI_API_KEY=your_key
export TTS_WS_URL=ws://localhost:8765
cd pipecat_client && python pipeline_demo.py
```

## Interruption Support

Both modes support user interruptions (barge-in):

- **GPU pipeline**: Pipecat's VAD detects user speaking → `DirectMegakernelTTSService._handle_interruption()` calls `decoder.cancel()` → decode stops within one step → audio stops immediately
- **Split mode**: VAD trigger → `MegakernelTTSService` sends `{"cancel": true}` over WebSocket → `server.py` cancels decode → replies `{"cancelled": true}` → TTS ready for next sentence

## Voice Consistency

Voice stability across sentences uses ECAPA-TDNN x-vector speaker embeddings:

- **Speaker reference** (`--speaker-ref`): Extracts a 1024-dim embedding at startup, cached and injected into every generation call
- **Low temperature** (0.3): Reduces prosody variability between utterances
- **Low top-k** (20): Tighter sampling for predictable output
- **Repetition detection**: Stops generation if the same codebook-0 token repeats 30 times

## WebSocket Protocol (Split Mode)

**Client → Server** (JSON):
```json
{
  "text": "Hello world",
  "language": "English",
  "temperature": 0.3,
  "top_k": 20,
  "chunk_tokens": 8
}
```

**Server → Client** (binary): Raw PCM audio chunks (float32, 24 kHz, mono), streamed frame-by-frame.

**Server → Client** (JSON, final):
```json
{"done": true, "duration_s": 3.04, "generation_s": 0.47, "ttfc_ms": 48.2, "rtf": 0.155, "samples": 72960}
```

**Cancellation**: Client sends `{"cancel": true}` at any time. Server stops at next decode step and acknowledges.

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
│   ├── megakernel_tts.py        # Pipecat TTSService (WebSocket client, split mode)
│   ├── pipeline_demo.py         # Full voice agent with local audio (split mode)
│   ├── test_client.py           # Standalone WebSocket test
│   └── requirements.txt         # Local machine dependencies
├── gpu_pipeline.py              # GPU-only Pipecat pipeline (recommended)
├── audio_client.py              # Local mic/speaker WebSocket client
├── server.py                    # WebSocket TTS server (split mode)
├── requirements.txt             # GPU server dependencies
├── SETUP.md                     # Detailed GPU instance setup guide
└── README.md
```

## Known Limitations

1. **Repetition at long utterances** — Codebook-0 token can enter repetition loops on long sentences. Mitigated by early-stop detection (30-repeat threshold).
2. **`torch.compile` warmup** — First server start takes ~30s for tracing. Subsequent calls are fast.
3. **Single-request processing** — Both modes handle one TTS request at a time. Concurrent requests queue.
4. **RTX 5090 only** — The kernel is compiled for sm_120 (Blackwell). It will not run on older GPUs.
5. **OpenAI dependency** — STT (Whisper) and LLM (GPT-4o-mini) require an OpenAI API key.

## Credits

- [AlpinDale/qwen_megakernel](https://github.com/AlpinDale/qwen_megakernel) — original Qwen3-0.6B megakernel for RTX 5090
- [Elliot Arledge's MegaQwen](https://github.com/ElliotArledge) — original RTX 3090 megakernel concept
- [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS) — the TTS model architecture
- [Pipecat](https://docs.pipecat.ai) — voice agent pipeline framework
