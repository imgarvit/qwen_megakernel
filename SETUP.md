# GPU Server Setup Guide

Host the Qwen3-TTS megakernel WebSocket server on a new GPU instance.

## Requirements

- **GPU**: RTX 5090 (sm_120 / Blackwell). The CUDA kernel is tuned for this architecture.
- **CUDA**: 12.8+
- **Python**: 3.12+
- **VRAM**: ~4 GB (model is bfloat16, no quantization)
- **Recommended**: Vast.ai RTX 5090 instance with PyTorch template

## 1. Get the code

```bash
# Clone or copy the qwen_megakernel directory to /workspace/
cd /workspace
git clone <your-repo-url> qwen_megakernel
cd qwen_megakernel
```

## 2. Install dependencies

```bash
# If using Vast.ai with a venv:
source /venv/main/bin/activate

pip install -r requirements.txt
```

## 3. First-run kernel compilation

The CUDA megakernel JIT-compiles on first import (~30 seconds). Trigger it once:

```bash
cd /workspace/qwen_megakernel
python -c "from qwen_megakernel.build_tts import get_extension; get_extension(); print('Kernel compiled.')"
```

## 4. Start the TTS server

```bash
python server.py --host 0.0.0.0 --port 8765

# Optional: use a speaker reference for consistent voice
python server.py --host 0.0.0.0 --port 8765 --speaker-ref /path/to/voice.wav
```

Startup takes ~10 seconds (loads model, captures CUDA graphs for the speech tokenizer decoder). You'll see:

```
Loading TTSDecoder...
TTSDecoder ready.
Starting WebSocket server on ws://0.0.0.0:8765
```

## 5. Verify it works

From any machine that can reach the GPU instance:

```bash
python pipecat_client/test_client.py --url ws://<gpu-ip>:8765 --text "Hello, how are you?"
```

This saves a WAV file and prints TTFC / RTF metrics.

## 6. Expose the port

**Vast.ai**: Add port 8765 to the instance's exposed ports in the dashboard, or use SSH tunneling:

```bash
# From your local machine:
ssh -p <ssh-port> -L 8765:localhost:8765 root@<gpu-ip>
# Then connect to ws://localhost:8765
```

**Direct**: If the instance has a public IP with port 8765 open, connect directly to `ws://<public-ip>:8765`.

## WebSocket Protocol

**Client sends** (JSON):
```json
{"text": "Hello world", "language": "English", "temperature": 0.3, "top_k": 15, "chunk_tokens": 8}
```

**Client cancels** (JSON):
```json
{"cancel": true}
```

**Server streams** (binary): float32 PCM chunks at 24 kHz, mono.

**Server ends with** (JSON):
```json
{"done": true, "duration_s": 3.04, "generation_s": 0.47, "rtf": 0.156, "samples": 72960}
```

## Performance (RTX 5090)

| Metric | Value |
|--------|-------|
| Kernel decode | 1,140 tok/s (0.88 ms/step) |
| TTFC | ~24 ms |
| RTF | 0.16-0.25 depending on text length |
| Audio sample rate | 24 kHz, mono, float32 |

## Troubleshooting

- **"flash-attn not installed" warning**: Safe to ignore. Only affects the speech tokenizer's internal transformer (already fast via CUDA graphs).
- **Port not reachable**: Use SSH tunneling (step 6) or check Vast.ai firewall rules.
- **Out of memory**: Ensure no other models are loaded. The server needs ~4 GB VRAM.
- **Kernel compile error**: Requires sm_120 (RTX 5090). Won't compile on older GPUs.
