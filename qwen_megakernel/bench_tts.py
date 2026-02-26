"""
TTS megakernel benchmarks.

Measures:
  - Talker backbone tok/s (megakernel only, no MTP or tokenizer)
  - TTFC (time to first audio chunk)
  - RTF (real-time factor)
  - End-to-end latency

Usage:
  python -m qwen_megakernel.bench_tts [--warmup 5] [--steps 100]
"""

import argparse
import time

import numpy as np
import torch
import soundfile as sf

from qwen_megakernel.tts_model import (
    TTSDecoder,
    SAMPLE_RATE,
    TOKENS_PER_SECOND,
    AUDIO_MS_PER_TOKEN,
    compute_mrope_cos_sin,
    _build_mrope_inv_freq,
    MAX_SEQ_LEN,
    HEAD_DIM,
    HIDDEN_SIZE,
    NUM_LAYERS,
    NUM_KV_HEADS,
    Q_SIZE,
    KV_SIZE,
    INTERMEDIATE_SIZE,
)


def bench_kernel_only(decoder: TTSDecoder, warmup: int = 5, steps: int = 100):
    """Benchmark the megakernel decode step alone (no embedding, no MTP)."""
    print("\n=== Kernel-Only Benchmark ===")

    inv_freq = _build_mrope_inv_freq()
    input_hidden = torch.randn(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")

    # Warmup
    decoder.reset()
    for i in range(warmup):
        cos_step, sin_step = compute_mrope_cos_sin(i, i, i, inv_freq)
        decoder._tts_decode(
            decoder._out_token, input_hidden,
            decoder._layer_weights_packed,
            decoder._final_norm_weight, decoder._lm_head_weight,
            cos_step, sin_step,
            decoder._k_cache, decoder._v_cache, decoder._hidden,
            decoder._act, decoder._res, decoder._q, decoder._k, decoder._v,
            decoder._attn_out, decoder._mlp_inter, decoder._norm_out,
            decoder._bmax_vals, decoder._bmax_idxs,
            NUM_LAYERS, i + 1, MAX_SEQ_LEN, decoder._attn_scale,
        )
    torch.cuda.synchronize()

    # Timed run
    decoder.reset()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(steps):
        cos_step, sin_step = compute_mrope_cos_sin(i, i, i, inv_freq)
        decoder._tts_decode(
            decoder._out_token, input_hidden,
            decoder._layer_weights_packed,
            decoder._final_norm_weight, decoder._lm_head_weight,
            cos_step, sin_step,
            decoder._k_cache, decoder._v_cache, decoder._hidden,
            decoder._act, decoder._res, decoder._q, decoder._k, decoder._v,
            decoder._attn_out, decoder._mlp_inter, decoder._norm_out,
            decoder._bmax_vals, decoder._bmax_idxs,
            NUM_LAYERS, i + 1, MAX_SEQ_LEN, decoder._attn_scale,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    ms_per_step = (elapsed / steps) * 1000
    tok_per_s = steps / elapsed
    print(f"  Steps:     {steps}")
    print(f"  Total:     {elapsed * 1000:.1f} ms")
    print(f"  Per step:  {ms_per_step:.3f} ms")
    print(f"  Tok/s:     {tok_per_s:.0f}")
    return tok_per_s, ms_per_step


def bench_streaming(decoder: TTSDecoder, text: str, chunk_tokens: int = 4,
                    save_audio: str = ""):
    """Benchmark end-to-end streaming TTS including TTFC and RTF."""
    print("\n=== Streaming Benchmark ===")
    print(f"  Text: '{text[:60]}...'")

    all_audio = []

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ttfc = None
    total_audio_samples = 0
    num_chunks = 0

    for audio_chunk in decoder.generate_speech_streaming(
        text=text, chunk_tokens=chunk_tokens,
    ):
        if ttfc is None:
            torch.cuda.synchronize()
            ttfc = time.perf_counter() - t0
        if isinstance(audio_chunk, np.ndarray):
            total_audio_samples += len(audio_chunk)
            all_audio.append(audio_chunk)
        elif isinstance(audio_chunk, torch.Tensor):
            arr = audio_chunk.cpu().numpy().flatten()
            total_audio_samples += len(arr)
            all_audio.append(arr)
        num_chunks += 1

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    audio_duration = total_audio_samples / SAMPLE_RATE
    rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")
    tokens_generated = num_chunks * chunk_tokens

    print(f"  Tokens:         {tokens_generated}")
    print(f"  Audio chunks:   {num_chunks}")
    print(f"  Audio duration: {audio_duration:.2f} s")
    print(f"  Wall time:      {elapsed * 1000:.1f} ms")
    print(f"  TTFC:           {(ttfc or 0) * 1000:.1f} ms")
    print(f"  RTF:            {rtf:.4f}")
    print(f"  E2E latency:    {elapsed * 1000:.1f} ms")

    if save_audio and all_audio:
        combined = np.concatenate(all_audio)
        combined = np.clip(combined, -1.0, 1.0)
        sf.write(save_audio, combined, SAMPLE_RATE)
        print(f"  Saved audio:    {save_audio} ({len(combined)} samples)")

    return {
        "tokens": tokens_generated,
        "audio_duration_s": audio_duration,
        "wall_time_ms": elapsed * 1000,
        "ttfc_ms": (ttfc or 0) * 1000,
        "rtf": rtf,
    }


def main():
    parser = argparse.ArgumentParser(description="TTS Megakernel Benchmarks")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument(
        "--text",
        default="Hello, this is a test of the Qwen three text to speech system powered by the megakernel.",
    )
    parser.add_argument("--chunk-tokens", type=int, default=4)
    parser.add_argument("--save-audio", default="/workspace/qwen_megakernel/test_output.wav")
    args = parser.parse_args()

    print("Initializing TTSDecoder ...")
    decoder = TTSDecoder(verbose=True)

    tok_s, ms_step = bench_kernel_only(decoder, args.warmup, args.steps)
    stream_results = bench_streaming(decoder, args.text, args.chunk_tokens,
                                     save_audio=args.save_audio)

    print("\n=== Summary ===")
    print(f"  Kernel tok/s:   {tok_s:.0f}")
    print(f"  Kernel ms/step: {ms_step:.3f}")
    print(f"  TTFC:           {stream_results['ttfc_ms']:.1f} ms (target < 60 ms)")
    print(f"  RTF:            {stream_results['rtf']:.4f} (target < 0.15)")
    print(f"  Audio duration: {stream_results['audio_duration_s']:.2f} s")
    print(f"  Wall time:      {stream_results['wall_time_ms']:.1f} ms")


if __name__ == "__main__":
    main()
