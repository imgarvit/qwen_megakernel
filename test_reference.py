"""Quick test: does the reference model generate proper speech with EOS?"""
import torch
import time
import numpy as np

print("Loading reference model...")
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

text = "Hello, this is a test of the Qwen three text to speech system powered by the megakernel."

print(f"\nGenerating speech for: '{text}'")
t0 = time.perf_counter()

wavs, sr = model.generate_voice_design(
    text=text,
    instruct="",
    language="English",
    do_sample=True,
    temperature=0.9,
    top_k=50,
    repetition_penalty=1.05,
)

elapsed = time.perf_counter() - t0

wav = wavs[0] if isinstance(wavs, list) else wavs
if isinstance(wav, torch.Tensor):
    wav = wav.cpu().numpy()
wav = wav.flatten()

audio_duration = len(wav) / sr
rtf = elapsed / audio_duration if audio_duration > 0 else float("inf")

print(f"\n=== Reference Model Results ===")
print(f"  Audio samples:  {len(wav)}")
print(f"  Sample rate:    {sr}")
print(f"  Audio duration: {audio_duration:.2f} s")
print(f"  Wall time:      {elapsed*1000:.1f} ms")
print(f"  RTF:            {rtf:.4f}")

import soundfile as sf
sf.write("/workspace/qwen_megakernel/test_reference.wav", wav.astype(np.float32), sr)
print(f"  Saved:          test_reference.wav")
