"""Test RoPE fix with real inputs and NaN detection at every kernel step."""
import torch
import types
import numpy as np
import soundfile as sf

TEXT = "Hello, this is a test."

from qwen_megakernel.tts_model import TTSDecoder, TTS_VOCAB_SIZE

decoder = TTSDecoder(verbose=True)

# Wrap _kernel_step to check for NaN/inf/OOB at every step
_real_step = TTSDecoder._kernel_step
_step_idx = [0]

def _checked_step(self, input_hidden, **kwargs):
    # Check input
    inp_nan = torch.isnan(input_hidden).any().item()
    inp_inf = torch.isinf(input_hidden).any().item()
    inp_norm = input_hidden.float().norm().item()
    
    if inp_nan or inp_inf:
        print(f"  !! step {_step_idx[0]}: INPUT has nan={inp_nan} inf={inp_inf} norm={inp_norm}")
        raise RuntimeError(f"NaN/Inf in input at step {_step_idx[0]}")
    
    token_id, hidden = _real_step(self, input_hidden, **kwargs)
    
    # Check outputs
    out_nan = torch.isnan(self._norm_out).any().item()
    out_inf = torch.isinf(self._norm_out).any().item()
    out_norm = self._norm_out.float().norm().item()
    raw_token = self._out_token.item()
    
    if out_nan or out_inf or raw_token < 0 or raw_token >= TTS_VOCAB_SIZE:
        print(f"  !! step {_step_idx[0]}: OUTPUT nan={out_nan} inf={out_inf} norm={out_norm} raw_token={raw_token} sampled_token={token_id}")
        raise RuntimeError(f"Bad output at step {_step_idx[0]}")
    
    if _step_idx[0] < 15 or _step_idx[0] % 20 == 0:
        print(f"     step {_step_idx[0]:3d}: inp_norm={inp_norm:8.2f} out_norm={out_norm:8.2f} token={token_id}")
    
    _step_idx[0] += 1
    return token_id, hidden

decoder._kernel_step = types.MethodType(_checked_step, decoder)

print("\n=== Real streaming generation (greedy, max 100 steps) ===")
try:
    chunks = []
    for chunk in decoder.generate_speech_streaming(
        TEXT, language="English", chunk_tokens=4,
        max_tokens=100, temperature=0.0, top_k=0,
    ):
        chunks.append(chunk)
    
    if chunks:
        combined = np.concatenate([
            c.flatten() if isinstance(c, np.ndarray) else c.cpu().numpy().flatten()
            for c in chunks
        ]).astype(np.float32)
        sf.write("/workspace/qwen_megakernel/test_ropefixed.wav", combined, 24000)
        print(f"\n  Audio: {len(combined)} samples, {len(combined)/24000:.2f}s")
    else:
        print("\n  No audio generated")
        
except RuntimeError as e:
    print(f"\n  FAILED: {e}")
    print(f"  Last step: {_step_idx[0]}")
