"""
Qwen3-TTS talker decoder backed by the adapted megakernel.

Loads weights from Qwen3-TTS-12Hz-0.6B-Base, runs the 28-layer backbone
via the CUDA megakernel, and delegates the MTP (code predictor) and speech
tokenizer decode to PyTorch.

Architecture (matches reference Qwen3TTSTalkerForConditionalGeneration):
  Prefill:  text_projection(text_embed(text_ids)) + codec_embed(codec_ids)
            → megakernel (28 layers + norm + LM head)
  Decode:   sum(all 16 codebook embeddings) + trailing_text_hidden
            → megakernel → codebook-0 token + hidden state
            → code_predictor(hidden, codec_embed(codebook-0)) → codebooks 1-15
            → speech_tokenizer.decode(all 16 codebook layers) → audio
"""

import math
import struct
from typing import Generator, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

NUM_LAYERS = 28
NUM_KV_HEADS = 8
NUM_Q_HEADS = 16
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = NUM_Q_HEADS * HEAD_DIM
KV_SIZE = NUM_KV_HEADS * HEAD_DIM
MAX_SEQ_LEN = 4096
TTS_VOCAB_SIZE = 3072
TEXT_HIDDEN_SIZE = 2048
NUM_CODE_GROUPS = 16

CP_NUM_LAYERS = 5
CP_MAX_SEQ_LEN = 32

MROPE_SECTIONS = [24, 20, 20]
ROPE_THETA = 1_000_000.0
SAMPLE_RATE = 24_000
TOKENS_PER_SECOND = 12.5
AUDIO_MS_PER_TOKEN = 1000.0 / TOKENS_PER_SECOND  # 80 ms

# Special token IDs (from model config)
CODEC_EOS_ID = 2150
CODEC_BOS_ID = 2149
CODEC_PAD_ID = 2148
CODEC_THINK_ID = 2154
CODEC_NOTHINK_ID = 2155
CODEC_THINK_BOS_ID = 2156
CODEC_THINK_EOS_ID = 2157

TTS_BOS_TOKEN_ID = 151672
TTS_EOS_TOKEN_ID = 151673
TTS_PAD_TOKEN_ID = 151671


def _build_mrope_inv_freq() -> torch.Tensor:
    """Build inverse frequency table for M-RoPE (64 pairs for 128-dim heads)."""
    return 1.0 / (
        ROPE_THETA
        ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )


def compute_mrope_cos_sin(
    pos_text: int,
    pos_audio_time: int,
    pos_audio_step: int,
    inv_freq: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a merged 128-element cos/sin pair for one decode step.

    Half-split layout: [cos(θ₀)..cos(θ₆₃), cos(θ₀)..cos(θ₆₃)], matching
    the reference model's ``cat(freqs, freqs)`` and the kernel's half-split
    RoPE which pairs dim k with dim k+64 for k < 64.

    M-RoPE sections [24, 20, 20] map frequencies to position axes:
      freq[0:24]  → pos_text
      freq[24:44] → pos_audio_time
      freq[44:64] → pos_audio_step
    """
    positions = torch.zeros(64, dtype=torch.float32)
    sec0, sec1, sec2 = MROPE_SECTIONS
    positions[:sec0] = pos_text
    positions[sec0 : sec0 + sec1] = pos_audio_time
    positions[sec0 + sec1 :] = pos_audio_step

    angles = positions * inv_freq  # [64]
    cos_vals = torch.cos(angles)
    sin_vals = torch.sin(angles)

    cos_step = torch.cat([cos_vals, cos_vals]).to(torch.bfloat16)  # [128]
    sin_step = torch.cat([sin_vals, sin_vals]).to(torch.bfloat16)  # [128]
    return cos_step.cuda().contiguous(), sin_step.cuda().contiguous()


def _pack_layer_weights(layer_weights: list[torch.Tensor], num_layers: int = NUM_LAYERS) -> torch.Tensor:
    """Pack 11-tensor-per-layer flat list into a device blob of LDGLayerWeights structs."""
    ptr_size = 8
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(num_layers * struct_bytes)
    for i in range(num_layers):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    t = torch.frombuffer(buf, dtype=torch.uint8).cuda()
    return t


def load_tts_weights(
    model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    verbose: bool = True,
):
    """
    Load Qwen3-TTS model and extract talker backbone weights for the megakernel.

    Q/K projection and norm weights are permuted from the model's half-split
    RoPE layout to the kernel's interleaved layout.
    """
    if verbose:
        print(f"Loading {model_name} ...")

    from qwen_tts import Qwen3TTSModel

    tts_model = Qwen3TTSModel.from_pretrained(
        model_name,
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )

    model = tts_model.model
    processor = tts_model.processor

    talker = model.talker
    state = talker.state_dict()

    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"model.layers.{i}."

        layer_weights.extend(
            [
                state[p + "input_layernorm.weight"].contiguous(),
                state[p + "self_attn.q_proj.weight"].contiguous(),
                state[p + "self_attn.k_proj.weight"].contiguous(),
                state[p + "self_attn.v_proj.weight"].contiguous(),
                state[p + "self_attn.q_norm.weight"].contiguous(),
                state[p + "self_attn.k_norm.weight"].contiguous(),
                state[p + "self_attn.o_proj.weight"].contiguous(),
                state[p + "post_attention_layernorm.weight"].contiguous(),
                state[p + "mlp.gate_proj.weight"].contiguous(),
                state[p + "mlp.up_proj.weight"].contiguous(),
                state[p + "mlp.down_proj.weight"].contiguous(),
            ]
        )

    final_norm_weight = state["model.norm.weight"].contiguous()
    lm_head_weight = state["codec_head.weight"].contiguous()

    weights = dict(
        layer_weights=layer_weights,
        final_norm_weight=final_norm_weight,
        lm_head_weight=lm_head_weight,
    )

    return weights, tts_model, processor


class TTSDecoder:
    """
    Stateful TTS decoder that uses the megakernel for the talker backbone
    and PyTorch for embeddings, code predictor, and speech tokenizer.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        verbose: bool = True,
    ):
        from qwen_megakernel.build_tts import get_extension

        get_extension()
        self._tts_decode = torch.ops.qwen_tts_megakernel_C.tts_decode

        weights, self._tts_model, self._processor = load_tts_weights(
            model_name, verbose=verbose
        )

        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]

        # Keep references to PyTorch modules (not raw weights)
        self._talker = self._tts_model.model.talker
        self._text_embed_fn = self._talker.get_text_embeddings()    # nn.Embedding(151936, 2048)
        self._codec_embed_fn = self._talker.get_input_embeddings()  # nn.Embedding(3072, 1024)
        self._text_projection = self._talker.text_projection        # MLP: 2048 → 1024
        self._code_predictor = self._talker.code_predictor
        self._speech_tokenizer = self._tts_model.model.speech_tokenizer
        self._st_decoder = self._speech_tokenizer.model.decoder
        self._st_upsample = int(self._st_decoder.total_upsample)

        self._inv_freq = _build_mrope_inv_freq()
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)
        self._position = 0

        # Backbone KV cache
        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        # Scratch buffers (shared between backbone and code predictor)
        f32 = dict(dtype=torch.float32, device="cuda")
        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        self._hidden = torch.empty(HIDDEN_SIZE, **bf16)
        self._act = torch.empty(HIDDEN_SIZE, **f32)
        self._res = torch.empty(HIDDEN_SIZE, **f32)
        self._q = torch.empty(Q_SIZE, **f32)
        self._k = torch.empty(KV_SIZE, **f32)
        self._v = torch.empty(KV_SIZE, **f32)
        self._attn_out = torch.empty(Q_SIZE, **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out = torch.empty(HIDDEN_SIZE, **f32)
        self._bmax_vals = torch.empty(4096, **f32)
        self._bmax_idxs = torch.empty(4096, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

        # Precompute float32 LM head weight (avoid per-call bf16→f32 conversion)
        self._lm_head_weight_f32 = self._lm_head_weight.float()

        # Pre-cache special token embeddings (used every call, saves ~10ms)
        with torch.no_grad():
            special_ids = torch.tensor(
                [[TTS_BOS_TOKEN_ID, TTS_EOS_TOKEN_ID, TTS_PAD_TOKEN_ID]],
                device="cuda",
            )
            special_embed = self._text_projection(self._text_embed_fn(special_ids))
            self._tts_bos_embed = special_embed[:, 0:1].clone()
            self._tts_eos_embed = special_embed[:, 1:2].clone()
            self._tts_pad_embed = special_embed[:, 2:3].clone()

        # Pre-cache suppress mask (saves ~15ms from Python loop)
        self._suppress_mask = torch.zeros(TTS_VOCAB_SIZE, dtype=torch.bool, device="cuda")
        self._suppress_mask[TTS_VOCAB_SIZE - 1024:] = True
        self._suppress_mask[CODEC_EOS_ID] = False

        # --- Code predictor megakernel setup ---
        self._tts_backbone = torch.ops.qwen_tts_megakernel_C.tts_backbone
        self._init_code_predictor_kernel(verbose=verbose)

        self._cancelled = False

        if verbose:
            print("Warming up speech tokenizer decoder...")
        self._warmup_st_decoder()
        self._warmup_sampling()
        if verbose:
            print("TTSDecoder ready.")

    def cancel(self):
        """Signal the decode loop to stop at the next step."""
        self._cancelled = True

    def reset(self):
        self._position = 0
        self._cancelled = False
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self) -> int:
        return self._position

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def _text_proj_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """text_projection(text_embedding(ids)), matching reference model."""
        with torch.no_grad():
            return self._text_projection(self._text_embed_fn(token_ids))

    def _codec_embed(self, token_ids: torch.Tensor) -> torch.Tensor:
        """codec_embedding(ids), matching reference model."""
        with torch.no_grad():
            return self._codec_embed_fn(token_ids)

    def _kernel_step(
        self,
        input_hidden: torch.Tensor,
        temperature: float = 0.0,
        top_k: int = 0,
        repetition_penalty: float = 1.0,
        generated_tokens: Optional[list] = None,
        suppress_tokens: Optional[torch.Tensor] = None,
    ) -> Tuple[int, torch.Tensor]:
        """
        Run one megakernel decode step.

        The kernel runs the 28-layer backbone + LM head argmax. When temperature > 0,
        we recompute the LM head on the host with sampling, repetition penalty, and
        token suppression to match the reference model's generation quality.
        """
        torch.cuda.synchronize()

        pos = self._position
        cos_step, sin_step = compute_mrope_cos_sin(pos, pos, pos, self._inv_freq)
        cache_len = self._position + 1

        self._tts_decode(
            self._out_token,
            input_hidden.contiguous(),
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            cos_step,
            sin_step,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            cache_len,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1

        hidden_state = self._norm_out.clone().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)  # [1, 1, 1024]

        if temperature > 0:
            with torch.no_grad():
                logits = self._norm_out.float() @ self._lm_head_weight.float().t()

                # Suppress special/reserved tokens (ref model suppresses 2048-3071 except EOS)
                if suppress_tokens is not None:
                    logits[suppress_tokens] = float('-inf')

                # Repetition penalty (ref model uses 1.05)
                if repetition_penalty != 1.0 and generated_tokens:
                    prev = torch.tensor(generated_tokens, device=logits.device, dtype=torch.long)
                    prev = prev.unique()
                    prev_logits = logits[prev]
                    logits[prev] = torch.where(
                        prev_logits > 0,
                        prev_logits / repetition_penalty,
                        prev_logits * repetition_penalty,
                    )

                logits /= temperature
                if top_k > 0:
                    tk = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, tk).values[-1]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                probs = F.softmax(logits, dim=-1)
                token_id = torch.multinomial(probs, 1).item()
        else:
            token_id = self._out_token.item()

        return token_id, hidden_state

    def _warmup_st_decoder(self):
        """Warm up and capture CUDA graphs for the speech tokenizer decoder."""
        self._st_graphs = {}
        self._st_static_in = {}
        self._st_static_out = {}
        self._st_graph_stream = torch.cuda.Stream()

        graph_lengths = [1, 2, 6, 10, 14, 18, 22, 26, 29]
        for T in graph_lengths:
            static_in = torch.zeros(1, 16, T, dtype=torch.long, device="cuda")
            with torch.no_grad():
                self._st_decoder(static_in)
                self._st_decoder(static_in)
            torch.cuda.synchronize()

            s = self._st_graph_stream
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s), torch.no_grad():
                self._st_decoder(static_in)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=s):
                static_out = self._st_decoder(static_in)
            torch.cuda.synchronize()

            self._st_graphs[T] = g
            self._st_static_in[T] = static_in
            self._st_static_out[T] = static_out

    def _st_decode_fast(self, codes_in: torch.Tensor) -> torch.Tensor:
        """Run speech tokenizer decoder using CUDA graph when available."""
        T = codes_in.shape[2]
        if T in self._st_graphs:
            self._st_static_in[T].copy_(codes_in)
            self._st_graphs[T].replay()
            return self._st_static_out[T]
        with torch.no_grad():
            return self._st_decoder(codes_in)

    def _warmup_sampling(self):
        """Warm up CUDA kernels for matmul, softmax, multinomial, top-k."""
        with torch.no_grad():
            dummy_h = torch.randn(HIDDEN_SIZE, dtype=torch.float32, device="cuda")
            # Backbone LM head matmul (1024 → 3072)
            logits = dummy_h @ self._lm_head_weight_f32.t()
            # Top-k
            _ = torch.topk(logits, 50)
            logits = logits.masked_fill(logits < logits[0], float('-inf'))
            # Softmax + multinomial
            probs = F.softmax(logits, dim=-1)
            _ = torch.multinomial(probs, 1).item()
            # Code predictor LM head matmul (1024 → 2048)
            logits2 = dummy_h @ self._cp_lm_heads_f32[0].t()
            probs2 = F.softmax(logits2 / 0.9, dim=-1)
            _ = torch.multinomial(probs2, 1).item()
        torch.cuda.synchronize()

    def _init_code_predictor_kernel(self, verbose: bool = True):
        """
        Set up the megakernel for the code predictor (5-layer Qwen3, same dims
        as the 28-layer backbone). Extracts weights, packs them, creates a
        dedicated KV cache, and precomputes RoPE tables.
        """
        cp = self._code_predictor
        inner = cp.model  # Qwen3TTSTalkerCodePredictorModel

        if verbose:
            print("Setting up code predictor megakernel (5 layers)...")

        # Extract layer weights in the same order as the backbone
        cp_layer_weights = []
        for i in range(CP_NUM_LAYERS):
            layer = inner.layers[i]
            cp_layer_weights.extend([
                layer.input_layernorm.weight.contiguous(),
                layer.self_attn.q_proj.weight.contiguous(),
                layer.self_attn.k_proj.weight.contiguous(),
                layer.self_attn.v_proj.weight.contiguous(),
                layer.self_attn.q_norm.weight.contiguous(),
                layer.self_attn.k_norm.weight.contiguous(),
                layer.self_attn.o_proj.weight.contiguous(),
                layer.post_attention_layernorm.weight.contiguous(),
                layer.mlp.gate_proj.weight.contiguous(),
                layer.mlp.up_proj.weight.contiguous(),
                layer.mlp.down_proj.weight.contiguous(),
            ])

        # Keep references alive (pointers are stored in packed blob)
        self._cp_layer_weights_list = cp_layer_weights
        self._cp_layer_weights_packed = _pack_layer_weights(cp_layer_weights, CP_NUM_LAYERS)
        self._cp_final_norm_weight = inner.norm.weight.contiguous()

        # LM heads (15, one per codebook 1-15) and codec embeddings
        self._cp_lm_heads = cp.lm_head  # nn.ModuleList of 15 Linear layers
        self._cp_lm_heads_f32 = [h.weight.float() for h in cp.lm_head]
        self._cp_codec_embeddings = inner.codec_embedding  # nn.ModuleList of 15 Embeddings

        # KV cache for code predictor (reset per backbone step)
        self._cp_k_cache = torch.zeros(
            CP_NUM_LAYERS, NUM_KV_HEADS, CP_MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._cp_v_cache = torch.zeros_like(self._cp_k_cache)

        # Precompute RoPE cos/sin for positions 0..CP_MAX_SEQ_LEN-1
        # Code predictor uses standard 1D RoPE (same theta/head_dim as backbone)
        self._cp_cos_cache = []
        self._cp_sin_cache = []
        for pos in range(CP_MAX_SEQ_LEN):
            cos_s, sin_s = compute_mrope_cos_sin(pos, pos, pos, self._inv_freq)
            self._cp_cos_cache.append(cos_s)
            self._cp_sin_cache.append(sin_s)

        # Warmup: run one full code predictor forward pass through the kernel
        if verbose:
            print("Warming up code predictor kernel...")
        self._cp_k_cache.zero_()
        self._cp_v_cache.zero_()
        dummy_input = torch.zeros(HIDDEN_SIZE, dtype=torch.bfloat16, device="cuda")
        for pos in range(NUM_CODE_GROUPS + 1):
            self._tts_backbone(
                dummy_input, self._cp_layer_weights_packed,
                self._cp_final_norm_weight,
                self._cp_cos_cache[pos], self._cp_sin_cache[pos],
                self._cp_k_cache, self._cp_v_cache,
                self._hidden, self._act, self._res,
                self._q, self._k, self._v, self._attn_out,
                self._mlp_inter, self._norm_out,
                CP_NUM_LAYERS, pos + 1, CP_MAX_SEQ_LEN, self._attn_scale,
            )
        torch.cuda.synchronize()
        self._cp_k_cache.zero_()
        self._cp_v_cache.zero_()
        if verbose:
            print("Code predictor kernel ready.")

    def _run_code_predictor(
        self,
        past_hidden: torch.Tensor,
        codebook0_id: int,
        do_sample: bool = True,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Run code predictor using the megakernel for the 5-layer transformer,
        with PyTorch LM heads for per-codebook sampling.

        All token IDs stay on GPU to avoid CPU-GPU sync overhead.

        Returns:
            (all_codes [16], predictor_sequences [1, 15]) for next-input computation.
        """
        self._cp_k_cache.zero_()
        self._cp_v_cache.zero_()

        hidden_1d = past_hidden.squeeze(0).squeeze(0).contiguous()
        self._tts_backbone(
            hidden_1d, self._cp_layer_weights_packed,
            self._cp_final_norm_weight,
            self._cp_cos_cache[0], self._cp_sin_cache[0],
            self._cp_k_cache, self._cp_v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v, self._attn_out,
            self._mlp_inter, self._norm_out,
            CP_NUM_LAYERS, 1, CP_MAX_SEQ_LEN, self._attn_scale,
        )

        cb0_tensor = torch.tensor([codebook0_id], device="cuda")
        with torch.no_grad():
            cb0_embed = self._codec_embed_fn(cb0_tensor)
        self._tts_backbone(
            cb0_embed.squeeze(0).contiguous(), self._cp_layer_weights_packed,
            self._cp_final_norm_weight,
            self._cp_cos_cache[1], self._cp_sin_cache[1],
            self._cp_k_cache, self._cp_v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v, self._attn_out,
            self._mlp_inter, self._norm_out,
            CP_NUM_LAYERS, 2, CP_MAX_SEQ_LEN, self._attn_scale,
        )

        token_buf = torch.empty(NUM_CODE_GROUPS - 1, dtype=torch.long, device="cuda")

        with torch.no_grad():
            logits = self._norm_out.float() @ self._cp_lm_heads_f32[0].t()
        tok_t = self._cp_sample_gpu(logits, do_sample, temperature, top_k)
        token_buf[0] = tok_t

        for step in range(1, NUM_CODE_GROUPS - 1):
            with torch.no_grad():
                tok_embed = self._cp_codec_embeddings[step - 1](tok_t.unsqueeze(0))

            pos = step + 1
            self._tts_backbone(
                tok_embed.squeeze(0).contiguous(), self._cp_layer_weights_packed,
                self._cp_final_norm_weight,
                self._cp_cos_cache[pos], self._cp_sin_cache[pos],
                self._cp_k_cache, self._cp_v_cache,
                self._hidden, self._act, self._res,
                self._q, self._k, self._v, self._attn_out,
                self._mlp_inter, self._norm_out,
                CP_NUM_LAYERS, pos + 1, CP_MAX_SEQ_LEN, self._attn_scale,
            )

            with torch.no_grad():
                logits = self._norm_out.float() @ self._cp_lm_heads_f32[step].t()
            tok_t = self._cp_sample_gpu(logits, do_sample, temperature, top_k)
            token_buf[step] = tok_t

        all_codes = torch.cat([cb0_tensor, token_buf])
        return all_codes, token_buf.unsqueeze(0)

    @staticmethod
    def _cp_sample_gpu(logits: torch.Tensor, do_sample: bool, temperature: float, top_k: int) -> torch.Tensor:
        """Sample a token, returning a scalar GPU tensor (no CPU sync)."""
        if not do_sample:
            return logits.argmax()
        logits = logits / temperature
        if top_k > 0:
            threshold = torch.topk(logits, min(top_k, logits.shape[-1])).values[-1]
            logits = logits.masked_fill(logits < threshold, float('-inf'))
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, 1).squeeze(0)

    def _compute_next_input(
        self,
        all_codes: torch.Tensor,
        trailing_text_hidden: Optional[torch.Tensor],
        generation_step: int,
        tts_pad_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the input embedding for the next decode step, matching reference:
          input = sum(all 16 codebook embeddings) + text_hidden

        all_codes: [16] tensor of codebook IDs (already on GPU).
        """
        with torch.no_grad():
            c0_embed = self._codec_embed_fn(all_codes[0:1]).unsqueeze(0)  # [1, 1, 1024]

            cp_embed_layers = self._code_predictor.get_input_embeddings()
            codec_sum = c0_embed  # start accumulating
            for i in range(NUM_CODE_GROUPS - 1):
                tok = all_codes[i + 1 : i + 2].unsqueeze(0)  # [1, 1]
                codec_sum = codec_sum + cp_embed_layers[i](tok)

            if trailing_text_hidden is not None and generation_step < trailing_text_hidden.shape[1]:
                text_hidden = trailing_text_hidden[:, generation_step:generation_step+1]
            else:
                text_hidden = tts_pad_embed

            input_embed = codec_sum + text_hidden

        return input_embed.squeeze(0).squeeze(0).contiguous()

    def decode_codes_to_audio(self, codes: list[torch.Tensor]) -> np.ndarray:
        """
        Decode a list of 16-layer codebook tensors into audio.
        Each entry in `codes` is shape [16] (one timestep, all codebook layers).
        """
        if not codes:
            return np.array([], dtype=np.float32)

        stacked = torch.stack(codes, dim=0)  # [T, 16]
        with torch.no_grad():
            wavs, sr = self._speech_tokenizer.decode([{"audio_codes": stacked}])
        return wavs[0] if isinstance(wavs, list) else wavs

    def _decode_chunk(
        self,
        all_codes: list[torch.Tensor],
        decode_cursor: int,
        left_context_size: int = 25,
    ) -> Optional[np.ndarray]:
        """
        Decode only the new codes (from decode_cursor onward) with left context,
        using the low-level decoder directly. Returns new audio samples as numpy,
        or None if nothing to decode.
        """
        T = len(all_codes)
        if T <= decode_cursor:
            return None
        ctx_start = max(0, decode_cursor - left_context_size)
        ctx_frames = decode_cursor - ctx_start

        stacked = torch.stack(all_codes[ctx_start:T], dim=0)       # [ctx+new, 16]
        codes_in = stacked.unsqueeze(0).transpose(1, 2)            # [1, 16, ctx+new]
        codes_in = torch.clamp(codes_in, min=0)

        with torch.no_grad():
            wav = self._st_decode_fast(codes_in)                     # [1, 1, samples]

        skip = ctx_frames * self._st_upsample
        new_wav = wav[0, 0, skip:]
        return new_wav.detach().float().cpu().numpy()

    def generate_speech_streaming(
        self,
        text: str,
        language: str = "English",
        speaker_ref: Optional[str] = None,
        chunk_tokens: int = 4,
        first_chunk_tokens: int = 4,
        max_tokens: int = 2048,
        temperature: float = 0.9,
        top_k: int = 50,
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream audio chunks from text input.

        Yields numpy arrays of float32 audio at 24 kHz.
        First chunk uses `first_chunk_tokens` (default 1) for fast TTFC,
        subsequent chunks use `chunk_tokens` for throughput.
        """
        import time as _time
        _t0 = _time.perf_counter()
        def _log(msg):
            elapsed = (_time.perf_counter() - _t0) * 1000
            print(f"  [{elapsed:8.1f} ms] {msg}", flush=True)

        self.reset()

        # --- Tokenize text ---
        input_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        input_ids = self._processor(
            text=input_text, return_tensors="pt", padding=False
        )["input_ids"].to("cuda")  # [1, seq_len]
        _log(f"Tokenized: {input_ids.shape[1]} tokens")

        # --- Speaker embedding (for voice cloning) ---
        spk_embed = None
        if speaker_ref is not None:
            try:
                import librosa
                wav, sr = librosa.load(speaker_ref, sr=None, mono=True)
                target_sr = self._tts_model.model.speaker_encoder_sample_rate
                if sr != target_sr:
                    wav = librosa.resample(y=wav, orig_sr=sr, target_sr=target_sr)
                spk_embed = self._tts_model.model.extract_speaker_embedding(
                    audio=wav, sr=target_sr
                )
            except Exception:
                pass

        # --- Language ID ---
        lang_map = {
            'english': 2050, 'chinese': 2055, 'german': 2053, 'italian': 2070,
            'portuguese': 2071, 'spanish': 2054, 'japanese': 2058, 'korean': 2064,
            'french': 2061, 'russian': 2069,
        }
        language_id = lang_map.get(language.lower())
        _log(f"Language: {language} -> id={language_id}")

        # --- Build prefill embeddings (streaming text mode) ---
        tts_bos_embed = self._tts_bos_embed
        tts_eos_embed = self._tts_eos_embed
        tts_pad_embed = self._tts_pad_embed
        suppress_mask = self._suppress_mask

        with torch.no_grad():
            role_embed = self._text_proj_embed(input_ids[:, :3])

            if language_id is None:
                codec_prefill_ids = [[CODEC_NOTHINK_ID, CODEC_THINK_BOS_ID, CODEC_THINK_EOS_ID]]
            else:
                codec_prefill_ids = [[CODEC_THINK_ID, CODEC_THINK_BOS_ID, language_id, CODEC_THINK_EOS_ID]]

            codec_embed_0 = self._codec_embed(
                torch.tensor(codec_prefill_ids, device="cuda")
            )
            codec_embed_1 = self._codec_embed(
                torch.tensor([[CODEC_PAD_ID, CODEC_BOS_ID]], device="cuda")
            )

            if spk_embed is not None:
                codec_input_embed = torch.cat(
                    [codec_embed_0, spk_embed.view(1, 1, -1), codec_embed_1], dim=1
                )
            else:
                codec_input_embed = torch.cat([codec_embed_0, codec_embed_1], dim=1)

            n_codec = codec_input_embed.shape[1]
            text_track = torch.cat([
                tts_pad_embed.expand(-1, n_codec - 2, -1),
                tts_bos_embed,
            ], dim=1)

            dual_track_prefill = text_track + codec_input_embed[:, :-1]
            first_text_embed = self._text_proj_embed(input_ids[:, 3:4]) + codec_input_embed[:, -1:]
            prefill_embeds = torch.cat([role_embed, dual_track_prefill, first_text_embed], dim=1)

            remaining_text_embeds = self._text_proj_embed(input_ids[:, 4:-5])
            trailing_text_hidden = torch.cat([remaining_text_embeds, tts_eos_embed], dim=1)

        _log(f"Built prefill embeds: {prefill_embeds.shape[1]} positions, trailing text: {trailing_text_hidden.shape[1]}")

        # --- Prefill (optimized: backbone-only kernel, no LM head, no per-step sync) ---
        prefill_len = prefill_embeds.shape[1]
        _log(f"Prefill starting ({prefill_len} steps)")

        # Precompute cos/sin for all prefill positions
        prefill_cos = []
        prefill_sin = []
        for i in range(prefill_len):
            c, s = compute_mrope_cos_sin(i, i, i, self._inv_freq)
            prefill_cos.append(c)
            prefill_sin.append(s)

        # Process all but last position with backbone-only kernel (no LM head, no sync)
        for i in range(prefill_len - 1):
            self._tts_backbone(
                prefill_embeds[0, i].contiguous(),
                self._layer_weights_packed,
                self._final_norm_weight,
                prefill_cos[i], prefill_sin[i],
                self._k_cache, self._v_cache,
                self._hidden, self._act, self._res,
                self._q, self._k, self._v, self._attn_out,
                self._mlp_inter, self._norm_out,
                NUM_LAYERS, i + 1, MAX_SEQ_LEN, self._attn_scale,
            )
        self._position = prefill_len - 1

        # Last prefill step: backbone-only (we'll do LM head in PyTorch for sampling)
        last_pos = prefill_len - 1
        self._tts_backbone(
            prefill_embeds[0, last_pos].contiguous(),
            self._layer_weights_packed,
            self._final_norm_weight,
            prefill_cos[last_pos], prefill_sin[last_pos],
            self._k_cache, self._v_cache,
            self._hidden, self._act, self._res,
            self._q, self._k, self._v, self._attn_out,
            self._mlp_inter, self._norm_out,
            NUM_LAYERS, last_pos + 1, MAX_SEQ_LEN, self._attn_scale,
        )
        self._position = prefill_len
        torch.cuda.synchronize()
        _log("Prefill kernels done")

        # Sample first codebook-0 token from the final prefill hidden state
        with torch.no_grad():
            logits = self._norm_out.float() @ self._lm_head_weight_f32.t()
            if suppress_mask is not None:
                logits[suppress_mask] = float('-inf')
            logits /= temperature
            if top_k > 0:
                tk = min(top_k, logits.size(-1))
                threshold = torch.topk(logits, tk).values[-1]
                logits = logits.masked_fill(logits < threshold, float('-inf'))
            probs = F.softmax(logits, dim=-1)
            first_codebook0 = torch.multinomial(probs, 1).item()
        past_hidden = self._norm_out.clone().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)

        _log(f"Prefill done ({prefill_len} steps), first codebook0={first_codebook0}")

        if first_codebook0 == CODEC_EOS_ID:
            _log("EOS immediately after prefill — nothing to generate")
            return

        # --- Autoregressive decode ---
        prev_codebook0 = first_codebook0
        all_codes_accumulated: list[torch.Tensor] = []
        decode_cursor = 0
        codes_since_last_decode = 0
        generation_step = 0
        decode_count = 0
        generated_tokens: list[int] = [first_codebook0]
        left_ctx = 25
        repeat_count = 0
        max_repeat = 30

        for _ in range(max_tokens):
            if self._cancelled:
                _log("Cancelled by caller")
                break

            _cp_t0 = _time.perf_counter()
            all_codes, pred_seq = self._run_code_predictor(past_hidden, prev_codebook0)
            _cp_ms = (_time.perf_counter() - _cp_t0) * 1000
            all_codes_accumulated.append(all_codes)
            codes_since_last_decode += 1

            if decode_count < 3 or decode_count % 50 == 0:
                _log(f"Decode step {decode_count}: codebook0={prev_codebook0}, codes={all_codes[:4].tolist()}..., cp={_cp_ms:.1f}ms")

            current_chunk_size = first_chunk_tokens if decode_count <= first_chunk_tokens else chunk_tokens
            if codes_since_last_decode >= current_chunk_size:
                T = len(all_codes_accumulated)
                ctx_start = max(0, decode_cursor - left_ctx)
                ctx_frames = decode_cursor - ctx_start
                stacked = torch.stack(all_codes_accumulated[ctx_start:T], dim=0)
                codes_in = stacked.unsqueeze(0).transpose(1, 2)
                codes_in = torch.clamp(codes_in, min=0)

                wav_out = self._st_decode_fast(codes_in)
                skip = ctx_frames * self._st_upsample
                audio_np = wav_out[0, 0, skip:].detach().float().cpu().numpy()
                if len(audio_np) > 0:
                    yield audio_np

                decode_cursor = T
                codes_since_last_decode = 0

            input_hidden = self._compute_next_input(
                all_codes,
                trailing_text_hidden, generation_step, tts_pad_embed,
            )
            generation_step += 1
            decode_count += 1

            if decode_count < 2:
                early_mask = suppress_mask.clone()
                early_mask[CODEC_EOS_ID] = True
            else:
                early_mask = suppress_mask

            # Run backbone-only kernel (no LM head needed, we do sampling in PyTorch)
            pos = self._position
            cos_s, sin_s = compute_mrope_cos_sin(pos, pos, pos, self._inv_freq)
            self._tts_backbone(
                input_hidden,
                self._layer_weights_packed,
                self._final_norm_weight,
                cos_s, sin_s,
                self._k_cache, self._v_cache,
                self._hidden, self._act, self._res,
                self._q, self._k, self._v, self._attn_out,
                self._mlp_inter, self._norm_out,
                NUM_LAYERS, pos + 1, MAX_SEQ_LEN, self._attn_scale,
            )
            self._position += 1
            # No device-wide sync — matmul is on same stream, .item() syncs current stream only

            # Sample with temperature/repetition penalty
            with torch.no_grad():
                logits = self._norm_out.float() @ self._lm_head_weight_f32.t()
                if early_mask is not None:
                    logits[early_mask] = float('-inf')
                if 1.05 != 1.0 and generated_tokens:
                    prev = torch.tensor(generated_tokens, device=logits.device, dtype=torch.long).unique()
                    prev_logits = logits[prev]
                    logits[prev] = torch.where(
                        prev_logits > 0,
                        prev_logits / 1.05,
                        prev_logits * 1.05,
                    )
                logits /= temperature
                if top_k > 0:
                    tk = min(top_k, logits.size(-1))
                    threshold = torch.topk(logits, tk).values[-1]
                    logits = logits.masked_fill(logits < threshold, float('-inf'))
                probs = F.softmax(logits, dim=-1)
                codebook0_token = torch.multinomial(probs, 1).item()

            past_hidden = self._norm_out.clone().to(torch.bfloat16).unsqueeze(0).unsqueeze(0)

            if codebook0_token == CODEC_EOS_ID:
                _log(f"EOS at decode step {decode_count}")
                break

            if codebook0_token == prev_codebook0:
                repeat_count += 1
                if repeat_count >= max_repeat:
                    _log(f"Repetition loop detected (token {codebook0_token} repeated {repeat_count}x), stopping at step {decode_count}")
                    break
            else:
                repeat_count = 0

            generated_tokens.append(codebook0_token)
            prev_codebook0 = codebook0_token

        # Final decode of remaining un-decoded codes
        if len(all_codes_accumulated) > decode_cursor:
            final_audio = self._decode_chunk(all_codes_accumulated, decode_cursor, left_ctx)
            if final_audio is not None and len(final_audio) > 0:
                yield final_audio

        _log(f"Done: {decode_count} decode steps total")
