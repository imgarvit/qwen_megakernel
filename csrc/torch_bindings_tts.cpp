#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdint>

#define _CONCAT(A, B) A##B
#define CONCAT(A, B) _CONCAT(A, B)
#define _STRINGIFY(A) #A
#define STRINGIFY(A) _STRINGIFY(A)

#define TORCH_LIBRARY_EXPAND(NAME, MODULE) TORCH_LIBRARY(NAME, MODULE)

#define REGISTER_EXTENSION(NAME) \
  PyMODINIT_FUNC CONCAT(PyInit_, NAME)() { \
    static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, \
                                        STRINGIFY(NAME), nullptr, 0, nullptr}; \
    return PyModule_Create(&module); \
  }

struct LDGLayerWeights {
  const void *input_layernorm_weight;
  const void *q_proj_weight;
  const void *k_proj_weight;
  const void *v_proj_weight;
  const void *q_norm_weight;
  const void *k_norm_weight;
  const void *o_proj_weight;
  const void *post_attn_layernorm_weight;
  const void *gate_proj_weight;
  const void *up_proj_weight;
  const void *down_proj_weight;
};

extern "C" void launch_ldg_tts_decode(
    const void *input_hidden, int *output_token_id,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *lm_head_weight, const void *cos_step, const void *sin_step,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized, void *block_max_vals,
    void *block_max_idxs, int num_layers, int cache_len, int max_seq_len,
    float attn_scale, cudaStream_t stream);

void tts_decode(torch::Tensor output_token, torch::Tensor input_hidden,
                torch::Tensor layer_weights_packed,
                torch::Tensor final_norm_weight, torch::Tensor lm_head_weight,
                torch::Tensor cos_step, torch::Tensor sin_step,
                torch::Tensor k_cache, torch::Tensor v_cache,
                torch::Tensor hidden_buffer, torch::Tensor activations,
                torch::Tensor residual, torch::Tensor q, torch::Tensor k,
                torch::Tensor v, torch::Tensor attn_out,
                torch::Tensor mlp_intermediate, torch::Tensor normalized,
                torch::Tensor block_max_vals, torch::Tensor block_max_idxs,
                int64_t num_layers, int64_t cache_len, int64_t max_seq_len,
                double attn_scale) {
  launch_ldg_tts_decode(
      input_hidden.data_ptr(),
      static_cast<int *>(output_token.data_ptr()),
      reinterpret_cast<const LDGLayerWeights *>(
          layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(), lm_head_weight.data_ptr(),
      cos_step.data_ptr(), sin_step.data_ptr(), k_cache.data_ptr(),
      v_cache.data_ptr(), hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), normalized.data_ptr(),
      block_max_vals.data_ptr(), block_max_idxs.data_ptr(),
      static_cast<int>(num_layers), static_cast<int>(cache_len),
      static_cast<int>(max_seq_len), static_cast<float>(attn_scale),
      c10::cuda::getCurrentCUDAStream().stream());
}

extern "C" void launch_ldg_tts_backbone_only(
    const void *input_hidden,
    const LDGLayerWeights *layer_weights, const void *final_norm_weight,
    const void *cos_step, const void *sin_step,
    void *k_cache, void *v_cache, void *hidden_buffer, void *g_activations,
    void *g_residual, void *g_q, void *g_k, void *g_v, void *g_attn_out,
    void *g_mlp_intermediate, void *g_normalized,
    int num_layers, int cache_len, int max_seq_len,
    float attn_scale, cudaStream_t stream);

void tts_backbone(torch::Tensor input_hidden,
                  torch::Tensor layer_weights_packed,
                  torch::Tensor final_norm_weight,
                  torch::Tensor cos_step, torch::Tensor sin_step,
                  torch::Tensor k_cache, torch::Tensor v_cache,
                  torch::Tensor hidden_buffer, torch::Tensor activations,
                  torch::Tensor residual, torch::Tensor q, torch::Tensor k,
                  torch::Tensor v, torch::Tensor attn_out,
                  torch::Tensor mlp_intermediate, torch::Tensor normalized,
                  int64_t num_layers, int64_t cache_len, int64_t max_seq_len,
                  double attn_scale) {
  launch_ldg_tts_backbone_only(
      input_hidden.data_ptr(),
      reinterpret_cast<const LDGLayerWeights *>(
          layer_weights_packed.data_ptr()),
      final_norm_weight.data_ptr(),
      cos_step.data_ptr(), sin_step.data_ptr(),
      k_cache.data_ptr(), v_cache.data_ptr(),
      hidden_buffer.data_ptr(), activations.data_ptr(),
      residual.data_ptr(), q.data_ptr(), k.data_ptr(), v.data_ptr(),
      attn_out.data_ptr(), mlp_intermediate.data_ptr(), normalized.data_ptr(),
      static_cast<int>(num_layers), static_cast<int>(cache_len),
      static_cast<int>(max_seq_len), static_cast<float>(attn_scale),
      c10::cuda::getCurrentCUDAStream().stream());
}

TORCH_LIBRARY_EXPAND(TORCH_EXTENSION_NAME, ops) {
  ops.def("tts_decode(Tensor output_token, Tensor input_hidden, "
          "Tensor layer_weights_packed, "
          "Tensor final_norm_weight, Tensor lm_head_weight, "
          "Tensor cos_step, Tensor sin_step, "
          "Tensor k_cache, Tensor v_cache, "
          "Tensor hidden_buffer, Tensor activations, Tensor residual, "
          "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
          "Tensor mlp_intermediate, Tensor normalized, "
          "Tensor block_max_vals, Tensor block_max_idxs, "
          "int num_layers, int cache_len, int max_seq_len, "
          "float attn_scale) -> ()");
  ops.impl("tts_decode", torch::kCUDA, &tts_decode);

  ops.def("tts_backbone(Tensor input_hidden, "
          "Tensor layer_weights_packed, "
          "Tensor final_norm_weight, "
          "Tensor cos_step, Tensor sin_step, "
          "Tensor k_cache, Tensor v_cache, "
          "Tensor hidden_buffer, Tensor activations, Tensor residual, "
          "Tensor q, Tensor k, Tensor v, Tensor attn_out, "
          "Tensor mlp_intermediate, Tensor normalized, "
          "int num_layers, int cache_len, int max_seq_len, "
          "float attn_scale) -> ()");
  ops.impl("tts_backbone", torch::kCUDA, &tts_backbone);
}

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
