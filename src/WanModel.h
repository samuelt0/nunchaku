#pragma once

#include "common.h"
#include "Tensor.h"
#include "Module.h"
#include "Linear.h"
#include "layernorm.h"

enum class AttentionImpl {
    FlashAttention2 = 0,
    NunchakuFP16,
};

class Attention : public Module {
public:
    static constexpr int POOL_SIZE = 128;
    
    Attention(int num_heads, int dim_head, Device device);
    Tensor forward(Tensor qkv);
    Tensor forward(Tensor qkv, Tensor pool_qkv, float sparsityRatio);
    
    static void setForceFP16(Module *module, bool value);
    
public:
    const int num_heads;
    const int dim_head;
    bool force_fp16;
    
private:
    Tensor cu_seqlens_cpu;
    Tensor headmask_type;
};

class MultiHeadCrossAttention : public Module {
public:
    MultiHeadCrossAttention(int num_heads, int head_dim, bool use_fp4, Tensor::ScalarType dtype, Device device);
    
    Tensor forward(Tensor x, Tensor cond, Tensor cu_seqlens_img, Tensor cu_seqlens_txt);
    
public:
    const int num_heads;
    const int head_dim;
    
private:
    GEMM_W4A4 q_linear;
    GEMM_F16  kv_linear;
    GEMM_W4A4 out_proj;
};

class WanTransformerBlock : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;
    
    WanTransformerBlock(int dim, int ffn_dim, int num_heads, str qk_norm = "rms_norm_across_heads", bool cross_attn_norm=False, float eps=1e-6);
    Tensor forward(Tensor hidden_states, Tensor encoder_hidden_states, Tensor temb, Tensor rotary_emb);
    
public: //here
    const int dim;
    const int dim_head;
    const int num_heads;
    const int mlp_hidden_dim;
    
    AttentionImpl attnImpl = AttentionImpl::FlashAttention2;
    
private:
    AdaLayerNormZeroSingle norm;
    GEMM mlp_fc1;
    GEMM mlp_fc2;
    GEMM qkv_proj;
    RMSNorm norm_q, norm_k;
    Attention attn;
    GEMM out_proj;
};