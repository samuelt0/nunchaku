#pragma once

#include "common.h"
#include "Tensor.h"
#include "Module.h"
#include "Linear.h"
#include "layernorm.h"

enum class WanAttentionImpl { FlashAttention2 = 0, NunchakuFP16 };

struct WanConfig {
    std::array<int, 3> patch_size{1, 2, 2};
    int num_layers = 40;
    int num_heads = 40;
    int head_dim = 128;
    int ffn_dim = 13824;
    int in_channels = 16;
    int out_channels = 16;
    int text_dim = 4096;
    int freq_dim = 256;
    int rope_max_seq_len = 1024;
    bool cross_attn_norm = true;
    bool qk_norm = true;
    float eps = 1e-6f;
    int image_dim = 0;
    int added_kv_proj_dim = 0;
    bool use_fp4 = true;
};

class WanRotaryPosEmbed : public Module {
public:
    WanRotaryPosEmbed(int head_dim, std::array<int, 3> patch, int max_seq_len, float theta, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor hidden_states);
private:
    const int head_dim;
    const std::array<int, 3> patch;
    const int max_seq_len;
    const float theta;
    Tensor freqs;
};

class Conv3dPatchEmbed : public Module {
public:
    Conv3dPatchEmbed(int in_ch, int out_ch, std::array<int, 3> kernel, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor x);
private:
    Tensor weight;
    Tensor bias;
    const std::array<int, 3> kernel;
};

class WanTimeTextImageEmbedding : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;
    WanTimeTextImageEmbedding(int inner_dim, int time_freq_dim, int text_dim, int image_dim, int time_proj_dim, int pos_embed_seq_len, bool use_fp4, Tensor::ScalarType dtype, Device device);
    std::tuple<Tensor, Tensor, Tensor, Tensor> forward(Tensor timestep, Tensor txt_tokens, Tensor img_tokens);
private:
    GEMM time_fc1;
    GEMM time_fc2;
    GEMM time_proj;
    GEMM text_fc1;
    GEMM text_fc2;
    std::optional<std::unique_ptr<Module>> image_embedder;
};

class WanTransformerBlock : public Module {
public:
    static constexpr bool USE_4BIT = true;
    using GEMM = std::conditional_t<USE_4BIT, GEMM_W4A4, GEMM_W8A8>;
    WanTransformerBlock(int dim, int ffn_dim, int num_heads, bool cross_attn_norm, int added_kv_proj_dim, bool use_fp4, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor hidden_states, Tensor encoder_hidden_states, Tensor temb, Tensor rotary_emb);
public:
    const int dim;
    const int num_heads;
    const int dim_head;
    WanAttentionImpl attnImpl = WanAttentionImpl::FlashAttention2;
private:
    LayerNorm norm1;
    LayerNorm norm2;
    LayerNorm norm3;
    GEMM self_qkv_proj;
    GEMM self_out_proj;
    GEMM cross_qkv_proj;
    GEMM cross_out_proj;
    std::optional<GEMM> added_k_proj;
    std::optional<GEMM> added_v_proj;
    GEMM ffn_fc1;
    GEMM ffn_fc2;
    Tensor scale_shift_table;
};

class WanModel : public Module {
public:
    WanModel(const WanConfig &cfg, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor hidden_states, Tensor timestep, Tensor txt_tokens, Tensor img_tokens = {}, bool skip_first_layer = false);
    void setAttentionImpl(WanAttentionImpl impl);
public:
    WanConfig config;
private:
    std::unique_ptr<WanRotaryPosEmbed> rope;
    std::unique_ptr<Conv3dPatchEmbed> patch_embedding;
    std::unique_ptr<WanTimeTextImageEmbedding> cond_embedder;
    std::vector<std::unique_ptr<WanTransformerBlock>> blocks;
    LayerNorm norm_out;
    GEMM_W4A4 proj_out;
    Tensor scale_shift_table;
    WanAttentionImpl attnImpl = WanAttentionImpl::FlashAttention2;
    bool use_fp4;
};
