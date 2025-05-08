#pragma once

#include "common.h"
#include "Tensor.h"
#include "Module.h"
#include "Linear.h"
#include "layernorm.h"


class WanSelfAttention : public Module {
    public:
    WanSelfAttention(int dim, int num_heads, bool bias, bool use_fp4, Tensor::ScalarType dtype, Device device);
    
    Tensor forward(Tensor x, Tensor out = {});
    
    public:
        const int dim;
        const int dim_pad;  
        const int num_heads;
        const int head_dim;
    
    private:
        GEMM_W4A4 qkv_proj;
        GEMM_W4A4 out_proj;
    
        std::optional<LayerNorm> norm_q;
        std::optional<LayerNorm> norm_k;
};
    
class WanCrossAttention : public Module {
    public:
        WanCrossAttention(int dim, int num_heads, int added_kv_proj_dim, bool bias, bool added_proj_bias, bool use_fp4, Tensor::ScalarType dtype, Device device);

        Tensor forward(Tensor x, Tensor cond, Tensor cu_seqlens_img, Tensor cu_seqlens_txt);

    public:
        const int num_heads;
        const int head_dim;
        const int dim;
    private:
        GEMM_W4A4 q_linear;
        GEMM_F16  kv_linear;
        GEMM_W4A4 out_proj;
        std::optional<GEMM_W4A4> add_k_proj;
        std::optional<GEMM_W4A4> add_v_proj;

        std::optional<LayerNorm> norm_q;
        std::optional<LayerNorm> norm_k;
        std::optional<LayerNorm> norm_added_k;
};

class WanTransformerBlock : public Module {
    public:
        WanTransformerBlock(int hidden_size, int intermediate_size, int num_heads, bool use_fp4, Tensor::ScalarType dtype, Device device);
    
        Tensor forward(Tensor hidden_states, Tensor encoder_hidden_states, Tensor timestep, Tensor rotary_emb, Tensor cu_seqlens_img = {}, Tensor cu_seqlens_txt = {});
    
    public:
        const int hidden_size;
        const int num_heads;
    
    private:
        Tensor scale_shift_table;
        // Tensor ones;
    
        WanSelfAttention attn1;
        WanCrossAttention attn2;
    
        LayerNorm norm1, norm2, norm3;
        GEMM_W4A4 mlp_fc1;
        GEMM_W4A4 mlp_fc2;
};

struct WanConfig {
    int num_layers;
    int num_attention_heads;    
    int attention_head_dim;
    //int num_cross_attention_heads;
    int ffn_dim;                    
    int in_channels;              
    int out_channels;              
    int text_dim;          
    int freq_dim;        
    std::tuple<int, int, int> patch_size;
    bool cross_attn_norm;
    bool use_fp4;
};

class WanModel : public Module {
public:
    WanModel(WanConfig config, Tensor::ScalarType dtype, Device device);
    Tensor forward(Tensor hidden_states,
        Tensor timestep,
        Tensor encoder_hidden_states,
        Tensor encoder_hidden_states_image = {},
        Tensor attention_kwargs = {});
public:
    const WanConfig config;

public:
    std::vector<std::unique_ptr<WanTransformerBlock>> blocks;
    LayerNorm norm_out;
    GEMM_W4A4 proj_out;
    Tensor scale_shift_table;
};