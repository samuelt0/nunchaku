#include <iostream>

#include "WanModel.h"
#include "kernels/zgemm/zgemm.h"
#include "flash_api.h"
#include "kernels/misc_kernels.h"
#include "kernels/gemm_batched.h"
#include "activation.h"

#include <nvtx3/nvToolsExt.h>

using spdlog::fmt_lib::format;
using namespace nunchaku;


WanSelfAttention::WanSelfAttention(int dim, int num_heads, bool bias, bool use_fp4, Tensor::ScalarType dtype, Device device)
    : dim(dim),
      dim_pad(ceilDiv(dim, 128) * 128),
      num_heads(num_heads),
      head_dim(dim_pad / num_heads),
      qkv_proj(dim, dim_pad * 3, bias, use_fp4, dtype, device),
      out_proj(dim_pad, dim, bias, use_fp4, dtype, device),
      norm_q(std::nullopt),
      norm_k(std::nullopt)
{
    registerChildren(qkv_proj, "qkv_proj");
    registerChildren(out_proj, "out_proj");
}

Tensor WanSelfAttention::forward(Tensor x, Tensor out) {
    constexpr int MIN_PAD = 256; 

    assert(x.ndims() == 3);
    const int B = x.shape[0];
    const int N = x.shape[1];
    const int C = x.shape[2];
    assert(C == dim);

    const int N_pad = ceilDiv(N, MIN_PAD) * MIN_PAD;

    if (N_pad != N) {
        spdlog::debug("WanSelfAttention: padding seq from {} to {}", N, N_pad);
        Tensor x_pad = Tensor::allocate({B, N_pad, dim}, x.dtype(), x.device());
        x_pad.zero_();
        for (int i = 0; i < B; ++i) {
            x_pad.slice(0, i, i + 1).slice(1, 0, N).copy_(x.slice(0, i, i + 1));
        }
        x = x_pad;
    }

    auto qkv_act = qkv_proj.quantize(x, false);

    Tensor qkv_out = Tensor::allocate({B, N_pad, dim_pad * 3}, x.dtype(), x.device());

    kernels::gemm_w4a4(
        qkv_act.act,
        qkv_proj.qweight,
        {}, {}, qkv_act.ascales, qkv_proj.wscales,
        {}, {}, qkv_act.lora_act, qkv_proj.lora_up, {}, {}, {}, {}, {},
        qkv_proj.bias, {},
        {}, qkv_out,
        qkv_act.is_unsigned, qkv_proj.lora_scales, false,
        qkv_proj.use_fp4,
        *qkv_proj.wtscale.data_ptr<float>(),
        qkv_proj.wcscales.numel() > 0 ? qkv_proj.wcscales : Tensor{},
        {}, {}, {}, 0
    );
    Tensor q = qkv_out.slice(2, 0, dim_pad);
    Tensor k = qkv_out.slice(2, dim_pad, dim_pad * 2);
    Tensor v = qkv_out.slice(2, dim_pad * 2, dim_pad * 3);

    q = q.view({B, N_pad, num_heads, head_dim}).view({B * num_heads, N_pad, head_dim});
    k = k.view({B, N_pad, num_heads, head_dim}).view({B * num_heads, N_pad, head_dim});
    v = v.view({B, N_pad, num_heads, head_dim}).view({B * num_heads, N_pad, head_dim});

    if (norm_q.has_value()) q = norm_q.value().forward(q);
    if (norm_k.has_value()) k = norm_k.value().forward(k);

    Tensor attn_output = mha_fwd(
        q, k, v,
        0.0f,                         
        1.0f / std::sqrt((float)head_dim),  
        false, -1, -1, false 
    ).front();

    attn_output = attn_output.view({B * num_heads, N_pad, head_dim}).view({B, N_pad, dim_pad});

    if (N_pad != N) {
        Tensor unpad = Tensor::allocate({B, N, dim_pad}, attn_output.dtype(), attn_output.device());
        for (int i = 0; i < B; ++i) {
            unpad.slice(0, i, i + 1).copy_(attn_output.slice(0, i, i + 1).slice(1, 0, N));
        }
        attn_output = unpad;
    }

    if (!out.valid()) {
        out = Tensor::allocate({B, N, dim}, attn_output.dtype(), attn_output.device());
    }

    out_proj.forward(attn_output, out);
    return out;
}


WanCrossAttention::WanCrossAttention(int dim, int num_heads, int added_kv_proj_dim, bool bias, 
    bool added_proj_bias, bool use_fp4, Tensor::ScalarType dtype, Device device) :
    dim(dim),
    num_heads(num_heads),
    head_dim(dim / num_heads),
    q_linear(dim, dim, bias, use_fp4, dtype, device),
    kv_linear(dim, dim * 2, bias, dtype, device),
    out_proj(dim, dim, bias, use_fp4, dtype, device),
    add_k_proj(std::nullopt),
    add_v_proj(std::nullopt),
    norm_q(std::nullopt),
    norm_k(std::nullopt),
    norm_added_k(std::nullopt)
{
    registerChildren(q_linear, "q_linear");
    registerChildren(kv_linear, "kv_linear");
    registerChildren(out_proj, "out_proj");
}

Tensor WanCrossAttention::forward(Tensor x, Tensor cond, Tensor cu_seqlens_img, Tensor cu_seqlens_txt) {
    const int B = x.shape[0];
    const int N = x.shape[1];         
    const int total_M = cond.shape[0];

    Tensor q = q_linear.forward(x);               
    q = q.view({B, N, num_heads, head_dim}).view({B * num_heads, N, head_dim});

    Tensor kv = kv_linear.forward(cond);             
    kv = kv.view({total_M, num_heads * 2, head_dim}); 

    Tensor k = kv.slice(1, 0, num_heads);            
    Tensor v = kv.slice(1, num_heads, 2 * num_heads); 

    if (norm_q.has_value()) {
        q = norm_q.value().forward(q);
    }
    if (norm_k.has_value()) {
        k = norm_k.value().forward(k);
    }

    Tensor hidden_states_img;

    if (add_k_proj.has_value() && add_v_proj.has_value()) {
        int img_tokens = total_M - 512;
        Tensor cond_img = cond.slice(0, 0, img_tokens);
        Tensor k_img = add_k_proj->forward(cond_img);
        k_img = k_img.view({img_tokens, num_heads, head_dim});
        if (norm_added_k.has_value()) {
            k_img = norm_added_k->forward(k_img);
        }

        Tensor v_img = add_v_proj->forward(cond_img);
        v_img = v_img.view({img_tokens, num_heads, head_dim});

        hidden_states_img = mha_varlen_fwd(
            q, k_img, v_img,
            cu_seqlens_img, cu_seqlens_txt,
            N, img_tokens,
            0.0f,
            1.0f / std::sqrt((float)head_dim),
            false, false,
            -1, -1,
            false
        ).front();
        hidden_states_img = hidden_states_img.view({B, N, num_heads * head_dim});
    }

    Tensor attn_output = mha_varlen_fwd(
        q, k, v,
        cu_seqlens_img, cu_seqlens_txt,
        N, total_M,
        0.0f,
        1.0f / std::sqrt((float)head_dim),
        false, false,
        -1, -1,
        false
    ).front(); // [B * H, N, D]
    attn_output = attn_output.view({B, N, num_heads * head_dim});

    // if (hidden_states_img.valid()) {
    //     attn_output = attn_output.add(hidden_states_img);
    // }

    return out_proj.forward(attn_output);
}


WanTransformerBlock::WanTransformerBlock(int hidden_size, int intermediate_size, int num_heads, bool use_fp4, Tensor::ScalarType dtype, Device device) :
    hidden_size(hidden_size), num_heads(num_heads),
    attn1(hidden_size, num_heads, true, use_fp4, dtype, device),
    attn2(hidden_size, num_heads, 512, true, true, use_fp4, dtype, device),
    mlp_fc1(hidden_size, intermediate_size, true, use_fp4, dtype, device),
    mlp_fc2(intermediate_size, hidden_size, true, use_fp4, dtype, device),
    norm1(hidden_size, 1e-6, false, dtype, device),
    norm2(hidden_size, 1e-6, false, dtype, device),
    norm3(hidden_size, 1e-6, false, dtype, device)
{
    this->scale_shift_table = Tensor::allocate({1, 6, hidden_size}, dtype, device);

    registerChildren(attn1, "attn1");
    registerChildren(attn2, "attn2");
    registerChildren(mlp_fc1, "mlp_fc1");
    registerChildren(mlp_fc1, "mlp_fc1");
    registerChildren(norm1, "norm1");
    registerChildren(norm2, "norm2");
    registerChildren(norm3, "norm3");

    registerParams
        (this->scale_shift_table, "scale_shift_table")
    ;
}

Tensor WanTransformerBlock::forward(Tensor hidden_states, Tensor encoder_hidden_states, Tensor temb, Tensor rotary_emb, Tensor cu_seqlens_img, Tensor cu_seqlens_txt) {
    nvtxRangePushA("WanTransformerBlock");

    const int batch_size = hidden_states.shape[0];
    const int dim = hidden_size;

    // [B, 6, C] = temb + scale_shift_table
    temb = temb.copy(temb.device()).view({batch_size, 6, dim});
    kernels::mul_add_batch(temb, {}, false, 0, scale_shift_table, false);

    std::array<Tensor, 6> chunked;
    for (int i = 0; i < 6; ++i) {
        chunked[i] = temb.slice(1, i, i + 1);
    }
    auto &&[shift_msa, scale_msa, gate_msa, shift_ff, scale_ff, gate_ff] = chunked;

    {
        nvtxRangePushA("SelfAttention");

        Tensor normed = norm1.forward(hidden_states);
        kernels::mul_add_batch(normed, scale_msa, true, 1, shift_msa, true);

        Tensor attn_out = attn1.forward(normed, rotary_emb);
        kernels::mul_add_batch(attn_out, gate_msa, true, 0.0f, hidden_states, true);

        hidden_states = attn_out;

        nvtxRangePop();
    }

    {
        nvtxRangePushA("CrossAttention");

        Tensor normed = norm2.forward(hidden_states);
        Tensor attn_out = attn2.forward(normed, encoder_hidden_states, {}, {});

        kernels::mul_add_batch(attn_out, {}, false, 0.0f, hidden_states, true);
        hidden_states = attn_out;

        nvtxRangePop();
    }
    {
        nvtxRangePushA("FeedForward");

        Tensor normed = norm3.forward(hidden_states);
        kernels::mul_add_batch(normed, scale_ff, true, 1, shift_ff, true);
        
        Tensor act = mlp_fc1.forward_silu(normed);
        Tensor ff_out = mlp_fc2.forward(act);

        kernels::mul_add_batch(ff_out, gate_ff, true, 0.0f, hidden_states, true);
        hidden_states = ff_out;

        nvtxRangePop();
    }

    nvtxRangePop();
    return hidden_states;
}


WanModel::WanModel(WanConfig config, Tensor::ScalarType dtype, Device device)
    : config(config),
      norm_out(config.num_attention_heads * config.attention_head_dim, 1e-6, false, dtype, device),
      proj_out(config.num_attention_heads * config.attention_head_dim,
               config.out_channels * std::get<0>(config.patch_size) *
               std::get<1>(config.patch_size) * std::get<2>(config.patch_size),
               true, config.use_fp4, dtype, device)
{
    const int inner_dim = config.num_attention_heads * config.attention_head_dim;

    for (int i = 0; i < config.num_layers; i++) {
        blocks.push_back(std::make_unique<WanTransformerBlock>(
            inner_dim,
            config.ffn_dim,
            config.num_attention_heads,
            config.use_fp4,
            dtype,
            device
        ));
        registerChildren(*blocks.back(), format("blocks.{}", i));
    }

    scale_shift_table = Tensor::allocate({1, 2, inner_dim}, dtype, device);

    registerChildren(norm_out, "norm_out");
    registerChildren(proj_out, "proj_out");

    registerParams(scale_shift_table, "scale_shift_table");
}


Tensor WanModel::forward(Tensor hidden_states, Tensor timestep, Tensor encoder_hidden_states, Tensor encoder_hidden_states_image, Tensor attention_kwargs) {
    for (int i = 0; i < config.num_layers; i++) {
        auto &&block = blocks[i];
        hidden_states = block->forward(
            hidden_states, timestep, encoder_hidden_states, encoder_hidden_states_image, attention_kwargs
        );
    }
    return hidden_states;
}
