// ───────────────────────────────────────────────────────
//  WanModel.cpp      (put under  src/  )
// ───────────────────────────────────────────────────────
#include "WanModel.h"

#include "activation.h"
#include "flash_api.h"
#include "kernels/misc_kernels.h"
#include "kernels/gemm_batched.h"
#include "kernels/zgemm/zgemm.h"

#include <nvtx3/nvToolsExt.h>

using namespace nunchaku;
using spdlog::fmt_lib::format;

// helper – identical to Flux/Sana
static inline Tensor
forward_mlp(GEMM_W4A4 &fc1, GEMM_W4A4 &fc2, Tensor x)
{
    Tensor h = fc2.forward_quant(
        std::get<GEMM_W4A4::QuantizedActivation>(
            fc1.forward(x, GEMM_W4A4::FuseOptions::GELU_QUANT, &fc2)
        ));
    return h;
}

// ───────────────────────────────────────────────────────
//  WanRotaryPosEmbed
//    very small – just holds a table; we don't
//    actually *apply* RoPE here (that happens in qkv kernels)
// ───────────────────────────────────────────────────────
WanRotaryPosEmbed::WanRotaryPosEmbed(
        int head_dim_,
        std::array<int,3> patch_,
        int max_seq_len_,
        float theta_,
        Tensor::ScalarType dtype,
        Device device)
    : head_dim(head_dim_),
      patch(patch_),
      max_seq_len(max_seq_len_),
      theta(theta_)
{
    freqs = Tensor::allocate({max_seq_len_, head_dim_}, dtype, device);
}

Tensor WanRotaryPosEmbed::forward(Tensor hidden_states)
{
    // ensure on same device/dtype
    if (freqs.device().type != hidden_states.device().type)
        return freqs.copy(hidden_states.device());
    if (freqs.scalar_type() != hidden_states.scalar_type())
        return freqs.cast(hidden_states.scalar_type());
    return freqs;
}

// ───────────────────────────────────────────────────────
//  Conv3dPatchEmbed
//    ***identity***  – replace with real conv later
// ───────────────────────────────────────────────────────
Conv3dPatchEmbed::Conv3dPatchEmbed(
        int in_ch, int out_ch,
        std::array<int,3> kernel_,
        Tensor::ScalarType dtype, Device device)
    : kernel(kernel_)
{
    weight = Tensor::allocate({out_ch, in_ch}, dtype, device);
    bias   = Tensor::allocate({out_ch},        dtype, device);
    registerParams(weight,"weight")(bias,"bias");
}

Tensor Conv3dPatchEmbed::forward(Tensor x)
{
    // pass-through – shapes stay the same
    return x;
}

// ───────────────────────────────────────────────────────
//  WanTimeTextImageEmbedding   (same pattern as Sana/Flux)
// ───────────────────────────────────────────────────────
WanTimeTextImageEmbedding::
WanTimeTextImageEmbedding(int inner_dim,
                          int time_freq_dim,
                          int text_dim,
                          int image_dim,
                          int time_proj_dim,
                          int /*pos_embed_seq_len*/,
                          bool use_fp4,
                          Tensor::ScalarType dtype,
                          Device device)
    : time_fc1(time_freq_dim, inner_dim, true, use_fp4, dtype, device),
      time_fc2(inner_dim,     inner_dim, true, use_fp4, dtype, device),
      time_proj(inner_dim,    time_proj_dim, true, use_fp4, dtype, device),
      text_fc1(text_dim,      inner_dim, true, use_fp4, dtype, device),
      text_fc2(inner_dim,     inner_dim, true, use_fp4, dtype, device)
{
    registerChildren(time_fc1,"time_fc1")(time_fc2,"time_fc2")
                    (time_proj,"time_proj")
                    (text_fc1,"text_fc1")(text_fc2,"text_fc2");

    if (image_dim) {
        image_embedder.emplace(
            std::make_unique<GEMM_W4A4>(image_dim, inner_dim,
                                        true, use_fp4, dtype, device));
        registerChildren(*image_embedder->get(),"image_embedder");
    }
}

std::tuple<Tensor,Tensor,Tensor,Tensor>
WanTimeTextImageEmbedding::forward(
        Tensor timestep,
        Tensor txt_tokens,
        Tensor img_tokens)
{
    Tensor temb = time_fc2.forward(
                        Silu::forward(time_fc1.forward(timestep)));

    Tensor t_proj = time_proj.forward(Silu::forward(temb));

    Tensor txt_emb = text_fc2.forward(
                         GELU::forward(text_fc1.forward(txt_tokens)));

    Tensor img_emb;
    if (image_embedder && img_tokens.valid())
        img_emb = (*image_embedder)->forward(img_tokens);

    return {temb, t_proj, txt_emb, img_emb};
}

// ───────────────────────────────────────────────────────
//  WanTransformerBlock
// ───────────────────────────────────────────────────────
WanTransformerBlock::
WanTransformerBlock(int dim_,
                    int ffn_dim,
                    int num_heads_,
                    bool cross_attn_norm,
                    int added_kv_proj_dim,
                    bool use_fp4,
                    Tensor::ScalarType dtype,
                    Device device)
    : dim(dim_), num_heads(num_heads_), dim_head(dim_/num_heads_),
      norm1(dim,1e-6,false,dtype,device),
      norm2(dim,1e-6,cross_attn_norm,dtype,device),
      norm3(dim,1e-6,false,dtype,device),
      self_qkv_proj(dim, dim*3, true, use_fp4, dtype, device),
      self_out_proj(dim, dim, true, use_fp4, dtype, device),
      cross_qkv_proj(dim, dim*3, true, use_fp4, dtype, device),
      cross_out_proj(dim, dim, true, use_fp4, dtype, device),
      ffn_fc1(dim, ffn_dim, true, use_fp4, dtype, device),
      ffn_fc2(ffn_dim, dim, true, use_fp4, dtype, device)
{
    scale_shift_table = Tensor::allocate({6, dim}, dtype, device);
    registerParams(scale_shift_table,"scale_shift_table");

    registerChildren(norm1,"norm1")(norm2,"norm2")(norm3,"norm3")
                    (self_qkv_proj,"self_qkv_proj")
                    (self_out_proj,"self_out_proj")
                    (cross_qkv_proj,"cross_qkv_proj")
                    (cross_out_proj,"cross_out_proj")
                    (ffn_fc1,"ffn_fc1")(ffn_fc2,"ffn_fc2");
}

Tensor WanTransformerBlock::forward(
        Tensor hidden_states,
        Tensor encoder_hidden_states,
        Tensor temb,
        Tensor rotary_emb)
{
    nvtxRangePushA("WanBlock");

    // ── build (shift,scale,gate) arrays just like Sana ──
    Tensor chunk = temb.view({temb.shape[0], 6, dim});
    kernels::mul_add_batch(chunk, {}, false, 0,
                           scale_shift_table, false);
    auto parts = kernels::split_mod<6>(chunk);
    auto &shift_msa = std::get<0>(parts);
    auto &scale_msa = std::get<1>(parts);
    auto &gate_msa  = std::get<2>(parts);
    auto &shift_mlp = std::get<3>(parts);
    auto &scale_mlp = std::get<4>(parts);
    auto &gate_mlp  = std::get<5>(parts);

    // ── self-attn ───────────────────────────────────────
    Tensor residual = hidden_states;
    Tensor norm_x   = norm1.forward(hidden_states);
    kernels::mul_add_batch(norm_x, scale_msa,true,1, shift_msa,true);

    Tensor qkv = self_qkv_proj.forward(norm_x);          // [B,T,3*dim]
    // slice → q,k,v
    qkv = qkv.view({qkv.shape[0], qkv.shape[1], 3, num_heads, dim_head});
    Tensor q = qkv.slice(2,0,1).reshape({-1,num_heads,dim_head});
    Tensor k = qkv.slice(2,1,2).reshape({-1,num_heads,dim_head});
    Tensor v = qkv.slice(2,2,3).reshape({-1,num_heads,dim_head});

    Tensor attn = mha_fwd(q,k,v,
                          0.f, pow(dim_head,-0.5f),
                          false,-1,-1,false).front()
                    .view({norm_x.shape[0],
                           norm_x.shape[1],
                           num_heads*dim_head});
    attn = self_out_proj.forward(attn);
    kernels::mul_add_batch(attn, gate_msa,true,0,
                           residual,true);
    hidden_states = attn;

    // ── cross-attn (img ⇄ text) ────────────────────────
    Tensor norm_c = norm2.forward(hidden_states);
    Tensor q_c = cross_qkv_proj.forward(norm_c);
    Tensor kv  = cross_qkv_proj.forward(encoder_hidden_states);

    q_c = q_c.view({q_c.shape[0],q_c.shape[1],3,num_heads,dim_head});
    kv  = kv .view({kv .shape[0],kv .shape[1],3,num_heads,dim_head});

    Tensor q2 = q_c.slice(2,0,1)
                     .reshape({-1,num_heads,dim_head});
    Tensor k2 = kv .slice(2,1,2)
                     .reshape({-1,num_heads,dim_head});
    Tensor v2 = kv .slice(2,2,3)
                     .reshape({-1,num_heads,dim_head});

    Tensor attn_c = mha_fwd(q2,k2,v2,
                            0.f, pow(dim_head,-0.5f),
                            false,-1,-1,false).front()
                       .view({norm_c.shape[0],
                              norm_c.shape[1],
                              num_heads*dim_head});
    attn_c = cross_out_proj.forward(attn_c);
    hidden_states = kernels::add(hidden_states, attn_c);

    // ── FFN ────────────────────────────────────────────
    Tensor norm_f = norm3.forward(hidden_states);
    kernels::mul_add_batch(norm_f, scale_mlp,true,1,
                           shift_mlp,true);

    Tensor ff = forward_mlp(ffn_fc1, ffn_fc2, norm_f);
    kernels::mul_add_batch(ff, gate_mlp,true,0,
                           hidden_states,true);

    nvtxRangePop();
    return ff;
}

// ───────────────────────────────────────────────────────
//  WanModel
// ───────────────────────────────────────────────────────
WanModel::WanModel(const WanConfig &cfg,
                   Tensor::ScalarType dtype,
                   Device device)
    : config(cfg),
      rope(std::make_unique<WanRotaryPosEmbed>(
              cfg.head_dim, cfg.patch_size,
              cfg.rope_max_seq_len, 10000.f,
              dtype, device)),
      patch_embedding(std::make_unique<Conv3dPatchEmbed>(
              cfg.in_channels,
              cfg.num_heads*cfg.head_dim,
              cfg.patch_size, dtype, device)),
      cond_embedder(std::make_unique<WanTimeTextImageEmbedding>(
              cfg.num_heads*cfg.head_dim,
              cfg.freq_dim,
              cfg.text_dim,
              cfg.image_dim,
              cfg.num_heads*cfg.head_dim*6,
              cfg.rope_max_seq_len,
              cfg.use_fp4, dtype, device)),
      norm_out(cfg.num_heads*cfg.head_dim, cfg.eps,false,dtype,device),
      proj_out(cfg.num_heads*cfg.head_dim,
               cfg.out_channels, true,
               cfg.use_fp4, dtype, device),
      use_fp4(cfg.use_fp4)
{
    scale_shift_table = Tensor::allocate({2, cfg.num_heads*cfg.head_dim},
                                         dtype, device);
    registerParams(scale_shift_table,"scale_shift_table");

    for (int i=0;i<cfg.num_layers;++i) {
        blocks.push_back(std::make_unique<WanTransformerBlock>(
            cfg.num_heads*cfg.head_dim, cfg.ffn_dim,
            cfg.num_heads, cfg.cross_attn_norm,
            cfg.added_kv_proj_dim,
            cfg.use_fp4, dtype, device));
        registerChildren(*blocks.back(), format("blocks.{}",i));
    }
}

Tensor WanModel::forward(
        Tensor hidden_states,     // input video or already-embedded
        Tensor timestep,
        Tensor txt_tokens,
        Tensor img_tokens,
        bool   skip_first_layer)
{
    nvtxRangePushA("WanModel");

    // 1. patch embed (identity at the moment)
    hidden_states = patch_embedding->forward(hidden_states);

    // reshape to tokens  B,N,C  (no permute/flatten)
    int B = hidden_states.shape[0];
    int C = hidden_states.shape[1];
    int D = hidden_states.shape.size()>2 ? hidden_states.shape[2] : 1;
    int N = hidden_states.numel() / (B*C);
    hidden_states = hidden_states.view({B,C,N})
                                .transpose(1,2);          // B,N,C

    // 2. condition embeddings
    Tensor temb, tproj, txt_emb, img_emb;
    std::tie(temb,tproj,txt_emb,img_emb) =
        cond_embedder->forward(timestep, txt_tokens, img_tokens);

    if (img_emb.valid()) { // prepend image extras to text
        Tensor concat = Tensor::allocate(
            {txt_emb.shape[0]+img_emb.shape[0],
             txt_emb.shape[1]}, txt_emb.scalar_type(),
            txt_emb.device());
        concat.slice(0,0,img_emb.shape[0]).copy_(img_emb);
        concat.slice(0,img_emb.shape[0],img_emb.shape[0]+txt_emb.shape[0])
              .copy_(txt_emb);
        txt_emb = concat;
    }

    // 3. rotary table (not applied here, just handed down)
    Tensor rotary = rope->forward(hidden_states);

    // 4. transformer stack
    for (int i=(skip_first_layer?1:0); i<config.num_layers; ++i)
        hidden_states = blocks[i]->forward(
                            hidden_states, txt_emb,
                            tproj, rotary);

    // 5. final norm + projection
    Tensor tmp = temb.view({temb.shape[0],2,hidden_states.shape[2]});
    kernels::mul_add_batch(tmp, {}, false, 0,
                           scale_shift_table,false);
    auto s2 = kernels::split_mod<2>(tmp);
    Tensor shift = std::get<0>(s2);
    Tensor scale = std::get<1>(s2);

    Tensor norm_x = norm_out.forward(hidden_states);
    kernels::mul_add_batch(norm_x, scale,true,1,
                           shift,true);

    Tensor out = proj_out.forward(norm_x);      // B,N,Cout
    nvtxRangePop();
    return out;
}

void WanModel::setAttentionImpl(WanAttentionImpl impl)
{
    attnImpl = impl;
    for (auto &b:blocks) b->attnImpl = impl;
}
