#pragma once
/*  Thin C++/PyBind wrapper around WanModel
 *  – handles config extraction from Python,
 *  – owns the underlying model on one CUDA device,
 *  – converts tensors torch ⇆ Tensor,
 *  – exposes forward / load / debug helpers supplied by ModuleWrapper.
 */
#include "interop/torch.h"
#include "WanModel.h"
#include "Serialization.h"
#include "debug.h"
#include "module.h"

#include <vector>
#include <string>
#include <pybind11/stl.h>   //  torch->std::vector casts

class QuantizedWanModel : public ModuleWrapper<WanModel>
{
public:
    /* ------------------------------------------------------------------ init */
    void init(pybind11::dict cfg, bool bf16, int8_t deviceId)
    {
        spdlog::info("Initializing QuantizedWanModel on device {}", deviceId);

        /* ----- fill WanConfig ------------------------------------------------ */
        WanConfig c;   // start with struct defaults declared in WanModel.h

        // patch_size : list[3]
        if (cfg.contains("patch_size"))
        {
            auto patch = cfg["patch_size"].cast<std::vector<int>>();
            if (patch.size() != 3)
                throw std::invalid_argument("patch_size must have 3 ints");
            c.patch_size = {patch[0], patch[1], patch[2]};
        }

        // standard integer & bool fields (present or left at defaults)
        if (cfg.contains("num_layers"))              c.num_layers           = cfg["num_layers"].cast<int>();
        if (cfg.contains("num_attention_heads"))     c.num_heads            = cfg["num_attention_heads"].cast<int>();
        if (cfg.contains("num_heads"))               c.num_heads            = cfg["num_heads"].cast<int>();     // alias
        if (cfg.contains("attention_head_dim"))      c.head_dim             = cfg["attention_head_dim"].cast<int>();
        if (cfg.contains("head_dim"))                c.head_dim             = cfg["head_dim"].cast<int>();      // alias
        if (cfg.contains("ffn_dim"))                 c.ffn_dim              = cfg["ffn_dim"].cast<int>();
        if (cfg.contains("in_channels"))             c.in_channels          = cfg["in_channels"].cast<int>();
        if (cfg.contains("out_channels"))            c.out_channels         = cfg["out_channels"].cast<int>();
        if (cfg.contains("text_dim"))                c.text_dim             = cfg["text_dim"].cast<int>();
        if (cfg.contains("freq_dim"))                c.freq_dim             = cfg["freq_dim"].cast<int>();
        if (cfg.contains("rope_max_seq_len"))        c.rope_max_seq_len     = cfg["rope_max_seq_len"].cast<int>();
        if (cfg.contains("image_dim"))               c.image_dim            = cfg["image_dim"].cast<int>();
        if (cfg.contains("added_kv_proj_dim"))       c.added_kv_proj_dim    = cfg["added_kv_proj_dim"].cast<int>();
        if (cfg.contains("cross_attn_norm"))         c.cross_attn_norm      = cfg["cross_attn_norm"].cast<bool>();
        if (cfg.contains("eps"))                     c.eps                  = cfg["eps"].cast<float>();
        if (cfg.contains("use_fp4"))                 c.use_fp4              = cfg["use_fp4"].cast<bool>();

        // qk_norm can arrive as bool *or* string ― coerce safely to bool
        if (cfg.contains("qk_norm"))
        {
            try {
                c.qk_norm = cfg["qk_norm"].cast<bool>();
            } catch (const pybind11::cast_error &) {
                std::string v = cfg["qk_norm"].cast<std::string>();
                c.qk_norm = (v == "1" || v == "true" || v == "True");
            }
        }
        /* -------------------------------------------------------------------- */

        ModuleWrapper::init(deviceId);               // allocates debug helpers
        CUDADeviceContext ctx(this->deviceId);       // sets current CUDA dev
        net = std::make_unique<WanModel>(
                c,
                bf16 ? Tensor::BF16 : Tensor::FP16,
                Device::cuda(static_cast<int>(deviceId)));
    }

    /* --------------------------------------------------------------- forward */
    torch::Tensor forward(torch::Tensor               hidden_states,
                          torch::Tensor               timestep,
                          torch::Tensor               encoder_hidden_states,          // text
                          std::optional<torch::Tensor> encoder_hidden_states_image = std::nullopt)
    {
        checkModel();
        CUDADeviceContext ctx(deviceId);

        hidden_states           = hidden_states.contiguous();
        timestep                = timestep.contiguous();
        encoder_hidden_states   = encoder_hidden_states.contiguous();
        if (encoder_hidden_states_image)
            encoder_hidden_states_image = encoder_hidden_states_image->contiguous();

        Tensor out = net->forward(
            from_torch(hidden_states),
            from_torch(timestep),
            from_torch(encoder_hidden_states),
            encoder_hidden_states_image ? from_torch(*encoder_hidden_states_image) : Tensor{},
            /*skip_first_layer=*/false
        );

        torch::Tensor result = to_torch(out);
        Tensor::synchronizeDevice();
        return result;
    }

    /* ---------------------------------------------- optional attention switch */
    void setAttentionImpl(const std::string &name = "flashattn2")
    {
        std::string n = name.empty() ? "flashattn2" : name;
        spdlog::info("Set WAN attention implementation to {}", n);

        if (n == "flashattn2" || n == "default")
            net->setAttentionImpl(WanAttentionImpl::FlashAttention2);
        else if (n == "nunchaku-fp16")
            net->setAttentionImpl(WanAttentionImpl::NunchakuFP16);
        else
            throw std::invalid_argument("Invalid attention implementation " + n);
    }
};
