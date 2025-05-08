#pragma once

#include "interop/torch.h"
#include "WanModel.h"
#include "Serialization.h"
#include "debug.h"
#include "module.h"

class QuantizedWanModel : public ModuleWrapper<WanModel> {
public:
    void init(pybind11::dict config, bool use_fp4, bool bf16, int8_t deviceId) {
        spdlog::info("Initializing QuantizedWanModel on device {}", deviceId);

        WanConfig cfg{
            .num_layers = config["num_layers"].cast<int>(),
            .num_attention_heads = config["num_attention_heads"].cast<int>(),
            .attention_head_dim = config["attention_head_dim"].cast<int>(),
            //.num_cross_attention_heads = config["num_cross_attention_heads"].cast<int>(),
            .ffn_dim = config["ffn_dim"].cast<int>(),
            .in_channels = config["in_channels"].cast<int>(),
            .out_channels = config["out_channels"].cast<int>(),
            .text_dim = config["text_dim"].cast<int>(),
            .freq_dim = config["freq_dim"].cast<int>(),
            .patch_size = config["patch_size"].cast<std::tuple<int, int, int>>(),
            .cross_attn_norm = config["cross_attn_norm"].cast<bool>(),
            .use_fp4 = use_fp4
        };

        ModuleWrapper::init(deviceId);

        CUDADeviceContext ctx(this->deviceId);
        net = std::make_unique<WanModel>(cfg, bf16 ? Tensor::BF16 : Tensor::FP16, Device::cuda(deviceId));
    }
    torch::Tensor forward(
        torch::Tensor hidden_states,
        torch::Tensor timestep,
        torch::Tensor encoder_hidden_states,
        torch::Tensor encoder_hidden_states_image = torch::Tensor(),
        torch::Tensor attention_kwargs = torch::Tensor())
    {
        checkModel();
        CUDADeviceContext ctx(deviceId);
    
        spdlog::debug("QuantizedWanModel forward");
    
        // Ensure inputs are contiguous for efficient memory access
        hidden_states = hidden_states.contiguous();
        timestep = timestep.contiguous();
        encoder_hidden_states = encoder_hidden_states.contiguous();
        if (encoder_hidden_states_image.defined()) {
            encoder_hidden_states_image = encoder_hidden_states_image.contiguous();
        }
        if (attention_kwargs.defined()) {
            attention_kwargs = attention_kwargs.contiguous();
        }
    
        Tensor result = net->forward(
            from_torch(hidden_states),
            from_torch(timestep),
            from_torch(encoder_hidden_states),
            encoder_hidden_states_image.defined() ? from_torch(encoder_hidden_states_image) : Tensor{},
            attention_kwargs.defined() ? from_torch(attention_kwargs) : Tensor{}
        );
    
        return to_torch(result);
    }
    
};