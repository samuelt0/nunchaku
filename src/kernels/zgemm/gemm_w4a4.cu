#include "zgemm.h"
#include "gemm_w4a4_launch.cuh"

namespace nunchaku::kernels {

// Global stub flag for profiling on unsupported hardware (e.g., B200 for FP4)
static bool g_stub_fp4_gemm = false;

void set_stub_fp4_gemm(bool enabled) {
    g_stub_fp4_gemm = enabled;
    spdlog::info("FP4 GEMM stub mode: {}", enabled ? "enabled" : "disabled");
}

bool get_stub_fp4_gemm() {
    return g_stub_fp4_gemm;
}

// for sm_75 only
struct FasterI2FMode {
    enum Mode {
        Disabled = 0,
        Enabled,
        Always,
    };
    inline static Mode mode = Disabled;
    static bool check(bool act_unsigned);
};

template<typename F>
static void invoke_launch(Tensor::ScalarType dtype, bool use_fp4, bool fasterI2F, F &&launch) {
    if (fasterI2F && dtype == Tensor::FP16) {
        launch.template operator()<GEMMConfig_W4A4_FP16_FasterI2F, false>();
    } else {
        dispatchBool(use_fp4, [&]<bool USE_FP4>() {
            if (dtype == Tensor::FP16) {
                launch.template operator()<GEMMConfig_W4A4_FP16, USE_FP4>();
            } else if (dtype == Tensor::BF16) {
                launch.template operator()<GEMMConfig_W4A4_BF16, USE_FP4>();
            } else {
                assert(false);
            }
        });
    }
}

void gemm_w4a4(Tensor act,            // packed act [M, K / 2]
               Tensor wgt,            // packed act [N, K / 2]
               Tensor out,            // linear     [M, N]
               Tensor qout,           // packed act [M, N / 2]
               Tensor ascales,        // packed as  [K / 64, M]
               Tensor wscales,        // packed ws  [K / 64, N]
               Tensor oscales,        // packed as  [N / 64, M]
               Tensor poolout,        // linear     [M / PoolSize, N]
               Tensor lora_act_in,    // packed lora_act [M, R]
               Tensor lora_up,        // packed lora_wgt [N, R]
               Tensor lora_down,      // packed lora_wgt [N, R]
               Tensor lora_act_out,   // packed lora_act [M, R]
               Tensor norm_q,         // linear     [HEAD_DIM]
               Tensor norm_k,         // linear     [HEAD_DIM]
               Tensor rotary_emb,     // linear     [M, HEAD_DIM / 2, 2, 2]
               Tensor bias,           // packed ws  [N]
               Tensor smooth_factor,  // packed ws  [N], for quantization of the next layer
               Tensor out_vk,         // linear     [B, num_heads, head_dim + 1, head_dim]
               Tensor out_linearattn, // linear     [B, (M), N / 3]
               bool act_unsigned,
               std::vector<float> lora_scales, // [R / 16]
               bool fuse_silu,
               bool fp4,
               float alpha,
               Tensor wcscales,
               Tensor out_q, // packed attention [B, H, M, D]
               Tensor out_k, // packed attention [B, H, M, D]
               Tensor out_v, // packed attention [B, H, M, D]
               int attn_tokens) {
    // Stub mode: skip FP4 GEMM, just zero the output tensors
    if (fp4 && g_stub_fp4_gemm) {
        if (out.valid()) {
            cudaMemsetAsync(out.data_ptr(), 0, out.numel() * out.scalar_size(), getCurrentCUDAStream());
        }
        if (qout.valid()) {
            cudaMemsetAsync(qout.data_ptr(), 0, qout.numel() * qout.scalar_size(), getCurrentCUDAStream());
        }
        if (oscales.valid()) {
            cudaMemsetAsync(oscales.data_ptr(), 0, oscales.numel() * oscales.scalar_size(), getCurrentCUDAStream());
        }
        if (poolout.valid()) {
            cudaMemsetAsync(poolout.data_ptr(), 0, poolout.numel() * poolout.scalar_size(), getCurrentCUDAStream());
        }
        if (out_vk.valid()) {
            cudaMemsetAsync(out_vk.data_ptr(), 0, out_vk.numel() * out_vk.scalar_size(), getCurrentCUDAStream());
        }
        if (out_linearattn.valid()) {
            cudaMemsetAsync(out_linearattn.data_ptr(), 0, out_linearattn.numel() * out_linearattn.scalar_size(), getCurrentCUDAStream());
        }
        if (out_q.valid()) {
            cudaMemsetAsync(out_q.data_ptr(), 0, out_q.numel() * out_q.scalar_size(), getCurrentCUDAStream());
        }
        if (out_k.valid()) {
            cudaMemsetAsync(out_k.data_ptr(), 0, out_k.numel() * out_k.scalar_size(), getCurrentCUDAStream());
        }
        if (out_v.valid()) {
            cudaMemsetAsync(out_v.data_ptr(), 0, out_v.numel() * out_v.scalar_size(), getCurrentCUDAStream());
        }
        if (lora_act_out.valid()) {
            cudaMemsetAsync(lora_act_out.data_ptr(), 0, lora_act_out.numel() * lora_act_out.scalar_size(), getCurrentCUDAStream());
        }
        return;
    }

    Tensor::ScalarType dtype = Tensor::INVALID_SCALAR_TYPE;
    if (!fp4) {
        dtype = ascales.dtype();
    } else {
        for (auto tensor : {out, bias, lora_up, lora_down, poolout, wcscales}) {
            if (tensor.valid()) {
                assert(dtype == Tensor::INVALID_SCALAR_TYPE || dtype == tensor.dtype());
                dtype = tensor.dtype();
            }
        }
    }
    invoke_launch(dtype, fp4, FasterI2FMode::check(act_unsigned), [&]<typename Config, bool USE_FP4>() {
        GEMM_W4A4_Launch<Config, USE_FP4>::gemm_w4a4(act,
                                                     wgt,
                                                     out,
                                                     qout,
                                                     ascales,
                                                     wscales,
                                                     oscales,
                                                     poolout,
                                                     lora_act_in,
                                                     lora_up,
                                                     lora_down,
                                                     lora_act_out,
                                                     norm_q,
                                                     norm_k,
                                                     rotary_emb,
                                                     bias,
                                                     smooth_factor,
                                                     out_vk,
                                                     out_linearattn,
                                                     act_unsigned,
                                                     lora_scales,
                                                     fuse_silu,
                                                     fp4,
                                                     alpha,
                                                     wcscales,
                                                     out_q,
                                                     out_k,
                                                     out_v,
                                                     attn_tokens);
    });
}

void linearattn_vk_mul_q(Tensor q, Tensor vk) {
    invoke_launch(q.dtype(), false, false, [&]<typename Config, bool USE_FP4>() {
        GEMM_W4A4_Launch<Config, false>::linearattn_vk_mul_q(q, vk);
    });
}

void quantize_w4a4_act_fuse_lora(Tensor input,
                                 Tensor output,
                                 Tensor oscales,
                                 Tensor lora_down,
                                 Tensor lora_act_out,
                                 Tensor smooth,
                                 bool fuse_glu,
                                 bool fp4) {
    // Stub mode: skip FP4 quantization, just zero the output tensors
    if (fp4 && g_stub_fp4_gemm) {
        if (output.valid()) {
            cudaMemsetAsync(output.data_ptr(), 0, output.numel() * output.scalar_size(), getCurrentCUDAStream());
        }
        if (oscales.valid()) {
            cudaMemsetAsync(oscales.data_ptr(), 0, oscales.numel() * oscales.scalar_size(), getCurrentCUDAStream());
        }
        if (lora_act_out.valid()) {
            cudaMemsetAsync(lora_act_out.data_ptr(), 0, lora_act_out.numel() * lora_act_out.scalar_size(), getCurrentCUDAStream());
        }
        return;
    }

    invoke_launch(input.dtype(), fp4, false, [&]<typename Config, bool USE_FP4>() {
        GEMM_W4A4_Launch<Config, USE_FP4>::quantize_w4a4_act_fuse_lora(
            input, output, oscales, lora_down, lora_act_out, smooth, fuse_glu, fp4);
    });
}

void quantize_w4a4_act(Tensor input, Tensor output, Tensor oscales) {
    invoke_launch(input.dtype(), false, false, [&]<typename Config, bool USE_FP4>() {
        GEMM_W4A4_Launch<Config, false>::quantize_w4a4_act(input, output, oscales);
    });
}
void quantize_w4a4_wgt(Tensor input, Tensor output, Tensor oscales) {
    invoke_launch(input.dtype(), false, false, [&]<typename Config, bool USE_FP4>() {
        GEMM_W4A4_Launch<Config, false>::quantize_w4a4_wgt(input, output, oscales);
    });
}

bool FasterI2FMode::check(bool act_unsigned) {
    auto *prop = getCurrentDeviceProperties();
    if (prop->major != 7 || prop->minor != 5) {
        return false;
    }

    if (mode == Always) {
        return true;
    } else if (mode == Enabled && !act_unsigned) {
        return true;
    } else {
        return false;
    }
}

void set_faster_i2f_mode(std::string mode) {
    static const std::map<std::string, FasterI2FMode::Mode> mapping = {
        {"disabled", FasterI2FMode::Disabled},
        {"enabled", FasterI2FMode::Enabled},
        {"always", FasterI2FMode::Always},
    };
    FasterI2FMode::mode = mapping.at(mode);
}

}; // namespace nunchaku::kernels
