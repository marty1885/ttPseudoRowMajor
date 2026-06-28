// Reader endpoint for per-head norms.
// A work unit is one tile-row of heads. Each row-block gathers `ctiles` x tiles
// into cb_x so the compute can reduce across the full head row. Optional
// gamma/beta vectors are gathered once and kept resident because they are reused
// for every row-block.
//
// Compile args: x TensorAccessorArgs, then gamma, then beta.   Define: TTPRM_X_VIEW.
// Runtime args: 0 x_addr 1 gamma_addr 2 beta_addr 3 n_rb 4 ctiles 5 rscale_bits 6 eps_bits
//               7 affine_kind  8.. x view block.

#include <cstdint>

#include "tensor_view.hpp"

#ifndef TTPRM_X_VIEW
#define TTPRM_X_VIEW ttprm::AffineTensorView
#endif

namespace {
constexpr uint32_t cb_x      = 0;
constexpr uint32_t cb_gamma  = 2;
constexpr uint32_t cb_beta   = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_eps    = 5;
constexpr uint32_t cb_zero   = 31;
constexpr uint32_t view_base = 8;
constexpr uint32_t AFFINE_NONE = 0, AFFINE_SCALE = 1, AFFINE_SCALE_BIAS = 2;
}  // namespace

static inline uint32_t tile_off(uint32_t r, uint32_t c) {
    uint32_t face = (r / 16) * 2 + (c / 16);
    return face * 256 + (r % 16) * 16 + (c % 16);
}

void kernel_main() {
    const uint32_t x_addr      = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr  = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr   = get_arg_val<uint32_t>(2);
    const uint32_t n_rb        = get_arg_val<uint32_t>(3);
    const uint32_t ctiles      = get_arg_val<uint32_t>(4);
    // Args 5 and 6 are consumed by compute, which generates scaler/eps tiles.
    const uint32_t affine_kind = get_arg_val<uint32_t>(7);
    const bool has_scale = affine_kind != AFFINE_NONE;
    const bool has_bias = affine_kind == AFFINE_SCALE_BIAS;

    using XArgs = ttprm::TensorViewArgs<TTPRM_X_VIEW, view_base>;
    uint32_t zero_rp = 0;
    if constexpr (XArgs::gather_needs_pad_seed) {
        cb_wait_front(cb_zero, 1);
        zero_rp = get_read_ptr(cb_zero);
    }

    const auto x_acc = TensorAccessor(TensorAccessorArgs<0>(), x_addr, get_tile_size(cb_x));
    auto x = XArgs::make(x_acc, zero_rp);
    const uint32_t x_tb = get_tile_size(cb_x);

    // gamma/beta are dense [1, Co] vectors and remain resident across row-blocks.
    if (has_scale) {
        constexpr uint32_t NX = TensorAccessorArgs<0>::num_compile_time_args();
        constexpr uint32_t NG = TensorAccessorArgs<NX>::num_compile_time_args();
        const auto g_acc = TensorAccessor(TensorAccessorArgs<NX>(),      gamma_addr, get_tile_size(cb_gamma));
        const auto b_acc = TensorAccessor(TensorAccessorArgs<NX + NG>(), beta_addr,  get_tile_size(cb_beta));
        cb_reserve_back(cb_gamma, ctiles);
        const uint32_t gwp = get_write_ptr(cb_gamma);
        const uint32_t g_tb = get_tile_size(cb_gamma);
        for (uint32_t p = 0; p < ctiles; p++) {
            noc_async_read(g_acc.get_noc_addr(p, 0), gwp + p * g_tb, g_tb);
        }
        if (has_bias) {
            cb_reserve_back(cb_beta, ctiles);
            const uint32_t bwp = get_write_ptr(cb_beta);
            const uint32_t b_tb = get_tile_size(cb_beta);
            for (uint32_t p = 0; p < ctiles; p++)
                noc_async_read(b_acc.get_noc_addr(p, 0), bwp + p * b_tb, b_tb);
        }
        noc_async_read_barrier();
        cb_push_back(cb_gamma, ctiles);
        if (has_bias) cb_push_back(cb_beta, ctiles);
    }

    // Stream one head row per row-block.
    for (uint32_t rb = 0; rb < n_rb; rb++) {
        cb_reserve_back(cb_x, ctiles);
        const uint32_t wp = get_write_ptr(cb_x);
        for (uint32_t i = 0; i < ctiles; i++) { x.gather_tile(wp + i * x_tb); x.advance(); }
        noc_async_read_barrier();
        cb_push_back(cb_x, ctiles);
    }
}
