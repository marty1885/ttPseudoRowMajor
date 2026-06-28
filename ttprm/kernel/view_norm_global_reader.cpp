// Reader endpoint for global folded norms.
// x and optional gamma/beta use the same folded view descriptor, differing only
// by tensor accessor. The whole folded block is one global reduction unit.
//
// Compile args: x TensorAccessorArgs, then gamma, then beta.   Define: TTPRM_X_VIEW (AFFINE).
// Runtime args: 0 x_addr 1 gamma_addr 2 beta_addr 3 ntiles 4 affine_kind  5.. folded view block.

#include <cstdint>

#include "tensor_view.hpp"

#ifndef TTPRM_X_VIEW
#define TTPRM_X_VIEW ttprm::AffineTensorView
#endif

namespace {
constexpr uint32_t cb_x      = 0;
constexpr uint32_t cb_gamma  = 2;
constexpr uint32_t cb_beta   = 3;
constexpr uint32_t cb_zero   = 31;
constexpr uint32_t view_base = 5;  // folded view args begin after the 5 scalar args
constexpr uint32_t AFFINE_NONE = 0, AFFINE_SCALE = 1, AFFINE_SCALE_BIAS = 2;
}  // namespace

void kernel_main() {
    const uint32_t x_addr      = get_arg_val<uint32_t>(0);
    const uint32_t gamma_addr  = get_arg_val<uint32_t>(1);
    const uint32_t beta_addr   = get_arg_val<uint32_t>(2);
    const uint32_t ntiles      = get_arg_val<uint32_t>(3);
    const uint32_t affine_kind = get_arg_val<uint32_t>(4);
    const bool has_scale = affine_kind != AFFINE_NONE;
    const bool has_bias = affine_kind == AFFINE_SCALE_BIAS;

    using VArgs = ttprm::TensorViewArgs<TTPRM_X_VIEW, view_base>;
    uint32_t zero_rp = 0;
    if constexpr (VArgs::gather_needs_pad_seed) {
        cb_wait_front(cb_zero, 1);
        zero_rp = get_read_ptr(cb_zero);
    }

    constexpr uint32_t NX = TensorAccessorArgs<0>::num_compile_time_args();
    constexpr uint32_t NG = TensorAccessorArgs<NX>::num_compile_time_args();
    const auto x_acc = TensorAccessor(TensorAccessorArgs<0>(),       x_addr,     get_tile_size(cb_x));
    const auto g_acc = TensorAccessor(TensorAccessorArgs<NX>(),      gamma_addr, get_tile_size(cb_gamma));
    const auto b_acc = TensorAccessor(TensorAccessorArgs<NX + NG>(), beta_addr,  get_tile_size(cb_beta));

    auto gather = [&](auto view, uint32_t cb) {
        const uint32_t tb = get_tile_size(cb);
        cb_reserve_back(cb, ntiles);
        const uint32_t wp = get_write_ptr(cb);
        for (uint32_t i = 0; i < ntiles; i++) { view.gather_tile(wp + i * tb); view.advance(); }
        noc_async_read_barrier();
        cb_push_back(cb, ntiles);
    };

    gather(VArgs::make(x_acc, zero_rp), cb_x);
    if (has_scale) {
        gather(VArgs::make(g_acc, zero_rp), cb_gamma);
    }
    if (has_bias) {
        gather(VArgs::make(b_acc, zero_rp), cb_beta);
    }
}
