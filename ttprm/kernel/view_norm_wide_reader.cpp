// Reader endpoint for wide per-row norms.
// cb_x holds one full row-block resident so compute can reduce and normalize
// without re-reading x. Gamma/beta stream one tile at a time during the
// normalize phase instead of remaining resident.
//
// Compile args: x, gamma, beta TensorAccessorArgs.   Define: TTPRM_X_VIEW.
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
    const uint32_t rscale_bits = get_arg_val<uint32_t>(5);
    const uint32_t eps_bits    = get_arg_val<uint32_t>(6);
    const uint32_t affine_kind = get_arg_val<uint32_t>(7);
    const bool has_scale = affine_kind != AFFINE_NONE;
    const bool has_bias = affine_kind == AFFINE_SCALE_BIAS;

    const uint16_t RSCALE = (uint16_t)(rscale_bits >> 16);
    const uint16_t EPS    = (uint16_t)(eps_bits >> 16);

    cb_reserve_back(cb_scaler, 1);
    {
        volatile tt_l1_ptr uint16_t* p = (volatile tt_l1_ptr uint16_t*)get_write_ptr(cb_scaler);
        for (uint32_t i = 0; i < 1024; i++) p[i] = 0;
        for (uint32_t k = 0; k < 4; k++)
            for (uint32_t j = 0; j < 16; j++) p[(k << 8) + j] = RSCALE;
    }
    cb_push_back(cb_scaler, 1);

    cb_reserve_back(cb_eps, 1);
    {
        volatile tt_l1_ptr uint16_t* p = (volatile tt_l1_ptr uint16_t*)get_write_ptr(cb_eps);
        for (uint32_t r = 0; r < 32; r++)
            for (uint32_t c = 0; c < 32; c++) p[tile_off(r, c)] = EPS;
    }
    cb_push_back(cb_eps, 1);

    using XArgs = ttprm::TensorViewArgs<TTPRM_X_VIEW, view_base>;
    uint32_t zero_rp = 0;
    if constexpr (XArgs::gather_needs_pad_seed) {
        cb_wait_front(cb_zero, 1);
        zero_rp = get_read_ptr(cb_zero);
    }
    const auto x_acc = TensorAccessor(TensorAccessorArgs<0>(), x_addr, get_tile_size(cb_x));
    auto x = XArgs::make(x_acc, zero_rp);
    const uint32_t x_tb = get_tile_size(cb_x);

    constexpr uint32_t NX = TensorAccessorArgs<0>::num_compile_time_args();
    constexpr uint32_t NG = TensorAccessorArgs<NX>::num_compile_time_args();
    const auto g_acc = TensorAccessor(TensorAccessorArgs<NX>(),      gamma_addr, get_tile_size(cb_gamma));
    const auto b_acc = TensorAccessor(TensorAccessorArgs<NX + NG>(), beta_addr,  get_tile_size(cb_beta));
    const uint32_t g_tb = get_tile_size(cb_gamma);
    const uint32_t b_tb = get_tile_size(cb_beta);

    for (uint32_t rb = 0; rb < n_rb; rb++) {
        // Gather the resident row-block.
        cb_reserve_back(cb_x, ctiles);
        const uint32_t wp = get_write_ptr(cb_x);
        for (uint32_t i = 0; i < ctiles; i++) { x.gather_tile(wp + i * x_tb); x.advance(); }
        noc_async_read_barrier();
        cb_push_back(cb_x, ctiles);

        // Stream normalize-phase gamma/beta one tile at a time.
        if (has_scale) {
            for (uint32_t i = 0; i < ctiles; i++) {
                cb_reserve_back(cb_gamma, 1);
                noc_async_read(g_acc.get_noc_addr(i, 0), get_write_ptr(cb_gamma), g_tb);
                if (has_bias) {
                    cb_reserve_back(cb_beta, 1);
                    noc_async_read(b_acc.get_noc_addr(i, 0), get_write_ptr(cb_beta), b_tb);
                }
                noc_async_read_barrier();
                cb_push_back(cb_gamma, 1);
                if (has_bias) cb_push_back(cb_beta, 1);
            }
        }
    }
}
