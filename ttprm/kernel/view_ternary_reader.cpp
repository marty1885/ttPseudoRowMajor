// Reader endpoint for fused ternary elementwise ops.
// Each input has its own TensorViewArgs block, chained as A, B, then C. The
// compute consumes cb_a/cb_b/cb_c and the writer scatters through the output
// view; no intermediate relayout is materialized in DRAM.
//
// Broadcast operands gather one `tpg`-tile group into a resident CB. Streaming
// operands gather one tile per output tile.
//
// Compile args: a, then b, then c TensorAccessorArgs.  Defines: TTPRM_A_VIEW/B_VIEW/C_VIEW.
// Runtime args: 0 a_addr 1 b_addr 2 c_addr 3 n_tiles 4 tpg 5 a_bcast 6 b_bcast 7 c_bcast
//               8.. view A  N.. view B  M.. view C  (N=AArgs::next(), M=BArgs::next()).

#include <cstdint>

#include "tensor_view.hpp"

#ifndef TTPRM_A_VIEW
#define TTPRM_A_VIEW ttprm::DenseTensorView
#endif
#ifndef TTPRM_B_VIEW
#define TTPRM_B_VIEW ttprm::DenseTensorView
#endif
#ifndef TTPRM_C_VIEW
#define TTPRM_C_VIEW ttprm::DenseTensorView
#endif

namespace {
constexpr uint32_t cb_a      = 0;
constexpr uint32_t cb_b      = 1;
constexpr uint32_t cb_c      = 2;
constexpr uint32_t cb_zero   = 31;
constexpr uint32_t view_base = 8;
}  // namespace

void kernel_main() {
    const uint32_t a_addr  = get_arg_val<uint32_t>(0);
    const uint32_t b_addr  = get_arg_val<uint32_t>(1);
    const uint32_t c_addr  = get_arg_val<uint32_t>(2);
    const uint32_t n_tiles = get_arg_val<uint32_t>(3);
    const uint32_t tpg     = get_arg_val<uint32_t>(4);
    const uint32_t a_bcast = get_arg_val<uint32_t>(5);
    const uint32_t b_bcast = get_arg_val<uint32_t>(6);
    const uint32_t c_bcast = get_arg_val<uint32_t>(7);

    // TensorAccessorArgs are packed A, then B, then C.
    constexpr uint32_t NA = TensorAccessorArgs<0>::num_compile_time_args();
    constexpr uint32_t NB = TensorAccessorArgs<NA>::num_compile_time_args();
    const auto a_acc = TensorAccessor(TensorAccessorArgs<0>(),       a_addr, get_tile_size(cb_a));
    const auto b_acc = TensorAccessor(TensorAccessorArgs<NA>(),      b_addr, get_tile_size(cb_b));
    const auto c_acc = TensorAccessor(TensorAccessorArgs<NA + NB>(), c_addr, get_tile_size(cb_c));

    // Runtime view blocks are packed A, then B, then C.
    using AArgs = ttprm::TensorViewArgs<TTPRM_A_VIEW, view_base>;
    using BArgs = ttprm::TensorViewArgs<TTPRM_B_VIEW, AArgs::next()>;
    using CArgs = ttprm::TensorViewArgs<TTPRM_C_VIEW, BArgs::next()>;

    // Affine gathers share the resident pad seed.
    uint32_t zero_rp = 0;
    if constexpr (AArgs::gather_needs_pad_seed || BArgs::gather_needs_pad_seed ||
                  CArgs::gather_needs_pad_seed) {
        cb_wait_front(cb_zero, 1);
        zero_rp = get_read_ptr(cb_zero);
    }
    auto a = AArgs::make(a_acc, zero_rp);
    auto b = BArgs::make(b_acc, zero_rp);
    auto c = CArgs::make(c_acc, zero_rp);
    const uint32_t a_tb = get_tile_size(cb_a);
    const uint32_t b_tb = get_tile_size(cb_b);
    const uint32_t c_tb = get_tile_size(cb_c);

    // Issue resident broadcast-group reads here; the first loop barrier publishes
    // them together with tile 0 streamed reads.
    if (a_bcast) {
        a.reset(0);
        cb_reserve_back(cb_a, tpg);
        const uint32_t wp = get_write_ptr(cb_a);
        for (uint32_t p = 0; p < tpg; p++) { a.gather_tile(wp + p * a_tb); a.advance(); }
    }
    if (b_bcast) {
        b.reset(0);
        cb_reserve_back(cb_b, tpg);
        const uint32_t wp = get_write_ptr(cb_b);
        for (uint32_t p = 0; p < tpg; p++) { b.gather_tile(wp + p * b_tb); b.advance(); }
    }
    if (c_bcast) {
        c.reset(0);
        cb_reserve_back(cb_c, tpg);
        const uint32_t wp = get_write_ptr(cb_c);
        for (uint32_t p = 0; p < tpg; p++) { c.gather_tile(wp + p * c_tb); c.advance(); }
    }
    bool resident_pending = (a_bcast || b_bcast || c_bcast);

    // Stream only non-broadcast operands; broadcast groups stay resident.
    for (uint32_t t = 0; t < n_tiles; t++) {
        if (!a_bcast) { cb_reserve_back(cb_a, 1); a.gather_tile(get_write_ptr(cb_a)); }
        if (!b_bcast) { cb_reserve_back(cb_b, 1); b.gather_tile(get_write_ptr(cb_b)); }
        if (!c_bcast) { cb_reserve_back(cb_c, 1); c.gather_tile(get_write_ptr(cb_c)); }
        // The first barrier also covers pending resident broadcast reads.
        const bool has_streamed = !a_bcast || !b_bcast || !c_bcast;
        if (has_streamed || resident_pending) noc_async_read_barrier();
        if (resident_pending) {
            if (a_bcast) cb_push_back(cb_a, tpg);
            if (b_bcast) cb_push_back(cb_b, tpg);
            if (c_bcast) cb_push_back(cb_c, tpg);
            resident_pending = false;
        }
        if (!a_bcast) { cb_push_back(cb_a, 1); a.advance(); }
        if (!b_bcast) { cb_push_back(cb_b, 1); b.advance(); }
        if (!c_bcast) { cb_push_back(cb_c, 1); c.advance(); }
    }
}
