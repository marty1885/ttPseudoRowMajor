// Reader endpoint for fused binary elementwise ops.
// Each input has its own TensorViewArgs block, so A and B can use different
// views while sharing the output tile grid. The compute consumes cb_a/cb_b and
// the writer scatters through the output view; no intermediate relayout is
// materialized in DRAM.
//
// Broadcast operands gather one `tpg`-tile group into a resident CB. Streaming
// operands gather one tile per output tile. The compute owns the group-cycle
// indexing for resident operands.
//
// Compile args: a TensorAccessorArgs, then b TensorAccessorArgs.  Defines: TTPRM_A_VIEW, TTPRM_B_VIEW.
// Runtime args: 0 a_addr 1 b_addr 2 n_tiles 3 tpg 4 a_bcast 5 b_bcast  6.. view A  N.. view B (N=AArgs::next()).

#include <cstdint>

#include "tensor_view.hpp"

#ifndef TTPRM_A_VIEW
#define TTPRM_A_VIEW ttprm::AffineTensorView
#endif
#ifndef TTPRM_B_VIEW
#define TTPRM_B_VIEW ttprm::DenseTensorView
#endif

namespace {
constexpr uint32_t cb_a      = 0;
constexpr uint32_t cb_b      = 1;
constexpr uint32_t cb_zero   = 31;
constexpr uint32_t view_base = 6;
}  // namespace

void kernel_main() {
    const uint32_t a_addr   = get_arg_val<uint32_t>(0);
    const uint32_t b_addr   = get_arg_val<uint32_t>(1);
    const uint32_t n_tiles  = get_arg_val<uint32_t>(2);
    const uint32_t tpg      = get_arg_val<uint32_t>(3);
    const uint32_t a_bcast  = get_arg_val<uint32_t>(4);
    const uint32_t b_bcast  = get_arg_val<uint32_t>(5);

    // B's TensorAccessorArgs immediately follow A's.
    constexpr uint32_t NA = TensorAccessorArgs<0>::num_compile_time_args();
    const auto a_acc = TensorAccessor(TensorAccessorArgs<0>(),  a_addr, get_tile_size(cb_a));
    const auto b_acc = TensorAccessor(TensorAccessorArgs<NA>(), b_addr, get_tile_size(cb_b));

    // B's runtime view block starts after A's.
    using AArgs = ttprm::TensorViewArgs<TTPRM_A_VIEW, view_base>;
    using BArgs = ttprm::TensorViewArgs<TTPRM_B_VIEW, AArgs::next()>;

    // Affine gathers share the resident pad seed.
    uint32_t zero_rp = 0;
    if constexpr (AArgs::gather_needs_pad_seed || BArgs::gather_needs_pad_seed) {
        cb_wait_front(cb_zero, 1);
        zero_rp = get_read_ptr(cb_zero);
    }
    auto a = AArgs::make(a_acc, zero_rp);
    auto b = BArgs::make(b_acc, zero_rp);
    const uint32_t a_tb = get_tile_size(cb_a);
    const uint32_t b_tb = get_tile_size(cb_b);

    // Issue all resident broadcast-group reads, then publish them after one barrier.
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
    if (a_bcast || b_bcast) noc_async_read_barrier();
    if (a_bcast) cb_push_back(cb_a, tpg);
    if (b_bcast) cb_push_back(cb_b, tpg);

    // Stream only non-broadcast operands; broadcast groups are already resident.
    for (uint32_t t = 0; t < n_tiles; t++) {
        if (!a_bcast) { cb_reserve_back(cb_a, 1); a.gather_tile(get_write_ptr(cb_a)); }
        if (!b_bcast) { cb_reserve_back(cb_b, 1); b.gather_tile(get_write_ptr(cb_b)); }
        if (!a_bcast || !b_bcast) noc_async_read_barrier();
        if (!a_bcast) { cb_push_back(cb_a, 1); a.advance(); }
        if (!b_bcast) { cb_push_back(cb_b, 1); b.advance(); }
    }
}
