// Compute endpoint for fused binary elementwise ops.
// Produces the resident zero tile in CB 31, then emits one output tile per work
// item: out = a OP b, where TTPRM_BINOP selects add, sub, or mul.
//
// Broadcast operands are resident `tpg`-tile groups. The compute waits once, then
// indexes tile `(i0 + t) % tpg` for each output tile. Streaming operands consume
// one front tile per output.
//
// Compile define: TTPRM_BINOP (add|sub|mul).
// Runtime args: 0 n_tiles, 1 tpg, 2 i0, 3 a_bcast, 4 b_bcast.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"

#ifndef TTPRM_BINOP
#define TTPRM_BINOP add
#endif
#define TTPRM_CAT_(a, b) a##b
#define TTPRM_CAT(a, b) TTPRM_CAT_(a, b)
#define TTPRM_OP_INIT  TTPRM_CAT(TTPRM_BINOP, _tiles_init)
#define TTPRM_OP_TILES TTPRM_CAT(TTPRM_BINOP, _tiles)

namespace {
constexpr uint32_t cb_a = 0, cb_b = 1, cb_out = 16, cb_zero = 31;
}  // namespace

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t tpg     = get_arg_val<uint32_t>(1);
    const uint32_t i0      = get_arg_val<uint32_t>(2);
    const uint32_t a_bcast = get_arg_val<uint32_t>(3);
    const uint32_t b_bcast = get_arg_val<uint32_t>(4);

    binary_op_init_common(cb_a, cb_b, cb_out);

    // Pad seed for affine input gathers; CB 31 remains resident.
    fill_tile_init();
    cb_reserve_back(cb_zero, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_zero);
    cb_push_back(cb_zero, 1);
    tile_regs_release();

    // Resident broadcast groups are re-indexed by tile position; streamed operands use FIFO fronts.
    if (a_bcast) cb_wait_front(cb_a, tpg);
    if (b_bcast) cb_wait_front(cb_b, tpg);

    TTPRM_OP_INIT(cb_a, cb_b);
    for (uint32_t t = 0; t < n_tiles; t++) {
        const uint32_t j = (i0 + t) % tpg;
        uint32_t ia = j, ib = j;
        if (!a_bcast) { cb_wait_front(cb_a, 1); ia = 0; }
        if (!b_bcast) { cb_wait_front(cb_b, 1); ib = 0; }
        tile_regs_acquire();
        TTPRM_OP_TILES(cb_a, cb_b, ia, ib, 0);
        tile_regs_commit();
        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();
        if (!a_bcast) cb_pop_front(cb_a, 1);
        if (!b_bcast) cb_pop_front(cb_b, 1);
    }
    if (a_bcast) cb_pop_front(cb_a, tpg);
    if (b_bcast) cb_pop_front(cb_b, tpg);
}
