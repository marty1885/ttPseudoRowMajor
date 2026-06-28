// Compute endpoint for fused ternary elementwise ops.
// Produces the resident zero tile in CB 31, then emits one output tile per work
// item. TTPRM_FMA selects out = a*b + c; TTPRM_LERP selects out = a + c*(b-a).
//
// Broadcast operands are resident `tpg`-tile groups. The compute waits once, then
// indexes tile `(i0 + t) % tpg` for each output tile. Streaming operands consume
// one front tile per output.
//
// Compile define: TTPRM_FMA or TTPRM_LERP (exactly one).
// Runtime args: 0 n_tiles, 1 tpg, 2 i0, 3 a_bcast, 4 b_bcast, 5 c_bcast.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"

#if defined(TTPRM_FMA)
#include "api/compute/eltwise_binary.h"
#elif defined(TTPRM_LERP)
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/lerp.h"
#else
#error "view_ternary_compute requires TTPRM_FMA or TTPRM_LERP"
#endif

namespace {
constexpr uint32_t cb_a = 0, cb_b = 1, cb_c = 2, cb_out = 16, cb_zero = 31;
}  // namespace

void kernel_main() {
    const uint32_t n_tiles = get_arg_val<uint32_t>(0);
    const uint32_t tpg     = get_arg_val<uint32_t>(1);
    const uint32_t i0      = get_arg_val<uint32_t>(2);
    const uint32_t a_bcast = get_arg_val<uint32_t>(3);
    const uint32_t b_bcast = get_arg_val<uint32_t>(4);
    const uint32_t c_bcast = get_arg_val<uint32_t>(5);

#if defined(TTPRM_FMA)
    binary_op_init_common(cb_a, cb_b, cb_out);
#else
    unary_op_init_common(cb_a, cb_out);
#endif

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
    if (c_bcast) cb_wait_front(cb_c, tpg);

    for (uint32_t t = 0; t < n_tiles; t++) {
        const uint32_t j = (i0 + t) % tpg;
        const uint32_t ia = a_bcast ? j : 0;
        const uint32_t ib = b_bcast ? j : 0;
        const uint32_t ic = c_bcast ? j : 0;
        if (!a_bcast) cb_wait_front(cb_a, 1);
        if (!b_bcast) cb_wait_front(cb_b, 1);
        if (!c_bcast) cb_wait_front(cb_c, 1);

        tile_regs_acquire();
#if defined(TTPRM_FMA)
        // dst0 = a*b; then add c through destination reuse.
        mul_tiles_init(cb_a, cb_b);
        mul_tiles(cb_a, cb_b, ia, ib, 0);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD,
                                     EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_c);
        binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD,
                                EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_c, ic, 0);
#else
        // SFPU ternary lerp uses one DST tile per operand.
        copy_tile_to_dst_init_short(cb_a); copy_tile(cb_a, ia, 0);
        copy_tile_to_dst_init_short(cb_b); copy_tile(cb_b, ib, 1);
        copy_tile_to_dst_init_short(cb_c); copy_tile(cb_c, ic, 2);
        lerp_tile_init();
        lerp_tile<DataFormat::Float16_b>(0, 1, 2, 0);
#endif
        tile_regs_commit();

        cb_reserve_back(cb_out, 1);
        tile_regs_wait();
        pack_tile(0, cb_out);
        cb_push_back(cb_out, 1);
        tile_regs_release();

        if (!a_bcast) cb_pop_front(cb_a, 1);
        if (!b_bcast) cb_pop_front(cb_b, 1);
        if (!c_bcast) cb_pop_front(cb_c, 1);
    }
    if (a_bcast) cb_pop_front(cb_a, tpg);
    if (b_bcast) cb_pop_front(cb_b, tpg);
    if (c_bcast) cb_pop_front(cb_c, tpg);
}
