// Compute endpoint for the fused RWKV7 k update:
//     out = k + (a - 1) * (k * k_a)
//         = k + a * tmp - tmp, tmp = k * k_a
//
// Uses the same reader ABI as view_ternary_reader.cpp:
//   cb_a = k, cb_b = a, cb_c = k_a.
// Broadcast operands are resident `tpg`-tile groups.
// Runtime args: 0 n_tiles, 1 tpg, 2 i0, 3 k_bcast, 4 a_bcast, 5 ka_bcast.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"

namespace {
constexpr uint32_t cb_k     = 0;
constexpr uint32_t cb_a     = 1;
constexpr uint32_t cb_ka    = 2;
constexpr uint32_t cb_out   = 16;
constexpr uint32_t cb_zero  = 31;
constexpr uint32_t cb_tmp   = 25;  // k * k_a
constexpr uint32_t cb_prod  = 26;  // a * tmp
constexpr uint32_t cb_delta = 27;  // prod - tmp

inline void emit1(uint32_t cb) {
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    pack_tile(0, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    const uint32_t n_tiles  = get_arg_val<uint32_t>(0);
    const uint32_t tpg      = get_arg_val<uint32_t>(1);
    const uint32_t i0       = get_arg_val<uint32_t>(2);
    const uint32_t k_bcast  = get_arg_val<uint32_t>(3);
    const uint32_t a_bcast  = get_arg_val<uint32_t>(4);
    const uint32_t ka_bcast = get_arg_val<uint32_t>(5);

    binary_op_init_common(cb_k, cb_ka, cb_out);

    // Pad seed for affine input gathers; CB 31 remains resident.
    fill_tile_init();
    cb_reserve_back(cb_zero, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    emit1(cb_zero);

    if (k_bcast) cb_wait_front(cb_k, tpg);
    if (a_bcast) cb_wait_front(cb_a, tpg);
    if (ka_bcast) cb_wait_front(cb_ka, tpg);

    for (uint32_t t = 0; t < n_tiles; t++) {
        const uint32_t j = (i0 + t) % tpg;
        const uint32_t ik  = k_bcast  ? j : 0;
        const uint32_t ia  = a_bcast  ? j : 0;
        const uint32_t ika = ka_bcast ? j : 0;
        if (!k_bcast) cb_wait_front(cb_k, 1);
        if (!a_bcast) cb_wait_front(cb_a, 1);
        if (!ka_bcast) cb_wait_front(cb_ka, 1);

        // tmp = k * k_a
        mul_tiles_init(cb_k, cb_ka);
        tile_regs_acquire();
        mul_tiles(cb_k, cb_ka, ik, ika, 0);
        tile_regs_commit();
        emit1(cb_tmp);

        // prod = a * tmp
        cb_wait_front(cb_tmp, 1);
        mul_tiles_init(cb_a, cb_tmp);
        tile_regs_acquire();
        mul_tiles(cb_a, cb_tmp, ia, 0, 0);
        tile_regs_commit();
        emit1(cb_prod);

        // delta = prod - tmp
        cb_wait_front(cb_prod, 1);
        sub_tiles_init(cb_prod, cb_tmp);
        tile_regs_acquire();
        sub_tiles(cb_prod, cb_tmp, 0, 0, 0);
        tile_regs_commit();
        emit1(cb_delta);
        cb_pop_front(cb_prod, 1);
        cb_pop_front(cb_tmp, 1);

        // out = delta + k
        cb_wait_front(cb_delta, 1);
        add_tiles_init(cb_delta, cb_k);
        tile_regs_acquire();
        add_tiles(cb_delta, cb_k, 0, ik, 0);
        tile_regs_commit();
        emit1(cb_out);
        cb_pop_front(cb_delta, 1);

        if (!k_bcast) cb_pop_front(cb_k, 1);
        if (!a_bcast) cb_pop_front(cb_a, 1);
        if (!ka_bcast) cb_pop_front(cb_ka, 1);
    }
    if (k_bcast) cb_pop_front(cb_k, tpg);
    if (a_bcast) cb_pop_front(cb_a, tpg);
    if (ka_bcast) cb_pop_front(cb_ka, tpg);
}
