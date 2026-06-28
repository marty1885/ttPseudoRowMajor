// Compute endpoint for fused mul_l2_norm: y = l2_norm(a * b) per head.
// The product tiles stay resident so they can be reused for both the square
// reduction and the final scale.
//
// Runtime args: 0 n_rb, 1 ctiles, 2 eps_bits, 3 rscale_bits (1.0 for l2).

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"

namespace {
constexpr uint32_t cb_a      = 0;
constexpr uint32_t cb_b      = 6;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_eps    = 5;
constexpr uint32_t cb_out    = 16;
constexpr uint32_t cb_zero   = 31;
constexpr uint32_t cb_msq    = 21, cb_rstd = 24, cb_sq = 25, cb_prod = 26;

inline void emitN(uint32_t cb, uint32_t n) {
    cb_reserve_back(cb, n);
    tile_regs_wait();
    for (uint32_t i = 0; i < n; i++) pack_tile(i, cb);
    cb_push_back(cb, n);
    tile_regs_release();
}
inline void emit1(uint32_t cb) { emitN(cb, 1); }
inline float bits_to_f32(uint32_t b) { union { uint32_t u; float f; } c; c.u = b; return c.f; }
}  // namespace

// Reduce scaler tile: scale value in face-row 0 of each face, zero elsewhere.
#ifdef TRISC_MATH
inline void gen_reduce_scaler_sfpi(float rscale) {
    using namespace sfpi;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    vFloat rs = rscale;
    for (int f = 0; f < 4; f++) {
        int b = f * 8;
        v_if (vConstTileId < 16) { dst_reg[b + 0] = rs; dst_reg[b + 1] = rs; }
        v_else                   { dst_reg[b + 0] = 0.0f; dst_reg[b + 1] = 0.0f; }
        v_endif;
        dst_reg[b + 2] = 0.0f; dst_reg[b + 3] = 0.0f;
        dst_reg[b + 4] = 0.0f; dst_reg[b + 5] = 0.0f;
        dst_reg[b + 6] = 0.0f; dst_reg[b + 7] = 0.0f;
    }
    math::clear_dst_reg_addr();
    TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);
    TTI_SETC16(2, 0);
}
#endif

void kernel_main() {
    const uint32_t n_rb        = get_arg_val<uint32_t>(0);
    const uint32_t ctiles      = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits    = get_arg_val<uint32_t>(2);
    const uint32_t rscale_bits = get_arg_val<uint32_t>(3);

    binary_op_init_common(cb_a, cb_b, cb_out);

    fill_tile_init();
    cb_reserve_back(cb_zero, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    emit1(cb_zero);

    cb_reserve_back(cb_eps, 1);
    tile_regs_acquire();
    fill_tile_bitcast(0, eps_bits);
    tile_regs_commit();
    emit1(cb_eps);

    cb_reserve_back(cb_scaler, 1);
    tile_regs_acquire();
    MATH(gen_reduce_scaler_sfpi(bits_to_f32(rscale_bits)));
    tile_regs_commit();
    emit1(cb_scaler);

    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    for (uint32_t rb = 0; rb < n_rb; rb++) {
        cb_wait_front(cb_a, ctiles);
        cb_wait_front(cb_b, ctiles);

        // prod_i = a_i * b_i.
        mul_tiles_init(cb_a, cb_b);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ctiles; i++) mul_tiles(cb_a, cb_b, i, i, i);
        tile_regs_commit();
        emitN(cb_prod, ctiles);
        cb_pop_front(cb_a, ctiles);
        cb_pop_front(cb_b, ctiles);

        // sq_i = prod_i^2.
        cb_wait_front(cb_prod, ctiles);
        mul_tiles_init(cb_prod, cb_prod);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ctiles; i++) mul_tiles(cb_prod, cb_prod, i, i, i);
        tile_regs_commit();
        emitN(cb_sq, ctiles);

        // ss = sum_i reduce_ROW(sq_i).
        cb_wait_front(cb_sq, ctiles);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_msq);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ctiles; i++)
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, i, 0, 0);
        tile_regs_commit();
        emit1(cb_msq);
        reduce_uninit();
        cb_pop_front(cb_sq, ctiles);

        // rstd = rsqrt(ss + eps).
        cb_wait_front(cb_msq, 1);
        add_tiles_init(cb_msq, cb_eps);
        tile_regs_acquire();
        add_tiles(cb_msq, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        emit1(cb_rstd);
        cb_pop_front(cb_msq, 1);

        // y_i = prod_i * rstd.
        cb_wait_front(cb_rstd, 1);
        mul_bcast_cols_init_short(cb_prod, cb_rstd);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ctiles; i++) mul_tiles_bcast_cols(cb_prod, cb_rstd, i, 0, i);
        tile_regs_commit();
        emitN(cb_out, ctiles);
        cb_pop_front(cb_rstd, 1);
        cb_pop_front(cb_prod, ctiles);
    }
}
