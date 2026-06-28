// Compute endpoint for global norms over folded-dense blocks.
// The reader gathers live elements as folded tiles and this kernel reduces over
// all folded tiles with REDUCE_SCALAR. The resulting mean/rstd is a single
// scalar broadcast back over the folded block. Gamma/beta use the same fold, so
// affine application is plain elementwise.
//
//   TTPRM_L2  : y = x * rsqrt(Σx² + eps)                         (scaler 1.0, no affine)
//   TTPRM_RMS : y = x * rsqrt(mean(x²)+eps) * weight              (weight optional)
//   TTPRM_LN  : mean=Σx/N ; var=Σx²/N - mean² ; y=(x-mean)*rstd ; y=y*gamma+beta
//
// Runtime args: 0 ntiles, 1 affine_kind, 2 eps_bits, 3 rscale_bits.
// REDUCE_SCALAR applies the scaler tile twice, so the host passes 1/sqrt(N)
// for RMS/LN and 1.0 for L2.
// Output: ntiles folded tiles in cb_out; the writer unfolds them back to flat[1,N].

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"

#if !defined(TTPRM_L2) && !defined(TTPRM_RMS) && !defined(TTPRM_LN)
#error "view_norm_global_compute requires TTPRM_L2, TTPRM_RMS, or TTPRM_LN"
#endif

namespace {
constexpr uint32_t cb_x      = 0;
constexpr uint32_t cb_gamma  = 2;
constexpr uint32_t cb_beta   = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_eps    = 5;
constexpr uint32_t cb_out    = 16;
constexpr uint32_t cb_zero   = 31;
constexpr uint32_t cb_mean = 20, cb_var = 23, cb_rstd = 24;
constexpr uint32_t cb_sq = 25, cb_xmm = 26, cb_norm = 27, cb_tmp = 28;
constexpr uint32_t AFFINE_NONE = 0, AFFINE_SCALE = 1, AFFINE_SCALE_BIAS = 2;

// Pack dst[0..n) after one tile_regs_wait; callers fill all n DST registers in
// the current acquire/commit cycle.
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

// REDUCE scaler tile: scale value in face-row 0 of every face, zero elsewhere.
// REDUCE_ROW and REDUCE_SCALAR both consume this pattern.
#ifdef TRISC_MATH
inline void gen_reduce_scaler_sfpi(float rscale) {
    using namespace sfpi;
    math::set_dst_write_addr<DstTileShape::Tile32x32, UnpackDestination::SrcRegs>(0);
    math::set_addr_mod_base();
    TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);
    vFloat rs = rscale;
    for (int f = 0; f < 4; f++) {
        int b = f * 8;
        v_if (vConstTileId < 16) {
            dst_reg[b + 0] = rs;
            dst_reg[b + 1] = rs;
        } v_else {
            dst_reg[b + 0] = 0.0f;
            dst_reg[b + 1] = 0.0f;
        }
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
    const uint32_t ntiles     = get_arg_val<uint32_t>(0);
    const uint32_t affine_kind = get_arg_val<uint32_t>(1);
    const uint32_t eps_bits   = get_arg_val<uint32_t>(2);
    const uint32_t rscale_bits = get_arg_val<uint32_t>(3);
    const bool has_scale = affine_kind != AFFINE_NONE;
    const bool has_bias = affine_kind == AFFINE_SCALE_BIAS;

    binary_op_init_common(cb_x, cb_x, cb_out);

    // Pad seed for affine input gathers; CB 31 remains resident.
    fill_tile_init();
    cb_reserve_back(cb_zero, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    emit1(cb_zero);

    // Generate uniform eps and face-row-0 reduce scaler on the SFPU.
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
    cb_wait_front(cb_x, ntiles);
    if (has_scale) cb_wait_front(cb_gamma, ntiles);
    if (has_bias) cb_wait_front(cb_beta, ntiles);

#if defined(TTPRM_L2) || defined(TTPRM_RMS)
    // ss/msq = sum_i reduce_SCALAR(x_i^2).
    mul_tiles_init(cb_x, cb_x);
    tile_regs_acquire();
    for (uint32_t i = 0; i < ntiles; i++) mul_tiles(cb_x, cb_x, i, i, i);
    tile_regs_commit();
    emitN(cb_sq, ntiles);
    cb_wait_front(cb_sq, ntiles);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, cb_var);
    tile_regs_acquire();
    for (uint32_t i = 0; i < ntiles; i++)
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, i, 0, 0);
    tile_regs_commit();
    emit1(cb_var);
    reduce_uninit();
    cb_pop_front(cb_sq, ntiles);
    // rstd = rsqrt(ss + eps) for L2, or rsqrt(mean_square + eps) for RMS.
    cb_wait_front(cb_var, 1);
    add_tiles_init(cb_var, cb_eps);
    tile_regs_acquire();
    add_tiles(cb_var, cb_eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    emit1(cb_rstd);
    cb_pop_front(cb_var, 1);
    // y_i = x_i * rstd.
    const uint32_t norm_dst =
#if defined(TTPRM_RMS)
        has_scale ? cb_norm : cb_out;
#else
        cb_out;
#endif
    cb_wait_front(cb_rstd, 1);
    mul_tiles_bcast_scalar_init_short(cb_x, cb_rstd);
    tile_regs_acquire();
    for (uint32_t i = 0; i < ntiles; i++) mul_tiles_bcast_scalar(cb_x, cb_rstd, i, 0, i);
    tile_regs_commit();
    emitN(norm_dst, ntiles);
    cb_pop_front(cb_rstd, 1);
    cb_pop_front(cb_x, ntiles);

#if defined(TTPRM_RMS)
    if (has_scale) {
        cb_wait_front(cb_norm, ntiles);
        mul_tiles_init(cb_norm, cb_gamma);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ntiles; i++) mul_tiles(cb_norm, cb_gamma, i, i, i);
        tile_regs_commit();
        emitN(cb_out, ntiles);
        cb_pop_front(cb_norm, ntiles);
    }
#endif

#else  // TTPRM_LN: two-pass variance over folded tiles.
    // mean = sum_i reduce_SCALAR(x_i).
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_x, cb_scaler, cb_mean);
    tile_regs_acquire();
    for (uint32_t i = 0; i < ntiles; i++)
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_x, cb_scaler, i, 0, 0);
    tile_regs_commit();
    emit1(cb_mean);
    reduce_uninit();

    // xmm_i = x_i - mean; reused for variance and normalization.
    cb_wait_front(cb_mean, 1);
    sub_tiles_bcast_scalar_init_short(cb_x, cb_mean);
    tile_regs_acquire();
    for (uint32_t i = 0; i < ntiles; i++) sub_tiles_bcast_scalar(cb_x, cb_mean, i, 0, i);
    tile_regs_commit();
    emitN(cb_xmm, ntiles);
    cb_pop_front(cb_mean, 1);
    cb_pop_front(cb_x, ntiles);

    // sq_i = xmm_i^2; var = sum_i reduce_SCALAR(sq_i).
    cb_wait_front(cb_xmm, ntiles);
    mul_tiles_init(cb_xmm, cb_xmm);
    tile_regs_acquire();
    for (uint32_t i = 0; i < ntiles; i++) mul_tiles(cb_xmm, cb_xmm, i, i, i);
    tile_regs_commit();
    emitN(cb_sq, ntiles);
    cb_wait_front(cb_sq, ntiles);
    reduce_init<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, cb_var);
    tile_regs_acquire();
    for (uint32_t i = 0; i < ntiles; i++)
        reduce_tile<PoolType::SUM, ReduceDim::REDUCE_SCALAR>(cb_sq, cb_scaler, i, 0, 0);
    tile_regs_commit();
    emit1(cb_var);
    reduce_uninit();
    cb_pop_front(cb_sq, ntiles);

    // rstd = rsqrt(var + eps).
    cb_wait_front(cb_var, 1);
    add_tiles_init(cb_var, cb_eps);
    tile_regs_acquire();
    add_tiles(cb_var, cb_eps, 0, 0, 0);
    rsqrt_tile_init();
    rsqrt_tile(0);
    tile_regs_commit();
    emit1(cb_rstd);
    cb_pop_front(cb_var, 1);

    // norm_i = xmm_i * rstd.
    const uint32_t norm_dst = has_scale ? cb_norm : cb_out;
    cb_wait_front(cb_rstd, 1);
    cb_wait_front(cb_xmm, ntiles);
    mul_tiles_bcast_scalar_init_short(cb_xmm, cb_rstd);
    tile_regs_acquire();
    for (uint32_t i = 0; i < ntiles; i++) mul_tiles_bcast_scalar(cb_xmm, cb_rstd, i, 0, i);
    tile_regs_commit();
    emitN(norm_dst, ntiles);
    cb_pop_front(cb_xmm, ntiles);
    cb_pop_front(cb_rstd, 1);

    if (has_scale) {
        // gamma/beta use the same fold as x, so affine is elementwise.
        cb_wait_front(cb_norm, ntiles);
        mul_tiles_init(cb_norm, cb_gamma);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ntiles; i++) mul_tiles(cb_norm, cb_gamma, i, i, i);
        if (has_bias) {
            binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD,
                                         EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_beta);
            for (uint32_t i = 0; i < ntiles; i++)
                binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD,
                                        EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_beta, i, i);
        }
        tile_regs_commit();
        emitN(cb_out, ntiles);
        cb_pop_front(cb_norm, ntiles);
    }
#endif
    if (has_scale) cb_pop_front(cb_gamma, ntiles);
    if (has_bias) cb_pop_front(cb_beta, ntiles);
}
