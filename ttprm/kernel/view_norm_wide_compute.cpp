// Compute endpoint for wide per-row norms.
// One full row-block stays resident in cb_x. The kernel reduces it, then reuses
// the same resident tiles for normalization. Gamma/beta stream one tile at a
// time to keep L1 bounded for large widths.
//
//   TTPRM_L2  : y_i = x_i * rsqrt(Σ x² + eps)
//   TTPRM_RMS : y_i = x_i * rsqrt(mean(x²) + eps) * weight_i
//   TTPRM_LN  : mean=Σx/N ; var=Σx²/N - mean² ; y_i=(x_i-mean)*rsqrt(var+eps) ; y=y*gamma+beta
//
// Runtime args: 0 n_rb, 1 ctiles, 2 affine_kind.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/rsqrt.h"

#if !defined(TTPRM_L2) && !defined(TTPRM_RMS) && !defined(TTPRM_LN)
#error "view_norm_wide_compute requires TTPRM_L2, TTPRM_RMS, or TTPRM_LN"
#endif

namespace {
constexpr uint32_t cb_x      = 0;
constexpr uint32_t cb_gamma  = 2;
constexpr uint32_t cb_beta   = 3;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_eps    = 5;
constexpr uint32_t cb_out    = 16;
constexpr uint32_t cb_zero   = 31;
constexpr uint32_t cb_mean = 20, cb_msq = 21, cb_m2 = 22, cb_var = 23, cb_rstd = 24;
constexpr uint32_t cb_sq = 25, cb_xmm = 26, cb_nrm = 27, cb_tmp = 28;
constexpr uint32_t AFFINE_NONE = 0, AFFINE_SCALE = 1, AFFINE_SCALE_BIAS = 2;

inline void emit1(uint32_t cb) {
    cb_reserve_back(cb, 1);
    tile_regs_wait();
    pack_tile(0, cb);
    cb_push_back(cb, 1);
    tile_regs_release();
}
}  // namespace

void kernel_main() {
    const uint32_t n_rb       = get_arg_val<uint32_t>(0);
    const uint32_t ctiles     = get_arg_val<uint32_t>(1);
    const uint32_t affine_kind = get_arg_val<uint32_t>(2);
    const bool has_scale = affine_kind != AFFINE_NONE;
    const bool has_bias = affine_kind == AFFINE_SCALE_BIAS;

    binary_op_init_common(cb_x, cb_x, cb_out);

    fill_tile_init();
    cb_reserve_back(cb_zero, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    emit1(cb_zero);

    cb_wait_front(cb_scaler, 1);
    cb_wait_front(cb_eps, 1);

    for (uint32_t rb = 0; rb < n_rb; rb++) {
        cb_wait_front(cb_x, ctiles);

        // sum(x^2) -> cb_sq, then reduce to a per-row scalar.
        mul_tiles_init(cb_x, cb_x);
        for (uint32_t i = 0; i < ctiles; i++) {
            tile_regs_acquire();
            mul_tiles(cb_x, cb_x, i, i, 0);
            tile_regs_commit();
            emit1(cb_sq);
        }
        cb_wait_front(cb_sq, ctiles);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, cb_msq);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ctiles; i++)
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_sq, cb_scaler, i, 0, 0);
        tile_regs_commit();
        emit1(cb_msq);
        reduce_uninit();
        cb_pop_front(cb_sq, ctiles);

#if defined(TTPRM_L2) || defined(TTPRM_RMS)
        // rstd = rsqrt(sum(x^2) + eps) for L2, or rsqrt(mean_square + eps) for RMS.
        cb_wait_front(cb_msq, 1);
        add_tiles_init(cb_msq, cb_eps);
        tile_regs_acquire();
        add_tiles(cb_msq, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        emit1(cb_rstd);
        cb_pop_front(cb_msq, 1);

        // y_i = x_i * rstd over the resident row-block.
        cb_wait_front(cb_rstd, 1);
        for (uint32_t i = 0; i < ctiles; i++) {
            // Re-init each iteration: the has_scale block below switches the bcast
            // mode to ROWS, so cols must be restored before the next col-bcast mul
            // (an init is a hardware mode switch, not one-time setup). Mirrors LN.
            mul_bcast_cols_init_short(cb_x, cb_rstd);
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_x, cb_rstd, i, 0, 0);
            tile_regs_commit();
#if defined(TTPRM_RMS)
            emit1(has_scale ? cb_nrm : cb_out);
            if (has_scale) {
                cb_wait_front(cb_nrm, 1);
                cb_wait_front(cb_gamma, 1);
                mul_bcast_rows_init_short(cb_nrm, cb_gamma);
                tile_regs_acquire();
                mul_tiles_bcast_rows(cb_nrm, cb_gamma, 0, 0, 0);
                tile_regs_commit();
                emit1(cb_out);
                cb_pop_front(cb_nrm, 1);
                cb_pop_front(cb_gamma, 1);
            }
#else
            emit1(cb_out);
#endif
        }
        cb_pop_front(cb_rstd, 1);
        cb_pop_front(cb_x, ctiles);

#else  // TTPRM_LN
        // mean = sum(x) / N.
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x, cb_scaler, cb_mean);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ctiles; i++)
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_x, cb_scaler, i, 0, 0);
        tile_regs_commit();
        emit1(cb_mean);
        reduce_uninit();

        // var = msq - mean^2; rstd = rsqrt(var + eps).
        cb_wait_front(cb_mean, 1);
        cb_wait_front(cb_msq, 1);
        mul_tiles_init(cb_mean, cb_mean);
        tile_regs_acquire();
        mul_tiles(cb_mean, cb_mean, 0, 0, 0);
        tile_regs_commit();
        emit1(cb_m2);
        cb_wait_front(cb_m2, 1);
        sub_tiles_init(cb_msq, cb_m2);
        tile_regs_acquire();
        sub_tiles(cb_msq, cb_m2, 0, 0, 0);
        tile_regs_commit();
        emit1(cb_var);
        cb_pop_front(cb_m2, 1);
        cb_pop_front(cb_msq, 1);
        cb_wait_front(cb_var, 1);
        add_tiles_init(cb_var, cb_eps);
        tile_regs_acquire();
        add_tiles(cb_var, cb_eps, 0, 0, 0);
        rsqrt_tile_init();
        rsqrt_tile(0);
        tile_regs_commit();
        emit1(cb_rstd);
        cb_pop_front(cb_var, 1);

        // Normalize each resident tile; gamma/beta stream one tile at a time.
        cb_wait_front(cb_rstd, 1);
        cb_wait_front(cb_mean, 1);
        for (uint32_t i = 0; i < ctiles; i++) {
            // xmm = x_i - mean.
            sub_bcast_cols_init_short(cb_x, cb_mean);
            tile_regs_acquire();
            sub_tiles_bcast_cols(cb_x, cb_mean, i, 0, 0);
            tile_regs_commit();
            emit1(cb_xmm);
            // nrm = xmm * rstd.
            cb_wait_front(cb_xmm, 1);
            mul_bcast_cols_init_short(cb_xmm, cb_rstd);
            tile_regs_acquire();
            mul_tiles_bcast_cols(cb_xmm, cb_rstd, 0, 0, 0);
            tile_regs_commit();
            emit1(has_scale ? cb_nrm : cb_out);
            cb_pop_front(cb_xmm, 1);

            if (has_scale) {
                cb_wait_front(cb_nrm, 1);
                cb_wait_front(cb_gamma, 1);
                if (has_bias) cb_wait_front(cb_beta, 1);
                mul_bcast_rows_init_short(cb_nrm, cb_gamma);
                tile_regs_acquire();
                mul_tiles_bcast_rows(cb_nrm, cb_gamma, 0, 0, 0);
                tile_regs_commit();
                emit1(cb_tmp);
                cb_pop_front(cb_nrm, 1);
                cb_wait_front(cb_tmp, 1);
                add_bcast_rows_init_short(cb_tmp, cb_beta);
                tile_regs_acquire();
                add_tiles_bcast_rows(cb_tmp, cb_beta, 0, 0, 0);
                tile_regs_commit();
                emit1(cb_out);
                cb_pop_front(cb_tmp, 1);
                cb_pop_front(cb_gamma, 1);
                if (has_bias) cb_pop_front(cb_beta, 1);
            }
        }
        cb_pop_front(cb_mean, 1);
        cb_pop_front(cb_rstd, 1);
        cb_pop_front(cb_x, ctiles);
#endif
    }
}
