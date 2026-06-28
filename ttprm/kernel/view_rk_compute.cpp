// Compute endpoint for the fused RWKV7 rk update:
//     rk[h]      = Σ_i r[h][i] * k[h][i] * r_k[h][i]     (reduce over the head's S lanes)
//     out[h][i]  = cur[h][i] + v[h][i] * rk[h]           (rk a per-row scalar -> bcast across cols)
// Two phases run over each row-block:
//   phase 1: pw = (r*k)*r_k ; rk = reduce_ROW(pw)   (raw sum -> reduce scaler = 1.0)
//   phase 2: out = cur + v * rk(bcast cols)
// The reduce scaler tile is generated on the SFPU with the same face-row-0
// pattern as the norm kernels.
//
// Runtime args: 0 n_rb, 1 ctiles, 2 rscale_bits (1.0),
//               3 w_grp, 4 w_phase0, 5 w_resident.

#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/reduce.h"
#include "api/compute/bcast.h"

namespace {
constexpr uint32_t cb_r      = 0;
constexpr uint32_t cb_k      = 6;
constexpr uint32_t cb_w      = 7;
constexpr uint32_t cb_v      = 8;
constexpr uint32_t cb_cur    = 9;
constexpr uint32_t cb_scaler = 4;
constexpr uint32_t cb_out    = 16;
constexpr uint32_t cb_zero   = 31;
constexpr uint32_t cb_ss   = 21;   // rk per-head scalar
constexpr uint32_t cb_pw   = 27;   // (r*k)*r_k

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
    const uint32_t rscale_bits = get_arg_val<uint32_t>(2);
    const uint32_t w_grp       = get_arg_val<uint32_t>(3);
    const uint32_t w_phase0    = get_arg_val<uint32_t>(4);
    const uint32_t w_resident  = get_arg_val<uint32_t>(5);

    binary_op_init_common(cb_r, cb_k, cb_out);

    // Pad seed for affine input gathers; CB 31 remains resident.
    fill_tile_init();
    cb_reserve_back(cb_zero, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    emit1(cb_zero);

    // Reduce scaler: face-row-0 pattern on the SFPU.
    cb_reserve_back(cb_scaler, 1);
    tile_regs_acquire();
    MATH(gen_reduce_scaler_sfpi(bits_to_f32(rscale_bits)));
    tile_regs_commit();
    emit1(cb_scaler);

    cb_wait_front(cb_scaler, 1);
    if (w_resident) cb_wait_front(cb_w, w_grp * ctiles);

    for (uint32_t rb = 0; rb < n_rb; rb++) {
        // Phase 1: rk[h] = sum_i r_i * k_i * r_k_i.
        cb_wait_front(cb_r, ctiles);
        cb_wait_front(cb_k, ctiles);
        if (!w_resident) cb_wait_front(cb_w, ctiles);
        const uint32_t w0 = w_resident ? ((w_phase0 + rb) % w_grp) * ctiles : 0u;
        mul_tiles_init(cb_r, cb_k);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ctiles; i++) mul_tiles(cb_r, cb_k, i, i, i);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWMUL,
                                     EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w);
        for (uint32_t i = 0; i < ctiles; i++)
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWMUL,
                                    EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_w, w0 + i, i);
        tile_regs_commit();
        emitN(cb_pw, ctiles);
        cb_pop_front(cb_r, ctiles);
        cb_pop_front(cb_k, ctiles);
        if (!w_resident) cb_pop_front(cb_w, ctiles);

        // rk = sum_i reduce_ROW(pw_i).
        cb_wait_front(cb_pw, ctiles);
        reduce_init<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_pw, cb_scaler, cb_ss);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ctiles; i++)
            reduce_tile<PoolType::SUM, ReduceDim::REDUCE_ROW>(cb_pw, cb_scaler, i, 0, 0);
        tile_regs_commit();
        emit1(cb_ss);
        reduce_uninit();
        cb_pop_front(cb_pw, ctiles);

        // Phase 2: out_i = cur_i + v_i * rk.
        cb_wait_front(cb_ss, 1);
        cb_wait_front(cb_v, ctiles);
        cb_wait_front(cb_cur, ctiles);
        mul_bcast_cols_init_short(cb_v, cb_ss);
        tile_regs_acquire();
        for (uint32_t i = 0; i < ctiles; i++) mul_tiles_bcast_cols(cb_v, cb_ss, i, 0, i);
        binary_dest_reuse_tiles_init<EltwiseBinaryType::ELWADD,
                                     EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cur);
        for (uint32_t i = 0; i < ctiles; i++)
            binary_dest_reuse_tiles<EltwiseBinaryType::ELWADD,
                                    EltwiseBinaryReuseDestType::DEST_TO_SRCA>(cb_cur, i, i);
        tile_regs_commit();
        emitN(cb_out, ctiles);
        cb_pop_front(cb_v, ctiles);
        cb_pop_front(cb_ss, 1);
        cb_pop_front(cb_cur, ctiles);
    }
    if (w_resident) cb_pop_front(cb_w, w_grp * ctiles);
}
