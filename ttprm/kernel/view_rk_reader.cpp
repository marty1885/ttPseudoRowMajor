// Reader endpoint for the fused RWKV7 rk update:
//     rk[h]  = Σ_i r[h][i] * k[h][i] * r_k[h][i]        (per-head scalar, reduce over the lanes)
//     out[h] = cur[h] + v[h] * rk[h]                    (rk broadcast across the head's lanes)
// Five operands are presented on the same head grid. r/k/v/cur stream one
// row-block at a time. r_k either streams with the same row-block FIFO protocol
// or, for token-broadcast when reuse is profitable for this core's assignment,
// is loaded once as a resident token group and re-indexed by compute.
//
// Compile args: r, k, r_k, v, cur TensorAccessorArgs (in that order).
// Defines: TTPRM_R_VIEW / TTPRM_K_VIEW / TTPRM_W_VIEW / TTPRM_V_VIEW / TTPRM_CUR_VIEW.
// Runtime args: 0 r_addr 1 k_addr 2 w_addr 3 v_addr 4 cur_addr 5 n_rb 6 ctiles
//               7 w_grp (rbs_per_group, 0=no bcast) 8 w_phase0 9 w_resident
//               10.. five view blocks.

#include <cstdint>

#include "tensor_view.hpp"

#ifndef TTPRM_R_VIEW
#define TTPRM_R_VIEW ttprm::AffineTensorView
#endif
#ifndef TTPRM_K_VIEW
#define TTPRM_K_VIEW ttprm::AffineTensorView
#endif
#ifndef TTPRM_W_VIEW
#define TTPRM_W_VIEW ttprm::AffineTensorView
#endif
#ifndef TTPRM_V_VIEW
#define TTPRM_V_VIEW ttprm::AffineTensorView
#endif
#ifndef TTPRM_CUR_VIEW
#define TTPRM_CUR_VIEW ttprm::AffineTensorView
#endif

namespace {
constexpr uint32_t cb_r   = 0;
constexpr uint32_t cb_k   = 6;
constexpr uint32_t cb_w   = 7;
constexpr uint32_t cb_v   = 8;
constexpr uint32_t cb_cur = 9;
constexpr uint32_t cb_zero = 31;
constexpr uint32_t view_base = 10;
}  // namespace

void kernel_main() {
    const uint32_t r_addr   = get_arg_val<uint32_t>(0);
    const uint32_t k_addr   = get_arg_val<uint32_t>(1);
    const uint32_t w_addr   = get_arg_val<uint32_t>(2);
    const uint32_t v_addr   = get_arg_val<uint32_t>(3);
    const uint32_t cur_addr = get_arg_val<uint32_t>(4);
    const uint32_t n_rb     = get_arg_val<uint32_t>(5);
    const uint32_t ctiles   = get_arg_val<uint32_t>(6);
    const uint32_t w_grp    = get_arg_val<uint32_t>(7);
    uint32_t       wphase   = get_arg_val<uint32_t>(8);
    const uint32_t w_resident = get_arg_val<uint32_t>(9);

    // Runtime view blocks are packed R, K, W, V, Cur.
    using RArgs   = ttprm::TensorViewArgs<TTPRM_R_VIEW,   view_base>;
    using KArgs   = ttprm::TensorViewArgs<TTPRM_K_VIEW,   RArgs::next()>;
    using WArgs   = ttprm::TensorViewArgs<TTPRM_W_VIEW,   KArgs::next()>;
    using VArgs   = ttprm::TensorViewArgs<TTPRM_V_VIEW,   WArgs::next()>;
    using CurArgs = ttprm::TensorViewArgs<TTPRM_CUR_VIEW, VArgs::next()>;

    uint32_t zero_rp = 0;
    if constexpr (RArgs::gather_needs_pad_seed || KArgs::gather_needs_pad_seed ||
                  WArgs::gather_needs_pad_seed || VArgs::gather_needs_pad_seed ||
                  CurArgs::gather_needs_pad_seed) {
        cb_wait_front(cb_zero, 1);
        zero_rp = get_read_ptr(cb_zero);
    }

    // TensorAccessorArgs are packed R, K, W, V, Cur.
    constexpr uint32_t NR = TensorAccessorArgs<0>::num_compile_time_args();
    constexpr uint32_t NK = TensorAccessorArgs<NR>::num_compile_time_args();
    constexpr uint32_t NW = TensorAccessorArgs<NR + NK>::num_compile_time_args();
    constexpr uint32_t NV = TensorAccessorArgs<NR + NK + NW>::num_compile_time_args();
    const auto r_acc   = TensorAccessor(TensorAccessorArgs<0>(),                r_addr,   get_tile_size(cb_r));
    const auto k_acc   = TensorAccessor(TensorAccessorArgs<NR>(),               k_addr,   get_tile_size(cb_k));
    const auto w_acc   = TensorAccessor(TensorAccessorArgs<NR + NK>(),          w_addr,   get_tile_size(cb_w));
    const auto v_acc   = TensorAccessor(TensorAccessorArgs<NR + NK + NW>(),     v_addr,   get_tile_size(cb_v));
    const auto cur_acc = TensorAccessor(TensorAccessorArgs<NR + NK + NW + NV>(), cur_addr, get_tile_size(cb_cur));

    auto r   = RArgs::make(r_acc, zero_rp);
    auto k   = KArgs::make(k_acc, zero_rp);
    auto w   = WArgs::make(w_acc, zero_rp);
    auto v   = VArgs::make(v_acc, zero_rp);
    auto cur = CurArgs::make(cur_acc, zero_rp);

    const uint32_t r_tb   = get_tile_size(cb_r);
    const uint32_t k_tb   = get_tile_size(cb_k);
    const uint32_t w_tb   = get_tile_size(cb_w);
    const uint32_t v_tb   = get_tile_size(cb_v);
    const uint32_t cur_tb = get_tile_size(cb_cur);

    if (w_resident) {
        const uint32_t w_tiles = w_grp * ctiles;
        cb_reserve_back(cb_w, w_tiles);
        const uint32_t wwp = get_write_ptr(cb_w);
        for (uint32_t i = 0; i < w_tiles; i++) { w.gather_tile(wwp + i * w_tb); w.advance(); }
        noc_async_read_barrier();
        cb_push_back(cb_w, w_tiles);
    }

    for (uint32_t rb = 0; rb < n_rb; rb++) {
        cb_reserve_back(cb_r, ctiles);
        cb_reserve_back(cb_k, ctiles);
        if (!w_resident) cb_reserve_back(cb_w, ctiles);
        const uint32_t rwp   = get_write_ptr(cb_r);
        const uint32_t kwp   = get_write_ptr(cb_k);
        for (uint32_t i = 0; i < ctiles; i++) { r.gather_tile(rwp + i * r_tb);     r.advance(); }
        for (uint32_t i = 0; i < ctiles; i++) { k.gather_tile(kwp + i * k_tb);     k.advance(); }
        if (!w_resident) {
            const uint32_t wwp = get_write_ptr(cb_w);
            // Rewind token-broadcast r_k at group boundaries.
            if (w_grp && wphase == w_grp) { w.reset(0); wphase = 0; }
            for (uint32_t i = 0; i < ctiles; i++) { w.gather_tile(wwp + i * w_tb); w.advance(); }
            if (w_grp) wphase++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_r, ctiles);
        cb_push_back(cb_k, ctiles);
        if (!w_resident) cb_push_back(cb_w, ctiles);

        cb_reserve_back(cb_v, ctiles);
        cb_reserve_back(cb_cur, ctiles);
        const uint32_t vwp   = get_write_ptr(cb_v);
        const uint32_t curwp = get_write_ptr(cb_cur);
        for (uint32_t i = 0; i < ctiles; i++) { v.gather_tile(vwp + i * v_tb);     v.advance(); }
        for (uint32_t i = 0; i < ctiles; i++) { cur.gather_tile(curwp + i * cur_tb); cur.advance(); }
        noc_async_read_barrier();
        cb_push_back(cb_v, ctiles);
        cb_push_back(cb_cur, ctiles);
    }
}
