// Device-side tensor-view types.
// The access strategy is selected by type, while dimensions remain runtime data.
// This keeps the hot path branch-free without recompiling for each shape:
//
//   DenseTensorView  -- the tensor IS the dense tiling. One whole-page (tile) transfer.
//   AffineTensorView -- the tensor is VIEWED. Walk the source coord (sr,sc) with a
//                       strength-reduced odometer; one 32B face-row per transfer.
//
// The same view type supports gather_tile for readers and scatter_tile for
// writers, so relayout and producer scatter use the same endpoint contract.
//
// gather_tile/scatter_tile only issue NoC transfers. Callers own barriers and
// may batch several tiles before waiting.

#pragma once

#include <cstdint>

#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"  // tt_memmove
#include "view_kind.hpp"                                                 // VIEW_*, view_num_args

namespace ttprm {

constexpr uint32_t FACE_ROW_BYTES = 32u;
constexpr uint32_t RIGHT_FACE     = 512u;

inline uint32_t f0(uint32_t r) { return (r >> 4) * 1024u + (r & 15u) * 32u; }

// Floored div/mod, matching the host implementation, keep scol in [0, Cs).
inline int32_t fdiv(int32_t a, int32_t b) { int32_t q = a / b, r = a % b; if (r != 0 && ((r < 0) != (b < 0))) q--; return q; }
inline int32_t fmod_(int32_t a, int32_t b) { int32_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r; }

// -----------------------------------------------------------------------------
// DenseTensorView: the tensor is laid out exactly as the dense tile grid. A tile is one
// page; gather/scatter is a single whole-page transfer; advance walks page ids.
template <typename Acc>
struct DenseTensorView {
    const Acc& acc;
    uint32_t page;  // current page id

    DenseTensorView(const Acc& a, uint32_t page0) : acc(a), page(page0) {}

    void gather_tile(uint32_t wp) const {
        noc_async_read(acc.get_noc_addr(page, 0), wp, acc.get_aligned_page_size());
    }
    void scatter_tile(uint32_t rp) const {
        noc_async_write(rp, acc.get_noc_addr(page, 0), acc.get_aligned_page_size());
    }
    void advance() { page += 1; }
    // Re-seed the cursor to output-tile ordinal t0 (the group schedule uses this to rewind a
    // broadcast input to its group start). For the dense tiling, page id IS the tile ordinal.
    void reset(int32_t t0) { page = (uint32_t)t0; }
};

// -----------------------------------------------------------------------------
// AffineMap is the runtime descriptor for a viewed access. Accepted views use
// k = base + oR*S_row + oC, with lane stride fixed at 1. The device derives
// odometer increments from this descriptor once in AffineTensorView's ctor.
struct AffineMap {
    int32_t base, S_row;  // the affine map: k(oR,oC) = base + oR*S_row + oC
    int32_t Cs;           // source width in elements (splits k -> page/face coords)
    int32_t nct;          // source col-tiles (source page stride per tile-row)
    int32_t Ro, Co;       // live logical extent of the output (padding guard)
    int32_t n_otc;        // output tile-cols (when a row of output tiles wraps)
};

// AffineTensorView walks the source with a strength-reduced odometer. It derives
// constant (sr, sc) increments once from the map and start tile; the hot loop is
// add/compare/subtract only.
template <typename Acc>
struct AffineTensorView {
    const Acc& acc;
    uint32_t zero_rp;     // pad-identity L1 source (gather seed); unused by scatter
    int32_t base_, S_row_;  // map values kept so reset() can re-seed the cursor
    int32_t Cs, nct, Ro, Co, n_otc;
    int32_t d_row_sr, d_row_sc;    // +1 logical row within a tile
    int32_t d_col_sr, d_col_sc;    // +1 output tile-col
    int32_t d_trow_sr, d_trow_sc;  // +1 output tile-row (with tile-col wrap)
    int32_t otr, otc;     // current output tile coord
    int32_t sr, sc;       // current source element coord (tile's top-left face-row)

    AffineTensorView(const Acc& a, uint32_t zero_rp_, const AffineMap& m, int32_t t0)
        : acc(a), zero_rp(zero_rp_), base_(m.base), S_row_(m.S_row),
          Cs(m.Cs), nct(m.nct), Ro(m.Ro), Co(m.Co), n_otc(m.n_otc) {
        // Odometer increments are constant over the whole walk.
        d_row_sr  = fdiv(m.S_row, Cs);  d_row_sc  = fmod_(m.S_row, Cs);   // +S_row per row
        d_col_sr  = fdiv(32, Cs);       d_col_sc  = fmod_(32, Cs);        // +32 per tile-col
        const int32_t dk_trow = 32 * m.S_row - (n_otc - 1) * 32;          // tile-row wrap
        d_trow_sr = fdiv(dk_trow, Cs);  d_trow_sc = fmod_(dk_trow, Cs);
        reset(t0);   // seed the start cursor from the first output tile
    }

    // Re-seed the cursor to output-tile ordinal t0.
    void reset(int32_t t0) {
        otr = t0 / n_otc;  otc = t0 % n_otc;
        const int32_t k = base_ + otr * 32 * S_row_ + otc * 32;
        sr = fdiv(k, Cs);  sc = fmod_(k, Cs);
    }

    inline void carry(int32_t& r, int32_t& c, int32_t dr, int32_t dc) const {
        c += dc; if (c >= Cs) { c -= Cs; r += 1; } r += dr;
    }
    inline uint32_t vpage(int32_t r, int32_t c) const { return (uint32_t)((r >> 5) * nct + (c >> 5)); }
    inline uint32_t voff(int32_t r, int32_t c) const {
        return f0((uint32_t)r & 31u) + (((uint32_t)c & 31u) >> 4) * RIGHT_FACE;
    }

    // Calls fn(l1_byte_off, viewed_page, viewed_off) for each live face-row.
    template <typename F>
    inline void each_face(F&& fn) const {
        int32_t r = sr, c = sc;
        for (uint32_t lr = 0; lr < 32; lr++) {
            if ((int32_t)(otr * 32 + (int32_t)lr) < Ro) {
                if (otc * 32 < Co) fn(f0(lr), vpage(r, c), voff(r, c));
                if (otc * 32 + 16 < Co) {
                    int32_t r2 = r, c2 = c; carry(r2, c2, 0, 16);
                    fn(f0(lr) + RIGHT_FACE, vpage(r2, c2), voff(r2, c2));
                }
            }
            carry(r, c, d_row_sr, d_row_sc);
        }
    }

    // Assemble the current tile into L1. Issues reads; caller owns barriers.
    void gather_tile(uint32_t wp) const {
        tt::data_movement::common::tt_memmove<true, false, true, 0>(wp, zero_rp, acc.get_aligned_page_size());
        each_face([&](uint32_t l1, uint32_t pg, uint32_t off) {
            noc_async_read(acc.get_noc_addr(pg, off), wp + l1, FACE_ROW_BYTES);
        });
    }

    // Scatter the current L1 tile to viewed slots. Issues writes; caller owns barriers.
    void scatter_tile(uint32_t rp) const {
        each_face([&](uint32_t l1, uint32_t pg, uint32_t off) {
            noc_async_write(rp + l1, acc.get_noc_addr(pg, off), FACE_ROW_BYTES);
        });
    }

    void advance() {
        otc += 1;
        if (otc == n_otc) { otc = 0; otr += 1; carry(sr, sc, d_trow_sr, d_trow_sc); }
        else carry(sr, sc, d_col_sr, d_col_sc);
    }
};

// -----------------------------------------------------------------------------
// GroupedTensorView generalizes the affine row walk to a two-level periodic row map:
//   source_row(oR) = A*(oR / P) + B*(oR % P)
//   source_elt     = base + source_row*S_row + oC
//   cur / x_prev : A=0, B=1  -> source_row = oR mod nt   (token block repeated per gate group)
//   weight       : A=1, B=0  -> source_row = oR div nt   (one weight row stretched over nt tokens)
// The hot loop seeds its (g,t) counter from each tile's top row, avoiding
// cross-tile state and keeping multi-core splits independent.
template <typename Acc>
struct GroupedTensorView {
    const Acc& acc;
    uint32_t zero_rp;                       // pad-identity L1 source (gather seed)
    int32_t base, S_row, Cs, nct, Ro, Co, n_otc;  // source affine row stride + output extent/grid
    int32_t P, A, B;                        // the grouped row map source_row = A*(oR/P) + B*(oR%P)
    int32_t otr, otc;                       // current output tile coord

    GroupedTensorView(const Acc& a, uint32_t zrp, int32_t base_, int32_t S_row_, int32_t Cs_,
                      int32_t nct_, int32_t Ro_, int32_t Co_, int32_t n_otc_, int32_t P_,
                      int32_t A_, int32_t B_, int32_t t0)
        : acc(a), zero_rp(zrp), base(base_), S_row(S_row_), Cs(Cs_), nct(nct_),
          Ro(Ro_), Co(Co_), n_otc(n_otc_), P(P_), A(A_), B(B_) { reset(t0); }

    void reset(int32_t t0) { otr = t0 / n_otc; otc = t0 % n_otc; }
    void advance() { otc += 1; if (otc == n_otc) { otc = 0; otr += 1; } }

    // page/offset of source element base + presented_row*S_row + presented_col in the source tensor.
    inline int32_t sk(int32_t sr, int32_t c) const { return base + sr * S_row + c; }
    inline uint32_t spage(int32_t sr, int32_t c) const {
        const int32_t k = sk(sr, c);
        // Page is tile-row major; soff handles the within-tile face-row.
        return (uint32_t)((fdiv(k, Cs) >> 5) * nct + (fmod_(k, Cs) >> 5));
    }
    inline uint32_t soff(int32_t sr, int32_t c) const {
        const int32_t k = sk(sr, c);
        const int32_t sc = fmod_(k, Cs);
        return f0((uint32_t)fdiv(k, Cs) & 31u) + (((uint32_t)sc & 31u) >> 4) * RIGHT_FACE;
    }

    // Whole-tile fast path for tile-aligned token-block repeats.
    inline bool whole_tile_ok(int32_t r0, int32_t scol) const {
        return base == 0 && A == 0 && B == 1 && S_row == Cs && (P & 31) == 0 && ((r0 % P) & 31) == 0 &&
               (scol & 31) == 0 && r0 + 32 <= Ro && otc * 32 + 32 <= Co;
    }

    // Assemble the current tile into L1. Issues reads; caller owns barriers.
    void gather_tile(uint32_t wp) const {
        const int32_t r0 = otr * 32;
        const int32_t scol = otc * 32;
        if (whole_tile_ok(r0, scol)) {
            noc_async_read(acc.get_noc_addr(spage(r0 % P, scol), 0), wp, acc.get_aligned_page_size());
            return;
        }
        tt::data_movement::common::tt_memmove<true, false, true, 0>(wp, zero_rp, acc.get_aligned_page_size());
        int32_t g = r0 / P, t = r0 % P;
        for (uint32_t lr = 0; lr < 32; lr++) {
            if (r0 + (int32_t)lr < Ro) {
                const int32_t sr = A * g + B * t;
                if (otc * 32 < Co)
                    noc_async_read(acc.get_noc_addr(spage(sr, scol), soff(sr, scol)),
                                   wp + f0(lr), FACE_ROW_BYTES);
                if (otc * 32 + 16 < Co)
                    noc_async_read(acc.get_noc_addr(spage(sr, scol + 16), soff(sr, scol + 16)),
                                   wp + f0(lr) + RIGHT_FACE, FACE_ROW_BYTES);
            }
            if (++t == P) { t = 0; ++g; }
        }
    }

    // Scatter the current L1 tile through the grouped row map. Issues writes;
    // caller owns barriers.
    void scatter_tile(uint32_t rp) const {
        const int32_t r0 = otr * 32;
        const int32_t scol = otc * 32;
        int32_t g = r0 / P, t = r0 % P;
        for (uint32_t lr = 0; lr < 32; lr++) {
            if (r0 + (int32_t)lr < Ro) {
                const int32_t dr = A * g + B * t;
                if (otc * 32 < Co)
                    noc_async_write(rp + f0(lr), acc.get_noc_addr(spage(dr, scol), soff(dr, scol)),
                                    FACE_ROW_BYTES);
                if (otc * 32 + 16 < Co)
                    noc_async_write(rp + f0(lr) + RIGHT_FACE,
                                    acc.get_noc_addr(spage(dr, scol + 16), soff(dr, scol + 16)),
                                    FACE_ROW_BYTES);
            }
            if (++t == P) { t = 0; ++g; }
        }
    }
};

// -----------------------------------------------------------------------------
// TensorViewArgs<ViewT, START> unpacks one runtime view block and constructs the
// selected device view. `next()` supports tightly chained view blocks.
template <template <typename> class ViewT, uint32_t START>
struct TensorViewArgs;

template <uint32_t START>
struct TensorViewArgs<DenseTensorView, START> {
    static constexpr bool gather_needs_pad_seed = false;
    static constexpr uint32_t num_args() { return view_num_args(VIEW_DENSE); }
    static constexpr uint32_t next() { return START + num_args(); }

    template <typename Acc>
    static DenseTensorView<Acc> make(const Acc& acc, uint32_t /*zero_rp*/ = 0) {
        return DenseTensorView<Acc>{acc, get_arg_val<uint32_t>(START)};
    }
};

template <uint32_t START>
struct TensorViewArgs<AffineTensorView, START> {
    static constexpr bool gather_needs_pad_seed = true;
    static constexpr uint32_t num_args() { return view_num_args(VIEW_AFFINE); }
    static constexpr uint32_t next() { return START + num_args(); }

    template <typename Acc>
    static AffineTensorView<Acc> make(const Acc& acc, uint32_t zero_rp = 0) {
        const AffineMap m{
            (int32_t)get_arg_val<uint32_t>(START + 1),  // base
            (int32_t)get_arg_val<uint32_t>(START + 2),  // S_row
            (int32_t)get_arg_val<uint32_t>(START + 3),  // Cs
            (int32_t)get_arg_val<uint32_t>(START + 4),  // nct
            (int32_t)get_arg_val<uint32_t>(START + 5),  // Ro
            (int32_t)get_arg_val<uint32_t>(START + 6),  // Co
            (int32_t)get_arg_val<uint32_t>(START + 7)}; // n_otc
        return AffineTensorView<Acc>{acc, zero_rp, m, (int32_t)get_arg_val<uint32_t>(START)};  // arg[START] = t0
    }
};

template <uint32_t START>
struct TensorViewArgs<GroupedTensorView, START> {
    static constexpr bool gather_needs_pad_seed = true;
    static constexpr uint32_t num_args() { return view_num_args(VIEW_GROUPED); }
    static constexpr uint32_t next() { return START + num_args(); }

    // Args (mirror append_view_args for GROUPED): [t0, base, S_row, Cs, nct, Ro, Co, n_otc, P, A, B].
    template <typename Acc>
    static GroupedTensorView<Acc> make(const Acc& acc, uint32_t zero_rp = 0) {
        return GroupedTensorView<Acc>{acc, zero_rp,
            (int32_t)get_arg_val<uint32_t>(START + 1),   // base
            (int32_t)get_arg_val<uint32_t>(START + 2),   // S_row
            (int32_t)get_arg_val<uint32_t>(START + 3),   // Cs
            (int32_t)get_arg_val<uint32_t>(START + 4),   // nct
            (int32_t)get_arg_val<uint32_t>(START + 5),   // Ro
            (int32_t)get_arg_val<uint32_t>(START + 6),   // Co
            (int32_t)get_arg_val<uint32_t>(START + 7),   // n_otc
            (int32_t)get_arg_val<uint32_t>(START + 8),   // P
            (int32_t)get_arg_val<uint32_t>(START + 9),   // A
            (int32_t)get_arg_val<uint32_t>(START + 10),  // B
            (int32_t)get_arg_val<uint32_t>(START)};      // t0
    }
};

}  // namespace ttprm
