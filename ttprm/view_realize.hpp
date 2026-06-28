// Host lowering from an accepted LayoutView to device affine runtime args.
// view_access.hpp proves every accepted view is a single affine map
//     k(oR, oC) = base + oR*S_row + oC          (inner/lane stride 1)
// and that the source tiling is periodic (mod 32 rows/cols, mod 16 faces). So the
// device can walk the source coordinate instead of using a per-tile copy list.
// The host computes the constant increments; the device hot loop is add/sub/compare.
//
//   k stride per level        host-precomputed (dsrow, dscol) = (k/Cs, k%Cs, floored)
//   ---------------------      --------------------------------------------------------
//   within-tile row (+S_row)   d_row
//   tile-col      (+32)        d_col
//   tile-row carry             d_trow   (last tile of a row -> first tile of the next)
//
// AffineDesc carries the map, output extent, and per-core seed math. The host
// simulator mirrors the device odometer and is tested against realize_ref.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "view_access.hpp"

namespace ttprm {

// Floored div/mod keep dscol in [0, Cs), including negative element strides.
inline int64_t fdiv(int64_t a, int64_t b) {
    int64_t q = a / b, r = a % b; if (r != 0 && ((r < 0) != (b < 0))) q--; return q;
}
inline int64_t fmod_(int64_t a, int64_t b) {
    int64_t r = a % b; if (r != 0 && ((r < 0) != (b < 0))) r += b; return r;
}

// Parameters consumed by the on-device affine generator.
struct AffineDesc {
    bool        ok = false;
    std::string why;
    AccessClass analysis = AccessClass::REJECT;   // TILE/ROW/SEGMENT (diagnostics only)

    int64_t Cs = 0, nct = 0;                       // source cols, source col-tiles (Cs/32)
    int64_t Ro = 0, Co = 0;                         // live output (for padding compares)
    int64_t Ro_pad = 0, Co_pad = 0, n_otc = 0, n_otr = 0;  // padded tile grid
    int64_t base = 0, S_row = 0;
    // GROUPED (prefill token-shift) row map: source_row(oR) = grp_A*(oR/grp_P) + grp_B*(oR%grp_P).
    // grp_P == 0 means a plain (linear) map; nonzero routes the endpoint to the GroupedTensorView.
    int64_t grp_P = 0, grp_A = 0, grp_B = 0;
    int64_t d_row_sr = 0, d_row_sc = 0;             // within-tile row step
    int64_t d_col_sr = 0, d_col_sc = 0;             // tile-col step (+32)
    int64_t d_trow_sr = 0, d_trow_sc = 0;           // tile-row carry

    int64_t out_tiles() const { return (Ro_pad / TILE) * (Co_pad / TILE); }

    // Source coord of flat output tile T's origin.
    void seed(int64_t T, int64_t& OTR, int64_t& OTC, int64_t& sr, int64_t& sc) const {
        OTR = T / n_otc; OTC = T % n_otc;
        int64_t k = base + OTR * TILE * S_row + OTC * TILE;
        sr = fdiv(k, Cs); sc = fmod_(k, Cs);
    }
};

// Lower an accepted view plan into device runtime args.
inline AffineDesc lower(const LayoutView& v, const LayoutView::Plan& p) {
    AffineDesc d;
    if (!p.ok()) { d.why = p.why; return d; }
    d.ok = true; d.analysis = p.cls;
    d.Cs = v.src_cols(); d.nct = d.Cs / TILE;
    d.Ro = p.Ro; d.Co = p.Co; d.Ro_pad = p.Ro_pad; d.Co_pad = p.Co_pad;
    d.n_otc = p.Co_pad / TILE; d.n_otr = p.Ro_pad / TILE;
    d.base = p.base; d.S_row = p.S_row;
    auto split = [&](int64_t dk, int64_t& dsr, int64_t& dsc) { dsr = fdiv(dk, d.Cs); dsc = fmod_(dk, d.Cs); };
    split(d.S_row, d.d_row_sr, d.d_row_sc);                            // +S_row per output row
    split(TILE, d.d_col_sr, d.d_col_sc);                              // +32 per tile-col
    split(TILE * d.S_row - (d.n_otc - 1) * TILE, d.d_trow_sr, d.d_trow_sc);  // row wrap
    return d;
}

inline AffineDesc lower(const LayoutView& v) { return lower(v, v.plan()); }

// Host simulator for the device odometer. Produces the source element id for
// each output physical slot.
inline std::vector<int32_t> realize_affine_sim(const LayoutView& v, const LayoutView::Plan& p) {
    AffineDesc d = lower(v, p);
    std::vector<int32_t> out((size_t)d.out_tiles() * PAGE_ELTS, -1);
    auto carry = [&](int64_t& sr, int64_t& sc, int64_t dsr, int64_t dsc) {
        sc += dsc; if (sc >= d.Cs) { sc -= d.Cs; sr++; } sr += dsr;
    };
    for (int64_t T = 0; T < d.out_tiles(); T++) {
        int64_t OTR, OTC, sr_t, sc_t; d.seed(T, OTR, OTC, sr_t, sc_t);
        int64_t sr = sr_t, sc = sc_t;
        for (int r = 0; r < TILE; r++) {
            int64_t oR = OTR * TILE + r;
            if (oR < d.Ro) {
                for (int fc = 0; fc < 2; fc++) {
                    int64_t oC0 = OTC * TILE + fc * FACE;
                    if (oC0 >= d.Co) continue;
                    int64_t s_r = sr, s_c = sc;
                    if (fc == 1) carry(s_r, s_c, 0, FACE);
                    int64_t dst_elt = in_page_elt(r, fc * FACE);
                    for (int e = 0; e < FACE_ROW_ELTS; e++)
                        out[(size_t)(T * PAGE_ELTS + dst_elt + e)] = (int32_t)(s_r * d.Cs + s_c + e);
                }
            }
            carry(sr, sc, d.d_row_sr, d.d_row_sc);
        }
    }
    return out;
}

}  // namespace ttprm
