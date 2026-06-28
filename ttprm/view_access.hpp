// Host-side layout algebra and face-granular access planner.
// A `Layout` (shape, stride, base) is an affine map
//     k(out_coord) = base + sum_a out_coord[a] * stride[a]
// from an OUTPUT logical coord to a SOURCE row-major element index. Builder verbs
// (slice/reshape/permute/fold/broadcast) fold a view-chain into one Layout.
//
// The SOURCE is a tiled [Rs, Cs] tensor (Cs % 32 == 0). A tt tile = 32x32 = four
// 16x16 faces (TL,TR,BL,BR) at byte 0/512/1024/1536 (bf16). The atomic NoC transfer
// is a 32B face-row = 16 contiguous bf16. So every transfer is one face-row, and a
// view is realizable iff each output face-row maps to 16 contiguous, FACE-ALIGNED
// (16-elt granularity) source elements:
//     stride[last] == 1                       (contiguous lanes)
//     base % 16 == 0 && every non-lane stride % 16 == 0   (no sub-face straddle)
// Violations are rejected. When it holds, plan() classifies the view as TILE,
// ROW, or SEGMENT and records the affine map consumed by the device kernels.
// This header is host-only and includes an executable oracle for address tests.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace ttprm {

constexpr int FACE = 16;   // face width (elements) == the alignment granularity
constexpr int TILE = 32;   // tile width (elements)
constexpr int FACE_ROW_ELTS = 16;          // 16 bf16 == 32 bytes == one NoC atom
constexpr int PAGE_ELTS = TILE * TILE;     // 1024 elements per tile-page

// Physical tiled addressing in elements. f0e mirrors the device f0() byte offset.
inline int64_t f0e(int64_t r) { return (r / FACE) * (FACE * TILE) + (r % FACE) * FACE; }

// Element offset, within its tile-page, of logical within-tile coord (r in [0,32),
// c in [0,32)). face_col = c/16 picks left/right (=+256 elts == +512 bytes).
inline int64_t in_page_elt(int64_t r, int64_t c) {
    return f0e(r) + (c / FACE) * (FACE * FACE) + (c % FACE);
}

// Full physical element index of logical (row,col) in a [.,C] tiled tensor.
inline int64_t phys(int64_t row, int64_t col, int64_t C) {
    int64_t page = (row / TILE) * (C / TILE) + (col / TILE);
    return page * PAGE_ELTS + in_page_elt(row % TILE, col % TILE);
}

// out coord -> source element index map.
struct Layout {
    std::vector<int64_t> shape;
    std::vector<int64_t> stride;
    int64_t base = 0;
};

enum class AccessClass : uint8_t { TILE, ROW, SEGMENT, REJECT };
inline const char* to_string(AccessClass c) {
    switch (c) {
        case AccessClass::TILE: return "TILE";
        case AccessClass::ROW: return "ROW";
        case AccessClass::SEGMENT: return "SEGMENT";
        default: return "REJECT";
    }
}

// Lazy view builder: folds a chain into one Layout.
class LayoutView {
public:
    // Anchor on a source [Rs, Cs] row-major tiled tensor (logical == padded; Cs % 32 == 0).
    LayoutView(int64_t Rs, int64_t Cs) : src_rows_(Rs), src_cols_(Cs) {
        L_.shape = {Rs, Cs};
        L_.stride = {Cs, 1};
        L_.base = 0;
    }

    // Anchor on a source whose LOGICAL shape is [lr, lc] but whose PHYSICAL tiling pads it
    // up to [pr, pc] (pr, pc % 32 == 0). The view ITERATES the logical live extent -- so the
    // builder verbs (reshape numel/contiguity, slice, fold) reason in LOGICAL terms -- while
    // the row stride and page addressing (src_rows_/src_cols_, phys()) use the PADDED grid.
    // This is the anchor a real tensor takes: a [16,32] tensor pads to a [32,32] tile, but a
    // reshape({1,512}) of it must see 512 logical elements, not the 1024 padded slots. When
    // logical == padded (tile-aligned), this is identical to the 2-arg constructor.
    static LayoutView source(int64_t lr, int64_t lc, int64_t pr, int64_t pc) {
        LayoutView v(pr, pc);          // src_rows_/src_cols_ = padded; row stride = pc
        v.L_.shape = {lr, lc};         // but ITERATE the logical live extent
        return v;
    }

    static LayoutView source_shape(const std::vector<int64_t>& logical, const std::vector<int64_t>& padded) {
        int64_t pr = 1;
        for (size_t i = 0; i + 1 < padded.size(); i++) pr *= padded[i];
        LayoutView v(pr, padded.back());
        v.L_.shape = logical;
        v.L_.stride = rowmajor(padded);
        return v;
    }

    const Layout& layout() const { return L_; }
    int64_t src_rows() const { return src_rows_; }
    int64_t src_cols() const { return src_cols_; }

    // reshape: row-major re-stride of a C-contiguous view. Non-contiguous -> POISON
    // strides, which plan() rejects.
    LayoutView reshape(const std::vector<int64_t>& ns) const {
        LayoutView v = *this;
        if (numel(ns) != numel(L_.shape) || !c_contiguous(L_)) {
            v.L_.shape = ns;
            v.L_.stride.assign(ns.size(), POISON);
            return v;
        }
        v.L_.shape = ns;
        v.L_.stride = rowmajor(ns);
        return v;
    }

    struct Range { int64_t begin, end, step = 1; };
    LayoutView slice(const std::vector<Range>& r) const {
        LayoutView v = *this;
        for (size_t a = 0; a < r.size() && a < L_.shape.size(); a++) {
            int64_t b = r[a].begin, e = r[a].end, s = r[a].step;
            // Invalid ranges poison the layout so plan() rejects cleanly.
            if (s <= 0 || b < 0 || e > L_.shape[a] || b > e) {
                v.L_.stride.assign(L_.shape.size(), POISON);
                return v;
            }
            v.L_.base += b * L_.stride[a];
            v.L_.shape[a] = (e - b + s - 1) / s;
            v.L_.stride[a] = L_.stride[a] * s;
        }
        return v;
    }

    LayoutView permute(const std::vector<int>& perm) const {
        LayoutView v = *this;
        for (size_t i = 0; i < perm.size(); i++) {
            v.L_.shape[i] = L_.shape[perm[i]];
            v.L_.stride[i] = L_.stride[perm[i]];
        }
        return v;
    }

    LayoutView fold(int axis) const {
        LayoutView v(src_rows_, src_cols_);
        // fold splits an axis into (axis/32, 32); non-multiple extents reject.
        if (axis < 0 || axis >= (int)L_.shape.size() || L_.shape[axis] % TILE != 0) {
            v.L_.shape = L_.shape;
            v.L_.stride.assign(L_.shape.size(), POISON);
            v.L_.base = L_.base;
            return v;
        }
        v.L_.shape.clear(); v.L_.stride.clear(); v.L_.base = L_.base;
        for (int a = 0; a < (int)L_.shape.size(); a++) {
            if (a == axis) {
                v.L_.shape.push_back(L_.shape[a] / TILE);
                v.L_.stride.push_back(L_.stride[a] * TILE);
                v.L_.shape.push_back(TILE);
                v.L_.stride.push_back(L_.stride[a]);
            } else {
                v.L_.shape.push_back(L_.shape[a]);
                v.L_.stride.push_back(L_.stride[a]);
            }
        }
        return v;
    }

    // Planner output is the view's presented shape folded to 2D:
    // [Ro, Co] = (prod(shape[:-1]), shape[-1]).
    struct Plan {
        AccessClass cls = AccessClass::REJECT;
        std::string why;
        int64_t Ro = 0, Co = 0;          // output LOGICAL (live) shape, folded 2D
        int64_t Ro_pad = 0, Co_pad = 0;  // padded up to tile-multiple = the physical tiling
        // Affine 2D map walked by the on-device generator:
        //     k(oR, oC) = base + oR*S_row + oC      (inner/lane stride is 1)
        // S_row is the (constant) source-element stride per output row -- the linearity
        // The linearity gate rejects any non-linear row_base.
        int64_t base = 0, S_row = 0;
        bool ok() const { return cls != AccessClass::REJECT; }
        int64_t out_tiles() const { return (Ro_pad / TILE) * (Co_pad / TILE); }
        bool has_pad() const { return Ro_pad != Ro || Co_pad != Co; }
    };

    Plan plan() const {
        Plan p;
        const int n = (int)L_.shape.size();
        if (n < 1) { p.why = "empty shape"; return p; }
        for (int64_t s : L_.stride) if (s == POISON) { p.why = "non-contiguous reshape (non-affine)"; return p; }

        int64_t Co = L_.shape[n - 1];
        int64_t Ro = 1; for (int a = 0; a < n - 1; a++) Ro *= L_.shape[a];
        p.Ro = Ro; p.Co = Co;
        p.Ro_pad = ((Ro + TILE - 1) / TILE) * TILE;
        p.Co_pad = ((Co + TILE - 1) / TILE) * TILE;

        if (L_.stride[n - 1] != 1) { p.why = "inner (lane) stride != 1 -> not a face-row"; return p; }
        // Face alignment: base and every non-lane stride must be face-aligned,
        // else an output face-row straddles a source face. The logical width
        // must also be face-aligned, else the live/pad column boundary cuts a face-row.
        if (Co % FACE != 0) { p.why = "logical width not 16-aligned (sub-face pad boundary)"; return p; }
        if (L_.base % FACE != 0) { p.why = "base not 16-aligned (sub-face)"; return p; }
        for (int a = 0; a < n - 1; a++)
            if (L_.stride[a] % FACE != 0) { p.why = "non-lane stride not 16-aligned (sub-face)"; return p; }

        // Affine gate: the device walks k = base + oR*S_row + oC, so row_base
        // must be linear in oR.
        p.base = L_.base;
        p.S_row = (Ro > 1) ? (row_base(1) - row_base(0)) : 0;
        for (int64_t oR = 1; oR < Ro; oR++)
            if (row_base(oR) != L_.base + oR * p.S_row) {
                p.why = "non-affine row map (permute / unsupported) -> REJECT";
                return p;
            }

        // Validate every live face-row maps in range, then classify by compression.
        // Padding face-rows are skipped, so padded tiles cannot use TILE/ROW fast paths.
        const int64_t Cs = src_cols_, total_src_pages = (src_rows_ / TILE) * (Cs / TILE);
        // Padding uses the generic seed+skip path.
        bool all_tile = !p.has_pad(), all_row = !p.has_pad();
        for (int64_t OTR = 0; OTR < p.Ro_pad / TILE; OTR++) {
            for (int64_t OTC = 0; OTC < p.Co_pad / TILE; OTC++) {
                int64_t prev_page = -1, delta = INT64_MIN;   // ROW: constant per-live-row page delta
                for (int r = 0; r < TILE; r++) {
                    int64_t oR = OTR * TILE + r;
                    if (oR >= Ro) { all_tile = false; continue; }   // pad row (no source)
                    int64_t row_k = row_base(oR);
                    int64_t page_l = -1, page_r = -1; bool col_pad = false;
                    for (int fc = 0; fc < 2; fc++) {
                        int64_t oC0 = OTC * TILE + fc * FACE;
                        if (oC0 >= Co) { all_tile = false; col_pad = true; continue; }   // pad col
                        int64_t k0 = row_k + oC0;
                        int64_t srow = k0 / Cs, scol = k0 % Cs;
                        if (k0 < 0 || srow >= src_rows_) { p.why = "face-row maps out of source range"; return p; }
                        int64_t pg = (srow / TILE) * (Cs / TILE) + (scol / TILE);
                        if (pg >= total_src_pages) { p.why = "page out of range"; return p; }
                        (fc == 0 ? page_l : page_r) = pg;
                        if ((srow % TILE) != r || (scol % TILE) / FACE != fc) all_tile = false;
                    }
                    if (!col_pad && page_l != page_r) all_row = false;
                    if (prev_page >= 0) {
                        int64_t d = page_l - prev_page;
                        if (delta == INT64_MIN) delta = d;
                        else if (d != delta) all_row = false;
                    }
                    prev_page = page_l;
                }
            }
        }
        p.cls = all_tile ? AccessClass::TILE : (all_row ? AccessClass::ROW : AccessClass::SEGMENT);
        p.why = p.has_pad() ? "face-aligned; realizable (padded)" : "face-aligned; realizable";
        return p;
    }

    // k for output row oR (column 0): base + sum over outer axes of idx[a]*stride[a].
    // PUBLIC so the host lowering can derive the affine map; the plan()'s linearity gate
    // guarantees this equals base + oR*S_row for every accepted view.
    int64_t row_base(int64_t oR) const {
        const int n = (int)L_.shape.size();
        int64_t k = L_.base, rem = oR;
        for (int a = n - 2; a >= 0; a--) {          // outer axes, last-outer fastest
            int64_t ext = L_.shape[a];
            int64_t idx = rem % ext; rem /= ext;
            k += idx * L_.stride[a];
        }
        return k;
    }

private:
    LayoutView() = default;
    static constexpr int64_t POISON = INT64_MIN;

    static int64_t numel(const std::vector<int64_t>& s) { int64_t n = 1; for (int64_t x : s) n *= x; return n; }
    static std::vector<int64_t> rowmajor(const std::vector<int64_t>& s) {
        std::vector<int64_t> st(s.size(), 1);
        for (int i = (int)s.size() - 2; i >= 0; i--) st[i] = st[i + 1] * s[i + 1];
        return st;
    }
    static bool c_contiguous(const Layout& L) {
        int64_t expect = 1;
        for (int i = (int)L.shape.size() - 1; i >= 0; i--) {
            if (L.shape[i] == 1) continue;
            if (L.stride[i] != expect) return false;
            expect *= L.shape[i];
        }
        return true;
    }

    int64_t src_rows_ = 0, src_cols_ = 0;
    Layout L_;
};

// ===========================================================================
// HOST-EXECUTABLE MODEL.
// A source buffer holds, at each physical element slot, the LOGICAL element id
// k = row*Cs + col. realize_ref writes the output by the affine map directly;
// view_realize.hpp simulates the device odometer for comparison.
// ===========================================================================

// Fill a tiled [R,C] source so phys-slot(row,col) holds id = row*C+col.
inline std::vector<int32_t> make_source(int64_t R, int64_t C) {
    std::vector<int32_t> buf((size_t)((R / TILE) * (C / TILE)) * PAGE_ELTS, -1);
    for (int64_t row = 0; row < R; row++)
        for (int64_t col = 0; col < C; col++)
            buf[(size_t)phys(row, col, C)] = (int32_t)(row * C + col);
    return buf;
}

// Brute-force oracle. Padding slots stay -1.
inline std::vector<int32_t> realize_ref(const LayoutView& v, const LayoutView::Plan& p) {
    const Layout& L = v.layout();
    std::vector<int32_t> out((size_t)p.out_tiles() * PAGE_ELTS, -1);
    const int n = (int)L.shape.size();
    for (int64_t oR = 0; oR < p.Ro; oR++) {
        int64_t k = L.base, rem = oR;
        for (int a = n - 2; a >= 0; a--) { int64_t e = L.shape[a]; k += (rem % e) * L.stride[a]; rem /= e; }
        for (int64_t oC = 0; oC < p.Co; oC++) {
            int64_t kk = k + oC * L.stride[n - 1];
            out[(size_t)phys(oR, oC, p.Co_pad)] = (int32_t)kk;   // padded physical width
        }
    }
    return out;
}

}  // namespace ttprm
