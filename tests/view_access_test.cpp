// view_access_test.cpp -- PURE-HOST derisk of the face-granular layout algebra.
// No device. Proves: (1) the offset+stride -> face-row address map is bit-exact vs a
// brute-force element oracle, for complex chains [slice->reshape], [reshape->reshape],
// folds, and a rank-N permute; (2) the 16-element (face) alignment rule REJECTS exactly
// the sub-face / non-contiguous cases; (3) the verdict classifies TILE/ROW/SEGMENT.
//
// Run: it builds with no TT deps. Prints RESULT: OK / FAIL.

#include "view_access.hpp"
#include "view_realize.hpp"

#include <cstdio>
#include <vector>

using namespace ttprm;

namespace {
int g_pass = 0, g_fail = 0;
void check(const char* name, bool ok, const char* detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name, detail[0] ? " -- " : "", detail);
    if (ok) ++g_pass; else ++g_fail;
}

// Realize the view two ways and assert bit-exact; assert the verdict class.
void expect_realizable(const char* name, const LayoutView& v, AccessClass want) {
    auto p = v.plan();
    if (!p.ok()) { check(name, false, p.why.c_str()); return; }
    auto ref = realize_ref(v, p);              // brute-force affine oracle
    auto got = realize_affine_sim(v, p);       // the EXACT device odometer, host-simulated
    bool exact = (ref == got);
    bool cls_ok = (p.cls == want);
    char d[160];
    std::snprintf(d, sizeof d, "class=%s(want %s) %s", to_string(p.cls), to_string(want),
                  exact ? "bit-exact" : "MISMATCH");
    check(name, exact && cls_ok, d);
}

void expect_reject(const char* name, const LayoutView& v, const char* why_contains) {
    auto p = v.plan();
    bool rejected = (p.cls == AccessClass::REJECT);
    check(name, rejected, rejected ? p.why.c_str() : "expected REJECT but accepted");
}
}  // namespace

int main() {
    std::printf(">>> view_access_test (pure-host layout-algebra derisk)\n");

    // ---- identity: a dense [64,64] source viewed as-is -> verbatim TILE.
    expect_realizable("identity [64,64]", LayoutView(64, 64), AccessClass::TILE);

    // ---- [reshape -> reshape -> store]: flat [1,4096] -> [64,64] -> [128,32].
    // Contiguous row-major all the way; width changes => per-face-row source page
    // walks => SEGMENT/ROW (not a verbatim page copy). Proves the chain composes.
    {
        LayoutView v = LayoutView(32, 4096)                 // a tiled source holding the 4096 lane (32 padded rows)
                     .slice({{0, 1, 1}, {0, 4096, 1}})  // logical [1,4096]
                     .reshape({64, 64})
                     .reshape({128, 32});
        // [128,32]: each output row r's 32 cols are flat[32r,32r+32), one source page,
        // constant per-row page delta -> ROW.
        expect_realizable("reshape->reshape [1,4096]->[64,64]->[128,32]", v, AccessClass::ROW);
    }

    // ---- flat [1,4096] -> [64,64] (the RWKV head reshape): each output row r is
    // flat cols [64r,64r+64), all in source row 0 -> per-row affine page => ROW.
    {
        LayoutView v = LayoutView(32, 4096).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        expect_realizable("reshape [1,4096]->[64,64] (head)", v, AccessClass::ROW);
    }

    // ---- [slice -> reshape -> store]: take a tile-aligned [64,64] sub-block of a
    // [128,128] source, then reshape to [128,32]. Contiguous within the slice? No --
    // a row-slice of width 64 from a width-128 source is NOT C-contiguous, so the
    // reshape poisons -> REJECT because the slice needs a relayout first.
    {
        LayoutView v = LayoutView(128, 128).slice({{0, 64, 1}, {0, 64, 1}}).reshape({128, 32});
        expect_reject("slice(width64 of 128)->reshape  [non-contiguous]", v, "non-contiguous");
    }

    // ---- [slice -> reshape -> store] that IS contiguous: slice full-width tile-aligned
    // rows [32:96] of [128,64], reshape [64,64]->[128,32]. Full-width slice stays
    // contiguous => realizable.
    {
        LayoutView v = LayoutView(128, 64).slice({{32, 96, 1}, {0, 64, 1}}).reshape({128, 32});
        expect_realizable("slice(rows32:96 full-width)->reshape->[128,32]", v, AccessClass::SEGMENT);
    }

    // ---- rank-N permute that REGROUPS tiles: row_base is no longer linear in oR, so the
    // affine gate REJECTs it (permute is out of scope -- the caller materializes via ttnn).
    {
        LayoutView v = LayoutView(128, 128)
                     .reshape({4, 32, 128})      // rows -> (tile_row, in_tile_row, cols)
                     .fold(2)                     // cols -> (tile_col, in_tile_col): [4,32,4,32]
                     .permute({2, 1, 0, 3});      // regroup tile axes
        expect_reject("rank-N tile regroup permute (non-affine)", v, "non-affine");
    }

    // ---- REJECT: a true element transpose (inner stride != 1) needs within-face
    // shuffle -> not a face-row -> REJECT.
    expect_reject("element transpose [64,64]", LayoutView(64, 64).permute({1, 0}), "stride != 1");

    // ---- REJECT: sub-face slice. Slice cols [8:40] of [64,64]: base col 8 not
    // 16-aligned -> output face-rows straddle source faces -> REJECT.
    expect_reject("sub-face col slice [8:40]", LayoutView(64, 64).slice({{0, 64, 1}, {8, 40, 1}}), "16-aligned");

    // ---- face-aligned col slice [16:48]: starts at a face boundary, width 32 -> OK.
    expect_realizable("face-aligned col slice [16:48]",
                      LayoutView(64, 64).slice({{0, 64, 1}, {16, 48, 1}}), AccessClass::SEGMENT);

    // ---- PADDING: logical [1,4096] -> physical [32,4096] (31 pad rows). Live row + pad
    // both reproduced (pad slots stay the -1 marker == device zero seed).
    expect_realizable("padded [1,4096] (1 live row of 32)",
                      LayoutView(32, 4096).slice({{0, 1, 1}, {0, 4096, 1}}), AccessClass::SEGMENT);

    // ---- PADDING (col): logical [64,48] -> physical [64,64]; 48%16==0, 48%32!=0.
    expect_realizable("padded [64,48] (col pad to 64)",
                      LayoutView(64, 64).slice({{0, 64, 1}, {0, 48, 1}}), AccessClass::SEGMENT);

    // ================================================================================
    // REGRESSION BATTERY (added after the logical-anchor fix + the audit). Each guards a
    // specific bug class found in source: builder-verb input validation, the logical-vs-
    // padded unfold boundary, and the documented REJECT contracts (sub-16 width, permute).
    // ================================================================================

    // ---- UNFOLD family: a folded [R,32] source (logical anchored on padded grid) reshaped
    // back to flat [1,R*32]. Pre-fix this REJECTed whenever R was not a tile-multiple because
    // the reshape numel check saw the PADDED slot count. R in {16,24,48} all pad to non-equal
    // grids; all must realize. (R=128 worked by tile-aligned luck even before the fix.)
    expect_realizable("unfold folded[16,32]->[1,512]  (R pads 16->32)",
                      LayoutView::source(16, 32, 32, 32).reshape({1, 512}), AccessClass::SEGMENT);
    expect_realizable("unfold folded[24,32]->[1,768]  (R pads 24->32)",
                      LayoutView::source(24, 32, 32, 32).reshape({1, 768}), AccessClass::SEGMENT);
    expect_realizable("unfold folded[48,32]->[1,1536] (R pads 48->64)",
                      LayoutView::source(48, 32, 64, 32).reshape({1, 1536}), AccessClass::SEGMENT);

    // ---- fold() VALIDATION (bug: shape[axis]/32 silently DROPPED the tail). A non-multiple-
    // of-32 fold axis, and an out-of-range axis, must REJECT instead of truncating.
    expect_reject("fold(0) on [48,32] (48 not mult of 32)", LayoutView::source(48, 32, 64, 32).fold(0), "");
    expect_reject("fold(axis=5) out of range",              LayoutView(64, 64).fold(5), "");

    // ---- slice() VALIDATION (bug: no range checks -> step=0 div-by-zero, negative/OOB ranges).
    expect_reject("slice step=0",        LayoutView(64, 64).slice({{0, 64, 0}, {0, 64, 1}}), "");
    expect_reject("slice begin<0",       LayoutView(64, 64).slice({{-1, 64, 1}, {0, 64, 1}}), "");
    expect_reject("slice end>extent",    LayoutView(64, 64).slice({{0, 64, 1}, {0, 96, 1}}), "");
    expect_reject("slice begin>end",     LayoutView(64, 64).slice({{48, 16, 1}, {0, 64, 1}}), "");

    // ---- sub-16 logical WIDTH is out of scope (face-row contract): even a plain identity and
    // a reshape to a last-dim < 16 must REJECT (decided: keep REJECT + ttnn fallback).
    expect_reject("identity logical width 100 (not 16-aligned)", LayoutView::source(1, 100, 32, 128), "");
    expect_reject("reshape [64,64]->[64,8,8] (last dim 8 < 16)", LayoutView(64, 64).reshape({64, 8, 8}), "");

    std::printf("  %d passed, %d failed\n", g_pass, g_fail);
    const bool ok = (g_fail == 0);
    std::printf("RESULT: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
