// grouped_gather_test.cpp -- PURE HOST derisk (no device, no TT link) for the PREFILL token-shift
// gather: the 2-level "grouped" row odometer that lets cur/x_prev (source row = r mod nt) and
// weight (source row = r div nt) feed a TIGHT [G*nt, n_embd] output when 1 < nt < 32.
// -----------------------------------------------------------------------------
// The linear affine map k = base + oR*S_row + oC cannot express a periodic/stretched row index.
// The grouped map generalizes the ROW part only (columns stay linear):
//
//     source_row(oR) = A*(oR div P) + B*(oR mod P)          P = nt (the period / group size)
//     source_elt      = base + source_row*S_row + oC
//
//   cur / x_prev : (A=0, B=1)  -> source_row = oR mod nt   (group-major token, repeated per group)
//   weight       : (A=1, B=0)  -> source_row = oR div nt   (one weight row stretched over nt tokens)
//   full operand : (A=P, B=1)  -> source_row = oR          (the plain linear case, sanity)
//
// This test proves the DEVICE-MIRROR odometer (a per-tile (g,t) counter: t++, wrap at P, g++ -- the
// exact loop the kernel each_face will run, seeded PER TILE so it is multi-core safe) reproduces the
// brute-force semantic intent bit-exact, BEFORE any kernel goes on the board. It is the host oracle
// the device GroupedTensorView must match (cf. realize_affine_sim vs AffineTensorView).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "view_access.hpp"   // TILE, FACE, PAGE_ELTS, FACE_ROW_ELTS, in_page_elt

namespace {
using namespace ttprm;

int g_pass = 0, g_fail = 0;
void check(const std::string& name, bool ok) {
    std::printf("  [%s] %s\n", ok ? "PASS" : "FAIL", name.c_str());
    if (ok) ++g_pass; else ++g_fail;
}

// The grouped-gather descriptor: the source geometry + output grid + the (P,A,B) row map.
struct GroupedDesc {
    int64_t base = 0, S_row = 0, Cs = 0, nct = 0;  // source affine rows + width/col-tiles
    int64_t Ro = 0, Co = 0;              // output LIVE extent (the pad guard)
    int64_t n_otr = 0, n_otc = 0;        // output tile grid
    int64_t P = 1, A = 0, B = 1;         // row map: source_row = A*(oR/P) + B*(oR%P)
    int64_t out_tiles() const { return n_otr * n_otc; }
};

int64_t source_elt(const GroupedDesc& d, int64_t src_row, int64_t col) {
    return d.base + src_row * d.S_row + col;
}

// BRUTE-FORCE reference: the semantic intent, computed straight from the definition. Per output
// physical slot (tile, in-page element) -> the source element id it must hold (-1 = pad/untouched).
std::vector<int32_t> grouped_ref(const GroupedDesc& d) {
    std::vector<int32_t> out((size_t)d.out_tiles() * PAGE_ELTS, -1);
    for (int64_t T = 0; T < d.out_tiles(); T++) {
        const int64_t OTR = T / d.n_otc, OTC = T % d.n_otc;
        for (int lr = 0; lr < TILE; lr++) {
            const int64_t oR = OTR * TILE + lr;
            if (oR >= d.Ro) continue;
            const int64_t src_row = d.A * (oR / d.P) + d.B * (oR % d.P);
            for (int fc = 0; fc < 2; fc++) {
                const int64_t oC0 = OTC * TILE + fc * FACE;
                if (oC0 >= d.Co) continue;
                for (int e = 0; e < FACE_ROW_ELTS; e++)
                    out[(size_t)(T * PAGE_ELTS + in_page_elt(lr, fc * FACE) + e)] =
                        (int32_t)source_elt(d, src_row, oC0 + e);
            }
        }
    }
    return out;
}

// DEVICE-MIRROR odometer: identical to the kernel each_face we will write -- a per-tile (g,t)
// counter (NO cross-tile state, so any start tile / core is correct), t++ wrapping at P -> g++.
std::vector<int32_t> grouped_sim(const GroupedDesc& d) {
    std::vector<int32_t> out((size_t)d.out_tiles() * PAGE_ELTS, -1);
    for (int64_t T = 0; T < d.out_tiles(); T++) {
        const int64_t OTR = T / d.n_otc, OTC = T % d.n_otc;
        const int64_t r0 = OTR * TILE;
        int64_t g = r0 / d.P, t = r0 % d.P;          // SEED the counter from this tile's top row
        for (int lr = 0; lr < TILE; lr++) {
            const int64_t oR = r0 + lr;
            const int64_t src_row = d.A * g + d.B * t;
            if (oR < d.Ro) {
                for (int fc = 0; fc < 2; fc++) {
                    const int64_t oC0 = OTC * TILE + fc * FACE;
                    if (oC0 < d.Co)
                        for (int e = 0; e < FACE_ROW_ELTS; e++)
                            out[(size_t)(T * PAGE_ELTS + in_page_elt(lr, fc * FACE) + e)] =
                                (int32_t)source_elt(d, src_row, oC0 + e);
                }
            }
            if (++t == d.P) { t = 0; ++g; }          // advance group counter (the device loop)
        }
    }
    return out;
}

bool same(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) if (a[i] != b[i]) return false;
    return true;
}

// Build the descriptor for the RWKV token-shift inputs at (nt, G), n_embd = E.
//   pattern 0 = cur/x_prev (mod), 1 = weight (div), 2 = full (identity, sanity).
GroupedDesc make(int64_t E, int64_t nt, int64_t G, int pattern) {
    GroupedDesc d;
    d.S_row = E; d.Cs = E; d.nct = E / TILE;
    d.Co = E; d.n_otc = E / TILE;
    d.Ro = G * nt;
    d.n_otr = (d.Ro + TILE - 1) / TILE;
    d.P = nt;
    if (pattern == 0)      { d.A = 0;  d.B = 1; }   // cur:    r mod nt
    else if (pattern == 1) { d.A = 1;  d.B = 0; }   // weight: r div nt
    else                   { d.A = nt; d.B = 1; }   // full:   r
    return d;
}

// Independent semantic check: cur[oR mod nt][oC], weight[oR div nt][oC], spelled out a different way
// than grouped_ref, so a shared bug in ref/sim can't hide. Confirms the (A,B) encoding == intent.
std::vector<int32_t> intent(int64_t E, int64_t nt, int64_t G, int pattern) {
    GroupedDesc d = make(E, nt, G, pattern);
    std::vector<int32_t> out((size_t)d.out_tiles() * PAGE_ELTS, -1);
    for (int64_t oR = 0; oR < d.Ro; oR++) {
        int64_t src_row;
        if (pattern == 0)      src_row = oR % nt;
        else if (pattern == 1) src_row = oR / nt;
        else                   src_row = oR;
        for (int64_t oC = 0; oC < d.Co; oC++)
            out[(size_t)phys(oR, oC, d.Co)] = (int32_t)(src_row * E + oC);
    }
    return out;
}
}  // namespace

int main() {
    std::printf(">>> grouped_gather_test (PREFILL token-shift grouped-row odometer, pure host)\n");

    struct Cfg { int64_t E, nt, G; };
    const std::vector<Cfg> cfgs = {
        {64, 1, 6}, {64, 5, 5}, {64, 8, 6}, {64, 8, 5},
        {128, 8, 6}, {512, 8, 6}, {64, 32, 3}, {64, 7, 10}, {256, 5, 7},
    };
    const char* pname[3] = {"cur(mod)", "weight(div)", "full(id)"};

    for (const auto& c : cfgs) {
        for (int p = 0; p < 3; p++) {
            char nm[96];
            std::snprintf(nm, sizeof nm, "E=%lld nt=%lld G=%lld  %s : sim==ref",
                          (long long)c.E, (long long)c.nt, (long long)c.G, pname[p]);
            const GroupedDesc d = make(c.E, c.nt, c.G, p);
            const bool ok = same(grouped_sim(d), grouped_ref(d));
            check(nm, ok);
            // and ref == independent intent (cur[r%nt] / weight[r/nt]) -- guards a shared ref/sim bug
            std::snprintf(nm, sizeof nm, "E=%lld nt=%lld G=%lld  %s : ref==intent",
                          (long long)c.E, (long long)c.nt, (long long)c.G, pname[p]);
            check(nm, same(grouped_ref(d), intent(c.E, c.nt, c.G, p)));
        }
    }

    // kk prefill broadcast shape: a flat [1, heads*S] weight is presented as [heads,S] and
    // repeated over token-major [nt*heads,S] rows. This is the affine-row grouped case used by
    // mul_l2_norm for k * time_mix_k_k.
    {
        const int64_t nt = 33, heads = 2, S = 64;
        GroupedDesc d;
        d.base = 0; d.S_row = S; d.Cs = heads * S; d.nct = d.Cs / TILE;
        d.Co = S; d.n_otc = S / TILE;
        d.Ro = nt * heads; d.n_otr = (d.Ro + TILE - 1) / TILE;
        d.P = heads; d.A = 0; d.B = 1;  // source row = output row % heads

        std::vector<int32_t> want((size_t)d.out_tiles() * PAGE_ELTS, -1);
        for (int64_t oR = 0; oR < d.Ro; oR++)
            for (int64_t oC = 0; oC < d.Co; oC++)
                want[(size_t)phys(oR, oC, d.Co)] = (int32_t)((oR % heads) * S + oC);

        check("kk row-repeat affine source rows: sim==ref", same(grouped_sim(d), grouped_ref(d)));
        check("kk row-repeat affine source rows: ref==intent", same(grouped_ref(d), want));
    }

    // Multi-core safety: every tile is seeded independently in the loops above, so a per-core tile
    // SUBRANGE produces byte-identical results. Verify by simulating tile-by-tile and splicing.
    {
        const GroupedDesc d = make(512, 8, 6, 0);   // 2 row-tiles x 16 col-tiles = 32 tiles
        const auto whole = grouped_sim(d);
        std::vector<int32_t> spliced((size_t)d.out_tiles() * PAGE_ELTS, -1);  // -1 = pad (as in sim)
        for (int64_t T = 0; T < d.out_tiles(); T++) {     // each "core" does ONE tile, seeded fresh
            const int64_t OTR = T / d.n_otc, OTC = T % d.n_otc, r0 = OTR * TILE;
            int64_t g = r0 / d.P, t = r0 % d.P;
            for (int lr = 0; lr < TILE; lr++) {
                const int64_t oR = r0 + lr; const int64_t sr = d.A * g + d.B * t;
                if (oR < d.Ro)
                    for (int fc = 0; fc < 2; fc++) {
                        const int64_t oC0 = OTC * TILE + fc * FACE;
                        if (oC0 < d.Co)
                            for (int e = 0; e < FACE_ROW_ELTS; e++)
                                spliced[(size_t)(T * PAGE_ELTS + in_page_elt(lr, fc * FACE) + e)] =
                                    (int32_t)source_elt(d, sr, oC0 + e);
                    }
                if (++t == d.P) { t = 0; ++g; }
            }
        }
        check("multi-core: per-tile independent seeding == whole-grid sim", same(whole, spliced));
    }

    std::printf("  %d passed, %d failed\n", g_pass, g_fail);
    const bool ok = (g_fail == 0);
    std::printf("RESULT: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
