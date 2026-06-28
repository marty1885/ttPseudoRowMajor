// view_lerp_rwkv_test.cpp -- Phase-0 derisk for the llama.cpp RWKV token-shift swap.
// -----------------------------------------------------------------------------
// THE RISK we retire here (no backend changes): the RWKV time-mix token-shift fuses the 6 gate
// groups into one lerp:
//     out = x + weight * (x_prev - x)            (ttprm::lerp(a=x, b=x_prev, w=weight))
// with the ggml shapes
//     cur / x_prev = [n_embd, n_tokens]          (the two operands that REPEAT across the 6 groups)
//     weight       = [n_embd, 1, 1, G]           (full per group, broadcast over n_tokens)
//     out (xxx)    = [n_embd, n_tokens, 1, G]
// Mapping ggml -> ttprm's 2D [rows, cols]: cols = n_embd (inner/contiguous), rows = the folded outer
// dims = G * n_tokens, GROUP-MAJOR (ne[3]=G outermost):  row = g * n_tokens + t.
//
// The worry: ttprm's accepted broadcast is TILE-granular (an operand presenting fewer *tiles* that
// divides the output is re-fed across whole-tile groups). But at decode n_tokens=1, each group is a
// SINGLE logical row, so cur/x_prev present 1 row that must broadcast to G(<32) rows INSIDE one
// 32-row tile -- a sub-tile row broadcast, which is NOT the whole-tile/face-granular class. Same at
// n_tokens=8 (groups are 8 rows, group boundaries fall mid-tile; weight must also repeat per-token).
// The existing lerp bcast coverage only uses groups spanning >= 32 rows, so this case is unproven.
//
// This harness builds the EXACT backend shapes for n_tokens in {1,8}, G in {5,6}, bf16, and asks
// ttprm::lerp to produce them, checking ok() + PCC>0.999 vs a CPU reference of the true logical
// broadcast. Two probes per (n_tokens,G):
//   [CONTROL] full operands (host-expanded, no broadcast) -- proves the harness/reference/kernel
//             math are right, so any PROBE failure is specifically the sub-tile-broadcast gap.
//   [PROBE]   the real shape (small operands + auto-broadcast) -- the decision gate.
//
// DECISION GATE: if the decode (n_tokens=1) PROBE rejects or returns wrong numbers, the swap is
// (a) restricted to prefill with ttnn::lerp kept at decode, or (b) logged as a ttprm lib gap
// (sub-tile group broadcast) to close later. RESULT: OK requires only the CONTROLs to pass (harness
// sanity); the PROBE verdicts are printed for the integrator to read.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::lerp;
using ttprm::View;
using ttprm::view_of;
using ttprm::view_of_shape;

namespace {
// n_embd lands on the COLUMN dim; it only scales the number of col-tiles streamed and is irrelevant
// to the row/group structure under test, so it is scaled down from the real 4096 to keep the test
// snappy while staying multi-tile (16 col-tiles) -- the sub-tile *row* broadcast is what we probe.
constexpr int64_t E = 512;

TensorSpec tspec(int64_t rows, int64_t cols) {
    return TensorSpec(ttnn::Shape({(uint32_t)rows, (uint32_t)cols}),
                      TensorLayout(DataType::BFLOAT16,
                                   PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
}

// Build a [rows,cols] bf16 device tensor with logical element (r,c) = f(r,c).
ttnn::Tensor make_t(distributed::MeshDevice* dev, int64_t rows, int64_t cols,
                    const std::function<float(int64_t, int64_t)>& f) {
    std::vector<float> h((size_t)rows * cols);
    for (int64_t r = 0; r < rows; r++)
        for (int64_t c = 0; c < cols; c++) h[(size_t)r * cols + c] = f(r, c);
    return ttnn::Tensor::from_vector(h, tspec(rows, cols), dev);
}

// Small, bf16-friendly operand value generators (lerp result stays well inside bf16 range, so the
// only thing a low PCC can mean is a STRUCTURAL (broadcast/addressing) error, not rounding).
inline float v_cur(int64_t t, int64_t e)   { return (float)(((t * E + e) % 17) - 8) * 0.25f; }
inline float v_prev(int64_t t, int64_t e)  { return (float)(((t * E + e) % 13) - 6) * 0.25f; }
inline float v_weight(int64_t g, int64_t e){ return (float)((g * E + e) % 6) * 0.2f; }   // 0..1.0

// The TRUE logical token-shift: out[g*nt+t, e] = cur[t,e] + weight[g,e]*(x_prev[t,e]-cur[t,e]).
std::vector<float> lerp_ref(int64_t nt, int64_t G) {
    std::vector<float> o((size_t)G * nt * E);
    for (int64_t g = 0; g < G; g++)
        for (int64_t t = 0; t < nt; t++)
            for (int64_t e = 0; e < E; e++) {
                const float a = v_cur(t, e), b = v_prev(t, e), w = v_weight(g, e);
                o[((size_t)(g * nt + t)) * E + e] = a + w * (b - a);
            }
    return o;
}

double pcc(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size() || a.empty()) return -2.0;
    double ma = 0, mb = 0;
    for (size_t i = 0; i < a.size(); i++) { ma += a[i]; mb += b[i]; }
    ma /= a.size(); mb /= a.size();
    double sab = 0, saa = 0, sbb = 0;
    for (size_t i = 0; i < a.size(); i++) {
        const double da = a[i] - ma, db = b[i] - mb;
        sab += da * db; saa += da * da; sbb += db * db;
    }
    if (saa == 0 || sbb == 0) return (saa == sbb) ? 1.0 : 0.0;
    return sab / std::sqrt(saa * sbb);
}

int g_ctrl_fail = 0;
}  // namespace

int main() {
    std::printf(">>> view_lerp_rwkv_test (RWKV token-shift lerp: does sub-tile group broadcast fuse?)\n");
    std::printf("    shapes: cur/x_prev=[nt,%lld]  weight=[G,%lld]  out=[G*nt,%lld] (group-major rows)\n",
                (long long)E, (long long)E, (long long)E);
    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    dev->enable_program_cache();

    struct Cfg { int64_t nt, G; };
    const std::vector<Cfg> cfgs = {{1, 5}, {1, 6}, {8, 5}, {8, 6},
                                   {32, 6}, {64, 3}, {128, 2}};   // large-batch nt (>=32, mult-of-32)

    // ---- CONTROL: host-expand the broadcast, run a pure-dense lerp. MUST pass. ----
    std::printf("\n  [CONTROL] full operands (no broadcast) -- harness/kernel sanity:\n");
    for (const auto& c : cfgs) {
        const int64_t R = c.G * c.nt;
        auto cur  = make_t(dev.get(), R, E, [&](int64_t r, int64_t e) { return v_cur(r % c.nt, e); });
        auto prev = make_t(dev.get(), R, E, [&](int64_t r, int64_t e) { return v_prev(r % c.nt, e); });
        auto wfull= make_t(dev.get(), R, E, [&](int64_t r, int64_t e) { return v_weight(r / c.nt, e); });
        auto r = lerp(view_of(cur), view_of(prev), view_of(wfull), view_of_shape(R, E));
        double p = r ? pcc(r.value().to_vector<float>(), lerp_ref(c.nt, c.G)) : -2.0;
        const bool ok = r && p > 0.999;
        if (!ok) ++g_ctrl_fail;
        std::printf("    nt=%lld G=%lld  ok=%s pcc=%s%.6f  [%s]\n",
                    (long long)c.nt, (long long)c.G, r ? "Y" : "N",
                    p < 0 ? "" : " ", p, ok ? "PASS" : (r ? "FAIL(bad pcc)" : "FAIL(reject)"));
        if (!r) std::printf("        reject: %s\n", r.error().c_str());
    }

    // ---- PROBE: the real backend shape. cur/x_prev present nt rows (broadcast over the G groups),
    // weight presents G rows (full per group). Auto-broadcast must honor the GROUP-MAJOR, possibly
    // SUB-TILE structure. This is the decision gate; we only REPORT its verdict. ----
    std::printf("\n  [PROBE] RWKV shape (cur/x_prev bcast over groups, weight per-group) -- decision gate:\n");
    bool decode_fuses = true, prefill_fuses = true;
    for (const auto& c : cfgs) {
        const int64_t R = c.G * c.nt;
        auto cur  = make_t(dev.get(), c.nt, E, [&](int64_t t, int64_t e) { return v_cur(t, e); });
        auto prev = make_t(dev.get(), c.nt, E, [&](int64_t t, int64_t e) { return v_prev(t, e); });
        auto wt   = make_t(dev.get(), c.G,  E, [&](int64_t g, int64_t e) { return v_weight(g, e); });
        auto r = lerp(view_of(cur), view_of(prev), view_of(wt), view_of_shape(R, E));
        double p = r ? pcc(r.value().to_vector<float>(), lerp_ref(c.nt, c.G)) : -2.0;
        const bool fused = r && p > 0.999;
        const char* verdict = !r           ? "REJECT (no device work -- caller falls back)"
                            : fused          ? "FUSED-OK"
                                             : "GAP: ok() but WRONG numbers (silent foot-gun!)";
        std::printf("    nt=%lld G=%lld  ok=%s pcc=%s%.6f  -> %s\n",
                    (long long)c.nt, (long long)c.G, r ? "Y" : "N",
                    p < 0 ? "" : " ", p, verdict);
        if (!r) std::printf("        reject: %s\n", r.error().c_str());
        (c.nt == 1 ? decode_fuses : prefill_fuses) &= fused;
    }

    // ---- CANONICAL: lerp(..., canonical_out=true) emits ttnn's batch-padded [G,1,nt,E] layout
    // directly (each gate group in its own ceil32(nt)-row tile-block), so the GGML consumer needs NO
    // post-lerp reshape. Output tensor is [G*Pad, E], Pad=ceil32(nt); only rows g*Pad+t (t<nt) are
    // live. We extract those live rows back into group-major order and PCC vs the same reference. ----
    std::printf("\n  [CANONICAL] canonical_out=true -> 4D logical [G,1,nt,E] (no post-lerp reshape):\n");
    int g_canon_fail = 0;
    for (const auto& c : cfgs) {
        const int64_t R = c.G * c.nt;
        auto cur  = make_t(dev.get(), c.nt, E, [&](int64_t t, int64_t e) { return v_cur(t, e); });
        auto prev = make_t(dev.get(), c.nt, E, [&](int64_t t, int64_t e) { return v_prev(t, e); });
        auto wt   = make_t(dev.get(), c.G,  E, [&](int64_t g, int64_t e) { return v_weight(g, e); });
        auto r = lerp(view_of(cur), view_of(prev), view_of(wt), view_of_shape(R, E), /*canonical=*/true);
        double p = -2.0;
        bool shape_ok = false;
        std::string shp;
        if (r) {
            // The returned tensor's LOGICAL shape must be 4D [G,1,nt,E]. to_vector() yields the logical
            // (unpadded) elements in row-major [G,1,nt,E] order = exactly lerp_ref's group-major order,
            // so the ttnn-internal nt->32 pad is invisible here -- the whole point of the 4D label.
            const auto& sh = r.value().logical_shape();
            shp = std::to_string(sh[0]) + "," + std::to_string(sh[1]) + "," +
                  std::to_string(sh[2]) + "," + std::to_string(sh[3]);
            shape_ok = sh.rank() == 4 && (int64_t)sh[0] == c.G && (int64_t)sh[1] == 1 &&
                       (int64_t)sh[2] == c.nt && (int64_t)sh[3] == E;
            auto v = r.value().to_vector<float>();
            if (shape_ok && (int64_t)v.size() == R * E) p = pcc(v, lerp_ref(c.nt, c.G));
        }
        const bool ok = r && shape_ok && p > 0.999;
        if (!ok) ++g_canon_fail;
        std::printf("    nt=%lld G=%lld  shape=[%s]  ok=%s pcc=%s%.6f  [%s]\n",
                    (long long)c.nt, (long long)c.G, shp.c_str(),
                    r ? "Y" : "N", p < 0 ? "" : " ", p,
                    ok ? "PASS" : (!r ? "FAIL(reject)" : !shape_ok ? "FAIL(shape)" : "FAIL(bad pcc)"));
        if (!r) std::printf("        reject: %s\n", r.error().c_str());
    }

    std::printf("\n  DECISION:\n");
    std::printf("    decode  (n_tokens=1): %s\n",
                decode_fuses ? "FUSES -- swap is safe at decode"
                             : "does NOT fuse -- keep ttnn::lerp at decode, OR close the ttprm "
                               "sub-tile-group-broadcast gap (option a/b)");
    std::printf("    prefill (n_tokens=8): %s\n",
                prefill_fuses ? "FUSES -- swap is safe at prefill"
                              : "does NOT fuse -- groups span <32 rows, same sub-tile gap");

    const bool ok = (g_ctrl_fail == 0) && (g_canon_fail == 0);
    std::printf("\nRESULT: %s\n",
                ok ? "OK (controls + canonical passed; probe verdicts above are the finding)"
                   : g_ctrl_fail ? "FAIL (a CONTROL case is broken -- harness/kernel bug, not the bcast gap)"
                                 : "FAIL (a CANONICAL densify case is broken)");
    return ok ? 0 : 1;
}
