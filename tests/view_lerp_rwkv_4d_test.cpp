// view_lerp_rwkv_4d_test.cpp -- reproduces the GGML-integration HANG.
// The passing view_lerp_rwkv_test builds every operand as a 2D [rows,cols] tensor. The real
// rwkv7-base.cpp backend hands ttprm the *realized* ggml shapes (ggml dims reversed):
//     cur, x_prev -> [1,1,nt,4096]      (was 2D [nt,4096] in the old test)
//     weight      -> [6,1,1,4096]       (was 2D [6,4096]  in the old test)   <-- the suspect
//     out         -> [6,1,nt,4096]
// weight's 4D shape TILE-pads dim -2 (the "1") to 32, so its padded grid is [6,1,32,4096]:
// logical weight row g lives at PHYSICAL row g*32, not g. The View ctor anchors the layout algebra
// on that padded grid (prows = product of padded dims except last = 6*32 = 192), so the grouped
// token-shift address map (which assumes weight rows packed 0,1,2,...,G-1) is wrong. This harness
// builds those exact shapes to see whether it hangs / mismatches.

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
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/data_movement/slice/slice.hpp>

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::lerp;
using ttprm::View;
using ttprm::view_of;
using ttprm::view_of_shape;

namespace {
constexpr int64_t E = 4096;

TensorSpec tspec_nd(const std::vector<uint32_t>& dims) {
    return TensorSpec(ttnn::Shape(dims),
                      TensorLayout(DataType::BFLOAT16,
                                   PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
}

// cur/x_prev: 4D [1,1,nt,E], logical element (t,e).
ttnn::Tensor make_tok(distributed::MeshDevice* dev, int64_t nt,
                      const std::function<float(int64_t, int64_t)>& f) {
    std::vector<float> h((size_t)nt * E);
    for (int64_t t = 0; t < nt; t++)
        for (int64_t e = 0; e < E; e++) h[(size_t)t * E + e] = f(t, e);
    return ttnn::Tensor::from_vector(h, tspec_nd({1, 1, (uint32_t)nt, (uint32_t)E}), dev);
}
// weight: 4D [G,1,1,E], logical element (g,e).
ttnn::Tensor make_w(distributed::MeshDevice* dev, int64_t G,
                    const std::function<float(int64_t, int64_t)>& f) {
    std::vector<float> h((size_t)G * E);
    for (int64_t g = 0; g < G; g++)
        for (int64_t e = 0; e < E; e++) h[(size_t)g * E + e] = f(g, e);
    return ttnn::Tensor::from_vector(h, tspec_nd({(uint32_t)G, 1, 1, (uint32_t)E}), dev);
}

inline float v_cur(int64_t t, int64_t e)   { return (float)(((t * E + e) % 17) - 8) * 0.25f; }
inline float v_prev(int64_t t, int64_t e)  { return (float)(((t * E + e) % 13) - 6) * 0.25f; }
inline float v_weight(int64_t g, int64_t e){ return (float)((g * E + e) % 6) * 0.2f; }

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
}  // namespace

int main() {
    std::printf(">>> view_lerp_rwkv_4d_test (real-model 4D shapes: weight=[G,1,1,E], tok=[1,1,nt,E])\n");
    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    dev->enable_program_cache();

    struct Cfg { int64_t nt, G; };
    // SAME shape repeated -> forces program-cache HITS from iter 2 (the backend reuses the lerp
    // program across all 32 layers; the original mixed-shape cfgs were all cache MISSES and never
    // exercised override_runtime_arguments -- the coverage gap that hid the backend hang).
    const std::vector<Cfg> cfgs = {{1, 6}, {1, 6}, {1, 6}, {1, 6}, {1, 6}, {1, 6}};

    // Keep every iteration's inputs ALIVE so the next allocation lands at a FRESH device address.
    // Freeing between iters would recycle the same address and the override patch would be a no-op,
    // masking the bug. The backend's per-layer lerps each have distinct addresses -> reproduce that.
    std::vector<ttnn::Tensor> keepalive;

    int fails = 0;
    for (const auto& c : cfgs) {
        const int64_t R = c.G * c.nt;
        std::printf("\n  nt=%lld G=%lld  building tensors...\n", (long long)c.nt, (long long)c.G);
        std::fflush(stdout);
        auto cur  = make_tok(dev.get(), c.nt, v_cur);
        auto prev = make_tok(dev.get(), c.nt, v_prev);
        auto wt   = make_w(dev.get(), c.G, v_weight);
        std::printf("    addrs: cur=0x%lx prev=0x%lx wt=0x%lx\n",
                    (unsigned long)cur.buffer()->address(), (unsigned long)prev.buffer()->address(),
                    (unsigned long)wt.buffer()->address());
        std::fflush(stdout);
        keepalive.push_back(cur); keepalive.push_back(prev); keepalive.push_back(wt);

        View va = view_of(cur), vb = view_of(prev), vw = view_of(wt);
        std::printf("    view rows/cols: a=[%lld,%lld] b=[%lld,%lld] w=[%lld,%lld]  out=[%lld,%lld]\n",
                    (long long)va.rows(), (long long)va.cols(),
                    (long long)vb.rows(), (long long)vb.cols(),
                    (long long)vw.rows(), (long long)vw.cols(), (long long)R, (long long)E);
        std::fflush(stdout);

        std::printf("    calling lerp(canonical=true)...\n"); std::fflush(stdout);
        auto r = lerp(va, vb, vw, view_of_shape(R, E), /*canonical=*/true);
        std::printf("    lerp returned: ok=%s\n", r ? "Y" : "N"); std::fflush(stdout);
        if (!r) { std::printf("    reject: %s  [FAIL]\n", r.error().c_str()); ++fails; continue; }
        // The GGML consumer reads a 4D [G,1,nt,E] tensor (no post-lerp reshape) -- assert that label.
        const auto& sh = r.value().logical_shape();
        const bool shape_ok = sh.rank() == 4 && (int64_t)sh[0] == c.G && (int64_t)sh[1] == 1 &&
                              (int64_t)sh[2] == c.nt && (int64_t)sh[3] == E;
        auto v = r.value().to_vector<float>();
        double p = ((int64_t)v.size() == R * E) ? pcc(v, lerp_ref(c.nt, c.G)) : -3.0;
        const bool ok = shape_ok && p > 0.999;
        if (!ok) ++fails;

        // ---- CONSUMER PROBE: feed the ttprm output into downstream ttnn ops, like the model does.
        // The standalone lerp PCC only proves the output VALUES; it never feeds the tensor to another
        // device op. The model hang appears only when a consumer reads it -- reproduce that here.
        std::printf("    [consume] ttnn::add(out,out)...\n"); std::fflush(stdout);
        auto y = ttnn::add(r.value(), r.value());
        auto vy = y.to_vector<float>();
        bool add_ok = vy.size() == v.size();
        if (add_ok) for (size_t i = 0; i < vy.size(); i++)
            if (std::abs(vy[i] - 2.0f * v[i]) > 0.05f) { add_ok = false; break; }
        std::printf("    [consume] add -> %s\n", add_ok ? "OK" : "WRONG/HANG-survived");
        std::fflush(stdout);
        // The real model consumer: slice group 0 out of [G,1,nt,E] -> [1,1,nt,E] (realize_ggml_view).
        std::printf("    [consume] ttnn::slice group0...\n"); std::fflush(stdout);
        auto g0 = ttnn::slice(r.value(),
                              ttnn::SmallVector<int>{0, 0, 0, 0},
                              ttnn::SmallVector<int>{1, 1, (int)c.nt, (int)E},
                              ttnn::SmallVector<int>{1, 1, 1, 1}, std::nullopt);
        auto vg = g0.to_vector<float>();
        std::printf("    [consume] slice -> ok size=%zu\n", vg.size()); std::fflush(stdout);
        std::printf("    shape=[%u,%u,%u,%u] pcc=%.6f size=%zu (want %lld)  [%s]\n",
                    sh.rank() >= 4 ? (unsigned)sh[0] : 0, sh.rank() >= 4 ? (unsigned)sh[1] : 0,
                    sh.rank() >= 4 ? (unsigned)sh[2] : 0, sh.rank() >= 4 ? (unsigned)sh[3] : 0,
                    p, v.size(), (long long)(R * E),
                    ok ? "PASS" : (!shape_ok ? "FAIL(shape)" : "FAIL(bad pcc)"));
        keepalive.push_back(r.value());  // retain the output too -> next iter's output addr shifts
    }
    std::printf("\nRESULT: %s\n", fails == 0 ? "OK" : "FAIL");
    return fails == 0 ? 0 : 1;
}
