// view_norm_trace_bench.cpp -- ttprm fused per-head/wide norm vs the NATIVE TTNN chain that RWKV
// actually runs, INCLUDING the reshapes, under tt-metal TRACE (pure device time).
// -----------------------------------------------------------------------------
// RWKV's per-head norms (time-mix group-norm, the kk l2-normalize) are wrapped in a relayout:
//
//     flat [1,4096]  --ttnn::reshape-->  head [64,64]  --norm-->  [64,64]  --ttnn::reshape-->  [1,4096]
//
// Each ttnn::reshape that changes the tile grid forces a physical untilize->retile (the "reshape
// tax"). ttprm fuses the whole chain into ONE op: the flat input is *viewed* as head[64,64] (an
// AFFINE face-row gather -- the reshape is never materialized), the per-head reduce + normalize
// run on the gathered tiles, and the result is scattered straight back to flat. So the native
// side here is the REAL competitor: reshape + norm + reshape, not just the bare norm.
//
// TTNN has no l2_norm; RWKV's kk-normalize (x / sqrt(sum x^2 + eps)) has the same op structure as
// RMS-norm (reduce -> rsqrt -> mul, differing only by the 1/N folded into the scaler), so
// ttnn::rms_norm is the fair native cost-proxy for the l2 path (labelled [rms~l2]).
//
// Family-1 (att/ffn LayerNorm over the full n_embd=4096) is NOT reshape-wrapped -- native is a bare
// ttnn::layer_norm. ttprm's only lever there is the single-token DECODE padding: [1,4096] is 1 live
// row in 32 (32x padded), and the AFFINE gather touches only the ~8 KB of live face-rows vs the
// native op streaming all 128 padded tiles (256 KB). That row measures whether large-layernorm is
// DRAM-bound enough for the live-only read to win.
//
// Eager numbers are host-dispatch-bound (hidden device cost); the TRACE columns are the real
// per-op device time. trace x = native_TRACE / ttprm_TRACE (>1 => ttprm wins).

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_ops.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>
#include <ttnn/operations/trace.hpp>

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::View;
using ttprm::view_of;
using ttprm::l2_norm;
using ttprm::layer_norm;

namespace {
inline float val(int64_t k) { return (float)(((k % 241) - 120) * 0.05f); }

TensorSpec tspec(int64_t rows, int64_t cols) {
    return TensorSpec(ttnn::Shape({(uint32_t)rows, (uint32_t)cols}),
                      TensorLayout(DataType::BFLOAT16,
                                   PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
}
TensorSpec tspec4(int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    return TensorSpec(ttnn::Shape({(uint32_t)d0, (uint32_t)d1, (uint32_t)d2, (uint32_t)d3}),
                      TensorLayout(DataType::BFLOAT16,
                                   PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
}
ttnn::Tensor make_input(distributed::MeshDevice* dev, int64_t rows, int64_t cols, float scale = 1.0f) {
    std::vector<float> h((size_t)rows * cols);
    for (int64_t i = 0; i < rows * cols; i++) h[i] = val(i) * scale;
    return ttnn::Tensor::from_vector(h, tspec(rows, cols), dev);
}
ttnn::Tensor make_empty4(distributed::MeshDevice* dev, int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
    return tt::tt_metal::create_device_tensor(tspec4(d0, d1, d2, d3), dev);
}
ttnn::Tensor make_weight(distributed::MeshDevice* dev, int64_t w, float base, float step) {
    std::vector<float> h((size_t)w);
    for (int64_t i = 0; i < w; i++) h[i] = base + step * (float)(i % 7);
    return ttnn::Tensor::from_vector(h, tspec(1, w), dev);
}

struct Case {
    std::string name;
    int64_t out_tiles;
    std::function<ttnn::Tensor()> ttprm_fn;
    std::function<ttnn::Tensor()> native_fn;
};
}  // namespace

int main() {
    std::printf(">>> view_norm_trace_bench (ttprm fused norm vs ttnn reshape+norm+reshape, under trace)\n");
    auto dev = distributed::MeshDevice::create_unit_mesh(0, /*l1_small=*/0, /*trace_region=*/64 << 20);
    dev->enable_program_cache();
    auto& cq = dev->mesh_command_queue();
    namespace tr = ttnn::operations::trace;

    // Cases capture stable POINTERS into `keep`; reserve so it never reallocates (dangling refs).
    std::vector<ttnn::Tensor> keep;
    keep.reserve(64);
    std::vector<Case> cases;
    auto hold = [&](ttnn::Tensor t) -> const ttnn::Tensor* { keep.push_back(std::move(t)); return &keep.back(); };

    const float EPS = 1e-5f;

    // head-view helper: slice the single live row of a flat [1,N], present as [H,S] (AFFINE gather).
    auto head_view = [](const ttnn::Tensor& t, int64_t H, int64_t S) {
        return view_of(t).slice({{0, 1, 1}, {0, H * S, 1}}).reshape({H, S});
    };

    // ---- Family 2: per-head LayerNorm WITH affine, the RWKV group-norm (reshape-wrapped) ----
    for (auto HS : std::vector<std::pair<int64_t,int64_t>>{{64,64},{256,64},{1024,64}}) {
        int64_t H = HS.first, S = HS.second, N = H * S;
        const ttnn::Tensor* flat = hold(make_input(dev.get(), 1, N));
        const ttnn::Tensor* g    = hold(make_weight(dev.get(), S, 1.0f, 0.05f));
        const ttnn::Tensor* b    = hold(make_weight(dev.get(), S, 0.0f, 0.02f));
        char nm[96];
        std::snprintf(nm, sizeof nm, "F2 LN+affine  [1,%lld]->[%lld,%lld]  (group-norm)", (long long)N, (long long)H, (long long)S);
        cases.push_back(Case{nm, H * S / 1024,
            [=]() { return layer_norm(head_view(*flat, H, S), g, b, EPS).value(); },
            [=]() {
                auto r = ttnn::reshape(*flat, ttnn::Shape({(uint32_t)H, (uint32_t)S}));
                auto y = ttnn::layer_norm(r, EPS, *g, *b);
                return ttnn::reshape(y, ttnn::Shape({1, (uint32_t)(H * S)}));
            }});
    }

    // ---- Family 2: per-head l2-normalize, the RWKV kk-norm (reshape-wrapped, native = rms~l2) ----
    for (auto HS : std::vector<std::pair<int64_t,int64_t>>{{64,64},{256,64}}) {
        int64_t H = HS.first, S = HS.second, N = H * S;
        const ttnn::Tensor* flat = hold(make_input(dev.get(), 1, N));
        char nm[96];
        std::snprintf(nm, sizeof nm, "F2 l2 [rms~l2] [1,%lld]->[%lld,%lld]  (kk-normalize)", (long long)N, (long long)H, (long long)S);
        cases.push_back(Case{nm, H * S / 1024,
            [=]() { return l2_norm(head_view(*flat, H, S), 1e-6f).value(); },
            [=]() {
                auto r = ttnn::reshape(*flat, ttnn::Shape({(uint32_t)H, (uint32_t)S}));
                auto y = ttnn::rms_norm(r, 1e-6f);
                return ttnn::reshape(y, ttnn::Shape({1, (uint32_t)(H * S)}));
            }});
    }

    // ---- Family 1: wide att/ffn LayerNorm over the full n_embd, single-token DECODE (NO reshape) ----
    // ttprm: flat [1,4096] viewed as the live single row of width 4096 (AFFINE => live-only ~8 KB read,
    // WIDE variant ctiles=128). native: bare ttnn::layer_norm streams all 128 padded tiles (256 KB).
    {
        int64_t N = 4096;
        const ttnn::Tensor* flat = hold(make_input(dev.get(), 1, N));
        const ttnn::Tensor* g    = hold(make_weight(dev.get(), N, 1.0f, 0.01f));
        const ttnn::Tensor* b    = hold(make_weight(dev.get(), N, 0.0f, 0.01f));
        cases.push_back(Case{"F1 wide LN+affine [1,4096] decode (live-only vs padded)", N / 1024,
            [=]() { return layer_norm(view_of(*flat).slice({{0,1,1},{0,N,1}}).reshape({1, N}), g, b, EPS).value(); },
            [=]() { return ttnn::layer_norm(*flat, EPS, *g, *b); }});
    }

    // Same F1 op, but with the explicit rank-4 bound output used by the RWKV lowering.
    // This checks whether binding the GGML-shaped output changes the ttprm path/cost.
    {
        int64_t N = 4096;
        const ttnn::Tensor* flat = hold(make_input(dev.get(), 1, N));
        const ttnn::Tensor* g    = hold(make_weight(dev.get(), N, 1.0f, 0.01f));
        const ttnn::Tensor* b    = hold(make_weight(dev.get(), N, 0.0f, 0.01f));
        const ttnn::Tensor* out  = hold(make_empty4(dev.get(), 1, 1, 1, N));
        cases.push_back(Case{"F1 wide LN+affine [1,4096] bound rank4 out (RWKV-like)", N / 1024,
            [=]() {
                auto xv = view_of(*flat).slice({{0,1,1},{0,N,1}}).reshape({1, N});
                auto ov = view_of(*out).slice({{0,1,1},{0,N,1}});
                return layer_norm(xv, g, b, EPS, ov).value();
            },
            [=]() { return ttnn::layer_norm(*flat, EPS, *g, *b); }});
    }

    // ---- DECOMPOSITION at the hot decode shape [1,4096]->[64,64] (isolate gather vs compute) ----
    // (D1) pure gather+scatter, NO compute: ttprm::realize of the head view vs a bare ttnn::reshape.
    //      => the cost of the AFFINE face-row gather + scatter alone.
    // (D2) l2 on an ALREADY-head [64,64] (DENSE whole-page gather) vs the AFFINE-gather l2 above.
    //      Same compute, different gather => the delta is the AFFINE gather penalty.
    {
        const ttnn::Tensor* flat = hold(make_input(dev.get(), 1, 4096));
        cases.push_back(Case{"D1 gather+scatter only [1,4096]->[64,64] (no compute)", 4,
            [=]() { return ttprm::realize(view_of(*flat).slice({{0,1,1},{0,4096,1}}).reshape({64,64})).value(); },
            [=]() { return ttnn::reshape(*flat, ttnn::Shape({64,64})); }});
        const ttnn::Tensor* head = hold(make_input(dev.get(), 64, 64));
        cases.push_back(Case{"D2 l2 DENSE head[64,64] (whole-page gather)", 4,
            [=]() { return l2_norm(view_of(*head), 1e-6f).value(); },
            [=]() { return ttnn::rms_norm(*head, 1e-6f); }});
    }

    const int WARMUP = 20, ITERS = 200;
    auto eager = [&](std::function<ttnn::Tensor()> fn) {
        for (int i = 0; i < WARMUP; i++) fn();
        distributed::Finish(cq);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) fn();
        distributed::Finish(cq);
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
    };
    auto traced = [&](std::function<ttnn::Tensor()> fn) {
        const auto qid = ttnn::QueueId(0);
        fn(); distributed::Finish(cq);
        auto tid = tr::begin_trace_capture(dev.get(), qid);
        fn();
        tr::end_trace_capture(dev.get(), tid, qid);
        distributed::Finish(cq);
        for (int i = 0; i < WARMUP; i++) tr::execute_trace(dev.get(), tid, qid, false);
        distributed::Finish(cq);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < ITERS; i++) tr::execute_trace(dev.get(), tid, qid, false);
        distributed::Finish(cq);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / ITERS;
        tr::release_trace(dev.get(), tid);
        return us;
    };

    std::printf("\n  %-46s %5s   %10s %11s   %10s %11s   %s\n",
                "case", "tiles", "ttprm eag", "ttprm TRACE", "ttnn eag", "ttnn TRACE", "trace x");
    for (auto& c : cases) {
        double pe = eager(c.ttprm_fn), pt = traced(c.ttprm_fn);
        double ne = eager(c.native_fn), nt = traced(c.native_fn);
        std::printf("  %-46s %5lld   %8.2f   %9.2f     %8.2f   %9.2f     %5.2fx\n",
                    c.name.c_str(), (long long)c.out_tiles, pe, pt, ne, nt, nt / pt);
    }
    std::printf("\nRESULT: OK\n");
    return 0;
}
