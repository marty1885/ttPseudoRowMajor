// view_trace_bench.cpp -- ttprm::realize (on-device affine generator) vs chained TTNN
// relayout, UNDER tt-metal TRACE, swept across shapes/classes so the overhead is visible.
// -----------------------------------------------------------------------------
// Eager timing is host-dispatch-bound and hides device cost. We capture each path ONCE
// into a device trace and time pure on-device REPLAY: host dispatch collapses to one trace
// launch for both sides, so what remains is DEVICE TIME -- the real overhead of the gather.
//
// The ttprm side is two-phase: prelower() computes the affine args (pure host, no upload),
// then realize(in, Lowered) is a pure cached device launch -- trace-legal, nothing to hoist.
//
// Cases include identity [64,64] (TILE): ttnn reshape-to-same-shape is a no-op, so that row
// shows ttprm's FLOOR cost (gather+scatter of N tiles with no relayout benefit) -- the
// baseline overhead the reshape cases must beat.

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
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/trace.hpp>

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::View;
using ttprm::view_of;

namespace {
inline float val(int64_t k) { return (float)((k % 241) - 120); }

TensorSpec tspec(int64_t rows, int64_t cols) {
    return TensorSpec(ttnn::Shape({(uint32_t)rows, (uint32_t)cols}),
                      TensorLayout(DataType::BFLOAT16,
                                   PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
}
ttnn::Tensor make_input(distributed::MeshDevice* dev, int64_t rows, int64_t cols) {
    std::vector<float> h((size_t)rows * cols);
    for (int64_t i = 0; i < rows * cols; i++) h[i] = val(i);
    return ttnn::Tensor::from_vector(h, tspec(rows, cols), dev);
}

struct Case {
    std::string name;
    int64_t out_tiles;
    std::function<ttnn::Tensor()> ttprm_fn;
    std::function<ttnn::Tensor()> native_fn;
};
}  // namespace

int main() {
    std::printf(">>> view_trace_bench (ttprm affine generator vs ttnn, swept, under trace)\n");
    auto dev = distributed::MeshDevice::create_unit_mesh(0, /*l1_small=*/0, /*trace_region=*/64 << 20);
    dev->enable_program_cache();
    auto& cq = dev->mesh_command_queue();
    namespace tr = ttnn::operations::trace;

    // Keep inputs + lowered handles alive for the whole run (cases hold references into these).
    std::vector<ttnn::Tensor> keep;
    std::vector<ttprm::Lowered> lows;
    std::vector<Case> cases;

    struct Spec { std::string name; int64_t R, C; std::vector<int64_t> rsh; ttnn::Shape tgt; };
    std::vector<Spec> specs = {
        {"identity [64,64]      (TILE floor)",   64,    64,   {},          ttnn::Shape({64,64})},
        {"[1,4096]->[64,64]     (head, ROW)",    1,     4096, {64,64},     ttnn::Shape({64,64})},
        {"[1,4096]->[128,32]    (folded, ROW)",  1,     4096, {128,32},    ttnn::Shape({128,32})},
        {"[1,16384]->[128,128]  (multi-core)",   1,     16384,{128,128},   ttnn::Shape({128,128})},
        {"[1,65536]->[256,256]  (big mc)",       1,     65536,{256,256},   ttnn::Shape({256,256})},
    };
    for (auto& s : specs) {
        keep.push_back(make_input(dev.get(), s.R, s.C));
        const ttnn::Tensor& in = keep.back();
        View v = view_of(in);
        if (!s.rsh.empty()) v = v.slice({{0, 1, 1}, {0, s.C, 1}}).reshape(s.rsh);
        auto low = ttprm::prelower(v);
        if (!low) { std::printf("  prelower FAILED (%s): %s\n", s.name.c_str(), low.error().c_str()); continue; }
        lows.push_back(low.value());
    }
    // bind cases against the stable kept vectors (indices align).
    for (size_t i = 0; i < specs.size(); i++) {
        const ttnn::Tensor& in = keep[i];
        const ttprm::Lowered& L = lows[i];
        ttnn::Shape tgt = specs[i].tgt;
        cases.push_back(Case{specs[i].name, L.in_desc.out_tiles(),
            [&L]() { return ttprm::realize(L); },
            [&in, tgt]() { return ttnn::reshape(in, tgt); }});
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

    std::printf("\n  %-34s %5s   %11s %11s   %11s %11s   %s\n",
                "case", "tiles", "ttprm eag", "ttprm TRACE", "ttnn eag", "ttnn TRACE", "trace x");
    for (auto& c : cases) {
        double pe = eager(c.ttprm_fn), pt = traced(c.ttprm_fn);
        double ne = eager(c.native_fn), nt = traced(c.native_fn);
        std::printf("  %-34s %5lld   %9.2f   %9.2f     %9.2f   %9.2f     %5.2fx\n",
                    c.name.c_str(), (long long)c.out_tiles, pe, pt, ne, nt, nt / pt);
    }
    std::printf("\nRESULT: OK\n");
    return 0;
}
