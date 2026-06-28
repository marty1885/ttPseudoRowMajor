// row_index_trace_bench.cpp -- is GET_ROWS / SET_ROWS a bottleneck? Measure achieved DRAM
// bandwidth under tt-metal TRACE (pure device time) and compare to peak.
// -----------------------------------------------------------------------------
// The op is a pure whole-tile copy: it READS M*T tiles and WRITES M*T tiles (data crosses the NoC
// twice -- DRAM->L1->DRAM -- there is no single-hop DRAM->DRAM). So bytes_moved = 2 * M*T * 2048.
// Wormhole_b0 DRAM peak is ~256-288 GB/s aggregate; if we reach a large fraction of that we are
// memory-bound (nothing to fix), if we are far below we are ISSUE-bound (per-tile barriers) and
// the batched-issue rewrite is worth it. That is how we decide whether it is a bottleneck.

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
#include <ttnn/operations/trace.hpp>

#include "ttprm.hpp"

using namespace tt::tt_metal;

namespace {
inline float val(int64_t k) { return (float)(((k % 241) - 120) * 0.05f); }

TensorSpec dspec(int64_t R, int64_t A, int64_t B) {
    return TensorSpec(ttnn::Shape({(uint32_t)R, (uint32_t)A, (uint32_t)B}),
                      TensorLayout(DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
}
ttnn::Tensor make_data(distributed::MeshDevice* dev, int64_t R, int64_t A, int64_t B) {
    std::vector<float> h((size_t)R * A * B);
    for (int64_t k = 0; k < R * A * B; k++) h[k] = val(k);
    return ttnn::Tensor::from_vector(h, dspec(R, A, B), dev);
}
ttnn::Tensor make_index(distributed::MeshDevice* dev, const std::vector<int32_t>& v) {
    TensorSpec s(ttnn::Shape({(uint32_t)v.size()}),
                 TensorLayout(DataType::INT32, PageConfig(tt::tt_metal::Layout::ROW_MAJOR), MemoryConfig{}));
    return ttnn::Tensor::from_vector(v, s, dev);
}
}  // namespace

int main() {
    std::printf(">>> row_index_trace_bench (GET_ROWS / SET_ROWS achieved DRAM bandwidth, under trace)\n");
    auto dev = distributed::MeshDevice::create_unit_mesh(0, /*l1_small=*/0, /*trace_region=*/64 << 20);
    dev->enable_program_cache();
    auto& cq = dev->mesh_command_queue();
    namespace tr = ttnn::operations::trace;

    const int WARMUP = 20, ITERS = 200;
    auto traced = [&](std::function<void()> fn) {
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

    // The [300,256,32] KV cache shape: T = 8 tiles/row. Sweep how many rows we touch.
    const int64_t R = 300, A = 256, B = 32, T = (A / 32) * (B / 32);
    auto cache = make_data(dev.get(), R, A, B);

    std::printf("\n  %-26s %8s %8s   %9s %9s\n", "case", "tiles", "us", "GB/s", "%peak(256)");
    for (int64_t M : std::vector<int64_t>{1, 4, 16, 32, 64, 128, 300}) {
        std::vector<int32_t> ids;
        for (int64_t i = 0; i < M; i++) ids.push_back((int32_t)((i * 7 + 3) % R));
        auto idx = make_index(dev.get(), ids);
        auto src = make_data(dev.get(), M, A, B);   // for set_rows

        const double bytes = 2.0 * (double)(M * T) * 2048.0;   // read + write
        double gus = traced([&]() { (void)ttprm::get_rows(cache, idx); });
        double sus = traced([&]() { (void)ttprm::set_rows(cache, idx, src); });
        auto line = [&](const char* nm, double us) {
            double gbs = bytes / (us * 1e3);   // bytes / (us*1e3) = GB/s
            std::printf("  %-26s %8lld %8.2f   %9.1f %9.1f\n", nm, (long long)(M * T), us, gbs, 100.0 * gbs / 256.0);
        };
        char gn[40], sn[40];
        std::snprintf(gn, sizeof gn, "get_rows M=%lld", (long long)M);
        std::snprintf(sn, sizeof sn, "set_rows M=%lld", (long long)M);
        line(gn, gus); line(sn, sus);
    }
    std::printf("\nRESULT: OK\n");
    return 0;
}
