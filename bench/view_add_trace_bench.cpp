// view_add_trace_bench.cpp -- ttprm::view_add (fused gather+add+scatter, no relayout) vs the
// native ttnn  reshape+add  it replaces, UNDER tt-metal TRACE (device-time, not host-dispatch).
// -----------------------------------------------------------------------------
// The GGML idiom  out[64,64] = a[1,4096] + b[64,64].reshape([1,4096])  (== out[h][w] =
// a_flat[h*S+w] + b[h][w]). The native path must MATERIALIZE the flatten/head relayout in DRAM
// (the reshape tax): reshape a[1,4096]->[64,64] forces an untilize->retile (128 tiles -> 4), then
// adds b. ttprm fakes a's head-tile on the read side and adds in one fused program -- zero
// reshape realized. We use the LEANEST native (one reshape + add) as the strongest baseline.
//
// Eager timing is host-dispatch-bound; we capture each path once into a device trace and time
// pure replay, so what remains is DEVICE TIME. view_add re-lowers its views per call (pure host,
// trace-legal) and launches the CACHED device_operation -> trace records only the device program.

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <list>
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
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/trace.hpp>

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::add;
using ttprm::View;
using ttprm::view_of;
using ttprm::view_of_shape;

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
    std::printf(">>> view_add_trace_bench (ttprm fused add-view vs ttnn reshape+add, under trace)\n");
    auto dev = distributed::MeshDevice::create_unit_mesh(0, /*l1_small=*/0, /*trace_region=*/64 << 20);
    dev->enable_program_cache();
    auto& cq = dev->mesh_command_queue();
    namespace tr = ttnn::operations::trace;

    // Keep operands alive for the whole run. std::list is node-based -- push_back never moves an
    // existing element, so the references the case lambdas capture stay valid no matter how many
    // cases we add later. `hold` stashes a tensor and hands back its stable reference at creation,
    // so we never re-index into the container (the trap a vector + numeric indices invites).
    std::list<ttnn::Tensor> keep;
    std::vector<Case> cases;
    auto hold = [&](ttnn::Tensor t) -> const ttnn::Tensor& { keep.push_back(std::move(t)); return keep.back(); };

    // RESHAPE-TAX cases: a flat [1,R*C] + b head [R,C]; native must reshape a -> [R,C] then add.
    struct Spec { std::string name; int64_t R, C; };
    std::vector<Spec> specs = {
        {"[1,4096]+[64,64]    -> [64,64]    (4t)",     64,  64},
        {"[1,16384]+[128,128] -> [128,128]  (16t mc)", 128, 128},
        {"[1,65536]+[256,256] -> [256,256]  (64t mc)", 256, 256},
    };
    for (auto& s : specs) {
        const ttnn::Tensor& a = hold(make_input(dev.get(), 1, s.R * s.C));  // a (flat)
        const ttnn::Tensor& b = hold(make_input(dev.get(), s.R, s.C));      // b (head)
        const int64_t R = s.R, C = s.C;
        const ttnn::Shape head({(uint32_t)R, (uint32_t)C});
        cases.push_back(Case{s.name, (R / 32) * (C / 32),
            [&a, &b, R, C]() {
                View av = view_of(a).slice({{0, 1, 1}, {0, R * C, 1}}).reshape({R, C});
                return add(av, view_of(b)).value();
            },
            [&a, &b, head]() { return ttnn::add(ttnn::reshape(a, head), b); }});
    }

    // TILE-ALIGNED FLOOR: both inputs already in the SAME [R,C] layout -> all three endpoints are
    // DENSE (the "plain" whole-page node). There is NO reshape tax for native to pay (bare
    // ttnn::add), so this row shows whether the plain DenseTensorView gather+scatter helps or just
    // adds overhead when there's nothing to relayout. Expected: a TIE or slight loss.
    std::vector<Spec> aligned = {
        {"[64,64]+[64,64]     -> [64,64]    (4t  aligned)",  64,  64},
        {"[256,256]+[256,256] -> [256,256]  (64t aligned)",  256, 256},
    };
    for (auto& s : aligned) {
        const ttnn::Tensor& a = hold(make_input(dev.get(), s.R, s.C));  // a (already head-shaped)
        const ttnn::Tensor& b = hold(make_input(dev.get(), s.R, s.C));  // b
        const int64_t R = s.R, C = s.C;
        cases.push_back(Case{s.name, (R / 32) * (C / 32),
            [&a, &b]() {  // a/b identity -> DENSE; out identity -> DENSE (no AFFINE anywhere)
                return add(view_of(a), view_of(b)).value();
            },
            [&a, &b]() { return ttnn::add(a, b); }});  // native: bare add, no reshape tax
    }

    // GROUP BROADCAST: a is ONE group [Rg,C] repeated over G groups; b/out are [G*Rg,C]. ttprm
    // auto-detects a as broadcast and (now) gathers its group ONCE, re-indexing it across the G
    // groups. Baseline = the SAME ttprm add but with a pre-EXPANDED full [G*Rg,C] operand (no
    // broadcast -> it streams all G*Rg a-tiles). Same result; the only difference is the broadcast
    // operand's read traffic: tpg tiles (resident) vs G*tpg (streamed) -- so this row isolates the
    // read-once win directly. Large G makes it visible.
    struct BSpec { std::string name; int64_t Rg, C, G; };
    std::vector<BSpec> bspecs = {
        {"bcast a[64,64]x8   + b[512,64]   (8 grp, 4t grp)",   64, 64, 8},
        {"bcast a[128,128]x8 + b[1024,128] (8 grp, 16t grp)",  128, 128, 8},
        {"bcast a[32,32]x256 + b[8192,32]  (256 grp, 1t grp)", 32, 32, 256},  // large G: re-read = 256*tpg
        {"bcast a[64,64]x128 + b[8192,64]  (128 grp, 4t grp)", 64, 64, 128},
    };
    for (auto& s : bspecs) {
        const int64_t Rg = s.Rg, C = s.C, G = s.G;
        const ttnn::Tensor& ag = hold(make_input(dev.get(), Rg, C));        // one group
        const ttnn::Tensor& b  = hold(make_input(dev.get(), G * Rg, C));    // full
        // a_full = ag tiled G times down the rows (the materialized repeat).
        std::vector<float> hf((size_t)G * Rg * C);
        for (int64_t g = 0; g < G; g++)
            for (int64_t i = 0; i < Rg * C; i++) hf[(size_t)g * Rg * C + i] = val(i);
        const ttnn::Tensor& af = hold(ttnn::Tensor::from_vector(hf, tspec(G * Rg, C), dev.get()));
        cases.push_back(Case{s.name, (G * Rg / 32) * (C / 32),
            [&ag, &b]() { return add(view_of(ag), view_of(b)).value(); },   // resident broadcast
            [&af, &b]() { return add(view_of(af), view_of(b)).value(); }}); // streamed full operand
    }

    // BOTH operands broadcast (the RWKV-lerp traffic shape: 2 of 3 streams are repeats). Re-read =
    // 2*G*tpg; resident = 2*cores*tpg. Baseline = both materialized to full [G*Rg,C] and streamed.
    // This is where faking the repeat should clearly win for large G (both reads drop to ~cores*tpg).
    std::vector<BSpec> bspecs2 = {
        {"bcast2 a[32,32]x256 + b[32,32]x256 (both, 256g)", 32, 32, 256},
        {"bcast2 a[64,64]x128 + b[64,64]x128 (both, 128g)", 64, 64, 128},
    };
    for (auto& s : bspecs2) {
        const int64_t Rg = s.Rg, C = s.C, G = s.G;
        const ttnn::Tensor& ag = hold(make_input(dev.get(), Rg, C));
        const ttnn::Tensor& bg = hold(make_input(dev.get(), Rg, C));
        std::vector<float> hf((size_t)G * Rg * C);
        for (int64_t g = 0; g < G; g++)
            for (int64_t i = 0; i < Rg * C; i++) hf[(size_t)g * Rg * C + i] = val(i);
        const ttnn::Tensor& af = hold(ttnn::Tensor::from_vector(hf, tspec(G * Rg, C), dev.get()));
        const ttnn::Tensor& bf = hold(ttnn::Tensor::from_vector(hf, tspec(G * Rg, C), dev.get()));
        cases.push_back(Case{s.name, (G * Rg / 32) * (C / 32),
            [&ag, &bg]() { return add(view_of(ag), view_of(bg)).value(); },  // both resident
            [&af, &bf]() { return add(view_of(af), view_of(bf)).value(); }}); // both streamed full
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

    std::printf("\n  %-36s %5s   %11s %11s   %11s %11s   %s\n",
                "case", "tiles", "ttprm eag", "ttprm TRACE", "ttnn eag", "ttnn TRACE", "trace x");
    for (auto& c : cases) {
        double pe = eager(c.ttprm_fn), pt = traced(c.ttprm_fn);
        double ne = eager(c.native_fn), nt = traced(c.native_fn);
        std::printf("  %-36s %5lld   %9.2f   %9.2f     %9.2f   %9.2f     %5.2fx\n",
                    c.name.c_str(), (long long)c.out_tiles, pe, pt, ne, nt, nt / pt);
    }
    std::printf("\nRESULT: OK\n");
    return 0;
}
