// view_ternary_trace_bench.cpp -- ttprm::fma / ttprm::lerp (fused gather+ternary+scatter, no
// relayout) vs the native ttnn op chain they replace, UNDER tt-metal TRACE (device time).
// -----------------------------------------------------------------------------
// fma(a,b,c)  = a*b + c          native (reshape-tax) = add(mul(reshape(a,head), b), c)
// lerp(a,b,w) = a + w*(b - a)    native (reshape-tax) = add(a, mul(w, sub(reshape(b,head), a)))
//
// Two question this answers:
//  1. CORRECTNESS ("are we off?"): every case cross-checks the ttprm result against the native
//     result with PCC -- printed as `xPCC`. bf16 rounding -> expect >0.999, not bit-exact.
//  2. SPEED: reshape-tax cases beat the native chain (it must MATERIALIZE the flatten relayout in
//     DRAM, then run 2-3 separate eltwise ops). Broadcast cases isolate the read-once win: the
//     baseline is the SAME ttprm op with pre-EXPANDED full operands (streams G*tpg tiles) vs the
//     resident path (gathers tpg once) -- the RWKV lerp shape (2 of 3 operands repeat) is where it
//     pays off, the read traffic drops from G*tpg to ~cores*tpg per broadcast operand.

#include <chrono>
#include <cmath>
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
#include <ttnn/operations/data_movement/repeat/repeat.hpp>
#include <ttnn/operations/eltwise/binary/binary.hpp>
#include <ttnn/operations/trace.hpp>

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::fma;
using ttprm::lerp;
using ttprm::View;
using ttprm::view_of;
using ttprm::view_of_shape;

namespace {
inline float val(int64_t k) { return (float)((k % 17) - 8); }   // small -> bf16 product stays sane

TensorSpec tspec(int64_t rows, int64_t cols) {
    return TensorSpec(ttnn::Shape({(uint32_t)rows, (uint32_t)cols}),
                      TensorLayout(DataType::BFLOAT16,
                                   PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
}
ttnn::Tensor make_input(distributed::MeshDevice* dev, int64_t rows, int64_t cols, int64_t seed = 0) {
    std::vector<float> h((size_t)rows * cols);
    for (int64_t i = 0; i < rows * cols; i++) h[i] = val(i + seed);
    return ttnn::Tensor::from_vector(h, tspec(rows, cols), dev);
}

// Pearson correlation on logical elements (the "are we off?" gate). 1.0 == identical.
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

struct Case {
    std::string name;
    int64_t out_tiles;
    std::function<ttnn::Tensor()> ttprm_fn;
    std::function<ttnn::Tensor()> native_fn;
};
}  // namespace

int main() {
    std::printf(">>> view_ternary_trace_bench (ttprm fused fma/lerp vs native ttnn chain, under trace)\n");
    auto dev = distributed::MeshDevice::create_unit_mesh(0, /*l1_small=*/0, /*trace_region=*/64 << 20);
    dev->enable_program_cache();
    auto& cq = dev->mesh_command_queue();
    namespace tr = ttnn::operations::trace;

    std::list<ttnn::Tensor> keep;
    std::vector<Case> cases;
    auto hold = [&](ttnn::Tensor t) -> const ttnn::Tensor& { keep.push_back(std::move(t)); return keep.back(); };

    // RESHAPE-TAX: a (or b) is flat [1,R*C] viewed as the head [R,C]; the native chain must
    // materialize the reshape, then run the eltwise ops. ttprm fakes it in one fused program.
    struct Spec { std::string name; int64_t R, C; };
    std::vector<Spec> specs = {
        {"[64,64]    (4t)",     64,  64},
        {"[128,128]  (16t mc)", 128, 128},
        {"[256,256]  (64t mc)", 256, 256},
    };
    for (auto& s : specs) {
        const int64_t R = s.R, C = s.C;
        const ttnn::Shape head({(uint32_t)R, (uint32_t)C});
        // fma: a flat-viewed, b & c head-shaped.
        const ttnn::Tensor& a = hold(make_input(dev.get(), 1, R * C, 1));
        const ttnn::Tensor& b = hold(make_input(dev.get(), R, C, 2));
        const ttnn::Tensor& c = hold(make_input(dev.get(), R, C, 3));
        cases.push_back(Case{"fma " + s.name, (R / 32) * (C / 32),
            [&a, &b, &c, R, C]() {
                View av = view_of(a).slice({{0, 1, 1}, {0, R * C, 1}}).reshape({R, C});
                return fma(av, view_of(b), view_of(c)).value();
            },
            [&a, &b, &c, head]() { return ttnn::add(ttnn::multiply(ttnn::reshape(a, head), b), c); }});
        // lerp: b flat-viewed, a & w head-shaped.
        const ttnn::Tensor& la = hold(make_input(dev.get(), R, C, 4));
        const ttnn::Tensor& lb = hold(make_input(dev.get(), 1, R * C, 5));
        const ttnn::Tensor& lw = hold(make_input(dev.get(), R, C, 6));
        cases.push_back(Case{"lerp " + s.name, (R / 32) * (C / 32),
            [&la, &lb, &lw, R, C]() {
                View bv = view_of(lb).slice({{0, 1, 1}, {0, R * C, 1}}).reshape({R, C});
                return lerp(view_of(la), bv, view_of(lw)).value();
            },
            [&la, &lb, &lw, head]() {
                return ttnn::add(la, ttnn::multiply(lw, ttnn::subtract(ttnn::reshape(lb, head), la)));
            }});
    }

    // RWKV BROADCAST shape: 2 of 3 operands are one group [Rg,C] repeated over G groups; the 3rd is
    // full [G*Rg,C]. ttprm fakes the repeat (gathers the two groups ONCE, resident). The baseline is
    // the SAME fused op, but it must FIRST materialize the repeat with ttnn::repeat -- the real DRAM
    // round-trip a caller pays to stack [Rg,C] into [G*Rg,C]. Materialization is TIMED (inside the
    // lambda), so this isolates exactly the cost ttprm avoids by never realizing the repeat.
    struct BSpec { std::string name; int64_t Rg, C, G; };
    std::vector<BSpec> bspecs = {
        {"a,b bcast x256 (1t grp)", 32, 32, 256},
        {"a,b bcast x128 (4t grp)", 64, 64, 128},
    };
    for (auto& s : bspecs) {
        const int64_t Rg = s.Rg, C = s.C, G = s.G;
        const ttnn::Tensor& ag = hold(make_input(dev.get(), Rg, C, 1));
        const ttnn::Tensor& bg = hold(make_input(dev.get(), Rg, C, 2));
        const ttnn::Tensor& full = hold(make_input(dev.get(), G * Rg, C, 3));  // the full 3rd operand
        const int64_t ot = (G * Rg / 32) * (C / 32);
        const ttnn::SmallVector<uint32_t> rep{(uint32_t)G, 1u};
        // fma: out = a*b + c, a&b broadcast. baseline materializes a&b via ttnn::repeat (timed).
        cases.push_back(Case{"fma " + s.name, ot,
            [&ag, &bg, &full]() { return fma(view_of(ag), view_of(bg), view_of(full)).value(); },
            [&ag, &bg, &full, rep]() {
                return fma(view_of(ttnn::repeat(ag, rep)), view_of(ttnn::repeat(bg, rep)),
                           view_of(full)).value();
            }});
        // lerp: out = a + w*(b-a), a&b broadcast, w full (the RWKV token-shift shape).
        cases.push_back(Case{"lerp " + s.name, ot,
            [&ag, &bg, &full]() { return lerp(view_of(ag), view_of(bg), view_of(full)).value(); },
            [&ag, &bg, &full, rep]() {
                return lerp(view_of(ttnn::repeat(ag, rep)), view_of(ttnn::repeat(bg, rep)),
                            view_of(full)).value();
            }});
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

    bool all_ok = true;
    std::printf("\n  %-26s %5s   %8s   %11s %11s   %11s %11s   %s\n",
                "case", "tiles", "xPCC", "ttprm eag", "ttprm TRACE", "base eag", "base TRACE", "trace x");
    for (auto& c : cases) {
        const double p = pcc(c.ttprm_fn().to_vector<float>(), c.native_fn().to_vector<float>());
        const bool ok = p > 0.999;
        all_ok = all_ok && ok;
        double pe = eager(c.ttprm_fn), pt = traced(c.ttprm_fn);
        double ne = eager(c.native_fn), nt = traced(c.native_fn);
        std::printf("  %-26s %5lld   %7.5f%s   %9.2f   %9.2f     %9.2f   %9.2f     %5.2fx\n",
                    c.name.c_str(), (long long)c.out_tiles, p, ok ? " " : "!", pe, pt, ne, nt, nt / pt);
    }
    std::printf("\nRESULT: %s\n", all_ok ? "OK" : "FAIL (a case is OFF -- xPCC <= 0.999)");
    return all_ok ? 0 : 1;
}
