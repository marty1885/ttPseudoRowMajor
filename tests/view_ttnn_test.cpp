// view_ttnn_test.cpp -- ttprm::realize on the ttnn::Tensor surface, compared DIRECTLY
// against chained TTNN ops (ttnn::reshape / ttnn::slice) as the oracle.
// -----------------------------------------------------------------------------
// This is the "speak TTNN + program cache" test currency: inputs are ttnn::Tensors,
// the op is the cached ttprm::realize device_operation, and correctness is parity with
// what TTNN itself would produce for the same logical reshape/slice -- no raw Metalium
// wiring per case. Part 2 of the ask lives here too: a battery of increasingly nasty
// reshape/slice/flatten/permute chains, with the expected verdict (fused vs REJECT)
// asserted, to flush out where the algebra breaks.
//
// Oracle: equal on LOGICAL elements. ttprm's gather is byte-identical on whole pages
// but TTNN leaves padding lanes uninitialized, so only live elements are compared.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>

#include <ttnn/tensor/tensor.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/operations/data_movement/reshape_view/reshape.hpp>
#include <ttnn/operations/normalization/layernorm/layernorm.hpp>
#include <ttnn/operations/normalization/rmsnorm/rmsnorm.hpp>

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::add;
using ttprm::fma;
using ttprm::l2_norm;
using ttprm::layer_norm;
using ttprm::lerp;
using ttprm::mul;
using ttprm::mul_l2_norm;
using ttprm::rk;
using ttprm::realize;
using ttprm::rms_norm;
using ttprm::sub;
using ttprm::View;
using ttprm::view_of;
using ttprm::view_of_shape;

namespace {
int g_pass = 0, g_fail = 0;
void check(const char* name, bool ok, const char* detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name, detail[0] ? " -- " : "", detail);
    if (ok) ++g_pass; else ++g_fail;
}

inline float val(int64_t k) { return (float)((k % 241) - 120); }

TensorSpec tspec(int64_t rows, int64_t cols) {
    return TensorSpec(ttnn::Shape({(uint32_t)rows, (uint32_t)cols}),
                      TensorLayout(DataType::BFLOAT16,
                                   PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
}

// A [rows,cols] device tensor with logical element (r,c) = val(r*cols+c).
ttnn::Tensor make_input(distributed::MeshDevice* dev, int64_t rows, int64_t cols) {
    std::vector<float> h((size_t)rows * cols);
    for (int64_t i = 0; i < rows * cols; i++) h[i] = val(i);
    return ttnn::Tensor::from_vector(h, tspec(rows, cols), dev);
}

std::vector<float> logical(const ttnn::Tensor& t) { return t.to_vector<float>(); }

bool exact(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) if (a[i] != b[i]) return false;
    return true;
}

// Realize `v` over `in`, expect it to FUSE, and assert the live logical output equals
// `oracle` (a vector produced by a TTNN op chain or a first-principles reference).
void expect_fused(const char* name, const View& v, const std::vector<float>& oracle) {
    auto r = realize(v);
    if (!r) { check(name, false, r.error().c_str()); return; }
    bool ok = exact(logical(r.value()), oracle);
    check(name, ok, ok ? "matches TTNN/oracle" : "MISMATCH vs oracle");
}

void expect_reject(const char* name, const View& v) {
    auto r = realize(v);
    check(name, !r, r ? "expected REJECT but fused" : r.error().c_str());
}

// First-principles reference for a contiguous flatten+reshape of a [1,N] live row:
// the logical sequence is just val(0..N-1) (row-major identity), independent of shape.
std::vector<float> seq_ref(int64_t n) {
    std::vector<float> o((size_t)n);
    for (int64_t i = 0; i < n; i++) o[i] = val(i);
    return o;
}

std::vector<float> add_vec(const std::vector<float>& a, const std::vector<float>& b) {
    std::vector<float> o(a.size());
    for (size_t i = 0; i < a.size(); i++) o[i] = a[i] + b[i];
    return o;
}

// A second device tensor, values shifted so the two operands are distinguishable. Integer-
// valued and |sum| < 256, so the bf16 add is EXACT -> exact() comparison is valid.
ttnn::Tensor make_input2(distributed::MeshDevice* dev, int64_t rows, int64_t cols) {
    std::vector<float> h((size_t)rows * cols);
    for (int64_t i = 0; i < rows * cols; i++) h[i] = (float)((i % 97) - 48);  // -48..48
    return ttnn::Tensor::from_vector(h, tspec(rows, cols), dev);
}

// A device tensor whose logical element i = fn(i). Used to keep fma/lerp operands SMALL so the
// bf16 result is an exact integer (|a*b+c| < 256, |a+w*(b-a)| < 256) -> exact() comparison holds.
template <typename Fn>
ttnn::Tensor make_fn(distributed::MeshDevice* dev, int64_t rows, int64_t cols, Fn fn) {
    std::vector<float> h((size_t)rows * cols);
    for (int64_t i = 0; i < rows * cols; i++) h[i] = (float)fn(i);
    return ttnn::Tensor::from_vector(h, tspec(rows, cols), dev);
}

// Relative comparison for float ops (mul) where the bf16 result is rounded, not exact.
bool close(const std::vector<float>& a, const std::vector<float>& b, float rel = 0.02f) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (std::abs(a[i] - b[i]) > rel * std::max(1.0f, std::abs(b[i]))) return false;
    return true;
}
}  // namespace

int main() {
    std::printf(">>> view_ttnn_test (ttprm::realize vs chained TTNN ops)\n");
    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    dev->enable_program_cache();  // prove the cached device_operation reuses across calls

    // ============ output memory_config forwarding ============
    {
        const MemoryConfig l1(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
        auto a = make_input(dev.get(), 32, 32);
        auto b = make_input2(dev.get(), 32, 32);

        auto rr = realize(view_of(a), std::nullopt, l1);
        check("realize forwards memory_config=L1", rr && rr.value().memory_config().buffer_type() == BufferType::L1);

        auto ar = add(view_of(a), view_of(b), std::nullopt, l1);
        check("add forwards memory_config=L1", ar && ar.value().memory_config().buffer_type() == BufferType::L1);

        auto nr = l2_norm(view_of(a), 1e-6f, std::nullopt, l1);
        check("l2_norm forwards memory_config=L1", nr && nr.value().memory_config().buffer_type() == BufferType::L1);
    }

    // ============ reshape / flatten chains -- oracle = ttnn::reshape ============
    {
        // [1,4096] -> [64,64] (the RWKV head reshape). TTNN oracle = ttnn::reshape.
        auto in = make_input(dev.get(), 1, 4096);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({64, 64})));
        View v = view_of(in).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        expect_fused("[1,4096]->[64,64] vs ttnn::reshape", v, oracle);
    }
    {
        // [1,4096] -> [128,32] (folded). TTNN oracle = ttnn::reshape.
        auto in = make_input(dev.get(), 1, 4096);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({128, 32})));
        View v = view_of(in).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({128, 32});
        expect_fused("[1,4096]->[128,32] vs ttnn::reshape", v, oracle);
    }
    {
        // chained: [1,4096] -> [64,64] -> [128,32]. Oracle = ttnn::reshape to final.
        auto in = make_input(dev.get(), 1, 4096);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({128, 32})));
        View v = view_of(in).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64}).reshape({128, 32});
        expect_fused("[1,4096]->[64,64]->[128,32] chained vs ttnn::reshape", v, oracle);
    }
    {
        // identity [64,64] (TILE verbatim). Oracle = the input itself.
        auto in = make_input(dev.get(), 64, 64);
        View v = view_of(in);
        expect_fused("identity [64,64]", v, logical(in));
    }
    {
        // a wider flatten chain: [2,2048] -> [4096] -> [64,64]. Oracle = ttnn::reshape.
        auto in = make_input(dev.get(), 2, 2048);   // padded to [32,2048]
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({64, 64})));
        // NB: source has 2 LIVE rows in a 32-row tile; a flatten of [2,2048] is NOT the
        // first 4096 contiguous physical elements -> this stresses multi-row slice+reshape.
        View v = view_of(in).slice({{0, 2, 1}, {0, 2048, 1}}).reshape({64, 64});
        expect_fused("[2,2048]->[64,64] vs ttnn::reshape", v, oracle);
    }

    // ============ slice chains -- oracle = first-principles sequence ============
    {
        // full-width tile-aligned row slice [32:96] of [128,64], then reshape [128,32].
        // Live elements = val(32*64 .. 96*64) reshaped -> contiguous sub-sequence.
        auto in = make_input(dev.get(), 128, 64);
        std::vector<float> oracle((size_t)64 * 64);
        for (int64_t i = 0; i < 64 * 64; i++) oracle[i] = val(32 * 64 + i);
        View v = view_of(in).slice({{32, 96, 1}, {0, 64, 1}}).reshape({128, 32});
        expect_fused("slice rows[32:96] full-width -> [128,32]", v, oracle);
    }
    {
        // face-aligned col slice [16:48] of [64,64] -> [64,32]. Live (r,c)=val(r*64+16+c).
        auto in = make_input(dev.get(), 64, 64);
        std::vector<float> oracle((size_t)64 * 32);
        for (int64_t r = 0; r < 64; r++)
            for (int64_t c = 0; c < 32; c++) oracle[r * 32 + c] = val(r * 64 + 16 + c);
        View v = view_of(in).slice({{0, 64, 1}, {16, 48, 1}});
        expect_fused("col slice [16:48] of [64,64]", v, oracle);
    }

    // ============ nastier chains -- try to break the algebra ====================
    {
        // rank-3 reshape roundtrip: [1,4096] -> [4,16,64]. Row-major contiguous, so the
        // logical sequence is unchanged -> oracle = ttnn::reshape to the flat [64,64].
        auto in = make_input(dev.get(), 1, 4096);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({64, 64})));
        View v = view_of(in).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({4, 16, 64});
        expect_fused("[1,4096]->[4,16,64] rank-3 vs ttnn::reshape", v, oracle);
    }
    {
        // flatten all the way to 1D [4096]. Co=4096, Ro=1 -> output [1,4096] padded.
        auto in = make_input(dev.get(), 64, 64);
        auto oracle = logical(in);   // identity logical sequence
        View v = view_of(in).reshape({4096});
        expect_fused("[64,64]->[4096] flatten-1D", v, oracle);
    }
    {
        // UNFOLD via reshape: source is FOLDED [dim/32,32]; reshape back to flat [1,dim].
        // out[0][j] = folded[i][k] with j = i*32+k. The GATHER mirror of the [1,4096]->[128,32]
        // fold above (inverse of the [128,32]->[1,4096] SCATTER below). Oracle = ttnn::reshape
        // of the folded source back to flat. dim=4096 -> folded rows 128 = 4 tiles (tile-aligned).
        const int64_t dim = 4096;
        auto in = make_input(dev.get(), dim / 32, 32);   // folded [128,32]
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({1, dim})));
        View v = view_of(in).reshape({1, dim});
        expect_fused("UNFOLD folded[128,32]->[1,4096] vs ttnn::reshape", v, oracle);
    }
    {
        // UNFOLD with a MISALIGNED folded-row count -- regression guard for the padded-anchor bug.
        // dim=512 -> folded [16,32]: 16 logical rows pad up to a full 32-row tile. The OLD View
        // anchored the algebra on the PADDED [32,32] grid, so reshape({1,512}) saw the padded
        // element count (32*32 != 512), poisoned the strides, and spuriously REJECTed -- even
        // though the unfold is a realizable face-row gather. Fixed by anchoring the iteration on
        // the LOGICAL extent (dense-packed tensors) while addressing on the padded grid.
        const int64_t dim = 512;
        auto in = make_input(dev.get(), dim / 32, 32);   // folded [16,32], pads to [32,32]
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({1, dim})));
        View v = view_of(in).reshape({1, dim});
        expect_fused("UNFOLD folded[16,32]->[1,512] (misaligned rows) vs ttnn::reshape", v, oracle);
    }
    {
        // strided row slice: rows [0:128:2] of [128,64] -> [64,64]. Non-unit OUTER stride;
        // out (r,c) = val((2r)*64 + c). Host reference (ttnn::slice step semantics vary).
        auto in = make_input(dev.get(), 128, 64);
        std::vector<float> oracle((size_t)64 * 64);
        for (int64_t r = 0; r < 64; r++)
            for (int64_t c = 0; c < 64; c++) oracle[r * 64 + c] = val((2 * r) * 64 + c);
        View v = view_of(in).slice({{0, 128, 2}, {0, 64, 1}});
        expect_fused("strided row slice [0:128:2] of [128,64]", v, oracle);
    }
    {
        // double slice: rows[32:96] then cols[16:48] of [128,64] -> [64,32].
        // out (r,c) = val((32+r)*64 + 16 + c).
        auto in = make_input(dev.get(), 128, 64);
        std::vector<float> oracle((size_t)64 * 32);
        for (int64_t r = 0; r < 64; r++)
            for (int64_t c = 0; c < 32; c++) oracle[r * 32 + c] = val((32 + r) * 64 + 16 + c);
        View v = view_of(in).slice({{32, 96, 1}, {0, 64, 1}}).slice({{0, 64, 1}, {16, 48, 1}});
        expect_fused("double slice rows[32:96] cols[16:48]", v, oracle);
    }
    {
        // reshape THEN slice: [1,4096]->[64,64], then col slice [0:32]. out (r,c)=val(r*64+c).
        auto in = make_input(dev.get(), 1, 4096);
        std::vector<float> oracle((size_t)64 * 32);
        for (int64_t r = 0; r < 64; r++)
            for (int64_t c = 0; c < 32; c++) oracle[r * 32 + c] = val(r * 64 + c);
        View v = view_of(in).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64}).slice({{0, 64, 1}, {0, 32, 1}});
        expect_fused("[1,4096]->[64,64] then col slice [0:32]", v, oracle);
    }

    {
        // BIG, multi-core: [1,16384] -> [128,128] (16 output tiles -> splits across cores).
        // Stresses the device_operation core partition + override_runtime_arguments path.
        auto in = make_input(dev.get(), 1, 16384);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({128, 128})));
        View v = view_of(in).slice({{0, 1, 1}, {0, 16384, 1}}).reshape({128, 128});
        expect_fused("[1,16384]->[128,128] multi-core vs ttnn::reshape", v, oracle);
    }

    // ============ SCATTER: dense input -> VIEWED output (affine WRITE path) ======
    // The mirror of realize: input is dense, the OUTPUT is viewed. Exercises the affine
    // scatter endpoint (OUT_KIND=AFFINE). out_view is over the output's padded grid,
    // reshaped to the input's tile grid -> the result is the inverse relayout.
    {
        // dense [128,32] scattered into [1,4096] viewed-as-[128,32] == ttnn::reshape to flat.
        auto in = make_input(dev.get(), 128, 32);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({1, 4096})));
        View ov = ttprm::view_of_shape(1, 4096).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({128, 32});
        auto r = realize(view_of(in), ov);
        if (!r) check("scatter [128,32]->[1,4096]", false, r.error().c_str());
        else check("scatter [128,32]->[1,4096] vs ttnn::reshape",
                   exact(logical(r.value()), oracle), "matches TTNN/oracle");
    }
    {
        // dense [64,64] scattered into [1,4096] viewed-as-[64,64] (head inverse).
        auto in = make_input(dev.get(), 64, 64);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({1, 4096})));
        View ov = ttprm::view_of_shape(1, 4096).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        auto r = realize(view_of(in), ov);
        if (!r) check("scatter [64,64]->[1,4096]", false, r.error().c_str());
        else check("scatter [64,64]->[1,4096] vs ttnn::reshape",
                   exact(logical(r.value()), oracle), "matches TTNN/oracle");
    }

    // ============ BOUND OUT: realize scatters into a preallocated dst (no alloc) =====
    // out view carries a real tensor -> the writer scatters into THAT buffer in place; the
    // returned handle IS dst, with its true logical metadata (NOT a fabricated padded shape).
    {
        // gather [1,4096]->[64,64] into a preallocated [64,64] dst.
        auto in = make_input(dev.get(), 1, 4096);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({64, 64})));
        auto dst = make_input(dev.get(), 64, 64);   // preexisting contents, to be overwritten
        View v  = view_of(in).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        auto r = realize(v, view_of(dst));
        if (!r) { check("bound-out gather [1,4096]->[64,64]", false, r.error().c_str()); }
        else {
            bool same_handle = r.value().logical_shape() == dst.logical_shape();
            bool wrote_dst = exact(logical(dst), oracle);            // dst mutated in place
            bool meta_ok = (r.value().logical_shape()[0] == 64u &&
                            r.value().logical_shape()[1] == 64u);    // logical, not padded
            check("bound-out gather [1,4096]->[64,64]",
                  same_handle && wrote_dst && meta_ok && exact(logical(r.value()), oracle),
                  "scatters into dst, logical metadata preserved");
        }
    }
    {
        // BOUND SCATTER: dense [128,32] in, scattered into a preallocated [1,4096] dst via a
        // reshape-map out view (rows()/cols() stay the dst's [1,4096]; layout() carries the map).
        auto in = make_input(dev.get(), 128, 32);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({1, 4096})));
        auto dst = make_input(dev.get(), 1, 4096);
        View ov = view_of(dst).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({128, 32});
        auto r = realize(view_of(in), ov);
        if (!r) { check("bound scatter [128,32]->dst[1,4096]", false, r.error().c_str()); }
        else {
            bool wrote_dst = exact(logical(dst), oracle);
            check("bound scatter [128,32]->dst[1,4096]",
                  wrote_dst && exact(logical(r.value()), oracle), "scatters into preallocated dst");
        }
    }
    {
        // non-bound out (view_of_shape) still allocates -> a [1,N] result keeps logical [1,N].
        auto in = make_input(dev.get(), 64, 64);
        auto oracle = logical(ttnn::reshape(in, ttnn::Shape({1, 4096})));
        View ov = ttprm::view_of_shape(1, 4096).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        auto r = realize(view_of(in), ov);
        bool meta_ok = r && r.value().logical_shape()[0] == 1u &&
                       r.value().logical_shape()[1] == 4096u;
        check("unbound out keeps logical [1,4096]",
              r && meta_ok && exact(logical(r.value()), oracle), "alloc path metadata");
    }

    // ============ FUSED ADD: out = a_view(a) + b_view(b), no relayout realized ====
    // The GGML idiom and its kin. Inputs are integer-valued with |sum| < 256 so the bf16
    // add is exact; oracle = elementwise a_logical + b_logical (head/flat orders coincide).
    {
        // a[1,4096] viewed as [64,64] (AFFINE) + b[64,64] (DENSE) -> out[64,64] (DENSE).
        // out[h][w] = a_flat[h*64+w] + b[h][w].  The canonical fused add-view.
        auto a = make_input(dev.get(), 1, 4096);
        auto b = make_input2(dev.get(), 64, 64);
        View av = view_of(a).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        View bv = view_of(b);
        View ov = view_of_shape(64, 64);
        auto r = add(av, bv, ov);
        if (!r) { check("add a[1,4096]-view + b[64,64] -> [64,64]", false, r.error().c_str()); }
        else check("add a[1,4096]-view + b[64,64] -> [64,64]",
                   exact(logical(r.value()), add_vec(logical(a), logical(b))), "matches oracle");
    }
    {
        // symmetric: a[64,64] (DENSE) + b[1,4096] viewed as [64,64] (AFFINE) -> [64,64].
        auto a = make_input(dev.get(), 64, 64);
        auto b = make_input2(dev.get(), 1, 4096);
        View av = view_of(a);
        View bv = view_of(b).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        View ov = view_of_shape(64, 64);
        auto r = add(av, bv, ov);
        if (!r) { check("add a[64,64] + b[1,4096]-view -> [64,64]", false, r.error().c_str()); }
        else check("add a[64,64] + b[1,4096]-view -> [64,64]",
                   exact(logical(r.value()), add_vec(logical(a), logical(b))), "matches oracle");
    }
    {
        // viewed OUTPUT (AFFINE scatter): a[64,64] + b[64,64], result STORED into flat [1,4096].
        // out_flat[h*64+w] = a[h][w] + b[h][w]. Exercises the affine WRITE endpoint in the fusion.
        auto a = make_input(dev.get(), 64, 64);
        auto b = make_input2(dev.get(), 64, 64);
        View av = view_of(a);
        View bv = view_of(b);
        View ov = view_of_shape(1, 4096).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        auto r = add(av, bv, ov);
        if (!r) { check("add a[64,64] + b[64,64] -> store flat[1,4096]", false, r.error().c_str()); }
        else check("add a[64,64] + b[64,64] -> store flat[1,4096]",
                   exact(logical(r.value()), add_vec(logical(a), logical(b))), "matches oracle");
    }
    {
        // BIG multi-core: a[1,16384] viewed [128,128] + b[128,128] -> [128,128] (16 tiles).
        auto a = make_input(dev.get(), 1, 16384);
        auto b = make_input2(dev.get(), 128, 128);
        View av = view_of(a).slice({{0, 1, 1}, {0, 16384, 1}}).reshape({128, 128});
        View bv = view_of(b);
        View ov = view_of_shape(128, 128);
        auto r = add(av, bv, ov);
        if (!r) { check("add a[1,16384]-view + b[128,128] -> [128,128] multi-core", false, r.error().c_str()); }
        else check("add a[1,16384]-view + b[128,128] -> [128,128] multi-core",
                   exact(logical(r.value()), add_vec(logical(a), logical(b))), "matches oracle");
    }

    // ============ sub / mul (the #ifdef TTPRM_BINOP extension) ===================
    {
        // sub: a[1,4096] viewed [64,64] - b[64,64]. out[h][w] = a_flat[h*64+w] - b[h][w]. exact.
        auto a = make_input(dev.get(), 1, 4096);
        auto b = make_input2(dev.get(), 64, 64);
        View av = view_of(a).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        auto r = sub(av, view_of(b), view_of_shape(64, 64));
        std::vector<float> oracle(4096);
        auto al = logical(a), bl = logical(b);
        for (size_t i = 0; i < 4096; i++) oracle[i] = al[i] - bl[i];
        if (!r) check("sub a[1,4096]-view - b[64,64]", false, r.error().c_str());
        else check("sub a[1,4096]-view - b[64,64]", exact(logical(r.value()), oracle), "matches oracle");
    }
    {
        // mul: a[1,4096] viewed [64,64] * b[64,64]. bf16 product -> close() (not exact).
        auto a = make_input(dev.get(), 1, 4096);
        auto b = make_input2(dev.get(), 64, 64);
        View av = view_of(a).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        auto r = mul(av, view_of(b), view_of_shape(64, 64));
        std::vector<float> oracle(4096);
        auto al = logical(a), bl = logical(b);
        for (size_t i = 0; i < 4096; i++) oracle[i] = al[i] * bl[i];
        if (!r) check("mul a[1,4096]-view * b[64,64]", false, r.error().c_str());
        else check("mul a[1,4096]-view * b[64,64]", close(logical(r.value()), oracle), "matches oracle (rel)");
    }

    // ============ BOUND OUT across the op surface (add / fma / l2_norm) ===========
    // Every op resolves its output through the shared resolve_output: a bound out (view_of(dst))
    // scatters into dst in place; the returned handle IS dst with its true logical metadata.
    {
        // add into a preallocated [64,64] dst.
        auto a = make_input(dev.get(), 1, 4096);
        auto b = make_input2(dev.get(), 64, 64);
        auto dst = make_input(dev.get(), 64, 64);
        View av = view_of(a).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        auto r = add(av, view_of(b), view_of(dst));
        std::vector<float> oracle(4096);
        auto al = logical(a), bl = logical(b);
        for (size_t i = 0; i < 4096; i++) oracle[i] = al[i] + bl[i];
        bool ok = r && exact(logical(dst), oracle) && exact(logical(r.value()), oracle);
        check("bound add -> dst[64,64]", ok, r ? "scatters into dst" : r.error().c_str());
    }
    {
        // fma into a preallocated [64,64] dst: out = a*b + c.
        auto a = make_fn(dev.get(), 64, 64, [](int64_t i){ return (i % 5) - 2; });
        auto b = make_fn(dev.get(), 64, 64, [](int64_t i){ return (i % 3) - 1; });
        auto c = make_fn(dev.get(), 64, 64, [](int64_t i){ return (i % 7) - 3; });
        auto dst = make_input(dev.get(), 64, 64);
        auto r = fma(view_of(a), view_of(b), view_of(c), view_of(dst));
        std::vector<float> oracle(4096);
        auto al = logical(a), bl = logical(b), cl = logical(c);
        for (size_t i = 0; i < 4096; i++) oracle[i] = al[i] * bl[i] + cl[i];
        bool ok = r && exact(logical(dst), oracle) && exact(logical(r.value()), oracle);
        check("bound fma -> dst[64,64]", ok, r ? "scatters into dst" : r.error().c_str());
    }
    {
        // l2_norm into a preallocated [64,64] dst (PCC vs the unbound result).
        auto x = make_input(dev.get(), 64, 64);
        auto ref = l2_norm(view_of(x));                 // unbound (allocated) reference
        auto dst = make_input(dev.get(), 64, 64);
        auto r = l2_norm(view_of(x), 1e-6f, view_of(dst));
        bool ok = r && ref && exact(logical(dst), logical(r.value())) &&
                  exact(logical(r.value()), logical(ref.value()));
        check("bound l2_norm -> dst[64,64]", ok, r ? "scatters into dst" : (r ? "" : r.error().c_str()));
    }

    // ============ GROUP BROADCAST (the RWKV7 REPEAT pattern, tested in add) =======
    // a is one group [Rg,C]; b/out are G stacked groups [G*Rg,C]. a is auto-detected as broadcast
    // (its tile grid divides the output) and re-fed across the G row-groups -- nothing materialized.
    // out[g*Rg+r, c] = a[r,c] + b[g*Rg+r, c].
    {
        const int64_t Rg = 32, C = 64, G = 3;  // a:[32,64]=2t, b/out:[96,64]=6t, tpg=2, G=3
        auto a = make_input(dev.get(), Rg, C);
        auto b = make_input2(dev.get(), G * Rg, C);
        auto r = add(view_of(a), view_of(b), view_of_shape(G * Rg, C));
        auto al = logical(a), bl = logical(b);
        std::vector<float> oracle((size_t)G * Rg * C);
        for (int64_t R = 0; R < G * Rg; R++)
            for (int64_t c = 0; c < C; c++)
                oracle[R * C + c] = al[(R % Rg) * C + c] + bl[R * C + c];
        if (!r) check("bcast add a[32,64] (x3 groups) + b[96,64]", false, r.error().c_str());
        else check("bcast add a[32,64] (x3 groups) + b[96,64]",
                   exact(logical(r.value()), oracle), "matches oracle");
    }
    {
        // bigger group broadcast (multi-tile groups, multi-core): a[64,64]=4t broadcast over
        // 4 groups -> b/out [256,64]=16t, tpg=4, G=4. Stresses the within-group counter + reset.
        const int64_t Rg = 64, C = 64, G = 4;
        auto a = make_input(dev.get(), Rg, C);
        auto b = make_input2(dev.get(), G * Rg, C);
        auto r = add(view_of(a), view_of(b), view_of_shape(G * Rg, C));
        auto al = logical(a), bl = logical(b);
        std::vector<float> oracle((size_t)G * Rg * C);
        for (int64_t R = 0; R < G * Rg; R++)
            for (int64_t c = 0; c < C; c++)
                oracle[R * C + c] = al[(R % Rg) * C + c] + bl[R * C + c];
        if (!r) check("bcast add a[64,64] (x4 groups) + b[256,64]", false, r.error().c_str());
        else check("bcast add a[64,64] (x4 groups) + b[256,64]",
                   exact(logical(r.value()), oracle), "matches oracle");
    }

    {
        // BOTH operands broadcast: a[32,64]=2t and b[32,64]=2t each repeated over G=3 groups ->
        // out[96,64]=6t = three identical copies of (a+b). Stresses two resident groups at once
        // (the RWKV lerp shape, where 2 of 3 operands are broadcast). tpg=2, G=3.
        const int64_t Rg = 32, C = 64, G = 3;
        auto a = make_input(dev.get(), Rg, C);
        auto b = make_input2(dev.get(), Rg, C);
        auto r = add(view_of(a), view_of(b), view_of_shape(G * Rg, C));
        auto al = logical(a), bl = logical(b);
        std::vector<float> oracle((size_t)G * Rg * C);
        for (int64_t R = 0; R < G * Rg; R++)
            for (int64_t c = 0; c < C; c++)
                oracle[R * C + c] = al[(R % Rg) * C + c] + bl[(R % Rg) * C + c];
        if (!r) check("bcast add a[32,64] + b[32,64] BOTH bcast (x3)", false, r.error().c_str());
        else check("bcast add a[32,64] + b[32,64] BOTH bcast (x3)",
                   exact(logical(r.value()), oracle), "matches oracle");
    }

    // ============ FUSED TERNARY: fma (a*b+c) and lerp (a+w*(b-a)) =================
    // Small integer operands so the bf16 result is an exact integer -> exact() comparison.
    {
        // fma: a[1,4096] viewed [64,64] (AFFINE) * b[64,64] (DENSE) + c[64,64] (DENSE) -> [64,64].
        // out[h][w] = a_flat[h*64+w]*b[h][w] + c[h][w].  a,b in 0..7, c in 0..63 -> result < 256.
        auto a = make_fn(dev.get(), 1, 4096, [](int64_t i) { return i % 8; });
        auto b = make_fn(dev.get(), 64, 64, [](int64_t i) { return i % 7; });
        auto c = make_fn(dev.get(), 64, 64, [](int64_t i) { return i % 64; });
        View av = view_of(a).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        auto r = fma(av, view_of(b), view_of(c), view_of_shape(64, 64));
        auto al = logical(a), bl = logical(b), cl = logical(c);
        std::vector<float> oracle(4096);
        for (size_t i = 0; i < 4096; i++) oracle[i] = al[i] * bl[i] + cl[i];
        if (!r) check("fma a[1,4096]-view * b[64,64] + c[64,64]", false, r.error().c_str());
        else check("fma a[1,4096]-view * b[64,64] + c[64,64]",
                   exact(logical(r.value()), oracle), "matches oracle");
    }
    {
        // lerp: a[64,64] (DENSE) + w*(b[1,4096]-view[64,64] - a). a,b in 0..7, w in 0..3 -> exact.
        // out[h][w] = a[h][w] + w_[h][w]*(b_flat[h*64+w] - a[h][w]).
        auto a = make_fn(dev.get(), 64, 64, [](int64_t i) { return i % 8; });
        auto b = make_fn(dev.get(), 1, 4096, [](int64_t i) { return i % 7; });
        auto w = make_fn(dev.get(), 64, 64, [](int64_t i) { return i % 4; });
        View bv = view_of(b).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
        auto r = lerp(view_of(a), bv, view_of(w), view_of_shape(64, 64));
        auto al = logical(a), bl = logical(b), wl = logical(w);
        std::vector<float> oracle(4096);
        for (size_t i = 0; i < 4096; i++) oracle[i] = al[i] + wl[i] * (bl[i] - al[i]);
        if (!r) check("lerp a[64,64] + w*(b[1,4096]-view - a)", false, r.error().c_str());
        else check("lerp a[64,64] + w*(b[1,4096]-view - a)",
                   exact(logical(r.value()), oracle), "matches oracle");
    }
    {
        // fma with ONE broadcast operand: a[32,64]=2t broadcast over G=3 groups, b/c[96,64]=6t full.
        // out[g*32+r, col] = a[r,col]*b[g*32+r,col] + c[g*32+r,col]. tpg=2, G=3.
        const int64_t Rg = 32, C = 64, G = 3;
        auto a = make_fn(dev.get(), Rg, C, [](int64_t i) { return i % 8; });
        auto b = make_fn(dev.get(), G * Rg, C, [](int64_t i) { return i % 7; });
        auto c = make_fn(dev.get(), G * Rg, C, [](int64_t i) { return i % 64; });
        auto r = fma(view_of(a), view_of(b), view_of(c), view_of_shape(G * Rg, C));
        auto al = logical(a), bl = logical(b), cl = logical(c);
        std::vector<float> oracle((size_t)G * Rg * C);
        for (int64_t R = 0; R < G * Rg; R++)
            for (int64_t col = 0; col < C; col++)
                oracle[R * C + col] = al[(R % Rg) * C + col] * bl[R * C + col] + cl[R * C + col];
        if (!r) check("fma bcast a[32,64] (x3) * b[96,64] + c[96,64]", false, r.error().c_str());
        else check("fma bcast a[32,64] (x3) * b[96,64] + c[96,64]",
                   exact(logical(r.value()), oracle), "matches oracle");
    }
    {
        // lerp the RWKV shape: TWO broadcast operands. a[32,64] & b[32,64] broadcast over G=3, w[96,64]
        // full -> out[g*32+r,col] = a[r,col] + w[g*32+r,col]*(b[r,col]-a[r,col]). Two resident groups.
        const int64_t Rg = 32, C = 64, G = 3;
        auto a = make_fn(dev.get(), Rg, C, [](int64_t i) { return i % 8; });
        auto b = make_fn(dev.get(), Rg, C, [](int64_t i) { return i % 7; });
        auto w = make_fn(dev.get(), G * Rg, C, [](int64_t i) { return i % 4; });
        auto r = lerp(view_of(a), view_of(b), view_of(w), view_of_shape(G * Rg, C));
        auto al = logical(a), bl = logical(b), wl = logical(w);
        std::vector<float> oracle((size_t)G * Rg * C);
        for (int64_t R = 0; R < G * Rg; R++)
            for (int64_t col = 0; col < C; col++) {
                const float av = al[(R % Rg) * C + col], bv = bl[(R % Rg) * C + col];
                oracle[R * C + col] = av + wl[R * C + col] * (bv - av);
            }
        if (!r) check("lerp BOTH-bcast a[32,64]+b[32,64] (x3), w[96,64] full", false, r.error().c_str());
        else check("lerp BOTH-bcast a[32,64]+b[32,64] (x3), w[96,64] full",
                   exact(logical(r.value()), oracle), "matches oracle");
    }
    {
        // ROW-BROADCAST decode shape (the RWKV token-shift at n_tokens=1): a SUB-TILE group bcast.
        // cur/x_prev are a single logical row [1,C] replicated DOWN G(<32) output rows in ONE tile
        // (not a whole-tile group); w[G,C] is full. out[g,col] = a[0,col] + w[g,col]*(b[0,col]-a[0,col]).
        // This regresses the "fix in L1": widen the single-row gather guard so row 0 fills the tile.
        const int64_t C = 64, G = 6;
        auto a = make_fn(dev.get(), 1, C, [](int64_t i) { return i % 8; });
        auto b = make_fn(dev.get(), 1, C, [](int64_t i) { return i % 7; });
        auto w = make_fn(dev.get(), G, C, [](int64_t i) { return i % 4; });
        auto r = lerp(view_of(a), view_of(b), view_of(w), view_of_shape(G, C));
        auto al = logical(a), bl = logical(b), wl = logical(w);
        std::vector<float> oracle((size_t)G * C);
        for (int64_t g = 0; g < G; g++)
            for (int64_t col = 0; col < C; col++)
                oracle[g * C + col] = al[col] + wl[g * C + col] * (bl[col] - al[col]);
        if (!r) check("lerp ROW-bcast cur/x_prev[1,64] (decode, x6 rows), w[6,64]", false, r.error().c_str());
        else check("lerp ROW-bcast cur/x_prev[1,64] (decode, x6 rows), w[6,64]",
                   exact(logical(r.value()), oracle), "matches oracle");
    }
    {
        // ROW-BROADCAST via add (the binary path shares the same classifier). a[1,128] replicated
        // down b[10,128]'s rows. out[r,col] = a[0,col] + b[r,col].  (10 rows -> still sub-tile.)
        const int64_t C = 128, R = 10;
        auto a = make_fn(dev.get(), 1, C, [](int64_t i) { return i % 9; });
        auto b = make_input2(dev.get(), R, C);
        auto r = add(view_of(a), view_of(b), view_of_shape(R, C));
        auto al = logical(a), bl = logical(b);
        std::vector<float> oracle((size_t)R * C);
        for (int64_t row = 0; row < R; row++)
            for (int64_t col = 0; col < C; col++) oracle[row * C + col] = al[col] + bl[row * C + col];
        if (!r) check("add ROW-bcast a[1,128] (x10 rows) + b[10,128]", false, r.error().c_str());
        else check("add ROW-bcast a[1,128] (x10 rows) + b[10,128]",
                   exact(logical(r.value()), oracle), "matches oracle");
    }
    {
        // PREFILL token-shift (the RWKV time-mix at n_tokens>1): SUB-TILE grouped broadcast. cur/x_prev
        // are [nt,C] repeated over G groups (source row = r mod nt); w is [G,C] stretched over nt tokens
        // (source row = r div nt); group-major out[g*nt+t] = cur[t] + w[g]*(x_prev[t]-cur[t]). The
        // grouped-row odometer (GroupedTensorView) gathers it with NO relayout -- exact integer oracle.
        const int64_t C = 64, nt = 8, G = 6;   // out [48,64], groups span 8<32 rows (sub-tile)
        auto cur = make_fn(dev.get(), nt, C, [](int64_t i) { return i % 8; });
        auto xpv = make_fn(dev.get(), nt, C, [](int64_t i) { return i % 7; });
        auto w   = make_fn(dev.get(), G,  C, [](int64_t i) { return i % 4; });
        auto r = lerp(view_of(cur), view_of(xpv), view_of(w), view_of_shape(G * nt, C));
        auto cl = logical(cur), xl = logical(xpv), wl = logical(w);
        std::vector<float> oracle((size_t)G * nt * C);
        for (int64_t g = 0; g < G; g++)
            for (int64_t t = 0; t < nt; t++)
                for (int64_t col = 0; col < C; col++) {
                    const float a = cl[t * C + col], bb = xl[t * C + col];
                    oracle[(g * nt + t) * C + col] = a + wl[g * C + col] * (bb - a);
                }
        if (!r) check("lerp PREFILL grouped cur/x_prev[8,64](mod) w[6,64](div) -> [48,64]", false, r.error().c_str());
        else check("lerp PREFILL grouped cur/x_prev[8,64](mod) w[6,64](div) -> [48,64]",
                   exact(logical(r.value()), oracle), "matches oracle");
    }
    {
        // NEGATIVE: a sub-tile broadcast that is NOT the lerp token-shift (here a plain add) must
        // REJECT, not silently mis-map -- a[8,64] repeated into [24,64] (same tile count, so the
        // tile-divisor test alone would miss it).
        const int64_t Rg = 8, C = 64, G = 3;
        auto a = make_fn(dev.get(), Rg, C, [](int64_t i) { return i % 8; });
        auto b = make_input2(dev.get(), G * Rg, C);
        auto r = add(view_of(a), view_of(b), view_of_shape(G * Rg, C));
        check("sub-tile bcast add a[8,64] (x3) REJECT", !r, r ? "expected REJECT but fused" : r.error().c_str());
    }

    // ============ REGRESSION: view live-extent vs original-tensor / padded grid ============
    // These guard the op-layer instances of the same "reason on the wrong extent" class that the
    // logical-anchor fix killed in the algebra: classify_operand read the ORIGINAL tensor's rows
    // (View::rows(), stale after slice) and the shape checks compared the PADDED grid.
    {
        // BUG (silent zeros): a SLICED single-row operand row-broadcast. a is row 0 of a [1,256]
        // tensor, sliced to width 128, then broadcast down b's 10 rows. Pre-fix, the slice made
        // is_plain_source false (Co 128 != Cs 256), the ROW-BROADCAST branch was skipped, the
        // operand was mis-classed FULL, and device rows 1..9 came out as 0+b instead of a[0]+b.
        const int64_t C = 128, R = 10;
        auto a_large = make_fn(dev.get(), 1, 256, [](int64_t i) { return i % 9; });
        auto b = make_input2(dev.get(), R, C);
        auto av = view_of(a_large).slice({{0, 1, 1}, {0, C, 1}});   // logical [1,128], base 0, Cs 256
        auto r = add(av, view_of(b), view_of_shape(R, C));
        auto al = logical(a_large), bl = logical(b);
        std::vector<float> oracle((size_t)R * C);
        for (int64_t row = 0; row < R; row++)
            for (int64_t col = 0; col < C; col++) oracle[row * C + col] = al[col] + bl[row * C + col];
        if (!r) check("REGRESSION sliced ROW-bcast a[1,256][:128] (x10) + b[10,128]", false, r.error().c_str());
        else check("REGRESSION sliced ROW-bcast a[1,256][:128] (x10) + b[10,128]",
                   exact(logical(r.value()), oracle), "matches oracle (rows 1..9 not zero)");
    }
    {
        // BUG (silent zeros): a SLICED SUB-TILE operand. a is 16 live rows sliced from [64,64],
        // broadcast into [48,64]. Pre-fix, op_rows came from a.rows()==64 so 64%32==0 passed the
        // sub-tile guard and it group-broadcast a tile holding 16 live + 16 pad rows -> wrong zeros.
        // Now classify reads the view's live d.Ro==16 -> REJECT.
        auto a = make_input(dev.get(), 64, 64);
        auto b = make_input2(dev.get(), 48, 64);
        auto av = view_of(a).slice({{0, 16, 1}, {0, 64, 1}});       // logical [16,64]
        auto r = add(av, view_of(b), view_of_shape(48, 64));
        check("REGRESSION sliced sub-tile bcast a[64,64][:16] (->48) REJECT", !r,
              r ? "expected REJECT but fused" : r.error().c_str());
    }
    {
        // BUG (OOB / desync): realize into a BOUND output whose presented shape is SMALLER than the
        // gathered input. Pre-fix prelower assumed "nothing to size-check" for a bound out; the writer
        // grid then mismatched the reader. Must REJECT.
        auto in = make_input(dev.get(), 64, 64);
        auto dst = make_input(dev.get(), 32, 64);
        auto r = realize(view_of(in), view_of(dst));
        check("REGRESSION realize -> bound dst[32,64] smaller than in[64,64] REJECT", !r,
              r ? "expected REJECT but fused" : r.error().c_str());
    }
    {
        // BUG (silent live-shape truncation): a norm whose bound output has FEWER live rows than the
        // input but the SAME padded grid ([63,64] and [64,64] both tile to [64,64]). Pre-fix the grid
        // check compared only n_otr/n_otc (padded) and passed. Now it also compares logical Ro/Co.
        auto x = make_input(dev.get(), 64, 64);
        auto dst = make_input(dev.get(), 63, 64);
        auto r = l2_norm(view_of(x), 1e-6f, view_of(dst));
        check("REGRESSION l2_norm -> bound dst[63,64] vs in[64,64] (same grid) REJECT", !r,
              r ? "expected REJECT but fused" : r.error().c_str());
    }
    {
        // CONTRACT: non-bf16 input is out of scope -> clean REJECT (not a crash). fp32 [64,64].
        std::vector<float> h(64 * 64);
        for (int i = 0; i < 64 * 64; i++) h[i] = val(i);
        TensorSpec fp32_spec(ttnn::Shape({64, 64}),
                             TensorLayout(DataType::FLOAT32, PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
        auto xf = ttnn::Tensor::from_vector(h, fp32_spec, dev.get());
        auto r = realize(view_of(xf).reshape({128, 32}));
        check("CONTRACT fp32 input REJECT (bf16-only)", !r,
              r ? "expected REJECT but fused" : r.error().c_str());
    }

    // ============ per-head NORM (the RWKV reshape-kill) =========================
    // PCC vs a first-principles CPU oracle (rsqrt/bf16 -> rounded, not exact). >0.999 = correct.
    {
        auto pcc = [](const std::vector<float>& a, const std::vector<float>& b) {
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
        };
        // CPU per-head L2:   y = x * rsqrt(Σ_lane x² + eps)
        auto l2_ref = [](const std::vector<float>& x, int64_t H, int64_t S, float eps) {
            std::vector<float> o(x.size());
            for (int64_t h = 0; h < H; h++) {
                double ss = 0;
                for (int64_t s = 0; s < S; s++) { float v = x[h * S + s]; ss += (double)v * v; }
                const double r = 1.0 / std::sqrt(ss + eps);
                for (int64_t s = 0; s < S; s++) o[h * S + s] = (float)(x[h * S + s] * r);
            }
            return o;
        };
        // CPU per-head RMSNorm: y = x * rsqrt(mean(x²) + eps) * weight.
        auto rms_ref = [](const std::vector<float>& x, int64_t H, int64_t S, float eps,
                          const std::vector<float>* w) {
            std::vector<float> o(x.size());
            for (int64_t h = 0; h < H; h++) {
                double ms = 0;
                for (int64_t s = 0; s < S; s++) { float v = x[h * S + s]; ms += (double)v * v; }
                ms /= S;
                const double r = 1.0 / std::sqrt(ms + eps);
                for (int64_t s = 0; s < S; s++) {
                    double y = x[h * S + s] * r;
                    if (w) y *= (*w)[s];
                    o[h * S + s] = (float)y;
                }
            }
            return o;
        };
        // CPU per-head LayerNorm:  y = (x-mean)*rstd*gamma + beta ; mean/var over S lanes.
        auto ln_ref = [](const std::vector<float>& x, int64_t H, int64_t S, float eps,
                         const std::vector<float>* g, const std::vector<float>* b) {
            std::vector<float> o(x.size());
            for (int64_t h = 0; h < H; h++) {
                double m = 0; for (int64_t s = 0; s < S; s++) m += x[h * S + s]; m /= S;
                double v = 0; for (int64_t s = 0; s < S; s++) { double d = x[h * S + s] - m; v += d * d; }
                v /= S;
                const double r = 1.0 / std::sqrt(v + eps);
                for (int64_t s = 0; s < S; s++) {
                    double y = (x[h * S + s] - m) * r;
                    if (g) y = y * (*g)[s] + (*b)[s];
                    o[h * S + s] = (float)y;
                }
            }
            return o;
        };
        auto run_pcc = [&](const char* name, ttprm::Result<ttnn::Tensor> r,
                           const std::vector<float>& oracle) {
            if (!r) { check(name, false, r.error().c_str()); return; }
            double p = pcc(logical(r.value()), oracle);
            char d[64]; std::snprintf(d, sizeof d, "pcc %.6f", p);
            check(name, p > 0.999, d);
        };

        const int64_t H = 64, S = 64;
        // small-ish values so squares don't overflow bf16 dynamic range badly.
        auto xs = make_fn(dev.get(), 1, H * S, [](int64_t i) { return (i % 23) - 11; });
        auto xl = logical(xs);

        // (1) l2_norm: flat[1,4096] presented as head[64,64] -> AFFINE gather (the reshape-kill).
        // Slice the one live row first (as the eltwise idiom does) so the reshape is contiguous.
        auto head_view = [&](const ttnn::Tensor& t) {
            return view_of(t).slice({{0, 1, 1}, {0, H * S, 1}}).reshape({H, S});
        };
        run_pcc("l2_norm flat[1,4096]->head[64,64] (AFFINE)",
                l2_norm(head_view(xs), 1e-6f),
                l2_ref(xl, H, S, 1e-6f));

        run_pcc("rms_norm flat[1,4096]->head[64,64] (AFFINE)",
                rms_norm(head_view(xs), 1e-6f),
                rms_ref(xl, H, S, 1e-6f, nullptr));

        // (2) l2_norm on an already-head [64,64] tensor -> DENSE whole-page gather.
        auto xh = make_fn(dev.get(), H, S, [](int64_t i) { return (i % 23) - 11; });
        run_pcc("l2_norm head[64,64] (DENSE)", l2_norm(view_of(xh), 1e-6f),
                l2_ref(logical(xh), H, S, 1e-6f));
        run_pcc("rms_norm head[64,64] (DENSE)", rms_norm(view_of(xh), 1e-6f),
                rms_ref(logical(xh), H, S, 1e-6f, nullptr));
        {
            auto tt_rms = ttnn::rms_norm(xh, 1e-6f);
            run_pcc("rms_norm head[64,64] vs ttnn::rms_norm",
                    rms_norm(view_of(xh), 1e-6f), tt_rms.to_vector<float>());
        }

        // (2b) mul_l2_norm: y = l2_norm(a ⊙ b) per head -- the RWKV kk-normalize.
        auto mul_l2_ref = [](const std::vector<float>& a, const std::vector<float>& b,
                             int64_t Hh, int64_t Ss, float eps) {
            std::vector<float> o(a.size());
            for (int64_t h = 0; h < Hh; h++) {
                double ss = 0;
                for (int64_t s = 0; s < Ss; s++) { float p = a[h*Ss+s]*b[h*Ss+s]; ss += (double)p*p; }
                const double r = 1.0 / std::sqrt(ss + eps);
                for (int64_t s = 0; s < Ss; s++) o[h*Ss+s] = (float)(a[h*Ss+s]*b[h*Ss+s] * r);
            }
            return o;
        };
        auto mul_l2_row_repeat_ref = [](const std::vector<float>& a, const std::vector<float>& b,
                                        int64_t R, int64_t Br, int64_t Ss, float eps) {
            std::vector<float> o(a.size());
            for (int64_t r = 0; r < R; r++) {
                double ss = 0;
                for (int64_t s = 0; s < Ss; s++) {
                    const float p = a[r*Ss+s] * b[(r%Br)*Ss+s];
                    ss += (double)p * p;
                }
                const double inv = 1.0 / std::sqrt(ss + eps);
                for (int64_t s = 0; s < Ss; s++)
                    o[r*Ss+s] = (float)(a[r*Ss+s] * b[(r%Br)*Ss+s] * inv);
            }
            return o;
        };
        // (i) both operands flat[1,4096] viewed as head[64,64] (AFFINE gather), the kk-norm shape.
        auto bf = make_fn(dev.get(), 1, H * S, [](int64_t i) { return (i % 13) - 6; });
        run_pcc("mul_l2_norm flat a,b ->head[64,64] (AFFINE)",
                mul_l2_norm(head_view(xs), head_view(bf), 1e-6f),
                mul_l2_ref(xl, logical(bf), H, S, 1e-6f));
        // (ii) both already-head [64,64] (DENSE whole-page gather).
        auto bh = make_fn(dev.get(), H, S, [](int64_t i) { return (i % 13) - 6; });
        run_pcc("mul_l2_norm head a,b [64,64] (DENSE)",
                mul_l2_norm(view_of(xh), view_of(bh), 1e-6f),
                mul_l2_ref(logical(xh), logical(bh), H, S, 1e-6f));
        // (iii) prefill kk-norm shape: token-major rows [nt*heads,S], with a [heads,S] weight
        // broadcast over tokens. nt*heads > 32 exercises the unequal n_otr case that used to reject.
        const int64_t nt = 33, Hr = 2, Sr = 64;
        auto kp = make_fn(dev.get(), nt, Hr * Sr, [](int64_t i) { return ((i % 17) - 8) * 0.25f; });
        auto kw = make_fn(dev.get(), 1, Hr * Sr, [](int64_t i) { return ((i % 11) - 5) * 0.25f; });
        auto kp_head = view_of(kp).slice({{0, nt, 1}, {0, Hr * Sr, 1}}).reshape({nt * Hr, Sr});
        auto kw_head = view_of(kw).slice({{0, 1, 1}, {0, Hr * Sr, 1}}).reshape({Hr, Sr});
        run_pcc("mul_l2_norm prefill k[nt*heads,S] * weight[heads,S] row-repeat",
                mul_l2_norm(kp_head, kw_head, 1e-6f),
                mul_l2_row_repeat_ref(logical(kp), logical(kw), nt * Hr, Hr, Sr, 1e-6f));

        // (2c) rk: rk[h]=Σ_i r·k·r_k over the head lanes ; out = cur + v*rk -- the RWKV7 rk update.
        auto rk_ref = [](const std::vector<float>& r, const std::vector<float>& k,
                         const std::vector<float>& rkw, const std::vector<float>& v,
                         const std::vector<float>& cur, int64_t Hh, int64_t Ss) {
            std::vector<float> o(cur.size());
            for (int64_t h = 0; h < Hh; h++) {
                double rkv = 0;
                for (int64_t s = 0; s < Ss; s++)
                    rkv += (double)r[h*Ss+s] * k[h*Ss+s] * rkw[h*Ss+s];
                for (int64_t s = 0; s < Ss; s++)
                    o[h*Ss+s] = (float)(cur[h*Ss+s] + (double)v[h*Ss+s] * rkv);
            }
            return o;
        };
        // small magnitudes: rk is a 64-term Σ of triple products, keep it inside bf16 range.
        auto rrt = make_fn(dev.get(), 1, H * S, [](int64_t i) { return ((i % 7) - 3) * 0.25; });
        auto krt = make_fn(dev.get(), 1, H * S, [](int64_t i) { return ((i % 5) - 2) * 0.25; });
        auto wrt = make_fn(dev.get(), 1, H * S, [](int64_t i) { return ((i % 3) - 1) * 0.5; });
        auto vrt = make_fn(dev.get(), 1, H * S, [](int64_t i) { return ((i % 11) - 5) * 0.25; });
        auto crt = make_fn(dev.get(), 1, H * S, [](int64_t i) { return ((i % 9) - 4) * 0.5; });
        // (i) all five flat[1,4096] viewed as head[64,64] (AFFINE gather) -- the decode shape.
        run_pcc("rk flat r,k,r_k,v,cur ->head[64,64] (AFFINE)",
                rk(head_view(rrt), head_view(krt), head_view(wrt), head_view(vrt), head_view(crt)),
                rk_ref(logical(rrt), logical(krt), logical(wrt), logical(vrt), logical(crt), H, S));
        // (ii) all five already-head [64,64] (DENSE whole-page gather).
        auto rrh = make_fn(dev.get(), H, S, [](int64_t i) { return ((i % 7) - 3) * 0.25; });
        auto krh = make_fn(dev.get(), H, S, [](int64_t i) { return ((i % 5) - 2) * 0.25; });
        auto wrh = make_fn(dev.get(), H, S, [](int64_t i) { return ((i % 3) - 1) * 0.5; });
        auto vrh = make_fn(dev.get(), H, S, [](int64_t i) { return ((i % 11) - 5) * 0.25; });
        auto crh = make_fn(dev.get(), H, S, [](int64_t i) { return ((i % 9) - 4) * 0.5; });
        run_pcc("rk head r,k,r_k,v,cur [64,64] (DENSE)",
                rk(view_of(rrh), view_of(krh), view_of(wrh), view_of(vrh), view_of(crh)),
                rk_ref(logical(rrh), logical(krh), logical(wrh), logical(vrh), logical(crh), H, S));

        // (iii) PREFILL: r/k/v/cur token-major [nt*H,S]; r_k=[1,H*S]->[H,S] broadcast over the nt
        // tokens (the prefill case that used to hit the 5-way grid-equality reject). hc=H=64 is a
        // multiple of 32, so each token-group is whole row-blocks -> clean token-axis broadcast.
        const int64_t ntk = 3;
        auto rrp = make_fn(dev.get(), 1, ntk * H * S, [](int64_t i) { return ((i % 7) - 3) * 0.25; });
        auto krp = make_fn(dev.get(), 1, ntk * H * S, [](int64_t i) { return ((i % 5) - 2) * 0.25; });
        auto vrp = make_fn(dev.get(), 1, ntk * H * S, [](int64_t i) { return ((i % 11) - 5) * 0.25; });
        auto crp = make_fn(dev.get(), 1, ntk * H * S, [](int64_t i) { return ((i % 9) - 4) * 0.5; });
        auto head_p = [&](const ttnn::Tensor& t) {
            return view_of(t).slice({{0, 1, 1}, {0, ntk * H * S, 1}}).reshape({ntk * H, S});
        };
        auto rk_bcast_ref = [](const std::vector<float>& r, const std::vector<float>& k,
                               const std::vector<float>& rkw, const std::vector<float>& v,
                               const std::vector<float>& cur, int64_t nt_, int64_t Hh, int64_t Ss) {
            std::vector<float> o(cur.size());
            for (int64_t t = 0; t < nt_; t++)
                for (int64_t h = 0; h < Hh; h++) {
                    const int64_t base = (t * Hh + h) * Ss;
                    double rkv = 0;
                    for (int64_t s = 0; s < Ss; s++) rkv += (double)r[base+s] * k[base+s] * rkw[h*Ss+s];
                    for (int64_t s = 0; s < Ss; s++)
                        o[base+s] = (float)(cur[base+s] + (double)v[base+s] * rkv);
                }
            return o;
        };
        run_pcc("rk prefill [nt*H,S], r_k[H,S] broadcast over tokens (AFFINE)",
                rk(head_p(rrp), head_p(krp), head_view(wrt), head_p(vrp), head_p(crp)),
                rk_bcast_ref(logical(rrp), logical(krp), logical(wrt), logical(vrp),
                             logical(crp), ntk, H, S));
        // Larger prefill: enough row-blocks per core to exercise resident r_k reuse.
        {
            const int64_t ntkr = 128;
            auto rrr = make_fn(dev.get(), 1, ntkr * H * S, [](int64_t i) { return ((i % 7) - 3) * 0.25; });
            auto krr = make_fn(dev.get(), 1, ntkr * H * S, [](int64_t i) { return ((i % 5) - 2) * 0.25; });
            auto vrr = make_fn(dev.get(), 1, ntkr * H * S, [](int64_t i) { return ((i % 11) - 5) * 0.25; });
            auto crr = make_fn(dev.get(), 1, ntkr * H * S, [](int64_t i) { return ((i % 9) - 4) * 0.5; });
            auto head_pr = [&](const ttnn::Tensor& t) {
                return view_of(t).slice({{0, 1, 1}, {0, ntkr * H * S, 1}}).reshape({ntkr * H, S});
            };
            run_pcc("rk resident-r_k prefill [nt*H,S], r_k[H,S] broadcast over tokens (AFFINE)",
                    rk(head_pr(rrr), head_pr(krr), head_view(wrt), head_pr(vrr), head_pr(crr)),
                    rk_bcast_ref(logical(rrr), logical(krr), logical(wrt), logical(vrr),
                                 logical(crr), ntkr, H, S));
        }

        // (3) layer_norm, no affine, flat->head (AFFINE).
        run_pcc("layer_norm flat[1,4096]->head[64,64] no-affine (AFFINE)",
                layer_norm(head_view(xs), nullptr, nullptr, 1e-5f),
                ln_ref(xl, H, S, 1e-5f, nullptr, nullptr));
        // Cross-check against ttnn::layer_norm (the bf16 op we replace), not just the fp64 oracle.
        {
            auto tt_ln = ttnn::layer_norm(xh, 1e-5f);   // reduces over last dim (=64, per head)
            run_pcc("layer_norm head[64,64] vs ttnn::layer_norm",
                    layer_norm(view_of(xh), nullptr, nullptr, 1e-5f), tt_ln.to_vector<float>());
        }

        // ---- WIDE variant (reduction width > 32 tiles -> the lean-resident/streamed kernel) ----
        // (5) full-width LayerNorm over W=2048 (ctiles=64): the att/ffn-norm shape, no reshape. Each
        // of the 32 rows is an independent token reduced over its 2048 lanes.
        {
            const int64_t Rw = 32, W = 2048;          // 32 tokens x 2048 channels
            auto xw = make_fn(dev.get(), Rw, W, [](int64_t i) { return ((i % 31) - 15); });
            auto xwl = logical(xw);
            run_pcc("WIDE layer_norm [32,2048] (ctiles=64, no reshape)",
                    layer_norm(view_of(xw), nullptr, nullptr, 1e-5f),
                    ln_ref(xwl, Rw, W, 1e-5f, nullptr, nullptr));
            run_pcc("WIDE l2_norm [32,2048] (ctiles=64)",
                    l2_norm(view_of(xw), 1e-6f), l2_ref(xwl, Rw, W, 1e-6f));
            run_pcc("WIDE rms_norm [32,2048] (ctiles=64)",
                    rms_norm(view_of(xw), 1e-6f), rms_ref(xwl, Rw, W, 1e-6f, nullptr));
            // vs ttnn::layer_norm (the bf16 large-layernorm path we replace)
            run_pcc("WIDE layer_norm [32,2048] vs ttnn::layer_norm",
                    layer_norm(view_of(xw), nullptr, nullptr, 1e-5f),
                    ttnn::layer_norm(xw, 1e-5f).to_vector<float>());
        }
        // (6) the DECODE reshape-kill: flat [1,4096] -> head-grouped wide LN. Reshape into a [2,2048]
        // grid (2 groups of 2048) presented from the flat row; AFFINE gather reads only live data.
        {
            const int64_t G = 2, W = 2048;            // [1,4096] -> [2,2048], reduce each 2048 group
            auto xf = make_fn(dev.get(), 1, G * W, [](int64_t i) { return ((i % 31) - 15); });
            auto xfl = logical(xf);
            auto wide_view = view_of(xf).slice({{0, 1, 1}, {0, G * W, 1}}).reshape({G, W});
            run_pcc("WIDE l2_norm flat[1,4096]->[2,2048] (AFFINE reshape-kill)",
                    l2_norm(wide_view, 1e-6f), l2_ref(xfl, G, W, 1e-6f));
            run_pcc("WIDE rms_norm flat[1,4096]->[2,2048] (AFFINE reshape-kill)",
                    rms_norm(wide_view, 1e-6f), rms_ref(xfl, G, W, 1e-6f, nullptr));
            run_pcc("WIDE layer_norm flat[1,4096]->[2,2048] (AFFINE reshape-kill)",
                    layer_norm(wide_view, nullptr, nullptr, 1e-5f),
                    ln_ref(xfl, G, W, 1e-5f, nullptr, nullptr));
        }
        // (7) WIDE layer_norm WITH per-lane gamma/beta [1,2048] (streamed, not resident).
        {
            const int64_t Rw = 32, W = 2048;
            auto xw = make_fn(dev.get(), Rw, W, [](int64_t i) { return ((i % 31) - 15); });
            auto gw = make_fn(dev.get(), 1, W, [](int64_t i) { return ((i % 5) - 2) * 0.5; });
            auto bw = make_fn(dev.get(), 1, W, [](int64_t i) { return ((i % 3) - 1) * 0.25; });
            auto gl = logical(gw), bl = logical(bw);
            run_pcc("WIDE layer_norm [32,2048] + gamma/beta[1,2048] (streamed affine)",
                    layer_norm(view_of(xw), &gw, &bw, 1e-5f),
                    ln_ref(logical(xw), Rw, W, 1e-5f, &gl, &bl));
            run_pcc("WIDE rms_norm [32,2048] + weight[1,2048] (streamed scale)",
                    rms_norm(view_of(xw), gw, 1e-6f),
                    rms_ref(logical(xw), Rw, W, 1e-6f, &gl));
        }

        // (7b) GLOBAL folded LN: flat[1,4096] is ONE norm over all 4096 (n_otr==1, wide) -> the
        // host folds it to dense [128,32] and reduces GLOBALLY (REDUCE_SCALAR). Oracle = single
        // head of width N (ln_ref H=1,S=N). Both no-affine and with folded gamma/beta[1,N].
        {
            const int64_t N = 4096;
            auto xg = make_fn(dev.get(), 1, N, [](int64_t i) { return ((i % 31) - 15); });
            auto xgl = logical(xg);
            auto gv = make_fn(dev.get(), 1, N, [](int64_t i) { return ((i % 5) - 2) * 0.5; });
            auto bv = make_fn(dev.get(), 1, N, [](int64_t i) { return ((i % 3) - 1) * 0.25; });
            auto gvl = logical(gv), bvl = logical(bv);
            auto global_view = [&](const ttnn::Tensor& t) {
                return view_of(t).slice({{0, 1, 1}, {0, N, 1}}).reshape({1, N});
            };
            run_pcc("GLOBAL layer_norm flat[1,4096] no-affine (folded REDUCE_SCALAR)",
                    layer_norm(global_view(xg), nullptr, nullptr, 1e-5f),
                    ln_ref(xgl, 1, N, 1e-5f, nullptr, nullptr));
            run_pcc("GLOBAL layer_norm flat[1,4096] + gamma/beta[1,4096] (folded affine)",
                    layer_norm(global_view(xg), &gv, &bv, 1e-5f),
                    ln_ref(xgl, 1, N, 1e-5f, &gvl, &bvl));
            run_pcc("GLOBAL l2_norm flat[1,4096] (folded REDUCE_SCALAR)",
                    l2_norm(global_view(xg), 1e-6f), l2_ref(xgl, 1, N, 1e-6f));
            run_pcc("GLOBAL rms_norm flat[1,4096] no-weight (folded REDUCE_SCALAR)",
                    rms_norm(global_view(xg), 1e-6f), rms_ref(xgl, 1, N, 1e-6f, nullptr));
            run_pcc("GLOBAL rms_norm flat[1,4096] + weight[1,4096] (folded scale)",
                    rms_norm(global_view(xg), gv, 1e-6f), rms_ref(xgl, 1, N, 1e-6f, &gvl));
        }

        // (4) layer_norm WITH per-lane gamma/beta [1,64] (the RWKV groupnorm affine).
        auto gamma = make_fn(dev.get(), 1, S, [](int64_t i) { return ((i % 5) - 2) * 0.5; });
        auto beta  = make_fn(dev.get(), 1, S, [](int64_t i) { return ((i % 3) - 1) * 0.25; });
        auto gl = logical(gamma), bl = logical(beta);
        run_pcc("rms_norm head[64,64] + weight[1,64] (AFFINE)",
                rms_norm(view_of(xh), gamma, 1e-6f),
                rms_ref(logical(xh), H, S, 1e-6f, &gl));
        run_pcc("layer_norm head[64,64] + gamma/beta[1,64] (AFFINE)",
                layer_norm(view_of(xh), &gamma, &beta, 1e-5f),
                ln_ref(logical(xh), H, S, 1e-5f, &gl, &bl));
    }

    // ============ adversarial: must REJECT (no native fallback) =================
    {
        // element transpose: inner stride != 1 -> sub-face shuffle -> REJECT.
        auto in = make_input(dev.get(), 64, 64);
        expect_reject("element transpose [64,64] (REJECT)", view_of(in).permute({1, 0}));
    }
    {
        // sub-face col slice [8:40]: base 8 not 16-aligned -> straddles faces -> REJECT.
        auto in = make_input(dev.get(), 64, 64);
        expect_reject("sub-face col slice [8:40] (REJECT)", view_of(in).slice({{0, 64, 1}, {8, 40, 1}}));
    }
    {
        // non-contiguous slice then reshape: width-64 slice of a width-128 source is not
        // C-contiguous -> reshape poisons -> REJECT.
        auto in = make_input(dev.get(), 128, 128);
        expect_reject("slice(w64 of 128)->reshape (REJECT)", view_of(in).slice({{0, 64, 1}, {0, 64, 1}}).reshape({128, 32}));
    }

    std::printf("  %d passed, %d failed\n", g_pass, g_fail);
    const bool ok = (g_fail == 0);
    std::printf("RESULT: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
