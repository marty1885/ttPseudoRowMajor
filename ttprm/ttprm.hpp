// Public TTNN tensor surface for ttprm view operations.
// Ops consume View operands, lower accepted layouts to cached device_operations,
// and return Result<Tensor> instead of throwing.
//
//   ttprm::View                    -- a tiled ttnn::Tensor bound to a layout view-chain.
//                                     A bare Tensor converts to a dense identity View.
//   ttprm::view_of(t)              -- View(t): an input operand seeded on t's tile grid.
//   ttprm::view_of_shape(r, c)     -- an OUTPUT operand of logical [r,c] (no tensor yet).
//   ttprm::realize(in, [out], [memory_config]) -> Result<ttnn::Tensor> (cached; trace-safe)
//   ttprm::add/sub/mul(a, b, [out])-> Result<ttnn::Tensor>   fused binary eltwise, no relayout
//
// realize(in, out) uses the same gather/scatter kernels for all endpoint combinations:
// viewed input gathers, viewed output scatters, dense/dense copies, viewed/viewed does both.
// out defaults to a dense result over in's presented shape.
//
// device_operation program caching keeps later launches trace-safe by patching
// buffer addresses only.
//
// Contract: inputs must be device-resident bf16 TILE tensors. Rejected views return
// Result::err before launching device work.

#pragma once

#include <cassert>
#include <optional>
#include <string>

#include <ttnn/tensor/tensor.hpp>

#include "view_access.hpp"
#include "view_realize.hpp"

namespace ttprm {

// Small non-throwing outcome type.
template <typename T>
class Result {
public:
    static Result ok(T v) { Result r; r.ok_ = true; r.val_.emplace(std::move(v)); return r; }
    static Result err(std::string e) { Result r; r.ok_ = false; r.err_ = std::move(e); return r; }
    bool is_ok() const { return ok_; }
    explicit operator bool() const { return ok_; }
    const T& value() const { assert(ok_ && "Result::value() on error"); return *val_; }
    const std::string& error() const { return err_; }
private:
    Result() = default;
    bool ok_ = false;
    std::optional<T> val_;
    std::string err_;
};

// ============================================================================
// View is a non-owning tensor binding plus a LayoutView chain.
//
// View::of_shape creates an output-only view with no bound tensor.
class View {
public:
    using Range = LayoutView::Range;

    View(const ttnn::Tensor& t);                          // implicit: bare tensor == dense view
    static View of_shape(int64_t rows, int64_t cols);     // output operand: shape only, no tensor

    View reshape(const std::vector<int64_t>& s) const { return with(lv_.reshape(s)); }
    View slice(const std::vector<Range>& r)      const { return with(lv_.slice(r)); }
    View permute(const std::vector<int>& p)      const { return with(lv_.permute(p)); }
    View fold(int axis)                          const { return with(lv_.fold(axis)); }
    View squeeze_dim(int dim)                    const;

    const ttnn::Tensor* tensor() const { return t_; }     // null for an output-shape view
    const LayoutView& layout() const { return lv_; }
    int64_t rows() const { return rows_; }                // logical (unpadded) shape this operand
    int64_t cols() const { return cols_; }                // represents as a whole tensor

private:
    View() : lv_(0, 0) {}
    View with(LayoutView lv) const { View v = *this; v.lv_ = std::move(lv); return v; }
    const ttnn::Tensor* t_ = nullptr;
    LayoutView lv_;
    int64_t rows_ = 0, cols_ = 0;
    std::vector<int64_t> shape_;
    std::vector<int64_t> padded_shape_;
};

// Factories (sugar over the constructors; kept for readable call sites).
inline View view_of(const ttnn::Tensor& t) { return View(t); }
inline View view_of_shape(int64_t rows, int64_t cols) { return View::of_shape(rows, cols); }

// Host-only lowered relayout parameters. prelower() computes these once; realize()
// launches the cached device_operation.
struct Lowered {
    const ttnn::Tensor* in = nullptr;       // the (gathered) input tensor
    const ttnn::Tensor* out = nullptr;      // bound destination (null -> op allocates a fresh tensor)
    AffineDesc in_desc, out_desc;
    uint32_t in_kind = 0, out_kind = 0;     // DENSE / AFFINE per endpoint
    int32_t out_rows = 0, out_cols = 0;     // OUTPUT tensor logical shape
};

// Plan and lower a relayout without device work.
Result<Lowered> prelower(const View& in, std::optional<View> out = std::nullopt);

// Launch an already-lowered relayout.
ttnn::Tensor realize(
    const Lowered& low,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

// Relayout op (prelower + launch). Kernel roles are decided by endpoint kind:
//   viewed IN, dense OUT   -> gather  (the reshape-kill / relayout)
//   dense  IN, viewed OUT  -> scatter (the producer / to_folded path)
//   dense  IN, dense OUT   -> verbatim copy   |   viewed IN, viewed OUT -> gather+scatter
// `out` defaults to a fresh dense tensor of in's presented shape. If the `out` view is BOUND to a
// real tensor (view_of(dst).<reshape...>), the writer scatters into THAT tensor in place instead of
// allocating. Bound outputs must be bf16 TILE device tensors whose presented
// tiled shape matches the input view. Aliasing a non-identity relayout is unsafe.
Result<ttnn::Tensor> realize(
    const View& in,
    std::optional<View> out = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

// Fused binary elementwise op. Each operand has its own view; out defaults to a
// dense result over the inputs' output grid. Tile-divisor operands are treated as
// resident group broadcasts.
Result<ttnn::Tensor> add(
    const View& a,
    const View& b,
    std::optional<View> out = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Result<ttnn::Tensor> sub(
    const View& a,
    const View& b,
    std::optional<View> out = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Result<ttnn::Tensor> mul(
    const View& a,
    const View& b,
    std::optional<View> out = std::nullopt,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

// Fused ternary elementwise op. Uses the same per-operand views and resident
// group-broadcast scheduling as add/sub/mul:
//   fma (a, b, c)  -> out = a*b + c        (FPU DST-accumulate)
//   lerp(a, b, w)  -> out = a + w*(b - a)  (native SFPU lerp; w is a per-element weight tile)
// Broadcast is detected per operand.
Result<ttnn::Tensor> fma(const View& a, const View& b, const View& c,
                         std::optional<View> out = std::nullopt,
                         const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
// canonical_out emits token-shift lerp in TTNN's batch-padded [G,1,nt,E]
// layout instead of packed [G*nt,E]. Valid only for the token-shift shape.
Result<ttnn::Tensor> lerp(const View& a, const View& b, const View& w,
                          std::optional<View> out = std::nullopt, bool canonical_out = false,
                          const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

// Fused per-head norm. x's view is the head presentation; the reduction is per
// logical row over Co lanes.
//   l2_norm(x, eps)              -> y = x * rsqrt(Σ_lane x² + eps)               (no affine)
//   rms_norm(x, weight, eps)     -> y = x * rsqrt(mean(x²)+eps) * weight         (weight optional)
//   layer_norm(x, gamma, beta)   -> y = (x-mean)*rsqrt(var+eps)*gamma + beta     (affine optional)
// weight/gamma/beta are per-lane [1,Co] tensors broadcast across rows. LayerNorm
// takes gamma+beta together, or neither.
// View::squeeze_dim(dim) is an explicit zero-copy view operation. The selected
// logical dimension must have extent 1. It never squeezes the last two
// dimensions; ttprm treats those as the physical tile-row/tile-column
// presentation consumed by kernels.
//
// Indexed row gather/scatter for KV/recurrent-state caches. The data tensor's
// OUTER dim is the row index; each row is the contiguous block of whole tiles formed by the inner
// dims (e.g. a [R,256,32] cache -> T=8 tiles/row). `idx` is an INT32 tensor selecting
// rows; the op copies whole tiles with the row chosen by idx. The logical
// [R, W=inner] "view" is just framing; the fast path uses the native whole-tile layout, so the row
// dim must be a batch dim (rank>=3) or a 32-multiple (whole-tile rows). Requirements: data bf16,
// device, Layout::TILE; idx INT32, device.
//
//   get_rows(data, idx, [memory_config]) -> y[M, inner] where y[i] = data[idx[i]]
//   set_rows(cache, idx, src)  -> writes src[i] into cache[idx[i]] IN PLACE, returns `cache`
Result<ttnn::Tensor> get_rows(
    const ttnn::Tensor& data,
    const ttnn::Tensor& idx,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Result<ttnn::Tensor> set_rows(const ttnn::Tensor& cache, const ttnn::Tensor& idx,
                              const ttnn::Tensor& src);

Result<ttnn::Tensor> l2_norm(const View& x, float eps = 1e-6f,
                             std::optional<View> out = std::nullopt,
                             const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Result<ttnn::Tensor> rms_norm(const View& x, float eps = 1e-6f,
                              std::optional<View> out = std::nullopt,
                              const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Result<ttnn::Tensor> rms_norm(const View& x, const ttnn::Tensor& weight,
                              float eps = 1e-6f, std::optional<View> out = std::nullopt,
                              const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Result<ttnn::Tensor> layer_norm(const View& x, const ttnn::Tensor* gamma = nullptr,
                                const ttnn::Tensor* beta = nullptr, float eps = 1e-5f,
                                std::optional<View> out = std::nullopt,
                                const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
Result<ttnn::Tensor> layer_norm(const View& x, const View& gamma, const View& beta,
                                float eps = 1e-5f, std::optional<View> out = std::nullopt,
                                const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

// Fused mul_l2_norm: y = l2_norm(a * b) over each head's lanes.
// a and b each carry their own view and must present the same head grid.
Result<ttnn::Tensor> mul_l2_norm(const View& a, const View& b, float eps = 1e-6f,
                                 std::optional<View> out = std::nullopt,
                                 const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

// Fused RWKV7 rk update:
//   rk[h]     = Σ_i r[h][i] * k[h][i] * r_k[h][i]     (per-head scalar, reduce over the head lanes)
//   out[h][i] = cur[h][i] + v[h][i] * rk[h]           (rk broadcast across the head's lanes)
// Each operand carries its own view and must present the same head grid.
Result<ttnn::Tensor> rk(const View& r, const View& k, const View& r_k, const View& v,
                        const View& cur, std::optional<View> out = std::nullopt,
                        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttprm
