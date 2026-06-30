// TTNN tensor surface and cached device_operations. See ttprm.hpp.
// This is the only ttprm translation unit that links TTNN.

#include <cstdlib>
#include <cmath>

#include "ttprm.hpp"
#include "tensor_view_args.hpp"

#include <cstdint>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/core_coord.hpp>

#include <ttnn/tensor/tensor_spec.hpp>
#include <ttnn/tensor/layout/tensor_layout.hpp>
#include <ttnn/device_operation.hpp>

namespace ttprm {

using namespace tt::tt_metal;
using distributed::MeshDevice;

namespace {

MemoryConfig output_memory_config(const std::optional<MemoryConfig>& memory_config) {
    return memory_config.value_or(MemoryConfig{});
}

MemoryConfig output_memory_config(
    const std::optional<ttnn::Tensor>& opt_out,
    const std::optional<MemoryConfig>& memory_config) {
    return opt_out ? opt_out->memory_config() : output_memory_config(memory_config);
}

TensorSpec tile_spec(int64_t rows, int64_t cols, const MemoryConfig& memory_config) {
    return TensorSpec(ttnn::Shape({(uint32_t)rows, (uint32_t)cols}),
                      TensorLayout(DataType::BFLOAT16,
                                   PageConfig(tt::tt_metal::Layout::TILE), memory_config));
}

std::string validate_device_tile_bf16(const ttnn::Tensor& t) {
    if (!t.is_allocated()) return "input not allocated";
    if (t.storage_type() != StorageType::DEVICE) return "input not on device";
    if (t.dtype() != DataType::BFLOAT16) return "input not bf16";
    if (t.layout() != tt::tt_metal::Layout::TILE) return "input not TILE layout";
    return "";
}

// Resolve a bound output tensor, or allocate a fresh tensor matching the op spec.
template <class Dev>
ttnn::Tensor resolve_output(const std::optional<ttnn::Tensor>& opt_out,
                            const ttnn::TensorSpec& spec, Dev* dev) {
    return opt_out ? *opt_out : create_device_tensor(spec, dev);
}

// Convert an optional output View into the bound tensor passed to invoke().
// Output-shape-only views return nullopt so the op allocates.
std::optional<ttnn::Tensor> bound_dst(const std::optional<View>& out, std::string& err) {
    err.clear();
    if (out && out->tensor()) {
        err = validate_device_tile_bf16(*out->tensor());
        if (!err.empty()) { err = "out: " + err; return std::nullopt; }
        return *out->tensor();
    }
    return std::nullopt;
}

// Dense identity endpoints can use whole-page transfers only when all tiles are
// fully live. Plain sources with padded rows still require the affine seed path.
bool is_dense_identity(const AffineDesc& d) {
    return d.grp_P == 0 && d.base == 0 && d.S_row == d.Cs && d.Co == d.Cs &&
           d.Ro_pad == d.Ro && d.Co_pad == d.Co;
}
// A plain source is row-major over its own tensor width. It may still have padded
// rows and therefore may not be dense-identity.
bool is_plain_source(const AffineDesc& d) {
    return d.grp_P == 0 && d.base == 0 && d.Co == d.Cs && d.Co_pad == d.Co &&
           (d.Ro <= 1 ? d.S_row == 0 : d.S_row == d.Cs);
}
uint32_t kind_of(const AffineDesc& d) {
    if (d.grp_P != 0) return VIEW_GROUPED;                      // 2-level (prefill) row map
    return is_dense_identity(d) ? VIEW_DENSE : VIEW_AFFINE;
}

// Build a GROUPED endpoint desc for remapped/repeated presented rows:
// source geometry (base/S_row/Cs/nct) from the operand, output tile grid + live extent from `dout`,
// and the 2-level row map source_row(oR) = A*(oR/P) + B*(oR%P). Streams the full output grid
// (out_tiles == wt, bcast=0). The device GroupedTensorView walks it.
AffineDesc make_grouped_desc(const AffineDesc& op, const AffineDesc& dout,
                             int64_t P, int64_t A, int64_t B) {
    AffineDesc d = dout;
    d.base = op.base; d.Cs = op.Cs; d.nct = op.nct;
    d.S_row = op.S_row;
    d.grp_P = P; d.grp_A = A; d.grp_B = B;
    return d;
}

// Classify one fused-eltwise operand against the output grid. The classifier
// mutates row-broadcast operands to widen their gather guard and returns the
// shared resident group size in `tpg`.
//
//   FULL          d covers the whole output grid (n == wt)            -> stream one tile per output.
//   ROW-BROADCAST d has one live row replicated down the output rows.
//   GROUP-BCAST   d is a proper tile-divisor of the output: it holds one group's tpg tiles, re-fed
//                 across G=wt/tpg groups. Group broadcasts require whole tile-rows.
std::string classify_operand(AffineDesc& d, const AffineDesc& dout,
                             int64_t out_rows, int64_t wt, int64_t& tpg, uint32_t& bcast) {
    const int64_t live_rows = d.Ro;
    // Detect row-broadcast by live rows rather than tile count; sub-32 outputs can
    // have the same tile count as the single-row operand.
    if (live_rows == 1 && out_rows > 1) {
        if (d.Co != dout.Co) return "row-broadcast operand width != output width";
        // Widen the live-row guard so S_row==0 replicates source row 0 through L1.
        d.S_row = 0;
        d.Ro = dout.Ro; d.Ro_pad = dout.Ro_pad; d.n_otr = dout.n_otr; d.n_otc = dout.n_otc;
        bcast = 0; return "";
    }
    // The linear odometer cannot express periodic sub-tile row repeats.
    if (is_plain_source(d) && live_rows > 1 && live_rows < out_rows && live_rows % TILE != 0)
        return "sub-tile row broadcast (operand rows not a multiple of 32) unsupported -- "
               "fall back to ttnn, or (for the lerp token-shift) use ttprm::lerp";
    const int64_t n = d.out_tiles();
    if (n == wt) { bcast = 0; return ""; }                // FULL
    if (n > 0 && wt % n == 0) {
        if (live_rows % TILE != 0)
            return "sub-tile group broadcast (operand rows not a multiple of 32) unsupported -- "
                   "fall back to ttnn, or use n_tokens a multiple of 32";
        bcast = 1;
        if (tpg != wt && tpg != n) return "broadcast operands disagree on group size";
        tpg = n; return "";
    }
    return "input tile grid is not the output grid nor a divisor of it";
}

void mkcb(Program& p, const CoreRangeSet& cr, tt::CBIndex cb, uint32_t page_bytes,
          uint32_t pages, tt::DataFormat fmt) {
    CircularBufferConfig c(pages * page_bytes, {{cb, fmt}});
    c.set_page_size(cb, page_bytes);
    CreateCircularBuffer(p, cr, c);
}

// Compact affine block stored in operation_attributes for program-cache keys.
struct ViewArgs7 {
    int32_t base = 0, S_row = 0, Cs = 0, nct = 0, Ro = 0, Co = 0, n_otc = 0;
    int32_t grp_P = 0, grp_A = 0, grp_B = 0;
};

AffineDesc to_desc(const ViewArgs7& p) {
    AffineDesc d;
    d.base = p.base; d.S_row = p.S_row; d.Cs = p.Cs; d.nct = p.nct;
    d.Ro = p.Ro; d.Co = p.Co; d.n_otc = p.n_otc;
    d.grp_P = p.grp_P; d.grp_A = p.grp_A; d.grp_B = p.grp_B;
    return d;
}
ViewArgs7 from_desc(const AffineDesc& d) {
    return {(int32_t)d.base, (int32_t)d.S_row, (int32_t)d.Cs, (int32_t)d.nct,
            (int32_t)d.Ro, (int32_t)d.Co, (int32_t)d.n_otc,
            (int32_t)d.grp_P, (int32_t)d.grp_A, (int32_t)d.grp_B};
}

// Relayout device_operation.
namespace vr {

struct operation_attributes_t {
    uint32_t in_kind, out_kind;          // TensorView KIND per endpoint (kernel defines)
    int32_t out_rows, out_cols;          // OUTPUT tensor logical shape (output spec)
    int32_t n_otr, n_otc;                // WORK grid (= the reader/writer tile-iteration count)
    MemoryConfig memory_config;
    ViewArgs7 in, out;
};
struct tensor_args_t { const ttnn::Tensor& in; std::optional<ttnn::Tensor> opt_out; };
using spec_return_value_t   = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {
struct ViewRealizeFactory {
    struct shared_variables_t { KernelHandle reader, writer; CoreRangeSet all_cores; };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    // Per-kernel runtime args: [addr, n_tiles] + this endpoint's view-arg block (packed by the
    // shared append_view_args). The device derives the odometer increments + per-core seed; only
    // the address + this endpoint's map differ between reader (in) and writer (out).
    static std::vector<uint32_t> args(uint32_t kind, const ViewArgs7& p,
                                      uint32_t addr, uint32_t t0, uint32_t count) {
        std::vector<uint32_t> v{addr, count};
        append_view_args(v, kind, t0, to_desc(p));
        return v;
    }

    static cached_program_t create(
        const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& out) {
        Program prog = CreateProgram();
        const uint32_t work_tiles = (uint32_t)(a.n_otr * a.n_otc);   // tiles each core iterates

        MeshDevice* dev = t.in.device();
        auto grid = dev->compute_with_storage_grid_size();
        auto [num_cores, all_cores, cg1, cg2, upc1, upc2] =
            tt::tt_metal::split_work_to_cores(grid, work_tiles);

        mkcb(prog, all_cores, tt::CBIndex::c_0, 2048u, 4, tt::DataFormat::Float16_b);  // io pipeline
        mkcb(prog, all_cores, tt::CBIndex::c_31, 2048u, 1, tt::DataFormat::Float16_b); // zero tile

        const std::string ttnn_inc = std::string(getenv("TT_METAL_HOME")) + "/ttnn";  // tt_memmove
        std::vector<uint32_t> rct; TensorAccessorArgs(*t.in.buffer()).append_to(rct);
        KernelHandle reader = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_reader.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                               .compile_args = rct, .defines = {{"TTPRM_IN_VIEW", view_type_name(a.in_kind)}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        std::vector<uint32_t> wct; TensorAccessorArgs(*out.buffer()).append_to(wct);
        KernelHandle writer = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_writer.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                               .compile_args = wct, .defines = {{"TTPRM_OUT_VIEW", view_type_name(a.out_kind)}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_compute.cpp", all_cores,
            ComputeConfig{.compile_args = {}});

        const uint32_t s_addr = (uint32_t)t.in.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
        uint32_t done = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord cc = cores[i];
            const uint32_t here = cg1.contains(cc) ? upc1 : upc2;
            SetRuntimeArgs(prog, reader, cc, args(a.in_kind,  a.in,  s_addr, done, here));
            SetRuntimeArgs(prog, writer, cc, args(a.out_kind, a.out, o_addr, done, here));
            done += here;
        }
        return {std::move(prog), shared_variables_t{reader, writer, all_cores}};
    }

    static void override_runtime_arguments(
        cached_program_t& cp, const operation_attributes_t&, const tensor_args_t& t,
        tensor_return_value_t& out) {
        auto& prog = cp.program;
        const uint32_t s_addr = (uint32_t)t.in.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        for (const auto& range : cp.shared_variables.all_cores.ranges())
            for (const auto& core : range) {
                GetRuntimeArgs(prog, cp.shared_variables.reader, core)[0] = s_addr;
                GetRuntimeArgs(prog, cp.shared_variables.writer, core)[0] = o_addr;
            }
    }
};
}  // namespace program

struct ViewRealizeDeviceOperation {
    using operation_attributes_t = vr::operation_attributes_t;
    using tensor_args_t          = vr::tensor_args_t;
    using spec_return_value_t    = vr::spec_return_value_t;
    using tensor_return_value_t  = vr::tensor_return_value_t;
    using program_factory_t      = std::variant<program::ViewRealizeFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return program::ViewRealizeFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t& a, const tensor_args_t&) {
        return tile_spec(a.out_rows, a.out_cols, a.memory_config);
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
        return resolve_output(t.opt_out, compute_output_specs(a, t), t.in.device());
    }
    // in gathers through in_desc; out scatters through out_desc. The work grid (reader/writer
    // tile-iteration count) is in_desc's presented tile grid -- which equals out_desc's by
    // construction (a relayout preserves element count -> tile count).
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& in, const AffineDesc& in_desc, const AffineDesc& out_desc,
        int32_t out_rows, int32_t out_cols, std::optional<ttnn::Tensor> opt_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        return { operation_attributes_t{
                     kind_of(in_desc), kind_of(out_desc), out_rows, out_cols,
                     (int32_t)in_desc.n_otr, (int32_t)in_desc.n_otc,
                     output_memory_config(opt_out, memory_config),
                     from_desc(in_desc), from_desc(out_desc)},
                 tensor_args_t{in, std::move(opt_out)} };
    }
};

}  // namespace vr

// Fused binary op: out = a OP b, laid out by an output view.
namespace va {

struct operation_attributes_t {
    uint32_t op = 0;                                 // binary op: 0 add, 1 sub, 2 mul (kernel define)
    uint32_t a_kind = 0, b_kind = 0, out_kind = 0;   // per-endpoint TensorView KIND (kernel defines)
    int32_t out_rows = 0, out_cols = 0;              // OUTPUT tensor logical shape (output spec)
    int32_t n_otr = 0, n_otc = 0;                    // OUTPUT tile grid = the work units
    int32_t tpg = 0;                                 // tiles per broadcast group (== work tiles if none)
    uint32_t a_bcast = 0, b_bcast = 0;               // input broadcast (group_stride 0) flags
    MemoryConfig memory_config;
    ViewArgs7 a, b, o;
};
struct tensor_args_t { const ttnn::Tensor& a; const ttnn::Tensor& b; std::optional<ttnn::Tensor> opt_out; };
using spec_return_value_t   = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {
struct ViewAddFactory {
    struct shared_variables_t { KernelHandle reader, writer; CoreRangeSet all_cores; };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    // reader: [a_addr, b_addr, n_tiles, tpg, a_bcast, b_bcast] + view A block + view B block.
    // Broadcast inputs seed at group start and remain resident; streaming
    // inputs start at the core's global tile.
    static std::vector<uint32_t> reader_args(const operation_attributes_t& a, uint32_t a_addr,
                                             uint32_t b_addr, uint32_t done, uint32_t count) {
        const uint32_t t0_a = a.a_bcast ? 0u : done;
        const uint32_t t0_b = a.b_bcast ? 0u : done;
        std::vector<uint32_t> v{a_addr, b_addr, count, (uint32_t)a.tpg, a.a_bcast, a.b_bcast};
        append_view_args(v, a.a_kind, t0_a, to_desc(a.a));
        append_view_args(v, a.b_kind, t0_b, to_desc(a.b));
        return v;
    }
    // compute: [n_tiles, tpg, i0, a_bcast, b_bcast]. i0 = this core's first within-group index;
    // compute re-indexes a resident broadcast group by (i0 + t) % tpg across the loop.
    static std::vector<uint32_t> compute_args(const operation_attributes_t& a, uint32_t done,
                                              uint32_t count) {
        return {count, (uint32_t)a.tpg, done % (uint32_t)a.tpg, a.a_bcast, a.b_bcast};
    }
    // writer: [o_addr, n_tiles] + output view block (same layout as the relayout writer).
    static std::vector<uint32_t> writer_args(const operation_attributes_t& a, uint32_t o_addr,
                                             uint32_t t0, uint32_t count) {
        std::vector<uint32_t> v{o_addr, count};
        append_view_args(v, a.out_kind, t0, to_desc(a.o));
        return v;
    }

    static cached_program_t create(
        const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& out) {
        Program prog = CreateProgram();
        // Broadcast work is split by group so each core owns whole resident
        // tpg-tile groups. Non-broadcast work is split by tile.
        const uint32_t wt    = (uint32_t)(a.n_otr * a.n_otc);
        const uint32_t tpg   = (uint32_t)a.tpg;
        const bool     bcast = a.a_bcast || a.b_bcast;
        const uint32_t units = bcast ? wt / tpg : wt;   // groups when broadcasting, else tiles

        MeshDevice* dev = t.a.device();
        auto grid = dev->compute_with_storage_grid_size();
        auto [num_cores, all_cores, cg1, cg2, upc1, upc2] =
            tt::tt_metal::split_work_to_cores(grid, units);

        // Resident broadcast operands need a CB sized to the full tpg-tile group.
        const uint32_t a_pages = a.a_bcast ? (uint32_t)a.tpg : 4u;
        const uint32_t b_pages = a.b_bcast ? (uint32_t)a.tpg : 4u;
        mkcb(prog, all_cores, tt::CBIndex::c_0,  2048u, a_pages, tt::DataFormat::Float16_b);  // cb_a
        mkcb(prog, all_cores, tt::CBIndex::c_1,  2048u, b_pages, tt::DataFormat::Float16_b);  // cb_b
        mkcb(prog, all_cores, tt::CBIndex::c_16, 2048u, 4, tt::DataFormat::Float16_b);  // cb_out
        mkcb(prog, all_cores, tt::CBIndex::c_31, 2048u, 1, tt::DataFormat::Float16_b);  // zero

        const std::string ttnn_inc = std::string(getenv("TT_METAL_HOME")) + "/ttnn";
        std::vector<uint32_t> rct;
        TensorAccessorArgs(*t.a.buffer()).append_to(rct);
        TensorAccessorArgs(*t.b.buffer()).append_to(rct);   // b's compile args follow a's
        KernelHandle reader = CreateKernel(prog, TTPRM_ROOT "/kernel/view_add_reader.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                               .compile_args = rct,
                               .defines = {{"TTPRM_A_VIEW", view_type_name(a.a_kind)},
                                           {"TTPRM_B_VIEW", view_type_name(a.b_kind)}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        std::vector<uint32_t> wct; TensorAccessorArgs(*out.buffer()).append_to(wct);
        KernelHandle writer = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_writer.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                               .compile_args = wct,
                               .defines = {{"TTPRM_OUT_VIEW", view_type_name(a.out_kind)},
                                           {"TTPRM_OUT_CB", "16"}},   // drain compute's cb_out
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        const char* binop = a.op == 1 ? "sub" : (a.op == 2 ? "mul" : "add");
        KernelHandle compute = CreateKernel(prog, TTPRM_ROOT "/kernel/view_add_compute.cpp", all_cores,
                                            ComputeConfig{.defines = {{"TTPRM_BINOP", binop}}});

        const uint32_t a_addr = (uint32_t)t.a.buffer()->address();
        const uint32_t b_addr = (uint32_t)t.b.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
        uint32_t done_u = 0;   // units (groups if broadcasting, else tiles) assigned so far
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord cc = cores[i];
            const uint32_t here_u = cg1.contains(cc) ? upc1 : upc2;
            const uint32_t done_t  = bcast ? done_u  * tpg : done_u;    // -> output tiles
            const uint32_t count_t = bcast ? here_u * tpg : here_u;
            SetRuntimeArgs(prog, reader, cc, reader_args(a, a_addr, b_addr, done_t, count_t));
            SetRuntimeArgs(prog, writer, cc, writer_args(a, o_addr, done_t, count_t));
            SetRuntimeArgs(prog, compute, cc, compute_args(a, done_t, count_t));
            done_u += here_u;
        }
        return {std::move(prog), shared_variables_t{reader, writer, all_cores}};
    }

    static void override_runtime_arguments(
        cached_program_t& cp, const operation_attributes_t&, const tensor_args_t& t,
        tensor_return_value_t& out) {
        auto& prog = cp.program;
        const uint32_t a_addr = (uint32_t)t.a.buffer()->address();
        const uint32_t b_addr = (uint32_t)t.b.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        for (const auto& range : cp.shared_variables.all_cores.ranges())
            for (const auto& core : range) {
                auto& ra = GetRuntimeArgs(prog, cp.shared_variables.reader, core);
                ra[0] = a_addr; ra[1] = b_addr;
                GetRuntimeArgs(prog, cp.shared_variables.writer, core)[0] = o_addr;
            }
    }
};
}  // namespace program

struct ViewAddDeviceOperation {
    using operation_attributes_t = va::operation_attributes_t;
    using tensor_args_t          = va::tensor_args_t;
    using spec_return_value_t    = va::spec_return_value_t;
    using tensor_return_value_t  = va::tensor_return_value_t;
    using program_factory_t      = std::variant<program::ViewAddFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return program::ViewAddFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t& a, const tensor_args_t&) {
        return tile_spec(a.out_rows, a.out_cols, a.memory_config);
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
        return resolve_output(t.opt_out, compute_output_specs(a, t), t.a.device());
    }
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        uint32_t op, const ttnn::Tensor& a, const ttnn::Tensor& b,
        const AffineDesc& da, const AffineDesc& db, const AffineDesc& dout,
        int32_t out_rows, int32_t out_cols, int32_t tpg, uint32_t a_bcast, uint32_t b_bcast,
        std::optional<ttnn::Tensor> opt_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        operation_attributes_t at;
        at.op = op;
        at.a_kind = kind_of(da); at.b_kind = kind_of(db); at.out_kind = kind_of(dout);
        at.out_rows = out_rows; at.out_cols = out_cols;
        at.n_otr = (int32_t)dout.n_otr; at.n_otc = (int32_t)dout.n_otc;
        at.tpg = tpg; at.a_bcast = a_bcast; at.b_bcast = b_bcast;
        at.memory_config = output_memory_config(opt_out, memory_config);
        at.a = from_desc(da); at.b = from_desc(db); at.o = from_desc(dout);
        return { at, tensor_args_t{a, b, std::move(opt_out)} };
    }
};

}  // namespace va

// Fused ternary op: out = TERNOP(a, b, c), laid out by an output view.
//   FMA  (op 0): out = a*b + c
//   LERP (op 1): out = a + c*(b - a)   (c is the weight w)
namespace vt {

struct operation_attributes_t {
    uint32_t op = 0;                                          // 0 fma, 1 lerp (kernel define)
    uint32_t a_kind = 0, b_kind = 0, c_kind = 0, out_kind = 0;
    int32_t out_rows = 0, out_cols = 0;                      // OUTPUT tensor logical shape (2D default)
    int32_t canon_G = 0, canon_nt = 0;                       // >0: emit 4D canonical [G,1,nt,E] spec
    int32_t n_otr = 0, n_otc = 0;                            // OUTPUT tile grid = the work units
    int32_t tpg = 0;                                         // tiles per broadcast group
    uint32_t a_bcast = 0, b_bcast = 0, c_bcast = 0;          // input broadcast (group_stride 0) flags
    MemoryConfig memory_config;
    ViewArgs7 a, b, c, o;
};
struct tensor_args_t { const ttnn::Tensor& a; const ttnn::Tensor& b; const ttnn::Tensor& c; std::optional<ttnn::Tensor> opt_out; };
using spec_return_value_t   = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {
struct ViewTernaryFactory {
    struct shared_variables_t { KernelHandle reader, writer; CoreRangeSet all_cores; };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    // reader: [a_addr,b_addr,c_addr,n_tiles,tpg,a_bcast,b_bcast,c_bcast] + view A + view B + view C.
    // Broadcast inputs seed at group start and remain resident; streaming
    // inputs start at the core's global tile.
    static std::vector<uint32_t> reader_args(const operation_attributes_t& a, uint32_t a_addr,
                                             uint32_t b_addr, uint32_t c_addr,
                                             uint32_t done, uint32_t count) {
        const uint32_t t0_a = a.a_bcast ? 0u : done;
        const uint32_t t0_b = a.b_bcast ? 0u : done;
        const uint32_t t0_c = a.c_bcast ? 0u : done;
        std::vector<uint32_t> v{a_addr, b_addr, c_addr, count, (uint32_t)a.tpg,
                                a.a_bcast, a.b_bcast, a.c_bcast};
        append_view_args(v, a.a_kind, t0_a, to_desc(a.a));
        append_view_args(v, a.b_kind, t0_b, to_desc(a.b));
        append_view_args(v, a.c_kind, t0_c, to_desc(a.c));
        return v;
    }
    // compute: [n_tiles, tpg, i0, a_bcast, b_bcast, c_bcast]. i0 = this core's first within-group
    // index; compute re-indexes each resident broadcast group by (i0 + t) % tpg.
    static std::vector<uint32_t> compute_args(const operation_attributes_t& a, uint32_t done,
                                              uint32_t count) {
        return {count, (uint32_t)a.tpg, done % (uint32_t)a.tpg, a.a_bcast, a.b_bcast, a.c_bcast};
    }
    // writer: [o_addr, n_tiles] + output view block (same layout as the relayout writer).
    static std::vector<uint32_t> writer_args(const operation_attributes_t& a, uint32_t o_addr,
                                             uint32_t t0, uint32_t count) {
        std::vector<uint32_t> v{o_addr, count};
        append_view_args(v, a.out_kind, t0, to_desc(a.o));
        return v;
    }

    static cached_program_t create(
        const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& out) {
        Program prog = CreateProgram();
        // Group-split when any operand broadcasts (so each core owns whole tpg-tile groups and the
        // resident gather amortizes); tile-split otherwise. Same as the binary op.
        const uint32_t wt    = (uint32_t)(a.n_otr * a.n_otc);
        const uint32_t tpg   = (uint32_t)a.tpg;
        const bool     bcast = a.a_bcast || a.b_bcast || a.c_bcast;
        const uint32_t units = bcast ? wt / tpg : wt;

        MeshDevice* dev = t.a.device();
        auto grid = dev->compute_with_storage_grid_size();
        auto [num_cores, all_cores, cg1, cg2, upc1, upc2] =
            tt::tt_metal::split_work_to_cores(grid, units);

        // A broadcast operand is held RESIDENT: its CB fits the whole tpg-tile group. A streaming
        // operand keeps a small double-buffer.
        const uint32_t a_pages = a.a_bcast ? (uint32_t)a.tpg : 4u;
        const uint32_t b_pages = a.b_bcast ? (uint32_t)a.tpg : 4u;
        const uint32_t c_pages = a.c_bcast ? (uint32_t)a.tpg : 4u;
        mkcb(prog, all_cores, tt::CBIndex::c_0,  2048u, a_pages, tt::DataFormat::Float16_b);  // cb_a
        mkcb(prog, all_cores, tt::CBIndex::c_1,  2048u, b_pages, tt::DataFormat::Float16_b);  // cb_b
        mkcb(prog, all_cores, tt::CBIndex::c_2,  2048u, c_pages, tt::DataFormat::Float16_b);  // cb_c
        mkcb(prog, all_cores, tt::CBIndex::c_16, 2048u, 4, tt::DataFormat::Float16_b);  // cb_out
        mkcb(prog, all_cores, tt::CBIndex::c_31, 2048u, 1, tt::DataFormat::Float16_b);  // zero

        const std::string ttnn_inc = std::string(getenv("TT_METAL_HOME")) + "/ttnn";
        std::vector<uint32_t> rct;
        TensorAccessorArgs(*t.a.buffer()).append_to(rct);
        TensorAccessorArgs(*t.b.buffer()).append_to(rct);   // b's compile args follow a's
        TensorAccessorArgs(*t.c.buffer()).append_to(rct);   // c's follow b's
        KernelHandle reader = CreateKernel(prog, TTPRM_ROOT "/kernel/view_ternary_reader.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                               .compile_args = rct,
                               .defines = {{"TTPRM_A_VIEW", view_type_name(a.a_kind)},
                                           {"TTPRM_B_VIEW", view_type_name(a.b_kind)},
                                           {"TTPRM_C_VIEW", view_type_name(a.c_kind)}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        std::vector<uint32_t> wct; TensorAccessorArgs(*out.buffer()).append_to(wct);
        KernelHandle writer = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_writer.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                               .compile_args = wct,
                               .defines = {{"TTPRM_OUT_VIEW", view_type_name(a.out_kind)},
                                           {"TTPRM_OUT_CB", "16"}},   // drain compute's cb_out
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        KernelHandle compute = CreateKernel(prog, TTPRM_ROOT "/kernel/view_ternary_compute.cpp", all_cores,
            ComputeConfig{.defines = {{a.op == 1 ? "TTPRM_LERP" : "TTPRM_FMA", "1"}}});

        const uint32_t a_addr = (uint32_t)t.a.buffer()->address();
        const uint32_t b_addr = (uint32_t)t.b.buffer()->address();
        const uint32_t c_addr = (uint32_t)t.c.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
        uint32_t done_u = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord cc = cores[i];
            const uint32_t here_u = cg1.contains(cc) ? upc1 : upc2;
            const uint32_t done_t  = bcast ? done_u  * tpg : done_u;
            const uint32_t count_t = bcast ? here_u * tpg : here_u;
            SetRuntimeArgs(prog, reader, cc, reader_args(a, a_addr, b_addr, c_addr, done_t, count_t));
            SetRuntimeArgs(prog, writer, cc, writer_args(a, o_addr, done_t, count_t));
            SetRuntimeArgs(prog, compute, cc, compute_args(a, done_t, count_t));
            done_u += here_u;
        }
        return {std::move(prog), shared_variables_t{reader, writer, all_cores}};
    }

    static void override_runtime_arguments(
        cached_program_t& cp, const operation_attributes_t&, const tensor_args_t& t,
        tensor_return_value_t& out) {
        auto& prog = cp.program;
        const uint32_t a_addr = (uint32_t)t.a.buffer()->address();
        const uint32_t b_addr = (uint32_t)t.b.buffer()->address();
        const uint32_t c_addr = (uint32_t)t.c.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        for (const auto& range : cp.shared_variables.all_cores.ranges())
            for (const auto& core : range) {
                auto& ra = GetRuntimeArgs(prog, cp.shared_variables.reader, core);
                ra[0] = a_addr; ra[1] = b_addr; ra[2] = c_addr;
                GetRuntimeArgs(prog, cp.shared_variables.writer, core)[0] = o_addr;
            }
    }
};
}  // namespace program

struct ViewTernaryDeviceOperation {
    using operation_attributes_t = vt::operation_attributes_t;
    using tensor_args_t          = vt::tensor_args_t;
    using spec_return_value_t    = vt::spec_return_value_t;
    using tensor_return_value_t  = vt::tensor_return_value_t;
    using program_factory_t      = std::variant<program::ViewTernaryFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return program::ViewTernaryFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t& a, const tensor_args_t&) {
        // Canonical token-shift output: logical [G,1,nt,E] over the same
        // physical buffer as [G*ceil32(nt), E].
        if (a.canon_nt > 0) {
            ttnn::SmallVector<uint32_t> dims{(uint32_t)a.canon_G, 1u,
                                             (uint32_t)a.canon_nt, (uint32_t)a.out_cols};
            return TensorSpec(ttnn::Shape(dims), TensorLayout(DataType::BFLOAT16,
                              PageConfig(tt::tt_metal::Layout::TILE), a.memory_config));
        }
        return tile_spec(a.out_rows, a.out_cols, a.memory_config);
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
        return resolve_output(t.opt_out, compute_output_specs(a, t), t.a.device());
    }
    // `dout` is the WORK grid (n_otr/n_otc = the compute/reader/writer tile iteration, = the packed
    // output the operands broadcast into). `dosc` is the OUTPUT SCATTER map: usually == dout (dense
    // packed result), but for the RWKV token-shift DENSIFY it is a GROUPED map that lands each packed
    // tile-row into ttnn's batch-padded [G,1,nt,E] canonical layout, while the iteration grid stays
    // packed. canon_nt>0 makes compute_output_specs hand back the 4D [G,1,nt,E] logical shape.
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        uint32_t op, const ttnn::Tensor& a, const ttnn::Tensor& b, const ttnn::Tensor& c,
        const AffineDesc& da, const AffineDesc& db, const AffineDesc& dc, const AffineDesc& dout,
        const AffineDesc& dosc, int32_t out_rows, int32_t out_cols, int32_t tpg,
        uint32_t a_bcast, uint32_t b_bcast, uint32_t c_bcast,
        int32_t canon_G = 0, int32_t canon_nt = 0, std::optional<ttnn::Tensor> opt_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        operation_attributes_t at;
        at.op = op;
        at.a_kind = kind_of(da); at.b_kind = kind_of(db); at.c_kind = kind_of(dc);
        at.out_kind = kind_of(dosc);
        at.out_rows = out_rows; at.out_cols = out_cols;
        at.canon_G = canon_G; at.canon_nt = canon_nt;
        at.n_otr = (int32_t)dout.n_otr; at.n_otc = (int32_t)dout.n_otc;
        at.tpg = tpg; at.a_bcast = a_bcast; at.b_bcast = b_bcast; at.c_bcast = c_bcast;
        at.memory_config = output_memory_config(opt_out, memory_config);
        at.a = from_desc(da); at.b = from_desc(db); at.c = from_desc(dc); at.o = from_desc(dosc);
        return { at, tensor_args_t{a, b, c, std::move(opt_out)} };
    }
};

}  // namespace vt

// Fused RWKV7 k update: out = k + (a - 1) * (k * k_a).
namespace vku {

struct operation_attributes_t {
    uint32_t k_kind = 0, a_kind = 0, ka_kind = 0, out_kind = 0;
    int32_t out_rows = 0, out_cols = 0;
    int32_t n_otr = 0, n_otc = 0;
    int32_t tpg = 0;
    uint32_t k_bcast = 0, a_bcast = 0, ka_bcast = 0;
    MemoryConfig memory_config;
    ViewArgs7 k, a, ka, o;
};
struct tensor_args_t {
    const ttnn::Tensor& k; const ttnn::Tensor& a; const ttnn::Tensor& ka;
    std::optional<ttnn::Tensor> opt_out;
};
using spec_return_value_t   = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {
struct ViewKUpdateFactory {
    struct shared_variables_t { KernelHandle reader, writer; CoreRangeSet all_cores; };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static std::vector<uint32_t> reader_args(const operation_attributes_t& a, uint32_t k_addr,
                                             uint32_t a_addr, uint32_t ka_addr,
                                             uint32_t done, uint32_t count) {
        const uint32_t t0_k  = a.k_bcast  ? 0u : done;
        const uint32_t t0_a  = a.a_bcast  ? 0u : done;
        const uint32_t t0_ka = a.ka_bcast ? 0u : done;
        std::vector<uint32_t> v{k_addr, a_addr, ka_addr, count, (uint32_t)a.tpg,
                                a.k_bcast, a.a_bcast, a.ka_bcast};
        append_view_args(v, a.k_kind,  t0_k,  to_desc(a.k));
        append_view_args(v, a.a_kind,  t0_a,  to_desc(a.a));
        append_view_args(v, a.ka_kind, t0_ka, to_desc(a.ka));
        return v;
    }
    static std::vector<uint32_t> compute_args(const operation_attributes_t& a, uint32_t done,
                                              uint32_t count) {
        return {count, (uint32_t)a.tpg, done % (uint32_t)a.tpg, a.k_bcast, a.a_bcast, a.ka_bcast};
    }
    static std::vector<uint32_t> writer_args(const operation_attributes_t& a, uint32_t o_addr,
                                             uint32_t t0, uint32_t count) {
        std::vector<uint32_t> v{o_addr, count};
        append_view_args(v, a.out_kind, t0, to_desc(a.o));
        return v;
    }

    static cached_program_t create(
        const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& out) {
        Program prog = CreateProgram();
        const uint32_t wt    = (uint32_t)(a.n_otr * a.n_otc);
        const uint32_t tpg   = (uint32_t)a.tpg;
        const bool     bcast = a.k_bcast || a.a_bcast || a.ka_bcast;
        const uint32_t units = bcast ? wt / tpg : wt;

        MeshDevice* dev = t.k.device();
        auto grid = dev->compute_with_storage_grid_size();
        auto [num_cores, all_cores, cg1, cg2, upc1, upc2] =
            tt::tt_metal::split_work_to_cores(grid, units);

        const uint32_t k_pages  = a.k_bcast  ? (uint32_t)a.tpg : 4u;
        const uint32_t a_pages  = a.a_bcast  ? (uint32_t)a.tpg : 4u;
        const uint32_t ka_pages = a.ka_bcast ? (uint32_t)a.tpg : 4u;
        mkcb(prog, all_cores, tt::CBIndex::c_0,  2048u, k_pages,  tt::DataFormat::Float16_b);
        mkcb(prog, all_cores, tt::CBIndex::c_1,  2048u, a_pages,  tt::DataFormat::Float16_b);
        mkcb(prog, all_cores, tt::CBIndex::c_2,  2048u, ka_pages, tt::DataFormat::Float16_b);
        mkcb(prog, all_cores, tt::CBIndex::c_16, 2048u, 4, tt::DataFormat::Float16_b);
        mkcb(prog, all_cores, tt::CBIndex::c_25, 2048u, 4, tt::DataFormat::Float16_b);
        mkcb(prog, all_cores, tt::CBIndex::c_26, 2048u, 4, tt::DataFormat::Float16_b);
        mkcb(prog, all_cores, tt::CBIndex::c_27, 2048u, 4, tt::DataFormat::Float16_b);
        mkcb(prog, all_cores, tt::CBIndex::c_31, 2048u, 1, tt::DataFormat::Float16_b);

        const std::string ttnn_inc = std::string(getenv("TT_METAL_HOME")) + "/ttnn";
        std::vector<uint32_t> rct;
        TensorAccessorArgs(*t.k.buffer()).append_to(rct);
        TensorAccessorArgs(*t.a.buffer()).append_to(rct);
        TensorAccessorArgs(*t.ka.buffer()).append_to(rct);
        KernelHandle reader = CreateKernel(prog, TTPRM_ROOT "/kernel/view_ternary_reader.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                               .compile_args = rct,
                               .defines = {{"TTPRM_A_VIEW", view_type_name(a.k_kind)},
                                           {"TTPRM_B_VIEW", view_type_name(a.a_kind)},
                                           {"TTPRM_C_VIEW", view_type_name(a.ka_kind)}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        std::vector<uint32_t> wct; TensorAccessorArgs(*out.buffer()).append_to(wct);
        KernelHandle writer = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_writer.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                               .compile_args = wct,
                               .defines = {{"TTPRM_OUT_VIEW", view_type_name(a.out_kind)},
                                           {"TTPRM_OUT_CB", "16"}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        KernelHandle compute = CreateKernel(prog, TTPRM_ROOT "/kernel/view_k_update_compute.cpp", all_cores,
            ComputeConfig{.compile_args = {}});

        const uint32_t k_addr  = (uint32_t)t.k.buffer()->address();
        const uint32_t a_addr  = (uint32_t)t.a.buffer()->address();
        const uint32_t ka_addr = (uint32_t)t.ka.buffer()->address();
        const uint32_t o_addr  = (uint32_t)out.buffer()->address();
        const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
        uint32_t done_u = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord cc = cores[i];
            const uint32_t here_u = cg1.contains(cc) ? upc1 : upc2;
            const uint32_t done_t  = bcast ? done_u * tpg : done_u;
            const uint32_t count_t = bcast ? here_u * tpg : here_u;
            SetRuntimeArgs(prog, reader, cc, reader_args(a, k_addr, a_addr, ka_addr, done_t, count_t));
            SetRuntimeArgs(prog, writer, cc, writer_args(a, o_addr, done_t, count_t));
            SetRuntimeArgs(prog, compute, cc, compute_args(a, done_t, count_t));
            done_u += here_u;
        }
        return {std::move(prog), shared_variables_t{reader, writer, all_cores}};
    }

    static void override_runtime_arguments(
        cached_program_t& cp, const operation_attributes_t&, const tensor_args_t& t,
        tensor_return_value_t& out) {
        auto& prog = cp.program;
        const uint32_t k_addr  = (uint32_t)t.k.buffer()->address();
        const uint32_t a_addr  = (uint32_t)t.a.buffer()->address();
        const uint32_t ka_addr = (uint32_t)t.ka.buffer()->address();
        const uint32_t o_addr  = (uint32_t)out.buffer()->address();
        for (const auto& range : cp.shared_variables.all_cores.ranges())
            for (const auto& core : range) {
                auto& ra = GetRuntimeArgs(prog, cp.shared_variables.reader, core);
                ra[0] = k_addr; ra[1] = a_addr; ra[2] = ka_addr;
                GetRuntimeArgs(prog, cp.shared_variables.writer, core)[0] = o_addr;
            }
    }
};
}  // namespace program

struct ViewKUpdateDeviceOperation {
    using operation_attributes_t = vku::operation_attributes_t;
    using tensor_args_t          = vku::tensor_args_t;
    using spec_return_value_t    = vku::spec_return_value_t;
    using tensor_return_value_t  = vku::tensor_return_value_t;
    using program_factory_t      = std::variant<program::ViewKUpdateFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return program::ViewKUpdateFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t& a, const tensor_args_t&) {
        return tile_spec(a.out_rows, a.out_cols, a.memory_config);
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
        return resolve_output(t.opt_out, compute_output_specs(a, t), t.k.device());
    }
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& k, const ttnn::Tensor& av, const ttnn::Tensor& ka,
        const AffineDesc& dk, const AffineDesc& da, const AffineDesc& dka,
        const AffineDesc& dout, int32_t out_rows, int32_t out_cols, int32_t tpg,
        uint32_t k_bcast, uint32_t a_bcast, uint32_t ka_bcast,
        std::optional<ttnn::Tensor> opt_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        operation_attributes_t at;
        at.k_kind = kind_of(dk); at.a_kind = kind_of(da); at.ka_kind = kind_of(dka);
        at.out_kind = kind_of(dout);
        at.out_rows = out_rows; at.out_cols = out_cols;
        at.n_otr = (int32_t)dout.n_otr; at.n_otc = (int32_t)dout.n_otc;
        at.tpg = tpg; at.k_bcast = k_bcast; at.a_bcast = a_bcast; at.ka_bcast = ka_bcast;
        at.memory_config = output_memory_config(opt_out, memory_config);
        at.k = from_desc(dk); at.a = from_desc(da); at.ka = from_desc(dka); at.o = from_desc(dout);
        return { at, tensor_args_t{k, av, ka, std::move(opt_out)} };
    }
};

}  // namespace vku

// Fused norm op: out = NORM(view_x(x)) per head, laid out by an output view.
// The reduction unit is one tile-row; optional gamma/beta are per-lane.
//   L2 (op 0): y = x * rsqrt(Σx² + eps)                       (no affine)
//   LN (op 1): y = (x-mean)*rsqrt(var+eps) ; y = y*gamma+beta (affine optional)
namespace vn {

enum class NormMode : uint32_t { L2 = 0, Rms = 1, Layer = 2 };
enum class NormAffine : uint32_t { None = 0, Scale = 1, ScaleBias = 2 };

struct operation_attributes_t {
    NormMode mode = NormMode::L2;          // kernel define selector
    uint32_t x_kind = 0, out_kind = 0;     // per-endpoint TensorView KIND (kernel defines)
    int32_t out_rows = 0, out_cols = 0;    // OUTPUT tensor logical shape (== input)
    int32_t n_otr = 0, n_otc = 0;          // OUTPUT tile grid: n_otr row-blocks of n_otc col-tiles
    uint32_t rscale_bits = 0, eps_bits = 0;
    NormAffine affine = NormAffine::None;
    uint32_t wide = 0;                     // 1 = streaming/lean-resident wide kernel (big ctiles)
    uint32_t global = 0;
    MemoryConfig memory_config;
    ViewArgs7 x, o;
};
struct tensor_args_t { const ttnn::Tensor& x; const ttnn::Tensor& gamma; const ttnn::Tensor& beta; std::optional<ttnn::Tensor> opt_out; };
using spec_return_value_t   = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {
struct ViewNormFactory {
    struct shared_variables_t { KernelHandle reader, writer; CoreRangeSet all_cores; };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    // reader: [x_addr, gamma_addr, beta_addr, n_rb, ctiles, rscale_bits, eps_bits, affine_kind] + x view.
    static std::vector<uint32_t> reader_args(const operation_attributes_t& a, uint32_t x_addr,
                                             uint32_t g_addr, uint32_t b_addr,
                                             uint32_t rb0, uint32_t n_rb) {
        std::vector<uint32_t> v{x_addr, g_addr, b_addr, n_rb, (uint32_t)a.n_otc,
                                a.rscale_bits, a.eps_bits, (uint32_t)a.affine};
        append_view_args(v, a.x_kind, rb0 * (uint32_t)a.n_otc, to_desc(a.x));   // x t0 = first tile
        return v;
    }
    // compute: [n_rb, ctiles, affine_kind, eps_bits, rscale_bits].  scalar tiles fill_tile'd here.
    static std::vector<uint32_t> compute_args(const operation_attributes_t& a, uint32_t n_rb) {
        return {n_rb, (uint32_t)a.n_otc, (uint32_t)a.affine, a.eps_bits, a.rscale_bits};
    }
    // writer: [o_addr, n_tiles] + output view block.
    static std::vector<uint32_t> writer_args(const operation_attributes_t& a, uint32_t o_addr,
                                             uint32_t rb0, uint32_t n_rb) {
        std::vector<uint32_t> v{o_addr, n_rb * (uint32_t)a.n_otc};
        append_view_args(v, a.out_kind, rb0 * (uint32_t)a.n_otc, to_desc(a.o));
        return v;
    }

    // Folded-global ABI: one core, all folded tiles, one reduction unit.
    static uint32_t g_ntiles(const operation_attributes_t& a) { return (uint32_t)(a.n_otr * a.n_otc); }
    // reader: [x_addr, gamma_addr, beta_addr, ntiles, affine_kind] + folded view block (t0=0).
    static std::vector<uint32_t> g_reader_args(const operation_attributes_t& a, uint32_t x_addr,
                                               uint32_t g_addr, uint32_t b_addr) {
        std::vector<uint32_t> v{x_addr, g_addr, b_addr, g_ntiles(a), (uint32_t)a.affine};
        append_view_args(v, a.x_kind, 0, to_desc(a.x));
        return v;
    }
    // compute: [ntiles, affine_kind, eps_bits, rscale_bits].
    static std::vector<uint32_t> g_compute_args(const operation_attributes_t& a) {
        return {g_ntiles(a), (uint32_t)a.affine, a.eps_bits, a.rscale_bits};
    }
    // writer: [o_addr, ntiles] + folded output view block (t0=0).
    static std::vector<uint32_t> g_writer_args(const operation_attributes_t& a, uint32_t o_addr) {
        std::vector<uint32_t> v{o_addr, g_ntiles(a)};
        append_view_args(v, a.out_kind, 0, to_desc(a.o));
        return v;
    }

    static cached_program_t create(
        const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& out) {
        Program prog = CreateProgram();
        const std::string ttnn_inc0 = std::string(getenv("TT_METAL_HOME")) + "/ttnn";

        // Folded-global path: one core reduces all folded tiles as one norm.
        // x/gamma/beta share the same folded view; writer scatters back through
        // the output view.
        if (a.global) {
            const uint32_t ntiles = g_ntiles(a);
            CoreRange core(CoreCoord{0, 0}, CoreCoord{0, 0});
            CoreRangeSet all_cores(core);
            mkcb(prog, all_cores, tt::CBIndex::c_0,  2048u, ntiles, tt::DataFormat::Float16_b);  // cb_x
            mkcb(prog, all_cores, tt::CBIndex::c_2,  2048u, ntiles, tt::DataFormat::Float16_b);  // cb_gamma
            mkcb(prog, all_cores, tt::CBIndex::c_3,  2048u, ntiles, tt::DataFormat::Float16_b);  // cb_beta
            mkcb(prog, all_cores, tt::CBIndex::c_16, 2048u, ntiles, tt::DataFormat::Float16_b);  // cb_out
            for (auto cb : {tt::CBIndex::c_25, tt::CBIndex::c_26, tt::CBIndex::c_27, tt::CBIndex::c_28})
                mkcb(prog, all_cores, cb, 2048u, ntiles, tt::DataFormat::Float16_b);             // sq/xmm/norm/tmp
            mkcb(prog, all_cores, tt::CBIndex::c_4,  2048u, 1, tt::DataFormat::Float16_b);        // cb_scaler
            mkcb(prog, all_cores, tt::CBIndex::c_5,  2048u, 1, tt::DataFormat::Float16_b);        // cb_eps
            mkcb(prog, all_cores, tt::CBIndex::c_31, 2048u, 1, tt::DataFormat::Float16_b);        // cb_zero
            for (auto cb : {tt::CBIndex::c_20, tt::CBIndex::c_23, tt::CBIndex::c_24})
                mkcb(prog, all_cores, cb, 2048u, 1, tt::DataFormat::Float16_b);                  // mean/var/rstd

            std::vector<uint32_t> rct;
            TensorAccessorArgs(*t.x.buffer()).append_to(rct);
            TensorAccessorArgs(*t.gamma.buffer()).append_to(rct);
            TensorAccessorArgs(*t.beta.buffer()).append_to(rct);
            KernelHandle reader = CreateKernel(prog, TTPRM_ROOT "/kernel/view_norm_global_reader.cpp", all_cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                                   .compile_args = rct,
                                   .defines = {{"TTPRM_X_VIEW", view_type_name(a.x_kind)}},
                                   .compiler_include_paths = {TTPRM_ROOT, ttnn_inc0}});
            std::vector<uint32_t> wct; TensorAccessorArgs(*out.buffer()).append_to(wct);
            KernelHandle writer = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_writer.cpp", all_cores,
                DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                                   .compile_args = wct,
                                   .defines = {{"TTPRM_OUT_VIEW", view_type_name(a.out_kind)},
                                               {"TTPRM_OUT_CB", "16"}},
                                   .compiler_include_paths = {TTPRM_ROOT, ttnn_inc0}});
            MathFidelity fid = MathFidelity::HiFi4; bool fp32d = true;
            if (const char* f = std::getenv("TTPRM_NORM_FIDELITY")) {
                int v = atoi(f);
                fid = v <= 0 ? MathFidelity::LoFi : v == 2 ? MathFidelity::HiFi2
                                                  : v == 3 ? MathFidelity::HiFi3 : MathFidelity::HiFi4;
            }
            if (const char* p = std::getenv("TTPRM_NORM_FP32")) fp32d = atoi(p) != 0;
            const char* mode_def = a.mode == NormMode::Layer ? "TTPRM_LN"
                                : a.mode == NormMode::Rms   ? "TTPRM_RMS"
                                                            : "TTPRM_L2";
            std::map<std::string, std::string> cdefs = {{mode_def, "1"}};
            KernelHandle compute = CreateKernel(prog, TTPRM_ROOT "/kernel/view_norm_global_compute.cpp", all_cores,
                ComputeConfig{.math_fidelity = fid, .fp32_dest_acc_en = fp32d,
                              .math_approx_mode = false, .defines = cdefs});

            const uint32_t x_addr = (uint32_t)t.x.buffer()->address();
            const uint32_t g_addr = (uint32_t)t.gamma.buffer()->address();
            const uint32_t b_addr = (uint32_t)t.beta.buffer()->address();
            const uint32_t o_addr = (uint32_t)out.buffer()->address();
            CoreCoord cc{0, 0};
            SetRuntimeArgs(prog, reader, cc, g_reader_args(a, x_addr, g_addr, b_addr));
            SetRuntimeArgs(prog, writer, cc, g_writer_args(a, o_addr));
            SetRuntimeArgs(prog, compute, cc, g_compute_args(a));
            return {std::move(prog), shared_variables_t{reader, writer, all_cores}};
        }

        // Work unit = ROW-BLOCK (one tile-row): the head reduction needs the whole row together, so
        // it cannot be split below a tile-row. Distribute the n_otr row-blocks across the grid.
        const uint32_t n_rb_total = (uint32_t)a.n_otr;
        const uint32_t ctiles     = (uint32_t)a.n_otc;

        MeshDevice* dev = t.x.device();
        auto grid = dev->compute_with_storage_grid_size();
        auto [num_cores, all_cores, cg1, cg2, upc1, upc2] =
            tt::tt_metal::split_work_to_cores(grid, n_rb_total);

        // Wide rows keep one row-block resident and stream gamma/beta. Narrow
        // rows double-buffer cb_x/cb_out and size scratch for the row.
        if (a.wide) {
            mkcb(prog, all_cores, tt::CBIndex::c_0,  2048u, ctiles, tt::DataFormat::Float16_b);  // cb_x
            mkcb(prog, all_cores, tt::CBIndex::c_2,  2048u, 4, tt::DataFormat::Float16_b);        // cb_gamma
            mkcb(prog, all_cores, tt::CBIndex::c_3,  2048u, 4, tt::DataFormat::Float16_b);        // cb_beta
            mkcb(prog, all_cores, tt::CBIndex::c_16, 2048u, 4, tt::DataFormat::Float16_b);        // cb_out
            mkcb(prog, all_cores, tt::CBIndex::c_25, 2048u, ctiles, tt::DataFormat::Float16_b);   // cb_sq
            for (auto cb : {tt::CBIndex::c_26, tt::CBIndex::c_27, tt::CBIndex::c_28})
                mkcb(prog, all_cores, cb, 2048u, 4, tt::DataFormat::Float16_b);                   // xmm/nrm/tmp
        } else {
            const uint32_t ct2 = ctiles * 2;  // double-buffer the row's tiles
            mkcb(prog, all_cores, tt::CBIndex::c_0,  2048u, ct2,    tt::DataFormat::Float16_b);  // cb_x
            mkcb(prog, all_cores, tt::CBIndex::c_2,  2048u, ctiles, tt::DataFormat::Float16_b);  // cb_gamma
            mkcb(prog, all_cores, tt::CBIndex::c_3,  2048u, ctiles, tt::DataFormat::Float16_b);  // cb_beta
            mkcb(prog, all_cores, tt::CBIndex::c_16, 2048u, ct2,    tt::DataFormat::Float16_b);  // cb_out
            for (auto cb : {tt::CBIndex::c_25, tt::CBIndex::c_26, tt::CBIndex::c_27, tt::CBIndex::c_28})
                mkcb(prog, all_cores, cb, 2048u, ctiles, tt::DataFormat::Float16_b);
        }
        mkcb(prog, all_cores, tt::CBIndex::c_4,  2048u, 1, tt::DataFormat::Float16_b);       // cb_scaler
        mkcb(prog, all_cores, tt::CBIndex::c_5,  2048u, 1, tt::DataFormat::Float16_b);       // cb_eps
        mkcb(prog, all_cores, tt::CBIndex::c_31, 2048u, 1, tt::DataFormat::Float16_b);       // cb_zero
        // scratch singles (mean/msq/m2/var/rstd)
        for (auto cb : {tt::CBIndex::c_20, tt::CBIndex::c_21, tt::CBIndex::c_22,
                        tt::CBIndex::c_23, tt::CBIndex::c_24})
            mkcb(prog, all_cores, cb, 2048u, 2, tt::DataFormat::Float16_b);

        const std::string ttnn_inc = std::string(getenv("TT_METAL_HOME")) + "/ttnn";
        std::vector<uint32_t> rct;
        TensorAccessorArgs(*t.x.buffer()).append_to(rct);
        TensorAccessorArgs(*t.gamma.buffer()).append_to(rct);   // gamma's compile args follow x's
        TensorAccessorArgs(*t.beta.buffer()).append_to(rct);    // beta's follow gamma's
        const char* reader_src = a.wide ? TTPRM_ROOT "/kernel/view_norm_wide_reader.cpp"
                                        : TTPRM_ROOT "/kernel/view_norm_reader.cpp";
        KernelHandle reader = CreateKernel(prog, reader_src, all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                               .compile_args = rct,
                               .defines = {{"TTPRM_X_VIEW", view_type_name(a.x_kind)}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        std::vector<uint32_t> wct; TensorAccessorArgs(*out.buffer()).append_to(wct);
        KernelHandle writer = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_writer.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                               .compile_args = wct,
                               .defines = {{"TTPRM_OUT_VIEW", view_type_name(a.out_kind)},
                                           {"TTPRM_OUT_CB", "16"}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        // fp32 dest accumulation: the reduce + (x-mean)² path is precision-sensitive (bf16 dest
        // loses ~1 digit per op -> LN drifts to ~0.998 PCC). fp32 dest holds the running sums and
        // the centered square in full precision (what ggml's own l2_norm does). HiFi4 for the muls.
        const char* compute_src = a.wide ? TTPRM_ROOT "/kernel/view_norm_wide_compute.cpp"
                                         : TTPRM_ROOT "/kernel/view_norm_compute.cpp";
        // Math fidelity / fp32-dest are tunable for benchmarking (HiFi4 is 4x math passes; the
        // reduce + (x-mean)² path is precision-sensitive but l2's sum-of-squares may tolerate less).
        MathFidelity fid = MathFidelity::HiFi4;
        bool fp32d = true;
        if (const char* f = std::getenv("TTPRM_NORM_FIDELITY")) {
            int v = atoi(f);
            fid = v <= 0 ? MathFidelity::LoFi : v == 2 ? MathFidelity::HiFi2
                                              : v == 3 ? MathFidelity::HiFi3 : MathFidelity::HiFi4;
        }
        if (const char* p = std::getenv("TTPRM_NORM_FP32")) fp32d = atoi(p) != 0;
        const char* mode_def = a.mode == NormMode::Layer ? "TTPRM_LN"
                            : a.mode == NormMode::Rms   ? "TTPRM_RMS"
                                                        : "TTPRM_L2";
        std::map<std::string, std::string> cdefs = {{mode_def, "1"}};
        KernelHandle compute = CreateKernel(prog, compute_src, all_cores,
            ComputeConfig{.math_fidelity = fid,
                          .fp32_dest_acc_en = fp32d,
                          .math_approx_mode = false,
                          .defines = cdefs});

        const uint32_t x_addr = (uint32_t)t.x.buffer()->address();
        const uint32_t g_addr = (uint32_t)t.gamma.buffer()->address();
        const uint32_t b_addr = (uint32_t)t.beta.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
        uint32_t done_rb = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord cc = cores[i];
            const uint32_t here_rb = cg1.contains(cc) ? upc1 : upc2;
            SetRuntimeArgs(prog, reader, cc, reader_args(a, x_addr, g_addr, b_addr, done_rb, here_rb));
            SetRuntimeArgs(prog, writer, cc, writer_args(a, o_addr, done_rb, here_rb));
            SetRuntimeArgs(prog, compute, cc, compute_args(a, here_rb));
            done_rb += here_rb;
        }
        return {std::move(prog), shared_variables_t{reader, writer, all_cores}};
    }

    static void override_runtime_arguments(
        cached_program_t& cp, const operation_attributes_t&, const tensor_args_t& t,
        tensor_return_value_t& out) {
        auto& prog = cp.program;
        const uint32_t x_addr = (uint32_t)t.x.buffer()->address();
        const uint32_t g_addr = (uint32_t)t.gamma.buffer()->address();
        const uint32_t b_addr = (uint32_t)t.beta.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        for (const auto& range : cp.shared_variables.all_cores.ranges())
            for (const auto& core : range) {
                auto& ra = GetRuntimeArgs(prog, cp.shared_variables.reader, core);
                ra[0] = x_addr; ra[1] = g_addr; ra[2] = b_addr;
                GetRuntimeArgs(prog, cp.shared_variables.writer, core)[0] = o_addr;
            }
    }
};
}  // namespace program

struct ViewNormDeviceOperation {
    using operation_attributes_t = vn::operation_attributes_t;
    using tensor_args_t          = vn::tensor_args_t;
    using spec_return_value_t    = vn::spec_return_value_t;
    using tensor_return_value_t  = vn::tensor_return_value_t;
    using program_factory_t      = std::variant<program::ViewNormFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return program::ViewNormFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t& a, const tensor_args_t&) {
        return tile_spec(a.out_rows, a.out_cols, a.memory_config);
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
        return resolve_output(t.opt_out, compute_output_specs(a, t), t.x.device());
    }
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        NormMode mode, const ttnn::Tensor& x, const ttnn::Tensor& gamma, const ttnn::Tensor& beta,
        const AffineDesc& dx, const AffineDesc& dout, int32_t out_rows, int32_t out_cols,
        uint32_t rscale_bits, uint32_t eps_bits, NormAffine affine, uint32_t wide,
        uint32_t global = 0,
        std::optional<ttnn::Tensor> opt_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        operation_attributes_t at;
        at.mode = mode;
        at.x_kind = kind_of(dx); at.out_kind = kind_of(dout);
        at.out_rows = out_rows; at.out_cols = out_cols;
        at.n_otr = (int32_t)dout.n_otr; at.n_otc = (int32_t)dout.n_otc;
        at.rscale_bits = rscale_bits; at.eps_bits = eps_bits; at.affine = affine;
        at.wide = wide; at.global = global;
        at.memory_config = output_memory_config(opt_out, memory_config);
        at.x = from_desc(dx); at.o = from_desc(dout);
        return { at, tensor_args_t{x, gamma, beta, std::move(opt_out)} };
    }
};

}  // namespace vn

// Row-index op: GET_ROWS / SET_ROWS as INT32-indexed whole-tile copies.
// Each row is T contiguous tiles. The indexed endpoint picks
// page = idx[ord/T]*T + ord%T; the other endpoint walks densely.
//   GATHER  (get_rows): reader indexed over src, writer dense to a fresh [M,inner] output.
//   SCATTER (set_rows): reader dense over src, writer indexed into dst IN PLACE (the cache).
// No relayout, pad seed, or odometer is involved.
namespace ri {

constexpr uint32_t MODE_GATHER = 0, MODE_SCATTER = 1;

struct operation_attributes_t {
    uint32_t mode = MODE_GATHER;
    int32_t  T = 1;                       // whole tiles per indexed row
    int32_t  work_tiles = 0;              // M*T (total output tiles)
    int32_t  idx_pages = 1;               // index buffer page count (loaded to L1)
    int32_t  n_idx_rows = 0;              // valid rows on the INDEXED side (bounds: idx in [0,n))
    int32_t  out_rank = 2;
    int32_t  os0 = 0, os1 = 0, os2 = 0, os3 = 0;  // output logical shape (GATHER); dst spec for SCATTER
    MemoryConfig memory_config;
};
struct tensor_args_t { const ttnn::Tensor& src; const ttnn::Tensor& idx; std::optional<ttnn::Tensor> dst; };
using spec_return_value_t   = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {
struct RowIndexFactory {
    struct shared_variables_t { KernelHandle reader, writer; CoreRangeSet all_cores; };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static constexpr uint32_t K = 8;          // batched-issue chunk: issue K tile transfers per barrier
    static constexpr uint32_t IO_DEPTH = 16;  // cb_io ring depth (multiple of K); deeper gave no gain

    // [data_addr, count, t0, T, K] + (indexed side only) [idx_addr, idx_pages, n_idx_rows].
    static std::vector<uint32_t> args(uint32_t addr, uint32_t count, uint32_t t0, uint32_t T,
                                      bool indexed, uint32_t idx_addr, uint32_t idx_pages,
                                      uint32_t n_idx_rows) {
        std::vector<uint32_t> v{addr, count, t0, T, K};
        if (indexed) { v.push_back(idx_addr); v.push_back(idx_pages); v.push_back(n_idx_rows); }
        return v;
    }

    static cached_program_t create(
        const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& out) {
        Program prog = CreateProgram();
        const bool gather = (a.mode == MODE_GATHER);
        const uint32_t work_tiles = (uint32_t)a.work_tiles;

        MeshDevice* dev = t.src.device();
        auto grid = dev->compute_with_storage_grid_size();
        auto [num_cores, all_cores, cg1, cg2, upc1, upc2] =
            tt::tt_metal::split_work_to_cores(grid, work_tiles);

        const uint32_t idx_pb = (uint32_t)t.idx.buffer()->aligned_page_size();
        // cb_io is a multiple of the K-tile chunk, so each chunk remains
        // contiguous in the ring while reader and writer overlap.
        mkcb(prog, all_cores, tt::CBIndex::c_0, 2048u, IO_DEPTH, tt::DataFormat::Float16_b);  // io pipeline
        mkcb(prog, all_cores, tt::CBIndex::c_1, idx_pb, (uint32_t)a.idx_pages, tt::DataFormat::UInt32);  // index L1

        const std::string ttnn_inc = std::string(getenv("TT_METAL_HOME")) + "/ttnn";
        // Reader is indexed for gather mode and dense for scatter mode.
        std::vector<uint32_t> rct; TensorAccessorArgs(*t.src.buffer()).append_to(rct);
        if (gather) TensorAccessorArgs(*t.idx.buffer()).append_to(rct);
        std::map<std::string, std::string> rdef;
        if (gather) rdef["ROW_INDEXED"] = "1";
        KernelHandle reader = CreateKernel(prog, TTPRM_ROOT "/kernel/row_index_reader.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                               .compile_args = rct, .defines = rdef,
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        // Writer is indexed for scatter mode and dense for gather mode.
        std::vector<uint32_t> wct; TensorAccessorArgs(*out.buffer()).append_to(wct);
        if (!gather) TensorAccessorArgs(*t.idx.buffer()).append_to(wct);
        std::map<std::string, std::string> wdef;
        if (!gather) wdef["ROW_INDEXED"] = "1";
        KernelHandle writer = CreateKernel(prog, TTPRM_ROOT "/kernel/row_index_writer.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                               .compile_args = wct, .defines = wdef,
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});

        const uint32_t s_addr   = (uint32_t)t.src.buffer()->address();
        const uint32_t o_addr   = (uint32_t)out.buffer()->address();
        const uint32_t idx_addr = (uint32_t)t.idx.buffer()->address();
        const uint32_t idx_pages = (uint32_t)a.idx_pages;
        const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
        uint32_t done = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord cc = cores[i];
            const uint32_t here = cg1.contains(cc) ? upc1 : upc2;
            const uint32_t n_idx_rows = (uint32_t)a.n_idx_rows;
            SetRuntimeArgs(prog, reader, cc, args(s_addr, here, done, (uint32_t)a.T,  gather, idx_addr, idx_pages, n_idx_rows));
            SetRuntimeArgs(prog, writer, cc, args(o_addr, here, done, (uint32_t)a.T, !gather, idx_addr, idx_pages, n_idx_rows));
            done += here;
        }
        return {std::move(prog), shared_variables_t{reader, writer, all_cores}};
    }

    static void override_runtime_arguments(
        cached_program_t& cp, const operation_attributes_t& a, const tensor_args_t& t,
        tensor_return_value_t& out) {
        auto& prog = cp.program;
        const uint32_t s_addr   = (uint32_t)t.src.buffer()->address();
        const uint32_t o_addr   = (uint32_t)out.buffer()->address();
        const uint32_t idx_addr = (uint32_t)t.idx.buffer()->address();
        const bool gather = (a.mode == MODE_GATHER);
        for (const auto& range : cp.shared_variables.all_cores.ranges())
            for (const auto& core : range) {
                auto& ra = GetRuntimeArgs(prog, cp.shared_variables.reader, core);
                auto& wa = GetRuntimeArgs(prog, cp.shared_variables.writer, core);
                ra[0] = s_addr; wa[0] = o_addr;
                if (gather) ra[5] = idx_addr; else wa[5] = idx_addr;  // patch the index buffer addr (arg 5)
            }
    }
};
}  // namespace program

struct RowIndexDeviceOperation {
    using operation_attributes_t = ri::operation_attributes_t;
    using tensor_args_t          = ri::tensor_args_t;
    using spec_return_value_t    = ri::spec_return_value_t;
    using tensor_return_value_t  = ri::tensor_return_value_t;
    using program_factory_t      = std::variant<program::RowIndexFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return program::RowIndexFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t& a, const tensor_args_t& t) {
        if (a.mode == MODE_SCATTER) return t.dst->tensor_spec();   // in-place: dst's own spec
        ttnn::SmallVector<uint32_t> dims;
        const int32_t os[4] = {a.os0, a.os1, a.os2, a.os3};
        for (int i = 0; i < a.out_rank; i++) dims.push_back((uint32_t)os[i]);
        return TensorSpec(ttnn::Shape(dims),
                          TensorLayout(DataType::BFLOAT16, PageConfig(tt::tt_metal::Layout::TILE), a.memory_config));
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
        if (a.mode == MODE_SCATTER) return *t.dst;                 // in-place: write into the cache
        return create_device_tensor(compute_output_specs(a, t), t.src.device());
    }
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        uint32_t mode, const ttnn::Tensor& src, const ttnn::Tensor& idx,
        std::optional<ttnn::Tensor> dst, int32_t T, int32_t work_tiles, int32_t idx_pages,
        int32_t n_idx_rows, int32_t out_rank, int32_t os0, int32_t os1, int32_t os2, int32_t os3,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        operation_attributes_t at;
        at.mode = mode; at.T = T; at.work_tiles = work_tiles; at.idx_pages = idx_pages;
        at.n_idx_rows = n_idx_rows;
        at.out_rank = out_rank; at.os0 = os0; at.os1 = os1; at.os2 = os2; at.os3 = os3;
        at.memory_config = output_memory_config(dst, memory_config);
        return { at, tensor_args_t{src, idx, std::move(dst)} };
    }
};

}  // namespace ri

// Fused mul_l2_norm: y = l2_norm(a * b) per head.
// Mirrors ViewNorm with a second input and product premultiply.
namespace vml {

struct operation_attributes_t {
    uint32_t a_kind = 0, b_kind = 0, out_kind = 0;
    int32_t out_rows = 0, out_cols = 0;
    int32_t n_otr = 0, n_otc = 0;
    uint32_t rscale_bits = 0, eps_bits = 0;
    MemoryConfig memory_config;
    ViewArgs7 a, b, o;
};
struct tensor_args_t { const ttnn::Tensor& a; const ttnn::Tensor& b; std::optional<ttnn::Tensor> opt_out; };
using spec_return_value_t   = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {
struct ViewMulL2Factory {
    struct shared_variables_t { KernelHandle reader, writer; CoreRangeSet all_cores; };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static std::vector<uint32_t> reader_args(const operation_attributes_t& a, uint32_t a_addr,
                                             uint32_t b_addr, uint32_t rb0, uint32_t n_rb) {
        std::vector<uint32_t> v{a_addr, b_addr, n_rb, (uint32_t)a.n_otc};
        append_view_args(v, a.a_kind, rb0 * (uint32_t)a.n_otc, to_desc(a.a));
        append_view_args(v, a.b_kind, rb0 * (uint32_t)a.n_otc, to_desc(a.b));
        return v;
    }
    static std::vector<uint32_t> compute_args(const operation_attributes_t& a, uint32_t n_rb) {
        return {n_rb, (uint32_t)a.n_otc, a.eps_bits, a.rscale_bits};
    }
    static std::vector<uint32_t> writer_args(const operation_attributes_t& a, uint32_t o_addr,
                                             uint32_t rb0, uint32_t n_rb) {
        std::vector<uint32_t> v{o_addr, n_rb * (uint32_t)a.n_otc};
        append_view_args(v, a.out_kind, rb0 * (uint32_t)a.n_otc, to_desc(a.o));
        return v;
    }

    static cached_program_t create(
        const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& out) {
        Program prog = CreateProgram();
        const uint32_t n_rb_total = (uint32_t)a.n_otr;
        const uint32_t ctiles     = (uint32_t)a.n_otc;
        const uint32_t ct2        = ctiles * 2;

        MeshDevice* dev = t.a.device();
        auto grid = dev->compute_with_storage_grid_size();
        auto [num_cores, all_cores, cg1, cg2, upc1, upc2] =
            tt::tt_metal::split_work_to_cores(grid, n_rb_total);

        mkcb(prog, all_cores, tt::CBIndex::c_0,  2048u, ct2,    tt::DataFormat::Float16_b);  // cb_a
        mkcb(prog, all_cores, tt::CBIndex::c_6,  2048u, ct2,    tt::DataFormat::Float16_b);  // cb_b
        mkcb(prog, all_cores, tt::CBIndex::c_16, 2048u, ct2,    tt::DataFormat::Float16_b);  // cb_out
        mkcb(prog, all_cores, tt::CBIndex::c_4,  2048u, 1, tt::DataFormat::Float16_b);       // cb_scaler
        mkcb(prog, all_cores, tt::CBIndex::c_5,  2048u, 1, tt::DataFormat::Float16_b);       // cb_eps
        mkcb(prog, all_cores, tt::CBIndex::c_31, 2048u, 1, tt::DataFormat::Float16_b);       // cb_zero
        mkcb(prog, all_cores, tt::CBIndex::c_25, 2048u, ctiles, tt::DataFormat::Float16_b);  // cb_sq
        mkcb(prog, all_cores, tt::CBIndex::c_26, 2048u, ctiles, tt::DataFormat::Float16_b);  // cb_prod
        mkcb(prog, all_cores, tt::CBIndex::c_21, 2048u, 2, tt::DataFormat::Float16_b);       // cb_msq
        mkcb(prog, all_cores, tt::CBIndex::c_24, 2048u, 2, tt::DataFormat::Float16_b);       // cb_rstd

        const std::string ttnn_inc = std::string(getenv("TT_METAL_HOME")) + "/ttnn";
        std::vector<uint32_t> rct;
        TensorAccessorArgs(*t.a.buffer()).append_to(rct);
        TensorAccessorArgs(*t.b.buffer()).append_to(rct);
        KernelHandle reader = CreateKernel(prog, TTPRM_ROOT "/kernel/view_mull2_reader.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                               .compile_args = rct,
                               .defines = {{"TTPRM_A_VIEW", view_type_name(a.a_kind)},
                                           {"TTPRM_B_VIEW", view_type_name(a.b_kind)}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        std::vector<uint32_t> wct; TensorAccessorArgs(*out.buffer()).append_to(wct);
        KernelHandle writer = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_writer.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                               .compile_args = wct,
                               .defines = {{"TTPRM_OUT_VIEW", view_type_name(a.out_kind)},
                                           {"TTPRM_OUT_CB", "16"}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        MathFidelity fid = MathFidelity::HiFi4;
        bool fp32d = true;
        if (const char* f = std::getenv("TTPRM_NORM_FIDELITY")) {
            int v = atoi(f);
            fid = v <= 0 ? MathFidelity::LoFi : v == 2 ? MathFidelity::HiFi2
                                              : v == 3 ? MathFidelity::HiFi3 : MathFidelity::HiFi4;
        }
        if (const char* p = std::getenv("TTPRM_NORM_FP32")) fp32d = atoi(p) != 0;
        KernelHandle compute = CreateKernel(prog, TTPRM_ROOT "/kernel/view_mull2_compute.cpp", all_cores,
            ComputeConfig{.math_fidelity = fid, .fp32_dest_acc_en = fp32d, .compile_args = {}});

        const uint32_t a_addr = (uint32_t)t.a.buffer()->address();
        const uint32_t b_addr = (uint32_t)t.b.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
        uint32_t done = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord cc = cores[i];
            const uint32_t here = cg1.contains(cc) ? upc1 : upc2;
            SetRuntimeArgs(prog, reader, cc, reader_args(a, a_addr, b_addr, done, here));
            SetRuntimeArgs(prog, writer, cc, writer_args(a, o_addr, done, here));
            SetRuntimeArgs(prog, compute, cc, compute_args(a, here));
            done += here;
        }
        return {std::move(prog), shared_variables_t{reader, writer, all_cores}};
    }

    static void override_runtime_arguments(
        cached_program_t& cp, const operation_attributes_t&, const tensor_args_t& t,
        tensor_return_value_t& out) {
        auto& prog = cp.program;
        const uint32_t a_addr = (uint32_t)t.a.buffer()->address();
        const uint32_t b_addr = (uint32_t)t.b.buffer()->address();
        const uint32_t o_addr = (uint32_t)out.buffer()->address();
        for (const auto& range : cp.shared_variables.all_cores.ranges())
            for (const auto& core : range) {
                auto& ra = GetRuntimeArgs(prog, cp.shared_variables.reader, core);
                ra[0] = a_addr; ra[1] = b_addr;
                GetRuntimeArgs(prog, cp.shared_variables.writer, core)[0] = o_addr;
            }
    }
};
}  // namespace program

struct ViewMulL2DeviceOperation {
    using operation_attributes_t = vml::operation_attributes_t;
    using tensor_args_t          = vml::tensor_args_t;
    using spec_return_value_t    = vml::spec_return_value_t;
    using tensor_return_value_t  = vml::tensor_return_value_t;
    using program_factory_t      = std::variant<program::ViewMulL2Factory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return program::ViewMulL2Factory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t& a, const tensor_args_t&) {
        return tile_spec(a.out_rows, a.out_cols, a.memory_config);
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
        return resolve_output(t.opt_out, compute_output_specs(a, t), t.a.device());
    }
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& a, const ttnn::Tensor& b, const AffineDesc& da, const AffineDesc& db,
        const AffineDesc& dout, int32_t out_rows, int32_t out_cols,
        uint32_t rscale_bits, uint32_t eps_bits, std::optional<ttnn::Tensor> opt_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        operation_attributes_t at;
        at.a_kind = kind_of(da); at.b_kind = kind_of(db); at.out_kind = kind_of(dout);
        at.out_rows = out_rows; at.out_cols = out_cols;
        at.n_otr = (int32_t)dout.n_otr; at.n_otc = (int32_t)dout.n_otc;
        at.rscale_bits = rscale_bits; at.eps_bits = eps_bits;
        at.memory_config = output_memory_config(opt_out, memory_config);
        at.a = from_desc(da); at.b = from_desc(db); at.o = from_desc(dout);
        return { at, tensor_args_t{a, b, std::move(opt_out)} };
    }
};

}  // namespace vml

// Fused RWKV7 rk update.
//   rk[h]     = Σ_i r[h][i] * k[h][i] * r_k[h][i]     (per-head scalar, reduce over the S lanes)
//   out[h][i] = cur[h][i] + v[h][i] * rk[h]           (rk broadcast across the head's lanes)
namespace vrk {

struct operation_attributes_t {
    uint32_t r_kind = 0, k_kind = 0, w_kind = 0, v_kind = 0, cur_kind = 0, out_kind = 0;
    int32_t out_rows = 0, out_cols = 0;
    int32_t n_otr = 0, n_otc = 0;
    uint32_t rscale_bits = 0;
    // r_k token-broadcast: one token-group of row-blocks is reused across
    // token groups. w_bcast == 0 is the decode/full-grid path.
    uint32_t w_bcast = 0;
    int32_t rbs_per_group = 0;
    MemoryConfig memory_config;
    ViewArgs7 r, k, w, v, cur, o;
};
struct tensor_args_t {
    const ttnn::Tensor& r; const ttnn::Tensor& k; const ttnn::Tensor& w;
    const ttnn::Tensor& v; const ttnn::Tensor& cur;
    std::optional<ttnn::Tensor> opt_out;
};
using spec_return_value_t   = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

namespace program {
struct ViewRkFactory {
    struct shared_variables_t { KernelHandle reader, writer; CoreRangeSet all_cores; };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static std::vector<uint32_t> reader_args(const operation_attributes_t& a, uint32_t r_addr,
                                             uint32_t k_addr, uint32_t w_addr, uint32_t v_addr,
                                             uint32_t cur_addr, uint32_t rb0, uint32_t n_rb,
                                             uint32_t w_resident) {
        const uint32_t t0 = rb0 * (uint32_t)a.n_otc;
        // Seed r_k at this core's within-group phase; reader rewinds at group boundaries.
        const uint32_t w_phase0 = a.w_bcast ? (rb0 % (uint32_t)a.rbs_per_group) : 0u;
        const uint32_t t0_w     = w_resident ? 0u : (a.w_bcast ? w_phase0 * (uint32_t)a.n_otc : t0);
        std::vector<uint32_t> v{r_addr, k_addr, w_addr, v_addr, cur_addr, n_rb, (uint32_t)a.n_otc,
                                a.w_bcast ? (uint32_t)a.rbs_per_group : 0u, w_phase0, w_resident};
        append_view_args(v, a.r_kind,   t0,   to_desc(a.r));
        append_view_args(v, a.k_kind,   t0,   to_desc(a.k));
        append_view_args(v, a.w_kind,   t0_w, to_desc(a.w));
        append_view_args(v, a.v_kind,   t0,   to_desc(a.v));
        append_view_args(v, a.cur_kind, t0,   to_desc(a.cur));
        return v;
    }
    static std::vector<uint32_t> compute_args(const operation_attributes_t& a, uint32_t rb0,
                                              uint32_t n_rb, uint32_t w_resident) {
        const uint32_t w_phase0 = a.w_bcast ? (rb0 % (uint32_t)a.rbs_per_group) : 0u;
        return {n_rb, (uint32_t)a.n_otc, a.rscale_bits,
                a.w_bcast ? (uint32_t)a.rbs_per_group : 0u, w_phase0, w_resident};
    }
    static std::vector<uint32_t> writer_args(const operation_attributes_t& a, uint32_t o_addr,
                                             uint32_t rb0, uint32_t n_rb) {
        std::vector<uint32_t> v{o_addr, n_rb * (uint32_t)a.n_otc};
        append_view_args(v, a.out_kind, rb0 * (uint32_t)a.n_otc, to_desc(a.o));
        return v;
    }

    static cached_program_t create(
        const operation_attributes_t& a, const tensor_args_t& t, tensor_return_value_t& out) {
        Program prog = CreateProgram();
        const uint32_t n_rb_total = (uint32_t)a.n_otr;
        const uint32_t ctiles     = (uint32_t)a.n_otc;
        const uint32_t ct2        = ctiles * 2;

        MeshDevice* dev = t.r.device();
        auto grid = dev->compute_with_storage_grid_size();
        auto [num_cores, all_cores, cg1, cg2, upc1, upc2] =
            tt::tt_metal::split_work_to_cores(grid, n_rb_total);

        const uint32_t w_group_pages = (uint32_t)a.rbs_per_group * ctiles;
        const uint32_t w_pages = a.w_bcast ? (w_group_pages > ct2 ? w_group_pages : ct2) : ct2;
        mkcb(prog, all_cores, tt::CBIndex::c_0,  2048u, ct2, tt::DataFormat::Float16_b);  // cb_r
        mkcb(prog, all_cores, tt::CBIndex::c_6,  2048u, ct2, tt::DataFormat::Float16_b);  // cb_k
        mkcb(prog, all_cores, tt::CBIndex::c_7,  2048u, w_pages, tt::DataFormat::Float16_b);  // cb_w (r_k)
        mkcb(prog, all_cores, tt::CBIndex::c_8,  2048u, ct2, tt::DataFormat::Float16_b);  // cb_v
        mkcb(prog, all_cores, tt::CBIndex::c_9,  2048u, ct2, tt::DataFormat::Float16_b);  // cb_cur
        mkcb(prog, all_cores, tt::CBIndex::c_16, 2048u, ct2, tt::DataFormat::Float16_b);  // cb_out
        mkcb(prog, all_cores, tt::CBIndex::c_4,  2048u, 1, tt::DataFormat::Float16_b);    // cb_scaler
        mkcb(prog, all_cores, tt::CBIndex::c_31, 2048u, 1, tt::DataFormat::Float16_b);    // cb_zero
        mkcb(prog, all_cores, tt::CBIndex::c_21, 2048u, 2, tt::DataFormat::Float16_b);    // cb_ss
        mkcb(prog, all_cores, tt::CBIndex::c_27, 2048u, ctiles, tt::DataFormat::Float16_b);  // cb_pw

        const std::string ttnn_inc = std::string(getenv("TT_METAL_HOME")) + "/ttnn";
        std::vector<uint32_t> rct;
        TensorAccessorArgs(*t.r.buffer()).append_to(rct);
        TensorAccessorArgs(*t.k.buffer()).append_to(rct);
        TensorAccessorArgs(*t.w.buffer()).append_to(rct);
        TensorAccessorArgs(*t.v.buffer()).append_to(rct);
        TensorAccessorArgs(*t.cur.buffer()).append_to(rct);
        KernelHandle reader = CreateKernel(prog, TTPRM_ROOT "/kernel/view_rk_reader.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                               .compile_args = rct,
                               .defines = {{"TTPRM_R_VIEW",   view_type_name(a.r_kind)},
                                           {"TTPRM_K_VIEW",   view_type_name(a.k_kind)},
                                           {"TTPRM_W_VIEW",   view_type_name(a.w_kind)},
                                           {"TTPRM_V_VIEW",   view_type_name(a.v_kind)},
                                           {"TTPRM_CUR_VIEW", view_type_name(a.cur_kind)}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        std::vector<uint32_t> wct; TensorAccessorArgs(*out.buffer()).append_to(wct);
        KernelHandle writer = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_writer.cpp", all_cores,
            DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default,
                               .compile_args = wct,
                               .defines = {{"TTPRM_OUT_VIEW", view_type_name(a.out_kind)},
                                           {"TTPRM_OUT_CB", "16"}},
                               .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
        MathFidelity fid = MathFidelity::HiFi4;
        bool fp32d = true;
        if (const char* f = std::getenv("TTPRM_NORM_FIDELITY")) {
            int v = atoi(f);
            fid = v <= 0 ? MathFidelity::LoFi : v == 2 ? MathFidelity::HiFi2
                                              : v == 3 ? MathFidelity::HiFi3 : MathFidelity::HiFi4;
        }
        if (const char* p = std::getenv("TTPRM_NORM_FP32")) fp32d = atoi(p) != 0;
        KernelHandle compute = CreateKernel(prog, TTPRM_ROOT "/kernel/view_rk_compute.cpp", all_cores,
            ComputeConfig{.math_fidelity = fid, .fp32_dest_acc_en = fp32d, .compile_args = {}});

        const uint32_t r_addr   = (uint32_t)t.r.buffer()->address();
        const uint32_t k_addr   = (uint32_t)t.k.buffer()->address();
        const uint32_t w_addr   = (uint32_t)t.w.buffer()->address();
        const uint32_t v_addr   = (uint32_t)t.v.buffer()->address();
        const uint32_t cur_addr = (uint32_t)t.cur.buffer()->address();
        const uint32_t o_addr   = (uint32_t)out.buffer()->address();
        const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/true);
        uint32_t done = 0;
        for (uint32_t i = 0; i < num_cores; i++) {
            const CoreCoord cc = cores[i];
            const uint32_t here = cg1.contains(cc) ? upc1 : upc2;
            // Resident r_k only helps when this core would otherwise reread more
            // row-blocks than the one-token r_k group contains.
            const uint32_t w_resident = (a.w_bcast && here > (uint32_t)a.rbs_per_group) ? 1u : 0u;
            SetRuntimeArgs(prog, reader, cc,
                           reader_args(a, r_addr, k_addr, w_addr, v_addr, cur_addr,
                                       done, here, w_resident));
            SetRuntimeArgs(prog, writer, cc, writer_args(a, o_addr, done, here));
            SetRuntimeArgs(prog, compute, cc, compute_args(a, done, here, w_resident));
            done += here;
        }
        return {std::move(prog), shared_variables_t{reader, writer, all_cores}};
    }

    static void override_runtime_arguments(
        cached_program_t& cp, const operation_attributes_t&, const tensor_args_t& t,
        tensor_return_value_t& out) {
        auto& prog = cp.program;
        const uint32_t r_addr   = (uint32_t)t.r.buffer()->address();
        const uint32_t k_addr   = (uint32_t)t.k.buffer()->address();
        const uint32_t w_addr   = (uint32_t)t.w.buffer()->address();
        const uint32_t v_addr   = (uint32_t)t.v.buffer()->address();
        const uint32_t cur_addr = (uint32_t)t.cur.buffer()->address();
        const uint32_t o_addr   = (uint32_t)out.buffer()->address();
        for (const auto& range : cp.shared_variables.all_cores.ranges())
            for (const auto& core : range) {
                auto& ra = GetRuntimeArgs(prog, cp.shared_variables.reader, core);
                ra[0] = r_addr; ra[1] = k_addr; ra[2] = w_addr; ra[3] = v_addr; ra[4] = cur_addr;
                GetRuntimeArgs(prog, cp.shared_variables.writer, core)[0] = o_addr;
            }
    }
};
}  // namespace program

struct ViewRkDeviceOperation {
    using operation_attributes_t = vrk::operation_attributes_t;
    using tensor_args_t          = vrk::tensor_args_t;
    using spec_return_value_t    = vrk::spec_return_value_t;
    using tensor_return_value_t  = vrk::tensor_return_value_t;
    using program_factory_t      = std::variant<program::ViewRkFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&) {
        return program::ViewRkFactory{};
    }
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&) {}
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&) {}
    static spec_return_value_t compute_output_specs(const operation_attributes_t& a, const tensor_args_t&) {
        return tile_spec(a.out_rows, a.out_cols, a.memory_config);
    }
    static tensor_return_value_t create_output_tensors(const operation_attributes_t& a, const tensor_args_t& t) {
        return resolve_output(t.opt_out, compute_output_specs(a, t), t.r.device());
    }
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& r, const ttnn::Tensor& k, const ttnn::Tensor& w,
        const ttnn::Tensor& v, const ttnn::Tensor& cur,
        const AffineDesc& dr, const AffineDesc& dk, const AffineDesc& dw,
        const AffineDesc& dv, const AffineDesc& dcur, const AffineDesc& dout,
        int32_t out_rows, int32_t out_cols, uint32_t rscale_bits,
        uint32_t w_bcast, int32_t rbs_per_group,
        std::optional<ttnn::Tensor> opt_out = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt) {
        operation_attributes_t at;
        at.r_kind = kind_of(dr); at.k_kind = kind_of(dk); at.w_kind = kind_of(dw);
        at.v_kind = kind_of(dv); at.cur_kind = kind_of(dcur); at.out_kind = kind_of(dout);
        at.out_rows = out_rows; at.out_cols = out_cols;
        at.n_otr = (int32_t)dout.n_otr; at.n_otc = (int32_t)dout.n_otc;
        at.rscale_bits = rscale_bits;
        at.w_bcast = w_bcast; at.rbs_per_group = rbs_per_group;
        at.memory_config = output_memory_config(opt_out, memory_config);
        at.r = from_desc(dr); at.k = from_desc(dk); at.w = from_desc(dw);
        at.v = from_desc(dv); at.cur = from_desc(dcur); at.o = from_desc(dout);
        return { at, tensor_args_t{r, k, w, v, cur, std::move(opt_out)} };
    }
};

}  // namespace vrk
}  // namespace

// Public surface.

// View: a tiled ttnn::Tensor bound to a layout view-chain. The constructors are the only place
// the ttnn::Tensor shape is read; everything downstream speaks LayoutView + the logical shape.
View::View(const ttnn::Tensor& t) : lv_(0, 0) {
    const auto& ps = t.padded_shape();
    int64_t pcols = (int64_t)ps[ps.rank() - 1], prows = 1;
    for (size_t i = 0; i + 1 < ps.rank(); i++) prows *= (int64_t)ps[i];
    const auto& ls = t.logical_shape();             // logical (unpadded) shape -> the operand shape
    cols_ = (int64_t)ls[ls.rank() - 1]; rows_ = 1;
    for (size_t i = 0; i + 1 < ls.rank(); i++) rows_ *= (int64_t)ls[i];
    shape_.reserve(ls.rank());
    padded_shape_.reserve(ps.rank());
    for (size_t i = 0; i < ls.rank(); i++) shape_.push_back((int64_t)ls[i]);
    for (size_t i = 0; i < ps.rank(); i++) padded_shape_.push_back((int64_t)ps[i]);
    // Dense-packed tensors have no padded middle axes, so collapsed logical rows
    // have a uniform physical stride. Inner-padded tensors keep the padded
    // anchor because logical rows do not have a uniform stride.
    bool dense_packed = true;
    for (size_t a = 1; a + 1 < ls.rank(); a++)
        if ((int64_t)ls[a] != (int64_t)ps[a]) dense_packed = false;
    lv_ = dense_packed ? LayoutView::source(rows_, cols_, prows, pcols)  // logical iterate
                       : LayoutView(prows, pcols);                       // legacy padded anchor
    t_ = &t;
}

View View::squeeze_dim(int dim) const {
    View v = *this;
    const int rank = (int)v.shape_.size();
    if (dim < 0) dim += rank;
    TT_FATAL(rank >= 3, "squeeze_dim requires rank >= 3 so the last two tile presentation dims are preserved");
    TT_FATAL(dim >= 0 && dim < rank, "squeeze_dim dim {} is out of range for rank {}", dim, rank);
    TT_FATAL(dim < rank - 2, "squeeze_dim dim {} targets a tile presentation dimension in rank {}", dim, rank);
    TT_FATAL(v.shape_[dim] == 1, "squeeze_dim dim {} has extent {}, expected 1", dim, v.shape_[dim]);
    v.shape_.erase(v.shape_.begin() + dim);
    v.padded_shape_.erase(v.padded_shape_.begin() + dim);
    v.rows_ = 1;
    for (size_t i = 0; i + 1 < v.shape_.size(); i++) v.rows_ *= v.shape_[i];
    v.cols_ = v.shape_.back();
    v.lv_ = LayoutView::source_shape(v.shape_, v.padded_shape_);
    return v;
}

View View::of_shape(int64_t rows, int64_t cols) {
    auto pad = [](int64_t x) { return ((x + 31) / 32) * 32; };
    View v;                                          // no tensor (output operand)
    v.lv_ = LayoutView::source(rows, cols, pad(rows), pad(cols));  // logical iterate, padded address
    v.rows_ = rows; v.cols_ = cols;
    v.shape_ = {rows, cols};
    v.padded_shape_ = {pad(rows), pad(cols)};
    return v;
}

// Lower `in` (gathered) into `out` (scattered). `out` defaults to a dense result over in's
// presented [Ro,Co]. Pure host: no device work.
Result<Lowered> prelower(const View& in, std::optional<View> out) {
    if (!in.tensor()) return Result<Lowered>::err("input view has no bound tensor");
    std::string vmsg = validate_device_tile_bf16(*in.tensor());
    if (!vmsg.empty()) return Result<Lowered>::err(vmsg);

    AffineDesc in_desc = lower(in.layout());
    if (!in_desc.ok) return Result<Lowered>::err("input view lowers to REJECT: " + in_desc.why);

    int64_t orows, ocols;
    LayoutView out_lv(0, 0);
    if (out) {
        out_lv = out->layout(); orows = out->rows(); ocols = out->cols();
    } else {
        const auto p = in.layout().plan();           // dense result of the gathered shape
        orows = p.Ro; ocols = p.Co;
        out_lv = View::of_shape(orows, ocols).layout();
    }
    AffineDesc out_desc = lower(out_lv);
    if (!out_desc.ok) return Result<Lowered>::err("output view lowers to REJECT: " + out_desc.why);

    // Reader and writer must present the same live extent and padded tile grid.
    // Compare view shapes, not backing tensor shapes: scatter views may reshape
    // the destination while preserving the presented grid.
    if (in_desc.Ro != out_desc.Ro || in_desc.Co != out_desc.Co ||
        in_desc.Ro_pad != out_desc.Ro_pad || in_desc.Co_pad != out_desc.Co_pad)
        return Result<Lowered>::err("output view shape does not match the gathered input shape");

    Lowered L;
    L.in = in.tensor();
    // A BOUND out view (view_of(dst)...) scatters into dst in place; an of_shape/absent out
    // leaves L.out null so the op allocates.
    if (out && out->tensor()) {
        std::string dmsg = validate_device_tile_bf16(*out->tensor());
        if (!dmsg.empty()) return Result<Lowered>::err("out: " + dmsg);
        L.out = out->tensor();
    }
    L.in_desc = in_desc; L.out_desc = out_desc;
    L.in_kind = kind_of(in_desc); L.out_kind = kind_of(out_desc);
    L.out_rows = (int32_t)orows; L.out_cols = (int32_t)ocols;
    return Result<Lowered>::ok(L);
}

ttnn::Tensor realize(const Lowered& low, const std::optional<MemoryConfig>& memory_config) {
    std::optional<ttnn::Tensor> opt_out;
    if (low.out) opt_out = *low.out;                  // bound destination -> scatter in place
    auto [attrs, args] = vr::ViewRealizeDeviceOperation::invoke(
        *low.in, low.in_desc, low.out_desc, low.out_rows, low.out_cols, std::move(opt_out), memory_config);
    return ttnn::device_operation::detail::launch<vr::ViewRealizeDeviceOperation>(attrs, args);
}

Result<ttnn::Tensor> realize(
    const View& in,
    std::optional<View> out,
    const std::optional<MemoryConfig>& memory_config) {
    auto low = prelower(in, out);
    if (!low) return Result<ttnn::Tensor>::err(low.error());
    return Result<ttnn::Tensor>::ok(realize(low.value(), memory_config));
}

// Fused binary elementwise. Each operand is lowered against the output grid.
// Proper tile-divisor operands are resident group broadcasts.
static Result<ttnn::Tensor> eltwise_bin(uint32_t op, const View& a, const View& b,
                                        std::optional<View> out,
                                        const std::optional<MemoryConfig>& memory_config) {
    if (!a.tensor()) return Result<ttnn::Tensor>::err("a: view has no bound tensor");
    if (!b.tensor()) return Result<ttnn::Tensor>::err("b: view has no bound tensor");
    std::string ma = validate_device_tile_bf16(*a.tensor());
    if (!ma.empty()) return Result<ttnn::Tensor>::err("a: " + ma);
    std::string mb = validate_device_tile_bf16(*b.tensor());
    if (!mb.empty()) return Result<ttnn::Tensor>::err("b: " + mb);

    AffineDesc da = lower(a.layout());
    if (!da.ok) return Result<ttnn::Tensor>::err("a view REJECT: " + da.why);
    AffineDesc db = lower(b.layout());
    if (!db.ok) return Result<ttnn::Tensor>::err("b view REJECT: " + db.why);

    // Output: explicit operand, or a dense result over the larger (full-grid) input's shape.
    int64_t out_rows, out_cols;
    LayoutView out_lv(0, 0);
    if (out) {
        out_lv = out->layout(); out_rows = out->rows(); out_cols = out->cols();
    } else {
        const auto pa = a.layout().plan();
        const auto pb = b.layout().plan();
        const auto& full = pa.out_tiles() >= pb.out_tiles() ? pa : pb;
        out_rows = full.Ro; out_cols = full.Co;
        out_lv = View::of_shape(out_rows, out_cols).layout();
    }
    AffineDesc dout = lower(out_lv);
    if (!dout.ok) return Result<ttnn::Tensor>::err("out view REJECT: " + dout.why);

    // Classify each input vs the output grid: equal -> normal; proper divisor -> group-broadcast.
    const int64_t wt = dout.out_tiles();
    int64_t tpg = wt;                 // group size = the broadcast operands' tile count (wt if none)
    uint32_t a_bcast = 0, b_bcast = 0;
    std::string e1 = classify_operand(da, dout, out_rows, wt, tpg, a_bcast);
    if (!e1.empty()) return Result<ttnn::Tensor>::err("a: " + e1);
    std::string e2 = classify_operand(db, dout, out_rows, wt, tpg, b_bcast);
    if (!e2.empty()) return Result<ttnn::Tensor>::err("b: " + e2);

    std::string oe; auto od = bound_dst(out, oe);
    if (!oe.empty()) return Result<ttnn::Tensor>::err(oe);

    auto [attrs, args] = va::ViewAddDeviceOperation::invoke(
        op, *a.tensor(), *b.tensor(), da, db, dout,
        (int32_t)out_rows, (int32_t)out_cols, (int32_t)tpg, a_bcast, b_bcast,
        std::move(od), memory_config);
    return Result<ttnn::Tensor>::ok(
        ttnn::device_operation::detail::launch<va::ViewAddDeviceOperation>(attrs, args));
}

// Fused ternary elementwise. Each operand is lowered against the output grid,
// and all resident broadcast operands share one group size.
static Result<ttnn::Tensor> eltwise_ternary(uint32_t op, const View& a, const View& b,
                                            const View& c, std::optional<View> out,
                                            bool canonical,
                                            const std::optional<MemoryConfig>& memory_config) {
    if (!a.tensor()) return Result<ttnn::Tensor>::err("a: view has no bound tensor");
    if (!b.tensor()) return Result<ttnn::Tensor>::err("b: view has no bound tensor");
    if (!c.tensor()) return Result<ttnn::Tensor>::err("c: view has no bound tensor");
    std::string ma = validate_device_tile_bf16(*a.tensor());
    if (!ma.empty()) return Result<ttnn::Tensor>::err("a: " + ma);
    std::string mb = validate_device_tile_bf16(*b.tensor());
    if (!mb.empty()) return Result<ttnn::Tensor>::err("b: " + mb);
    std::string mc = validate_device_tile_bf16(*c.tensor());
    if (!mc.empty()) return Result<ttnn::Tensor>::err("c: " + mc);

    AffineDesc da = lower(a.layout());
    if (!da.ok) return Result<ttnn::Tensor>::err("a view REJECT: " + da.why);
    AffineDesc db = lower(b.layout());
    if (!db.ok) return Result<ttnn::Tensor>::err("b view REJECT: " + db.why);
    AffineDesc dc = lower(c.layout());
    if (!dc.ok) return Result<ttnn::Tensor>::err("c view REJECT: " + dc.why);

    // Output: explicit operand, or a dense result over the largest (full-grid) input's shape.
    int64_t out_rows, out_cols;
    LayoutView out_lv(0, 0);
    if (out) {
        out_lv = out->layout(); out_rows = out->rows(); out_cols = out->cols();
    } else {
        const auto pa = a.layout().plan();
        const auto pb = b.layout().plan();
        const auto pc = c.layout().plan();
        const auto* full = &pa;
        if (pb.out_tiles() > full->out_tiles()) full = &pb;
        if (pc.out_tiles() > full->out_tiles()) full = &pc;
        out_rows = full->Ro; out_cols = full->Co;
        out_lv = View::of_shape(out_rows, out_cols).layout();
    }
    AffineDesc dout = lower(out_lv);
    if (!dout.ok) return Result<ttnn::Tensor>::err("out view REJECT: " + dout.why);

    const int64_t wt = dout.out_tiles();
    int64_t tpg = wt;
    uint32_t a_bcast = 0, b_bcast = 0, c_bcast = 0;

    // Prefill token-shift uses a two-level periodic row map. For lerp,
    // a/b are token values [nt,E], c is per-group weight [G,E], and
    // group-major output rows satisfy out_rows == nt * G.
    const int64_t nt = a.rows(), G = c.rows();
    // Detect token-shift before mutating descriptors so canonical output can
    // use the original shape information.
    const bool is_tokenshift = op == 1 /*lerp*/ && G >= 1 && nt >= 1 && b.rows() == nt &&
                               nt * G == out_rows && is_plain_source(da) &&
                               is_plain_source(db) && is_plain_source(dc);
    // Weights may be packed [G,E] or row-strided [G,1,1,E]. Row-strided
    // weights need a per-group source-row multiplier in the grouped map.
    const bool w_strided = is_tokenshift && G > 1 && dc.n_otr == G;
    const int64_t wstride = w_strided ? (dc.Ro_pad / G) : 1;        // 32 (one tile-block/group) or 1
    if (is_tokenshift && (nt > 1 || w_strided)) {
        da = make_grouped_desc(da, dout, /*P=*/nt, /*A=*/0, /*B=*/1);         // cur:    oR mod nt
        db = make_grouped_desc(db, dout, /*P=*/nt, /*A=*/0, /*B=*/1);         // x_prev: oR mod nt
        dc = make_grouped_desc(dc, dout, /*P=*/nt, /*A=*/wstride, /*B=*/0);   // weight: (oR div nt)*stride
        // All three stream the full output grid; no resident group.
    } else {
        // Classify each input vs the output grid: equal -> normal; proper divisor -> group-broadcast.
        std::string e1 = classify_operand(da, dout, out_rows, wt, tpg, a_bcast);
        if (!e1.empty()) return Result<ttnn::Tensor>::err("a: " + e1);
        std::string e2 = classify_operand(db, dout, out_rows, wt, tpg, b_bcast);
        if (!e2.empty()) return Result<ttnn::Tensor>::err("b: " + e2);
        std::string e3 = classify_operand(dc, dout, out_rows, wt, tpg, c_bcast);
        if (!e3.empty()) return Result<ttnn::Tensor>::err("c: " + e3);
    }

    // Canonical output scatters packed [G*nt,E] rows into TTNN's batch-padded
    // [G,1,nt,E] layout. When nt is tile-aligned, the packed layout already
    // matches the canonical physical tiling.
    AffineDesc dosc = dout;
    int64_t canon_G = 0, canon_nt = 0;
    if (canonical) {
        if (!is_tokenshift)
            return Result<ttnn::Tensor>::err(
                "canonical output is only valid for the RWKV token-shift lerp shape "
                "(a=b=[nt,E], w=[G,E], out rows = G*nt group-major)");
        // The returned tensor's LOGICAL shape is 4D [G,1,nt,E]; ttnn's TILE layout pads nt->32
        // internally, so the buffer matches the writer's physical scatter exactly (no caller pad math).
        canon_G = G; canon_nt = nt;
        const int64_t Pad = ((nt + 31) / 32) * 32;       // ttnn pads the nt (token) dim to a tile
        if (nt % 32 != 0) {                              // real densify: decode nt=1, or sub-32 prefill
            out_rows = G * Pad;                          // physical buffer = G batch-blocks of Pad rows
            dosc.grp_P = nt; dosc.grp_A = Pad; dosc.grp_B = 1;   // packed oR -> Pad*(oR/nt) + (oR%nt)
            // dosc.Cs/nct/base unchanged: output width is still E (only the row count grew).
        }
        // nt%32==0: packed already IS the canonical tiling -> dense scatter, just the 4D shape label.
    }

    std::string oe; auto od = bound_dst(out, oe);
    if (!oe.empty()) return Result<ttnn::Tensor>::err(oe);

    auto [attrs, args] = vt::ViewTernaryDeviceOperation::invoke(
        op, *a.tensor(), *b.tensor(), *c.tensor(), da, db, dc, dout, dosc,
        (int32_t)out_rows, (int32_t)out_cols, (int32_t)tpg, a_bcast, b_bcast, c_bcast,
        (int32_t)canon_G, (int32_t)canon_nt, std::move(od), memory_config);
    return Result<ttnn::Tensor>::ok(
        ttnn::device_operation::detail::launch<vt::ViewTernaryDeviceOperation>(attrs, args));
}

Result<ttnn::Tensor> fma(
    const View& a,
    const View& b,
    const View& c,
    std::optional<View> out,
    const std::optional<MemoryConfig>& memory_config) {
    return eltwise_ternary(0, a, b, c, out, false, memory_config);
}
Result<ttnn::Tensor> lerp(const View& a, const View& b, const View& w, std::optional<View> out,
                          bool canonical_out, const std::optional<MemoryConfig>& memory_config) {
    return eltwise_ternary(1, a, b, w, out, canonical_out, memory_config);
}

// Fused per-head norm. Reduction is per logical row over Co lanes. L2 uses
// raw-sum scale 1.0; RMS/LayerNorm use mean scale 1/Co.
static inline uint32_t f32_bits(float f) { uint32_t u; __builtin_memcpy(&u, &f, 4); return u; }

static Result<ttnn::Tensor> norm_impl(vn::NormMode mode, const View& x, const ttnn::Tensor* gamma,
                                      const ttnn::Tensor* beta, float eps,
                                      std::optional<View> out,
                                      const View* gamma_view = nullptr,
                                      const View* beta_view = nullptr,
                                      const std::optional<MemoryConfig>& memory_config = std::nullopt) {
    if (!x.tensor()) return Result<ttnn::Tensor>::err("x: view has no bound tensor");
    std::string mx = validate_device_tile_bf16(*x.tensor());
    if (!mx.empty()) return Result<ttnn::Tensor>::err("x: " + mx);

    AffineDesc dx = lower(x.layout());
    if (!dx.ok) return Result<ttnn::Tensor>::err("x view REJECT: " + dx.why);

    // Reduction width is the presented head width, not the source tensor width.
    // RMS/LayerNorm divide sumsq by N (mean); l2 keeps the raw sum (scaler 1.0).
    const int64_t N = dx.Co;
    if (N <= 0) return Result<ttnn::Tensor>::err("norm: reduction width (Co) is non-positive");
    const float rscale = (mode == vn::NormMode::L2) ? 1.0f : 1.0f / (float)N;

    int64_t out_rows, out_cols;
    LayoutView out_lv(0, 0);
    if (out) {
        out_lv = out->layout(); out_rows = out->rows(); out_cols = out->cols();
    } else {
        const auto p = x.layout().plan();
        out_rows = p.Ro; out_cols = p.Co;
        out_lv = View::of_shape(out_rows, out_cols).layout();
    }
    AffineDesc dout = lower(out_lv);
    if (!dout.ok) return Result<ttnn::Tensor>::err("out view REJECT: " + dout.why);
    if (dout.n_otr != dx.n_otr || dout.n_otc != dx.n_otc || dout.Ro != dx.Ro || dout.Co != dx.Co)
        return Result<ttnn::Tensor>::err("norm output grid must match the input (shape-preserving)");

    vn::NormAffine affine = vn::NormAffine::None;
    if (mode == vn::NormMode::L2) {
        if (gamma || beta) return Result<ttnn::Tensor>::err("l2_norm does not take affine tensors");
    } else if (mode == vn::NormMode::Rms) {
        if (beta) return Result<ttnn::Tensor>::err("rms_norm takes an optional weight tensor, not beta");
        affine = gamma ? vn::NormAffine::Scale : vn::NormAffine::None;
    } else {
        if ((gamma == nullptr) != (beta == nullptr))
            return Result<ttnn::Tensor>::err("layer_norm needs both gamma and beta, or neither");
        affine = gamma ? vn::NormAffine::ScaleBias : vn::NormAffine::None;
    }
    if (gamma) {
        std::string mg = validate_device_tile_bf16(*gamma);
        if (!mg.empty()) return Result<ttnn::Tensor>::err("gamma: " + mg);
        if (gamma_view) {
            AffineDesc dg = lower(gamma_view->layout());
            if (!dg.ok) return Result<ttnn::Tensor>::err("gamma view REJECT: " + dg.why);
            if (gamma_view->tensor() != gamma || !is_plain_source(dg) || dg.Ro != 1 || dg.Co != N)
                return Result<ttnn::Tensor>::err("gamma view must present a plain [1, Co] vector");
        } else {
            const auto gs = gamma->logical_shape();
            if (gs.rank() != 2 || gs[0] != 1u || gs[1] != (uint32_t)N)
                return Result<ttnn::Tensor>::err("gamma/weight must have logical shape [1, Co]");
        }
    }
    if (beta) {
        std::string mb = validate_device_tile_bf16(*beta);
        if (!mb.empty()) return Result<ttnn::Tensor>::err("beta: " + mb);
        if (beta_view) {
            AffineDesc db = lower(beta_view->layout());
            if (!db.ok) return Result<ttnn::Tensor>::err("beta view REJECT: " + db.why);
            if (beta_view->tensor() != beta || !is_plain_source(db) || db.Ro != 1 || db.Co != N)
                return Result<ttnn::Tensor>::err("beta view must present a plain [1, Co] vector");
        } else {
            const auto bs = beta->logical_shape();
            if (bs.rank() != 2 || bs[0] != 1u || bs[1] != (uint32_t)N)
                return Result<ttnn::Tensor>::err("beta must have logical shape [1, Co]");
        }
    }
    // gamma/beta accessors must always exist (compiled in); when there is no affine, alias x so the
    // accessor is valid but the reader never gathers it.
    const ttnn::Tensor& g = gamma ? *gamma : *x.tensor();
    const ttnn::Tensor& b = beta  ? *beta  : *x.tensor();

    // Narrow rows use the resident double-buffered kernel. Wider rows use the
    // lean-resident kernel with streamed gamma/beta and smaller scratch.
    const uint32_t wide = (dx.n_otc > 32) ? 1u : 0u;

    // Folded-global path: a single wide row [1,N] is one norm over N. Fold
    // live elements to dense [N/32,32] and reduce with REDUCE_SCALAR. The
    // writer may target either a fresh dense output or a bound output view, but
    // it must present the same folded tile grid as the compute input.
    uint32_t global = 0;
    AffineDesc dxf = dx, dof = dout;
    if (wide && dx.n_otr == 1 && (N % 1024) == 0) {
        AffineDesc xf = lower(x.reshape({N / 32, 32}).layout());
        AffineDesc of = out ? lower(out->reshape({N / 32, 32}).layout())
                            : lower(View::of_shape(out_rows, out_cols).reshape({N / 32, 32}).layout());
        if (xf.ok && of.ok &&
            xf.Ro == of.Ro && xf.Co == of.Co &&
            xf.n_otr == of.n_otr && xf.n_otc == of.n_otc &&
            xf.n_otc == 1 && xf.n_otr == N / 1024) {
            global = 1; dxf = xf; dof = of;
        }
    }

    // The folded global path uses REDUCE_SCALAR, whose LLK applies the scaler tile TWICE (pool faces,
    // then a final pool over the transposed partial column -- see llk_math_reduce.h REDUCE_SCALAR).
    // So the canonical scaler for a scalar mean is 1/sqrt(N) (per reduce.h + TTNN's reduce helper),
    // not 1/N: (1/sqrt(N))^2 = 1/N for RMS/LN reductions. The per-row REDUCE_ROW
    // path applies the scaler once, so it keeps 1/N. L2 keeps 1.0 either way (1^2 == 1).
    const float reduce_rscale = (mode != vn::NormMode::L2 && global) ? 1.0f / std::sqrt((float)N) : rscale;

    std::string oe; auto od = bound_dst(out, oe);
    if (!oe.empty()) return Result<ttnn::Tensor>::err(oe);

    auto [attrs, args] = vn::ViewNormDeviceOperation::invoke(
        mode, *x.tensor(), g, b, dxf, dof, (int32_t)out_rows, (int32_t)out_cols,
        f32_bits(reduce_rscale), f32_bits(eps), affine, wide, global, std::move(od), memory_config);
    return Result<ttnn::Tensor>::ok(
        ttnn::device_operation::detail::launch<vn::ViewNormDeviceOperation>(attrs, args));
}

Result<ttnn::Tensor> l2_norm(
    const View& x,
    float eps,
    std::optional<View> out,
    const std::optional<MemoryConfig>& memory_config) {
    return norm_impl(vn::NormMode::L2, x, nullptr, nullptr, eps, out, nullptr, nullptr, memory_config);
}
Result<ttnn::Tensor> rms_norm(
    const View& x,
    float eps,
    std::optional<View> out,
    const std::optional<MemoryConfig>& memory_config) {
    return norm_impl(vn::NormMode::Rms, x, nullptr, nullptr, eps, out, nullptr, nullptr, memory_config);
}
Result<ttnn::Tensor> rms_norm(const View& x, const ttnn::Tensor& weight, float eps,
                              std::optional<View> out,
                              const std::optional<MemoryConfig>& memory_config) {
    return norm_impl(vn::NormMode::Rms, x, &weight, nullptr, eps, out, nullptr, nullptr, memory_config);
}
Result<ttnn::Tensor> layer_norm(const View& x, const ttnn::Tensor* gamma, const ttnn::Tensor* beta,
                                float eps, std::optional<View> out,
                                const std::optional<MemoryConfig>& memory_config) {
    return norm_impl(vn::NormMode::Layer, x, gamma, beta, eps, out, nullptr, nullptr, memory_config);
}
Result<ttnn::Tensor> layer_norm(const View& x, const View& gamma, const View& beta,
                                float eps, std::optional<View> out,
                                const std::optional<MemoryConfig>& memory_config) {
    if (!gamma.tensor()) return Result<ttnn::Tensor>::err("gamma: view has no bound tensor");
    if (!beta.tensor()) return Result<ttnn::Tensor>::err("beta: view has no bound tensor");
    return norm_impl(vn::NormMode::Layer, x, gamma.tensor(), beta.tensor(), eps, out,
                     &gamma, &beta, memory_config);
}

    // FUSED mul_l2_norm: y = l2_norm(a ⊙ b) per head (the RWKV kk-normalize). a and b are each lowered
    // against their own tile grid (flat[1,N]->head AFFINE or dense). They either present the same head
    // grid, or one same-width row grid evenly repeats over the other (RWKV prefill k * time_mix_k_k).
    // The reduction is per logical row over the Co lanes (raw-sum scaler 1.0, like l2). `out` defaults
    // to a dense result over the full broadcast grid.
Result<ttnn::Tensor> mul_l2_norm(
    const View& a,
    const View& b,
    float eps,
    std::optional<View> out,
    const std::optional<MemoryConfig>& memory_config) {
    if (!a.tensor()) return Result<ttnn::Tensor>::err("a: view has no bound tensor");
    if (!b.tensor()) return Result<ttnn::Tensor>::err("b: view has no bound tensor");
    std::string ma = validate_device_tile_bf16(*a.tensor());
    if (!ma.empty()) return Result<ttnn::Tensor>::err("a: " + ma);
    std::string mb = validate_device_tile_bf16(*b.tensor());
    if (!mb.empty()) return Result<ttnn::Tensor>::err("b: " + mb);

    AffineDesc da = lower(a.layout());
    if (!da.ok) return Result<ttnn::Tensor>::err("a view REJECT: " + da.why);
    AffineDesc db = lower(b.layout());
    if (!db.ok) return Result<ttnn::Tensor>::err("b view REJECT: " + db.why);
    AffineDesc dfull = da;
    const View* shape_view = &a;
    enum class Bcast { None, AToB, BToA };
    Bcast bcast = Bcast::None;
    auto same_grid = [](const AffineDesc& x, const AffineDesc& y) {
        return x.n_otr == y.n_otr && x.n_otc == y.n_otc && x.Ro == y.Ro && x.Co == y.Co;
    };
    auto can_row_repeat = [](const AffineDesc& small, const AffineDesc& full) {
        return small.Co == full.Co && small.n_otc == full.n_otc &&
               small.Ro > 0 && small.Ro < full.Ro && full.Ro % small.Ro == 0;
    };
    if (same_grid(da, db)) {
        // Fast path: no broadcast.
    } else if (can_row_repeat(db, da)) {
        bcast = Bcast::BToA; dfull = da; shape_view = &a;
    } else if (can_row_repeat(da, db)) {
        bcast = Bcast::AToB; dfull = db; shape_view = &b;
    } else {
        return Result<ttnn::Tensor>::err(
            "mul_l2_norm: a and b must present the same head grid, or one same-width row grid "
            "must evenly repeat over the other");
    }

    const int64_t N = dfull.Co;
    if (N <= 0) return Result<ttnn::Tensor>::err("mul_l2_norm: reduction width (Co) is non-positive");

    int64_t out_rows, out_cols;
    LayoutView out_lv(0, 0);
    if (out) {
        out_lv = out->layout(); out_rows = out->rows(); out_cols = out->cols();
    } else {
        const auto p = shape_view->layout().plan();
        out_rows = p.Ro; out_cols = p.Co;
        out_lv = View::of_shape(out_rows, out_cols).layout();
    }
    AffineDesc dout = lower(out_lv);
    if (!dout.ok) return Result<ttnn::Tensor>::err("out view REJECT: " + dout.why);
    if (dout.n_otr != dfull.n_otr || dout.n_otc != dfull.n_otc || dout.Ro != dfull.Ro || dout.Co != dfull.Co)
        return Result<ttnn::Tensor>::err("mul_l2_norm output grid must match the inputs (shape-preserving)");

    if (bcast == Bcast::BToA) {
        db = make_grouped_desc(db, dout, /*P=*/db.Ro, /*A=*/0, /*B=*/1);
    } else if (bcast == Bcast::AToB) {
        da = make_grouped_desc(da, dout, /*P=*/da.Ro, /*A=*/0, /*B=*/1);
    }

    std::string oe; auto od = bound_dst(out, oe);
    if (!oe.empty()) return Result<ttnn::Tensor>::err(oe);

    auto [attrs, args] = vml::ViewMulL2DeviceOperation::invoke(
        *a.tensor(), *b.tensor(), da, db, dout, (int32_t)out_rows, (int32_t)out_cols,
        f32_bits(1.0f), f32_bits(eps), std::move(od), memory_config);
    return Result<ttnn::Tensor>::ok(
        ttnn::device_operation::detail::launch<vml::ViewMulL2DeviceOperation>(attrs, args));
}

// Fused RWKV7 rk update. All full-grid operands must present the same head
// grid; r_k may match that grid or broadcast over token groups.
Result<ttnn::Tensor> rk(const View& r, const View& k, const View& r_k, const View& v,
                        const View& cur, std::optional<View> out,
                        const std::optional<MemoryConfig>& memory_config) {
    struct In { const char* name; const View& view; };
    // r/k/v/cur carry the full output grid. r_k may match it or present one
    // token-group to broadcast over the token axis.
    const In ins[5] = {{"r", r}, {"k", k}, {"r_k", r_k}, {"v", v}, {"cur", cur}};
    AffineDesc d[5];
    for (int i = 0; i < 5; i++) {
        if (!ins[i].view.tensor())
            return Result<ttnn::Tensor>::err(std::string(ins[i].name) + ": view has no bound tensor");
        std::string m = validate_device_tile_bf16(*ins[i].view.tensor());
        if (!m.empty()) return Result<ttnn::Tensor>::err(std::string(ins[i].name) + ": " + m);
        d[i] = lower(ins[i].view.layout());
        if (!d[i].ok)
            return Result<ttnn::Tensor>::err(std::string(ins[i].name) + " view REJECT: " + d[i].why);
        if (i == 2) continue;                              // r_k classified separately below
        if (d[i].n_otr != d[0].n_otr || d[i].n_otc != d[0].n_otc || d[i].Ro != d[0].Ro || d[i].Co != d[0].Co)
            return Result<ttnn::Tensor>::err("rk: r/k/v/cur must present the same head grid");
    }
    const AffineDesc& dr = d[0]; const AffineDesc& dk = d[1]; AffineDesc& dw = d[2];
    const AffineDesc& dv = d[3]; const AffineDesc& dcur = d[4];

    // Classify r_k against the full grid (= dr). The token axis is the broadcast axis: r_k repeats
    // across G token-groups, each rbs_per_group = dw.n_otr whole row-blocks. hc (= dw.Ro) is a
    // multiple of 32 in RWKV7, so a group is whole row-blocks (no sub-tile hazard) and the output is
    // token-major, so token g <-> contiguous row-blocks [g*rbs_per_group, ...) with r_k's tile order
    // matching the group's intra-group position 1:1.
    uint32_t w_bcast = 0;
    int32_t  rbs_per_group = dw.n_otr;
    if (dw.n_otr == dr.n_otr && dw.n_otc == dr.n_otc && dw.Ro == dr.Ro && dw.Co == dr.Co) {
        w_bcast = 0;                                       // r_k IS the full grid (decode)
    } else if (dw.n_otc == dr.n_otc && dw.Co == dr.Co && dw.n_otr > 0 &&
               dr.n_otr % dw.n_otr == 0 && dw.Ro % TILE == 0) {
        w_bcast = 1; rbs_per_group = dw.n_otr;             // token-broadcast (prefill)
    } else {
        return Result<ttnn::Tensor>::err(
            "r_k must present the full head grid (decode) or one token's [hc,hs] with hc a multiple "
            "of 32 broadcast over the token axis -- fall back to ttnn");
    }

    int64_t out_rows, out_cols;
    LayoutView out_lv(0, 0);
    if (out) {
        out_lv = out->layout(); out_rows = out->rows(); out_cols = out->cols();
    } else {
        const auto p = cur.layout().plan();
        out_rows = p.Ro; out_cols = p.Co;
        out_lv = View::of_shape(out_rows, out_cols).layout();
    }
    AffineDesc dout = lower(out_lv);
    if (!dout.ok) return Result<ttnn::Tensor>::err("out view REJECT: " + dout.why);
    if (dout.n_otr != dcur.n_otr || dout.n_otc != dcur.n_otc || dout.Ro != dcur.Ro || dout.Co != dcur.Co)
        return Result<ttnn::Tensor>::err("rk output grid must match the inputs (shape-preserving)");

    std::string oe; auto od = bound_dst(out, oe);
    if (!oe.empty()) return Result<ttnn::Tensor>::err(oe);

    auto [attrs, args] = vrk::ViewRkDeviceOperation::invoke(
        *r.tensor(), *k.tensor(), *r_k.tensor(), *v.tensor(), *cur.tensor(),
        dr, dk, dw, dv, dcur, dout, (int32_t)out_rows, (int32_t)out_cols, f32_bits(1.0f),
        w_bcast, rbs_per_group, std::move(od), memory_config);
    return Result<ttnn::Tensor>::ok(
        ttnn::device_operation::detail::launch<vrk::ViewRkDeviceOperation>(attrs, args));
}

Result<ttnn::Tensor> k_update(const View& k, const View& a, const View& k_a,
                              std::optional<View> out,
                              const std::optional<MemoryConfig>& memory_config) {
    struct In { const char* name; const View& view; };
    const In ins[3] = {{"k", k}, {"a", a}, {"k_a", k_a}};
    AffineDesc d[3];
    for (int i = 0; i < 3; i++) {
        if (!ins[i].view.tensor())
            return Result<ttnn::Tensor>::err(std::string(ins[i].name) + ": view has no bound tensor");
        std::string m = validate_device_tile_bf16(*ins[i].view.tensor());
        if (!m.empty()) return Result<ttnn::Tensor>::err(std::string(ins[i].name) + ": " + m);
        d[i] = lower(ins[i].view.layout());
        if (!d[i].ok)
            return Result<ttnn::Tensor>::err(std::string(ins[i].name) + " view REJECT: " + d[i].why);
    }

    AffineDesc& dk  = d[0];
    AffineDesc& da  = d[1];
    AffineDesc& dka = d[2];
    if (da.n_otr != dk.n_otr || da.n_otc != dk.n_otc || da.Ro != dk.Ro || da.Co != dk.Co)
        return Result<ttnn::Tensor>::err("k_update: k and a must present the same head grid");

    int64_t out_rows, out_cols;
    LayoutView out_lv(0, 0);
    if (out) {
        out_lv = out->layout(); out_rows = out->rows(); out_cols = out->cols();
    } else {
        const auto p = k.layout().plan();
        out_rows = p.Ro; out_cols = p.Co;
        out_lv = View::of_shape(out_rows, out_cols).layout();
    }
    AffineDesc dout = lower(out_lv);
    if (!dout.ok) return Result<ttnn::Tensor>::err("out view REJECT: " + dout.why);
    if (dout.n_otr != dk.n_otr || dout.n_otc != dk.n_otc || dout.Ro != dk.Ro || dout.Co != dk.Co)
        return Result<ttnn::Tensor>::err("k_update output grid must match k/a (shape-preserving)");

    const int64_t wt = dout.out_tiles();
    int64_t tpg = wt;
    uint32_t k_bcast = 0, a_bcast = 0, ka_bcast = 0;
    std::string e;
    e = classify_operand(dk, dout, out_rows, wt, tpg, k_bcast);
    if (!e.empty()) return Result<ttnn::Tensor>::err("k: " + e);
    e = classify_operand(da, dout, out_rows, wt, tpg, a_bcast);
    if (!e.empty()) return Result<ttnn::Tensor>::err("a: " + e);
    e = classify_operand(dka, dout, out_rows, wt, tpg, ka_bcast);
    if (!e.empty()) return Result<ttnn::Tensor>::err("k_a: " + e);

    std::string oe; auto od = bound_dst(out, oe);
    if (!oe.empty()) return Result<ttnn::Tensor>::err(oe);

    auto [attrs, args] = vku::ViewKUpdateDeviceOperation::invoke(
        *k.tensor(), *a.tensor(), *k_a.tensor(), dk, da, dka, dout,
        (int32_t)out_rows, (int32_t)out_cols, (int32_t)tpg, k_bcast, a_bcast, ka_bcast,
        std::move(od), memory_config);
    return Result<ttnn::Tensor>::ok(
        ttnn::device_operation::detail::launch<vku::ViewKUpdateDeviceOperation>(attrs, args));
}

// ---- INDEXED ROW GATHER / SCATTER (KV / state cache) ------------------------------
namespace {
std::string validate_index(const ttnn::Tensor& idx) {
    if (!idx.is_allocated()) return "index not allocated";
    if (idx.storage_type() != StorageType::DEVICE) return "index not on device";
    if (idx.dtype() != DataType::INT32) return "index not INT32";
    return "";
}
// Tiles-per-row for an indexed data tensor: total tiles / outer dim. Must divide evenly (whole-tile
// rows) -- the fast path needs the row dim to be a batch dim or a 32-multiple. 0 => not whole-tile.
int64_t tiles_per_row(const ttnn::Tensor& t, int64_t R) {
    const int64_t tiles = (int64_t)t.buffer()->num_pages();
    return (R > 0 && tiles % R == 0) ? tiles / R : 0;
}
}  // namespace

Result<ttnn::Tensor> get_rows(
    const ttnn::Tensor& data,
    const ttnn::Tensor& idx,
    const std::optional<MemoryConfig>& memory_config) {
    std::string md = validate_device_tile_bf16(data); if (!md.empty()) return Result<ttnn::Tensor>::err("data: " + md);
    std::string mi = validate_index(idx);             if (!mi.empty()) return Result<ttnn::Tensor>::err("idx: " + mi);

    const auto ds = data.logical_shape();
    const int64_t R = ds[0];
    const int64_t T = tiles_per_row(data, R);
    if (T == 0) return Result<ttnn::Tensor>::err("get_rows: rows are not whole-tile blocks "
                                                 "(outer dim must be a batch dim or 32-multiple)");
    const int64_t M = idx.logical_shape().volume();
    // output shape = [M, inner dims of data]; rank preserved.
    int32_t os[4] = {(int32_t)M, 0, 0, 0};
    const int rank = (int)ds.rank();
    if (rank > 4) return Result<ttnn::Tensor>::err("get_rows: rank > 4 unsupported");
    for (int i = 1; i < rank; i++) os[i] = (int32_t)ds[i];

    auto [attrs, args] = ri::RowIndexDeviceOperation::invoke(
        ri::MODE_GATHER, data, idx, std::nullopt, (int32_t)T, (int32_t)(M * T),
        (int32_t)idx.buffer()->num_pages(), (int32_t)R, rank, os[0], os[1], os[2], os[3],
        memory_config);
    return Result<ttnn::Tensor>::ok(
        ttnn::device_operation::detail::launch<ri::RowIndexDeviceOperation>(attrs, args));
}

Result<ttnn::Tensor> set_rows(const ttnn::Tensor& cache, const ttnn::Tensor& idx,
                              const ttnn::Tensor& src) {
    std::string mc = validate_device_tile_bf16(cache); if (!mc.empty()) return Result<ttnn::Tensor>::err("cache: " + mc);
    std::string ms = validate_device_tile_bf16(src);   if (!ms.empty()) return Result<ttnn::Tensor>::err("src: " + ms);
    std::string mi = validate_index(idx);              if (!mi.empty()) return Result<ttnn::Tensor>::err("idx: " + mi);

    const int64_t R = cache.logical_shape()[0];
    const int64_t T = tiles_per_row(cache, R);
    if (T == 0) return Result<ttnn::Tensor>::err("set_rows: cache rows are not whole-tile blocks");
    const int64_t M = idx.logical_shape().volume();
    if ((int64_t)src.buffer()->num_pages() != M * T)
        return Result<ttnn::Tensor>::err("set_rows: src tile count must equal M*T (rows to write)");

    // SCATTER is in-place into `cache`; the output spec is the cache's own.
    auto [attrs, args] = ri::RowIndexDeviceOperation::invoke(
        ri::MODE_SCATTER, src, idx, cache, (int32_t)T, (int32_t)(M * T),
        (int32_t)idx.buffer()->num_pages(), (int32_t)R, (int)cache.logical_shape().rank(), 0, 0, 0, 0);
    return Result<ttnn::Tensor>::ok(
        ttnn::device_operation::detail::launch<ri::RowIndexDeviceOperation>(attrs, args));
}

Result<ttnn::Tensor> add(
    const View& a,
    const View& b,
    std::optional<View> out,
    const std::optional<MemoryConfig>& memory_config) {
    return eltwise_bin(0, a, b, out, memory_config);
}
Result<ttnn::Tensor> sub(
    const View& a,
    const View& b,
    std::optional<View> out,
    const std::optional<MemoryConfig>& memory_config) {
    return eltwise_bin(1, a, b, out, memory_config);
}
Result<ttnn::Tensor> mul(
    const View& a,
    const View& b,
    std::optional<View> out,
    const std::optional<MemoryConfig>& memory_config) {
    return eltwise_bin(2, a, b, out, memory_config);
}

}  // namespace ttprm
