// view_realize_test.cpp -- ON-DEVICE derisk of the generic copy-list gather->scatter.
// -----------------------------------------------------------------------------
// Proves ttprm::view_realize (the pure relayout: face-row gather -> whole-tile write,
// no compute) round-trips complex view chains BIT-EXACT on wormhole_b0, single-core AND
// multi-core. Oracle = the host realize_ref() (already proven in view_access_test): it
// gives the exact source element id k that each output slot must hold; the source is
// filled with a bf16-exact function of k, so a byte-correct gather reproduces it exactly.
//
// MOCK-first: this is gather+scatter with NO math -- if the address map is wrong it shows
// here as a mismatch before any compute is layered on.

#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "view_access.hpp"
#include "view_realize.hpp"

using namespace tt::tt_metal;
using ttprm::LayoutView;
using ttprm::TILE;
using ttprm::PAGE_ELTS;

namespace {
int g_pass = 0, g_fail = 0;
void check(const char* name, bool ok, const char* detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name, detail[0] ? " -- " : "", detail);
    if (ok) ++g_pass; else ++g_fail;
}

// bf16-exact value carried by source element id k (integers in [-120,120] are exact).
inline float val(int32_t k) { return (float)((k % 241) - 120); }

std::shared_ptr<distributed::MeshBuffer> MakeBuf(distributed::MeshDevice* dev,
                                                 uint32_t n_pages, uint32_t page_bytes) {
    distributed::DeviceLocalBufferConfig c{.page_size = page_bytes, .buffer_type = BufferType::DRAM};
    distributed::ReplicatedBufferConfig b{.size = (uint64_t)page_bytes * n_pages};
    return distributed::MeshBuffer::create(b, c, dev);
}

// Run view_realize for `v` and return readback (bf16, tiled, size out_tiles*PAGE_ELTS).
// num_cores==1 -> single core; else split out tiles across the grid.
std::vector<bfloat16> run_realize(distributed::MeshDevice* dev, const LayoutView& v,
                                  const LayoutView::Plan& p, bool multicore) {
    auto& cq = dev->mesh_command_queue();
    const uint32_t ts = sizeof(bfloat16) * PAGE_ELTS;             // 2048
    const uint32_t src_tiles = (uint32_t)((v.src_rows() / TILE) * (v.src_cols() / TILE));
    const uint32_t out_tiles = (uint32_t)p.out_tiles();          // padded tile grid

    // Lower the planned view onto a device executor (the seam). Today this always
    // selects the SEGMENT executor + its copy-list descriptor; the kernel's MapKind
    // define and descriptor come from here, never hand-built by the test.
    const ttprm::AccessPlan ap = ttprm::lower(v, p);

    // --- source: fill phys slot(row,col) = val(row*Cs+col) ---
    std::vector<bfloat16> src_host((size_t)src_tiles * PAGE_ELTS, bfloat16(0.0f));
    for (int64_t row = 0; row < v.src_rows(); row++)
        for (int64_t col = 0; col < v.src_cols(); col++)
            src_host[(size_t)ttprm::phys(row, col, v.src_cols())] =
                bfloat16(val((int32_t)(row * v.src_cols() + col)));
    auto bsrc = MakeBuf(dev, src_tiles, ts);
    distributed::EnqueueWriteMeshBuffer(cq, bsrc, src_host, false);

    // --- descriptor: the lowered executor's runtime input (SEGMENT: out_tiles*512B) ---
    auto bdesc = MakeBuf(dev, out_tiles, 512u);
    distributed::EnqueueWriteMeshBuffer(cq, bdesc, ap.descriptor, false);

    auto bout = MakeBuf(dev, out_tiles, ts);

    // --- program ---
    Program prog = CreateProgram();
    const auto grid = dev->compute_with_storage_grid_size();
    uint32_t num_cores = 1;
    CoreRangeSet all_cores(CoreRange{{0, 0}, {0, 0}});
    std::vector<CoreCoord> cores{{0, 0}};
    std::vector<uint32_t> ostart{0}, count{out_tiles};
    uint32_t cg1_units = out_tiles;
    CoreRangeSet cg1, cg2;
    if (multicore) {
        auto [nc, ac, c1, c2, u1, u2] = split_work_to_cores(grid, out_tiles);
        num_cores = nc; all_cores = ac; cg1 = c1; cg2 = c2; cg1_units = u1;
        cores = corerange_to_cores(ac, nc, /*row_wise=*/true);
        ostart.assign(nc, 0); count.assign(nc, 0);
        uint32_t t = 0;
        for (uint32_t i = 0; i < nc; i++) {
            uint32_t u = cg1.contains(cores[i]) ? u1 : u2;
            ostart[i] = t; count[i] = u; t += u;
        }
    }

    CircularBufferConfig cb0(2 * ts, {{tt::CBIndex::c_0, tt::DataFormat::Float16_b}});
    cb0.set_page_size(tt::CBIndex::c_0, ts);
    CreateCircularBuffer(prog, all_cores, cb0);
    CircularBufferConfig cb1(2 * 512u, {{tt::CBIndex::c_1, tt::DataFormat::UInt32}});
    cb1.set_page_size(tt::CBIndex::c_1, 512u);
    CreateCircularBuffer(prog, all_cores, cb1);
    // CB 31: the resident zero tile (pad identity) the compute engine fills via fill_tile.
    CircularBufferConfig cb31(ts, {{tt::CBIndex::c_31, tt::DataFormat::Float16_b}});
    cb31.set_page_size(tt::CBIndex::c_31, ts);
    CreateCircularBuffer(prog, all_cores, cb31);

    std::vector<uint32_t> ct;
    TensorAccessorArgs(*bsrc).append_to(ct);
    TensorAccessorArgs(*bout).append_to(ct);
    TensorAccessorArgs(*bdesc).append_to(ct);
    const std::string ttnn_inc = std::string(getenv("TT_METAL_HOME")) + "/ttnn";  // tt_memmove
    // The JIT specialization seam: the MapKind executor lower() chose becomes a compile
    // define. The kernel binary is keyed on (source-hash + defines), so this is the
    // per-MapKind kernel-source cache key the design calls for.
    std::map<std::string, std::string> dm_defines{
        {"TTPRM_MAPKIND", std::to_string(ap.mapkind_define())}};
    auto dm = CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_dm.cpp", all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default,
                           .compile_args = ct, .defines = dm_defines,
                           .compiler_include_paths = {TTPRM_ROOT, ttnn_inc}});
    // compute kernel: fill one zero tile into CB 31 (the pad identity).
    CreateKernel(prog, TTPRM_ROOT "/kernel/view_realize_compute.cpp", all_cores,
        ComputeConfig{.compile_args = {}});

    for (uint32_t i = 0; i < num_cores; i++)
        SetRuntimeArgs(prog, dm, cores[i], std::vector<uint32_t>{
            (uint32_t)bsrc->address(), (uint32_t)bout->address(), (uint32_t)bdesc->address(),
            ostart[i], ostart[i] + count[i]});

    distributed::MeshWorkload wl;
    wl.add_program(distributed::MeshCoordinateRange(dev->shape()), std::move(prog));
    distributed::EnqueueMeshWorkload(cq, wl, false);
    distributed::Finish(cq);

    std::vector<bfloat16> got((size_t)out_tiles * PAGE_ELTS);
    distributed::EnqueueReadMeshBuffer(cq, got, bout, true);
    return got;
}

// Compare a readback against the host oracle (realize_ref ids -> val()). A pad slot
// (ref id == -1) must read back as the zero pad identity (0.0f), proving the compute
// fill_tile -> CB31 -> reader-seed path filled it (not DRAM garbage).
bool matches_oracle(const LayoutView& v, const LayoutView::Plan& p, const std::vector<bfloat16>& got) {
    auto ref = ttprm::realize_ref(v, p);   // int32 source id per output phys slot (-1 = pad)
    if (got.size() != ref.size()) return false;
    for (size_t m = 0; m < ref.size(); m++) {
        float want = (ref[m] == -1) ? 0.0f : val(ref[m]);
        if (float(got[m]) != want) return false;
    }
    return true;
}

void run_case(distributed::MeshDevice* dev, const char* name, const LayoutView& v,
              ttprm::AccessClass want) {
    auto p = v.plan();
    if (!p.ok()) { check(name, false, p.why.c_str()); return; }
    bool cls_ok = (p.cls == want);
    bool sc = matches_oracle(v, p, run_realize(dev, v, p, /*multicore=*/false));
    bool mc = matches_oracle(v, p, run_realize(dev, v, p, /*multicore=*/true));
    char d[160];
    std::snprintf(d, sizeof d, "class=%s single=%s multi=%s",
                  ttprm::to_string(p.cls), sc ? "ok" : "MISMATCH", mc ? "ok" : "MISMATCH");
    check(name, cls_ok && sc && mc, d);
}
}  // namespace

int main() {
    std::printf(">>> view_realize_test (on-device generic copy-list gather->scatter)\n");
    auto dev = distributed::MeshDevice::create_unit_mesh(0);

    // [reshape -> reshape -> store]  [1,4096]->[64,64]->[128,32]  (ROW)
    run_case(dev.get(), "reshape->reshape [1,4096]->[64,64]->[128,32]",
             LayoutView(32, 4096).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64}).reshape({128, 32}),
             ttprm::AccessClass::ROW);

    // [reshape] head  [1,4096]->[64,64]  (ROW)
    run_case(dev.get(), "reshape [1,4096]->[64,64] (head)",
             LayoutView(32, 4096).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64}),
             ttprm::AccessClass::ROW);

    // [slice -> reshape -> store]  full-width tile-aligned slice  (SEGMENT)
    run_case(dev.get(), "slice(rows32:96 full-width)->reshape->[128,32]",
             LayoutView(128, 64).slice({{32, 96, 1}, {0, 64, 1}}).reshape({128, 32}),
             ttprm::AccessClass::SEGMENT);

    // identity verbatim  (TILE)
    run_case(dev.get(), "identity [64,64]", LayoutView(64, 64), ttprm::AccessClass::TILE);

    // face-aligned col slice [16:48]  (SEGMENT)
    run_case(dev.get(), "face-aligned col slice [16:48]",
             LayoutView(64, 64).slice({{0, 64, 1}, {16, 48, 1}}), ttprm::AccessClass::SEGMENT);

    // PADDING (row pad): logical [1,4096] -> physical [32,4096]; 1 live row + 31 pad rows.
    // The 31 padded rows must read back as exact 0 (compute zero tile, not DRAM garbage).
    run_case(dev.get(), "padded [1,4096] (1 live row of 32)",
             LayoutView(32, 4096).slice({{0, 1, 1}, {0, 4096, 1}}), ttprm::AccessClass::SEGMENT);

    // PADDING (col pad): logical [64,48] -> physical [64,64]; right face of every 2nd
    // tile-col is pad. 48%16==0 (face-aligned width) but 48%32!=0 (tile-padded).
    run_case(dev.get(), "padded [64,48] (col pad to 64)",
             LayoutView(64, 64).slice({{0, 64, 1}, {0, 48, 1}}), ttprm::AccessClass::SEGMENT);

    std::printf("  %d passed, %d failed\n", g_pass, g_fail);
    const bool ok = (g_fail == 0);
    std::printf("RESULT: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
