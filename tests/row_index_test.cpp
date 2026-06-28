// row_index_test.cpp -- ON-DEVICE derisk of GET_ROWS / SET_ROWS (the indexed whole-tile copy).
// -----------------------------------------------------------------------------
// MOCK-first: pure data movement, no compute. Each row of the data tensor is a distinct,
// bf16-exact function of its element id, so a byte-correct indexed gather/scatter reproduces
// the selected rows exactly. This also empirically pins down how TTNN lays out an INT32
// ROW_MAJOR index tensor (the kernel reads it raw from L1).
//
//   get_rows(data[R,A,B], idx[M])      -> y[M,A,B],  y[i] == data[idx[i]]
//   set_rows(cache[R,A,B], idx, src)   -> cache[idx[i]] := src[i]  (in place)

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

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::get_rows;
using ttprm::set_rows;

namespace {
int g_pass = 0, g_fail = 0;
void check(const char* name, bool ok, const char* detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name, detail[0] ? " -- " : "", detail);
    if (ok) ++g_pass; else ++g_fail;
}

// bf16-exact value carried by element id k (integers in [-120,120] are exact in bf16).
inline float val(int64_t k) { return (float)((k % 241) - 120); }

TensorSpec dspec(int64_t R, int64_t A, int64_t B) {
    return TensorSpec(ttnn::Shape({(uint32_t)R, (uint32_t)A, (uint32_t)B}),
                      TensorLayout(DataType::BFLOAT16,
                                   PageConfig(tt::tt_metal::Layout::TILE), MemoryConfig{}));
}
// data[r,i,j] = val(r*A*B + i*B + j) -- a distinct exact value per element (per row).
ttnn::Tensor make_data(distributed::MeshDevice* dev, int64_t R, int64_t A, int64_t B, int64_t base = 0) {
    std::vector<float> h((size_t)R * A * B);
    for (int64_t k = 0; k < R * A * B; k++) h[k] = val(base + k);
    return ttnn::Tensor::from_vector(h, dspec(R, A, B), dev);
}
// INT32 ROW_MAJOR index tensor of length M.
ttnn::Tensor make_index(distributed::MeshDevice* dev, const std::vector<int32_t>& v) {
    TensorSpec s(ttnn::Shape({(uint32_t)v.size()}),
                 TensorLayout(DataType::INT32, PageConfig(tt::tt_metal::Layout::ROW_MAJOR), MemoryConfig{}));
    return ttnn::Tensor::from_vector(v, s, dev);
}

bool exact(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++) if (a[i] != b[i]) return false;
    return true;
}
}  // namespace

int main() {
    std::printf(">>> row_index_test (GET_ROWS / SET_ROWS indexed whole-tile copy)\n");
    auto dev = distributed::MeshDevice::create_unit_mesh(0);
    dev->enable_program_cache();

    // ============ GET_ROWS ============
    {
        const int64_t R = 2, A = 32, B = 32;
        const MemoryConfig l1(TensorMemoryLayout::INTERLEAVED, BufferType::L1);
        auto data = make_data(dev.get(), R, A, B);
        auto idx = make_index(dev.get(), std::vector<int32_t>{1});
        auto r = get_rows(data, idx, l1);
        check("get_rows forwards memory_config=L1", r && r.value().memory_config().buffer_type() == BufferType::L1);
    }
    {
        // [6,64,64] (T=4 tiles/row), gather rows [3,0,5,1] -> [4,64,64].
        const int64_t R = 6, A = 64, B = 64;
        auto data = make_data(dev.get(), R, A, B);
        std::vector<int32_t> ids{3, 0, 5, 1};
        auto idx = make_index(dev.get(), ids);
        auto r = get_rows(data, idx);
        if (!r) { check("get_rows [6,64,64] idx{3,0,5,1}", false, r.error().c_str()); }
        else {
            auto dl = data.to_vector<float>();
            auto ol = r.value().to_vector<float>();
            std::vector<float> want((size_t)ids.size() * A * B);
            for (size_t i = 0; i < ids.size(); i++)
                for (int64_t k = 0; k < A * B; k++) want[i * A * B + k] = dl[ids[i] * A * B + k];
            check("get_rows [6,64,64] idx{3,0,5,1}", exact(ol, want), "vs data[idx[i]]");
        }
    }
    {
        // duplicate + repeated indices (a gather may repeat rows): idx{2,2,0,4,4,4}.
        const int64_t R = 5, A = 32, B = 32;          // T=1 tile/row
        auto data = make_data(dev.get(), R, A, B, 1000);
        std::vector<int32_t> ids{2, 2, 0, 4, 4, 4};
        auto idx = make_index(dev.get(), ids);
        auto r = get_rows(data, idx);
        if (!r) { check("get_rows [5,32,32] dup idx", false, r.error().c_str()); }
        else {
            auto dl = data.to_vector<float>(); auto ol = r.value().to_vector<float>();
            std::vector<float> want((size_t)ids.size() * A * B);
            for (size_t i = 0; i < ids.size(); i++)
                for (int64_t k = 0; k < A * B; k++) want[i * A * B + k] = dl[ids[i] * A * B + k];
            check("get_rows [5,32,32] dup idx{2,2,0,4,4,4}", exact(ol, want), "vs data[idx[i]]");
        }
    }
    {
        // BIG, multi-core: the [300,256,32] KV shape, T=8. Gather 64 rows -> 64*8=512 tiles.
        const int64_t R = 300, A = 256, B = 32;
        auto data = make_data(dev.get(), R, A, B);
        std::vector<int32_t> ids;
        for (int i = 0; i < 64; i++) ids.push_back((int32_t)((i * 7 + 3) % R));   // scattered rows
        auto idx = make_index(dev.get(), ids);
        auto r = get_rows(data, idx);
        if (!r) { check("get_rows [300,256,32] 64 rows (multi-core)", false, r.error().c_str()); }
        else {
            auto dl = data.to_vector<float>(); auto ol = r.value().to_vector<float>();
            bool ok = true;
            for (size_t i = 0; i < ids.size() && ok; i++)
                for (int64_t k = 0; k < A * B; k++)
                    if (ol[i * A * B + k] != dl[ids[i] * A * B + k]) { ok = false; break; }
            check("get_rows [300,256,32] 64 rows (multi-core)", ok, "vs data[idx[i]]");
        }
    }

    // ============ SET_ROWS (in-place) ============
    {
        // cache[6,64,64] seeded; write src[4,64,64] into rows {3,0,5,1}; verify in place.
        const int64_t R = 6, A = 64, B = 64;
        auto cache = make_data(dev.get(), R, A, B, 0);          // initial cache contents
        std::vector<int32_t> ids{3, 0, 5, 1};
        auto idx = make_index(dev.get(), ids);
        auto src = make_data(dev.get(), ids.size(), A, B, 50000);  // distinct new rows
        auto before = cache.to_vector<float>();
        auto sl = src.to_vector<float>();
        auto r = set_rows(cache, idx, src);
        if (!r) { check("set_rows [6,64,64] idx{3,0,5,1}", false, r.error().c_str()); }
        else {
            auto after = r.value().to_vector<float>();
            // expected: rows in ids replaced by src; all others unchanged.
            std::vector<float> want = before;
            for (size_t i = 0; i < ids.size(); i++)
                for (int64_t k = 0; k < A * B; k++) want[ids[i] * A * B + k] = sl[i * A * B + k];
            check("set_rows [6,64,64] idx{3,0,5,1} in-place", exact(after, want), "rows replaced, rest intact");
        }
    }
    {
        // BIG multi-core scatter: write 50 rows into a [300,256,32] cache.
        const int64_t R = 300, A = 256, B = 32;
        auto cache = make_data(dev.get(), R, A, B, 0);
        std::vector<int32_t> ids;
        for (int i = 0; i < 50; i++) ids.push_back((int32_t)((i * 11 + 5) % R));
        auto idx = make_index(dev.get(), ids);
        auto src = make_data(dev.get(), ids.size(), A, B, 200000);
        auto before = cache.to_vector<float>(); auto sl = src.to_vector<float>();
        auto r = set_rows(cache, idx, src);
        if (!r) { check("set_rows [300,256,32] 50 rows (multi-core)", false, r.error().c_str()); }
        else {
            auto after = r.value().to_vector<float>();
            std::vector<float> want = before;
            for (size_t i = 0; i < ids.size(); i++)   // later writes win on duplicate ids (none here)
                for (int64_t k = 0; k < A * B; k++) want[ids[i] * A * B + k] = sl[i * A * B + k];
            check("set_rows [300,256,32] 50 rows (multi-core)", exact(after, want), "in-place scatter");
        }
    }

    // ============ OUT-OF-BOUNDS INDEX (safety: must NOT hang / corrupt DRAM) ============
    // An index < 0 or >= R would compute an out-of-range DRAM page. The kernels guard it: the
    // GATHER reader CLAMPS an OOB row to 0, the SCATTER writer SKIPS the write. These cases pin
    // that deterministic behavior (and, more importantly, that the op returns instead of hanging).
    {
        // GET_ROWS with a mix of valid and OOB indices: -1 (negative) and 6 (== R) both clamp to 0.
        const int64_t R = 6, A = 64, B = 64;
        auto data = make_data(dev.get(), R, A, B);
        std::vector<int32_t> ids{3, -1, 6, 1};
        auto idx = make_index(dev.get(), ids);
        auto r = get_rows(data, idx);
        if (!r) { check("get_rows OOB idx{3,-1,6,1} (no hang)", false, r.error().c_str()); }
        else {
            auto dl = data.to_vector<float>(); auto ol = r.value().to_vector<float>();
            std::vector<float> want((size_t)ids.size() * A * B);
            for (size_t i = 0; i < ids.size(); i++) {
                const int64_t row = (ids[i] < 0 || ids[i] >= R) ? 0 : ids[i];   // OOB -> clamped to 0
                for (int64_t k = 0; k < A * B; k++) want[i * A * B + k] = dl[row * A * B + k];
            }
            check("get_rows OOB idx{3,-1,6,1} clamps to row 0", exact(ol, want), "no hang, deterministic");
        }
    }
    {
        // SET_ROWS with an OOB index (7 >= R): that row's write is SKIPPED, the cache is otherwise
        // updated. Valid rows {3,1} are written; row targeted by the OOB id stays untouched.
        const int64_t R = 6, A = 64, B = 64;
        auto cache = make_data(dev.get(), R, A, B, 0);
        std::vector<int32_t> ids{3, 7, 1};
        auto idx = make_index(dev.get(), ids);
        auto src = make_data(dev.get(), ids.size(), A, B, 50000);
        auto before = cache.to_vector<float>(); auto sl = src.to_vector<float>();
        auto r = set_rows(cache, idx, src);
        if (!r) { check("set_rows OOB idx{3,7,1} (no hang)", false, r.error().c_str()); }
        else {
            auto after = r.value().to_vector<float>();
            std::vector<float> want = before;                       // OOB id 7 -> skipped (no write)
            for (size_t i = 0; i < ids.size(); i++) {
                if (ids[i] < 0 || ids[i] >= R) continue;
                for (int64_t k = 0; k < A * B; k++) want[ids[i] * A * B + k] = sl[i * A * B + k];
            }
            check("set_rows OOB idx{3,7,1} skips OOB write", exact(after, want), "no hang, cache intact");
        }
    }

    std::printf("  %d passed, %d failed\n", g_pass, g_fail);
    const bool ok = (g_fail == 0);
    std::printf("RESULT: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
