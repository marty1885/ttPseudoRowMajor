// Reader endpoint for indexed whole-tile copies.
// Each indexed row is T contiguous tile pages. ROW_INDEXED gathers from the
// indexed source row; the dense mode gathers page ordinals directly.
//
//   ord  = this core's global output-tile ordinal (t0 + k)
//   row  = ord / T ,  tir = ord % T          (which selected row, which tile within it)
//   page = idx[row]*T + tir   (ROW_INDEXED, the gather)  |   page = ord   (dense source)
//
// Batched issue: emit K reads, use one barrier, then push K tiles.
//
// Compile args: src TensorAccessorArgs [, idx TensorAccessorArgs if ROW_INDEXED].
// Runtime args: 0 src_addr  1 count  2 t0  3 T  4 K  [5 idx_addr  6 idx_pages  7 n_idx_rows].
//
// Bounds: invalid indices are clamped to row 0 to keep reads in range.

#include <cstdint>

namespace {
constexpr uint32_t cb_io  = 0;
constexpr uint32_t cb_idx = 1;
}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t count    = get_arg_val<uint32_t>(1);
    const uint32_t t0       = get_arg_val<uint32_t>(2);
    const uint32_t T        = get_arg_val<uint32_t>(3);
    const uint32_t K        = get_arg_val<uint32_t>(4);

    constexpr auto src_args = TensorAccessorArgs<0>();
    const uint32_t tile_bytes = get_tile_size(cb_io);
    const auto src = TensorAccessor(src_args, src_addr, tile_bytes);

#ifdef ROW_INDEXED
    const uint32_t idx_addr   = get_arg_val<uint32_t>(5);
    const uint32_t idx_pages  = get_arg_val<uint32_t>(6);
    const uint32_t n_idx_rows = get_arg_val<uint32_t>(7);
    constexpr auto idx_args = TensorAccessorArgs<src_args.num_compile_time_args()>();
    const uint32_t idx_pb = get_tile_size(cb_idx);
    const auto idx = TensorAccessor(idx_args, idx_addr, idx_pb);
    // Load the full index once; each core's ordinal range may reference any row.
    cb_reserve_back(cb_idx, idx_pages);
    const uint32_t il1 = get_write_ptr(cb_idx);
    for (uint32_t p = 0; p < idx_pages; p++)
        noc_async_read(idx.get_noc_addr(p, 0), il1 + p * idx_pb, idx_pb);
    noc_async_read_barrier();
    volatile tt_l1_ptr int32_t* I = reinterpret_cast<volatile tt_l1_ptr int32_t*>(il1);
#endif

    for (uint32_t k = 0; k < count; k += K) {
        const uint32_t chunk = (count - k < K) ? (count - k) : K;
        cb_reserve_back(cb_io, chunk);
        const uint32_t wp = get_write_ptr(cb_io);
        for (uint32_t j = 0; j < chunk; j++) {
            const uint32_t ord = t0 + k + j;
#ifdef ROW_INDEXED
            const uint32_t row  = ord / T;
            const uint32_t ri   = (uint32_t)I[row];
            const uint32_t page = (ri < n_idx_rows ? ri : 0u) * T + (ord - row * T);
#else
            const uint32_t page = ord;
#endif
            noc_async_read(src.get_noc_addr(page, 0), wp + j * tile_bytes, tile_bytes);
        }
        noc_async_read_barrier();
        cb_push_back(cb_io, chunk);
    }
}
