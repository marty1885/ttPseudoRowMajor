// Reader endpoint for realizing a view into cb_io.
// TTPRM_IN_VIEW selects either DenseTensorView for whole-page reads or
// AffineTensorView for face-row gathers. Dimensions are runtime arguments.
//
// Compile args: src TensorAccessorArgs. Define: TTPRM_IN_VIEW.
// Runtime args use the shared reader/writer layout. AffineTensorView derives the
// odometer increments and per-core seed from this block:
//   0 addr  1 n_tiles  2 t0 (this core's start tile / start page for DENSE)
//   3 base  4 S_row  5 Cs  6 nct  7 Ro  8 Co  9 n_otc

#include <cstdint>

#include "tensor_view.hpp"

#ifndef TTPRM_IN_VIEW
#define TTPRM_IN_VIEW ttprm::AffineTensorView
#endif

namespace {
constexpr uint32_t cb_io     = 0;
constexpr uint32_t cb_zero   = 31;
constexpr uint32_t view_base = 2;  // after addr and n_tiles
}  // namespace

void kernel_main() {
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles  = get_arg_val<uint32_t>(1);

    constexpr auto src_args = TensorAccessorArgs<0>();
    const auto src = TensorAccessor(src_args, src_addr, get_tile_size(cb_io));

    using InArgs = ttprm::TensorViewArgs<TTPRM_IN_VIEW, view_base>;
    uint32_t zero_rp = 0;
    if constexpr (InArgs::gather_needs_pad_seed) {
        cb_wait_front(cb_zero, 1);
        zero_rp = get_read_ptr(cb_zero);
    }
    auto in = InArgs::make(src, zero_rp);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_reserve_back(cb_io, 1);
        in.gather_tile(get_write_ptr(cb_io));
        noc_async_read_barrier();
        cb_push_back(cb_io, 1);
        in.advance();
    }
}
