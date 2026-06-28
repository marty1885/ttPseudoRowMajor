// Writer endpoint for scattering cb_io to the output view.
// TTPRM_OUT_VIEW selects DenseTensorView for whole-page writes or AffineTensorView
// for face-row scatters. The writer runs on RISCV_1/NOC_1 so reads and writes can
// overlap through cb_io.
//
// Compile args: dst TensorAccessorArgs. Define: TTPRM_OUT_VIEW.
// Runtime args: same shared layout as the reader (see view_realize_reader.cpp).

#include <cstdint>

#include "tensor_view.hpp"

#ifndef TTPRM_OUT_VIEW
#define TTPRM_OUT_VIEW ttprm::DenseTensorView
#endif
#ifndef TTPRM_OUT_CB
#define TTPRM_OUT_CB 0
#endif

namespace {
constexpr uint32_t cb_io     = TTPRM_OUT_CB;
constexpr uint32_t view_base = 2;  // after addr and n_tiles
}  // namespace

void kernel_main() {
    const uint32_t dst_addr = get_arg_val<uint32_t>(0);
    const uint32_t n_tiles  = get_arg_val<uint32_t>(1);

    constexpr auto dst_args = TensorAccessorArgs<0>();
    const auto dst = TensorAccessor(dst_args, dst_addr, get_tile_size(cb_io));

    // Scatter writes the live face-rows from the provided tile; no pad seed is needed.
    auto out = ttprm::TensorViewArgs<TTPRM_OUT_VIEW, view_base>::make(dst);

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_io, 1);
        out.scatter_tile(get_read_ptr(cb_io));
        noc_async_write_barrier();
        cb_pop_front(cb_io, 1);
        out.advance();
    }
}
