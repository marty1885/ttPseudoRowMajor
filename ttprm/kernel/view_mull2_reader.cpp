// Reader endpoint for fused mul_l2_norm.
// Each row-block gathers matching A and B head rows into cb_a/cb_b. Each
// operand has its own view descriptor.
//
// Compile args: a TensorAccessorArgs, then b.  Defines: TTPRM_A_VIEW, TTPRM_B_VIEW.
// Runtime args: 0 a_addr 1 b_addr 2 n_rb 3 ctiles  4.. a view block, then b view block.

#include <cstdint>

#include "tensor_view.hpp"

#ifndef TTPRM_A_VIEW
#define TTPRM_A_VIEW ttprm::AffineTensorView
#endif
#ifndef TTPRM_B_VIEW
#define TTPRM_B_VIEW ttprm::AffineTensorView
#endif

namespace {
constexpr uint32_t cb_a    = 0;
constexpr uint32_t cb_b    = 6;
constexpr uint32_t cb_zero = 31;
constexpr uint32_t view_base = 4;  // views begin after a_addr(0), b_addr(1), n_rb(2), ctiles(3)
}  // namespace

void kernel_main() {
    const uint32_t a_addr  = get_arg_val<uint32_t>(0);
    const uint32_t b_addr  = get_arg_val<uint32_t>(1);
    const uint32_t n_rb    = get_arg_val<uint32_t>(2);
    const uint32_t ctiles  = get_arg_val<uint32_t>(3);

    using AArgs = ttprm::TensorViewArgs<TTPRM_A_VIEW, view_base>;
    using BArgs = ttprm::TensorViewArgs<TTPRM_B_VIEW, AArgs::next()>;

    uint32_t zero_rp = 0;
    if constexpr (AArgs::gather_needs_pad_seed || BArgs::gather_needs_pad_seed) {
        cb_wait_front(cb_zero, 1);
        zero_rp = get_read_ptr(cb_zero);
    }

    constexpr uint32_t NA = TensorAccessorArgs<0>::num_compile_time_args();
    const auto a_acc = TensorAccessor(TensorAccessorArgs<0>(),  a_addr, get_tile_size(cb_a));
    const auto b_acc = TensorAccessor(TensorAccessorArgs<NA>(), b_addr, get_tile_size(cb_b));
    auto a = AArgs::make(a_acc, zero_rp);
    auto b = BArgs::make(b_acc, zero_rp);
    const uint32_t a_tb = get_tile_size(cb_a);
    const uint32_t b_tb = get_tile_size(cb_b);

    for (uint32_t rb = 0; rb < n_rb; rb++) {
        cb_reserve_back(cb_a, ctiles);
        cb_reserve_back(cb_b, ctiles);
        const uint32_t awp = get_write_ptr(cb_a);
        const uint32_t bwp = get_write_ptr(cb_b);
        for (uint32_t i = 0; i < ctiles; i++) { a.gather_tile(awp + i * a_tb); a.advance(); }
        for (uint32_t i = 0; i < ctiles; i++) { b.gather_tile(bwp + i * b_tb); b.advance(); }
        noc_async_read_barrier();
        cb_push_back(cb_a, ctiles);
        cb_push_back(cb_b, ctiles);
    }
}
