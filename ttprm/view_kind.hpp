// Shared view-strategy tokens and per-kind runtime-arg counts.
// Kept host/device safe so append_view_args and TensorViewArgs use the same ABI.
// A view's runtime args (everything past the kernel's addr/n_tiles) are:
//   DENSE   : [t0]                                  -- start page; whole-page transfer
//   AFFINE  : [t0, base, S_row, Cs, nct, Ro, Co, n_otc]  -- affine map + start tile
//   GROUPED : [t0, base, S_row, Cs, nct, Ro, Co, n_otc, P, A, B]  -- AFFINE block + the 2-level row
//             map source_row(oR) = A*(oR/P) + B*(oR%P), then source element
//             base + source_row*S_row + oC. See PREFILL token-shift
//             (cur/x_prev: A=0,B=1 -> r%nt ; weight: A=1,B=0 -> r/nt).
// Odometer increments are derived on device; see kernel/tensor_view.hpp.

#pragma once

#include <cstdint>

namespace ttprm {

constexpr uint32_t VIEW_DENSE   = 0u;
constexpr uint32_t VIEW_ROW     = 1u;   // reserved (constant per-row page delta); routes to AFFINE
constexpr uint32_t VIEW_AFFINE  = 2u;
constexpr uint32_t VIEW_GROUPED = 3u;   // 2-level (periodic) row map; AFFINE block + [P, A, B]

// Number of runtime args consumed by a view of `kind`.
constexpr uint32_t view_num_args(uint32_t kind) {
    if (kind == VIEW_DENSE)   return 1u;
    if (kind == VIEW_GROUPED) return 11u;
    return 8u;
}

}  // namespace ttprm
