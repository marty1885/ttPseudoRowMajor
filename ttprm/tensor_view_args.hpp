// Host packer for device TensorViewArgs runtime blocks.
// view_kind.hpp defines the shared ABI consumed by kernel/tensor_view.hpp.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "view_kind.hpp"
#include "view_realize.hpp"  // AffineDesc

namespace ttprm {

// Return the device view class name used in kernel defines.
inline std::string view_type_name(uint32_t kind) {
    if (kind == VIEW_DENSE)   return "ttprm::DenseTensorView";
    if (kind == VIEW_GROUPED) return "ttprm::GroupedTensorView";
    return "ttprm::AffineTensorView";
}

// Append a view's runtime-arg block. t0 is this core's first output tile.
inline void append_view_args(std::vector<uint32_t>& v, uint32_t kind, uint32_t t0, const AffineDesc& d) {
    v.push_back(t0);
    if (kind != VIEW_DENSE) {
        v.push_back((uint32_t)d.base);  v.push_back((uint32_t)d.S_row);
        v.push_back((uint32_t)d.Cs);    v.push_back((uint32_t)d.nct);
        v.push_back((uint32_t)d.Ro);    v.push_back((uint32_t)d.Co);
        v.push_back((uint32_t)d.n_otc);
    }
    if (kind == VIEW_GROUPED) {
        v.push_back((uint32_t)d.grp_P); v.push_back((uint32_t)d.grp_A); v.push_back((uint32_t)d.grp_B);
    }
}

}  // namespace ttprm
