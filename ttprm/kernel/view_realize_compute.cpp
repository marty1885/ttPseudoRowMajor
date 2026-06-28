// Produces the pad-identity tile used when a reader assembles partial output tiles.
// The tile is generated once in CB 31 and remains resident; readers peek it as the
// seed buffer before overlaying live face-rows.

#include "api/compute/compute_kernel_api.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"

namespace { constexpr uint32_t cb_zero = 31; }

void kernel_main() {
    unary_op_init_common(cb_zero, cb_zero);
    fill_tile_init();
    cb_reserve_back(cb_zero, 1);
    tile_regs_acquire();
    fill_tile(0, 0.0f);
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_zero);
    cb_push_back(cb_zero, 1);
    tile_regs_release();
}
