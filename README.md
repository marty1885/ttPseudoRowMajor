# ttPseudoRowMajor

`ttprm` is a small TTNN-facing library for running view-shaped tensor work without
first materializing the view in DRAM. It targets Tenstorrent `TILE` tensors and
uses cached device operations to gather/scatter through layout views.

The current public API lives in:

```text
ttprm/ttprm.hpp
```

## What It Provides

- `ttprm::View`: a lightweight view wrapper around a `ttnn::Tensor`.
- `ttprm::realize`: materialize a view, or scatter into a bound output view.
- Fused view-aware ops: `add`, `sub`, `mul`, `fma`, `lerp`.
- View-aware norms: `l2_norm`, `rms_norm`, `layer_norm`, `mul_l2_norm`.
- RWKV helper: `rk`.
- Indexed row cache ops: `get_rows`, `set_rows`.
- Optional `memory_config` on fresh-output ops, matching TTNN-style output
  memory configuration.

Inputs are expected to be device-resident BF16 `TILE` tensors unless the specific
API says otherwise.

## Build

Set the TT-Metal paths used by your local checkout:

```sh
export TT_METAL_HOME=/home/marty/Documents/tt/tt-metal
export TT_METAL_RUNTIME_ROOT=/home/marty/Documents/tt/tt-metal
```

Configure and build:

```sh
cmake -S . -B build
cmake --build build --target ttprm_ttnn
```

Build the tests:

```sh
cmake --build build --target view_ttnn_test row_index_test view_lerp_rwkv_test view_lerp_rwkv_4d_test
```

Run the configured test suite:

```sh
TT_METAL_HOME=/home/marty/Documents/tt/tt-metal \
TT_METAL_RUNTIME_ROOT=/home/marty/Documents/tt/tt-metal \
ctest --test-dir build --output-on-failure
```

## Basic Usage

```cpp
#include <stdexcept>

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::view_of;

void example(ttnn::Tensor x) {
    // Present a flat [1, 4096] tensor as [64, 64] without calling ttnn::reshape.
    auto head = view_of(x).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});

    auto y = ttprm::realize(head);
    if (!y) {
        throw std::runtime_error(y.error());
    }

    ttnn::Tensor out = y.value();
}
```

## Fused Elementwise

```cpp
#include <stdexcept>

#include "ttprm.hpp"

using ttprm::view_of;
using ttprm::view_of_shape;

void add_example(ttnn::Tensor a, ttnn::Tensor b) {
    auto av = view_of(a).slice({{0, 1, 1}, {0, 4096, 1}}).reshape({64, 64});
    auto bv = view_of(b);
    auto ov = view_of_shape(64, 64);

    auto y = ttprm::add(av, bv, ov);
    if (!y) {
        throw std::runtime_error(y.error());
    }
}
```

## Output Memory Config

Fresh-output ops accept an optional output `MemoryConfig`. Bound output views
use the bound tensor's existing memory configuration.

```cpp
#include <stdexcept>

#include "ttprm.hpp"

using namespace tt::tt_metal;
using ttprm::view_of;

void memory_config_example(ttnn::Tensor a, ttnn::Tensor b) {
    MemoryConfig l1(TensorMemoryLayout::INTERLEAVED, BufferType::L1);

    auto y = ttprm::add(view_of(a), view_of(b), std::nullopt, l1);
    if (!y) {
        throw std::runtime_error(y.error());
    }

    // y.value() is allocated using L1 memory config.
}
```

## Indexed Rows

```cpp
#include <stdexcept>

#include "ttprm.hpp"

void kv_cache_example(ttnn::Tensor cache, ttnn::Tensor idx, ttnn::Tensor src) {
    auto gathered = ttprm::get_rows(cache, idx);
    if (!gathered) {
        throw std::runtime_error(gathered.error());
    }

    auto updated_cache = ttprm::set_rows(cache, idx, src);
    if (!updated_cache) {
        throw std::runtime_error(updated_cache.error());
    }
}
```

## Notes

- `Result<T>` is non-throwing; check it before calling `.value()`.
- `view_of(tensor)` binds a view to an existing tensor.
- `view_of_shape(rows, cols)` creates an output-shape view with no tensor.
- A bound output like `view_of(dst)` writes into `dst` in place.
- Unsupported view patterns return `Result::err(...)` before launching device work.
