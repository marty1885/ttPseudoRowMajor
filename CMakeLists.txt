cmake_minimum_required(VERSION 3.15)

project(ttPseudoRowMajor)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

list(PREPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
include(CPM)

if(CMAKE_VERSION VERSION_LESS 3.25)
    add_subdirectory(dependencies EXCLUDE_FROM_ALL)
else()
    add_subdirectory(dependencies EXCLUDE_FROM_ALL SYSTEM)
endif()

if("$ENV{TT_METAL_HOME}" STREQUAL "")
    message(FATAL_ERROR "TT_METAL_HOME is not set")
endif()

# tt-metal include/link flags (copied from the wkv7 standalone harness).
add_library(ttmetal IMPORTED INTERFACE)
target_include_directories(ttmetal INTERFACE
    $ENV{TT_METAL_HOME}/build/include
    $ENV{TT_METAL_HOME}
    $ENV{TT_METAL_HOME}/tt_metal
    $ENV{TT_METAL_HOME}/tt_metal/api/
    $ENV{TT_METAL_HOME}/tt_metal/hostdevcommon/api
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api/
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api/umd
    $ENV{TT_METAL_HOME}/tt_metal/third_party/umd/device/api/umd/device
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public/tracy
    $ENV{TT_METAL_HOME}/tt_metal/third_party/tracy/public/
    $ENV{TT_METAL_HOME}/tt_metal/hw/inc/
    $ENV{TT_METAL_HOME}/tt_stl/
)

target_compile_definitions(ttmetal INTERFACE
    FMT_HEADER_ONLY
    SPDLOG_FMT_EXTERNAL
)

target_link_directories(ttmetal INTERFACE
    $ENV{TT_METAL_HOME}/build/tt_metal
    $ENV{TT_METAL_HOME}/build/tt_stl
)

target_link_libraries(ttmetal INTERFACE
    fmt
    yaml-cpp
    Boost::core
    Boost::container
    tt_metal
    tt_stl
)

# --- ttnn (high-level op library: native TM/compute ops + the trace API) ------
add_library(ttnnlib IMPORTED INTERFACE)
target_include_directories(ttnnlib INTERFACE
    $ENV{TT_METAL_HOME}/ttnn
    $ENV{TT_METAL_HOME}/ttnn/api
    $ENV{TT_METAL_HOME}/ttnn/cpp
)
target_link_libraries(ttnnlib INTERFACE $ENV{TT_METAL_HOME}/build/lib/_ttnncpp.so)

# =============================================================================
# SQUARE 1 — clean rebuild. Phase-1 implementation is archived in ../old/ (its
# CMakeLists.txt builds it standalone). Read LEARNINGS.md before adding targets.
#
# The build scaffolding above (ttmetal + ttnnlib imported targets, CPM deps) is
# validated and reusable. The library lives in src/ (empty for now); tests in tests/.
# When src/ has sources, add e.g. `add_subdirectory(src)` here and link it in tests/.
# =============================================================================

add_subdirectory(ttprm)

enable_testing()
add_subdirectory(tests)
add_subdirectory(bench)
