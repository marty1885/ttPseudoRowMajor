# ttprm library (square-1) --------------------------------------------------------
# Header-only composer (view_access.hpp) + host lowering (view_realize.hpp) + the
# ttnn::Tensor surface (view_realize_op.cpp). The ttnn surface is the only TU that
# links ttnn; it carries the cached, trace-safe device_operation.
#
# Requires the `ttmetal` and `ttnnlib` IMPORTED INTERFACE targets from the parent.

add_library(ttprm_ttnn STATIC view_realize_op.cpp)
target_include_directories(ttprm_ttnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(ttprm_ttnn PUBLIC ttmetal ttnnlib)

# Absolute root so kernel sources + JIT compiler_include_paths resolve regardless of cwd.
target_compile_definitions(ttprm_ttnn PUBLIC TTPRM_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")
