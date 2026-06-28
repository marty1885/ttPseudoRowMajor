# Square-1 benchmarks.
#  - view_trace_bench: ttprm::realize vs chained ttnn ops under tt-metal TRACE
#                      (device-time numbers; eager is host-dispatch-bound).

add_executable(view_trace_bench view_trace_bench.cpp)
target_link_libraries(view_trace_bench PRIVATE ttprm_ttnn)

#  - view_add_trace_bench: ttprm::view_add (fused gather+add+scatter) vs ttnn reshape+add.
add_executable(view_add_trace_bench view_add_trace_bench.cpp)
target_link_libraries(view_add_trace_bench PRIVATE ttprm_ttnn)

#  - view_ternary_trace_bench: ttprm::fma / ttprm::lerp vs the native ttnn op chain; cross-checks
#    correctness (xPCC) and isolates the resident-broadcast read-once win (RWKV lerp shape).
add_executable(view_ternary_trace_bench view_ternary_trace_bench.cpp)
target_link_libraries(view_ternary_trace_bench PRIVATE ttprm_ttnn)

#  - view_norm_trace_bench: ttprm fused l2_norm / layer_norm vs the native ttnn reshape+norm+reshape
#    chain RWKV actually runs (Family-2 per-head, reshape-wrapped) + Family-1 wide decode LN.
add_executable(view_norm_trace_bench view_norm_trace_bench.cpp)
target_link_libraries(view_norm_trace_bench PRIVATE ttprm_ttnn)

#  - row_index_trace_bench: GET_ROWS / SET_ROWS achieved DRAM bandwidth vs peak (bottleneck check).
add_executable(row_index_trace_bench row_index_trace_bench.cpp)
target_link_libraries(row_index_trace_bench PRIVATE ttprm_ttnn)
