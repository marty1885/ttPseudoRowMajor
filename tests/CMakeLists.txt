# Square-1 tests.
#  - view_access_test : PURE HOST (no device, no TT link) -- proves the face-granular
#                       layout algebra bit-exact vs a brute-force oracle.
#  - view_realize_test: on-device -- the generic copy-list gather->scatter (view_realize)
#                       round-trips complex view chains bit-exact, single- and multi-core.
# TTPRM_ROOT is the absolute path to the library (host headers in ttprm/, kernels in
# ttprm/kernel/), baked in so kernel paths + include dirs resolve regardless of cwd.

set(TTPRM_ROOT "${CMAKE_SOURCE_DIR}/ttprm")

# --- pure-host layout-algebra derisk (builds & runs without a board) ---
add_executable(view_access_test view_access_test.cpp)
target_include_directories(view_access_test PRIVATE ${TTPRM_ROOT})
add_test(NAME view_access_test COMMAND view_access_test)

# --- pure-host PREFILL grouped-row gather derisk (step 1: the mod/div odometer math) ---
add_executable(grouped_gather_test grouped_gather_test.cpp)
target_include_directories(grouped_gather_test PRIVATE ${TTPRM_ROOT})
add_test(NAME grouped_gather_test COMMAND grouped_gather_test)

# NOTE: the old raw-Metalium view_realize_test (DRAM copy-list table) was RETIRED when the
# device path moved to the on-device affine generator. Device correctness is now proven by
# view_ttnn_test below, against actually-running TTNN ops.

# --- ttnn-facing view_realize: cached device_operation, oracle = chained TTNN ops ---
add_executable(view_ttnn_test view_ttnn_test.cpp)
target_link_libraries(view_ttnn_test PRIVATE ttprm_ttnn)
add_test(NAME view_ttnn_test COMMAND view_ttnn_test)

# --- indexed row gather/scatter (GET_ROWS / SET_ROWS): the KV / state-cache primitive ---
add_executable(row_index_test row_index_test.cpp)
target_link_libraries(row_index_test PRIVATE ttprm_ttnn)
add_test(NAME row_index_test COMMAND row_index_test)

# --- Phase-0 derisk: RWKV token-shift lerp sub-tile group broadcast (decode/prefill) ---
# Decision gate for the llama.cpp swap -- does ttprm::lerp fuse the [n_embd,n_tokens] x [.,1,1,G]
# token-shift when each gate group is a sub-tile (<32) row span? Reports FUSE vs REJECT vs GAP.
add_executable(view_lerp_rwkv_test view_lerp_rwkv_test.cpp)
target_link_libraries(view_lerp_rwkv_test PRIVATE ttprm_ttnn)
add_test(NAME view_lerp_rwkv_test COMMAND view_lerp_rwkv_test)

add_executable(view_lerp_rwkv_4d_test view_lerp_rwkv_4d_test.cpp)
target_link_libraries(view_lerp_rwkv_4d_test PRIVATE ttprm_ttnn)
add_test(NAME view_lerp_rwkv_4d_test COMMAND view_lerp_rwkv_4d_test)
