// ttprm_test.cpp -- square-1 test skeleton.
// -----------------------------------------------------------------------------
// Establishes the PASS/FAIL convention used across the project: print "RESULT: OK" (or
// "RESULT: FAIL") on the last line and return non-zero on failure, so a `timeout stdbuf -oL`
// run reports a verdict even if device teardown segfaults.
//
// Methodology to follow as src/ fills in (see LEARNINGS.md / CLAUDE.md):
//   1. MOCK passthrough FIRST -- prove the gather/scatter address map round-trips bit-exact
//      (gather -> scatter, no compute) before adding any math.
//   2. Oracle = ttnn::reshape(+ math) vs the ttprm kernel: bit-exact for int fills, PCC vs CPU
//      for float. Smallest shapes first.
//   3. Test multi-core partitioning in the oracle, not just single-core.
//   4. Measure perf under tt-metal TRACE (eager is host-dispatch-bound).

#include <cstdint>
#include <cstdio>

namespace {
int g_pass = 0, g_fail = 0;

// Record a boolean check. `detail` is optional context shown on the line.
void check(const char* name, bool ok, const char* detail = "") {
    std::printf("  [%s] %s%s%s\n", ok ? "PASS" : "FAIL", name,
                detail[0] ? " -- " : "", detail);
    if (ok) ++g_pass; else ++g_fail;
}
}  // namespace

int main() {
    std::printf(">>> ttprm_test (square-1 skeleton)\n");

    // TODO(square-1): replace with real cases as the library lands in src/, e.g.
    //   check("MOCK gather->scatter round-trip", mock_roundtrip_bit_exact());
    //   check("fused op vs CPU oracle (PCC)",    pcc(fused, oracle) > 0.99f);
    //   check("multi-core partition vs oracle",  multicore_matches_oracle());
    check("skeleton harness wired", true);

    std::printf("  %d passed, %d failed\n", g_pass, g_fail);
    const bool ok = (g_fail == 0);
    std::printf("RESULT: %s\n", ok ? "OK" : "FAIL");
    return ok ? 0 : 1;
}
