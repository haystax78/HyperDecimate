#!/usr/bin/env bash
# Run every suite and report honestly.
#
# Two traps this guards against, both of which reported false success earlier:
#   * Blender exits 0 even when a --python script raises, so an exit code alone is
#     not a pass signal. Every suite must also print its verdict line.
#   * Reading an exit code through $(...) gives the status of the substitution,
#     not of the command, so exit codes are captured separately here.
set -u
BLENDER="${BLENDER:-D:/blender_test/blender.exe}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ADDONS="$(dirname "$ROOT")"
OUT="$(mktemp -d)"
FAILED=0

report() {  # name rc logfile
  local name="$1" rc="$2" log="$3"
  local verdict
  verdict="$(tr -d '\000' < "$log" | grep -oE 'all checks passed|[0-9]+ check\(s\) failed|[0-9]+/[0-9]+ required probes passed|within budget|OVER BUDGET' | tail -1)"
  if [ -z "$verdict" ]; then
    verdict="NO VERDICT (script probably crashed)"
  fi
  case "$verdict" in
    *"all checks passed"*|*"required probes passed"*|*"within budget"*) ;;
    *) FAILED=1 ;;
  esac
  [ "$rc" -eq 0 ] || FAILED=1
  printf '%-34s rc=%-3s %s\n' "$name" "$rc" "$verdict"
}

# tools/run_suite.py binds the `hyper_decimate` import name to THIS checkout
# before running its target, so a sibling add-on of the same name cannot shadow
# it and have every suite below test that instead.
run_py() {  # name module
  ( cd "$ROOT" && python tools/run_suite.py "$2" ) > "$OUT/$1.log" 2>&1
  report "$1" "$?" "$OUT/$1.log"
}

run_blender() {  # name backend script [blendfile]
  local name="$1" backend="$2" script="$3" blend="${4:-}"
  if [ -n "$blend" ]; then
    ( cd "$ROOT" && "$BLENDER" --background --factory-startup --gpu-backend "$backend" "$blend" --python tools/run_suite.py -- "$script" ) > "$OUT/$name.log" 2>&1
  else
    ( cd "$ROOT" && "$BLENDER" --background --factory-startup --gpu-backend "$backend" --python tools/run_suite.py -- "$script" ) > "$OUT/$name.log" 2>&1
  fi
  report "$name" "$?" "$OUT/$name.log"
}

echo "logs in $OUT"
for B in opengl vulkan; do
  run_blender "probe [$B]"        "$B" tools/probe_gpu.py
  run_blender "gpu context [$B]"  "$B" tests/test_gpu_context.py
  run_blender "gpu prims [$B]"    "$B" tests/test_gpu_prims.py
  run_blender "gpu pipeline [$B]" "$B" tests/test_gpu_simplify.py
done
run_blender "blender io" opengl tests/test_blender_io.py
run_blender "addon (operator + UI)" vulkan tests/test_addon.py

# Not run here: tools/gui_smoke.py needs a real window, because background mode
# has no event loop and so cannot drive a modal operator, and no sidebar to draw
# the progress bar into. It is the only cover for either. Run it by hand:
#   HD_SMOKE_LOG=smoke.log blender --factory-startup --python tools/gui_smoke.py
# then check the log ends in PASS.

echo
if [ "$FAILED" -eq 0 ]; then
  echo "ALL SUITES PASSED"
else
  echo "SOME SUITES FAILED"
fi
exit "$FAILED"
