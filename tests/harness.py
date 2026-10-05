"""The shared scaffolding for the test scripts. No test logic lives here.

Every suite runs as `blender --background --factory-startup --python tests/<x>.py`
and reports by printing, because there is no pytest inside Blender and adding one
would mean shipping a dependency to run three files. That is fine, but it did
mean five copies of the same `check`, the same result block and the same exit
dance, including the nine-line comment explaining the exit dance, which is the
sort of thing that gets fixed in four files out of five.
"""

from __future__ import annotations

import os
import sys

FAILURES = []


def check(name, cond, detail=""):
    """Record one assertion. Printed as it happens so a hang shows its place."""
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}"
          + (f" — {detail}" if detail else ""), flush=True)
    if not cond:
        FAILURES.append(name)


def report():
    """Print the summary and return the process exit code."""
    print("\n=== Result ===", flush=True)
    if FAILURES:
        print(f"  {len(FAILURES)} check(s) failed:", flush=True)
        for name in FAILURES:
            print(f"    - {name}", flush=True)
        return 1
    print("  all checks passed", flush=True)
    return 0


def finish(code):
    """Leave the process immediately, keeping `code` for CI.

    Do not let SystemExit escape a Blender `--python` script after GPU work.
    Blender then tears down along a path that frees the GPU context out of order
    and the process dies with a segmentation fault *after* every check has
    already passed, which reads exactly like a real failure and is not one. A gc
    census at that point shows no live GPU objects, so this is Blender's shutdown
    order rather than a reference the tests are holding.

    Everything is finished and flushed by the time this is called, so there is
    nothing to lose by leaving at once.
    """
    try:
        from hyper_decimate.gpu_backend import context as ctx
        ctx.release_all()
    except Exception:  # noqa: BLE001 - a suite that never touched the GPU
        pass
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
