"""Tests for the GPU scan and compaction primitives. Must run inside Blender.

    blender --background --factory-startup --python tests/test_gpu_prims.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hyper_decimate.gpu_backend import context as ctx  # noqa: E402
from hyper_decimate.gpu_backend import prims as pr  # noqa: E402

from hyper_decimate.tests.harness import check, finish, report  # noqa: E402


def test_scan(p):
    print("\nexclusive prefix sum", flush=True)
    rng = np.random.default_rng(7)
    # Sizes chosen to hit every recursion depth and the awkward boundaries:
    # under one block, exactly one block, just over, two and three levels deep.
    for n in (1, 2, 255, 256, 257, 512, 4095, 4096, 65_536, 300_000, 5_000_000):
        vals = rng.integers(0, 16, size=n, endpoint=False).astype(np.uint32)
        src = ctx.Array(f"sc_src_{n}", n, fmt="R32UI", data=vals)
        dst = ctx.Array(f"sc_dst_{n}", n, fmt="R32UI")
        dst.clear(0)
        p.scan_exclusive(src, dst, n)
        got = dst.download(n)
        want = pr.exclusive_scan_reference(vals)
        ok = np.array_equal(got, want)
        detail = f"n={n:,}"
        if not ok:
            first = int(np.flatnonzero(got != want)[0])
            detail += (f", first mismatch at {first}: got {got[first]} "
                       f"want {want[first]}")
        check(f"scan exact at n={n:,}", ok, detail)

        total = p.scan_total(src, dst, n)
        check(f"total exact at n={n:,}", total == int(vals.sum()),
              f"got {total:,} want {int(vals.sum()):,}")


def test_scan_wide_values(p):
    print("\nscan with values that stress 32-bit accumulation", flush=True)
    n = 100_000
    vals = np.full(n, 40_000, dtype=np.uint32)  # sums to 4.0e9, near 2**32
    src = ctx.Array("wide_src", n, fmt="R32UI", data=vals)
    dst = ctx.Array("wide_dst", n, fmt="R32UI")
    p.scan_exclusive(src, dst, n)
    got = dst.download(n)
    want = (np.arange(n, dtype=np.uint64) * 40_000).astype(np.uint32)
    check("scan matches modulo 2**32", np.array_equal(got, want),
          f"total would be {n * 40_000:,}, wraps at 2**32")


def test_compact_one(p):
    print("\nstream compaction, single array", flush=True)
    rng = np.random.default_rng(11)
    for n in (1000, 262_144, 2_000_000):
        vals = (np.arange(n) * 7 + 1).astype(np.uint32)
        flags = (rng.random(n) < 0.35).astype(np.uint32)
        src = ctx.Array(f"cp_src_{n}", n, fmt="R32UI", data=vals)
        fl = ctx.Array(f"cp_fl_{n}", n, fmt="R32UI", data=flags)
        dst = ctx.Array(f"cp_dst_{n}", n, fmt="R32UI")
        dst.clear(0)
        kept = p.compact_one(src, dst, fl, n)
        want = vals[flags == 1]
        got = dst.download(kept)
        check(f"kept count correct at n={n:,}", kept == want.size,
              f"got {kept:,} want {want.size:,}")
        check(f"compacted values and order correct at n={n:,}",
              np.array_equal(got, want),
              f"{int((got != want).sum()) if got.size == want.size else 'size differs'}")


def test_compact_three(p):
    print("\nstream compaction, three parallel arrays", flush=True)
    n = 500_000
    rng = np.random.default_rng(13)
    a = (np.arange(n) % 60_000).astype(np.uint32)
    b = ((np.arange(n) * 3) % 60_000).astype(np.uint32)
    c = ((np.arange(n) * 5) % 60_000).astype(np.uint32)
    flags = (rng.random(n) < 0.5).astype(np.uint32)

    srcs = [ctx.Array(f"c3_s{i}", n, fmt="R32UI", data=x)
            for i, x in enumerate((a, b, c))]
    dsts = [ctx.Array(f"c3_d{i}", n, fmt="R32UI") for i in range(3)]
    for d in dsts:
        d.clear(0)
    fl = ctx.Array("c3_fl", n, fmt="R32UI", data=flags)

    kept = p.compact_three(srcs, dsts, fl, n)
    keepmask = flags == 1
    check("kept count correct", kept == int(keepmask.sum()),
          f"got {kept:,} want {int(keepmask.sum()):,}")
    for i, x in enumerate((a, b, c)):
        got = dsts[i].download(kept)
        want = x[keepmask]
        check(f"component {i} compacted correctly", np.array_equal(got, want))


def test_scan_timing(p):
    print("\ntiming at production scale", flush=True)
    n = 25_000_000
    vals = np.ones(n, dtype=np.uint32)
    src = ctx.Array("big_src", n, fmt="R32UI", data=vals)
    dst = ctx.Array("big_dst", n, fmt="R32UI")

    # Warm up at the SAME size. A smaller warm-up is worse than none: the
    # recursion only allocates its scratch textures when it first reaches each
    # depth, so warming up at 1024 exercises only the serial path and leaves
    # three texture allocations inside the timed region. That mistake made this
    # scan look 22x slower on Vulkan than on OpenGL, when steady-state compute is
    # actually comparable on the two backends.
    p.scan_exclusive(src, dst, n)
    p.scan_total(src, dst, n)

    best = min(_time_scan(p, src, dst, n) for _ in range(3))
    total = p.scan_total(src, dst, n)
    print(f"  scan of {n:,} elements, warm, best of 3: {best * 1000:.1f} ms",
          flush=True)
    check("scan of 25M is correct", total == n, f"total {total:,} want {n:,}")
    check("scan of 25M is under 100 ms", best < 0.100, f"{best * 1000:.1f} ms")

    # And report the cold cost separately, because the first call in a real
    # session pays it once and it is worth knowing how much it is.
    fresh = pr.Prims()
    t0 = time.perf_counter()
    fresh.scan_exclusive(src, dst, n)
    fresh.scan_total(src, dst, n)
    print(f"  same scan cold, including scratch allocation: "
          f"{(time.perf_counter() - t0) * 1000:.1f} ms", flush=True)


def _time_scan(p, src, dst, n):
    t0 = time.perf_counter()
    p.scan_exclusive(src, dst, n)
    p.scan_total(src, dst, n)
    return time.perf_counter() - t0


def main():
    print("Hyper Decimate — GPU primitives tests", flush=True)
    info = ctx.probe()
    print(f"  {info['backend']} / {info['renderer']}", flush=True)
    p = pr.Prims()

    test_scan(p)
    test_scan_wide_values(p)
    test_compact_one(p)
    test_compact_three(p)
    test_scan_timing(p)

    return report()


if __name__ == "__main__":
    finish(main())
