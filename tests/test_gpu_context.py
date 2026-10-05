"""Tests for gpu_backend/context.py. Must run inside Blender.

    blender --background --factory-startup --python tests/test_gpu_context.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hyper_decimate.gpu_backend import context as ctx  # noqa: E402

from hyper_decimate.tests.harness import check, finish, report  # noqa: E402


def test_probe():
    print("\nprobe", flush=True)
    info = ctx.probe()
    print(f"  {info['backend']} / {info['renderer']}", flush=True)
    check("capacity covers 25M elements", info["capacity"] >= 25_000_000,
          f"{info['capacity']:,}")
    check("reported max_images is recorded but not trusted",
          "reported_max_images" in info, f"{info['reported_max_images']:,}")


def test_roundtrip():
    print("\nhost round trip", flush=True)
    for fmt, dtype in (("R32UI", np.uint32), ("R32F", np.float32)):
        n = 1_000_003  # deliberately not a multiple of PACK_W
        src = (np.arange(n) % 65521).astype(dtype)
        arr = ctx.Array(f"rt_{fmt}", n, fmt=fmt, data=src)
        got = arr.download()
        check(f"{fmt} round trip exact", np.array_equal(got, src),
              f"{n:,} elements, {int((got != src).sum())} wrong")

    # The staged 16-bit split must be exact across the whole 32-bit range, which
    # is the whole reason it exists rather than a single float per value.
    edges = np.array([
        0, 1, 65535, 65536, 65537, 1 << 20, (1 << 24) - 1, 1 << 24,
        (1 << 24) + 1, 25_000_000, (1 << 31) - 1, 1 << 31, 0xFFFFFFFE,
        0xFFFFFFFF,
    ], dtype=np.uint32)
    arr = ctx.Array("rt_edges", edges.size, fmt="R32UI", data=edges)
    got = arr.download()
    bad = [(int(w), int(g)) for w, g in zip(edges, got) if int(w) != int(g)]
    check("R32UI exact at 32-bit boundary values", not bad,
          f"{bad}" if bad else f"{edges.size} values including 2**24 and 2**32-1")

    vec = np.stack([edges, edges[::-1], edges, edges], axis=1)
    arrv = ctx.Array("rt_vecui", edges.size, fmt="RGBA32UI")
    check("RGBA32UI upload is refused rather than silently wrong",
          _raises(lambda: arrv.upload(vec), NotImplementedError))


def test_vector_roundtrip():
    print("\nvector round trip", flush=True)
    n = 250_007
    src = np.random.default_rng(0).random((n, 4)).astype(np.float32)
    arr = ctx.Array("rt_vec", n, fmt="RGBA32F", data=src)
    got = arr.download()
    check("RGBA32F round trip exact", np.array_equal(got, src),
          f"max diff {float(np.abs(got - src).max()):.2e}")


def test_addressing():
    print("\npacked addressing matches between host and shader", flush=True)
    n = 300_000
    out = ctx.Array("out_arr", n, fmt="R32UI")
    out.clear(0)
    kernel = ctx.Kernel(
        "addr",
        """
        void main() {
          uint i = GID;
          if (i >= uint(N)) return;
          // Recompute the flat index from the texel this thread addresses. If
          // IDX2 and the host agree, this is the identity.
          ivec2 c = IDX2(i);
          uint back = uint(c.y) * uint(PACK_W) + uint(c.x);
          imageStore(out_arr, IDX2(i), uvec4(back));
        }
        """,
        [out],
        defines={"N": n},
    )
    kernel.run(n)
    got = out.download()
    want = np.arange(n, dtype=np.uint32)
    check("shader index equals host index", np.array_equal(got, want),
          f"{int((got != want).sum())} of {n:,} disagree")


def test_hash_matches_host():
    print("\nGLSL hash matches the NumPy hash", flush=True)
    from hyper_decimate.tests.fixtures import hash_u32

    n = 100_000
    out = ctx.Array("hash_out", n, fmt="R32UI")
    kernel = ctx.Kernel(
        "hashk",
        """
        void main() {
          uint i = GID;
          if (i >= uint(N)) return;
          imageStore(hash_out, IDX2(i), uvec4(hd_hash(i)));
        }
        """,
        [out],
        defines={"N": n},
    )
    kernel.run(n)
    got = out.download()
    want = hash_u32(np.arange(n))
    check("hash agrees element for element", np.array_equal(got, want),
          f"{int((got != want).sum())} of {n:,} differ")


def test_layered():
    print("\nlayered array read/write and per-layer download", flush=True)
    n = 200_003
    layers = 3
    arr = ctx.Array("lay", n, fmt="R32F", layers=layers)
    kernel = ctx.Kernel(
        "fill_layers",
        """
        void main() {
          uint i = GID;
          if (i >= uint(N)) return;
          for (int l = 0; l < LAYERS; l++)
            imageStore(lay, IDX3(i, l), vec4(float(i) + float(l) * 1000000.0));
        }
        """,
        [arr],
        defines={"N": n, "LAYERS": layers},
    )
    kernel.run(n)

    check("direct download of a layered array is refused",
          _raises(lambda: arr.download(), RuntimeError))

    for layer in range(layers):
        got = ctx.download_layer(arr, layer)
        want = np.arange(n, dtype=np.float32) + layer * 1_000_000.0
        check(f"layer {layer} flattened correctly", np.array_equal(got, want),
              f"{int((got != want).sum())} of {n:,} wrong")


def test_guards():
    print("\nguards", flush=True)
    check("reserved GLSL name is rejected",
          _raises(lambda: ctx.Array("flat", 16), ValueError))
    check("unknown format is rejected",
          _raises(lambda: ctx.Array("bad", 16, fmt="RGB8"), ValueError))
    check("signed integer format is rejected with a reason",
          _raises(lambda: ctx.Array("signed", 16, fmt="R32I"), ValueError))
    arrays = [ctx.Array(f"a{i}", 16) for i in range(ctx.IMAGE_BUDGET + 1)]
    check("over-budget kernel is rejected",
          _raises(lambda: ctx.Kernel("too_many", "void main() {}", arrays),
                  ValueError))
    check("oversized upload is rejected",
          _raises(lambda: ctx.Array("small", 4).upload(np.zeros(10_000_000, np.uint32)),
                  ValueError))


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception as other:
        print(f"      (raised {type(other).__name__} instead)", flush=True)
        return False
    return False


def test_integer_upload_paths():
    """Both integer upload routes must be exact, not just the one in use.

    The fast route reinterprets each value's bits through an R32F image, which is
    5.3x faster than splitting into 16-bit halves but only correct where the
    driver leaves NaN payloads alone -- and an index array read as float is full
    of NaNs, NO_INDEX being 0xFFFFFFFF. `bitcast_ok` probes for that, so the
    fallback is what some other device will actually run, and it is tested here
    whether or not this one needs it.
    """
    print("\ninteger upload, both routes", flush=True)
    check("this device round-trips float32 bit patterns", ctx.bitcast_ok(),
          "fast path in use" if ctx.bitcast_ok() else "using the 16-bit split")

    n = 1_000_003
    vals = (np.arange(n) % 65521).astype(np.uint32)
    edges = np.array([0, 1, 0xFFFF, 0x10000, (1 << 24) - 1, 1 << 24,
                      (1 << 31) - 1, 1 << 31, 0x7F800000, 0x7FC00000,
                      0xFFFFFFFE, 0xFFFFFFFF], dtype=np.uint32)

    saved = ctx._BITCAST_OK
    try:
        for label, flag in (("bitcast", True), ("16-bit split", False)):
            ctx._BITCAST_OK = flag
            arr = ctx.Array(f"up_{int(flag)}", n, fmt="R32UI", data=vals)
            check(f"{label}: {n:,} values exact",
                  np.array_equal(arr.download(), vals),
                  f"{int((arr.download() != vals).sum())} wrong")
            e = ctx.Array(f"upe_{int(flag)}", edges.size, fmt="R32UI", data=edges)
            got = e.download()
            check(f"{label}: NaN, infinity and sentinel bit patterns exact",
                  np.array_equal(got, edges),
                  f"{[(hex(int(a)), hex(int(b))) for a, b in zip(edges, got) if a != b]}")

            # A short upload must clear the elements it does not write, since the
            # pack kernel runs over the whole array.
            part = ctx.Array(f"upp_{int(flag)}", 4096, fmt="R32UI")
            part.fill(0xDEADBEEF)
            part.upload(vals[:100])
            back = part.download()
            check(f"{label}: short upload zeroes the tail",
                  np.array_equal(back[:100], vals[:100])
                  and not back[100:].any(),
                  f"{int(back[100:].sum())} left over")
    finally:
        ctx._BITCAST_OK = saved


def test_packing_width():
    """The width must widen for arrays past one texture, and only then.

    This is the 52M-triangle case: its 156M adjacency entries need 38,205 rows at
    the 4096 default, past a 32,768px device, and the failure used to arrive as
    "gpu.texture.new(...) failed with 'unknown error'" from inside GPUState.
    """
    print("\npacking width planning", flush=True)
    limit = ctx.max_texture_size()
    start = ctx.PACK_W
    try:
        check("default width is the minimum for a small mesh",
              ctx.width_for(1_000_000) == ctx.PACK_W_MIN,
              f"{ctx.width_for(1_000_000)}")

        fits = ctx.PACK_W_MIN * limit
        check("exactly one texture's worth still fits at the minimum",
              ctx.width_for(fits) == ctx.PACK_W_MIN, f"{fits:,} elements")
        check("one element more widens the packing",
              ctx.width_for(fits + 1) == 2 * ctx.PACK_W_MIN,
              f"{fits + 1:,} -> {ctx.width_for(fits + 1)}")

        corners = 3 * 52_162_560  # head_scan_huge_test.blend, triangulated
        width = ctx.width_for(corners)
        check("the 52M-triangle adjacency fits some width",
              ctx.rows_for(corners, width) <= limit,
              f"{corners:,} -> {width} x {ctx.rows_for(corners, width):,}")

        check("beyond the widest packing it reports unavailable rather than failing"
              " in GPUTexture",
              _raises(lambda: ctx.width_for(limit * limit + 1),
                      ctx.GPUUnavailable))

        # A real round trip at a widened packing, because the define, the host
        # arithmetic and the texture shape all have to move together.
        ctx.set_pack_width(2 * ctx.PACK_W_MIN)
        n = 1_000_003
        src = (np.arange(n) % 65521).astype(np.uint32)
        arr = ctx.Array("wide_rt", n, fmt="R32UI", data=src)
        check("round trip exact at a widened packing",
              np.array_equal(arr.download(), src), f"width {ctx.PACK_W}")

        # `arr` stays behind at the wide packing, so a kernel compiled after the
        # width is restored addresses it wrongly. Silently wrong indices are the
        # worst outcome available here, so the bind has to refuse.
        ctx.set_pack_width(ctx.PACK_W_MIN)
        narrow = ctx.Kernel(
            "stale_width",
            "void main() { if (GID == 0u) imageStore(wide_rt, IDX2(0u), uvec4(1)); }",
            [("wide_rt", "R32UI")],
        )
        check("a kernel refuses an Array packed at another width",
              _raises(lambda: narrow.run(1, bind={"wide_rt": arr}), RuntimeError))
    finally:
        ctx.set_pack_width(start)


def main():
    print("Hyper Decimate — GPU context tests", flush=True)
    test_probe()
    test_integer_upload_paths()
    test_packing_width()
    test_roundtrip()
    test_vector_roundtrip()
    test_addressing()
    test_hash_matches_host()
    test_layered()
    test_guards()

    return report()


if __name__ == "__main__":
    finish(main())
