"""Measure every stage of a GPU run except the kernels, on a real mesh.

The Tier 2 simplifier takes 22 minutes on the head scan, which makes it useless
for timing the stages around it. This script substitutes a synthetic 10%
decimation result of the right shape and sizes, so ingest, GPU upload, GPU
readback and egress can all be measured on production data in seconds.

The synthetic result is geometrically meaningless and that is fine: every stage
measured here is index shuffling and bulk transfer whose cost depends on array
sizes, not on which vertices were chosen.

    blender --background --factory-startup --gpu-backend opengl \
        tests/head_scan_test.blend --python tools/bench_stages.py -- 0.10
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hyper_decimate.core import egress, ingest  # noqa: E402
from hyper_decimate.gpu_backend import context as ctx  # noqa: E402

ROWS = []


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<46} {dt:7.3f}s", flush=True)
    ROWS.append((label, dt))
    return out


def fake_result(positions, tris, ratio):
    """A decimation result with realistic shapes and a valid remap contract."""
    step = max(1, int(round(1.0 / ratio)))
    keep_faces = np.arange(0, tris.shape[0], step, dtype=np.int64)
    sub = tris[keep_faces]
    used = np.unique(sub)

    old_to_new = np.zeros(positions.shape[0], dtype=np.int32)
    old_to_new[used] = np.arange(used.size, dtype=np.int32)
    survived = np.zeros(positions.shape[0], dtype=bool)
    survived[used] = True
    # Non-survivors have to point at *some* survivor for the contract to hold.
    remap = np.zeros(positions.shape[0], dtype=np.int32)
    remap[used] = old_to_new[used]

    return (positions[used], old_to_new[sub].astype(np.int32), remap, survived,
            keep_faces.astype(np.int32))


def main():
    import bpy

    argv = sys.argv
    ratio = float(argv[argv.index("--") + 1]) if "--" in argv else 0.10

    obj = max((o for o in bpy.data.objects if o.type == "MESH"),
              key=lambda o: len(o.data.vertices))
    mesh = obj.data
    print(f"\nsubject: {obj.name!r} verts={len(mesh.vertices):,} "
          f"polys={len(mesh.polygons):,} loops={len(mesh.loops):,}", flush=True)

    print("\n--- ingest ---", flush=True)
    positions, tris, tri_loops, method = timed(
        "read_mesh", lambda: ingest.read_mesh(mesh)
    )[:4]
    print(f"    path={method}  {positions.shape[0]:,} verts, "
          f"{tris.shape[0]:,} triangles", flush=True)

    print("\n--- GPU upload, what the kernels would need ---", flush=True)
    info = ctx.probe()
    print(f"    {info['backend']} / {info['renderer']}", flush=True)
    pos_f = np.ascontiguousarray(positions, dtype=np.float32)
    v0 = np.ascontiguousarray(tris[:, 0], dtype=np.uint32)
    v1 = np.ascontiguousarray(tris[:, 1], dtype=np.uint32)
    v2 = np.ascontiguousarray(tris[:, 2], dtype=np.uint32)

    padded = np.zeros((pos_f.shape[0], 4), dtype=np.float32)
    padded[:, :3] = pos_f
    gpos = timed(
        "upload positions (RGBA32F, float fast path)",
        lambda: ctx.Array("bs_pos", pos_f.shape[0], fmt="RGBA32F", data=padded),
    )
    gv = timed(
        "upload 3 corner arrays (staged integer path)",
        lambda: [ctx.Array(f"bs_v{i}", x.size, fmt="R32UI", data=x)
                 for i, x in enumerate((v0, v1, v2))],
    )
    mb = (padded.nbytes + 3 * v0.nbytes * 4) / 1e6
    print(f"    ~{mb:.0f} MB of texture payload", flush=True)

    print("\n--- GPU readback of a 10% result ---", flush=True)
    nk = max(1, int(tris.shape[0] * ratio))
    small_u = ctx.Array("bs_small_u", nk, fmt="R32UI")
    small_f = ctx.Array("bs_small_f", nk, fmt="RGBA32F")
    timed("readback 3 corner arrays", lambda: [
        small_u.download(nk) for _ in range(3)
    ])
    timed("readback positions", lambda: small_f.download(nk))

    print("\n--- egress of a synthetic 10% result ---", flush=True)
    out_p, out_t, remap, survived, face_origin = fake_result(
        positions, tris.astype(np.int64), ratio
    )
    print(f"    {out_p.shape[0]:,} verts, {out_t.shape[0]:,} triangles",
          flush=True)
    out = timed("build_mesh", lambda: egress.build_mesh("bs_out", out_p, out_t))
    timed("transfer_point_attributes",
          lambda: egress.transfer_point_attributes(mesh, out, remap, survived))
    timed("transfer_uv_layers",
          lambda: egress.transfer_uv_layers(mesh, out, tri_loops, face_origin,
                                            remap, survived))

    print("\n=== everything except the kernels ===", flush=True)
    total = sum(dt for _, dt in ROWS)
    for label, dt in ROWS:
        print(f"  {label:<46} {dt:7.3f}s")
    print(f"  {'TOTAL':<46} {total:7.3f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
