"""Milestone 2 gate: measure Blender mesh I/O at production scale.

PLAN.md 5.4 budgets 0.5-1.5 s to read a 24.6M-vertex mesh and 1.0-2.5 s to build
the decimated result. PLAN.md milestone M2 says: if read plus write alone exceeds
about 5 seconds, stop and fix that before writing a single GPU kernel.

This script answers that. It builds a large grid directly from NumPy, so the
build is itself the egress measurement at full scale, then reads it back, then
builds a 10% result, which is the realistic egress size.

    blender --background --factory-startup --python tools/bench_blender.py -- 3200

The trailing number is the grid side; 3200 gives about 10.2M vertices and 20.5M
triangles, close to a 24.6M-vertex mesh.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hyper_decimate.core import egress, ingest  # noqa: E402

ROWS = []


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    print(f"  {label:<44} {dt:7.2f}s", flush=True)
    ROWS.append((label, dt))
    return out, dt


def make_grid(n, quads=False):
    """Flat n x n grid as arrays. Quads or triangles."""
    xs = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pos = np.empty((n * n, 3), dtype=np.float32)
    pos[:, 0] = gx.ravel()
    pos[:, 1] = gy.ravel()
    pos[:, 2] = 0.0

    idx = np.arange(n * n, dtype=np.int32).reshape(n, n)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    if quads:
        return pos, np.stack([a, b, c, d], axis=1)
    return pos, np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]
    )


def build_quad_mesh(name, positions, quads):
    """Egress variant for quads, used only to make a quad test subject."""
    import bpy

    nv, nf = positions.shape[0], quads.shape[0]
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(nv)
    mesh.loops.add(nf * 4)
    mesh.polygons.add(nf)
    mesh.vertices.foreach_set("co", positions.reshape(-1))
    mesh.loops.foreach_set("vertex_index", quads.reshape(-1))
    mesh.polygons.foreach_set("loop_start", np.arange(nf, dtype=np.int32) * 4)
    mesh.update(calc_edges=True)
    return mesh


def main():
    import bpy

    argv = sys.argv
    n = int(argv[argv.index("--") + 1]) if "--" in argv else 3200

    pos, tris = make_grid(n)
    print(
        f"\nSubject: flat grid {n}x{n} = {pos.shape[0]:,} verts, "
        f"{tris.shape[0]:,} triangles",
        flush=True,
    )

    print("\n--- egress, full size ---", flush=True)
    mesh, t_build_edges = timed(
        "build_mesh, calc_edges=True",
        lambda: egress.build_mesh("bench_tris", pos, tris, calc_edges=True),
    )
    _, t_build_noedges = timed(
        "build_mesh, calc_edges=False",
        lambda: egress.build_mesh("bench_noedge", pos, tris, calc_edges=False),
    )

    print("\n--- ingest ---", flush=True)
    method = ingest.triangulation_method(mesh)
    print(f"  detected path: {method}", flush=True)
    timed("read_positions", lambda: ingest.read_positions(mesh))
    (rt, rl, _), t_read_tris = timed(
        "read_triangles (all-triangle fast path)",
        lambda: ingest.read_triangles(mesh, method),
    )
    ok = np.array_equal(rt, tris)
    print(f"  round trip exact: {ok}", flush=True)

    print("\n--- ingest, quad mesh of the same vertex count ---", flush=True)
    qpos, quads = make_grid(n, quads=True)
    qmesh, _ = timed(
        "build quad subject", lambda: build_quad_mesh("bench_quads", qpos, quads)
    )
    qmethod = ingest.triangulation_method(qmesh)
    print(f"  detected path: {qmethod}", flush=True)
    timed(
        "read_triangles (quad split fast path)",
        lambda: ingest.read_triangles(qmesh, qmethod),
    )
    timed(
        "calc_loop_triangles for comparison, same mesh",
        lambda: qmesh.calc_loop_triangles(),
    )

    print("\n--- egress, 10% result ---", flush=True)
    keep = tris.shape[0] // 10
    sub = tris[:keep]
    used = np.unique(sub)
    old_to_new = np.full(pos.shape[0], 0, dtype=np.int32)
    old_to_new[used] = np.arange(used.size, dtype=np.int32)
    _, t_build_small = timed(
        f"build_mesh at 10% ({keep:,} tris)",
        lambda: egress.build_mesh(
            "bench_small", pos[used], old_to_new[sub], calc_edges=True
        ),
    )

    print("\n=== M2 gate ===", flush=True)
    read_total = sum(
        dt for label, dt in ROWS
        if label.startswith("read_positions") or "all-triangle fast path" in label
    )
    write_total = t_build_small
    print(f"  read (positions + triangles):          {read_total:7.2f}s")
    print(f"  write (10% result, with edges):        {write_total:7.2f}s")
    print(f"  read + write round trip:               {read_total + write_total:7.2f}s")
    print(f"  full-size build, for reference:        {t_build_edges:7.2f}s")
    print(f"  calc_edges costs:                      "
          f"{t_build_edges - t_build_noedges:7.2f}s")
    budget = 5.0
    passed = (read_total + write_total) <= budget
    print("  " + ("within the 5s M2 gate, proceed to M3"
                  if passed else
                  "OVER the 5s M2 gate, fix I/O before writing kernels"))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
