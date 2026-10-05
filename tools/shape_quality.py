"""Triangle shape and shading noise, ours against Blender's Decimate.

    blender --background --factory-startup tests/head_scan_test.blend \
        --python tools/shape_quality.py -- 2000000

`tools/quality.py` measures how far the decimated surface sits from the original,
and by that measure this backend does well. It says nothing about the *shape* of
the triangles that approximate it, and that is a separate axis you can fail while
passing the first one: a long thin sliver can lie exactly on the surface and still
wreck the shading, because a vertex normal is the area-weighted mean of its faces'
normals and a sliver's normal is numerically unstable.

That is the failure this measures. On a smooth region every neighbour of a vertex
lies in nearly the same plane, so the quadric is rank-deficient and every collapse
direction scores nearly zero; the choice is then made by floating-point noise and
nothing resists anisotropy. The result is fans of slivers, which is what a close
render of a decimated scan shows against Blender's output of the same count.

Reported per mesh:

  shape q      4*sqrt(3)*A / (a^2+b^2+c^2). 1.0 equilateral, 0 degenerate.
               The mean is insensitive; the low percentiles are the story.
  min angle    smallest angle in each triangle, in degrees.
  dihedral     angle between the normals of each pair of adjacent faces. This is
               the shading noise directly: a smooth surface should have a small
               dihedral everywhere, and the high percentiles are what the eye
               picks up as speckle.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hyper_decimate.core import dispatch, egress, ingest  # noqa: E402
from hyper_decimate.core.options import Options  # noqa: E402


def say(*a):
    print(*a, flush=True)


# ------------------------------------------------------------------ measures

def triangle_metrics(positions, triangles):
    """(shape quality, min angle in degrees) per triangle."""
    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    a = p[t[:, 1]] - p[t[:, 0]]
    b = p[t[:, 2]] - p[t[:, 1]]
    c = p[t[:, 0]] - p[t[:, 2]]

    cross = np.cross(a, -c)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    l2 = (np.einsum("ij,ij->i", a, a) + np.einsum("ij,ij->i", b, b)
          + np.einsum("ij,ij->i", c, c))
    quality = np.where(l2 > 0.0, 4.0 * np.sqrt(3.0) * area / np.maximum(l2, 1e-300), 0.0)

    # The smallest angle sits opposite the shortest edge; computing all three and
    # taking the minimum is simpler than reasoning about which that is.
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)
    ang = []
    for (u, v, w) in ((la, lb, lc), (lb, lc, la), (lc, la, lb)):
        denom = np.maximum(2.0 * u * v, 1e-300)
        cosang = np.clip((u * u + v * v - w * w) / denom, -1.0, 1.0)
        ang.append(np.degrees(np.arccos(cosang)))
    return quality, np.min(np.stack(ang, axis=1), axis=1)


def dihedral_angles(positions, triangles):
    """Angle in degrees between the normals of every pair of adjacent faces."""
    p = np.asarray(positions, dtype=np.float64)
    t = np.asarray(triangles, dtype=np.int64)
    n = np.cross(p[t[:, 1]] - p[t[:, 0]], p[t[:, 2]] - p[t[:, 0]])
    ln = np.linalg.norm(n, axis=1)
    good = ln > 0.0
    n[good] /= ln[good, None]

    lo = np.minimum(np.stack([t[:, 0], t[:, 1], t[:, 2]], axis=1),
                    np.stack([t[:, 1], t[:, 2], t[:, 0]], axis=1))
    hi = np.maximum(np.stack([t[:, 0], t[:, 1], t[:, 2]], axis=1),
                    np.stack([t[:, 1], t[:, 2], t[:, 0]], axis=1))
    key = lo.reshape(-1).astype(np.int64) * (1 << 32) + hi.reshape(-1)
    face = np.repeat(np.arange(t.shape[0], dtype=np.int64), 3)

    order = np.argsort(key, kind="stable")
    key, face = key[order], face[order]
    # An interior edge appears exactly twice; take consecutive equal pairs.
    same = key[1:] == key[:-1]
    f0, f1 = face[:-1][same], face[1:][same]
    both = good[f0] & good[f1]
    f0, f1 = f0[both], f1[both]
    if f0.size == 0:
        return np.zeros(0)
    cosang = np.clip(np.einsum("ij,ij->i", n[f0], n[f1]), -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def describe(label, positions, triangles):
    q, ang = triangle_metrics(positions, triangles)
    dih = dihedral_angles(positions, triangles)
    say(f"\n  {label}  ({triangles.shape[0]:,} triangles)")
    say(f"    shape q     mean {q.mean():.3f}   p50 {np.percentile(q, 50):.3f}"
        f"   p5 {np.percentile(q, 5):.3f}   p1 {np.percentile(q, 1):.3f}")
    say(f"                q<0.3 {100.0 * (q < 0.3).mean():5.2f}%"
        f"   q<0.1 {100.0 * (q < 0.1).mean():5.2f}%"
        f"   q<0.01 {100.0 * (q < 0.01).mean():5.2f}%")
    say(f"    min angle   mean {ang.mean():5.1f}d  p5 {np.percentile(ang, 5):5.1f}d"
        f"  p1 {np.percentile(ang, 1):5.1f}d"
        f"   <10d {100.0 * (ang < 10.0).mean():5.2f}%"
        f"   <5d {100.0 * (ang < 5.0).mean():5.2f}%")
    if dih.size:
        say(f"    dihedral    mean {dih.mean():5.2f}d  p95 {np.percentile(dih, 95):5.2f}d"
            f"  p99 {np.percentile(dih, 99):5.2f}d"
            f"  >30d {100.0 * (dih > 30.0).mean():5.3f}%"
            f"  >60d {100.0 * (dih > 60.0).mean():5.3f}%")
    return {"q": q, "ang": ang, "dih": dih}


# ------------------------------------------------------------------- drivers

def blender_decimate(positions, triangles, target):
    """Blender's own Decimate modifier, as the reference to beat.

    Built with `egress.build_mesh`, not `from_pydata`: the latter wants lists of
    tuples and at 13M triangles that is a Python loop long enough to dominate the
    whole measurement.
    """
    import bpy

    mesh = egress.build_mesh("hd_ref_src", positions, triangles)
    obj = bpy.data.objects.new("hd_ref_obj", mesh)
    bpy.context.collection.objects.link(obj)

    mod = obj.modifiers.new("hd_ref", 'DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = min(1.0, target / max(triangles.shape[0], 1))
    dg = bpy.context.evaluated_depsgraph_get()
    out = obj.evaluated_get(dg).to_mesh()
    pos = ingest.read_positions(out).astype(np.float64)
    tris, _, _ = ingest.read_triangles(out)
    pos, tris = pos.copy(), tris.astype(np.int64).copy()
    obj.evaluated_get(dg).to_mesh_clear()
    bpy.data.objects.remove(obj, do_unlink=True)
    bpy.data.meshes.remove(mesh, do_unlink=True)
    return pos, tris


def main():
    argv = sys.argv
    target = int(argv[argv.index("--") + 1]) if "--" in argv else 2_000_000

    import bpy
    obj = max((o for o in bpy.data.objects if o.type == "MESH"),
              key=lambda o: len(o.data.vertices))
    positions, tris, _, method = ingest.read_mesh(obj.data)
    say(f"source: {obj.name!r}  {positions.shape[0]:,} verts, "
        f"{tris.shape[0]:,} triangles ({method})")
    say(f"target: {target:,} triangles")

    t0 = time.perf_counter()
    result, info = dispatch.run(positions, tris, target, Options())
    say(f"\nours: {info['tier']} in {time.perf_counter() - t0:.1f}s, "
        f"{result.triangles.shape[0]:,} triangles")
    describe("ours", result.positions, result.triangles)

    if "--skip-blender" not in argv:
        t0 = time.perf_counter()
        b_pos, b_tris = blender_decimate(positions, tris, target)
        say(f"\nblender decimate in {time.perf_counter() - t0:.1f}s, "
            f"{b_tris.shape[0]:,} triangles")
        describe("blender", b_pos, b_tris)
    return 0


if __name__ == "__main__":
    import os
    code = 0
    try:
        code = main()
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    finally:
        dispatch.release()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(code)
