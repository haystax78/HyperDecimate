"""Measure decimation quality against a reference. PLAN.md milestone M4, 4.3a.

    blender --background --factory-startup --gpu-backend vulkan \
        --python tools/quality.py -- [fixture...] [--ratios 0.10,0.05,0.01]

With no arguments it runs every built-in fixture at 10%, 5% and 1%. Pass a .blend
on Blender's own command line to measure the largest mesh in it instead.

**The reference is Blender's own Decimate modifier, not meshoptimizer.** The plan
named meshoptimizer as the yardstick, and it is still the better one: it is the
state of the art for this algorithm and its error at a given triangle count is the
number worth beating. But it has no wheel on PyPI and this machine has no MSVC
toolchain, so the extension cannot be built here. Blender's Decimate is the honest
substitute: it is also quadric collapse, it is what users would otherwise reach
for, and "is this better than what Blender already does" is the question they
actually care about. Treat the numbers as relative to Blender, not absolute.
meshoptimizer still informed the algorithm; see PLAN.md 4.3a.

**The fixtures are generated here, not loaded.** They used to be whatever mesh
happened to be in a .blend, which made every number unreproducible and un-chased.
Three of them, because one is not enough to tune against:

  sphere     uniform triangles, convex everywhere. The easy case.
  stretched  the same sphere scaled anisotropically, so triangle area varies
             about tenfold. A uniform mesh cannot tell a weight-normalised cost
             from an unnormalised one; this one can.
  torus      genus 1, curvature of both signs, high-frequency ripples. The hard
             case, and consistently the worst result. It was held out while the
             constants were tuned, which is how a stricter fold limit was caught
             looking good on spheres while failing to reach the target at all on
             a saddle.

Distance is point-to-surface in both directions, using `mathutils.bvhtree`, not
point-to-point. A point-to-point measure flatters a decimator that keeps original
vertices, which this one does by construction, so it would be measuring the wrong
thing. Both directions matter and they say different things: source to result
catches detail that was thrown away, result to source catches geometry that was
invented, which is what a fold or a spike looks like.

**The triangle count is reported alongside the error, and it is not decoration.**
A variant that cannot reach the requested count keeps more triangles and so posts
a lower error for free. Any row whose `x/tgt` is above about 1.02 is not
comparable with the others and its error should be ignored.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hyper_decimate.core import dispatch, egress, ingest  # noqa: E402
from hyper_decimate.core.options import Options  # noqa: E402

# Points sampled per direction. Exhaustive comparison needs a nearest-surface
# query per vertex in Python, which is far too slow at these counts, so this is a
# sampled estimate. Fixed seed, so runs are comparable.
SAMPLES = 20_000
SEED = 7
DEFAULT_RATIOS = (0.10, 0.05, 0.01)

# PLAN.md M4's target, as a ratio of Blender's mean surface error.
TARGET = 1.15

# One pipeline ships, so the table is one row: the reference is Blender's own
# Decimate and the gate is applied to ours.
OURS_LABEL = "Hyper"
DEFAULT_LABEL = OURS_LABEL
REFERENCE_LABEL = "Blender Decimate"
VARIANTS = [(OURS_LABEL, "ours"), (REFERENCE_LABEL, None)]


# ------------------------------------------------------------------- fixtures

def _icosphere(level):
    from hyper_decimate.tests.fixtures import sphere
    return sphere(level)


def noisy_sphere(level=6):
    """82k triangles with three frequencies of radial detail."""
    pos, tris = _icosphere(level)
    pos = pos.astype(np.float64)
    r = np.linalg.norm(pos, axis=1, keepdims=True)
    unit = pos / np.where(r > 0, r, 1.0)
    d = (0.10 * np.sin(3.0 * pos[:, 0]) * np.cos(3.1 * pos[:, 1])
         + 0.04 * np.sin(11.0 * pos[:, 1]) * np.cos(9.0 * pos[:, 2])
         + 0.015 * np.sin(29.0 * pos[:, 2]) * np.cos(31.0 * pos[:, 0]))
    return (unit * (r + d[:, None])).astype(np.float32), tris.astype(np.int32)


def stretched(level=6):
    """The same sphere scaled anisotropically; triangle area varies ~10x."""
    pos, tris = noisy_sphere(level)
    return (pos.astype(np.float64) * np.array([1.0, 0.28, 2.6])
            ).astype(np.float32), tris


def torus(rings=320, sides=160, major=1.0, minor=0.38):
    """102k triangles, genus 1, curvature of both signs, rippled."""
    u = np.arange(rings) * (2.0 * np.pi / rings)
    v = np.arange(sides) * (2.0 * np.pi / sides)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    rr = minor * (1.0 + 0.18 * np.sin(5.0 * uu) * np.cos(7.0 * vv)
                  + 0.05 * np.sin(23.0 * vv))
    pos = np.stack([(major + rr * np.cos(vv)) * np.cos(uu),
                    (major + rr * np.cos(vv)) * np.sin(uu),
                    rr * np.sin(vv)], axis=-1).reshape(-1, 3).astype(np.float32)

    i = np.arange(rings)[:, None]
    j = np.arange(sides)[None, :]
    a = (i * sides + j).reshape(-1)
    b = (((i + 1) % rings) * sides + j).reshape(-1)
    c = (((i + 1) % rings) * sides + (j + 1) % sides).reshape(-1)
    d = (i * sides + (j + 1) % sides).reshape(-1)
    return pos, np.concatenate([np.stack([a, b, c], axis=1),
                                np.stack([a, c, d], axis=1)]).astype(np.int32)


def big_torus(rings=520, sides=400):
    """A torus large enough for v2's first stage to engage at every ratio.

    The torus shape is kept because it is the hard case: genus 1, curvature of
    both signs, and a tube whose two sides are close enough in space that a
    careless collapse confuses them.
    """
    return torus(rings=rings, sides=sides)


FIXTURES = {"sphere": noisy_sphere, "stretched": stretched, "torus": torus,
            "big_torus": big_torus}

# The default set leaves `big_torus` out, because it is four times the work and
# the three originals are what the tuning history in PLAN.md 4.3a refers to.
# Name it explicitly, or pass `all`, to measure it.
DEFAULT_FIXTURES = ("sphere", "stretched", "torus")


# -------------------------------------------------------------- measurement

def bvh_from_arrays(positions, triangles):
    from mathutils.bvhtree import BVHTree
    return BVHTree.FromPolygons([tuple(map(float, p)) for p in positions],
                                [tuple(map(int, t)) for t in triangles],
                                all_triangles=True)


def surface_distances(points, tree):
    """Distance from each point to the nearest surface point of `tree`."""
    out = np.empty(points.shape[0], dtype=np.float64)
    for i, p in enumerate(points):
        location, _normal, _index, dist = tree.find_nearest(
            (float(p[0]), float(p[1]), float(p[2])))
        out[i] = dist if location is not None else np.nan
    return out


def sample(points, rng):
    if points.shape[0] <= SAMPLES:
        return points
    return points[rng.choice(points.shape[0], SAMPLES, replace=False)]


# The source BVH, kept between rows. `BVHTree.FromPolygons` takes Python lists
# of tuples, so building one over a few hundred thousand triangles is tens of
# seconds -- and the source does not change between variants or ratios, so
# rebuilding it per row was most of the harness's runtime and the reason the
# fixtures had to stay small enough not to exercise v2 at all.
_SRC_BVH = {}


def source_bvh(src_pos, src_tris):
    key = (id(src_pos), id(src_tris), src_tris.shape[0])
    tree = _SRC_BVH.get(key)
    if tree is None:
        _SRC_BVH.clear()
        tree = bvh_from_arrays(src_pos, src_tris)
        _SRC_BVH[key] = tree
    return tree


def compare(src_pos, src_tris, out_pos, out_tris, diag):
    """Two-sided sampled surface distance, as a fraction of the bbox diagonal."""
    if out_tris.shape[0] == 0:
        return None
    src_bvh = source_bvh(src_pos, src_tris)
    out_bvh = bvh_from_arrays(out_pos, out_tris)
    fwd = surface_distances(sample(src_pos, np.random.default_rng(SEED)),
                            out_bvh) / diag
    rev = surface_distances(sample(out_pos, np.random.default_rng(SEED)),
                            src_bvh) / diag
    fwd, rev = fwd[np.isfinite(fwd)], rev[np.isfinite(rev)]
    return {"mean": float(max(fwd.mean(), rev.mean())),
            "hausdorff": float(max(fwd.max(), rev.max()))}


def blender_decimate(positions, triangles, ratio):
    """Blender's own Decimate modifier, as arrays. The reference."""
    import bpy
    mesh = egress.build_mesh("hd_quality_ref", positions, triangles)
    obj = bpy.data.objects.new("hd_quality_ref", mesh)
    bpy.context.collection.objects.link(obj)
    mod = obj.modifiers.new("hd_ref", 'DECIMATE')
    mod.decimate_type = 'COLLAPSE'
    mod.ratio = ratio
    t0 = time.perf_counter()
    evaluated = obj.evaluated_get(bpy.context.evaluated_depsgraph_get())
    out = bpy.data.meshes.new_from_object(evaluated)
    elapsed = time.perf_counter() - t0
    pos, tris, _, _ = ingest.read_mesh(out)
    pos, tris = pos.copy(), tris.astype(np.int64).copy()
    bpy.data.meshes.remove(out)
    bpy.data.objects.remove(obj, do_unlink=True)
    return pos, tris, elapsed


def _ours(positions, tris64, triangles, target, plan, variant):
    """One run of our pipeline, landing on the target itself, as (pos, tris).

    There is no second decimator to hand the endgame to: optimal placement is
    part of the collapse, so the pass loop reaches the target on its own and
    what is measured here is the whole of what ships.
    """
    result, info = dispatch.run(positions, tris64, target, Options(), plan)
    return result.positions, result.triangles.astype(np.int64), info


def run_one(name, positions, triangles, ratios, plan):
    tris64 = triangles.astype(np.int64)
    diag = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    print("")
    print(f"--- {name}: {positions.shape[0]:,} verts, "
          f"{triangles.shape[0]:,} triangles, bbox diagonal {diag:.4f}",
          flush=True)
    header = (f"{'ratio':>7} {'method':<18} {'tris':>9} {'x/tgt':>6} "
              f"{'time':>7} {'mean':>10} {'hausdorff':>10}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    rows = []
    for ratio in ratios:
        target = max(4, int(triangles.shape[0] * ratio))
        measured = {}
        for label, variant in VARIANTS:
            shown = label
            t0 = time.perf_counter()
            if variant is None:
                out_pos, out_tris, elapsed = blender_decimate(
                    positions, triangles, ratio)
            else:
                out_pos, out_tris, info = _ours(
                    positions, tris64, triangles, target, plan, variant)
                elapsed = time.perf_counter() - t0
            stats = compare(positions, tris64, out_pos, out_tris, diag)
            measured[label] = stats
            print(f"{ratio:>7.3f} {shown:<20} {out_tris.shape[0]:>9,} "
                  f"{out_tris.shape[0] / target:>6.2f} {elapsed:>6.2f}s "
                  f"{stats['mean']:>10.3e} {stats['hausdorff']:>10.3e}",
                  flush=True)

        ref = measured[REFERENCE_LABEL]
        for label, variant in VARIANTS:
            if variant is None:
                continue
            ours = measured[label]
            rm = ours["mean"] / max(ref["mean"], 1e-30)
            rh = ours["hausdorff"] / max(ref["hausdorff"], 1e-30)
            rows.append((name, ratio, label, rm, rh))
            print(f"{'':>7} {'-> / Blender':<18} {label:>9} {'':>6} {'':>7} "
                  f"{rm:>9.2f}x {rh:>9.2f}x", flush=True)
    return rows


def main():
    import bpy

    argv = sys.argv
    args = argv[argv.index("--") + 1:] if "--" in argv else []
    ratios = list(DEFAULT_RATIOS)
    if "--ratios" in args:
        i = args.index("--ratios")
        ratios = [float(x) for x in args[i + 1].split(",")]
        args = args[:i] + args[i + 2:]
    if "all" in args:
        names = list(FIXTURES)
    else:
        names = [a for a in args if a in FIXTURES] or list(DEFAULT_FIXTURES)

    plan = dispatch.choose()
    print(f"backend: {plan.tier} - {plan.reason}", flush=True)
    print(f"{SAMPLES:,} samples per direction, seed {SEED}", flush=True)

    # Only treat the scene as the subject when a .blend was actually opened.
    # --factory-startup ships a default Cube, and measuring that reports a
    # cheerful 0.50x against Blender on twelve triangles, which is how this
    # first "passed".
    scene_meshes = ([o for o in bpy.data.objects if o.type == 'MESH']
                    if bpy.data.filepath else [])
    rows = []
    if scene_meshes:
        obj = max(scene_meshes, key=lambda o: len(o.data.vertices))
        pos, tris, _, method = ingest.read_mesh(obj.data)
        rows += run_one(f"{obj.name} ({method})", pos, tris, ratios, plan)
    else:
        for name in names:
            pos, tris = FIXTURES[name]()
            rows += run_one(name, pos, tris, ratios, plan)

    print("")
    print("=== verdict ===", flush=True)
    print(f"  PLAN.md M4 target: mean error within {TARGET:.2f}x of Blender",
          flush=True)
    for name, ratio, label, rm, rh in rows:
        print(f"  {name:<12} {ratio:.3f} {label:<14}: "
              f"mean {rm:.2f}x  hausdorff {rh:.2f}x", flush=True)
    labels = []
    for _n, _r, label, _m, _h in rows:
        if label not in labels:
            labels.append(label)
    ok = True
    for label in labels:
        mine = [(rm, rh) for _n, _r, lab, rm, rh in rows if lab == label]
        worst = max(rm for rm, _ in mine)
        avg = sum(rm for rm, _ in mine) / len(mine)
        avg_h = sum(rh for _, rh in mine) / len(mine)
        print(f"  {label:<14} mean ratio: average {avg:.2f}x, "
              f"worst {worst:.2f}x; hausdorff average {avg_h:.2f}x", flush=True)
        # Only the shipped default has to clear the gate. Measuring a variant
        # is the point of being able to name one, and a variant that fails is
        # information rather than a build break.
        if label == DEFAULT_LABEL:
            ok = worst <= TARGET
            print(f"  {label:<14} worst mean ratio {worst:.2f}x - "
                  f"{'within target' if ok else 'OVER target'}", flush=True)

    dispatch.release()
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
