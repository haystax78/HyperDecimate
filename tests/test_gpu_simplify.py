"""Check the GPU stages. Must run inside Blender.

    blender --background --factory-startup --python tests/test_gpu_simplify.py

Each stage is checked against `core.*` before the stage above it is trusted.
The adjacency is compared as a set per row, because the GPU scatter uses
atomic cursors and so does not fix an order within a row; every consumer
treats a row as a set.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hyper_decimate.core import quadrics as qd  # noqa: E402
from hyper_decimate.core import topology as tp  # noqa: E402
from hyper_decimate.gpu_backend import context as ctx  # noqa: E402
from hyper_decimate.gpu_backend.state import GPUState  # noqa: E402
from hyper_decimate.core.options import Options  # noqa: E402
from hyper_decimate.tests.fixtures import (  # noqa: E402
    accumulated_quadric_error, assert_face_origin, assert_remap_consistent,
    assert_wellformed, grid, longest_incident_edge as _longest_incident_edge,
    seamed_grid, sphere, tube)

from hyper_decimate.tests.harness import check, finish, report  # noqa: E402


def _raises(fn, exc):
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


def rows_as_sets(offsets, values):
    return [frozenset(values[offsets[i]:offsets[i + 1]].tolist())
            for i in range(offsets.size - 1)]


def case(label, pos, tris):
    print(f"\n{label}: V={pos.shape[0]:,} F={tris.shape[0]:,}", flush=True)
    tris = np.ascontiguousarray(tris, dtype=np.int64)

    state = GPUState(pos, tris)
    state.build_adjacency()
    state.build_quadrics()

    # --- adjacency -------------------------------------------------------
    g_offs, g_adj = state.read_adjacency()
    c_offs, c_adj = tp.build_csr(tris, pos.shape[0])

    check(f"{label}: CSR offsets match",
          np.array_equal(g_offs.astype(np.int64), c_offs),
          f"{int((g_offs.astype(np.int64) != c_offs).sum())} of {c_offs.size} differ")
    if np.array_equal(g_offs.astype(np.int64), c_offs):
        g_rows = rows_as_sets(g_offs.astype(np.int64), g_adj.astype(np.int64))
        c_rows = rows_as_sets(c_offs, c_adj.astype(np.int64))
        bad = sum(1 for a, b in zip(g_rows, c_rows) if a != b)
        check(f"{label}: CSR rows hold the same corners", bad == 0,
              f"{bad} of {len(c_rows)} rows differ")

    # --- boundary detection ---------------------------------------------
    g_boundary = state.read_boundary_flags()
    border_verts, _ = tp.find_boundary_edges(tris)
    c_boundary = np.zeros(pos.shape[0], dtype=bool)
    if border_verts.size:
        c_boundary[np.unique(border_verts)] = True
    check(f"{label}: boundary vertices match",
          np.array_equal(g_boundary, c_boundary),
          f"{int((g_boundary != c_boundary).sum())} of {c_boundary.size} differ")

    # --- quadrics --------------------------------------------------------
    work, _, _ = qd.normalize_positions(np.ascontiguousarray(pos, np.float64))
    c_quad = qd.build_vertex_quadrics(work, tris)
    if border_verts.size:
        normals, _, _ = qd.face_planes(work, tris)
        _, bface = tp.find_boundary_edges(tris)
        qd.add_constraint_planes(c_quad, work, border_verts, normals[bface])
    g_quad = state.read_quadrics()

    # float32 on the GPU against float64 on the host, over an area-weighted sum,
    # so compare relatively against the magnitude of each row.
    scale = np.maximum(np.abs(c_quad).max(axis=1, keepdims=True), 1e-12)
    rel = np.abs(g_quad - c_quad) / scale
    worst = float(rel.max())
    check(f"{label}: quadrics agree to float32 precision", worst < 2e-4,
          f"max relative error {worst:.2e}")

    # Check the quadric where the algorithm actually evaluates it: at a
    # neighbour's position, which is what a half-edge collapse cost is. Not at
    # the vertex's own position, where the error is zero by construction for both
    # backends and a relative comparison is just dividing noise by noise.
    nbr = np.arange(pos.shape[0])
    have = np.diff(c_offs) > 0
    first_corner = c_adj[c_offs[:-1][have]]
    nbr[have] = tris.reshape(-1)[
        (first_corner // 3) * 3 + (first_corner + 1) % 3
    ]
    probe_at = work[nbr]
    g_err = qd.quadric_error(g_quad, probe_at)
    c_err = qd.quadric_error(c_quad, probe_at)

    # The right bound is absolute, scaled by the quadric's own magnitude, not
    # relative to the cost. Evaluating v^T Q v sums about ten terms each up to
    # |Q| * |p|^2, with |p| <= 0.5 after unit-cube normalisation, so float32 gives
    # an absolute error near 1e-7 * |Q|. Where the true cost is far below that,
    # as it is everywhere on a smooth sphere, the relative error is meaningless
    # and only the absolute figure says anything. Allow a 10x margin.
    qmag = np.abs(c_quad).max(axis=1)
    tol = 1e-6 * np.maximum(qmag, 1e-30)
    worst_abs = float((np.abs(g_err - c_err) / np.maximum(qmag, 1e-30)).max())
    check(f"{label}: collapse cost agrees within float32",
          bool((np.abs(g_err - c_err) <= tol).all()),
          f"worst error {worst_abs:.2e} x |Q|, cost range "
          f"{c_err.min():.2e} to {c_err.max():.2e}")
    return state


def test_remap_and_compaction():
    print("\nkernel 1 and 2: remap then compact", flush=True)
    pos, tris = sphere(2)
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    state = GPUState(pos, tris)

    # Collapse a handful of vertices onto neighbours, mirroring one pass.
    step = np.arange(pos.shape[0], dtype=np.int64)
    offsets, corners = tp.build_csr(tris, pos.shape[0])
    src, dst, _ = tp.directed_edges(tris)
    picked = []
    taken = np.zeros(pos.shape[0], dtype=bool)
    for a, b in zip(src, dst):
        if not taken[a] and not taken[b]:
            ring_a = tris[corners[offsets[a]:offsets[a + 1]] // 3].reshape(-1)
            if taken[ring_a].any():
                continue
            taken[ring_a] = True
            taken[a] = taken[b] = True
            picked.append((int(a), int(b)))
    for a, b in picked:
        step[a] = b
    print(f"  collapsing {len(picked)} independent pairs", flush=True)

    state.remap.upload(step.astype(np.uint32))
    # Verify the input before blaming the kernel that consumes it.
    back = state.remap.download(state.v_count).astype(np.int64)
    check("remap array arrived on the GPU intact", np.array_equal(back, step),
          f"{int((back != step).sum())} of {step.size} differ, "
          f"gpu[:6]={back[:6].tolist()} want {step[:6].tolist()}")

    state.apply_remap()
    g_tris = state.read_triangles().astype(np.int64)
    g_alive = state.alive.download(state.f_count).astype(bool)

    c_tris, c_dead = tp.apply_remap(tris, step)
    check("remapped corners match", np.array_equal(g_tris, c_tris),
          f"{int((g_tris != c_tris).any(axis=1).sum())} faces differ")
    check("dead flags match", np.array_equal(g_alive, ~c_dead),
          f"{int((g_alive != ~c_dead).sum())} flags differ")

    check("in-place compaction is refused",
          _raises(lambda: state.prims.compact_three(
              [state.c0, state.c1, state.c2], [state.c0, state.c1, state.c2],
              state.alive, state.f_count), ValueError))

    kept = state.compact_faces()
    want = c_tris[~c_dead]
    check("compacted face count matches", kept == want.shape[0],
          f"{kept} vs {want.shape[0]}")
    got = np.stack([state.c0.download(kept), state.c1.download(kept),
                    state.c2.download(kept)], axis=1).astype(np.int64)
    check("compacted faces match, in order", np.array_equal(got, want),
          f"{int((got != want).any(axis=1).sum())} differ"
          if got.shape == want.shape else "shape differs")


def test_end_to_end():
    """The whole GPU pipeline, checked for well-formedness."""
    from hyper_decimate.gpu_backend.simplify import simplify as gpu_simplify

    print("\nend to end: full GPU pass loop", flush=True)
    for label, (pos, tris), ratio in (
        ("grid 48x48", grid(48), 0.20),
        ("icosphere 4", sphere(4), 0.20),
        ("tube 3-sided", tube(8, 3), 0.50),
    ):
        tris = np.ascontiguousarray(tris, dtype=np.int64)
        target = max(4, int(tris.shape[0] * ratio))
        stats = {}
        res = gpu_simplify(pos, tris, target, Options(), stats=stats)
        print(f"  {label}: {tris.shape[0]} -> {res.triangles.shape[0]} faces "
              f"in {stats['passes']} passes", flush=True)

        ok_target = res.triangles.shape[0] <= target
        detail = f"{res.triangles.shape[0]} vs target {target}"
        if not ok_target:
            # A stall says far more with its pass schedule attached:
            # (pass, candidates, winners, faces before the pass).
            detail += f", schedule={stats['schedule'][:6]}"
        check(f"{label}: reached the target", ok_target, detail)
        assert_wellformed(label, res.positions, res.triangles, res.remap,
                          res.survived, pos.shape[0])
        assert_face_origin(label, res, tris.shape[0])
        assert_remap_consistent(label, tris, res.triangles, res.remap)

        # Placement is part of the collapse: the survivor moves to the pair
        # quadric's minimiser as the collapse commits. The alternative the
        # option selects is the edge midpoint, not standing still -- the
        # reference's --centroid control -- so what is worth checking is that
        # the placed result beats the midpoint one by the quadric's own
        # reckoning, and that nobody was thrown outside the patch it
        # absorbed: a minimiser may drift several collapses' worth inside
        # the region it merged, so the reach bound is that patch's extent,
        # which a remap bug -- a corner sent to the wrong vertex entirely --
        # exceeds by orders of magnitude.
        live = np.flatnonzero(res.survived)
        orig = pos[live]
        got = res.positions[res.remap[live]]
        drift = np.linalg.norm(orig - got, axis=1) if orig.size else np.zeros(0)
        n_out = res.positions.shape[0]
        lo = np.full((n_out, 3), np.inf)
        hi = np.full((n_out, 3), -np.inf)
        np.minimum.at(lo, res.remap, pos)
        np.maximum.at(hi, res.remap, pos)
        extent = np.linalg.norm(np.where(np.isfinite(lo), hi - lo, 0.0),
                                axis=1)
        over = int((drift > extent[res.remap[live]] + 1e-5).sum())
        check(f"{label}: no survivor moved outside the patch it absorbed",
              over == 0,
              f"{over} of {drift.size} did, worst "
              f"{float(np.max(drift, initial=0.0)):.2e}")

        mid = gpu_simplify(pos, tris, target,
                           Options(optimal_placement=False))
        before = accumulated_quadric_error(pos, tris, mid)
        after = accumulated_quadric_error(pos, tris, res)
        # "Never worse" is the real invariant: a planar mesh leaves nothing
        # for the solve to improve, so both runs sit at exactly zero there,
        # and two different greedy runs are not strictly comparable anyway.
        check(f"{label}: optimal placement no worse than the midpoint",
              after <= before * (1.0 + 1e-6) + 1e-9,
              f"{before:.4e} -> {after:.4e}")

        # No duplicate faces and no edge shared by three or more triangles.
        faces = {tuple(sorted(t)) for t in res.triangles}
        keys = tp.undirected_edge_keys(res.triangles.astype(np.int64))
        _, counts = np.unique(keys, return_counts=True)
        check(f"{label}: no duplicate faces",
              len(faces) == res.triangles.shape[0],
              f"{res.triangles.shape[0] - len(faces)} duplicated")
        check(f"{label}: no over-shared edges", int((counts > 2).sum()) == 0,
              f"{int((counts > 2).sum())} edges used by 3+ faces")


def test_param_inputs_released():
    """write_params must release its inputs without changing what it folded in.

    `vlock` and `vdensity` exist only for that one dispatch and are 0.78 GiB on a
    104M-vertex mesh, where the backend's arrays already come to 21.8 GiB of a
    24 GiB card. Releasing them is only safe if the queued dispatch has actually
    read them first, so what matters here is that the folded values survive --
    a premature free would show up as the state or density channel being wrong.
    """
    print("\nwrite_params releases its inputs", flush=True)
    pos, tris = sphere(3)
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    n = pos.shape[0]

    state = GPUState(pos, tris, freeze_borders=False)
    check("not allocated before write_params",
          state.vlock is None and state.vdensity is None)
    state.build_adjacency()
    state.build_quadrics()

    locked = np.zeros(n, dtype=bool)
    locked[::7] = True
    density = np.full(n, 3.0, dtype=np.float32)
    density[::5] = 0.25
    state.write_params(locked=locked, density=density)
    check("released after write_params",
          state.vlock is None and state.vdensity is None)

    # Layer 2's .z is the packed lock/seam/border state bits and .w the
    # density, both written by WRITE_PARAMS from the arrays that have now been
    # freed. Bit 0 is the lock; a sphere has no border or seam to set the rest.
    folded = ctx.download_layer(state.quad, 2)[:n]
    got_state = folded[:, 2].copy().view(np.uint32)
    got_density = folded[:, 3]
    want_density = np.maximum(density, 1e-6)
    check("lock state folded in correctly",
          np.array_equal(got_state & np.uint32(1),
                         locked.astype(np.uint32)),
          f"{int(((got_state & np.uint32(1)) != locked.astype(np.uint32)).sum())}"
          f" of {n} wrong")
    check("density folded in correctly",
          np.allclose(got_density, want_density, rtol=0, atol=0),
          f"max diff {float(np.abs(got_density - want_density).max()):.2e}")

    # Binding a released array must say so rather than failing obscurely.
    try:
        state._bind("vlock")
        check("binding a released array is refused", False, "no error raised")
    except RuntimeError as exc:
        check("binding a released array is refused",
              "write_params" in str(exc), str(exc)[:60])

    # And folding a fresh set in again has to work, reallocating as it goes.
    state.write_params(density=np.full(n, 2.0, dtype=np.float32))
    again = ctx.download_layer(state.quad, 2)[:n]
    check("a second write_params reallocates and folds again",
          np.allclose(again[:, 3], 2.0) and state.vdensity is None,
          f"density now {float(again[0, 3]):.3f}")

    # The whole pass loop must still run with the inputs gone.
    from hyper_decimate.gpu_backend.simplify import simplify as gpu_simplify
    out = gpu_simplify(pos, tris, max(4, tris.shape[0] // 4),
                       Options(locked=locked))
    check("a full run works with the inputs released",
          out.triangles.shape[0] <= tris.shape[0]
          and bool(np.isfinite(out.positions).all()),
          f"{tris.shape[0]} -> {out.triangles.shape[0]} faces")
    check("locked vertices did not move",
          _locked_held(pos, out, locked),
          "every surviving locked vertex kept its position")
    del state


def _locked_held(pos, res, locked):
    """Every locked vertex that survived must be where it started."""
    keep = np.flatnonzero(res.survived)
    held = keep[locked[keep]]
    if held.size == 0:
        return True
    moved = np.abs(res.positions[res.remap[held]] - pos[held]).max()
    return float(moved) < 1e-5


def test_seams_gpu():
    """The GPU must honour UV seams. PLAN.md 4.3."""
    from hyper_decimate.core import seams as sm
    from hyper_decimate.gpu_backend.simplify import simplify as gpu_simplify

    print("\nUV seams on the GPU", flush=True)
    n = 25
    # The same fixture the CPU seam test uses, so the two tiers are compared on
    # one definition of it rather than two copies that can drift.
    pos, tris, corner_vert, uv, _ = seamed_grid(n)
    tris = np.ascontiguousarray(tris, dtype=np.int64)
    mask = sm.seam_vertices_from_uvs(corner_vert, uv, pos.shape[0])
    check("fixture has a seam to preserve", int(mask.sum()) == n,
          f"{int(mask.sum())} seam vertices")

    target = max(4, int(tris.shape[0] * 0.15))
    opts = Options(seams=mask, freeze_borders=True)
    res = gpu_simplify(pos, tris, target, opts)

    survivor_is_seam = np.zeros(res.positions.shape[0], dtype=bool)
    survivor_is_seam[res.remap[np.flatnonzero(mask & res.survived)]] = True
    reps = res.remap[np.flatnonzero(mask)]
    check("seam vertices only collapse onto the seam",
          bool(survivor_is_seam[reps].all()),
          f"{int((~survivor_is_seam[reps]).sum())} of {int(mask.sum())} left it")

    kept = res.positions[res.remap[np.flatnonzero(mask & res.survived)]]
    worst = float(np.abs(kept[:, 0]).max()) if kept.size else 0.0
    check("surviving seam stays on the split", worst < 1e-6,
          f"max |x| on the seam = {worst:.2e}")

    # And the same run without seams should be free to cross the split, which is
    # what shows the constraint is doing something rather than being a no-op.
    plain = gpu_simplify(pos, tris, target, Options(freeze_borders=True))
    plain_survivor_seam = np.zeros(plain.positions.shape[0], dtype=bool)
    plain_survivor_seam[plain.remap[np.flatnonzero(mask & plain.survived)]] = True
    crossed = int((~plain_survivor_seam[plain.remap[np.flatnonzero(mask)]]).sum())
    print(f"  without the seam constraint, {crossed} seam vertices left the seam",
          flush=True)


def test_uv_folds_gpu():
    """With per-vertex UVs given, UVs follow the vertices and never fold.

    Two failures, both seen on a decimated head scan. A vertex kept its UV
    while its position moved with every collapse it survived, so its faces
    were textured as if it had not moved -- 43% of an edge out at 1% -- and
    the 3D normal test alone let 5% of the triangles fold over in UV.

    A bumpy height field under a smooth, fold-free UV warp reproduces both at
    a size a test can afford, and has an exact answer to check against: the
    surface is z = f(x, y) and the UV is a function of (x, y), so the right UV
    for any output vertex is the warp of where it now sits.
    """
    from hyper_decimate.gpu_backend.simplify import simplify as gpu_simplify

    print("\nUV tracking on the GPU", flush=True)
    pos, tris = grid(80)
    x, y = pos[:, 0], pos[:, 1]
    pos[:, 2] = 0.15 * np.sin(5.0 * x) * np.cos(4.0 * y)

    def warp(p):
        # Jacobian 1 - 0.5625 cos(3x) cos(3y) > 0 everywhere: no source folds.
        return np.stack([p[:, 0] + 0.25 * np.sin(3.0 * p[:, 1]),
                         p[:, 1] + 0.25 * np.sin(3.0 * p[:, 0])], axis=1)

    uvs = warp(pos).astype(np.float32)
    tris = np.ascontiguousarray(tris, dtype=np.int64)

    def signed(u):
        u = u.astype(np.float64)
        b, c = u[:, 1] - u[:, 0], u[:, 2] - u[:, 0]
        return b[:, 0] * c[:, 1] - b[:, 1] * c[:, 0]

    def untracked(res):
        """Each output vertex's UV as it was before tracking: its own."""
        keep = np.flatnonzero(res.survived)
        new_to_old = np.zeros(int(res.survived.sum()), dtype=np.int64)
        new_to_old[res.remap[keep]] = keep
        return uvs[new_to_old]

    src = np.sign(signed(uvs[tris]))
    check("fixture has no UV folds of its own", bool((src > 0).all()))

    target = int(tris.shape[0] * 0.05)
    plain = gpu_simplify(pos, tris, target, Options())
    tracked = gpu_simplify(pos, tris, target, Options(uvs=uvs))
    check("an untracked run returns no UVs", plain.uvs is None)
    check("a tracked run returns one UV per output vertex",
          tracked.uvs is not None
          and tracked.uvs.shape == (tracked.positions.shape[0], 2))

    def folds(res, vert_uv):
        out = np.sign(signed(vert_uv[res.triangles]))
        return int((out != src[res.face_origin]).sum())

    print(f"  without tracking, {folds(plain, untracked(plain))} triangles "
          "fold over in UV", flush=True)
    guarded = folds(tracked, tracked.uvs)
    check("no triangle folds over in UV with tracking", guarded == 0,
          f"{guarded} folded")

    # Accuracy, in units of the output's own median UV edge.
    def error(res, vert_uv):
        e = np.linalg.norm(vert_uv - warp(res.positions), axis=1)
        u = vert_uv[res.triangles]
        edge = np.median(np.linalg.norm(u[:, 1] - u[:, 0], axis=1))
        return float(np.median(e) / edge), float(e.max() / edge)

    before = error(plain, untracked(plain))
    after = error(tracked, tracked.uvs)
    print(f"  UV error, median/max in edges: untracked {before[0]:.3f}/"
          f"{before[1]:.3f}, tracked {after[0]:.4f}/{after[1]:.4f}", flush=True)
    check("tracked UVs sit where the surface maps the vertex",
          after[0] < 0.02 and after[1] < 0.1,
          f"median {after[0]:.4f}, max {after[1]:.4f} of an edge")
    check("tracking is what fixed it", after[0] < before[0] / 5,
          f"{after[0]:.4f} against {before[0]:.4f}")


def main():
    print("Hyper Decimate — GPU stage tests", flush=True)
    info = ctx.probe()
    print(f"  {info['backend']} / {info['renderer']}", flush=True)

    case("grid 32x32", *grid(32))
    case("icosphere 3", *sphere(3))
    case("tube 3-sided", *tube(4, 3))
    test_remap_and_compaction()
    test_end_to_end()
    test_param_inputs_released()
    test_seams_gpu()
    test_uv_folds_gpu()

    return report()


if __name__ == "__main__":
    finish(main())
