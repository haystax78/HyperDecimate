"""End-to-end test of ingest, simplify and egress inside Blender.

    blender --background --factory-startup --python tests/test_blender_io.py

Exercises all three triangulation paths, checks the decimated mesh passes
Blender's own validator, and checks that point attributes and UVs arrive intact.
Meshes are kept small because the checks are the point, not the benchmark;
path; speed at scale is `tools/bench_blender.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hyper_decimate.core import egress, ingest, seams  # noqa: E402
from hyper_decimate.core.options import Options
from hyper_decimate.gpu_backend.simplify import simplify  # noqa: E402

from hyper_decimate.tests.harness import check, finish, report  # noqa: E402


# ------------------------------------------------------------------- fixtures

def grid_arrays(n, ngon=False, quads=False):
    xs = np.linspace(-1.0, 1.0, n, dtype=np.float32)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel(), np.zeros(n * n, np.float32)], axis=1)
    # Give it some relief so quadric costs are not all zero.
    pos[:, 2] = 0.15 * np.sin(4.0 * pos[:, 0]) * np.cos(4.0 * pos[:, 1])
    idx = np.arange(n * n, dtype=np.int32).reshape(n, n)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    if quads or ngon:
        return pos, np.stack([a, b, c, d], axis=1)
    return pos, np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]
    )


def make_poly_mesh(name, positions, faces, per_face):
    """Build a mesh whose faces all have `per_face` corners."""
    nv, nf = positions.shape[0], faces.shape[0]
    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(nv)
    mesh.loops.add(nf * per_face)
    mesh.polygons.add(nf)
    mesh.vertices.foreach_set("co", positions.reshape(-1))
    mesh.loops.foreach_set("vertex_index", faces.reshape(-1))
    mesh.polygons.foreach_set(
        "loop_start", np.arange(nf, dtype=np.int32) * per_face
    )
    mesh.update(calc_edges=True)
    return mesh


def make_mixed_mesh(n, quad_count):
    """A grid where the first `quad_count` cells stay quads and the rest are split.

    Each cell is used exactly once, so the surface tiles the plane without
    overlaps. Getting this wrong produces coincident faces, and the symptom is
    `mesh.validate()` failing on the *output* while the input looks fine, which is
    a confusing way to spend an afternoon.
    """
    pos, quads = grid_arrays(n, quads=True)
    quad_count = min(quad_count, quads.shape[0])
    keep_q = quads[:quad_count]
    split = quads[quad_count:]
    tris = np.concatenate([split[:, [0, 1, 2]], split[:, [0, 2, 3]]])

    nf_q, nf_t = keep_q.shape[0], tris.shape[0]
    mesh = bpy.data.meshes.new("src_mixed")
    mesh.vertices.add(pos.shape[0])
    mesh.loops.add(nf_q * 4 + nf_t * 3)
    mesh.polygons.add(nf_q + nf_t)
    mesh.vertices.foreach_set("co", pos.reshape(-1))
    mesh.loops.foreach_set(
        "vertex_index",
        np.concatenate([keep_q.reshape(-1), tris.reshape(-1)]),
    )
    mesh.polygons.foreach_set("loop_start", np.concatenate([
        np.arange(nf_q, dtype=np.int32) * 4,
        nf_q * 4 + np.arange(nf_t, dtype=np.int32) * 3,
    ]))
    mesh.update(calc_edges=True)
    return mesh


def add_attributes(mesh):
    """Add a point float attribute, a point colour, and a UV layer."""
    nv = len(mesh.vertices)
    nl = len(mesh.loops)

    tag = mesh.attributes.new("hd_tag", "FLOAT", "POINT")
    tag.data.foreach_set("value", np.arange(nv, dtype=np.float32))

    col = mesh.attributes.new("hd_col", "FLOAT_COLOR", "POINT")
    rgba = np.zeros((nv, 4), dtype=np.float32)
    rgba[:, 0] = np.linspace(0.0, 1.0, nv, dtype=np.float32)
    rgba[:, 3] = 1.0
    col.data.foreach_set("color", rgba.reshape(-1))

    uv = mesh.uv_layers.new(name="hd_uv", do_init=False)
    corner_vert = ingest.read_corner_verts(mesh)
    pos = ingest.read_positions(mesh)
    # Planar UVs derived from position, so every loop of a vertex agrees and the
    # transfer in egress is exact rather than approximate.
    coords = (pos[corner_vert, :2] * 0.5 + 0.5).astype(np.float32)
    uv.uv.foreach_set("vector", coords.reshape(-1))
    return nl


# ---------------------------------------------------------------------- tests

def run_case(label, mesh, expect_method, ratio=0.35):
    print(f"\n{label}", flush=True)
    # Validate the fixture first. A bad source mesh shows up as a failure on the
    # decimated output, which points the finger at entirely the wrong code.
    check(f"{label}: source fixture is valid", not mesh.validate(verbose=False))
    add_attributes(mesh)

    positions, tris, tri_loops, method = ingest.read_mesh(mesh)
    check(f"{label}: triangulation path", method == expect_method,
          f"got {method}")
    check(f"{label}: triangles are in range",
          tris.min() >= 0 and tris.max() < positions.shape[0],
          f"V={positions.shape[0]} F={tris.shape[0]}")
    check(f"{label}: tri_loops in range",
          tri_loops.min() >= 0 and tri_loops.max() < len(mesh.loops))

    target = max(4, int(tris.shape[0] * ratio))
    res = simplify(positions, tris, target, Options(freeze_borders=True))
    check(f"{label}: reached target", res.triangles.shape[0] <= target,
          f"{tris.shape[0]} -> {res.triangles.shape[0]}, target {target}")

    out = egress.build_mesh(f"{label}_out", res.positions, res.triangles)
    check(f"{label}: output counts match arrays",
          len(out.vertices) == res.positions.shape[0]
          and len(out.polygons) == res.triangles.shape[0],
          f"{len(out.vertices)} verts, {len(out.polygons)} faces")

    # Blender's own validator. It returns True when it had to fix something,
    # which is the single most informative check available here.
    corrected = out.validate(verbose=False)
    check(f"{label}: mesh.validate() found nothing to fix", not corrected)

    copied = egress.transfer_point_attributes(mesh, out, res.remap, res.survived)
    check(f"{label}: point attributes copied",
          "hd_tag" in copied and "hd_col" in copied, f"copied {copied}")

    # hd_tag held the original vertex index, so after transfer each output vertex
    # must carry the index of the original vertex it came from.
    got = np.empty(len(out.vertices), dtype=np.float32)
    out.attributes["hd_tag"].data.foreach_get("value", got)
    keep = np.flatnonzero(res.survived)
    want = np.zeros(len(out.vertices), dtype=np.float32)
    want[res.remap[keep]] = keep.astype(np.float32)
    check(f"{label}: point attribute values follow the remap",
          np.array_equal(got, want),
          f"{int((got != want).sum())} of {got.size} wrong")

    uvs = egress.transfer_uv_layers(
        mesh, out, tri_loops, res.face_origin, res.remap, res.survived
    )
    check(f"{label}: UV layer copied", "hd_uv" in uvs, f"copied {uvs}")
    if "hd_uv" in uvs:
        uv_got = np.empty(len(out.loops) * 2, dtype=np.float32)
        out.uv_layers["hd_uv"].uv.foreach_get("vector", uv_got)
        uv_got = uv_got.reshape(-1, 2)
        out_cv = ingest.read_corner_verts(out)
        # Against the *source* vertex each output vertex descends from, not
        # against its output position. Optimal placement moves survivors after
        # the collapses (core/placement.py), so a UV copied faithfully from the
        # source no longer equals the same function of the new position. The
        # transfer's contract is provenance, not geometry, and that is what this
        # checks; deriving the expectation from `res.positions` instead would
        # make this a test of the placement step wearing a UV test's name.
        keep_v = np.flatnonzero(res.survived)
        new_to_old = np.zeros(res.positions.shape[0], dtype=np.int64)
        new_to_old[res.remap[keep_v]] = keep_v
        src_pos = ingest.read_positions(mesh)
        expect = (src_pos[new_to_old[out_cv], :2] * 0.5 + 0.5).astype(np.float32)
        err = float(np.abs(uv_got - expect).max())
        check(f"{label}: UVs match the planar layout exactly", err < 1e-6,
              f"max UV error {err:.2e}")
    return res


def _seams_triangle_space(mesh, tri_loops, triangles):
    """`read_seams` as it was: gather every triangle corner's UV through
    `tri_loops` and group in triangle-corner space."""
    vertex_count = len(mesh.vertices)
    mask = np.zeros(vertex_count, dtype=bool)
    n_loops = len(mesh.loops)
    for layer in mesh.uv_layers:
        uv = np.empty(n_loops * 2, dtype=np.float32)
        mesh.attributes[layer.name].data.foreach_get("vector", uv)
        uv = uv.reshape(n_loops, 2)
        flat = np.asarray(tri_loops).reshape(-1)
        mask |= seams.seam_vertices_from_uvs(
            np.asarray(triangles).reshape(-1), uv[flat], vertex_count)
    return mask


def _give_seamed_uvs(mesh):
    """Per-face UVs with a jump across x == 0, so a real seam exists.

    The default fixture derives UVs from position, which makes every loop of a
    vertex agree and would let a seam test pass while proving nothing.
    """
    if not mesh.uv_layers:
        mesh.uv_layers.new(name="hd_uv_seam", do_init=False)
    corner_vert = ingest.read_corner_verts(mesh)
    pos = ingest.read_positions(mesh)
    coords = (pos[corner_vert, :2] * 0.5 + 0.5).astype(np.float32)

    # Which polygon each loop belongs to, and which side of the split it is on.
    starts = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_start", starts)
    sizes = np.empty(len(mesh.polygons), dtype=np.int32)
    mesh.polygons.foreach_get("loop_total", sizes)
    poly_of_loop = np.repeat(np.arange(starts.size, dtype=np.int64), sizes)
    centre_x = np.zeros(starts.size, dtype=np.float64)
    np.add.at(centre_x, poly_of_loop, pos[corner_vert, 0])
    centre_x /= sizes
    coords[(centre_x > 0.0)[poly_of_loop], 0] += 10.0

    mesh.uv_layers[-1].uv.foreach_set("vector", coords.reshape(-1))
    return coords


def test_read_seams_loop_space():
    """Loop space must give the same mask as triangle-corner space.

    `read_seams` used to gather the UV of every triangle corner through
    `tri_loops`; it now reads the mesh's own loops, which is 26.08M corners
    instead of 39.12M on the head scan and drops the gather entirely. The
    triangulation only ever repeats loops, so the set of (vertex, UV) pairs is the
    same -- asserted here on all three triangulation paths rather than assumed.
    """
    print("\nread_seams in loop space", flush=True)
    cases = [
        ("tris", make_poly_mesh("seam_tris", *grid_arrays(40), 3)),
        ("quads", make_poly_mesh("seam_quads", *grid_arrays(40, quads=True), 4)),
        ("mixed", make_mixed_mesh(40, quad_count=200)),
    ]
    for label, mesh in cases:
        _give_seamed_uvs(mesh)
        positions, tris, tri_loops, method = ingest.read_mesh(mesh)
        loop_mask = seams.read_seams(mesh)
        tri_mask = _seams_triangle_space(mesh, tri_loops, tris)
        check(f"{label}: the fixture contains a real seam",
              int(loop_mask.sum()) > 0, f"{int(loop_mask.sum())} seam vertices")
        check(f"{label}: loop space equals triangle-corner space",
              np.array_equal(loop_mask, tri_mask),
              f"{int(loop_mask.sum())} vs {int(tri_mask.sum())}, "
              f"{int((loop_mask != tri_mask).sum())} differ ({method})")
        # The seam must be a thin band near the split rather than most of the
        # mesh, or the comparison above could agree for the wrong reason. The
        # grid has an even number of columns, so there is no vertex exactly at
        # x == 0 and the band is one or two columns wide depending on whether
        # face centres are quad centres or triangle centroids.
        xs = positions[loop_mask, 0]
        width = float(xs.max() - xs.min()) if xs.size else 0.0
        check(f"{label}: the seam is a thin band at the split",
              loop_mask.sum() < positions.shape[0] // 8 and width < 0.11
              and abs(float(xs.mean())) < 0.06,
              f"{int(loop_mask.sum())} verts, x span {width:.3f}, "
              f"centred {float(xs.mean()):+.3f}")

    empty = make_poly_mesh("seam_nouv", *grid_arrays(8), 3)
    check("a mesh with no UV layer gives an empty mask",
          not seams.read_seams(empty).any())


def test_fast_int_read():
    """The pointer route must agree with `foreach_get`, element for element.

    `read_corner_verts` reads Blender's own attribute storage through an address
    rather than going through `foreach_get`, which is 55x faster and is the single
    biggest read in the addon. The API promises nothing about that memory, so the
    agreement is worth asserting on every triangulation shape rather than trusting
    the layout check in `_int_attribute_view` alone.
    """
    print("\nfast integer attribute read", flush=True)
    cases = [
        ("tris", make_poly_mesh("fast_tris", *grid_arrays(40), 3)),
        ("quads", make_poly_mesh("fast_quads", *grid_arrays(40, quads=True), 4)),
        ("mixed", make_mixed_mesh(40, quad_count=200)),
    ]
    for label, mesh in cases:
        n = len(mesh.loops)
        slow = np.empty(n, dtype=np.int32)
        mesh.attributes[".corner_vert"].data.foreach_get("value", slow)

        view = ingest._int_attribute_view(mesh.attributes[".corner_vert"].data, n)
        check(f"{label}: layout check accepts this mesh", view is not None,
              f"{n:,} corners")
        if view is not None:
            check(f"{label}: pointer route equals foreach_get",
                  np.array_equal(np.asarray(view), slow),
                  f"{int((np.asarray(view) != slow).sum())} of {n:,} differ")

        fast = ingest.read_corner_verts(mesh)
        check(f"{label}: read_corner_verts exact and int32",
              np.array_equal(fast, slow) and fast.dtype == np.int32,
              f"dtype {fast.dtype}")
        # Returning the view itself would hand the caller an array backed by
        # memory Blender is free to move or free.
        check(f"{label}: result owns its memory", fast.flags.owndata)

        # The documented escape hatch has to actually work.
        ingest.FAST_INT_READ = False
        try:
            check(f"{label}: FAST_INT_READ = False falls back and agrees",
                  np.array_equal(ingest.read_corner_verts(mesh), slow))
        finally:
            ingest.FAST_INT_READ = True

    # A one-element collection cannot have its stride measured, so it must be
    # refused rather than guessed at.
    one = make_poly_mesh("fast_one", *grid_arrays(2), 3)
    check("a collection too short to measure is refused",
          ingest._int_attribute_view(
              one.attributes[".corner_vert"].data, 1) is None)


def main():
    print("Hyper Decimate — Blender ingest/egress tests", flush=True)

    test_fast_int_read()
    test_read_seams_loop_space()

    pos, tris = grid_arrays(40)
    run_case("all-tris", make_poly_mesh("src_tris", pos, tris, 3),
             ingest.TRIS_DIRECT)

    qpos, quads = grid_arrays(40, quads=True)
    run_case("all-quads", make_poly_mesh("src_quads", qpos, quads, 4),
             ingest.QUADS_SPLIT)

    run_case("mixed quads+tris", make_mixed_mesh(40, quad_count=200),
             ingest.TESSELLATED)

    return report()


if __name__ == "__main__":
    finish(main())
