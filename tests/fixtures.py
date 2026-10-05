"""Shared mesh fixtures and assertion helpers for the test suites.

These used to live at the top of the CPU backend's test file, and every other
suite imported them from there because there was no better place. With the CPU
backend gone the helpers keep the fixtures: one definition of "the test mesh"
is what makes the GPU checks comparable to each other.
"""

from __future__ import annotations

import numpy as np

from . import harness as _h

check = _h.check


# ----------------------------------------------------------------- generators

def grid(n):
    """Flat n x n triangulated plane in the z=0 plane."""
    xs = np.linspace(-1.0, 1.0, n)
    gx, gy = np.meshgrid(xs, xs, indexing="ij")
    pos = np.stack([gx.ravel(), gy.ravel(), np.zeros(n * n)], axis=1)
    idx = np.arange(n * n).reshape(n, n)
    a = idx[:-1, :-1].ravel()
    b = idx[1:, :-1].ravel()
    c = idx[1:, 1:].ravel()
    d = idx[:-1, 1:].ravel()
    tris = np.concatenate(
        [np.stack([a, b, c], axis=1), np.stack([a, c, d], axis=1)]
    )
    return pos, tris


def sphere(subdiv=4):
    """Icosphere by recursive subdivision. Closed, all triangles, no boundary."""
    t = (1.0 + 5.0 ** 0.5) / 2.0
    verts = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ], dtype=np.int64)

    for _ in range(subdiv):
        cache = {}
        new_faces = []
        vlist = list(verts)

        def midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key not in cache:
                cache[key] = len(vlist)
                vlist.append((verts[i] + verts[j]) * 0.5)
            return cache[key]

        for f in faces:
            i, j, k = int(f[0]), int(f[1]), int(f[2])
            a, b, c = midpoint(i, j), midpoint(j, k), midpoint(k, i)
            new_faces += [[i, a, c], [j, b, a], [k, c, b], [a, b, c]]
        verts = np.array(vlist)
        faces = np.array(new_faces, dtype=np.int64)

    verts = verts / np.linalg.norm(verts, axis=1)[:, None]
    return verts, faces


def seamed_grid(n=25):
    """A flat grid plus per-face UVs that put a real seam down the x = 0 column.

    `n` must be odd so there is a vertex column at exactly x = 0, which is the
    seam the tests then look for.

    The UV has to be a property of the *face*, not of the vertex. Deriving it
    from the vertex position gives every corner of a vertex the same UV, so no
    discontinuity can exist and the test passes while proving nothing, which is
    exactly what the first version of this fixture did.

    Returns (positions, triangles, corner_verts, corner_uvs, face_side), the
    last being which side of the split each face is on, which the UV transfer
    check needs to know what each output corner should have ended up with.
    """
    pos, tris = grid(n)
    corner_vert = tris.reshape(-1)
    face_side = pos[tris, 0].mean(axis=1) > 0.0
    uv = np.stack([pos[corner_vert, 0], pos[corner_vert, 1]], axis=1)
    uv[np.repeat(face_side, 3), 0] += 10.0
    return pos, tris, corner_vert, uv, face_side


def tube(rings=2, sides=3):
    """A closed-ended tube with very few sides, which violates the link
    condition all over the place. A 3-sided tube is the smallest mesh where
    two adjacent vertices share more neighbours than they share faces."""
    pos = []
    for r in range(rings):
        for s in range(sides):
            ang = 2.0 * np.pi * s / sides
            pos.append([np.cos(ang), np.sin(ang), float(r)])
    tris = []
    for r in range(rings - 1):
        for s in range(sides):
            a = r * sides + s
            b = r * sides + (s + 1) % sides
            c = (r + 1) * sides + s
            d = (r + 1) * sides + (s + 1) % sides
            tris += [[a, b, d], [a, d, c]]
    # Cap both ends so there is no boundary to confuse the picture.
    lo = list(range(sides))
    hi = [(rings - 1) * sides + s for s in range(sides)]
    pos.append([0.0, 0.0, -0.5])
    pos.append([0.0, 0.0, float(rings - 1) + 0.5])
    c_lo, c_hi = len(pos) - 2, len(pos) - 1
    for s in range(sides):
        tris.append([lo[(s + 1) % sides], lo[s], c_lo])
        tris.append([hi[s], hi[(s + 1) % sides], c_hi])
    return np.array(pos, dtype=np.float64), np.array(tris, dtype=np.int64)


def hash_u32(values, seed=0):
    """Deterministic, spatially decorrelated 32-bit hash (lowbias32).

    The oracle for the GPU's `hd_hash_seeded`: the candidate kernel breaks
    cost ties between targets by a hash of the target, reseeded every pass, so
    that a flat region -- where thousands of targets cost the same and the
    quadric cannot tell them apart -- does not funnel every vertex onto
    whichever neighbour the ring walk reached first.

    `seed` is xored in before mixing, which gives an unrelated ordering of
    the same indices.
    """
    x = values.astype(np.uint32)
    x ^= np.uint32(seed)
    x ^= x >> np.uint32(16)
    x *= np.uint32(0x7FEB352D)
    x ^= x >> np.uint32(15)
    x *= np.uint32(0x846CA68B)
    x ^= x >> np.uint32(16)
    return x


# ------------------------------------------------------------------- helpers

def accumulated_quadric_error(src_pos, src_tris, res):
    """Total quadric error of the output, summed over surviving vertices.

    The quantity the collapse's optimal placement minimises. Rebuilt here from
    the source rather than taken from the backend, so the test is checking the
    result and not the backend's own bookkeeping.
    """
    from ..core import quadrics as qd

    q = qd.build_vertex_quadrics(src_pos.astype(np.float64),
                                 np.asarray(src_tris, dtype=np.int64))
    n = res.positions.shape[0]
    acc = np.zeros((n, q.shape[1]), dtype=np.float64)
    remap = np.asarray(res.remap, dtype=np.int64)
    for k in range(q.shape[1]):
        acc[:, k] = np.bincount(remap, weights=q[:, k], minlength=n)[:n]
    return float(qd.quadric_error(acc, res.positions.astype(np.float64)).sum())


def longest_incident_edge(positions, triangles):
    """Per-vertex distance to the furthest vertex sharing a face with it."""
    from ..core import topology as tp

    tris = np.asarray(triangles, dtype=np.int64)
    src, dst, _ = tp.directed_edges(tris)
    d = np.linalg.norm(positions[dst] - positions[src], axis=1)
    out = np.zeros(positions.shape[0], dtype=np.float64)
    np.maximum.at(out, src, d)
    return out


def hausdorff_to_plane(pos):
    """Max |z| for a mesh that should have stayed in the z=0 plane."""
    return float(np.abs(pos[:, 2]).max()) if pos.size else 0.0


def max_radial_error(pos):
    """Deviation from the unit sphere."""
    r = np.linalg.norm(pos, axis=1)
    return float(np.abs(r - 1.0).max()) if pos.size else 0.0


def collapse_is_clean(triangles, v, w):
    """Perform v -> w and report whether the result stays a valid manifold."""
    step = np.arange(triangles.max() + 1, dtype=np.int64)
    step[v] = w
    mapped = step[triangles]
    alive = ~(
        (mapped[:, 0] == mapped[:, 1])
        | (mapped[:, 1] == mapped[:, 2])
        | (mapped[:, 2] == mapped[:, 0])
    )
    t = mapped[alive]
    edges = np.concatenate([t[:, :2], t[:, 1:3], t[:, [2, 0]]])
    keys = np.sort(edges, axis=1)
    uk, counts = np.unique(keys, axis=0, return_counts=True)
    return counts.max() <= 2


# ------------------------------------------------------------------ assertions

def assert_wellformed(name, pos, tris, remap, survived, orig_verts):
    check(f"{name}: indices in range",
          tris.size == 0 or (tris.min() >= 0 and tris.max() < pos.shape[0]),
          f"V={pos.shape[0]} F={tris.shape[0]}")
    degen = (
        (tris[:, 0] == tris[:, 1])
        | (tris[:, 1] == tris[:, 2])
        | (tris[:, 2] == tris[:, 0])
    )
    check(f"{name}: no degenerate triangles", not degen.any(),
          f"{int(degen.sum())} degenerate")
    check(f"{name}: no unreferenced vertices",
          np.unique(tris).size == pos.shape[0],
          f"{pos.shape[0] - np.unique(tris).size} orphaned")
    check(f"{name}: remap covers original vertices", remap.shape[0] == orig_verts)
    check(f"{name}: every remap target is a live vertex",
          remap.size == 0 or (remap.min() >= 0 and remap.max() < pos.shape[0]))
    check(f"{name}: survived count equals output vertex count",
          int(survived.sum()) == pos.shape[0],
          f"{int(survived.sum())} survived vs {pos.shape[0]} output verts")
    # Survivors must map onto the output vertices bijectively.
    idx = np.flatnonzero(survived)
    check(f"{name}: survivors map onto output vertices bijectively",
          np.array_equal(np.sort(remap[idx]), np.arange(pos.shape[0])))
    check(f"{name}: no NaN in output", np.isfinite(pos).all())


def assert_remap_consistent(name, orig_tris, out_tris, remap):
    """Pushing the original triangles through remap must rebuild the output.

    This is the strongest single statement about correctness available without a
    second implementation to diff against: it ties the vertex mapping, the
    collapse chain and the surviving topology together.
    """
    mapped = remap[orig_tris]
    alive = ~(
        (mapped[:, 0] == mapped[:, 1])
        | (mapped[:, 1] == mapped[:, 2])
        | (mapped[:, 2] == mapped[:, 0])
    )
    got = {tuple(sorted(t)) for t in mapped[alive]}
    want = {tuple(sorted(t)) for t in out_tris}
    check(f"{name}: remap reproduces output topology", got == want,
          f"{len(got)} mapped vs {len(want)} output faces, "
          f"{len(got ^ want)} differ")


def assert_face_origin(name, res, orig_faces):
    """Face provenance must be in range, unique, and consistent with the remap.

    Output face j came from original face face_origin[j], so pushing that
    original face's vertices through remap must give exactly the output face.
    """
    fo = res.face_origin
    check(f"{name}: face_origin in range",
          fo.size == 0 or (fo.min() >= 0 and fo.max() < orig_faces),
          f"{fo.size} faces, max {int(fo.max()) if fo.size else 0} of {orig_faces}")
    check(f"{name}: face_origin has no duplicates",
          np.unique(fo).size == fo.size,
          f"{fo.size - np.unique(fo).size} duplicated")


def assert_face_origin_maps(name, res, orig_tris):
    mapped = res.remap[orig_tris[res.face_origin]]
    got = np.sort(mapped, axis=1)
    want = np.sort(res.triangles, axis=1)
    check(f"{name}: face_origin agrees with remap",
          np.array_equal(got, want),
          f"{int((got != want).any(axis=1).sum())} of {len(want)} faces disagree")
