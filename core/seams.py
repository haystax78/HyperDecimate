"""UV seam detection. PLAN.md 4.3, milestone M4.

A UV seam is not a topological property, so unlike the mesh boundary it cannot be
found from connectivity alone and has to come from the UV layer itself.

What counts as a seam here is a *UV discontinuity*, not Blender's `use_seam` edge
flag. The flag records what the user marked for unwrapping and can be stale or
absent on an imported model; the discontinuity is what actually matters, because it
is what makes a collapse across it visibly tear the texture. A vertex is on a seam
when its incident corners do not all carry the same UV.

The rule applied to seams mirrors the one for boundaries (4.3): a seam vertex may
only collapse to another seam vertex, so the seam curve can shorten along itself
but never be pulled off its path.

**This is an approximation, deliberately.** Constraint planes are added for ring
edges whose *both* endpoints are seam vertices, which is not quite the same as
"this edge is a seam edge": two separate seams passing close together can have
adjacent vertices that are each on a seam without the edge between them being one.
Per-vertex is what the GPU can act on inside its image budget, and the failure mode
is an over-constrained edge rather than a torn texture. Exact per-wedge handling
needs attribute quadrics and is out of scope here.

That approximation is about which edges get held in place. It says nothing about
which UV each output corner ends up with, which *is* exact and belongs to
`core.egress.transfer_uv_layers`. Getting that part wrong is what welds the two
sides of a seam together; see PLAN.md 5.1a.
"""

from __future__ import annotations

import numpy as np

from . import ingest

# UVs are float32 and usually authored, so anything above this is a real
# discontinuity rather than rounding.
UV_TOLERANCE = 1e-6

# Corners per slice in `seam_vertices_from_uvs`. 16K to 1M measured within 3% of
# each other on a 26M-corner scan, so this is only about bounding the scratch.
_COMPARE_CHUNK = 1 << 18


def seam_vertices_from_uvs(corner_verts, corner_uvs, vertex_count,
                           tolerance=UV_TOLERANCE):
    """Per-vertex boolean: does this vertex carry more than one UV?

    corner_verts: (C,) vertex index per face corner.
    corner_uvs:   (C, 2) UV per face corner.

    Every corner is compared against one chosen corner of its own vertex and the
    results are OR'd per vertex. Grouping by vertex the obvious way means sorting
    the corner array, which on the 13M-triangle head scan is 39M elements and 2.1 s
    of the 5.0 s this used to cost. Neither the representative nor the OR needs
    that ordering: a reverse scatter picks the representative and a masked scatter
    does the OR, both linear, for a bit-identical answer in a fraction of the time.
    UVs stay float32 for the same reason, since promoting what Blender already
    stores as float32 buys no precision and doubles the memory traffic over the
    largest array here.

    Every version of this has returned the same mask as the sort did, and the
    tests assert that against hand-built fixtures, random meshes, and the values
    that distinguish a bitwise comparison from a numeric one.
    """
    corner_verts = np.asarray(corner_verts).reshape(-1)
    out = np.zeros(vertex_count, dtype=bool)
    n = corner_verts.size
    if n == 0:
        return out
    corner_uvs = np.ascontiguousarray(corner_uvs, dtype=np.float32).reshape(-1, 2)

    # The representative's UV is scattered straight into a per-vertex table, as
    # the eight raw bytes of the pair rather than as a corner index. When fancy
    # indexing repeats an index the last write wins, so walking backwards leaves
    # the lowest corner index's value in place -- the same representative the sort
    # picked, and the same one the index version picked.
    #
    # Scattering the value instead of the index removes three full-size arrays
    # that the index version needed: the reversed arange, the gather of
    # representative indices, and the gather of representative UVs through them.
    # A vertex with no corners is never written and never read, since every index
    # used below comes out of corner_verts.
    try:
        key = corner_uvs.view(np.uint64).reshape(-1)
    except (TypeError, ValueError):  # a view needs 8-byte alignment
        corner_uvs = corner_uvs.copy()
        key = corner_uvs.view(np.uint64).reshape(-1)
    rep = np.empty(vertex_count, dtype=np.uint64)
    rep[corner_verts[::-1]] = key[::-1]
    rep_uv = rep.view(np.float32).reshape(vertex_count, 2)

    # Compared as bits first, and only the corners that fail that get the real
    # tolerance test. A corner whose UV is bit-identical to its representative's
    # has a difference of exactly zero and so can never exceed the tolerance, and
    # on a normally unwrapped mesh almost every corner is in that case: the head
    # scan has 3,929 bit-different corners out of 26.08M. The arithmetic the old
    # version did over the whole array now runs over a few thousand elements.
    #
    # The bit test is a prefilter and nothing more, so the tolerance still decides
    # every answer. That matters for the values where the two disagree -- +0.0
    # against -0.0 differs in bits and not in value, and a NaN differs in bits
    # while failing every comparison -- and both come out as they did before.
    #
    # Sliced rather than done whole so the intermediates stay in cache instead of
    # being four more arrays the size of the corner list.
    for lo in range(0, n, _COMPARE_CHUNK):
        hi = min(lo + _COMPARE_CHUNK, n)
        verts = corner_verts[lo:hi]
        moved = key[lo:hi] != rep[verts]
        if not moved.any():
            continue
        picked = np.flatnonzero(moved)
        vs = verts[picked]
        delta = np.abs(corner_uvs[lo + picked] - rep_uv[vs])
        out[vs[(delta[:, 0] > tolerance) | (delta[:, 1] > tolerance)]] = True
    return out


def seam_edges_from_vertices(triangles, seam_mask):
    """Ring edges with a seam vertex at both ends, plus a face for each.

    Returns (edge_verts (E,2) int32, face_index (E,) int32), deduplicated, in the
    form `quadrics.add_constraint_planes` expects. See the module docstring for why
    this is derived per-vertex rather than per-edge.
    """
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.size == 0 or not seam_mask.any():
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.int32)

    a = np.concatenate([triangles[:, 0], triangles[:, 1], triangles[:, 2]])
    b = np.concatenate([triangles[:, 1], triangles[:, 2], triangles[:, 0]])
    face = np.tile(np.arange(triangles.shape[0], dtype=np.int64), 3)

    keep = seam_mask[a] & seam_mask[b]
    if not keep.any():
        return np.zeros((0, 2), dtype=np.int32), np.zeros(0, dtype=np.int32)
    a, b, face = a[keep], b[keep], face[keep]

    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    key = lo * (1 << 32) + hi
    _, first = np.unique(key, return_index=True)
    return (np.stack([lo[first], hi[first]], axis=1).astype(np.int32),
            face[first].astype(np.int32))


def read_seams(mesh):
    """Per-vertex seam mask for a Blender mesh, across every UV layer.

    Computed over the mesh's own loops, which is what the module docstring's
    definition says -- a vertex is on a seam when its incident corners do not all
    carry the same UV -- and is also what `egress._CornerUVs` has always used.

    This used to work in triangle-corner space, gathering the UV of every triangle
    corner through `tri_loops`, on the reasoning that a quad split or a
    tessellation must not be able to invent a discontinuity the original mesh did
    not have. Loop space cannot invent one either, and it is strictly less work:
    the triangulation only ever repeats loops, never introduces a corner with a
    new UV, so the set of (vertex, UV) pairs it sees is the same set. For the head
    scan it is 26.08M corners against 39.12M, the gather of 39.12M UV pairs
    disappears, and the mask is identical -- which the tests check on all three
    triangulation paths rather than taking on trust.

    The one case where the two could differ is a polygon corner that the
    tessellator drops entirely, which loop space would still count. That direction
    over-constrains rather than tearing a texture, which is the trade 4.3 already
    makes everywhere else.
    """
    vertex_count = len(mesh.vertices)
    mask = np.zeros(vertex_count, dtype=bool)
    layers = list(mesh.uv_layers)
    if not layers:
        return mask

    corner_verts = ingest.read_corner_verts(mesh)
    uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    for layer in layers:
        mask |= seam_vertices_from_uvs(
            corner_verts, ingest.read_uvs(mesh, layer, uv), vertex_count)
    return mask


def read_seams_and_uvs(mesh):
    """`read_seams`, plus one UV per vertex from the active UV layer.

    Returns (mask, uvs) with `uvs` a (V, 2) float32 array, or (mask, None)
    when the mesh has no UV layer. The per-vertex UV is what the collapse
    tests for UV folds; see `vertex_uvs`. Read here because the corner array
    and the UVs are already in hand.
    """
    vertex_count = len(mesh.vertices)
    mask = np.zeros(vertex_count, dtype=bool)
    layers = list(mesh.uv_layers)
    if not layers:
        return mask, None

    active = mesh.uv_layers.active or layers[0]
    corner_verts = ingest.read_corner_verts(mesh)
    uv = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    uvs = None
    for layer in layers:
        layer_uv = ingest.read_uvs(mesh, layer, uv)
        mask |= seam_vertices_from_uvs(corner_verts, layer_uv, vertex_count)
        if layer.name == active.name:
            uvs = vertex_uvs(corner_verts, layer_uv, vertex_count)
    return mask, uvs


def vertex_uvs(corner_verts, corner_uvs, vertex_count):
    """(V, 2) float32: the UV of each vertex's lowest-indexed corner.

    Exact away from seams, where every corner of a vertex agrees. On a seam
    vertex it is one of its wedges, which is why the fold test skips faces
    that touch a seam. A vertex with no corners gets (0, 0).
    """
    corner_verts = np.asarray(corner_verts).reshape(-1)
    corner_uvs = np.asarray(corner_uvs, dtype=np.float32).reshape(-1, 2)
    out = np.zeros((vertex_count, 2), dtype=np.float32)
    # Last write wins on repeated indices, so walking backwards leaves the
    # lowest corner's UV, the same representative `seam_vertices_from_uvs`
    # and `egress` pick.
    out[corner_verts[::-1]] = corner_uvs[::-1]
    return out
