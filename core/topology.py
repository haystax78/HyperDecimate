"""Connectivity helpers: CSR adjacency, boundary detection, remap and compaction.

Everything here is index arithmetic on flat arrays, deliberately mirroring what
the GPU kernels will do, so the two backends can be diffed pass for pass.

Corner numbering: corner c belongs to triangle c // 3 and is the (c % 3)'th
vertex of it. So triangles.reshape(-1)[c] is the vertex at corner c.
"""

from __future__ import annotations

import numpy as np


def build_csr(triangles, vertex_count):
    """Map each vertex to the corners that reference it.

    Returns (offsets (V+1,), corners (3F,)) where the corners touching vertex v
    are corners[offsets[v]:offsets[v + 1]].

    argsort with kind="stable" picks a radix sort for integer keys, which is
    linear in the corner count rather than n log n. That matters: this is the
    single hottest CPU-side operation in the whole pipeline.
    """
    flat = triangles.reshape(-1)
    counts = np.bincount(flat, minlength=vertex_count)
    offsets = np.zeros(vertex_count + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    corners = np.argsort(flat, kind="stable").astype(np.int32)
    return offsets, corners


def segment_starts(offsets):
    """Group start indices for the non-empty rows of a CSR, plus their row ids.

    reduceat misbehaves on empty segments, so every segmented reduction in this
    codebase filters through here first.
    """
    counts = np.diff(offsets)
    rows = np.flatnonzero(counts > 0)
    return rows, offsets[rows]


def segmented_min(values, offsets):
    """Minimum of values within each CSR row. Empty rows yield +inf."""
    out = np.full(offsets.shape[0] - 1, np.inf)
    if values.size == 0:
        return out
    rows, starts = segment_starts(offsets)
    if rows.size:
        out[rows] = np.minimum.reduceat(values, starts)
    return out


def segmented_argmin(values, offsets):
    """Index into values of the minimum within each CSR row, or -1 if empty."""
    out = np.full(offsets.shape[0] - 1, -1, dtype=np.int64)
    if values.size == 0:
        return out
    rows, starts = segment_starts(offsets)
    if not rows.size:
        return out
    row_min = np.minimum.reduceat(values, starts)
    # Expand the per-row minimum back over the flat array, then take the first
    # position that attains it. Ties resolve to the lowest index, which is what
    # keeps the whole algorithm deterministic.
    counts = np.diff(offsets)[rows]
    expanded = np.repeat(row_min, counts)
    hit = np.flatnonzero(values == expanded)
    which = np.repeat(np.arange(rows.size), counts)[hit]
    first = np.full(rows.size, -1, dtype=np.int64)
    # Reverse order so the earliest hit wins the final write.
    first[which[::-1]] = hit[::-1]
    out[rows] = first
    return out


def directed_edges(triangles):
    """Every half-edge as (source, target) pairs, plus its owning triangle.

    Each triangle yields six directed edges. A half-edge collapse v -> w is a
    candidate exactly when (v, w) appears here, so this is the candidate set.
    """
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    src = np.concatenate([v0, v1, v1, v2, v2, v0])
    dst = np.concatenate([v1, v0, v2, v1, v0, v2])
    face = np.tile(np.arange(triangles.shape[0], dtype=np.int32), 6)
    return src, dst, face


def undirected_edge_keys(triangles):
    """Canonical (min, max) key per half-edge, as one int64 for fast grouping."""
    v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    a = np.concatenate([v0, v1, v2]).astype(np.int64)
    b = np.concatenate([v1, v2, v0]).astype(np.int64)
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return lo * (1 << 32) + hi


def edge_keys(a, b):
    """Canonical int64 key for arbitrary vertex pairs, matching
    `undirected_edge_keys`."""
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    return np.minimum(a, b) * (1 << 32) + np.maximum(a, b)


def link_condition_ok(triangles, offsets, corners, sources, targets,
                      vertex_count):
    """Which collapses `sources[i] -> targets[i]` preserve topology.

    The link condition: collapsing edge (v, w) is safe exactly when the number of
    vertices adjacent to both v and w equals the number of triangles containing
    the edge (v, w). Two for a manifold interior edge, one for a boundary edge.
    An extra shared neighbour means v and w are already joined by a second path,
    and merging them folds that path into a duplicate face or a non-manifold
    edge. Dey et al., "Topology Preserving Edge Contraction", 1999.

    This is not optional, and it is not covered by the normal-flip test. Without
    it, synthetic grids and spheres still come out clean, which is exactly what
    makes it easy to miss: a regular mesh rarely violates the condition. A real
    scan violates it constantly, around thin features and creases, and the only
    symptom is that Blender's own `mesh.validate()` starts finding things to fix
    while every other check still passes.
    """
    sources = np.asarray(sources, dtype=np.int64)
    targets = np.asarray(targets, dtype=np.int64)
    if sources.size == 0:
        return np.zeros(0, dtype=bool)

    all_keys = np.sort(undirected_edge_keys(triangles))
    unique_keys = np.unique(all_keys)

    # Triangles containing the edge (v, w).
    vw = edge_keys(sources, targets)
    face_count = (
        np.searchsorted(all_keys, vw, side="right")
        - np.searchsorted(all_keys, vw, side="left")
    )

    # Distinct vertices adjacent to v, other than w.
    counts = (offsets[sources + 1] - offsets[sources]).astype(np.int64)
    total = int(counts.sum())
    if total == 0:
        return face_count == 0
    owner = np.repeat(np.arange(sources.size, dtype=np.int64), counts)
    ramp = np.arange(total, dtype=np.int64) - np.repeat(
        np.cumsum(counts) - counts, counts
    )
    incident = corners[np.repeat(offsets[sources], counts) + ramp]
    tri = triangles[incident.astype(np.int64) // 3]

    # Each incident face contributes its two vertices that are not v.
    v_of = sources[owner]
    other = np.where(tri == v_of[:, None], -1, tri)
    flat_other = other.reshape(-1)
    flat_owner = np.repeat(owner, 3)
    valid = flat_other >= 0
    flat_other = flat_other[valid]
    flat_owner = flat_owner[valid]

    # Deduplicate (owner, neighbour): a neighbour is shared by two faces.
    packed = np.unique(flat_owner * np.int64(vertex_count) + flat_other)
    own = packed // np.int64(vertex_count)
    nbr = packed % np.int64(vertex_count)

    keep = nbr != targets[own]
    own = own[keep]
    nbr = nbr[keep]

    # Is that neighbour also adjacent to w?
    wu = edge_keys(targets[own], nbr)
    pos = np.searchsorted(unique_keys, wu)
    pos_clamped = np.minimum(pos, unique_keys.size - 1)
    shared = unique_keys[pos_clamped] == wu
    shared_count = np.bincount(own[shared], minlength=sources.size)

    return shared_count == face_count


def find_boundary_edges(triangles):
    """Edges used by exactly one triangle, with the index of that triangle.

    Returns (edge_verts (E,2) int32, face_index (E,) int32).
    """
    keys = undirected_edge_keys(triangles)
    face = np.tile(np.arange(triangles.shape[0], dtype=np.int32), 3)

    order = np.argsort(keys, kind="stable")
    sorted_keys = keys[order]

    # A key appearing once is a boundary edge. Two is manifold interior. More
    # than two is non-manifold, and is treated as a constraint too: locking it
    # is far cheaper than trying to collapse it correctly.
    unique, first, counts = np.unique(
        sorted_keys, return_index=True, return_counts=True
    )
    picked = counts != 2
    if not picked.any():
        empty_v = np.zeros((0, 2), dtype=np.int32)
        return empty_v, np.zeros(0, dtype=np.int32)

    key = unique[picked]
    idx = order[first[picked]]
    lo = (key >> 32).astype(np.int32)
    hi = (key & 0xFFFFFFFF).astype(np.int32)
    return np.stack([lo, hi], axis=1), face[idx]


def apply_remap(triangles, remap):
    """Rewrite every corner through remap, and report which triangles died.

    A triangle with two equal corners has collapsed to a line and carries no
    area, so it is removed. Because collapses within a pass form an independent
    set, remap is only ever one level deep and needs no path compression.
    """
    out = remap[triangles]
    degenerate = (
        (out[:, 0] == out[:, 1])
        | (out[:, 1] == out[:, 2])
        | (out[:, 2] == out[:, 0])
    )
    return out, degenerate


def compact_vertices(positions, triangles, extra=None):
    """Drop unreferenced vertices and renumber. Returns (P, T, old_to_new).

    old_to_new is -1 for vertices that no longer exist, which is what attribute
    transfer on the way out needs.
    """
    used = np.zeros(positions.shape[0], dtype=bool)
    used[triangles.reshape(-1)] = True
    keep = np.flatnonzero(used)

    old_to_new = np.full(positions.shape[0], -1, dtype=np.int32)
    old_to_new[keep] = np.arange(keep.size, dtype=np.int32)

    new_tris = old_to_new[triangles]
    new_pos = positions[keep]
    if extra is None:
        return new_pos, new_tris, old_to_new
    return new_pos, new_tris, old_to_new, [e[keep] for e in extra]
