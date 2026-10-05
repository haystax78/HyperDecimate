"""Quadric error metric math, shared by the CPU and GPU backends.

A quadric is the symmetric 4x4 matrix Q = n n^T for a homogeneous plane
n = (a, b, c, d) with unit normal, scaled by the triangle's area. The error of
placing a vertex at v is v^T Q v, which is the area-weighted sum of squared
distances to the planes that built Q.

Only the 10 unique coefficients are stored, in this order:

    index   0   1   2   3   4   5   6   7   8   9
    term    aa  ab  ac  ad  bb  bc  bd  cc  cd  dd

which lays out the matrix as

    [ q0 q1 q2 q3 ]
    [ q1 q4 q5 q6 ]
    [ q2 q5 q7 q8 ]
    [ q3 q6 q8 q9 ]

Garland & Heckbert, "Surface Simplification Using Quadric Error Metrics",
SIGGRAPH 1997.
"""

from __future__ import annotations

import numpy as np

# Weight applied to the artificial planes that hold boundary and seam loops in
# place, matching the reference's `--border-weight` default of 8. The plane is
# scaled by the edge's squared length, so 8 is strong enough to keep the border
# collapse near its own polyline while still allowing it to thin.
BOUNDARY_WEIGHT = 8.0

QUADRIC_LEN = 10


# Rows per slice in `normalize_bounds`. The slice is transposed into a scratch
# buffer, so it wants to be small enough to stay in cache -- 16K to 256K all
# measured within 2 ms of each other on a 6.5M-vertex scan, and 1M was 45% worse.
_BOUNDS_CHUNK = 1 << 18


def _axis_bounds(positions):
    """Per-column (min, max) of an (N, 3) array, in the array's own dtype.

    `positions.min(axis=0)` is the obvious spelling and it is 20 times slower
    than it needs to be: reducing along the outer axis of an (N, 3) array gives
    NumPy a three-element inner loop with nothing to vectorise, which measured
    350 MB/s -- 0.216 s on a 75 MB array, making it the single largest piece of
    host work in the whole GPU path. Transposing a slice at a time turns it into
    three contiguous reductions over cache-resident data: 0.011 s for the same
    answer.

    Exactly the same answer, not merely a close one. min and max are exact in any
    precision and do not care in what order they see the values, so slicing and
    combining cannot change the result -- unlike a sum, which this must never
    become.
    """
    n = positions.shape[0]
    dtype = positions.dtype if positions.dtype.kind == "f" else np.float64
    lo = np.full(3, np.inf, dtype=dtype)
    hi = np.full(3, -np.inf, dtype=dtype)
    for start in range(0, n, _BOUNDS_CHUNK):
        block = np.ascontiguousarray(positions[start:start + _BOUNDS_CHUNK].T)
        for axis in range(3):
            row = block[axis]
            lo[axis] = min(lo[axis], row.min())
            hi[axis] = max(hi[axis], row.max())
    return lo, hi


def normalize_bounds(positions):
    """The (centre, scale) that `normalize_positions` would apply.

    Separate from the transform so a caller that is going to write the result
    straight into a float32 destination never has to materialise the scaled
    float64 copy. The bounds are reduced in float64 whatever the input dtype:
    min and max are exact in any precision, but `(lo + hi) * 0.5` is not, so
    promoting first is what keeps the centre identical for float32 and float64
    input and therefore keeps the two backends comparable.
    """
    positions = np.asarray(positions)
    if positions.shape[0] == 0:
        return np.zeros(3, dtype=np.float64), 1.0
    lo, hi = _axis_bounds(positions)
    lo = lo.astype(np.float64)
    hi = hi.astype(np.float64)
    centre = (lo + hi) * 0.5
    extent = float((hi - lo).max())
    scale = 1.0 / extent if extent > 0.0 else 1.0
    return centre, scale


def normalize_positions(positions):
    """Fit positions into the unit cube centred on the origin.

    Quadric coefficients are products of plane coefficients, so they grow with
    the square of coordinate magnitude. On a sculpt with world coordinates in
    the hundreds, float32 quadrics accumulated over a high-valence vertex lose
    most of their significant digits to cancellation. Normalising first is not
    an optimisation, it is what makes the metric usable in single precision.

    Returns (scaled_positions, centre, scale) so the caller can invert it.
    """
    centre, scale = normalize_bounds(positions)
    return (positions - centre) * scale, centre, scale


def denormalize_positions(positions, centre, scale):
    """Invert normalize_positions."""
    return positions / scale + centre


def face_planes(positions, triangles):
    """Per-triangle unit normal, plane offset and area.

    Returns (normals (F,3), d (F,), areas (F,)). Degenerate triangles get a
    zero normal and zero area, so they contribute nothing to any quadric.
    """
    p0 = positions[triangles[:, 0]]
    p1 = positions[triangles[:, 1]]
    p2 = positions[triangles[:, 2]]

    cross = np.cross(p1 - p0, p2 - p0)
    twice_area = np.linalg.norm(cross, axis=1)
    areas = twice_area * 0.5

    # Guard the divide rather than masking afterwards, so no NaN ever enters a
    # quadric. A zero normal yields a zero quadric, which is the right answer
    # for a degenerate triangle.
    safe = np.where(twice_area > 0.0, twice_area, 1.0)
    normals = cross / safe[:, None]
    normals[twice_area <= 0.0] = 0.0

    d = -np.einsum("ij,ij->i", normals, p0)
    return normals, d, areas


def plane_quadrics(normals, d, weights):
    """Build the 10 coefficients of w * n n^T for each plane. Shape (N, 10)."""
    a, b, c = normals[:, 0], normals[:, 1], normals[:, 2]
    out = np.empty((normals.shape[0], QUADRIC_LEN), dtype=np.float64)
    out[:, 0] = a * a
    out[:, 1] = a * b
    out[:, 2] = a * c
    out[:, 3] = a * d
    out[:, 4] = b * b
    out[:, 5] = b * c
    out[:, 6] = b * d
    out[:, 7] = c * c
    out[:, 8] = c * d
    out[:, 9] = d * d
    out *= weights[:, None]
    return out


def scatter_to_vertices(triangles, face_quadrics, vertex_count):
    """Sum each triangle's quadric onto its three vertices. Shape (V, 10).

    Uses one bincount per coefficient. bincount is a tight C loop, which is
    roughly an order of magnitude faster than np.add.at for this shape.
    """
    corners = triangles.reshape(-1)
    out = np.zeros((vertex_count, QUADRIC_LEN), dtype=np.float64)
    for k in range(QUADRIC_LEN):
        # Each triangle contributes the same quadric to all three of its
        # corners, hence the repeat.
        out[:, k] = np.bincount(
            corners,
            weights=np.repeat(face_quadrics[:, k], 3),
            minlength=vertex_count,
        )
    return out


def build_vertex_quadrics(positions, triangles, vertex_count=None):
    """Surface quadric for every vertex, weighted by twice the triangle area.

    The reference implementation weights each face's plane by |e1 x e2|, the
    parallelogram area rather than the triangle's, which is twice-area
    weighting -- the same shape as Garland & Heckbert's area weighting, scaled
    by a constant.
    """
    if vertex_count is None:
        vertex_count = positions.shape[0]
    normals, d, areas = face_planes(positions, triangles)
    return scatter_to_vertices(
        triangles, plane_quadrics(normals, d, 2.0 * areas), vertex_count
    )


def add_constraint_planes(quadrics, positions, edge_verts, edge_face_normals,
                          weight=BOUNDARY_WEIGHT):
    """Add a perpendicular holding plane for each constrained edge.

    For a boundary or seam edge (u, v) lying in a face with normal n_f, the
    constraint plane contains the edge and is perpendicular to the face. Adding
    it to both endpoints makes moving either one off the edge expensive, which
    is what "freeze borders" and "keep UVs" actually mean in practice.

    edge_verts is (E, 2), edge_face_normals is (E, 3).
    """
    if edge_verts.shape[0] == 0:
        return quadrics

    pu = positions[edge_verts[:, 0]]
    pv = positions[edge_verts[:, 1]]
    edge = pv - pu

    normals = np.cross(edge, edge_face_normals)
    length = np.linalg.norm(normals, axis=1)
    safe = np.where(length > 0.0, length, 1.0)
    normals = normals / safe[:, None]
    normals[length <= 0.0] = 0.0

    d = -np.einsum("ij,ij->i", normals, pu)

    # Scale by squared edge length so the constraint keeps pace with the
    # area-weighted surface quadrics as the mesh coarsens.
    edge_len_sq = np.einsum("ij,ij->i", edge, edge)
    weights = np.full(edge_verts.shape[0], weight) * edge_len_sq

    q = plane_quadrics(normals, d, weights)
    for k in range(QUADRIC_LEN):
        quadrics[:, k] += np.bincount(
            edge_verts.reshape(-1),
            weights=np.repeat(q[:, k], 2),
            minlength=quadrics.shape[0],
        )
    return quadrics


def quadric_weight(quadrics):
    """The total plane weight accumulated in each quadric. Shape (N,).

    Free, with no extra storage: every plane is added with a *unit* normal, so
    its 3x3 block is w * n n^T whose trace is w * |n|^2 = w. Quadrics add
    linearly, so the trace keeps summing the weights through every merge.

    This is what turns the raw quadric value into a distance-like quantity.
    `v^T Q v` is a sum over incident planes, so it grows with how much surface a
    vertex carries; dividing by the weight makes it a weighted *mean* squared
    distance instead, which is the thing the quality metric actually measures.
    meshoptimizer's `quadricError` does the same division.
    """
    return quadrics[..., 0] + quadrics[..., 4] + quadrics[..., 7]


def quadric_error(quadrics, points):
    """Evaluate v^T Q v for matching rows of quadrics (N,10) and points (N,3).

    Expanded rather than assembled into 4x4 matrices, because at 25M rows the
    matrix form costs an order of magnitude more memory traffic for no benefit.
    """
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    q = quadrics
    err = (
        q[:, 0] * x * x
        + 2.0 * q[:, 1] * x * y
        + 2.0 * q[:, 2] * x * z
        + 2.0 * q[:, 3] * x
        + q[:, 4] * y * y
        + 2.0 * q[:, 5] * y * z
        + 2.0 * q[:, 6] * y
        + q[:, 7] * z * z
        + 2.0 * q[:, 8] * z
        + q[:, 9]
    )
    # The metric is a sum of squares, so negatives are pure rounding noise.
    return np.maximum(err, 0.0)
