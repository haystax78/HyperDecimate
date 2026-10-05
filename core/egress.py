"""Flat NumPy arrays -> Blender mesh, plus attribute transfer. PLAN.md 5.3.

`from_pydata` is not viable at these counts. The fast recipe is to grow the
collections in one call each and then fill them with `foreach_set`.
"""

from __future__ import annotations

import numpy as np

from . import ingest, seams


def build_mesh(name, positions, triangles, calc_edges=True):
    """Create a new mesh datablock from a vertex array and a triangle array."""
    import bpy

    positions = np.ascontiguousarray(positions, dtype=np.float32)
    triangles = np.ascontiguousarray(triangles, dtype=np.int32)
    nv = positions.shape[0]
    nf = triangles.shape[0]

    mesh = bpy.data.meshes.new(name)
    mesh.vertices.add(nv)
    mesh.loops.add(nf * 3)
    mesh.polygons.add(nf)

    # Same asymmetry as on the read side, see core/ingest.py. Measured at
    # 10.24M verts and 61.4M corners, writing through the attribute API took
    # 3.24 s against 5.20 s through mesh.vertices / mesh.loops.
    pos_attr = mesh.attributes.get("position")
    if pos_attr is not None:
        pos_attr.data.foreach_set("vector", positions.reshape(-1))
    else:
        mesh.vertices.foreach_set("co", positions.reshape(-1))

    corner_attr = mesh.attributes.get(".corner_vert")
    if corner_attr is not None:
        corner_attr.data.foreach_set("value", triangles.reshape(-1))
    else:
        mesh.loops.foreach_set("vertex_index", triangles.reshape(-1))
    # Every face is a triangle, so loop starts are a stride-3 ramp. Blender
    # derives each face's loop count from the next face's start, so loop_total
    # does not need setting and is read-only in current versions.
    mesh.polygons.foreach_set(
        "loop_start", np.arange(nf, dtype=np.int32) * 3
    )

    mesh.update(calc_edges=calc_edges)
    return mesh


def set_shading(mesh, smooth):
    """Shade every face of `mesh` smooth or flat.

    Flat is a `sharp_face` attribute of all True, written in one call.
    `Mesh.shade_flat` sets it one face at a time from Python, which took
    0.5 s on a 650K-face result.
    """
    attr = mesh.attributes.get("sharp_face")
    if smooth:
        if attr is not None:
            mesh.attributes.remove(attr)
        return
    if attr is None:
        attr = mesh.attributes.new("sharp_face", 'BOOLEAN', 'FACE')
    attr.data.foreach_set("value", np.ones(len(mesh.polygons), dtype=bool))


def transfer_point_attributes(src_mesh, dst_mesh, remap, survived):
    """Copy point-domain generic attributes through the collapse mapping.

    A surviving vertex keeps its own value; there is nothing to blend, because a
    half-edge collapse does not invent positions and so should not invent
    attribute values either. Returns the list of names copied.
    """
    copied = []
    keep = np.flatnonzero(survived)
    order = np.argsort(remap[keep])
    src_index = keep[order]  # source vertex for each destination vertex, in order

    for attr in src_mesh.attributes:
        if attr.domain != "POINT" or attr.name == "position":
            continue
        # Dot-prefixed names are Blender's own internals (.select_vert and
        # friends). Blender maintains them on the new mesh itself, so copying
        # them by hand is at best redundant.
        if attr.name.startswith("."):
            continue
        spec = _ATTR_SPECS.get(attr.data_type)
        if spec is None:
            continue
        prop, width, dtype = spec

        n_src = len(src_mesh.vertices)
        # A four-byte integer attribute has no memcpy path in foreach_get, so it
        # goes the same way as the corner array; see core/ingest.py. INT8 and
        # BOOLEAN are one byte and fail that check, falling through to the loop.
        view = (ingest._int_attribute_view(attr.data, n_src)
                if attr.data_type == "INT" else None)
        if view is not None:
            src = np.array(view, dtype=dtype, copy=True)
        else:
            src = np.empty(n_src * width, dtype=dtype)
            attr.data.foreach_get(prop, src)
        src = src.reshape(-1, width) if width > 1 else src.reshape(-1, 1)

        dst_attr = dst_mesh.attributes.get(attr.name)
        if dst_attr is None:
            dst_attr = dst_mesh.attributes.new(
                attr.name, attr.data_type, "POINT"
            )
        dst_attr.data.foreach_set(prop, src[src_index].reshape(-1))
        copied.append(attr.name)
    return copied


def transfer_uv_layers(src_mesh, dst_mesh, tri_loops, face_origin, remap,
                       survived, seam_mask=None, tracked_uvs=None,
                       tracked_layer=None):
    """Copy UV layers onto the decimated mesh, per corner, keeping seams split.

    The unit a UV belongs to is the *corner*, not the vertex, and that distinction
    is the whole of this function. A vertex on a UV seam carries a different UV in
    each UV island meeting it, so handing every corner around that vertex one
    representative UV welds the islands together along the seam: the border on one
    side gets dragged onto the border opposite. Keeping the split means answering,
    for each output corner separately, which source corner it inherits from.

    That question has an exact answer, because a collapse never reorders corners.
    `topology.apply_remap` rewrites corner slots elementwise, the GPU's
    REMAP_CORNERS does the same, and compaction only ever drops whole faces. So
    output corner `k` of face `f` descends from corner `k` of source triangle
    `face_origin[f]`, which is source loop `tri_loops[face_origin[f], k]`.

    Two cases follow from that, and only the second involves a choice:

    * The corner still references the vertex that source loop belonged to. Its own
      UV is then exactly right, seam or not, and is used verbatim. Every corner of
      every *surviving* seam vertex lands here, which is what keeps a seam split.
    * The corner's vertex was collapsed into a survivor, which may itself hold
      several UVs; the one wanted is the one in the same island. It is taken as the
      survivor's UV nearest to the removed corner's own UV, which *is* the right
      island's UV at an adjacent vertex. The metric being the UV itself is what
      makes that safe rather than merely plausible: two candidates close in UV are
      interchangeable in the output whichever island they came from, so the only
      way to be visibly wrong is for a foreign island to fall nearer in UV than the
      vertex's own wedge, which takes a pathological layout.

    `seam_mask` is the per-vertex mask from `core.seams.read_seams`, passed in when
    the caller already has it. Only multi-UV vertices need the nearest-UV search,
    and knowing which they are up front saves a pass over every source corner.

    All of that hands each corner a UV the source already had, at whatever
    vertex it came from -- which is wrong wherever the collapse *moved* the
    vertex, and with optimal placement that is nearly every survivor. When the
    collapse tracked UVs (`tracked_uvs`, per output vertex, for the layer named
    `tracked_layer`), every corner of a vertex off a seam takes its tracked UV
    instead: the UV of the source surface where the vertex now is. Seam
    vertices only ever sit on an endpoint, so their source UVs stay exact, and
    the corner tracing above still decides them.
    """
    uv_layers = [la for la in src_mesh.uv_layers]
    if not uv_layers:
        return []

    n_src_loops = len(src_mesh.loops)
    # Two full corner reads, which is why `read_corner_verts` is worth the pointer
    # route it uses: 0.017 s each on a 26M-corner source against 0.92 s through
    # foreach_get. See the table in core/ingest.py.
    src_loop_vert = ingest.read_corner_verts(src_mesh)
    dst_tris = ingest.read_corner_verts(dst_mesh)

    keep = np.flatnonzero(survived)
    new_to_old = np.zeros(int(survived.sum()), dtype=np.int64)
    new_to_old[remap[keep]] = keep

    trace = _CornerUVs(src_loop_vert, len(src_mesh.vertices), tri_loops,
                       face_origin, new_to_old[dst_tris])

    copied = []
    uv_buf = np.empty(n_src_loops * 2, dtype=np.float32)
    for layer in uv_layers:
        uv = ingest.read_uvs(src_mesh, layer, uv_buf)

        dst_layer = dst_mesh.uv_layers.get(layer.name)
        if dst_layer is None:
            dst_layer = dst_mesh.uv_layers.new(name=layer.name, do_init=False)
        values = uv[trace.loops_for(uv, seam_mask)]
        if tracked_uvs is not None and layer.name == tracked_layer:
            values = values.copy()
            multi = trace._multi_uv(uv, seam_mask)
            free = ~multi[trace.dst_orig_vert]
            values[free] = np.asarray(tracked_uvs, dtype=np.float32)[
                dst_tris[free]]
        _uv_write(dst_mesh, dst_layer, values.reshape(-1))
        copied.append(layer.name)
    return copied


def corner_uv_loops(src_loop_vert, vertex_count, tri_loops, face_origin,
                    dst_orig_vert, uv, seam_mask=None):
    """`transfer_uv_layers`' corner tracing, without Blender. For tests."""
    return _CornerUVs(src_loop_vert, vertex_count, tri_loops, face_origin,
                      dst_orig_vert).loops_for(uv, seam_mask)


class _CornerUVs:
    """Which source loop each output corner takes its UV from.

    Pure NumPy, and sized by the output rather than the source apart from a couple
    of linear passes over the source corner array. See `transfer_uv_layers` for
    why the tracing is exact and where the one choice is made.

    src_loop_vert: (L,) source vertex per source loop.
    tri_loops:     (F_src, 3) source loop per source triangle corner.
    face_origin:   (F_out,) source triangle each output triangle came from.
    dst_orig_vert: (F_out * 3,) *original* vertex index of each output corner.
    """

    def __init__(self, src_loop_vert, vertex_count, tri_loops, face_origin,
                 dst_orig_vert):
        # Left at whatever width they came in at. These are the source-sized
        # arrays, and widening 26M corners to int64 just to index with them costs
        # a 200 MB copy for nothing; NumPy indexes happily with int32.
        self.src_loop_vert = np.asarray(src_loop_vert).reshape(-1)
        self.vertex_count = int(vertex_count)
        self.dst_orig_vert = np.asarray(dst_orig_vert, dtype=np.int64).reshape(-1)

        # Gather first, widen after: the other way round materialises the whole
        # source triangle-to-loop table at double width.
        tri_loops = np.asarray(tri_loops).reshape(-1, 3)
        face_origin = np.asarray(face_origin).reshape(-1)
        self.origin_loop = tri_loops[face_origin].reshape(-1).astype(np.int64)

        # Corners whose vertex is no longer the one their source loop belonged
        # to. Only these need a UV chosen for them; every other corner inherits
        # its own, exactly.
        self.moved = np.flatnonzero(
            self.src_loop_vert[self.origin_loop] != self.dst_orig_vert)

        # Lowest-indexed loop per source vertex, by reverse scatter. When fancy
        # indexing repeats an index the last write wins, so walking backwards
        # leaves the lowest loop index in place. That is O(n) and beats
        # np.unique(..., return_index=True), which sorts 26M elements to learn
        # the same thing.
        self.rep_loop = np.zeros(self.vertex_count, dtype=np.int64)
        self.rep_loop[self.src_loop_vert[::-1]] = np.arange(
            self.src_loop_vert.size - 1, -1, -1, dtype=np.int64
        )

    def loops_for(self, uv, seam_mask=None):
        """(F_out * 3,) source loop index per output corner, for one UV layer."""
        out = self.origin_loop.copy()
        if self.moved.size == 0:
            return out

        # A moved corner's default is any loop of its survivor, which is already
        # exact wherever that survivor carries a single UV.
        survivor = self.dst_orig_vert[self.moved]
        out[self.moved] = self.rep_loop[survivor]

        multi = self._multi_uv(uv, seam_mask)
        query = self.moved[multi[survivor]]
        if query.size == 0:
            return out

        cand, offsets = self._candidates(multi)
        q_vert = self.dst_orig_vert[query]
        counts = offsets[q_vert + 1] - offsets[q_vert]
        if not counts.any():
            return out

        # Expand each query over its candidate loops. np.repeat lays the groups
        # out contiguously and in order, so a lexsort by (group, distance) leaves
        # each group's nearest candidate first.
        group = np.repeat(np.arange(query.size, dtype=np.int64), counts)
        starts = np.zeros(query.size + 1, dtype=np.int64)
        np.cumsum(counts, out=starts[1:])
        within = np.arange(group.size, dtype=np.int64) - starts[group]
        loops = cand[offsets[q_vert][group] + within]

        want = uv[self.origin_loop[query]][group]
        dist = np.abs(uv[loops] - want).sum(axis=1)
        order = np.lexsort((dist, group))
        sorted_group = group[order]
        first = np.concatenate(
            ([0], np.flatnonzero(np.diff(sorted_group)) + 1))
        out[query[sorted_group[first]]] = loops[order[first]]
        return out

    def _multi_uv(self, uv, seam_mask):
        """Per-vertex: does this vertex carry more than one UV?

        A mask supplied by the caller is the union across every UV layer, which is
        safe to use for one layer: a vertex wrongly included just gets a
        nearest-UV search among loops that all agree, and so the same answer.
        """
        if seam_mask is not None:
            mask = np.asarray(seam_mask, dtype=bool).reshape(-1)
            if mask.size == self.vertex_count:
                return mask
        return seams.seam_vertices_from_uvs(self.src_loop_vert, uv,
                                            self.vertex_count)

    def _candidates(self, multi):
        """Source loops grouped by vertex, multi-UV vertices only, as CSR.

        Restricting to those vertices is what keeps the sort affordable. A seam is
        a curve through the mesh, so this is thousands of loops where sorting all
        of them would be tens of millions, and a full-mesh sort here would cost
        more than the decimation it is supporting.
        """
        sel = np.flatnonzero(multi[self.src_loop_vert])
        verts = self.src_loop_vert[sel]
        cand = sel[np.argsort(verts, kind="stable")]
        offsets = np.zeros(self.vertex_count + 1, dtype=np.int64)
        np.cumsum(np.bincount(verts, minlength=self.vertex_count),
                  out=offsets[1:])
        return cand, offsets


def _uv_write(mesh, layer, values):
    """The write half of `ingest.read_uvs`, which decides the attribute route."""
    attr = ingest.uv_attribute(mesh, layer)
    if attr is not None:
        attr.data.foreach_set("vector", values)
    else:
        layer.uv.foreach_set("vector", values)


# data_type -> (property name, components, numpy dtype)
_ATTR_SPECS = {
    "FLOAT": ("value", 1, np.float32),
    "INT": ("value", 1, np.int32),
    "INT8": ("value", 1, np.int32),
    "BOOLEAN": ("value", 1, np.int32),
    "FLOAT2": ("vector", 2, np.float32),
    "FLOAT_VECTOR": ("vector", 3, np.float32),
    "FLOAT_COLOR": ("color", 4, np.float32),
    "BYTE_COLOR": ("color", 4, np.float32),
    "QUATERNION": ("value", 4, np.float32),
}
