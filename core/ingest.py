"""Blender mesh -> flat NumPy arrays. PLAN.md 5.1.

The only rule that matters here: never touch a `bpy_prop_collection` element by
element. Every read goes through `foreach_get` into a preallocated array.

The second rule: avoid `calc_loop_triangles()`. On a 50M-triangle mesh
tessellation costs seconds, and for the meshes this addon exists to handle it is
usually unnecessary, because sculpt output is all triangles or all quads. See
`triangulation_method` for how that is detected.
"""

from __future__ import annotations

import ctypes

import numpy as np

# How the triangle list was obtained, for reporting and for the benchmark.
TRIS_DIRECT = "all-triangles (free)"
QUADS_SPLIT = "all-quads, split (free)"
TESSELLATED = "tessellated via calc_loop_triangles (slow)"


def triangulation_method(mesh):
    """Classify the mesh without tessellating it.

    Every polygon has at least three loops, so a total loop count of exactly
    3 x polygons forces every polygon to be a triangle. The same argument gives
    the all-quad case. Anything else has at least one n-gon or a mix.
    """
    polys = len(mesh.polygons)
    loops = len(mesh.loops)
    if polys == 0:
        return TRIS_DIRECT
    if loops == 3 * polys:
        return TRIS_DIRECT
    if loops == 4 * polys:
        return QUADS_SPLIT
    return TESSELLATED


# Blender's `foreach_get` has a memcpy fast path for float-vector attributes but
# not for the older per-element RNA properties, and not for integer scalars.
# Measured on a 10.24M-vertex, 61.4M-corner mesh:
#
#   positions via attributes['position'].foreach_get('vector')     0.02 s
#   positions via mesh.vertices.foreach_get('co')                  0.26 s
#   corner verts via attributes['.corner_vert'].foreach_get(...)   2.10 s
#   corner verts via mesh.loops.foreach_get('vertex_index')        3.32 s
#   corner verts via mesh.polygons.foreach_get('vertices')         3.49 s
#
# So positions are effectively free through the attribute API, 13x faster, while
# integer corner data has no fast path anywhere and costs about 34 ns per element
# whichever route is used. The attribute route is still 1.6x better, so use it.
#
# `.corner_vert` is an internal attribute name, so every fast path here is
# attempted and silently falls back to the public API if it is missing.
#
# That 34 ns per element is not Blender's floor, though, only `foreach_get`'s.
# An attribute's values live in one contiguous C array, and `as_pointer()` on an
# element hands back its address, so the whole span can be wrapped as a NumPy
# array and copied out in one memcpy. Re-measured on the 26.08M-corner head scan
# under Blender 5.2:
#
#   corner verts via attributes['.corner_vert'].foreach_get(...)    0.922 s
#   corner verts via as_pointer + NumPy view + copy                 0.017 s
#   corner verts via mesh.loops.foreach_get('vertex_index')         1.451 s
#
# 55x, and bit-identical. Float attributes gain nothing -- they already have the
# memcpy path, and the pointer route measured 0.8x on positions -- so this is used
# for integer attributes only, where there is no alternative.
#
# It does read Blender's own memory through an address, which the Python API makes
# no promises about. `_int_attribute_view` therefore proves the layout before
# trusting it -- the stride between the first two elements, the distance to the
# last element, and the values at both ends and the middle against RNA -- and
# returns None on any surprise, which puts the caller back on `foreach_get`. The
# view is copied immediately and never retained, because it aliases memory Blender
# may move or free at any time. Set FAST_INT_READ = False to take the slow route
# everywhere, which is the first thing to try if a mesh ever reads back wrong.
FAST_INT_READ = True


def _int_attribute_view(data, count, ctype=None):
    """Zero-copy NumPy view over an integer attribute's own storage, or None.

    `data` is the attribute's `.data` collection and `count` its length. None
    means the layout was not exactly what this assumes and the caller must fall
    back to `foreach_get`. The result aliases Blender's memory: copy it, do not
    keep it.
    """
    if not FAST_INT_READ or count < 2:
        return None
    try:
        ctype = ctype or ctypes.c_int32
        itemsize = ctypes.sizeof(ctype)
        base = data[0].as_pointer()

        # Consecutive elements one itemsize apart, and the last element exactly
        # count-1 strides along, which together rule out both a different element
        # size (INT8 and BOOLEAN attributes are one byte, not four) and storage
        # that is not one flat array.
        if data[1].as_pointer() - base != itemsize:
            return None
        if data[count - 1].as_pointer() - base != itemsize * (count - 1):
            return None

        view = np.ctypeslib.as_array(
            ctypes.cast(base, ctypes.POINTER(ctype)), shape=(count,))
        for i in (0, 1, count // 2, count - 2, count - 1):
            if int(view[i]) != int(data[i].value):
                return None
        return view
    except Exception:  # noqa: BLE001 - any failure means use foreach_get
        return None


def read_positions(mesh):
    """(V, 3) float32 vertex positions."""
    n = len(mesh.vertices)
    buf = np.empty(n * 3, dtype=np.float32)
    attr = mesh.attributes.get("position")
    if attr is not None:
        attr.data.foreach_get("vector", buf)
    else:
        mesh.vertices.foreach_get("co", buf)
    return buf.reshape(n, 3)


def uv_attribute(mesh, layer):
    """The CORNER/FLOAT2 attribute behind a UV layer, or None.

    `layer.uv` is the public route and works everywhere, but it is the slow
    per-element one; the attribute of the same name has the memcpy path. A UV
    layer is not guaranteed to have a matching attribute, and a same-named
    attribute is not guaranteed to be the right domain or type, so this is the
    one place that decides -- `read_uvs` here and `egress` both go through it
    rather than each repeating the rule and drifting apart.
    """
    attr = mesh.attributes.get(layer.name)
    if (attr is not None and attr.domain == "CORNER"
            and attr.data_type == "FLOAT2"):
        return attr
    return None


def read_uvs(mesh, layer, out=None):
    """(n_loops, 2) float32 UVs for one layer, into `out` if given."""
    n = len(mesh.loops)
    if out is None:
        out = np.empty(n * 2, dtype=np.float32)
    attr = uv_attribute(mesh, layer)
    if attr is not None:
        attr.data.foreach_get("vector", out)
    else:
        layer.uv.foreach_get("vector", out)
    return out.reshape(n, 2)


def read_corner_verts(mesh):
    """(n_loops,) int32 vertex index per face corner.

    The hottest read in the addon: the ingest needs it, and `transfer_uv_layers`
    needs it again for both the source and the decimated mesh.
    """
    n = len(mesh.loops)
    attr = mesh.attributes.get(".corner_vert")
    if attr is not None:
        view = _int_attribute_view(attr.data, n)
        if view is not None:
            return np.array(view, dtype=np.int32, copy=True)
        buf = np.empty(n, dtype=np.int32)
        attr.data.foreach_get("value", buf)
        return buf
    buf = np.empty(n, dtype=np.int32)
    mesh.loops.foreach_get("vertex_index", buf)
    return buf


def read_triangles(mesh, method=None):
    """(F, 3) int32 triangles, plus (F, 3) int32 originating loop indices.

    The loop indices are what makes corner-domain attribute transfer possible
    later, and they are free to produce here, so they are always returned.
    """
    method = method or triangulation_method(mesh)
    polys = len(mesh.polygons)

    if method == TRIS_DIRECT:
        # The corner array *is* the triangle array: corners are stored in face
        # order, three per face. No tessellation, no extra pass.
        tris = read_corner_verts(mesh).reshape(polys, 3)
        loops = np.arange(polys * 3, dtype=np.int32).reshape(polys, 3)
        return tris, loops, method

    if method == QUADS_SPLIT:
        quad = read_corner_verts(mesh).reshape(polys, 4)
        # Split each quad along the 0-2 diagonal, matching Blender's own choice
        # closely enough for decimation purposes.
        #
        # Written column by column into one preallocated array rather than built
        # with fancy indexing and `concatenate`, which materialised both halves
        # and then copied them again. Same output, same face order -- all the
        # (0,1,2) triangles then all the (0,2,3) ones -- for a third less time and
        # without the 1 GB of temporaries a 26M-quad mesh used to need. The loop
        # indices are pure arithmetic, 4i + k, so they need no source array
        # either.
        tris = np.empty((2 * polys, 3), dtype=np.int32)
        for col, src in enumerate((0, 1, 2)):
            tris[:polys, col] = quad[:, src]
        for col, src in enumerate((0, 2, 3)):
            tris[polys:, col] = quad[:, src]

        loops = np.empty((2 * polys, 3), dtype=np.int32)
        base = np.arange(0, polys * 4, 4, dtype=np.int32)
        for col, off in enumerate((0, 1, 2)):
            loops[:polys, col] = base + off if off else base
        for col, off in enumerate((0, 2, 3)):
            loops[polys:, col] = base + off if off else base
        return tris, loops, method

    # n-gons present: fall back to Blender's tessellator.
    mesh.calc_loop_triangles()
    n = len(mesh.loop_triangles)
    tris = np.empty(n * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", tris)
    loops = np.empty(n * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("loops", loops)
    return tris.reshape(n, 3), loops.reshape(n, 3), method


def read_mesh(mesh):
    """(positions, triangles, tri_loops, method)."""
    method = triangulation_method(mesh)
    positions = read_positions(mesh)
    tris, loops, method = read_triangles(mesh, method)
    return positions, tris, loops, method


# ----------------------------------------------------------------- weight maps

def read_vertex_group_weights(obj, group_name, default=1.0):
    """(V,) float32 weights from a vertex group, or None if it is absent.

    Vertex groups have no bulk accessor, so this is the one place a per-vertex
    Python loop is unavoidable. It is only reached when the user actually picks a
    group, and the cost is reported by the caller.
    """
    group = obj.vertex_groups.get(group_name)
    if group is None:
        return None
    mesh = obj.data
    weights = np.full(len(mesh.vertices), default, dtype=np.float32)
    index = group.index
    for i, vert in enumerate(mesh.vertices):
        for g in vert.groups:
            if g.group == index:
                weights[i] = g.weight
                break
    return weights


def read_color_attribute_luminance(mesh, name):
    """(V,) float32 luminance of a colour attribute, or None if unusable.

    This is how a painted colour attribute steers local resolution. Only
    point-domain colour is handled; a corner-domain layer is averaged onto points
    by the caller if needed, which is not implemented yet.
    """
    layer = mesh.color_attributes.get(name) if name else None
    if layer is None or layer.domain != "POINT":
        return None
    n = len(mesh.vertices)
    buf = np.empty(n * 4, dtype=np.float32)
    layer.data.foreach_get("color", buf)
    rgba = buf.reshape(n, 4)
    # Rec. 709 luma, which matches how the value reads on screen.
    return (
        0.2126 * rgba[:, 0] + 0.7152 * rgba[:, 1] + 0.0722 * rgba[:, 2]
    ).astype(np.float32)


def density_from_weights(weights, strength):
    """Turn a 0..1 weight map into a cost multiplier.

    A weight of 0 leaves the cost untouched, a weight of 1 multiplies it by
    `strength`, so painted areas resist collapse and keep their resolution.
    Interpolation is geometric rather than linear because cost spans orders of
    magnitude.
    """
    if weights is None:
        return None
    w = np.clip(weights.astype(np.float32), 0.0, 1.0)
    return np.exp(w * np.log(max(strength, 1.0))).astype(np.float32)
