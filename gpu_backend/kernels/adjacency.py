"""GLSL sources for the adjacency and quadric stages. PLAN.md 4.3, 4.4.

Kept as Python strings rather than separate .glsl files so the defines, the
prelude and the resource names stay in one place, and so a typo is a Python error
rather than a shader compile error pointing into Blender's own prelude.

No resource here may be named `flat`, `smooth`, `texture`, `sample`, `filter` or
`main`; `context.GLSL_RESERVED` enforces that.
"""

# Maximum one-ring size a kernel will gather into registers. Two arrays of this
# length live in registers per thread, so it is a real occupancy cost and not a
# free constant. 24 matches the valence guard in PLAN.md 4.7, so any ring that
# does not fit is one the guard would have rejected anyway.
#
# Overflow is reported rather than clamped. Silently truncating a ring would make
# the link condition and the boundary test quietly wrong on exactly the messy
# vertices where they matter most.
MAX_RING = 24
RING_OVERFLOW = 0xFFFFFFFF


# ---------------------------------------------------------------- shared helpers

# The adjacency stores corner indices, so a row gives both the faces incident to
# a vertex and which of each face's three slots that vertex occupies. Both matter:
# the slot yields the other two vertices *in winding order*, which the normal-flip
# test depends on.
#
# RING_BEGIN opens its own block so the same function can walk two rings without
# redeclaring anything.
RING = """
#define MAX_RING %(max_ring)d
#define RING_OVERFLOW 0xffffffffu

#define RING_BEGIN(v) { \\
  uint _lo = imageLoad(offs, IDX2(v)).r; \\
  uint _hi = imageLoad(offs, IDX2(v + 1u)).r; \\
  for (uint _k = _lo; _k < _hi; _k++) { \\
    uint _corner = imageLoad(adj, IDX2(_k)).r; \\
    uint face = _corner / 3u; \\
    uint slot = _corner - face * 3u; \\
    uvec3 tri = uvec3(imageLoad(c0, IDX2(face)).r, \\
                      imageLoad(c1, IDX2(face)).r, \\
                      imageLoad(c2, IDX2(face)).r);

#define RING_END } }

uvec2 ring_others(uvec3 tri, uint slot) {
  if (slot == 0u) return uvec2(tri.y, tri.z);
  if (slot == 1u) return uvec2(tri.z, tri.x);
  return uvec2(tri.x, tri.y);
}

// Gather the neighbours of v into registers, with the count of incident faces
// each one shares with v. One image pass instead of the quadratic re-walk a naive
// boundary test would do, which matters because every ring kernel needs this.
uint ring_gather(uint v, out uint nbr[MAX_RING], out uint shared_faces[MAX_RING]) {
  uint count = 0u;
  bool overflow = false;
  RING_BEGIN(v)
    uvec2 o = ring_others(tri, slot);
    for (int w = 0; w < 2; w++) {
      uint u = (w == 0) ? o.x : o.y;
      bool found = false;
      for (uint i = 0u; i < count; i++) {
        if (nbr[i] == u) { shared_faces[i] += 1u; found = true; break; }
      }
      if (!found) {
        if (count >= uint(MAX_RING)) { overflow = true; }
        else {
          nbr[count] = u;
          shared_faces[count] = 1u;
          count++;
        }
      }
    }
  RING_END
  return overflow ? RING_OVERFLOW : count;
}
""" % {"max_ring": MAX_RING}

# `pos` is layered: layer 0 the position, layer 1 the vertex's UV.
VPOS = """
vec3 vpos(uint v) { return imageLoad(pos, IDX3(v, 0)).xyz; }
vec2 vuv(uint v) { return imageLoad(pos, IDX3(v, 1)).xy; }
"""

# Evaluating v^T Q v from the layered quadric image. Layout, 12 floats over 3
# layers, of which 11 are used:
#   layer 0: (q0, q1, q2, q3)   layer 1: (q4, q5, q6, q7)   layer 2: (q8, q9, -, -)
QUADRIC_EVAL = """
void quad_load(uint v, out vec4 a, out vec4 b, out vec4 c) {
  a = imageLoad(quad, IDX3(v, 0));
  b = imageLoad(quad, IDX3(v, 1));
  c = imageLoad(quad, IDX3(v, 2));
}

// The accumulated plane weight, recovered from the quadric itself: every plane
// is added with a unit normal, so the 3x3 trace is the sum of the weights. In
// this packing the diagonal is a.x, b.x and b.w. See core.quadrics.quadric_weight.
float quad_weight(vec4 a, vec4 b) {
  return a.x + b.x + b.w;
}

float quad_error(vec4 a, vec4 b, vec4 c, vec3 p) {
  float e = a.x * p.x * p.x + 2.0 * a.y * p.x * p.y + 2.0 * a.z * p.x * p.z
          + 2.0 * a.w * p.x
          + b.x * p.y * p.y + 2.0 * b.y * p.y * p.z + 2.0 * b.z * p.y
          + b.w * p.z * p.z
          + 2.0 * c.x * p.z + c.y;
  return max(e, 0.0);
}
"""


# ------------------------------------------------------------------- the kernels

# Sort each CSR row by corner id, so the adjacency is a function of the mesh
# alone rather than of the scatter's scheduling.
#
# The scatter fills a row through atomic cursors, so the order inside it
# follows whichever thread got there first. Every consumer treats a row as a
# set, so that was harmless -- until the placement moved onto the device, and
# with it two sums that do care: the quadric build adds a row's faces in row
# order, and the candidate kernel breaks cost ties by first-found among
# neighbours. Two runs of one mesh could then differ in the last float bit of
# a quadric, or in which of two equal-cost targets a vertex picks, and the
# results came apart -- the reference pipeline solved the same problem by
# sorting its rings for exactly this reason.
#
# An insertion sort over the row in memory: rings are short, the walk is a
# handful of swaps, and a row longer than the valence guard would allow is
# already a vertex the candidate kernel rejects.
SORT_RINGS = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  uint lo = imageLoad(offs, IDX2(v)).r;
  uint hi = imageLoad(offs, IDX2(v + 1u)).r;
  for (uint i = lo + 1u; i < hi; i++) {
    uint x = imageLoad(adj, IDX2(i)).r;
    uint j = i;
    while (j > lo && imageLoad(adj, IDX2(j - 1u)).r > x) {
      imageStore(adj, IDX2(j), uvec4(imageLoad(adj, IDX2(j - 1u)).r));
      j--;
    }
    imageStore(adj, IDX2(j), uvec4(x));
  }
}
"""

REMAP_CORNERS = """
void main() {
  uint f = GID;
  if (f >= uint(n)) return;
  uint a = imageLoad(remap, IDX2(imageLoad(c0, IDX2(f)).r)).r;
  uint b = imageLoad(remap, IDX2(imageLoad(c1, IDX2(f)).r)).r;
  uint c = imageLoad(remap, IDX2(imageLoad(c2, IDX2(f)).r)).r;
  imageStore(c0, IDX2(f), uvec4(a));
  imageStore(c1, IDX2(f), uvec4(b));
  imageStore(c2, IDX2(f), uvec4(c));
  bool dead = (a == b) || (b == c) || (c == a);
  imageStore(alive, IDX2(f), uvec4(dead ? 0u : 1u));
}
"""

# Kernel 3: one atomic per corner into a per-vertex counter. The cheap atomic
# pass; the scatter below is the expensive one.
ADJ_COUNT = """
void main() {
  uint f = GID;
  if (f >= uint(n)) return;
  if (imageLoad(alive, IDX2(f)).r == 0u) return;
  imageAtomicAdd(counts, IDX2(imageLoad(c0, IDX2(f)).r), 1u);
  imageAtomicAdd(counts, IDX2(imageLoad(c1, IDX2(f)).r), 1u);
  imageAtomicAdd(counts, IDX2(imageLoad(c2, IDX2(f)).r), 1u);
}
"""

# Kernel 5: fill the CSR corner list. Each corner claims a slot by bumping its
# vertex's cursor, so order within a row follows the scheduler; SORT_RINGS,
# which runs straight after this, puts it back to corner order so the row is a
# function of the mesh alone.
ADJ_SCATTER = """
void main() {
  uint f = GID;
  if (f >= uint(n)) return;
  if (imageLoad(alive, IDX2(f)).r == 0u) return;
  uint v0 = imageLoad(c0, IDX2(f)).r;
  uint v1 = imageLoad(c1, IDX2(f)).r;
  uint v2 = imageLoad(c2, IDX2(f)).r;
  uint s0 = imageLoad(offs, IDX2(v0)).r + imageAtomicAdd(cursor, IDX2(v0), 1u);
  uint s1 = imageLoad(offs, IDX2(v1)).r + imageAtomicAdd(cursor, IDX2(v1), 1u);
  uint s2 = imageLoad(offs, IDX2(v2)).r + imageAtomicAdd(cursor, IDX2(v2), 1u);
  imageStore(adj, IDX2(s0), uvec4(f * 3u + 0u));
  imageStore(adj, IDX2(s1), uvec4(f * 3u + 1u));
  imageStore(adj, IDX2(s2), uvec4(f * 3u + 2u));
}
"""

# Quadric accumulation, gathered rather than scattered.
#
# Garland-Heckbert scatters each triangle's quadric onto its three vertices, which
# needs float atomics. Those are not portable, so each vertex instead gathers the
# faces it already knows through the adjacency and sums them itself. Same result,
# no atomics, and it reuses the ring walk.
#
# Boundary planes come free from the same gather: an edge (v, u) is on the
# boundary exactly when only one of v's faces contains u, and ring_gather has
# already counted that. No global edge table is needed.
BUILD_QUADRIC = """
void accumulate(inout vec4 a0, inout vec4 a1, inout vec4 a2,
                vec3 nrm, float d, float w) {
  a0 += w * vec4(nrm.x * nrm.x, nrm.x * nrm.y, nrm.x * nrm.z, nrm.x * d);
  a1 += w * vec4(nrm.y * nrm.y, nrm.y * nrm.z, nrm.y * d, nrm.z * nrm.z);
  a2 += w * vec4(nrm.z * d, d * d, 0.0, 0.0);
}

void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  vec3 p = vpos(v);
  vec4 a0 = vec4(0.0), a1 = vec4(0.0), a2 = vec4(0.0);

  // Surface quadrics, weighted by twice the triangle area -- the length of
  // the cross product itself -- which is the reference implementation's own
  // weighting and the same shape as Garland & Heckbert's area weighting.
  RING_BEGIN(v)
    uvec2 o = ring_others(tri, slot);
    vec3 cr = cross(vpos(o.x) - p, vpos(o.y) - p);
    float len = length(cr);
    if (len > 0.0) {
      vec3 nrm = cr / len;
      accumulate(a0, a1, a2, nrm, -dot(nrm, p), len);
    }
  RING_END

  // Boundary and seam constraint planes.
  //
  // vflags is both input and output here, which is how the seam flag reaches this
  // kernel without a ninth image slot:
  //   bit 0  boundary, written below
  //   bit 1  UV seam, uploaded before this runs and preserved
  // Reading a neighbour's bit 1 is safe against the concurrent writes because
  // those preserve bit 1 unchanged.
  uint seam_v = (imageLoad(vflags, IDX2(v)).r >> 1u) & 1u;

  uint nbr[MAX_RING];
  uint shared_faces[MAX_RING];
  uint count = ring_gather(v, nbr, shared_faces);
  uint boundary = 0u;
  // A ring that did not fit in registers comes back as RING_OVERFLOW: looping
  // to that count is a device-lost hang, not a skip. The same kernel answers
  // the question for border.py, which takes the conservative side and calls
  // such a vertex constrained; so does this one, and it matches what the
  // candidate pass will do with it anyway (reject on the same overflow).
  if (count == RING_OVERFLOW) {
    boundary = 1u;
    count = 0u;
  }
  for (uint i = 0u; i < count; i++) {
    uint u = nbr[i];
    bool on_boundary = (shared_faces[i] == 1u);
    // The seam rule is per-vertex, so "both ends are seam vertices" stands in for
    // "this edge is a seam edge". core/seams.py explains why and what it costs.
    bool on_seam = (seam_v == 1u)
                && (((imageLoad(vflags, IDX2(u)).r >> 1u) & 1u) == 1u);
    if (on_boundary) boundary = 1u;
    if (!on_boundary && !on_seam) continue;
    vec3 edge = vpos(u) - p;
    // Any incident face gives a usable normal for the perpendicular plane.
    vec3 face_n = vec3(0.0);
    RING_BEGIN(v)
      if (tri.x == u || tri.y == u || tri.z == u) {
        uvec2 o = ring_others(tri, slot);
        face_n = cross(vpos(o.x) - p, vpos(o.y) - p);
      }
    RING_END
    vec3 cn = cross(edge, face_n);
    float cl = length(cn);
    if (cl > 0.0) {
      vec3 nrm = cn / cl;
      accumulate(a0, a1, a2, nrm, -dot(nrm, p),
                 BOUNDARY_WEIGHT * dot(edge, edge));
    }
  }

  imageStore(quad, IDX3(v, 0), a0);
  imageStore(quad, IDX3(v, 1), a1);
  imageStore(quad, IDX3(v, 2), a2);
  imageStore(vflags, IDX2(v), uvec4(boundary | (seam_v << 1u)));
}
"""
