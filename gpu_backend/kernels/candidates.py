"""Candidate selection and the independent set. PLAN.md 4.5, 4.6, 4.7.

Slot budget notes, because this is where the 8-image limit actually bites.

The candidate kernel needs c0, c1, c2, offs, adj, pos, quad and cand. That is
exactly 8, so the per-vertex lock flag and density multiplier have nowhere to go
as arrays of their own. They ride in the two unused channels of the quadric's
third layer instead: a quadric is 10 floats spread over 3 RGBA32F layers, which
leaves layer 2's .z and .w free. `WRITE_PARAMS` fills them once at setup.

    quad layer 0: (q0, q1, q2, q3)
    quad layer 1: (q4, q5, q6, q7)
    quad layer 2: (q8, q9, state, density)

`state` is the bit pattern of a small flag set, stored as its own uint bits:

    bit 0  locked outright (user lock, or a border under freeze_borders)
    bit 1  on a UV seam; slides along it, never off it
    bit 2  on an open border; slides along it, never off it

Bits, not a small enum, because the placement rules below need each class on
its own and because the border bit is inherited by the survivor of a collapse
(see MERGE_QUADRICS), which an enum cannot express in one channel.

`cand` is likewise packed, holding its outputs in one layered RGBA32F image so
it still costs a single image unit:

    layer 0: (best cost, best target as uintBitsToFloat -- NO_TARGET when
              there is none, rejection reason, faces shared with the target)
    layer 1: (placement x, y, z, swap flag)
    layer 2: (placement u, v, -, -), written when UVs are tracked

The placement is part of the candidate rather than a post-process, which is the
reference pipeline's own arrangement: the cost is evaluated where the merged
vertex will actually go, the flip test checks the geometry the collapse will
really produce, and the record is what the collapse log replays later.
"""

NO_TARGET = 0xFFFFFFFF

# The claim array holds, per *face*, the cheapest collapse that wants it:
# layer 0 its monotone cost key, layer 1 the owner's vertex id among key
# winners. Two layers keep it inside one image unit.
CLAIM_KEY_CLEAR = 0xFFFFFFFF

# Fold the boundary, lock, seam and density inputs into the quadric's spare
# channels. Runs once, so it can afford its own kernel.
WRITE_PARAMS = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  vec4 q2 = imageLoad(quad, IDX3(v, 2));
  uint flags = imageLoad(vflags, IDX2(v)).r;
  uint on_boundary = flags & 1u;
  uint on_seam = (flags >> 1u) & 1u;
  uint user_lock = imageLoad(vlock, IDX2(v)).r;

  uint state = (on_boundary != 0u ? 4u : 0u) | (on_seam != 0u ? 2u : 0u);
  if (user_lock != 0u || (FREEZE_BORDERS != 0 && on_boundary != 0u))
    state |= 1u;
  float density = max(imageLoad(vdensity, IDX2(v)).r, 1e-6);
  imageStore(quad, IDX3(v, 2), vec4(q2.x, q2.y, uintBitsToFloat(state), density));
  // One texel into a scratch array, so the host has something this dispatch
  // wrote that it can read back to wait on it. `quad` is layered and cannot be
  // read directly, and reading anything this dispatch did *not* write does not
  // wait at all; see context.sync.
  if (v == 0u) imageStore(probe_dst, IDX2(0u), uvec4(1u));
}
"""

# Kernel 6. Cheapest target per vertex, with every validity test.
#
# The ordering matters and mirrors the CPU reference exactly: pick the minimum
# cost over the one-ring, then run the validity tests on the winner alone.
# Testing every candidate would cost six times as much, and the reference's
# semantics are that a vertex whose best collapse is invalid simply sits out
# this pass, so matching it keeps the two backends diffable.
#
# The cost itself is the reference's: the pair quadric Q_v + Q_w evaluated at
# the point the merged vertex will actually occupy -- raw, with no weight
# normalisation, so the histogram's log range and the claims' ordering see
# exactly the figure the reference scores a candidate with.
#
# The placement rules, in the reference's own order:
#   * both ends locked: the pair can never merge;
#   * one end locked: the merged vertex sits on it -- a lock fixes the
#     position, not the vertex id, so a locked source may still die;
#   * both ends flagged (border or seam at both): the cheaper endpoint,
#     so a border pair slides along its own polyline;
#   * one end flagged: that end, for the same reason;
#   * otherwise: the minimiser of the pair quadric, or the edge midpoint when
#     the system is too near singular to trust (and when optimal placement
#     is switched off, always the midpoint -- the reference's --centroid
#     control).
#
# And the pair rules the reference applies before the tests: two border
# vertices merge only along the boundary itself (an edge shared by exactly
# one face), and an edge shared by three or more faces -- a weld, as the
# clustering stage can leave behind -- is refused outright.
CANDIDATE = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;

  // No candidate, until proven otherwise.
  // cand layer 0 .z carries a rejection reason, purely for diagnostics:
  // 1 locked, 2 ring empty or over MAX_RING, 3 no legal target,
  // 4 valence guard, 5 link condition, 6 normal flip, 7 shared-face count,
  // 8 UV fold, 0 accepted.
  // layer 0 .w carries the faces shared with the chosen target, which the
  // validity tests compute and the collapse log needs.
  #define REJECT(code) { \
    imageStore(cand, IDX3(v, 0), vec4(1e30, uintBitsToFloat(NO_TARGET_U), \
                                      float(code), 0.0)); \
    imageStore(cand, IDX3(v, 1), vec4(0.0)); \
    return; }
  imageStore(cand, IDX3(v, 0), vec4(1e30, uintBitsToFloat(NO_TARGET_U), 0.0, 0.0));
  imageStore(cand, IDX3(v, 1), vec4(0.0));

  vec4 qa, qb, qc;
  quad_load(v, qa, qb, qc);
  uint st_v = floatBitsToUint(qc.z);
  bool v_locked = (st_v & 1u) != 0u;
  bool v_border = (st_v & 4u) != 0u;
  bool v_flagged = (st_v & 6u) != 0u;  // border or seam, from the kept flags
  float density = max(qc.w, 1e-6);

  uint nbr[MAX_RING];
  uint shared_faces[MAX_RING];
  uint count = ring_gather(v, nbr, shared_faces);
  // Overflowed the register window, so the ring is larger than any collapse we
  // would allow anyway. Sitting out is both safe and correct.
  if (count == 0u || count == RING_OVERFLOW) REJECT(2)

  // --- cheapest target ------------------------------------------------
  vec3 p = vpos(v);
  float best = 1e30;
  uint tgt = NO_TARGET_U;
  uint best_i = 0u;
  uint best_hash = 0u;
  uint best_st = 0u;
  vec3 best_np = vec3(0.0);
  for (uint i = 0u; i < count; i++) {
    uint w = nbr[i];
    // The target's own state rides in the quadric it was just loaded from.
    vec4 wa, wb, wc;
    quad_load(w, wa, wb, wc);
    uint st_w = floatBitsToUint(wc.z);
    bool w_locked = (st_w & 1u) != 0u;
    bool w_border = (st_w & 4u) != 0u;
    bool w_flagged = (st_w & 6u) != 0u;

    // A pair of border vertices merges only along the boundary itself: the
    // edge between them is a boundary edge exactly when one face uses it.
    // A pair of locked ends never merges at all.
    if (v_border && w_border && shared_faces[i] != 1u) continue;
    if (v_locked && w_locked) continue;

    // The pair quadric. Only the ten surface coefficients add; the state and
    // density channels stay the target's own, because the target survives.
    vec4 pa = qa + wa;
    vec4 pb = qb + wb;
    vec4 pc = vec4(qc.x + wc.x, qc.y + wc.y, wc.z, wc.w);

    vec3 pw = vpos(w);
    vec3 np;
    if (v_locked) {
      np = p;
    } else if (w_locked) {
      np = pw;
    } else if (v_flagged && w_flagged) {
      float ev = quad_error(pa, pb, pc, p);
      float ew = quad_error(pa, pb, pc, pw);
      np = (ew < ev) ? pw : p;
    } else if (v_flagged) {
      np = p;
    } else if (w_flagged) {
      np = pw;
    } else {
      mat3 a = mat3(pa.x, pa.y, pa.z,
                    pa.y, pb.x, pb.y,
                    pa.z, pb.y, pb.w);
      // The reference's own singularity guard: |det| > 1e-10 * scale^3
      // for the largest diagonal entry of the 3x3 block. Under float32 a
      // solve can pass that and still be garbage, so the midpoint score
      // double-checks it: the pair quadric is positive-semidefinite, which
      // means no genuine minimiser scores above the midpoint's.
      float scale = max(pa.x, max(pb.x, pb.w));
      float det = determinant(a);
      vec3 mid = 0.5 * (p + pw);
      np = mid;
      if (OPTIMAL_PLACEMENT != 0 && abs(det) > 1e-10 * scale * scale * scale) {
        vec3 solved = inverse(a) * (-vec3(pa.w, pb.z, pc.x));
        if (!(quad_error(pa, pb, pc, solved)
              > quad_error(pa, pb, pc, mid))) {
          np = solved;
        }
      }
    }

    float e = quad_error(pa, pb, pc, np) * density;
    // The reference clamps the score: a negative -- or NaN -- quadric error
    // counts as zero, so every pair at the floor competes on the hash alone.
    if (!(e >= 0.0)) e = 0.0;

    // TIES BREAK BY A PER-PASS HASH of the target, not by which neighbour
    // the ring walk happened to reach first. The ring order follows the
    // scatter's atomics, so first-found would make the choice depend on the
    // scheduler; a hash reseeded every pass makes it a pure function of
    // (target, pass) instead, on both backends. The tolerance is what lets
    // float32 rounding on this side agree with float64 on the CPU side
    // about which pairs are ties at all.
    uint hu = hd_hash_seeded(w, uint(SEED));
    if (tgt == NO_TARGET_U) {
      best = e; tgt = w; best_i = i; best_hash = hu; best_np = np;
      best_st = st_w;
    } else {
      float tol = 1e-6 * max(abs(e), abs(best));
      if (e < best - tol
          || (abs(e - best) <= tol && hu < best_hash)) {
        best = e; tgt = w; best_i = i; best_hash = hu; best_np = np;
        best_st = st_w;
      }
    }
  }
  if (tgt == NO_TARGET_U) REJECT(3)

  // A pair must share one or two faces: an edge is a mesh edge or a boundary
  // edge, and anything with three or more faces on it is a weld the collapse
  // cannot untangle. The reference refuses those outright.
  uint nshared = shared_faces[best_i];
  if (nshared == 0u || nshared > 2u) REJECT(7)

  // --- valence guard --------------------------------------------------
  // The merged ring is both rings less the faces the collapse removes,
  // twice over because each shared face dies once per ring. The reference
  // caps the merged ring itself, not the sum.
  uint deg_v = imageLoad(offs, IDX2(v + 1u)).r - imageLoad(offs, IDX2(v)).r;
  uint deg_w = imageLoad(offs, IDX2(tgt + 1u)).r - imageLoad(offs, IDX2(tgt)).r;
  if (deg_v + deg_w - 2u * nshared > uint(MAX_VALENCE)) REJECT(4)

  // --- link condition, PLAN.md 4.7 ------------------------------------
  // Shared neighbours of v and tgt must number exactly as many as the faces
  // on the edge (v, tgt): two inside a manifold, one on a boundary. A bitmask
  // over v's ring deduplicates without a second register array.
  uint mask = 0u;
  RING_BEGIN(tgt)
    uvec2 o = ring_others(tri, slot);
    for (int k = 0; k < 2; k++) {
      uint z = (k == 0) ? o.x : o.y;
      if (z == v) continue;
      for (uint i = 0u; i < count; i++) {
        if (nbr[i] == z) { mask |= (1u << i); break; }
      }
    }
  RING_END
  if (uint(bitCount(mask)) != nshared) REJECT(5)

  // --- normal flip test -----------------------------------------------
  // The survivor moves to best_np, so the faces that change shape are both
  // rings': v's ring loses the corner v and gains the corner tgt-at-np, and
  // the target's own ring has its tgt corner moved. Faces containing both
  // die and cannot fold. Faces of the target's ring are only walked when the
  // placement actually moves it. A degenerate face fails the test at a dot
  // of zero, matching the reference's normal helper, so it is not skipped.
  vec3 np = best_np;
  float worst = 1.0;
  RING_BEGIN(v)
    uvec2 o = ring_others(tri, slot);
    if (o.x == tgt || o.y == tgt) continue;    // this face collapses away
    vec3 b = vpos(o.x);
    vec3 c = vpos(o.y);
    vec3 n_old = cross(b - p, c - p);
    float l_old = length(n_old);
    if (l_old <= 0.0) { worst = 0.0; continue; }
    vec3 n_new = cross(b - np, c - np);
    float l_new = length(n_new);
    if (l_new <= 0.0) { worst = -1.0; continue; }
    worst = min(worst, dot(n_old / l_old, n_new / l_new));
  RING_END
  if (np.x != vpos(tgt).x || np.y != vpos(tgt).y || np.z != vpos(tgt).z) {
    vec3 pw = vpos(tgt);
    RING_BEGIN(tgt)
      uvec2 o = ring_others(tri, slot);
      if (o.x == v || o.y == v) continue;      // this face collapses away
      vec3 b = vpos(o.x);
      vec3 c = vpos(o.y);
      vec3 n_old = cross(b - pw, c - pw);
      float l_old = length(n_old);
      if (l_old <= 0.0) { worst = 0.0; continue; }
      vec3 n_new = cross(b - np, c - np);
      float l_new = length(n_new);
      if (l_new <= 0.0) { worst = -1.0; continue; }
      worst = min(worst, dot(n_old / l_old, n_new / l_new));
    RING_END
  }
  if (worst < FLIP_LIMIT) REJECT(6)

  // Which end's lineage survives the pair. The acceptor is the default, but
  // the reference's k_pair swaps a flagged proposer into the survivor slot --
  // the flag swap runs border first, then seam on the swapped pair, so two
  // swaps cancel -- and its k_apply always hands a locked end's identity to
  // the survivor, which makes the locked end the surviving lineage whichever
  // side it sits on. The bit rides in the spare channel of the placement
  // record so the commit, merge and log kernels all see the same direction.
  //
  // When both ends are flagged the placement is whichever end is cheaper,
  // and the flag swaps above say nothing about which that was. The survivor
  // is then made the end the placement sits on. The geometry is the same
  // either way, but the survivor's identity is what carries its UVs and
  // every other per-vertex attribute: the other end would put one vertex's
  // texture coordinates at the other's position, an edge or more out along a
  // border or a seam.
  uint st_t = best_st;
  bool s1 = (st_v & 4u) != 0u && (st_t & 4u) == 0u;
  uint bflags = s1 ? st_t : st_v;
  uint aflags = s1 ? st_v : st_t;
  bool s2 = (bflags & 2u) != 0u && (aflags & 2u) == 0u;
  bool swp = s1 != s2;
  if (v_flagged && (st_t & 6u) != 0u) {
    swp = (np == p);
  }
  if ((st_v & 1u) != 0u || (st_t & 1u) != 0u) {
    swp = (st_v & 1u) != 0u;
  }

  // --- UV placement and fold test ------------------------------------
  // The merged vertex's UV moves with it. A free pair is placed off both its
  // endpoints, and keeping the survivor's old UV there leaves every face
  // around it textured as if the vertex had not moved -- 43% of an edge out
  // on a 1% head scan. So the UV is read off the surface the pair covers now,
  // at the placement: the nearest face of either ring, by barycentrics of the
  // placement clamped into it. A flagged or locked end pins the placement to
  // that end, which is the survivor, so its own UV is already exact.
  //
  // Then no face may turn over in UV space for it, which the normal test
  // cannot promise: it judges 3D shape, and UV distortion is the map's own.
  //
  // Faces touching a seam are skipped, by both halves: a seam vertex has one
  // UV per island and `pos` holds only one of them, so interpolating from it
  // or testing against it would be judging the wrong wedge. The seam rules
  // already keep those vertices on their seams, at an endpoint.
  vec2 nuv = vec2(0.0);
  if (UV_CHECK != 0) {
    uint die = swp ? tgt : v;
    uint keep = swp ? v : tgt;
    nuv = vuv(keep);
    if (((st_v | st_t) & 2u) == 0u) {
      if (((st_v | st_t) & 7u) == 0u) {
        float near_d = 1e30;
        for (int side = 0; side < 2; side++) {
          uint ctr = (side == 0) ? v : tgt;
          vec3 pc = vpos(ctr);
          vec2 uc = vuv(ctr);
          RING_BEGIN(ctr)
            uvec2 o = ring_others(tri, slot);
            uint so = floatBitsToUint(imageLoad(quad, IDX3(o.x, 2)).z)
                    | floatBitsToUint(imageLoad(quad, IDX3(o.y, 2)).z);
            if ((so & 2u) != 0u) continue;
            vec3 e1 = vpos(o.x) - pc;
            vec3 e2 = vpos(o.y) - pc;
            vec3 d = np - pc;
            float d11 = dot(e1, e1), d12 = dot(e1, e2), d22 = dot(e2, e2);
            float den = d11 * d22 - d12 * d12;
            if (!(den > 0.0)) continue;  // degenerate face
            float d1 = dot(d, e1), d2 = dot(d, e2);
            float l1 = (d22 * d1 - d12 * d2) / den;
            float l2 = (d11 * d2 - d12 * d1) / den;
            vec3 l = max(vec3(1.0 - l1 - l2, l1, l2), vec3(0.0));
            l /= (l.x + l.y + l.z);
            vec3 q = np - (pc + l.y * e1 + l.z * e2);
            float dist = dot(q, q);
            if (dist < near_d) {
              near_d = dist;
              nuv = l.x * uc + l.y * vuv(o.x) + l.z * vuv(o.y);
            }
          RING_END
        }
      }

      // Every face that changes: the dying end's ring takes the new UV in
      // place of its own, and the survivor's ring in place of the survivor's.
      // Faces on the collapsing edge die and cannot fold.
      bool folds = false;
      for (int side = 0; side < 2; side++) {
        uint ctr = (side == 0) ? die : keep;
        uint other = (side == 0) ? keep : die;
        vec2 uo = vuv(ctr);
        if (side == 1 && uo == nuv) break;  // the survivor's UV is unchanged
        RING_BEGIN(ctr)
          uvec2 o = ring_others(tri, slot);
          if (o.x == other || o.y == other) continue;
          uint so = floatBitsToUint(imageLoad(quad, IDX3(o.x, 2)).z)
                  | floatBitsToUint(imageLoad(quad, IDX3(o.y, 2)).z);
          if ((so & 2u) != 0u) continue;
          vec2 b = vuv(o.x) - uo;
          vec2 c = vuv(o.y) - uo;
          float a_old = b.x * c.y - b.y * c.x;
          if (a_old == 0.0) continue;  // already degenerate in UV
          vec2 b2 = vuv(o.x) - nuv;
          vec2 c2 = vuv(o.y) - nuv;
          float a_new = b2.x * c2.y - b2.y * c2.x;
          if (a_new == 0.0 || (a_new > 0.0) != (a_old > 0.0)) folds = true;
        RING_END
      }
      if (folds) REJECT(8)
    }
  }

  imageStore(cand, IDX3(v, 0),
             vec4(best, uintBitsToFloat(tgt), 0.0, float(nshared)));
  imageStore(cand, IDX3(v, 1), vec4(np, swp ? 1.0 : 0.0));
  imageStore(cand, IDX3(v, 2), vec4(nuv, 0.0, 0.0));
}
"""

# The pairing pass. Every vertex proposes its cheapest edge; each vertex then
# accepts at most one proposal -- the cheapest offered to it -- which turns
# the proposal list into candidate pairs owned by the acceptor, exactly the
# reference's k_pair. A vertex whose proposal nobody accepted has no
# candidate this pass. The contest is keyed the way the face claims are:
# the cost's float bits with the sign cleared first, then the proposer id
# among equal keys, so exactly one proposer can be accepted.
ACCEPT_CLEAR = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  imageStore(accept, IDX3(v, 0), uvec4(uint(CLAIM_KEY_CLEAR)));
  imageStore(accept, IDX3(v, 1), uvec4(uint(CLAIM_KEY_CLEAR)));
}
"""

ACCEPT_KEY = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  vec4 c = imageLoad(cand, IDX3(v, 0));
  uint t = floatBitsToUint(c.y);
  if (t == NO_TARGET_U) return;
  uint key = floatBitsToUint(c.x) & 0x7fffffffu;
  imageAtomicMin(accept, IDX3(t, 0), key);
}
"""

ACCEPT_ID = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  vec4 c = imageLoad(cand, IDX3(v, 0));
  uint t = floatBitsToUint(c.y);
  if (t == NO_TARGET_U) return;
  uint key = floatBitsToUint(c.x) & 0x7fffffffu;
  if (imageLoad(accept, IDX3(t, 0)).r != key) return;
  imageAtomicMin(accept, IDX3(t, 1), v);
}
"""

# Kernel 7. Log-spaced histogram of candidate costs, so the host can pick an
# admission threshold from a 4 KB readback instead of sorting anything.
HISTOGRAM = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  vec4 c = imageLoad(cand, IDX3(v, 0));
  if (floatBitsToUint(c.y) == NO_TARGET_U) return;
  float e = max(c.x, 1e-30);
  // log10 mapped onto the bucket range. LOG_LO and LOG_HI come from the host.
  float t = (log2(e) * 0.30103 - LOG_LO) / max(LOG_HI - LOG_LO, 1e-6);
  int b = clamp(int(t * float(BUCKETS)), 0, BUCKETS - 1);
  imageAtomicAdd(hist, IDX2(uint(b)), 1u);
}
"""

# Candidate status for the claim rounds: a plain flag array is all the claims
# need, because the contest itself is decided by cost key and owner id on the
# faces. NOT_CANDIDATE sits above both status bits, so `prio >= NOT_CANDIDATE`
# covers "cannot compete this round", whatever the reason.
#
#   bit 31  taken. Set by MARK_TAKEN on a committed collapse, its target and
#           both of their rings, so later rounds work around them.
#   bit 30  blocked. Set by MARK_BLOCKED when the vertex's ring touches a
#           reservation, so it cannot win for the rest of this pass.
#
# The two bits must be distinct because MARK_BLOCKED reads bit 31 on neighbours
# while writing bit 30 on itself. Sharing one bit would let a concurrent write
# masquerade as a reservation and cascade the blocking across the mesh.
SET_PRIORITY = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  vec4 c = imageLoad(cand, IDX3(v, 0));
  uint t = floatBitsToUint(c.y);
  uint key = floatBitsToUint(c.x) & 0x7fffffffu;
  // The proposal is a candidate exactly when the target accepted it: it was
  // the cheapest offered to that vertex this pass. Rejected proposals and
  // costs past the admission threshold sit the round out together.
  bool paired = (t != NO_TARGET_U)
      && imageLoad(accept, IDX3(t, 0)).r == key
      && imageLoad(accept, IDX3(t, 1)).r == v;
  uint p = (!paired || c.x > THRESHOLD)
           ? uint(NOT_CANDIDATE_U) : 0u;
  imageStore(prio, IDX2(v), uvec4(p));
}
"""

TAKEN_BIT = 0x80000000
BLOCKED_BIT = 0x40000000
NOT_CANDIDATE = 0x3FFFFFFF

# The claim arrays start each pass as "no owner": clearing 0xFFFFFFFF through
# a kernel rather than Array.clear, because Blender marshals clear values
# through a signed int and refuses anything above 2**31-1.
CLAIM_CLEAR = """
void main() {
  uint f = GID;
  if (f >= uint(n)) return;
  imageStore(claim, IDX3(f, 0), uvec4(uint(CLAIM_KEY_CLEAR)));
  imageStore(claim, IDX3(f, 1), uvec4(uint(CLAIM_KEY_CLEAR)));
}
"""

# THE CLAIMS. The affected faces of a collapse are both rings': the source's
# corner is rewritten and the survivor moves, so a face in either ring changes
# shape, and two collapses may never share one. Each candidate therefore
# claims every face of both rings with its cost key -- the float bits of the
# cost with the sign cleared, which is monotone in the cost for the positive
# floats every cost here is -- and the cheapest claim on a face wins it.
#
# Round one records the cheapest key; round two resolves key ties by owner id,
# so exactly one candidate can end up owning a face, whatever the costs.
CLAIM1 = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  if (imageLoad(prio, IDX2(v)).r >= uint(NOT_CANDIDATE_U)) return;
  uint w = floatBitsToUint(imageLoad(cand, IDX3(v, 0)).y);
  uint key = floatBitsToUint(imageLoad(cand, IDX3(v, 0)).x) & 0x7fffffffu;
  RING_BEGIN(v)
    imageAtomicMin(claim, IDX3(face, 0), key);
  RING_END
  RING_BEGIN(w)
    imageAtomicMin(claim, IDX3(face, 0), key);
  RING_END
}
"""

CLAIM2 = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  if (imageLoad(prio, IDX2(v)).r >= uint(NOT_CANDIDATE_U)) return;
  uint w = floatBitsToUint(imageLoad(cand, IDX3(v, 0)).y);
  uint key = floatBitsToUint(imageLoad(cand, IDX3(v, 0)).x) & 0x7fffffffu;
  bool mine = true;
  RING_BEGIN(v)
    if (imageLoad(claim, IDX3(face, 0)).r != key) mine = false;
  RING_END
  RING_BEGIN(w)
    if (imageLoad(claim, IDX3(face, 0)).r != key) mine = false;
  RING_END
  if (!mine) return;
  RING_BEGIN(v)
    imageAtomicMin(claim, IDX3(face, 1), v);
  RING_END
  RING_BEGIN(w)
    imageAtomicMin(claim, IDX3(face, 1), v);
  RING_END
}
"""

# A candidate wins exactly when it still owns every face it claimed, by key
# and by id. Two candidates sharing a face can never both pass: the shared
# face has one key and one owner.
#
# The `w` guard is load-bearing, not redundant: a vertex with no faces left
# has an empty ring, its RING loops make zero passes over the checks, and
# without the guard it "confirms" into a phantom winner that commits a
# NO_TARGET remap. Any vertex that still has a legal candidate has a
# non-empty ring by construction, so the check costs the pass nothing.
CONFIRM = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  uint w = floatBitsToUint(imageLoad(cand, IDX3(v, 0)).y);
  if (w == NO_TARGET_U) return;
  uint key = floatBitsToUint(imageLoad(cand, IDX3(v, 0)).x) & 0x7fffffffu;
  RING_BEGIN(v)
    uvec2 cl = uvec2(imageLoad(claim, IDX3(face, 0)).r,
                     imageLoad(claim, IDX3(face, 1)).r);
    if (cl.x != key || cl.y != v) return;
  RING_END
  RING_BEGIN(w)
    uvec2 cl = uvec2(imageLoad(claim, IDX3(face, 0)).r,
                     imageLoad(claim, IDX3(face, 1)).r);
    if (cl.x != key || cl.y != v) return;
  RING_END
  imageStore(winner, IDX2(v), uvec4(1u));
}
"""

# Any candidate whose ring touches a reservation is out for the rest of the
# pass. Runs before the claims so blocked candidates never enter the contest
# at all; with claims exact, this is what stops a later round from re-opening
# a face a committed collapse already owns.
#
# Checking the candidate's own ring is enough even though the claims cover
# the target's ring too: the target is a neighbour, so it is in the own ring,
# and every vertex of a shared face is a ring neighbour of the candidate's
# pair (see MARK_TAKEN for the full argument).
MARK_BLOCKED = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  uint prio_v = imageLoad(prio, IDX2(v)).r;
  if (prio_v >= uint(NOT_CANDIDATE_U)) return;  // no candidate, blocked, or taken
  bool blocked = false;
  RING_BEGIN(v)
    uvec2 o = ring_others(tri, slot);
    if ((imageLoad(prio, IDX2(o.x)).r & TAKEN_BIT_U) != 0u) blocked = true;
    if ((imageLoad(prio, IDX2(o.y)).r & TAKEN_BIT_U) != 0u) blocked = true;
  RING_END
  if (blocked) imageStore(prio, IDX2(v), uvec4(prio_v | BLOCKED_BIT_U));
}
"""

# Reserve a winner's whole affected neighbourhood so later rounds in the same
# pass keep clear of it: the winner itself, its target, and both of their
# rings. Writes only ever set the bit, so concurrent writers agree and no
# atomic is needed.
#
# Both rings, not just the winner's: any conflict between two collapses shows
# up as a face of one pair's rings sharing a corner with the other pair, and
# that corner is inside one of these two rings -- so a candidate blocked on
# its own ring (which contains its target) is excluded from every conflict.
MARK_TAKEN = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  if (imageLoad(winner, IDX2(v)).r == 0u) return;
  uint w = floatBitsToUint(imageLoad(cand, IDX3(v, 0)).y);
  imageStore(prio, IDX2(v),
             uvec4(imageLoad(prio, IDX2(v)).r | TAKEN_BIT_U));
  imageStore(prio, IDX2(w),
             uvec4(imageLoad(prio, IDX2(w)).r | TAKEN_BIT_U));
  RING_BEGIN(v)
    uvec2 o = ring_others(tri, slot);
    imageStore(prio, IDX2(o.x),
               uvec4(imageLoad(prio, IDX2(o.x)).r | TAKEN_BIT_U));
    imageStore(prio, IDX2(o.y),
               uvec4(imageLoad(prio, IDX2(o.y)).r | TAKEN_BIT_U));
  RING_END
  RING_BEGIN(w)
    uvec2 o = ring_others(tri, slot);
    imageStore(prio, IDX2(o.x),
               uvec4(imageLoad(prio, IDX2(o.x)).r | TAKEN_BIT_U));
    imageStore(prio, IDX2(o.y),
               uvec4(imageLoad(prio, IDX2(o.y)).r | TAKEN_BIT_U));
  RING_END
}
"""

# Commit. Writes *every* vertex, not just the winners: a winner maps onto its
# target and everything else maps to itself.
#
# The winner's thread also moves the survivor onto the placement the pass
# tested. Targets are unique across a pass -- the claims own the target's ring
# faces, so no two winners share one -- so the write has no race.
#
# Deliberately not "seed the array with the identity, then overwrite the
# winners". That needs a second kernel to have landed first, and on OpenGL it
# intermittently had not, leaving remap full of zeros so every vertex
# collapsed onto vertex 0 in a single pass and a 4,418-face grid became 4,418
# degenerate faces. Writing the whole array from one kernel removes the
# ordering dependency and one dispatch per pass.
COMMIT_REMAP = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  uint target = v;
  vec4 c0 = imageLoad(cand, IDX3(v, 0));
  uint w = floatBitsToUint(c0.y);
  bool swp = imageLoad(cand, IDX3(v, 1)).w != 0.0;
  if (imageLoad(winner, IDX2(v)).r != 0u && w != NO_TARGET_U) {
    if (swp) {
      // The flagged or locked end keeps its lineage: the proposer's own
      // slot survives at the placement, and the acceptor dies instead --
      // its own thread rewrites its remap below.
      imageStore(pos, IDX3(v, 0),
                 vec4(imageLoad(cand, IDX3(v, 1)).xyz, 0.0));
    } else {
      target = w;
      imageStore(pos, IDX3(w, 0),
                 vec4(imageLoad(cand, IDX3(v, 1)).xyz, 0.0));
    }
    // The UV moves with the position; see the candidate kernel.
    if (UV_CHECK != 0) {
      imageStore(pos, IDX3(swp ? v : w, 1), imageLoad(cand, IDX3(v, 2)));
    }
  } else {
    // A vertex also dies without a candidate of its own when the proposal
    // it accepted won as a swapped pair: the proposer keeps the lineage,
    // so this end's corners rewrite to it.
    uint u = imageLoad(accept, IDX3(v, 1)).r;
    if (u != uint(CLAIM_KEY_CLEAR) && u != v) {
      vec4 uc = imageLoad(cand, IDX3(u, 0));
      bool paired = imageLoad(accept, IDX3(v, 0)).r
                    == (floatBitsToUint(uc.x) & 0x7fffffffu);
      bool u_won = imageLoad(winner, IDX2(u)).r != 0u;
      bool u_swap = imageLoad(cand, IDX3(u, 1)).w != 0.0;
      if (paired && u_won && u_swap) target = u;
    }
  }
  imageStore(remap, IDX2(v), uvec4(target));
}
"""

# Q_target += Q_v for every winner. Targets are unique across a pass, which
# the claims guarantee, so no atomics and no read-write race.
#
# The border flag is inherited, not summed: a collapse cannot create a border,
# so any border either end already carried is one the merged vertex still
# sits on. The reference pipeline does the same, and it is why the border bit
# stays trustworthy through a whole run without ever being recomputed.
MERGE_QUADRICS = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  if (imageLoad(winner, IDX2(v)).r == 0u) return;
  uint w = floatBitsToUint(imageLoad(cand, IDX3(v, 0)).y);
  if (w == NO_TARGET_U) return;
  // The survivor is the acceptor, except on a swapped pair where the
  // flagged or locked proposer keeps the lineage; see the CANDIDATE kernel.
  bool swp = imageLoad(cand, IDX3(v, 1)).w != 0.0;
  uint surv = swp ? v : w;
  uint dead = swp ? w : v;
  for (int l = 0; l < 3; l++) {
    vec4 a = imageLoad(quad, IDX3(dead, l));
    vec4 b = imageLoad(quad, IDX3(surv, l));
    if (l == 2) {
      // Layer 2 carries state and density in .z/.w, which must not be
      // summed. Keep the survivor's density, and give it every flag the
      // dead end carried: a collapse cannot create a border or a seam, so
      // any either end had is one the merged vertex still sits on, and a
      // lock -- which fixes a position, not a vertex id -- passes to the
      // replacement vertex the same way the reference inherits it.
      uint state = floatBitsToUint(b.z) | (floatBitsToUint(a.z) & 7u);
      imageStore(quad, IDX3(surv, l),
                 vec4(a.x + b.x, a.y + b.y, uintBitsToFloat(state), b.w));
    } else {
      imageStore(quad, IDX3(surv, l), a + b);
    }
  }
}
"""

# THE COLLAPSE LOG, written on the device. One record per committed collapse,
# in the original numbering, so the pass loop's compactions cannot touch it:
#
#   layer 0  the source vertex, by its base id
#   layer 1  the survivor, by its base id
#   layer 2  the placement x, as float bits
#   layer 3  the placement y
#   layer 4  the placement z
#   layer 5  how many faces the collapse removed
#   layer 6  the live face count after the whole pass
#   layer 7  the record's rank within the pass, which is also the order the
#            replay applies them in
#   layer 8  the placement u, as float bits (zero when UVs are not tracked)
#   layer 9  the placement v
#
# The per-record face count is what lets the host reconstruct a running count
# through the pass on replay, so a target that lands inside a pass can be hit
# exactly rather than to the nearest pass.
LOG_APPEND = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  if (imageLoad(winner, IDX2(v)).r == 0u) return;
  uint slot = uint(base) + imageLoad(rank, IDX2(v)).r;
  if (slot >= uint(cap)) return;
  uint w = floatBitsToUint(imageLoad(cand, IDX3(v, 0)).y);
  if (w == NO_TARGET_U) return;
  vec4 c0 = imageLoad(cand, IDX3(v, 0));
  vec3 np = imageLoad(cand, IDX3(v, 1)).xyz;
  // A swapped pair dies at its acceptor and survives at the proposer; the
  // record's convention is always (dead, survivor).
  bool swp = imageLoad(cand, IDX3(v, 1)).w != 0.0;
  uint dead = swp ? w : v;
  uint surv = swp ? v : w;
  imageStore(log, IDX3(slot, 0), uvec4(imageLoad(vorigin, IDX2(dead)).r));
  imageStore(log, IDX3(slot, 1), uvec4(imageLoad(vorigin, IDX2(surv)).r));
  imageStore(log, IDX3(slot, 2), uvec4(floatBitsToUint(np.x)));
  imageStore(log, IDX3(slot, 3), uvec4(floatBitsToUint(np.y)));
  imageStore(log, IDX3(slot, 4), uvec4(floatBitsToUint(np.z)));
  imageStore(log, IDX3(slot, 5), uvec4(uint(c0.w)));
  imageStore(log, IDX3(slot, 6), uvec4(uint(live)));
  imageStore(log, IDX3(slot, 7), uvec4(imageLoad(rank, IDX2(v)).r));
  vec2 nuv = (UV_CHECK != 0) ? imageLoad(cand, IDX3(v, 2)).xy : vec2(0.0);
  imageStore(log, IDX3(slot, 8), uvec4(floatBitsToUint(nuv.x)));
  imageStore(log, IDX3(slot, 9), uvec4(floatBitsToUint(nuv.y)));
  if (v == 0u) imageStore(probe_dst, IDX2(0u), uvec4(1u));
}
"""
