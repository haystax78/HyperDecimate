"""Vertex compaction. The reference pipeline's `k_compact_*`.

**Why.** The pass loop dispatches over the vertex count, and until now that was
the *original* vertex count for the whole run. Faces were compacted every pass;
vertices never were. On the 13.04M-triangle head scan reduced to 5,000
triangles that is plainly visible in the timings:

| v1 pass loop | passes | seconds | per pass |
| --- | --- | --- | --- |
| 13.04M -> 500,000 | 87 | 2.76 | 31.7 ms |
| 13.04M -> 5,000 | 162 | 4.15 | 25.6 ms |

The 75 passes between those two rows cost 1.40 s, 18.6 ms each, while the mesh
they were working on fell from 500,000 triangles to 5,000. They were not doing
less work than the early passes, because the work was never a function of how
many vertices were still alive -- a dozen kernels per pass, every one of them
walking a 6.52M-entry array of which, by the end, 0.07% held a live vertex. The
reference says the same thing about its own batches before it compacted: "batch
24 with 31% alive cost the same 49 ms as batch 2".

**What is compacted and what is not.** Three arrays carry state that has to
survive: `pos`, `quad` and `vflags`. Those are gathered to the front. The
per-pass scratch -- `cand`, `prio`, `winner`, `claim`, `remap`, `counts`,
`offs`, `cursor` -- holds nothing between passes, so it is reallocated at the
new size rather than copied, which is both cheaper and where most of the VRAM
saving comes from. `chain` is the exception that is reallocated never: it is
indexed by *original* vertex, so it keeps its length for the life of the
session and only has its values pushed through the renumbering.

**Why the destinations are allocated fresh each time.** Compaction cannot write
into the array it reads -- `prims.py` explains the race -- so it needs a
destination, and sizing that destination at the live count rather than at the
original one is what makes the memory go down instead of up. The transient peak
is the old array plus the new, so compacting only once the live count has
roughly halved keeps that peak at about 1.5x of the vertex arrays rather than
2x. PLAN.md 2.3a is why that margin is worth caring about: on a 104M-vertex
scan the backend already sits at 21.8 GiB of a 24 GiB card, and going over does
not fail, it pages, which costs far more than any kernel here.
"""

from __future__ import annotations

# A vertex is alive exactly when some surviving triangle still names it. Run
# after the faces have been compacted, so "surviving" means what it says.
#
# Every write stores 1, so concurrent writers to the same slot agree and no
# atomic is needed -- the same argument as MARK_TAKEN in candidates.py.
MARK_VERTS_ALIVE = """
void main() {
  uint f = GID;
  if (f >= uint(n)) return;
  imageStore(vlive, IDX2(imageLoad(c0, IDX2(f)).r), uvec4(1u));
  imageStore(vlive, IDX2(imageLoad(c1, IDX2(f)).r), uvec4(1u));
  imageStore(vlive, IDX2(imageLoad(c2, IDX2(f)).r), uvec4(1u));
}
"""

# Gather the positions to the front, both layers: (x, y, z) and the UV the
# fold test reads. `offsets` is the exclusive prefix sum of `vlive`, so a live
# vertex's new index is already computed.
COMPACT_POS = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  if (imageLoad(vlive, IDX2(i)).r == 0u) return;
  uint slot = imageLoad(offsets, IDX2(i)).r;
  imageStore(pos_dst, IDX3(slot, 0), imageLoad(pos_src, IDX3(i, 0)));
  imageStore(pos_dst, IDX3(slot, 1), imageLoad(pos_src, IDX3(i, 1)));
}
"""

# The same, for the layered quadric. Three layers in one dispatch rather than
# three dispatches of one layer: the gather is memory bound and the index
# arithmetic is shared.
COMPACT_QUADRIC = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  if (imageLoad(vlive, IDX2(i)).r == 0u) return;
  uint slot = imageLoad(offsets, IDX2(i)).r;
  for (int l = 0; l < 3; l++) {
    imageStore(quad_dst, IDX3(slot, l), imageLoad(quad_src, IDX3(i, l)));
  }
}
"""

# Push the triangle corners through the renumbering. Safe in place: every
# corner names a live vertex, whose new index is at most its old one, and each
# corner is written by exactly one thread from a value it read itself.
RENUMBER_CORNERS = """
void main() {
  uint f = GID;
  if (f >= uint(n)) return;
  imageStore(c0, IDX2(f),
             uvec4(imageLoad(offsets, IDX2(imageLoad(c0, IDX2(f)).r)).r));
  imageStore(c1, IDX2(f),
             uvec4(imageLoad(offsets, IDX2(imageLoad(c1, IDX2(f)).r)).r));
  imageStore(c2, IDX2(f),
             uvec4(imageLoad(offsets, IDX2(imageLoad(c2, IDX2(f)).r)).r));
}
"""

# And the original-to-current chain.
#
# In place for the same reason as the corners: one thread per entry, reading
# and writing only its own slot.
#
# **The clamp is not belt-and-braces, it is the whole correctness of this
# kernel.** It is tempting to assume every value in `chain` names a live vertex
# -- a collapse does rewrite the entries pointing at the vertex that died, to
# the one that survived -- but a vertex can also stop being live without ever
# collapsing. If every face around it degenerates in the same pass, because
# each of those faces had its other two corners merged, the vertex is left with
# no faces at all. It was nobody's collapse target, so nothing rewrote the
# chain entries naming it, and it is not live, so it has no slot in the
# compacted arrays.
#
# `offsets` is an exclusive prefix sum, so for such a vertex it holds the number
# of live vertices *before* it -- which is `live` exactly when every live vertex
# precedes it. That is one past the end. Measured on the 13M head scan reduced
# to 5,000, eight entries were out of range at the second compaction and 161 by
# the eighth, and the symptom was not a crash on the device but an
# `IndexError` in the host readback eleven compactions later, by which time the
# values had been pushed through the renumbering nine more times and looked like
# nothing in particular.
#
# Clamping lands those entries on a live vertex: the next live one in the old
# ordering, or the last one when there is none after it. An original vertex
# whose geometry has entirely vanished has no right answer here, and what the
# contract in `SimplifyResult` does require is that its remap is a real output
# vertex rather than a negative number. `survived` is unaffected either way --
# it is derived from `vorigin`, and a vertex with no faces has no output slot
# to survive into.
RENUMBER_CHAIN = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  uint slot = imageLoad(offsets, IDX2(imageLoad(chain, IDX2(i)).r)).r;
  imageStore(chain, IDX2(i), uvec4(min(slot, uint(live) - 1u)));
}
"""
