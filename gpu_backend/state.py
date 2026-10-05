"""GPU-side mesh state: the arrays, the adjacency build, the quadric build.

Split out from the pass loop so each stage can be checked on its own. Every
stage here has a matching check in `tests/test_gpu_simplify.py`; nothing in
the pass loop is trusted until the stage below it verifies.

Image slot budget is 8 (PLAN.md 2.5), and the quadric build sits exactly at it:
c0, c1, c2, offs, adj, pos, quad, vflags. Adding anything to that kernel means
taking something out.

The other budget is VRAM, and on the largest meshes it is the binding one: the
arrays here come to 21.8 GiB on a 104M-vertex, 208M-triangle scan, against a
24 GiB card. PLAN.md 2.3a has the breakdown and the reasoning behind allocating
`vlock` and `vdensity` only for the one dispatch that reads them.
"""

from __future__ import annotations

import numpy as np

from ..core import quadrics as qd
from ..core import replay as core_replay
from . import context as ctx
from . import prims as pr
from .kernels import adjacency as ka
from .kernels import candidates as kc
from .kernels import compaction as kv

U = "R32UI"
F1 = "R32F"
F4 = "RGBA32F"
INT1 = (("INT", "n"),)

# Histogram of log10(cost). The range is fixed and generous rather than measured
# per pass: costs live between about 1e-20 and 1 after unit-cube normalisation, and
# a wrong-but-wide range only costs threshold resolution, while a wrong-but-narrow
# one would clip real candidates into the end buckets.
HIST_BUCKETS = 1024
HIST_LOG_LO = -30.0
HIST_LOG_HI = 2.0

# The collapse log is 32 bytes per committed collapse and every collapse
# kills exactly one vertex, so the vertex count bounds the record count. A
# mesh too big for the buffer allocates nothing and simply cannot replay.
LOG_LAYERS = 10

# Vertices per slice when normalising into the upload buffer. Large enough that
# the per-slice NumPy overhead is irrelevant, small enough that the float64
# intermediate stays in cache instead of being a second copy of the mesh. 256K
# measured fastest on a 6.5M-vertex scan; 4M was 11% slower and unchunked 14%.
_NORM_CHUNK = 1 << 18

# remap[i] = i, written on the GPU.
#
# This used to be a host upload of np.arange every pass, which was both wasteful
# and wrong: the integer upload path allocates a staging texture, runs a pack
# kernel and forces a sync, all inside the hot loop, and on OpenGL after enough
# textures had been created and dropped it stopped taking effect. The symptom was
# a remap of all zeros, so every vertex collapsed onto vertex 0 in a single pass
# and the whole mesh became 4,418 degenerate faces. It reproduced 5 times out of 5
# on OpenGL and 0 out of 5 on Vulkan, and not at all when the same call ran in
# isolation. Writing the identity with a kernel avoids the upload entirely.
IDENTITY_SRC = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  imageStore(remap, IDX2(i), uvec4(i));
}
"""

# Carry the accumulated original-to-current mapping forward by one pass.
CHAIN_SRC = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  imageStore(chain, IDX2(i), uvec4(imageLoad(remap, IDX2(imageLoad(chain, IDX2(i)).r)).r));
}
"""


class GPUState:
    """Mesh arrays on the GPU, plus the stages that derive from them."""

    def __init__(self, positions, triangles, boundary_weight=qd.BOUNDARY_WEIGHT,
                 max_normal_flip_deg=78.46304096718453, max_valence=48,
                 freeze_borders=False, optimal_placement=True, uvs=None):
        # Whatever dtype the caller has. Nothing here needs float64 positions or
        # int64 indices: the destinations are float32 and uint32, so forcing the
        # wide dtypes first only bought a pair of full-size copies -- 1.9 GB of
        # them on a 52M-triangle mesh -- to throw away immediately afterwards.
        positions = np.ascontiguousarray(positions)
        triangles = np.ascontiguousarray(triangles)

        # Normalise on the host, exactly as the CPU path does, so quadrics are
        # conditioned the same way and the two backends are comparable. See
        # PLAN.md 4.1 for why this is not optional.
        self.centre, self.scale = qd.normalize_bounds(positions)

        self.v_count = positions.shape[0]
        self.f_count = triangles.shape[0]
        self.face_capacity = triangles.shape[0]
        self.boundary_weight = float(boundary_weight)
        self.flip_limit = float(np.cos(np.radians(max_normal_flip_deg)))
        self.max_valence = int(max_valence)
        self.freeze_borders = bool(freeze_borders)
        # Where a free pair lands: the pair quadric's minimiser, or the edge
        # midpoint -- the reference's --centroid control. The ruled placements
        # (locked and flagged endpoints) apply either way.
        self.optimal_placement = bool(optimal_placement)
        # Per-vertex UVs switch on the UV fold test; see Options.uvs.
        self.check_uvs = uvs is not None
        # Normalise straight into the float32 upload buffer, a slice at a time.
        # `centre` and `scale` are float64 scalars, so each chunk is still
        # evaluated in float64 and rounded once on the way into float32 -- the
        # same arithmetic and the same result as scaling the whole array in
        # float64 and casting, but without the two full-size float64 temporaries,
        # which were 1.25 GB together on a 26M-vertex mesh.
        padded = np.zeros((self.v_count, 4), dtype=np.float32)
        for lo in range(0, self.v_count, _NORM_CHUNK):
            hi = min(lo + _NORM_CHUNK, self.v_count)
            padded[lo:hi, :3] = (positions[lo:hi] - self.centre) * self.scale

        V, F, C = self.v_count, self.f_count, 3 * self.f_count

        # Widen the texture packing before anything is allocated, since the
        # adjacency's C entries are the longest array in the backend and a width
        # that cannot hold them fails inside GPUTexture with an opaque "unknown
        # error". Doing it here rather than per Array keeps one addressing scheme
        # for the whole session, which is what the kernels compile against. It
        # raises GPUUnavailable when no width fits, and dispatch takes that as the
        # signal to fall back to the CPU tier.
        self.pack_width = ctx.plan_packing(max(C, V + 1))

        # Built after the width is settled: Prims compiles its own kernels and
        # allocates scratch, and both would bake in the old addressing.
        self.prims = pr.Prims()

        # Two layers: (x, y, z, -) and (u, v, -, -). The UVs ride with the
        # positions so the candidate kernel can read them without a ninth
        # image unit, and so vertex compaction carries them along for free.
        # A commit writes layer 0 only, so a survivor keeps its own UV, which
        # is the UV the transfer will give it.
        texels = ctx.PACK_W * ctx.rows_for(V, ctx.PACK_W)
        layered = np.zeros((2, texels, 4), dtype=np.float32)
        layered[0, :V] = padded
        del padded
        if uvs is not None:
            layered[1, :V, :2] = np.asarray(uvs, dtype=np.float32).reshape(V, 2)
        self.pos = ctx.Array("hd_pos", V, fmt=F4, layers=2, data=layered)
        del layered
        self.c0 = ctx.Array("hd_c0", F, fmt=U,
                            data=triangles[:, 0].astype(np.uint32))
        self.c1 = ctx.Array("hd_c1", F, fmt=U,
                            data=triangles[:, 1].astype(np.uint32))
        self.c2 = ctx.Array("hd_c2", F, fmt=U,
                            data=triangles[:, 2].astype(np.uint32))

        # Compaction destinations. Compacting in place is a race, see prims.py,
        # so the corner arrays are double buffered and swapped.
        self.c0b = ctx.Array("hd_c0b", F, fmt=U)
        self.c1b = ctx.Array("hd_c1b", F, fmt=U)
        self.c2b = ctx.Array("hd_c2b", F, fmt=U)
        # Written by the identity kernel at the end of __init__, not uploaded.
        self.forigin = ctx.Array("hd_forigin", F, fmt=U)
        self.foriginb = ctx.Array("hd_foriginb", F, fmt=U)

        self.alive = ctx.Array("hd_alive", F, fmt=U)
        self.counts = ctx.Array("hd_counts", V + 1, fmt=U)
        self.offs = ctx.Array("hd_offs", V + 1, fmt=U)
        self.cursor = ctx.Array("hd_cursor", V, fmt=U)
        self.adj = ctx.Array("hd_adj", C, fmt=U)
        self.quad = ctx.Array("hd_quad", V, fmt=F4, layers=3)
        self.vflags = ctx.Array("hd_vflags", V, fmt=U)
        self.remap = ctx.Array("hd_remap", V, fmt=U)

        # Candidate selection and the independent set. `accept` holds the
        # pairing contest: layer 0 the cheapest proposal's cost key, layer 1
        # the proposer id among equal keys, mirroring the face claims.
        self.cand = ctx.Array("hd_cand", V, fmt=F4, layers=3)
        self.accept = ctx.Array("hd_accept", V, fmt=U, layers=2)
        self.prio = ctx.Array("hd_prio", V, fmt=U)
        self.winner = ctx.Array("hd_winner", V, fmt=U)
        self.hist = ctx.Array("hd_hist", HIST_BUCKETS, fmt=U)

        # The claim rounds: one entry per face, key on layer 0 and owner id on
        # layer 1. Sized to the face capacity rather than the live count,
        # because face compaction swaps corner arrays of a fixed capacity and
        # the claim array is never re-gathered.
        self.claim = ctx.Array("hd_claim", self.face_capacity, fmt=U, layers=2)

        # The collapse log, one 32-byte record per committed collapse in the
        # original vertex numbering. A collapse kills exactly one vertex, so V
        # records bound the whole run and the buffer is sized by vertices, not
        # faces -- on a scan that is half of what an F-sized allocation costs.
        # A run with no log still decimates, it just cannot be replayed to
        # another target, so the allocation is attempted and a failure falls
        # back to logless the same way the old face cap did.
        self.log = None
        self.log_count = 0
        self.log_truncated = False
        try:
            self.log = ctx.Array("hd_log", V, fmt=U, layers=LOG_LAYERS)
        except ctx.GPUUnavailable:
            pass

        # Per-vertex user inputs, folded into the quadric's spare channels by
        # write_params so the candidate kernel stays inside its 8 slots.
        #
        # Allocated by write_params and released again as soon as it has folded
        # them in, because that one dispatch is the only thing that ever reads
        # them: WRITE_PARAMS loads both and writes the result into the quadric's
        # layer 2, and no later kernel binds either. Holding them for the whole
        # session cost 0.78 GiB on a 104M-vertex mesh, where the backend's arrays
        # already come to 21.8 GiB of a 24 GiB card, so this is the difference
        # between fitting and paging. Not allocating them until then also keeps
        # them out of the peak during the corner uploads.
        self.vlock = None
        self.vdensity = None

        # Original vertex index -> its current representative. Identity to start,
        # also written by the kernel.
        self.chain = ctx.Array("hd_chain", V, fmt=U)

        # Current vertex index -> the original vertex it is. The vertex analogue
        # of `forigin`, and it exists because vertex compaction renumbers.
        #
        # Before compaction existed, "did original vertex i survive" was
        # answered by `chain[i] == i`: a vertex that never collapsed still
        # pointed at itself. Compaction breaks that -- vertex i's current index
        # is no longer i -- and it breaks it *silently*, which is the dangerous
        # part: the comparison still type-checks, still returns a plausible
        # number of survivors, and the attribute transfer then reads the wrong
        # source vertex for every output vertex. Carrying the origin forward
        # makes the question structural instead of a numbering coincidence, and
        # it is the same answer with or without compaction.
        self.vorigin = ctx.Array("hd_vorigin", V, fmt=U)

        # Vertices that have died since the last vertex compaction. Counted
        # rather than scanned for: `count_winners` already tells the pass loop
        # how many collapses it committed, and each one kills exactly one
        # vertex, so the compaction trigger is free. Scanning `vlive` to find
        # out would cost a pass over the vertex array every pass, which is the
        # very thing compaction exists to stop doing.
        self._dead_since_compaction = 0
        self.compactions = 0
        # The original vertex count, which `chain` is indexed by and which
        # therefore never shrinks. `v_count` is the live count and does.
        self.v_origin = self.v_count

        self.vflags.clear(0)
        self.alive.fill(1)
        self._build_kernels()

        # np.arange uploads, until they were not. `forigin` and `chain` both start
        # as the identity, and an identity is cheaper to compute than to send: the
        # host array, the staging texture and the transfer all disappear, which on
        # a 52M-triangle mesh is 1.25 GB of allocation and about a third of a
        # second. IDENTITY_SRC already existed for `remap` and says more about why.
        self.k_identity.run(F, bind={"remap": self.forigin}, n=F)
        self.k_identity.run(V, bind={"remap": self.chain}, n=V)
        self.k_identity.run(V, bind={"remap": self.vorigin}, n=V)

    # ------------------------------------------------------------- kernels

    def _build_kernels(self):
        """Compile every kernel once.

        Slot names are the short ones the GLSL uses, not the Array names. Kernel
        declares the slots, `run(bind=...)` attaches whichever Array belongs in
        each, so an Array's own name never has to match a shader variable.
        """
        # Include only the helpers a kernel can back with declared images.
        # ring_* needs c0/c1/c2/offs/adj, vpos needs pos, quad_* needs quad.
        # Handing a kernel a helper whose image it has not declared fails with
        # "undefined variable" pointing at the helper, not at the kernel.
        ring = ka.RING
        ring_pos = ka.RING + ka.VPOS
        ring_pos_quad = ka.RING + ka.VPOS + ka.QUADRIC_EVAL

        self.k_remap = ctx.Kernel(
            "hd_remap_corners", ka.REMAP_CORNERS,
            [("c0", U), ("c1", U), ("c2", U), ("remap", U), ("alive", U)],
            push_constants=INT1,
        )
        self.k_count = ctx.Kernel(
            "hd_adj_count", ka.ADJ_COUNT,
            [("c0", U), ("c1", U), ("c2", U), ("counts", U), ("alive", U)],
            push_constants=INT1,
        )
        self.k_scatter = ctx.Kernel(
            "hd_adj_scatter", ka.ADJ_SCATTER,
            [("c0", U), ("c1", U), ("c2", U), ("offs", U), ("cursor", U),
             ("adj", U), ("alive", U)],
            push_constants=INT1,
        )
        self.k_quadric = ctx.Kernel(
            "hd_build_quadric", ring_pos + ka.BUILD_QUADRIC,
            [("c0", U), ("c1", U), ("c2", U), ("offs", U), ("adj", U),
             ("pos", F4, 2), ("quad", F4, 3), ("vflags", U)],
            defines={"BOUNDARY_WEIGHT": f"{self.boundary_weight:.1f}"},
            push_constants=INT1,
        )

        common = {
            "NO_TARGET_U": f"{kc.NO_TARGET}u",
            "MAX_VALENCE": self.max_valence,
            "BUCKETS": HIST_BUCKETS,
            "FREEZE_BORDERS": 1 if self.freeze_borders else 0,
            "TAKEN_BIT_U": f"{kc.TAKEN_BIT}u",
            "BLOCKED_BIT_U": f"{kc.BLOCKED_BIT}u",
            "NOT_CANDIDATE_U": f"{kc.NOT_CANDIDATE}u",
            "CLAIM_KEY_CLEAR": f"{kc.CLAIM_KEY_CLEAR}u",
            "OPTIMAL_PLACEMENT": 1 if self.optimal_placement else 0,
            "UV_CHECK": 1 if self.check_uvs else 0,
        }
        self.k_sort_rings = ctx.Kernel(
            "hd_sort_rings", ka.SORT_RINGS,
            [("offs", U), ("adj", U)],
            push_constants=INT1,
        )
        self.k_params = ctx.Kernel(
            "hd_write_params", kc.WRITE_PARAMS,
            [("quad", F4, 3), ("vflags", U), ("vlock", U), ("vdensity", F1),
             ("probe_dst", U)],
            defines=common, push_constants=INT1,
        )
        self.k_candidate = ctx.Kernel(
            "hd_candidate", ring_pos_quad + kc.CANDIDATE,
            [("c0", U), ("c1", U), ("c2", U), ("offs", U), ("adj", U),
             ("pos", F4, 2), ("quad", F4, 3), ("cand", F4, 3)],
            defines=common,
            push_constants=(("INT", "n"), ("FLOAT", "FLIP_LIMIT"),
                            ("INT", "SEED")),
        )
        self.k_hist = ctx.Kernel(
            "hd_histogram", kc.HISTOGRAM,
            [("cand", F4, 3), ("hist", U)],
            defines=common,
            push_constants=(("INT", "n"), ("FLOAT", "LOG_LO"),
                            ("FLOAT", "LOG_HI")),
        )
        self.k_accept_clear = ctx.Kernel(
            "hd_accept_clear", kc.ACCEPT_CLEAR,
            [("accept", U, 2)],
            defines=common, push_constants=INT1,
        )
        self.k_accept_key = ctx.Kernel(
            "hd_accept_key", kc.ACCEPT_KEY,
            [("cand", F4, 3), ("accept", U, 2)],
            defines=common, push_constants=INT1,
        )
        self.k_accept_id = ctx.Kernel(
            "hd_accept_id", kc.ACCEPT_ID,
            [("cand", F4, 3), ("accept", U, 2)],
            defines=common, push_constants=INT1,
        )
        self.k_prio = ctx.Kernel(
            "hd_set_priority", kc.SET_PRIORITY,
            [("cand", F4, 3), ("accept", U, 2), ("prio", U)],
            defines=common,
            push_constants=(("INT", "n"), ("FLOAT", "THRESHOLD")),
        )
        self.k_blocked = ctx.Kernel(
            "hd_mark_blocked", ring + kc.MARK_BLOCKED,
            [("c0", U), ("c1", U), ("c2", U), ("offs", U), ("adj", U),
             ("prio", U)],
            defines=common, push_constants=INT1,
        )
        # The claim kernels walk two rings -- the candidate's and its
        # target's -- so they need the candidate record alongside the ring
        # images and the eligibility flags; that is all eight image slots.
        # CONFIRM also needs `winner`, which pushes it past the budget, so its
        # source skips the `prio` check instead: only a candidate that claimed
        # can confirm, and a previous winner re-confirming is the same write.
        claim_images = [("c0", U), ("c1", U), ("c2", U), ("offs", U),
                        ("adj", U), ("cand", F4, 3), ("claim", U, 2),
                        ("prio", U)]
        self.k_claim1 = ctx.Kernel(
            "hd_claim1", ring + kc.CLAIM1, claim_images,
            defines=common, push_constants=INT1,
        )
        self.k_claim2 = ctx.Kernel(
            "hd_claim2", ring + kc.CLAIM2, claim_images,
            defines=common, push_constants=INT1,
        )
        self.k_confirm = ctx.Kernel(
            "hd_confirm", ring + kc.CONFIRM,
            claim_images[:-1] + [("winner", U)],
            defines=common, push_constants=INT1,
        )
        self.k_clear_claim = ctx.Kernel(
            "hd_clear_claim", kc.CLAIM_CLEAR,
            [("claim", U, 2)],
            defines=common, push_constants=INT1,
        )
        self.k_commit = ctx.Kernel(
            "hd_commit_remap", kc.COMMIT_REMAP,
            [("winner", U), ("cand", F4, 3), ("remap", U), ("pos", F4, 2),
             ("accept", U, 2)],
            defines=common, push_constants=INT1,
        )
        self.k_taken = ctx.Kernel(
            "hd_mark_taken", ring + kc.MARK_TAKEN,
            [("c0", U), ("c1", U), ("c2", U), ("offs", U), ("adj", U),
             ("winner", U), ("prio", U), ("cand", F4, 3)],
            defines=common, push_constants=INT1,
        )
        self.k_merge = ctx.Kernel(
            "hd_merge_quadrics", kc.MERGE_QUADRICS,
            [("winner", U), ("cand", F4, 3), ("quad", F4, 3)],
            defines=common, push_constants=INT1,
        )
        if self.log is not None:
            self.k_log = ctx.Kernel(
                "hd_log_append", kc.LOG_APPEND,
                [("winner", U), ("cand", F4, 3), ("vorigin", U),
                 ("rank", U), ("log", U, LOG_LAYERS), ("probe_dst", U)],
                defines=common,
                push_constants=(("INT", "n"), ("INT", "base"), ("INT", "cap"),
                                ("INT", "live")),
            )
        self.k_identity = ctx.Kernel(
            "hd_identity", IDENTITY_SRC,
            [("remap", U)], push_constants=INT1,
        )
        self.k_chain = ctx.Kernel(
            "hd_chain", CHAIN_SRC,
            [("chain", U), ("remap", U)],
            push_constants=INT1,
        )

        # Vertex compaction. See kernels/compaction.py.
        self.k_vmark = ctx.Kernel(
            "hd_mark_verts_alive", kv.MARK_VERTS_ALIVE,
            [("c0", U), ("c1", U), ("c2", U), ("vlive", U)],
            push_constants=INT1,
        )
        self.k_vcompact4 = ctx.Kernel(
            "hd_compact_pos", kv.COMPACT_POS,
            [("pos_src", F4, 2), ("pos_dst", F4, 2), ("vlive", U),
             ("offsets", U)],
            push_constants=INT1,
        )
        self.k_vcompactq = ctx.Kernel(
            "hd_compact_quadric", kv.COMPACT_QUADRIC,
            [("quad_src", F4, 3), ("quad_dst", F4, 3), ("vlive", U),
             ("offsets", U)],
            push_constants=INT1,
        )
        self.k_vrenumber = ctx.Kernel(
            "hd_renumber_corners", kv.RENUMBER_CORNERS,
            [("c0", U), ("c1", U), ("c2", U), ("offsets", U)],
            push_constants=INT1,
        )
        self.k_vchain = ctx.Kernel(
            "hd_renumber_chain", kv.RENUMBER_CHAIN,
            [("chain", U), ("offsets", U)],
            push_constants=(("INT", "n"), ("INT", "live")),
        )

    def _bind(self, *names):
        table = {
            "c0": self.c0, "c1": self.c1, "c2": self.c2,
            "alive": self.alive, "counts": self.counts,
            "offs": self.offs, "cursor": self.cursor, "adj": self.adj,
            "pos": self.pos, "quad": self.quad, "vflags": self.vflags,
            "remap": self.remap, "cand": self.cand, "prio": self.prio,
            "accept": self.accept,
            "winner": self.winner, "hist": self.hist, "vlock": self.vlock,
            "vdensity": self.vdensity, "chain": self.chain,
            "claim": self.claim, "log": self.log, "vorigin": self.vorigin,
        }
        bound = {}
        for name in names:
            array = table[name]
            # `vlock` and `vdensity` are released once write_params has folded
            # them in, so a name can legitimately be absent. Binding None would
            # fail inside Kernel.run on a missing attribute, which says nothing
            # about why.
            if array is None:
                raise RuntimeError(
                    f"{name!r} is not allocated; write_params released it after "
                    "folding it into the quadric. Call write_params again if a "
                    "kernel needs it."
                )
            bound[name] = array
        return bound

    # -------------------------------------------------------------- stages

    def apply_remap(self):
        """Kernel 1. Rewrite corners through `remap` and flag dead triangles."""
        self.k_remap.run(
            self.f_count,
            bind=self._bind("c0", "c1", "c2", "remap", "alive"),
            n=self.f_count,
        )

    def build_adjacency(self):
        """Kernels 3 to 5. Rebuild the CSR from the live triangles."""
        self.counts.clear(0)
        self.cursor.clear(0)
        self.k_count.run(
            self.f_count,
            bind=self._bind("c0", "c1", "c2", "counts", "alive"),
            n=self.f_count,
        )
        self.prims.scan_exclusive(self.counts, self.offs, self.v_count + 1)
        self.k_scatter.run(
            self.f_count,
            bind=self._bind("c0", "c1", "c2", "offs", "cursor",
                            "adj", "alive"),
            n=self.f_count,
        )
        # Row order follows the scatter's atomics, and the quadric build's
        # gather sums in row order, so the sort is what makes the whole pass
        # reproducible; see kernels/adjacency.SORT_RINGS.
        self.k_sort_rings.run(
            self.v_count, bind=self._bind("offs", "adj"), n=self.v_count,
        )

    def build_quadrics(self):
        """Area-weighted surface quadrics plus boundary and seam planes, gathered.

        Does not clear `vflags`: the seam bits placed by `set_seams` are an input
        here, and the kernel preserves them while writing the boundary bit.
        """
        self.quad.clear(0.0)
        self.k_quadric.run(
            self.v_count,
            bind=self._bind("c0", "c1", "c2", "offs", "adj",
                            "pos", "quad", "vflags"),
            n=self.v_count,
        )

    def reset_remap_identity(self):
        """remap[i] = i, on the GPU. See IDENTITY_SRC for why not an upload."""
        self.k_identity.run(
            self.v_count, bind=self._bind("remap"), n=self.v_count,
        )

    def set_seams(self, seams):
        """Upload the per-vertex UV seam mask.

        Must be called before `build_quadrics`, which reads the seam bit out of
        `vflags` to place seam constraint planes and then writes the boundary bit
        back alongside it. Bit 0 is boundary, bit 1 is seam.
        """
        if seams is None:
            self.vflags.clear(0)
            return
        mask = np.asarray(seams, dtype=bool).astype(np.uint32) << np.uint32(1)
        self.vflags.upload(mask)

    def _allocate_param_inputs(self):
        """Allocate `vlock` and `vdensity`, with their neutral defaults.

        Separate from `__init__` because `write_params` releases them again, and
        a caller is still allowed to fold a fresh set of parameters in later.
        """
        if self.vlock is None:
            self.vlock = ctx.Array("hd_vlock", self.v_count, fmt=U)
            self.vlock.clear(0)
        if self.vdensity is None:
            # Uniform default, so a clear says it: an np.ones upload cost a host
            # array and a transfer the size of the vertex count to say the same.
            self.vdensity = ctx.Array("hd_vdensity", self.v_count, fmt=F1)
            self.vdensity.clear(1.0)

    def write_params(self, locked=None, density=None):
        """Fold boundary, lock and density into the quadric's spare channels.

        The inputs are released on the way out; see `__init__`. Calling this again
        allocates them afresh, so changing the parameters mid-session still works
        and simply pays for the arrays again.
        """
        self._allocate_param_inputs()
        if locked is not None:
            self.vlock.upload(np.asarray(locked, dtype=bool).astype(np.uint32))
        if density is not None:
            self.vdensity.upload(np.asarray(density, dtype=np.float32))
        probe = ctx.probe_array()
        bound = self._bind("quad", "vflags", "vlock", "vdensity")
        bound["probe_dst"] = probe
        self.k_params.run(self.v_count, bind=bound, n=self.v_count)
        # The dispatch above is queued, not finished, and it reads both arrays.
        # Dropping them before it runs would hand the kernel freed memory, so the
        # queue is drained first. It has to be read back through something this
        # dispatch wrote, which is what `probe_dst` is for; ctx.sync has the
        # measurements showing why an unrelated read would not wait.
        ctx.sync(probe)
        self.vlock = None
        self.vdensity = None

    def select_candidates(self, flip_limit=None, seed=0):
        """Kernel 6. Best target, its placement and every validity test.

        `flip_limit` is the cosine of the steepest fold allowed. It is a push
        constant rather than a compiled-in define so a caller can vary it
        without paying for a new shader; nothing in the shipped pipeline does,
        but the alternative is a shader recompile per value and this costs
        nothing. `seed` reseeds the target tie-break hash; the pass loop passes
        its pass number, so each pass breaks equal-cost ties differently.
        """
        self.k_candidate.run(
            self.v_count,
            bind=self._bind("c0", "c1", "c2", "offs", "adj", "pos", "quad",
                            "cand"),
            n=self.v_count,
            FLIP_LIMIT=self.flip_limit if flip_limit is None else float(flip_limit),
            SEED=int(seed),
        )

    def choose_threshold(self, admit_fraction):
        """Kernel 7 plus the host side. Returns a cost threshold, or None.

        The only readback inside a pass: 1024 buckets, 4 KB.
        """
        self.hist.clear(0)
        self.k_hist.run(
            self.v_count, bind=self._bind("cand", "hist"),
            n=self.v_count, LOG_LO=HIST_LOG_LO, LOG_HI=HIST_LOG_HI,
        )
        counts = self.hist.download(HIST_BUCKETS).astype(np.int64)
        total = int(counts.sum())
        if total == 0:
            return None, 0
        want = max(1, int(total * admit_fraction))
        bucket = int(np.searchsorted(np.cumsum(counts), want))
        bucket = min(bucket, HIST_BUCKETS - 1)
        # Upper edge of the chosen bucket, so everything counted is admitted.
        span = HIST_LOG_HI - HIST_LOG_LO
        exponent = HIST_LOG_LO + (bucket + 1) / HIST_BUCKETS * span
        return float(10.0 ** exponent), total

    def independent_set(self, threshold, rounds=4, seed=0):
        """The claim rounds. Returns nothing; winners are flagged.

        Each round: block the candidates whose ring touches a committed
        reservation, claim both rings' faces of the rest by cost key and then
        owner id, confirm the candidates that own every face they claimed, and
        reserve the winners' neighbourhoods for the later rounds. The claim
        arrays are cleared once per pass rather than per round: a key left
        behind by a candidate that never won keeps holding its faces against
        later rounds' claimers, which can only lower the yield, never let two
        collapses share a face.
        """
        self.winner.clear(0)
        self.k_clear_claim.run(
            self.f_count, bind=self._bind("claim"), n=self.f_count,
        )
        # The pairing pass: each vertex accepts the cheapest proposal made to
        # it, turning the per-vertex proposals into candidate pairs before
        # the threshold and the claims decide which of them run.
        self.k_accept_clear.run(
            self.v_count, bind=self._bind("accept"), n=self.v_count,
        )
        self.k_accept_key.run(
            self.v_count, bind=self._bind("cand", "accept"), n=self.v_count,
        )
        self.k_accept_id.run(
            self.v_count, bind=self._bind("cand", "accept"), n=self.v_count,
        )
        self.k_prio.run(
            self.v_count, bind=self._bind("cand", "accept", "prio"),
            n=self.v_count, THRESHOLD=threshold,
        )
        for _ in range(rounds):
            self.k_blocked.run(
                self.v_count,
                bind=self._bind("c0", "c1", "c2", "offs", "adj", "prio"),
                n=self.v_count,
            )
            self.k_claim1.run(
                self.v_count,
                bind=self._bind("c0", "c1", "c2", "offs", "adj", "cand",
                                "claim", "prio"),
                n=self.v_count,
            )
            self.k_claim2.run(
                self.v_count,
                bind=self._bind("c0", "c1", "c2", "offs", "adj", "cand",
                                "claim", "prio"),
                n=self.v_count,
            )
            self.k_confirm.run(
                self.v_count,
                bind=self._bind("c0", "c1", "c2", "offs", "adj", "cand",
                                "claim", "winner"),
                n=self.v_count,
            )
            # Reserve this round's winners so the next round works around them.
            self.k_taken.run(
                self.v_count,
                bind=self._bind("c0", "c1", "c2", "offs", "adj", "winner",
                                "prio", "cand"),
                n=self.v_count,
            )

    def commit(self):
        """Write the step remap for the winners and merge their quadrics."""
        # No identity seed needed: the commit kernel writes every vertex.
        self.k_commit.run(
            self.v_count,
            bind=self._bind("winner", "cand", "remap", "pos", "accept"),
            n=self.v_count,
        )
        self.k_merge.run(
            self.v_count, bind=self._bind("winner", "cand", "quad"),
            n=self.v_count,
        )
        # Over `v_origin`, not `v_count`. `chain` is the one per-vertex array
        # indexed by the *original* numbering, so it keeps its full length for
        # the life of the session while every other per-vertex array shrinks.
        #
        # Running it over `v_count` was correct for exactly as long as the two
        # were equal, which was until vertex compaction landed. After the first
        # compaction it left every entry past the live count frozen at whatever
        # it last pointed at, so an original vertex whose representative later
        # collapsed went on naming a vertex that no longer existed. On the 13M
        # head scan reduced to 500,000 that was 2,313,817 of 6,521,569 entries,
        # and it surfaced as a negative remap rather than as anything resembling
        # its cause.
        self.k_chain.run(
            self.v_origin, bind=self._bind("chain", "remap"),
            n=self.v_origin,
        )

    def log_pass(self, live_faces):
        """Append this pass's collapses to the log, in original numbering.

        Must run after `compact_faces` -- the live face count it records is the
        pass's result -- and before any vertex compaction, which renumbers the
        `vorigin` array the records are written in. The records are placed by a
        scan of the winner flags, so the order within a pass is vertex order and
        the log is the same whatever the scheduler did.

        A pass whose records do not fit the buffer marks the log truncated and
        stops trying: the run still finishes correctly, it just cannot be
        replayed to a target below wherever the log gave up.
        """
        if self.log is None or self.log_truncated:
            return
        rank, won = self.prims.scan_flags(self.winner, self.v_count)
        if self.log_count + won > self.log.count:
            self.log_truncated = True
            return
        probe = ctx.probe_array()
        bound = self._bind("winner", "cand", "vorigin", "log")
        bound["rank"] = rank
        bound["probe_dst"] = probe
        self.k_log.run(
            self.v_count, bind=bound,
            n=self.v_count, base=self.log_count, cap=self.log.count,
            live=int(live_faces),
        )
        # The probe is what the log kernel wrote, so reading it both drains the
        # dispatch and proves it ran; see context.sync.
        ctx.sync(probe)
        self.log_count += won

    def compact_faces(self):
        """Drop dead triangles, keeping face provenance alongside. Returns the
        new face count."""
        offs, kept = self.prims.scan_flags(self.alive, self.f_count)
        self.prims.apply_compact_three(
            (self.c0, self.c1, self.c2), (self.c0b, self.c1b, self.c2b),
            self.alive, offs, self.f_count,
        )
        self.prims.apply_compact_one(
            self.forigin, self.foriginb, self.alive, offs, self.f_count,
        )
        # Swap rather than copy back; compaction cannot write where it reads.
        self.c0, self.c0b = self.c0b, self.c0
        self.c1, self.c1b = self.c1b, self.c1
        self.c2, self.c2b = self.c2b, self.c2
        self.forigin, self.foriginb = self.foriginb, self.forigin
        self.f_count = kept

        # Every surviving face is alive by construction, so the flags have to be
        # reset. Leaving them is a genuinely nasty bug: `alive` still holds the
        # pre-compaction pattern, so the next adjacency rebuild skips real faces
        # wherever a zero happens to land below the new face count. The rings it
        # then reports are incomplete, and an incomplete ring makes the link
        # condition and the normal-flip test pass collapses they should reject.
        # On a 13M-triangle scan that produced two non-manifold edges and two
        # duplicate faces out of nine million, appearing only at pass three, which
        # is about as thin a thread as a correctness bug can hang by.
        self.alive.fill(1)
        return kept

    def count_winners(self):
        won = int(self.winner.download(self.v_count).sum())
        # Each winner collapses itself onto its target, so each one kills
        # exactly one vertex. This is what decides when to compact, without a
        # scan; see __init__.
        self._dead_since_compaction += won
        return won

    def should_compact_vertices(self, fraction=0.5):
        """Whether enough vertices have died to be worth gathering up.

        Compaction is not free -- it is a scan, five dispatches and a round of
        texture allocation -- so it has to be amortised rather than run every
        pass. The threshold is a halving, which bounds two things at once: the
        number of compactions over a whole run at log2 of the vertex count,
        about twenty-three in the worst case, and the transient memory peak at
        roughly 1.5x the vertex arrays rather than 2x, because the destination
        is allocated at the live size while the source is still held.
        """
        return self._dead_since_compaction >= fraction * self.v_count

    def compact_vertices(self):
        """Gather the live vertices to the front and renumber everything.

        Returns the new live vertex count. The caller must have compacted the
        faces first, because "live" is defined as "named by a surviving
        triangle", and must rebuild the adjacency afterwards, because every
        index in it has just changed.
        """
        vlive = ctx.Array("hd_vlive", self.v_count, fmt=U)
        vlive.clear(0)
        self.k_vmark.run(
            self.f_count,
            bind={"c0": self.c0, "c1": self.c1, "c2": self.c2,
                  "vlive": vlive},
            n=self.f_count,
        )
        offs, kept = self.prims.scan_flags(vlive, self.v_count)
        if kept == 0 or kept == self.v_count:
            # Nothing to gain, and `kept == 0` would leave zero-length arrays
            # for the pass loop to dispatch over. Either way the counter has to
            # be cleared or every later pass would retry this immediately.
            self._dead_since_compaction = 0
            return self.v_count

        # Renumber before the sources are replaced: the corners and the chain
        # are rewritten in place through `offs`, which is still indexed by the
        # old numbering.
        self.k_vrenumber.run(
            self.f_count,
            bind={"c0": self.c0, "c1": self.c1, "c2": self.c2,
                  "offsets": offs},
            n=self.f_count,
        )
        self.k_vchain.run(
            self.v_origin,
            bind={"chain": self.chain, "offsets": offs},
            n=self.v_origin, live=kept,
        )

        # The three arrays that carry state across passes.
        pos_dst = ctx.Array("hd_pos", kept, fmt=F4, layers=2)
        self.k_vcompact4.run(
            self.v_count,
            bind={"pos_src": self.pos, "pos_dst": pos_dst,
                  "vlive": vlive, "offsets": offs},
            n=self.v_count,
        )
        quad_dst = ctx.Array("hd_quad", kept, fmt=F4, layers=3)
        self.k_vcompactq.run(
            self.v_count,
            bind={"quad_src": self.quad, "quad_dst": quad_dst,
                  "vlive": vlive, "offsets": offs},
            n=self.v_count,
        )
        vflags_dst = ctx.Array("hd_vflags", kept, fmt=U)
        self.prims.apply_compact_one(self.vflags, vflags_dst, vlive, offs,
                                     self.v_count)
        vorigin_dst = ctx.Array("hd_vorigin", kept, fmt=U)
        self.prims.apply_compact_one(self.vorigin, vorigin_dst, vlive, offs,
                                     self.v_count)

        # The gathers above are queued, not finished, and the next statement
        # drops the last reference to the arrays they are reading. Drain first;
        # `context.sync` has the measurements showing why it has to be a read
        # of something this work wrote, and this is the last of the four.
        ctx.sync(vorigin_dst)
        self.pos, self.quad = pos_dst, quad_dst
        self.vflags, self.vorigin = vflags_dst, vorigin_dst

        # The per-pass scratch carries nothing between passes, so it is
        # reallocated at the new size rather than gathered. This is where most
        # of the memory comes back: nine arrays against three.
        self.v_count = kept
        self.cand = ctx.Array("hd_cand", kept, fmt=F4, layers=3)
        self.prio = ctx.Array("hd_prio", kept, fmt=U)
        self.winner = ctx.Array("hd_winner", kept, fmt=U)
        self.remap = ctx.Array("hd_remap", kept, fmt=U)
        self.counts = ctx.Array("hd_counts", kept + 1, fmt=U)
        self.offs = ctx.Array("hd_offs", kept + 1, fmt=U)
        self.cursor = ctx.Array("hd_cursor", kept, fmt=U)

        self._dead_since_compaction = 0
        self.compactions += 1
        return kept

    # ---------------------------------------------------------- readback

    def read_triangles(self):
        return np.stack([
            self.c0.download(self.f_count),
            self.c1.download(self.f_count),
            self.c2.download(self.f_count),
        ], axis=1)

    def read_adjacency(self):
        offs = self.offs.download(self.v_count + 1)
        return offs, self.adj.download(int(offs[-1]))

    def read_quadrics(self):
        """(V, 10) in the same coefficient order as core.quadrics."""
        a, b, c = (ctx.download_layer(self.quad, i) for i in range(3))
        out = np.empty((self.v_count, 10), dtype=np.float64)
        out[:, 0:4] = a[:, :4]
        out[:, 4:8] = b[:, :4]
        out[:, 8:10] = c[:, :2]
        return out

    def read_boundary_flags(self):
        return self.vflags.download(self.v_count).astype(bool)

    def read_candidates(self):
        """(best_target, best_cost) with -1 where there is no legal collapse."""
        raw = ctx.download_layer(self.cand, 0)[:self.v_count]
        target = raw[:, 1].view(np.uint32).astype(np.int64)
        cost = raw[:, 0].astype(np.float64)
        none = target == kc.NO_TARGET
        target = np.where(none, -1, target)
        cost = np.where(none, np.inf, cost)
        return target, cost

    def read_reject_reasons(self):
        """Tally of why each vertex had no candidate. See kernels/candidates.py."""
        raw = ctx.download_layer(self.cand, 0)[:self.v_count]
        codes = raw[:, 2].astype(np.int64)
        names = {0: "accepted", 1: "locked", 2: "ring size", 3: "no target",
                 4: "valence", 5: "link condition", 6: "normal flip",
                 7: "shared faces", 8: "uv fold"}
        return {names.get(int(c), f"code {int(c)}"): int((codes == c).sum())
                for c in np.unique(codes)}

    def read_log(self):
        """The collapse records as a structured array, or None.

        One row per committed collapse, in original vertex numbering:
        `a` the vertex that died, `b` the survivor, `p` the placement it moved
        to, `removed` the faces the collapse deleted, `live` the face count
        after the whole pass, `rank` the record's order within its pass. The
        per-row `removed` and `live` together give the running face count a
        replay needs to stop at any target, including one inside a pass.
        """
        if self.log is None or self.log_count == 0:
            return None
        rows = np.empty(self.log_count, dtype=core_replay.LOG_DTYPE)
        for field, layer in (("a", 0), ("b", 1),
                            ("removed", 5), ("live", 6), ("rank", 7)):
            rows[field] = ctx.download_layer(self.log, layer)[:self.log_count]
        # The placement layers hold float bits in a uint image: view them as
        # float32 *before* assigning, or numpy converts the integer value and
        # the placement comes back as ~3e9 instead of ~0.5.
        for field, layer in (("px", 2), ("py", 3), ("pz", 4),
                             ("pu", 8), ("pv", 9)):
            rows[field] = ctx.download_layer(
                self.log, layer)[:self.log_count].view(np.float32)
        if self.log_truncated:
            return None
        return rows

    def read_chain(self):
        """Original vertex -> its current index. Length is the *original* count.

        `v_origin`, not `v_count`: `chain` is the one per-vertex array indexed
        by the original numbering, so vertex compaction shrinks every other
        array and leaves this one alone. Downloading `v_count` of it quietly
        returned a prefix once compaction landed.
        """
        return self.chain.download(self.v_origin).astype(np.int64)

    def read_vertex_origin(self):
        """Current vertex -> the original vertex it is. See `vorigin`."""
        return self.vorigin.download(self.v_count).astype(np.int64)

    def read_face_origin(self):
        return self.forigin.download(self.f_count).astype(np.int64)
