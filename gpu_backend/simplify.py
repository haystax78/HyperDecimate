"""The entry point. PLAN.md 4.4, 4.6.

Exposed two ways over the same code:

`simplify()` runs to completion and returns a `SimplifyResult`.

`Session` runs one pass at a time. The operator needs that: a 13M-triangle mesh
takes about six seconds of GPU work, and a modal operator stepping one pass per
timer tick gives a real progress bar and a working Escape key, where a single
blocking call would just freeze Blender. The pass loop was already a sequence
of discrete passes, so this costs nothing but a little plumbing.

The only readback inside the loop is the cost histogram, 4 KB per pass.

**The floor and the log.** A session may be asked to run past its target, to a
`floor` below it, while logging every collapse. The result it returns is then a
replay of that log back up to the target -- which is exactly the prefix of the
loop the target asked for, down to the mid-pass record -- and the log itself is
what a later run at a different target replays instead of recomputing. See
`core/replay.py` and `core/cache.py`.
"""

from __future__ import annotations

import numpy as np

from ..core import quadrics as qd
from ..core import replay as rp
from ..core import topology as tp
from ..core.options import Options
from ..core.result import SimplifyResult
from . import context as ctx
from .state import GPUState


class Session:
    """A decimation in progress, advanced one pass at a time."""

    def __init__(self, positions, triangles, target_triangles, opts=None,
                 floor=None):
        self.opts = opts or Options()
        # Kept in whatever dtype the caller read them in. These are only held
        # for their shapes and to hand to GPUState, which wants float32 and
        # uint32, so promoting them to float64 and int64 here was a pair of
        # full-size copies that nothing ever read -- 1.9 GB on a 52M-triangle
        # mesh, alive for the whole session.
        self.positions = np.ascontiguousarray(positions)
        self.triangles = np.ascontiguousarray(triangles)
        self.original_count = self.positions.shape[0]
        self.target = int(target_triangles)
        # The loop stops at the floor, not the target, when a log is wanted:
        # the extra passes below the target are what make coarser targets
        # replayable later. Without logging there is nothing to gain by
        # running past the target, so the floor is the target.
        self.floor = int(floor) if floor is not None else self.target
        self.start_faces = self.triangles.shape[0]

        self.passes = 0
        self.schedule = []
        self.done = False

        self.state = GPUState(
            self.positions, self.triangles,
            boundary_weight=self.opts.boundary_weight,
            max_normal_flip_deg=self.opts.max_normal_flip_deg,
            max_valence=self.opts.max_valence,
            freeze_borders=self.opts.freeze_borders,
            optimal_placement=self.opts.optimal_placement,
            uvs=self.opts.uvs,
        )
        self.stop_at = self.floor if self.state.log is not None else self.target
        # The adjacency must exist before the quadric build, which gathers
        # through it, and before write_params, which needs its boundary flags.
        self.state.build_adjacency()
        # Seams are an input to the quadric build, so they go up first.
        self.state.set_seams(self.opts.seams)
        self.state.build_quadrics()
        self.state.write_params(locked=self.opts.locked,
                                density=self.opts.density)

    @property
    def face_count(self):
        return self.state.f_count

    @property
    def progress(self):
        """0..1, by how far the face count has moved toward the stopping point."""
        span = self.start_faces - self.stop_at
        if span <= 0:
            return 1.0
        moved = self.start_faces - self.state.f_count
        return float(min(max(moved / span, 0.0), 1.0))

    def step(self):
        """Run one pass. Returns True while there is more to do."""
        if self.done:
            return False
        if (self.state.f_count <= self.stop_at
                or self.passes >= self.opts.max_passes):
            self.done = True
            return False

        self.passes += 1
        st = self.state
        st.select_candidates(seed=self.passes - 1)
        threshold, candidates = st.choose_threshold(self.opts.admit_fraction)
        if threshold is None:
            self.done = True
            return False
        if self.opts.max_error is not None:
            threshold = min(threshold, float(self.opts.max_error))

        st.independent_set(threshold, rounds=self.opts.claim_rounds)
        won = st.count_winners()
        if won == 0:
            # Nothing survives the validity tests, so this is as coarse as the
            # collapse rules allow. Not a failure: the replay below simply
            # returns what the log already reached.
            self.done = True
            return False

        self.schedule.append((self.passes - 1, candidates, won, st.f_count))
        st.commit()
        st.apply_remap()
        kept = st.compact_faces()
        # The log records the pass's result, so it sits between the face
        # compaction -- which is what tells it how many faces are left -- and
        # any vertex compaction, which renumbers the origins it writes.
        st.log_pass(kept)
        # Vertices, once enough of them have died to amortise it. This has to
        # sit between the face compaction and the adjacency rebuild: "live"
        # means "named by a surviving triangle", so the faces must be settled
        # first, and every index the adjacency holds is about to change.
        #
        # It is what stops a pass costing the same when the mesh is a hundred
        # times smaller. kernels/compaction.py has the measurement.
        if self.opts.compact_vertices and st.should_compact_vertices():
            st.compact_vertices()
        st.build_adjacency()
        return True

    def finish(self):
        """Bring the result back to the host."""
        st = self.state
        log = st.read_log()
        if log is not None:
            return self._finish_from_log(log)

        # No log: the loop stopped at the target itself and the mesh on the
        # device is the answer, in the working (normalised) space.
        chain = st.read_chain()
        vertex_origin = st.read_vertex_origin()
        tri_current = st.read_triangles().astype(np.int64)
        face_origin = st.read_face_origin()
        work_pos = ctx.download_layer(st.pos, 0)[:st.v_count, :3].astype(
            np.float64)

        out_pos, out_tris, old_to_new = tp.compact_vertices(work_pos, tri_current)
        out_uv = None
        if st.check_uvs:
            cur_uv = ctx.download_layer(st.pos, 1)[:st.v_count, :2]
            out_uv = np.zeros((out_pos.shape[0], 2), dtype=np.float32)
            live = old_to_new >= 0
            out_uv[old_to_new[live]] = cur_uv[live]
        remap = old_to_new[chain]
        # A vertex whose representative lost every face has no output vertex
        # to name; the contract says remap is never negative, and replay.py
        # makes the same choice for the same reason.
        remap = np.where(remap < 0, 0, remap)

        # Which original vertices are present in their own right.
        #
        # Answered by construction rather than by comparing the chain against
        # the identity, which compaction's renumbering quietly broke; see
        # `GPUState.vorigin`.
        survived = np.zeros(self.original_count, dtype=bool)
        kept = old_to_new >= 0
        survived[vertex_origin[kept]] = True

        out_pos = qd.denormalize_positions(out_pos, st.centre, st.scale)
        return SimplifyResult.build(out_pos, out_tris, remap, survived,
                                    face_origin, uvs=out_uv)

    def _finish_from_log(self, log):
        """The target's result, as a prefix of the recorded log."""
        st = self.state
        # The normalised working positions, rebuilt exactly as the upload
        # built them: the same centre and scale, the same float64 arithmetic
        # rounded once into float32. The log's placements were recorded in
        # this space, so the replay of them lands on the same values the loop
        # itself produced. Only the survivors are normalised; see `rp.replay`.
        def to_work(p):
            return ((p - st.centre) * st.scale).astype(np.float32)

        pos, tris, remap, survived, face_origin, uv = rp.replay(
            self.positions, self.triangles, log, self.target,
            to_work=to_work, uvs=self.opts.uvs)
        out_pos = qd.denormalize_positions(pos.astype(np.float64),
                                           st.centre, st.scale)
        return SimplifyResult.build(out_pos, tris, remap, survived,
                                     face_origin, uvs=uv)

    @property
    def log(self):
        """The recorded collapse log, with everything a replay needs.

        None when this run did not log: the mesh was past the log's memory
        ceiling, or the loop recorded nothing. `floor_faces` is how far the
        log reaches; a target below it cannot be replayed and a target above
        the base the run started from cannot be either.
        """
        st = self.state
        records = st.read_log() if st is not None else None
        if records is None:
            return None
        return {
            "records": records,
            "base_faces": self.start_faces,
            "floor_faces": st.f_count,
            "centre": st.centre,
            "scale": st.scale,
            "positions_dtype": np.dtype(np.float32),
        }

    def release(self):
        """Drop GPU objects. See context.release_all for why this matters."""
        self.state = None


def simplify(positions, triangles, target_triangles, opts=None, stats=None,
             floor=None):
    """Reduce `triangles` toward `target_triangles` on the GPU.

    Returns a `SimplifyResult`. Pass a dict as `stats` to also receive the
    pass count and per-pass diagnostics, and a `floor` below the target to
    record a replayable log down to (see `Session`).
    """
    session = Session(positions, triangles, target_triangles, opts, floor=floor)
    while session.step():
        pass
    result = session.finish()
    if stats is not None:
        stats["passes"] = session.passes
        stats["schedule"] = session.schedule
        stats["log"] = session.log
    return result
