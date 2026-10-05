"""Replay a collapse log to any target. The reference pipeline's preprocess.

**What the log is.** Every committed collapse of a run, in order, one record
per collapse: the vertex that died, the survivor it died into, the position
the survivor was moved to, how many faces the collapse removed, the live
face count at the end of that record's pass, and the record's rank within the
pass. All of it in the *base* numbering -- the ids of the mesh the collapse
stage started from -- because the pass loop compacts and renumbers as it goes
and the log must not care.

**What replay is.** The pass loop's decisions are already in the log; replay
is the cheap part of the loop without the expensive part. Each pass's records
are an independent set, so they can be applied in bulk: rewrite the dying
vertices onto their survivors, move the survivors to their recorded
placements, and drop the faces whose corners met. A prefix of a pass is as
valid as a whole one -- any subset of an independent set is still independent
-- and the per-record removal counts give the running face count through the
pass, so a target that lands between two passes is hit exactly rather than
rounded to the nearest pass.

That is what makes "preprocess once, decimate to any target afterwards" work:
the first run of a mesh goes to a floor below any target the user is likely
to ask next, and every later target between the floor and the base is a
prefix of the same log, applied in milliseconds instead of passes.

**Positions stay in the collapse stage's working space.** The log's
placements were computed on the normalised mesh, so replay is handed the
normalised positions and returns normalised ones; denormalising is the
caller's business and both backends already do it their own precision.

Nothing here imports a backend, and neither backend's result assembly is
repeated: `replay` returns plain arrays in the `SimplifyResult` field order
and the caller wraps them.
"""

from __future__ import annotations

import numpy as np

# One collapse record. The layout mirrors the reference pipeline's own
# 32-byte log entry -- two vertex ids, three placement floats, and three
# counts -- plus the UV the survivor was given, when UVs are tracked. `live` is the pass's, not the record's -- every record of a pass
# carries the same value, which is also how pass boundaries are found.
LOG_DTYPE = np.dtype([
    ("a", np.uint32),      # the vertex that died, base id
    ("b", np.uint32),      # the survivor, base id
    ("px", np.float32),    # where the survivor was moved to
    ("py", np.float32),
    ("pz", np.float32),
    ("removed", np.uint32),  # faces this collapse deleted
    ("live", np.uint32),     # live faces after this record's whole pass
    ("rank", np.uint32),     # the record's order within its pass
    ("pu", np.float32),      # the survivor's UV at its placement; zero when
    ("pv", np.float32),      # the run did not track UVs
])


def make_log(a, b, p, removed, live, rank, uv=None):
    """Assemble a log from column arrays. Columns may be shorter than the
    record count of the largest dtype; everything is cast, not checked."""
    n = len(a)
    out = np.zeros(n, dtype=LOG_DTYPE)
    if uv is not None:
        out["pu"] = uv[:, 0]
        out["pv"] = uv[:, 1]
    out["a"] = a
    out["b"] = b
    out["px"] = p[:, 0]
    out["py"] = p[:, 1]
    out["pz"] = p[:, 2]
    out["removed"] = removed
    out["live"] = live
    out["rank"] = rank
    return out


def replay(positions, triangles, log, target, to_work=None, uvs=None):
    """Apply `log` to the mesh it was recorded from, down to `target`.

    Returns (positions, triangles, remap, survived, face_origin) in the
    `SimplifyResult` field order: positions in the caller's working space,
    everything in base numbering folded onto the surviving vertices, and
    `face_origin` naming the base face each output face descends from.

    `to_work`, when given, maps rows of `positions` into the working space
    the log's placements were recorded in. It is applied only to the vertices
    that survive, which on a deep replay is a few percent of the mesh, so the
    caller need not normalise the whole source to read a fraction of it back.
    It must act row by row.

    `uvs`, the (V, 2) per-vertex UVs the run was recorded with, makes the
    replay carry them as well: a survivor takes the UV its last collapse
    recorded, any other vertex keeps its own. The sixth return value is then
    the output's per-vertex UVs, and None without them.

    A `target` below what the log reaches is not an error: the replay simply
    stops where the log does, and the caller -- who chose the floor the log
    was recorded to -- decides what to do about it.
    """
    positions = np.asarray(positions)
    vertex_count = positions.shape[0]
    tri = np.asarray(triangles)
    if tri.dtype not in (np.int32, np.int64):
        tri = tri.astype(np.int64)
    # Indices stay at the width they came in at, and int32 where the vertex
    # count allows: the corner gather below touches every corner of the
    # source, and int64 doubles its traffic for nothing.
    index = np.int32 if vertex_count < 2**31 else np.int64
    faces = tri.shape[0]

    # Which records apply is decided from the log alone. Each record carries
    # the faces it removed and each pass the live count after it, so the
    # running face count -- and with it the cut -- is known without touching
    # the mesh. The mesh is then rewritten once, at the end, rather than once
    # per pass: per-pass rewrites of every corner and every vertex cost 30 s
    # of a 31 s replay on a 13M-face scan.
    records = np.asarray(log)
    lives = records["live"]
    # A pass is a run of equal `live`.
    bounds = np.concatenate((
        [0], np.flatnonzero(lives[1:] != lives[:-1]) + 1, [records.shape[0]]))
    chosen = []
    for i, j in zip(bounds[:-1].tolist(), bounds[1:].tolist()):
        if faces <= target:
            break
        live = int(lives[i])

        # The records of one pass, in the order they were ranked. The scan
        # that placed them already wrote them in rank order; sorting again is
        # a no-op that keeps the replay honest about its own contract.
        idx = i + np.argsort(records["rank"][i:j], kind="stable")

        removed = records["removed"][idx].astype(np.int64)
        before = live + int(removed.sum())
        # How many records this pass may apply. The first prefix whose
        # cumulative removal brings the count to the target lands the mesh
        # at or just under it, which is the same contract the pass loop's
        # own stopping gives -- a fresh run overshoots downwards by whatever
        # its last pass removes in one go, and the replay has the finer stop.
        # A pass that cannot reach the target applies in full.
        cum = np.cumsum(removed)
        m = int(np.searchsorted(cum, before - target, side="left")) + 1
        m = min(m, idx.size)
        if m <= 0:
            break
        chosen.append(idx[:m])
        faces = before - int(cum[m - 1])

    # Each vertex's representative: the vertex it ends up folded into. A
    # pass's survivors are alive while it runs and its dying vertices are
    # distinct from them, so walking the passes backwards, every survivor's
    # own fate is already settled when its pass is reached, and one gather
    # per record settles the vertex that died into it.
    rep = np.arange(vertex_count, dtype=index)
    for idx in reversed(chosen):
        rep[records["a"][idx]] = rep[records["b"][idx]]

    # A face dies when two of its corners meet, and corners that have met
    # stay met under every later collapse, so remapping once through the
    # composed `rep` drops exactly the faces the passes dropped, in order.
    tri = rep[tri]
    alive = (
        (tri[:, 0] != tri[:, 1])
        & (tri[:, 1] != tri[:, 2])
        & (tri[:, 2] != tri[:, 0])
    )
    face_origin = np.flatnonzero(alive)
    tri = tri[face_origin]

    # Drop unreferenced vertices and renumber, mirroring the pass loop's own
    # compaction.
    used = np.zeros(vertex_count, dtype=bool)
    used[tri.reshape(-1)] = True
    keep = np.flatnonzero(used)
    old_to_new = np.full(vertex_count, -1, dtype=index)
    old_to_new[keep] = np.arange(keep.size, dtype=index)
    tri = old_to_new[tri]

    # The caller's working precision is kept, and the recorded placements are
    # cast into it. The log stores its placements as float32 -- the collapse
    # stage's own working precision on the GPU -- so a float64 caller's replay
    # positions are rounded to what the log actually recorded, which is the
    # honest answer: the replay cannot know more than the log does.
    pos = positions[keep]
    if to_work is not None:
        pos = to_work(pos)
    uv = None
    if uvs is not None:
        uv = np.asarray(uvs, dtype=np.float32).reshape(-1, 2)[keep]
    # A survivor placed more than once keeps its last placement: with
    # repeated indices the last write wins, and `order` is application order.
    if chosen:
        order = np.concatenate(chosen)
        last = np.full(vertex_count, -1, dtype=np.int64)
        last[records["b"][order]] = order
        placed = last[keep]
        hit = np.flatnonzero(placed >= 0)
        rows = records[placed[hit]]
        pos[hit] = np.stack([rows["px"], rows["py"], rows["pz"]],
                            axis=1).astype(pos.dtype, copy=False)
        if uv is not None:
            uv[hit] = np.stack([rows["pu"], rows["pv"]], axis=1)

    # A vertex survived in its own right exactly when nothing ever collapsed
    # into it and it still has a face; the latter is what compaction's
    # old_to_new says, and relying on it rather than on `rep == identity`
    # alone keeps the survivor count equal to the output vertex count.
    survived = (rep == np.arange(vertex_count, dtype=index)) & (old_to_new >= 0)

    # Every original vertex still names a real output vertex. A vertex whose
    # representative lost every face has no geometry to inherit, so where it
    # lands is arbitrary -- vertex 0, matching nothing in particular, which
    # is exactly what the collapse itself decided about such vertices.
    remap = old_to_new[rep]
    remap = np.where(remap < 0, 0, remap).astype(np.int32)

    return pos, tri, remap, survived, face_origin, uv


def from_log(log, positions, triangles, target, uvs=None):
    """The SimplifyResult of replaying a cached log to `target`.

    `log` is what a session's `log` property returns and `core.cache` kept:
    the records plus the normalisation they were written in. `positions` and
    `triangles` are the source mesh -- the cache key proved it is the mesh the
    log was recorded against. `uvs` are its per-vertex UVs, when the run
    tracked them; the cache key covers those too.
    """
    from . import quadrics as qd
    from .result import SimplifyResult

    # The tier's normalised working positions, rebuilt as it built them:
    # float64 normalisation rounded once into the working dtype. Casting
    # before normalising would round the source first, which is not what the
    # backend did. Only the survivors are ever normalised; see `replay`.
    def to_work(p):
        return ((p - log["centre"]) * log["scale"]).astype(
            log["positions_dtype"])

    pos, tris, rem, surv, forg, uv = replay(
        positions, triangles, log["records"], target, to_work=to_work,
        uvs=uvs)
    out = qd.denormalize_positions(pos.astype(np.float64),
                                   log["centre"], log["scale"])
    return SimplifyResult.build(out, tris, rem, surv, forg, uvs=uv)
