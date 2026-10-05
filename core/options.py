"""The collapse tunables. Defaults are the ones the reference pipeline argues
for.
"""

from __future__ import annotations

from . import quadrics as qd


class Options:
    """Tunables for one collapse run.

    Per-vertex arrays (`seams`, `locked`, `density`) are inputs, not settings:
    they steer the sequence the same way the scalars do, which is why the
    session cache keys them into the mesh fingerprint rather than here.
    """

    def __init__(
        self,
        freeze_borders=False,
        boundary_weight=qd.BOUNDARY_WEIGHT,
        max_normal_flip_deg=78.46304096718453,
        max_valence=48,
        check_link_condition=True,
        seams=None,
        admit_fraction=0.10,
        claim_rounds=8,
        max_error=None,
        max_passes=600,
        optimal_placement=True,
        locked=None,
        density=None,
        compact_vertices=True,
        uvs=None,
    ):
        # Off by default, and that default changed after it was reported as a
        # bug. Freezing is a *hard lock*: a frozen border vertex can never
        # collapse, so on a scanned bust whose neck is an open boundary the
        # border keeps every one of its original vertices while the interior
        # coarsens around it.
        #
        # Off, the boundary is still *held*: constraint planes keep it from
        # being dragged inward, and a boundary vertex may only collapse along
        # its own boundary, so the border keeps its path and merely gets fewer
        # vertices along it. That is what Blender's own Decimate does, and
        # what the reference pipeline does.
        self.freeze_borders = freeze_borders
        self.boundary_weight = boundary_weight
        # Reject a collapse that would turn a face normal past this angle.
        # The reference's --flip is the cosine of it: its default 0.2 is
        # 78.463 degrees here.
        self.max_normal_flip_deg = max_normal_flip_deg
        # The cap is on the merged ring: both one-rings less the faces the
        # collapse removes. The reference's own default is 48.
        self.max_valence = max_valence
        # Topological safeguard. Leave this on. Turning it off makes each pass
        # cheaper and produces meshes that Blender has to repair on load.
        self.check_link_condition = check_link_condition
        # Per-vertex UV seam mask, from core.seams. A seam vertex may only
        # collapse to another seam vertex, so the seam can shorten along
        # itself but never be dragged off its path and tear the texture.
        self.seams = seams
        # Fraction of ranked candidates admitted per pass, cheapest first.
        # 0.10: lower starves the tail of the mesh and stalls above the target;
        # higher trades error for a little speed.
        self.admit_fraction = admit_fraction
        # Rounds of claims per pass. Eight; sixteen produces a bit-identical
        # result, so the independent set is saturated by then.
        self.claim_rounds = claim_rounds
        self.max_error = max_error
        # Where the merged vertex goes: the minimiser of the pair quadric
        # (the reference's own rule), or the edge midpoint -- the reference's
        # --centroid control. The locked and flagged-endpoint rules apply
        # either way.
        self.optimal_placement = optimal_placement
        # Hard stop, not a schedule. Stopping here means finishing short of
        # the target silently, which is exactly the bug that 160 caused: a 13M
        # scan reduced to a 10,000-triangle target ran out of budget at 11,636
        # while its last pass still had 2,000 usable candidates.
        #
        # Late passes are nearly free, because by then the mesh is tiny -- 160
        # passes cost 3.0 s against 2.3 s for 100 -- so the budget is set by
        # what an aggressive target actually needs rather than by what is
        # typical. Reaching 1,000 triangles from 13M takes about 250.
        self.max_passes = max_passes
        self.locked = locked
        self.density = density
        # Gather the live vertices to the front of every per-vertex array once
        # enough of them have died. What it buys is a pass that does not cost
        # the original vertex count forever after most of it has died.
        self.compact_vertices = compact_vertices
        # Per-vertex UVs, (V, 2), from core.seams.read_seams_and_uvs. When
        # given, a collapse that would fold a triangle over in UV space is
        # refused, the same way one that folds it over in 3D is. Without it
        # the UVs are only checked by the 3D test, which they drift away from:
        # a vertex keeps its UV while its position moves with every collapse
        # it survives, and 5% of a 10% head scan came out folded.
        self.uvs = uvs
