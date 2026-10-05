"""The simplify contract: what the collapse returns.

A NamedTuple so the callers that build results and the callers that consume
them share exactly one definition of the field order.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np


class SimplifyResult(NamedTuple):
    """What the backend returns.

    positions    (V_out, 3) float
    triangles    (F_out, 3) int32
    remap        (V_orig,) int32. Every original vertex index mapped to the
                 output index of the vertex it was merged into. Never negative:
                 a removed vertex still has a representative, which is what
                 attribute transfer needs. Do not read `remap >= 0` as "kept".
    survived     (V_orig,) bool. True where the original vertex is still present
                 in its own right. This is the "was it kept" question.
    face_origin  (F_out,) int32. The original triangle each output triangle came
                 from. An edge collapse never creates a face, it only remaps
                 corners or deletes the face, so this provenance is exact. It
                 is what makes corner-domain attribute transfer (UVs) possible.
    uvs          (V_out, 2) float32 or None. Each output vertex's UV in the
                 tracked layer, moved with the vertex as it was placed. Exact
                 for every vertex off a seam; a seam vertex's entry is one of
                 its wedges and the transfer does not use it. None when the
                 run did not track UVs.
    """

    positions: np.ndarray
    triangles: np.ndarray
    remap: np.ndarray
    survived: np.ndarray
    face_origin: np.ndarray
    uvs: np.ndarray | None = None

    @classmethod
    def build(cls, positions, triangles, remap, survived, face_origin,
              dtype=np.float64, uvs=None):
        """Apply the dtypes the contract above specifies, in one place.

        `asarray` rather than `astype`, so an array that already has the right
        dtype is passed through instead of copied -- the positions of a
        10M-vertex result are 240 MB.
        """
        return cls(
            positions=np.asarray(positions, dtype=dtype),
            triangles=np.asarray(triangles, dtype=np.int32),
            remap=np.asarray(remap, dtype=np.int32),
            survived=np.asarray(survived, dtype=bool),
            face_origin=np.asarray(face_origin, dtype=np.int32),
            uvs=None if uvs is None else np.asarray(uvs, dtype=np.float32),
        )
