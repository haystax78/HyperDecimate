"""The session cache: preprocess once, decimate to any target afterwards.

The pipeline's headline feature. The first decimation of a mesh records a
collapse log down to a floor below the target (see `core.replay`); every later
target between that floor and the base the log started from is a prefix of
the same log, applied in milliseconds instead of passes.

**Scope: this session's memory.** The cache lives in a module global for the
Blender session, keyed by the mesh's content and every option that affects
the collapse sequence. A restart re-preprocesses, which is the honest trade
for now: the mesh is in memory anyway, the log is a fraction of its size,
and a disk format is a decision about persistence and versioning that does
not belong inside an algorithm change. The module is shaped so a persistent
store can slot in behind the same two calls.

**What makes a cacheable run.** The log is only meaningful against the exact
mesh it was recorded from, so the key covers:

* the positions and the triangles, by content hash;
* the per-vertex inputs -- locks, density, seams -- when present, hashed
  the same way, because they steer the collapse;
* every `Options` field that changes the sequence.

The target itself is deliberately *not* in the key: a different target on
the same mesh and settings is the whole point.

**The floor and the base.** A cached entry covers targets in
`[floor_faces, base_faces]`. Below the floor the log ran out -- a coarser
request re-preprocesses with a deeper floor. Above the base, the mesh the
log starts from has fewer triangles than the user wants -- a finer request
re-preprocesses too.
"""

from __future__ import annotations

import hashlib

import numpy as np

# How many meshes to keep. Each entry holds a log (32 bytes per collapse,
# which is at most one record per face of the input) -- proportional to the
# mesh, but not free on a 100M-vertex scan.
MAX_ENTRIES = 4

_CACHE: dict = {}
_ORDER: list = []


def _hash_array(h, array):
    """Fold an array's shape and content into `h` without copying it."""
    array = np.asarray(array)
    h.update(str(array.dtype.str).encode())
    h.update(str(array.shape).encode())
    if array.dtype == np.bool_:
        # Hashing a view as bytes works for any dtype; bools are no exception,
        # but the packbits round trip is a quarter of the traffic for meshes
        # where the mask is the biggest array being keyed.
        packed = np.packbits(array.reshape(-1))
        h.update(packed.data)
    else:
        h.update(np.ascontiguousarray(array).data)


def mesh_key(positions, triangles, masks=()):
    """A content hash of the mesh and any per-vertex masks that steer it.

    `masks` is a sequence of (label, array-or-None) pairs. None contributes
    nothing but its label, so "no density" and "density of all ones" stay
    different keys -- they produce the same sequence, but proving that from
    here would mean the caller re-deriving the array it decided not to build.
    """
    h = hashlib.blake2b(digest_size=16)
    _hash_array(h, positions)
    _hash_array(h, triangles)
    for label, mask in masks:
        h.update(label.encode())
        if mask is not None:
            _hash_array(h, mask)
    return h.hexdigest()


def options_key(opts):
    """The options fingerprint: everything that changes the collapse sequence.

    Every field of `Options` that steers which collapses are chosen or where
    the survivors land is in here, including the threshold fraction -- a
    different `admit_fraction` admits a different set each pass, so the log
    it records is a different one. The per-vertex arrays (seams, locks,
    density) are not `Options` scalars; the caller keys them into `mesh_key`
    as masks instead.
    """
    return repr((
        opts.freeze_borders,
        opts.boundary_weight,
        opts.max_normal_flip_deg,
        opts.max_valence,
        opts.check_link_condition,
        opts.admit_fraction,
        opts.claim_rounds,
        opts.max_error,
        opts.max_passes,
        opts.optimal_placement,
        opts.compact_vertices,
    ))


def store(key, log_info):
    """Keep a run's replayable state under `key`.

    `log_info` is what a backend's `log` property returns: the records, the
    base and floor face counts, and the normalisation.
    """
    if log_info is None or log_info["records"] is None:
        return
    if key in _CACHE:
        _ORDER.remove(key)
    _CACHE[key] = log_info
    _ORDER.append(key)
    while len(_ORDER) > MAX_ENTRIES:
        del _CACHE[_ORDER.pop(0)]


def lookup(key, target):
    """The cached log when `target` is inside its replayable range.

    Returns (log_info, reason). `log_info` is None when this target must be
    re-preprocessed, with `reason` saying why for the report: nothing cached,
    target below the cached floor, or target above the cached base.
    """
    log = _CACHE.get(key)
    if log is None:
        return None, "not preprocessed yet"
    if target < log["floor_faces"]:
        return None, (f"target below the cached floor "
                      f"({log['floor_faces']:,} faces)")
    if target > log["base_faces"]:
        return None, (f"target above the cached base "
                      f"({log['base_faces']:,} faces)")
    return log, "replayed from the cache"


def clear():
    """Drop every entry. The addon's reload calls this so a stale log can
    never outlive the code that produced it."""
    _CACHE.clear()
    _ORDER.clear()
