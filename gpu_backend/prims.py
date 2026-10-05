"""Parallel primitives: exclusive prefix sum and stream compaction.

Kernels 2 and 4 of PLAN.md 4.4 are both a scan, so it is worth getting right
once. Two constraints shape the implementation:

  * No atomics are used. Every thread writes only to slots it owns, so the
    result is deterministic and the code runs on a device where integer image
    atomics are missing.
  * No shared memory and no subgroup operations, so nothing depends on features
    Blender's GLSL translation layer may or may not forward to Metal.

The structure is the standard three-phase block scan. Each thread serially
reduces a block of BLOCK elements, the block sums are scanned recursively, then
each thread serially writes its block's running total offset by its block's
prefix. Serial work inside a block is fine: it trades a little parallelism for
complete portability, and the whole thing is memory-bound regardless.

Depth is log(n) / log(BLOCK), so 25M elements need three levels.
"""

from __future__ import annotations

import numpy as np

from . import context as ctx

BLOCK = 256

_SRC_BLOCK_SUMS = """
void main() {
  uint b = GID;
  if (b >= uint(nblocks)) return;
  uint start = b * uint(BLOCK);
  uint end = min(start + uint(BLOCK), uint(n));
  uint acc = 0u;
  for (uint i = start; i < end; i++) acc += imageLoad(src, IDX2(i)).r;
  imageStore(sums, IDX2(b), uvec4(acc));
}
"""

# One thread, used only once the recursion has shrunk the array to a single
# block. At BLOCK = 256 that is at most 256 serial adds, which is nothing.
_SRC_SERIAL = """
void main() {
  if (GID != 0u) return;
  uint acc = 0u;
  for (uint i = 0u; i < uint(n); i++) {
    uint v = imageLoad(src, IDX2(i)).r;
    imageStore(dst, IDX2(i), uvec4(acc));
    acc += v;
  }
}
"""

_SRC_WRITE = """
void main() {
  uint b = GID;
  if (b >= uint(nblocks)) return;
  uint start = b * uint(BLOCK);
  uint end = min(start + uint(BLOCK), uint(n));
  uint acc = imageLoad(offsets, IDX2(b)).r;
  for (uint i = start; i < end; i++) {
    uint v = imageLoad(src, IDX2(i)).r;
    imageStore(dst, IDX2(i), uvec4(acc));
    acc += v;
  }
}
"""

# total = exclusive_prefix[n-1] + src[n-1], computed on the GPU so the host only
# reads back a single-element array instead of two whole textures.
_SRC_TOTAL = """
void main() {
  if (GID != 0u) return;
  uint last = uint(n) - 1u;
  uint t = imageLoad(dst, IDX2(last)).r + imageLoad(src, IDX2(last)).r;
  imageStore(sums, IDX2(0u), uvec4(t));
}
"""

_SRC_COMPACT = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  if (imageLoad(flags, IDX2(i)).r == 0u) return;
  uint slot = imageLoad(offsets, IDX2(i)).r;
  imageStore(dst, IDX2(slot), uvec4(imageLoad(src, IDX2(i)).r));
}
"""

# Compacting several parallel arrays at once. Triangles are held as three
# separate uint arrays rather than one RGBA array, because integer upload is only
# implemented for single-component arrays (see context.Array.upload).
_SRC_COMPACT3 = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  if (imageLoad(flags, IDX2(i)).r == 0u) return;
  uint slot = imageLoad(offsets, IDX2(i)).r;
  imageStore(dst_a, IDX2(slot), uvec4(imageLoad(src_a, IDX2(i)).r));
  imageStore(dst_b, IDX2(slot), uvec4(imageLoad(src_b, IDX2(i)).r));
  imageStore(dst_c, IDX2(slot), uvec4(imageLoad(src_c, IDX2(i)).r));
}
"""


class Prims:
    """Compiled primitives plus a pool of scratch arrays.

    Shaders are compiled once per session. Scratch arrays are reused across
    calls, keyed by the element count they need to hold, because reallocating a
    texture per scan would cost more than the scan.
    """

    def __init__(self):
        u = "R32UI"
        defines = {"BLOCK": BLOCK}
        ints = (("INT", "n"), ("INT", "nblocks"))

        self.block_sums = ctx.Kernel(
            "scan_block_sums", _SRC_BLOCK_SUMS,
            [("src", u), ("sums", u)], defines=defines, push_constants=ints,
        )
        self.serial = ctx.Kernel(
            "scan_serial", _SRC_SERIAL,
            [("src", u), ("dst", u)], defines=defines,
            push_constants=(("INT", "n"),), local_size=1,
        )
        self.write = ctx.Kernel(
            "scan_write", _SRC_WRITE,
            [("src", u), ("dst", u), ("offsets", u)],
            defines=defines, push_constants=ints,
        )
        self.total = ctx.Kernel(
            "scan_total", _SRC_TOTAL,
            [("src", u), ("dst", u), ("sums", u)],
            defines=defines, push_constants=(("INT", "n"),), local_size=1,
        )
        self.compact = ctx.Kernel(
            "compact", _SRC_COMPACT,
            [("src", u), ("dst", u), ("flags", u), ("offsets", u)],
            defines=defines, push_constants=(("INT", "n"),),
        )
        self.compact3 = ctx.Kernel(
            "compact3", _SRC_COMPACT3,
            [("src_a", u), ("src_b", u), ("src_c", u),
             ("dst_a", u), ("dst_b", u), ("dst_c", u),
             ("flags", u), ("offsets", u)],
            defines=defines, push_constants=(("INT", "n"),),
        )
        self._scratch = {}
        self._one = ctx.Array("prims_one", 1, fmt="R32UI")

    def scratch(self, key, count):
        """A reusable R32UI array of at least `count` elements."""
        have = self._scratch.get(key)
        if have is None or have.count < count:
            have = ctx.Array(f"prims_{key}", max(count, BLOCK), fmt="R32UI")
            self._scratch[key] = have
        return have

    # -- scan ---------------------------------------------------------------

    def scan_exclusive(self, src, dst, count, depth=0):
        """dst[i] = sum(src[0:i]). src and dst must be distinct R32UI arrays."""
        if count <= 0:
            return
        if count <= BLOCK:
            self.serial.run(1, bind={"src": src, "dst": dst}, n=count)
            return

        nblocks = (count + BLOCK - 1) // BLOCK
        sums = self.scratch(f"sums{depth}", nblocks)
        offs = self.scratch(f"offs{depth}", nblocks)

        self.block_sums.run(
            nblocks, bind={"src": src, "sums": sums}, n=count, nblocks=nblocks
        )
        self.scan_exclusive(sums, offs, nblocks, depth + 1)
        self.write.run(
            nblocks, bind={"src": src, "dst": dst, "offsets": offs},
            n=count, nblocks=nblocks,
        )

    def scan_total(self, src, dst, count):
        """Total of src, given dst already holds its exclusive prefix sum."""
        if count <= 0:
            return 0
        self.total.run(
            1, bind={"src": src, "dst": dst, "sums": self._one}, n=count
        )
        return int(self._one.download(1)[0])

    # -- compaction ---------------------------------------------------------

    # Compaction must never write into the array it is reading.
    #
    # A surviving element at index i moves to slot j <= i. With every thread
    # running at once, a thread writing slot j can clobber index j before the
    # thread that owns index j has read it. In-place compaction therefore
    # corrupts data, non-deterministically and only when the timing lines up,
    # which is the worst way to find out. Callers pass distinct destinations and
    # swap, and these methods refuse the aliased case outright.

    def scan_flags(self, flags, count):
        """Exclusive prefix sum of `flags`, plus the total. Returns (offs, kept)."""
        offs = self.scratch("cmp_offs", count)
        self.scan_exclusive(flags, offs, count)
        return offs, self.scan_total(flags, offs, count)

    def apply_compact_one(self, src, dst, flags, offs, count):
        if src is dst:
            raise ValueError("compaction cannot be done in place")
        self.compact.run(
            count, bind={"src": src, "dst": dst, "flags": flags,
                         "offsets": offs}, n=count,
        )

    def apply_compact_three(self, src, dst, flags, offs, count):
        if any(a is b for a, b in zip(src, dst)):
            raise ValueError("compaction cannot be done in place")
        self.compact3.run(
            count,
            bind={
                "src_a": src[0], "src_b": src[1], "src_c": src[2],
                "dst_a": dst[0], "dst_b": dst[1], "dst_c": dst[2],
                "flags": flags, "offsets": offs,
            },
            n=count,
        )

    def compact_one(self, src, dst, flags, count):
        """Copy the flagged elements of src into the front of dst, keeping order."""
        offs, kept = self.scan_flags(flags, count)
        self.apply_compact_one(src, dst, flags, offs, count)
        return kept

    def compact_three(self, src, dst, flags, count):
        """compact_one over three parallel arrays, e.g. triangle corners."""
        offs, kept = self.scan_flags(flags, count)
        self.apply_compact_three(src, dst, flags, offs, count)
        return kept


def exclusive_scan_reference(values):
    """NumPy equivalent, for tests."""
    out = np.zeros(values.size, dtype=np.uint64)
    np.cumsum(values[:-1].astype(np.uint64), out=out[1:])
    return out.astype(np.uint32)
