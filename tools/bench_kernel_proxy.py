"""Measure the per-pass GPU cost with stand-in kernels, on a real mesh.

The ten kernels of PLAN.md 4.4 are not written yet, so the pass cost in 4.6 is
still an estimate. This script replaces the estimate with a measurement by
running kernels that have the same shape, the same memory access pattern and the
same array sizes as the real ones, over the real head scan.

What each proxy stands in for:

  remap_corners  kernel 1. Random gather through a remap array, per corner.
  adj_count      kernel 3. imageAtomicAdd per corner into a per-vertex counter.
  adj_scatter    kernel 5. Atomic cursor bump plus a write, per corner.
  ring_walk      kernel 6, the heaviest. Per vertex, walk the one-ring through
                 the CSR, read each incident face's corners, load an 11-float
                 quadric for the candidate target from a layered image, and do
                 the quadric arithmetic.
  claim          kernels 8 and 9. Per vertex, walk the ring, imageAtomicMin.

Not proxied: the prefix sums and compaction, which prims.py already measures.

    blender --background --factory-startup --gpu-backend opengl \
        tests/head_scan_test.blend --python tools/bench_kernel_proxy.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hyper_decimate.core import ingest  # noqa: E402
from hyper_decimate.gpu_backend import context as ctx  # noqa: E402
from hyper_decimate.gpu_backend import prims as pr  # noqa: E402

U = "R32UI"
F4 = "RGBA32F"

K_REMAP = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  uint a = imageLoad(c0, IDX2(i)).r;
  uint b = imageLoad(c1, IDX2(i)).r;
  uint c = imageLoad(c2, IDX2(i)).r;
  uint ra = imageLoad(remap, IDX2(a)).r;
  uint rb = imageLoad(remap, IDX2(b)).r;
  uint rc = imageLoad(remap, IDX2(c)).r;
  imageStore(c0, IDX2(i), uvec4(ra));
  imageStore(c1, IDX2(i), uvec4(rb));
  imageStore(c2, IDX2(i), uvec4(rc));
  uint dead = (ra == rb || rb == rc || rc == ra) ? 0u : 1u;
  imageStore(alive, IDX2(i), uvec4(dead));
}
"""

K_ADJ_COUNT = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  imageAtomicAdd(counts, IDX2(imageLoad(c0, IDX2(i)).r), 1u);
  imageAtomicAdd(counts, IDX2(imageLoad(c1, IDX2(i)).r), 1u);
  imageAtomicAdd(counts, IDX2(imageLoad(c2, IDX2(i)).r), 1u);
}
"""

K_ADJ_SCATTER = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  uint v0 = imageLoad(c0, IDX2(i)).r;
  uint v1 = imageLoad(c1, IDX2(i)).r;
  uint v2 = imageLoad(c2, IDX2(i)).r;
  uint s0 = imageLoad(offs, IDX2(v0)).r + imageAtomicAdd(cursor, IDX2(v0), 1u);
  uint s1 = imageLoad(offs, IDX2(v1)).r + imageAtomicAdd(cursor, IDX2(v1), 1u);
  uint s2 = imageLoad(offs, IDX2(v2)).r + imageAtomicAdd(cursor, IDX2(v2), 1u);
  imageStore(adj, IDX2(s0), uvec4(i * 3u + 0u));
  imageStore(adj, IDX2(s1), uvec4(i * 3u + 1u));
  imageStore(adj, IDX2(s2), uvec4(i * 3u + 2u));
}
"""

# The expensive one. Reads the ring, the incident faces, and a full quadric per
# candidate target, then evaluates the quadric error the way 4.3 specifies.
K_RING_WALK = """
float qerr(uint v, vec3 p) {
  vec4 a = imageLoad(quad, IDX3(v, 0));
  vec4 b = imageLoad(quad, IDX3(v, 1));
  vec4 c = imageLoad(quad, IDX3(v, 2));
  float e = a.x * p.x * p.x + 2.0 * a.y * p.x * p.y + 2.0 * a.z * p.x * p.z
          + 2.0 * a.w * p.x + b.x * p.y * p.y + 2.0 * b.y * p.y * p.z
          + 2.0 * b.z * p.y + b.w * p.z * p.z + 2.0 * c.x * p.z + c.y;
  return max(e, 0.0);
}

void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  uint lo = imageLoad(offs, IDX2(v)).r;
  uint hi = imageLoad(offs, IDX2(v + 1u)).r;
  float best = 1e30;
  uint tgt = 0xffffffffu;
  for (uint k = lo; k < hi; k++) {
    uint corner = imageLoad(adj, IDX2(k)).r;
    uint f = corner / 3u;
    uint a = imageLoad(c0, IDX2(f)).r;
    uint b = imageLoad(c1, IDX2(f)).r;
    uint c = imageLoad(c2, IDX2(f)).r;
    uint o1 = (a == v) ? b : ((b == v) ? c : a);
    uint o2 = (a == v) ? c : ((b == v) ? a : b);
    vec3 p1 = imageLoad(pos, IDX2(o1)).xyz;
    vec3 p2 = imageLoad(pos, IDX2(o2)).xyz;
    float e1 = qerr(o1, p1);
    float e2 = qerr(o2, p2);
    if (e1 < best) { best = e1; tgt = o1; }
    if (e2 < best) { best = e2; tgt = o2; }
  }
  imageStore(target, IDX2(v), uvec4(tgt));
}
"""

K_CLAIM = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  uint prio = hd_hash(v);
  uint lo = imageLoad(offs, IDX2(v)).r;
  uint hi = imageLoad(offs, IDX2(v + 1u)).r;
  for (uint k = lo; k < hi; k++) {
    uint f = imageLoad(adj, IDX2(k)).r / 3u;
    imageAtomicMin(claim, IDX2(imageLoad(c0, IDX2(f)).r), prio);
    imageAtomicMin(claim, IDX2(imageLoad(c1, IDX2(f)).r), prio);
    imageAtomicMin(claim, IDX2(imageLoad(c2, IDX2(f)).r), prio);
  }
}
"""

# The same independent set with no atomics at all.
#
# "v wins" means v has the lowest priority among every candidate whose ring
# touches v's ring. Written as a scatter that is a claim, and it needs atomics.
# But ring membership is symmetric: u is in ring(w) exactly when w is in ring(u).
# So the minimum priority claiming slot u equals the minimum priority over
# ring(u), which each thread can gather for itself. Two gather passes replace the
# atomic claim and the verify, and nothing contends.
#
#   pass A:  ring_min[u] = min over w in ring(u) of prio(w)
#   pass B:  v wins iff prio(v) == min over u in ring(v) of ring_min[u]
K_RING_MIN = """
void main() {
  uint u = GID;
  if (u >= uint(n)) return;
  uint best = hd_hash(u);
  uint lo = imageLoad(offs, IDX2(u)).r;
  uint hi = imageLoad(offs, IDX2(u + 1u)).r;
  for (uint k = lo; k < hi; k++) {
    uint f = imageLoad(adj, IDX2(k)).r / 3u;
    best = min(best, hd_hash(imageLoad(c0, IDX2(f)).r));
    best = min(best, hd_hash(imageLoad(c1, IDX2(f)).r));
    best = min(best, hd_hash(imageLoad(c2, IDX2(f)).r));
  }
  imageStore(ring_min, IDX2(u), uvec4(best));
}
"""

K_WINNER = """
void main() {
  uint v = GID;
  if (v >= uint(n)) return;
  uint prio = hd_hash(v);
  uint lo = imageLoad(offs, IDX2(v)).r;
  uint hi = imageLoad(offs, IDX2(v + 1u)).r;
  uint worst = prio;
  for (uint k = lo; k < hi; k++) {
    uint f = imageLoad(adj, IDX2(k)).r / 3u;
    worst = min(worst, imageLoad(ring_min, IDX2(imageLoad(c0, IDX2(f)).r)).r);
    worst = min(worst, imageLoad(ring_min, IDX2(imageLoad(c1, IDX2(f)).r)).r);
    worst = min(worst, imageLoad(ring_min, IDX2(imageLoad(c2, IDX2(f)).r)).r);
  }
  imageStore(claim, IDX2(v), uvec4(worst == prio ? 1u : 0u));
}
"""

RESULTS = []


def bench(label, kernel, threads, bind, sync_array, reps=3, **consts):
    """Time a kernel, forcing a real sync through an array it actually wrote."""
    kernel.run(threads, bind=bind, **consts)   # warm
    sync_array.download(1)
    best = None
    for _ in range(reps):
        t0 = time.perf_counter()
        kernel.run(threads, bind=bind, **consts)
        sync_array.download(1)
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    print(f"  {label:<36} {best * 1000:8.2f} ms   ({threads:,} threads)",
          flush=True)
    RESULTS.append((label, best))
    return best


def main():
    import bpy

    obj = max((o for o in bpy.data.objects if o.type == "MESH"),
              key=lambda o: len(o.data.vertices))
    positions, tris, _, method = ingest.read_mesh(obj.data)
    V = positions.shape[0]
    F = tris.shape[0]
    C = 3 * F
    info = ctx.probe()
    print(f"\n{info['backend']} / {info['renderer']}", flush=True)
    print(f"subject: {V:,} verts, {F:,} triangles, {C:,} corners "
          f"({method})\n", flush=True)

    padded = np.zeros((V, 4), dtype=np.float32)
    padded[:, :3] = positions
    pos = ctx.Array("kp_pos", V, fmt=F4, data=padded)
    c0 = ctx.Array("kp_c0", F, fmt=U, data=tris[:, 0].astype(np.uint32))
    c1 = ctx.Array("kp_c1", F, fmt=U, data=tris[:, 1].astype(np.uint32))
    c2 = ctx.Array("kp_c2", F, fmt=U, data=tris[:, 2].astype(np.uint32))
    remap = ctx.Array("kp_remap", V, fmt=U,
                      data=np.arange(V, dtype=np.uint32))
    alive = ctx.Array("kp_alive", F, fmt=U)
    counts = ctx.Array("kp_counts", V + 1, fmt=U)
    offs = ctx.Array("kp_offs", V + 1, fmt=U)
    cursor = ctx.Array("kp_cursor", V, fmt=U)
    adj = ctx.Array("kp_adj", C, fmt=U)
    quad = ctx.Array("kp_quad", V, fmt=F4, layers=3)
    target = ctx.Array("kp_target", V, fmt=U)
    claim = ctx.Array("kp_claim", V, fmt=U)
    for a in (alive, counts, offs, cursor, adj, target):
        a.clear(0)
    claim.clear(0xFFFFFFFF)
    quad.clear(0.0)

    p = pr.Prims()
    ints = (("INT", "n"),)

    k_remap = ctx.Kernel("remap_corners", K_REMAP, [
        ("c0", U), ("c1", U), ("c2", U), ("remap", U), ("alive", U)],
        push_constants=ints)
    k_count = ctx.Kernel("adj_count", K_ADJ_COUNT, [
        ("c0", U), ("c1", U), ("c2", U), ("counts", U)], push_constants=ints)
    k_scatter = ctx.Kernel("adj_scatter", K_ADJ_SCATTER, [
        ("c0", U), ("c1", U), ("c2", U), ("offs", U), ("cursor", U),
        ("adj", U)], push_constants=ints)
    k_ring = ctx.Kernel("ring_walk", K_RING_WALK, [
        ("c0", U), ("c1", U), ("c2", U), ("offs", U), ("adj", U),
        ("pos", F4), ("quad", F4, 3), ("target", U)], push_constants=ints)
    k_claim = ctx.Kernel("claim", K_CLAIM, [
        ("c0", U), ("c1", U), ("c2", U), ("offs", U), ("adj", U),
        ("claim", U)], push_constants=ints)

    print("--- per-pass kernels, at full mesh size ---", flush=True)
    bench("kernel 1 remap corners", k_remap, F,
          {"c0": c0, "c1": c1, "c2": c2, "remap": remap, "alive": alive},
          alive, n=F)
    bench("kernel 3 adjacency count", k_count, F,
          {"c0": c0, "c1": c1, "c2": c2, "counts": counts}, counts, n=F)

    t0 = time.perf_counter()
    p.scan_exclusive(counts, offs, V + 1)
    offs.download(1)
    t_scan = time.perf_counter() - t0
    print(f"  {'kernel 4 scan offsets':<36} {t_scan * 1000:8.2f} ms   "
          f"({V + 1:,} threads)", flush=True)
    RESULTS.append(("kernel 4 scan offsets", t_scan))

    bench("kernel 5 adjacency scatter", k_scatter, F,
          {"c0": c0, "c1": c1, "c2": c2, "offs": offs, "cursor": cursor,
           "adj": adj}, adj, n=F)
    bench("kernel 6 ring walk + quadrics", k_ring, V,
          {"c0": c0, "c1": c1, "c2": c2, "offs": offs, "adj": adj,
           "pos": pos, "quad": quad, "target": target}, target, n=V)
    t_claim = bench("(atomic claim, for comparison only)", k_claim, V,
                    {"c0": c0, "c1": c1, "c2": c2, "offs": offs, "adj": adj,
                     "claim": claim}, claim, n=V)
    RESULTS.pop()  # measured for comparison, not part of the chosen design

    t0 = time.perf_counter()
    p.compact_three([c0, c1, c2], [c0, c1, c2], alive, F)
    t_compact = time.perf_counter() - t0
    print(f"  {'kernel 2 compact triangles':<36} {t_compact * 1000:8.2f} ms   "
          f"({F:,} threads)", flush=True)
    RESULTS.append(("kernel 2 compact triangles", t_compact))

    print("\n--- atomic-free replacement for kernels 8/9 ---", flush=True)
    ring_min = ctx.Array("kp_ringmin", V, fmt=U)
    ring_min.clear(0)
    k_ringmin = ctx.Kernel("ring_min", K_RING_MIN, [
        ("c0", U), ("c1", U), ("c2", U), ("offs", U), ("adj", U),
        ("ring_min", U)], push_constants=ints)
    k_winner = ctx.Kernel("winner", K_WINNER, [
        ("c0", U), ("c1", U), ("c2", U), ("offs", U), ("adj", U),
        ("ring_min", U), ("claim", U)], push_constants=ints)
    t_ringmin = bench("gather pass A, ring minimum", k_ringmin, V,
                      {"c0": c0, "c1": c1, "c2": c2, "offs": offs, "adj": adj,
                       "ring_min": ring_min}, ring_min, n=V)
    t_winner = bench("gather pass B, winners", k_winner, V,
                     {"c0": c0, "c1": c1, "c2": c2, "offs": offs, "adj": adj,
                      "ring_min": ring_min, "claim": claim}, claim, n=V)
    t_gather_round = t_ringmin + t_winner
    print(f"    one round: {t_gather_round * 1000:.2f} ms gathering against "
          f"{2 * t_claim * 1000:.2f} ms with atomics, "
          f"{2 * t_claim / t_gather_round:.1f}x faster", flush=True)

    ROUNDS = 4
    RESULTS.append((f"kernels 8/9, {ROUNDS} gather rounds",
                    ROUNDS * t_gather_round))
    print(f"  {'kernels 8/9, ' + str(ROUNDS) + ' gather rounds':<36} "
          f"{ROUNDS * t_gather_round * 1000:8.2f} ms", flush=True)

    per_pass = sum(dt for _, dt in RESULTS)
    print(f"\n  one full-size pass, total          {per_pass * 1000:8.2f} ms",
          flush=True)
    print(f"  (the atomic claim measured above is excluded; the gather "
          f"replaces it)", flush=True)

    # Geometric series: the mesh shrinks ~9% per pass (PLAN.md 4.6), so the sum
    # over all passes is one full-size pass times (1 - ratio) / reduction.
    factor = (1.0 - 0.10) / 0.09
    print(f"  26 passes, shrinking, factor {factor:.1f}x   "
          f"{per_pass * factor * 1000:8.2f} ms", flush=True)
    print(f"\n  => predicted GPU kernel time: "
          f"{per_pass * factor:.2f} s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
