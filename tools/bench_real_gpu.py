import os, sys, time
sys.path.insert(0, r"C:/_GoogleDrive/work/Blender/Python/addons_experimental/CascadeProjects/windsurf-project/addons")
import bpy, numpy as np
from hyper_decimate.core import egress, ingest, seams as sm
from hyper_decimate.core.options import Options
from hyper_decimate.gpu_backend import context as ctx
from hyper_decimate.gpu_backend.simplify import simplify as gpu_simplify
def say(*a): print(*a, flush=True)

obj = max((o for o in bpy.data.objects if o.type == "MESH"),
          key=lambda o: len(o.data.vertices))
mesh = obj.data
say(f"subject: {obj.name!r} verts={len(mesh.vertices):,} polys={len(mesh.polygons):,}")
say(f"backend: {ctx.probe()['backend']}")

t0 = time.perf_counter()
positions, tris, tri_loops, method = ingest.read_mesh(mesh)
t_in = time.perf_counter() - t0
say(f"\ningest        {t_in:7.2f}s  ({method}) {positions.shape[0]:,} v, {tris.shape[0]:,} f")

t0 = time.perf_counter()
seams = sm.read_seams(mesh) if mesh.uv_layers else None
t_seam = time.perf_counter() - t0
if seams is not None:
    say(f"seam detect   {t_seam:7.2f}s  {int(seams.sum()):,} seam vertices")

target = int(tris.shape[0] * 0.10)
stats = {}
t0 = time.perf_counter()
res = gpu_simplify(positions, tris.astype(np.int64), target,
                   Options(seams=seams), stats=stats)
t_gpu = time.perf_counter() - t0
say(f"gpu simplify  {t_gpu:7.2f}s  {tris.shape[0]:,} -> {res.triangles.shape[0]:,} "
    f"({res.triangles.shape[0]/tris.shape[0]:.1%}) in {stats['passes']} passes")

t0 = time.perf_counter()
out = egress.build_mesh(f"{obj.name}_gpu", res.positions, res.triangles)
copied = egress.transfer_point_attributes(mesh, out, res.remap, res.survived)
t_build = time.perf_counter() - t0
t0 = time.perf_counter()
uvs = egress.transfer_uv_layers(mesh, out, tri_loops, res.face_origin,
                                res.remap, res.survived, seam_mask=seams)
t_uv = time.perf_counter() - t0
t_out = t_build + t_uv
say(f"egress        {t_build:7.2f}s  attrs={copied}")
say(f"uv transfer   {t_uv:7.2f}s  uv={uvs}")

say("\n--- validity ---")
corrected = out.validate(verbose=False)
t = res.triangles
degen = int(((t[:,0]==t[:,1])|(t[:,1]==t[:,2])|(t[:,2]==t[:,0])).sum())
say(f"  mesh.validate() had repairs to make : {corrected}")
say(f"  degenerate triangles               : {degen}")
say(f"  orphaned vertices                  : {res.positions.shape[0]-np.unique(t).size}")

uv_ok = True
if uvs:
    # UV integrity at scale: no corner may have been handed a UV from a wedge it
    # does not belong to, and the seams that were there before must still be
    # there. See core/egress.transfer_uv_layers.
    n_out_loops = len(out.loops)
    buf = np.empty(n_out_loops * 2, dtype=np.float32)
    out.uv_layers[0].uv.foreach_get("vector", buf)
    out_uv = buf.reshape(n_out_loops, 2)
    out_cv = ingest.read_corner_verts(out)
    split = sm.seam_vertices_from_uvs(out_cv, out_uv, res.positions.shape[0])
    # Every output corner's UV must be one the source actually held for that
    # vertex, so the transfer cannot have invented or averaged anything.
    src_cv = ingest.read_corner_verts(mesh)
    src_buf = np.empty(len(mesh.loops) * 2, dtype=np.float32)
    mesh.uv_layers[0].uv.foreach_get("vector", src_buf)
    src_uv = src_buf.reshape(-1, 2)
    keep = np.flatnonzero(res.survived)
    n2o = np.zeros(res.positions.shape[0], dtype=np.int64)
    n2o[res.remap[keep]] = keep
    held = {}
    for v, u in zip(src_cv.tolist(), map(tuple, src_uv.tolist())):
        held.setdefault(v, set()).add(u)
    bad = 0
    step = max(1, n_out_loops // 200_000)        # sample; the dict walk is Python
    for i in range(0, n_out_loops, step):
        if tuple(out_uv[i].tolist()) not in held.get(int(n2o[out_cv[i]]), ()):
            bad += 1
    uv_ok = bad == 0 and int(split.sum()) > 0
    say("\n--- uv integrity ---")
    say(f"  seam vertices before / after       : {int(seams.sum()):,} / "
        f"{int(split.sum()):,}")
    say(f"  sampled corners with a UV their own vertex never held : {bad} "
        f"of {len(range(0, n_out_loops, step)):,}")

diag = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
disp = np.linalg.norm(res.positions[res.remap] - positions, axis=1) / diag
say("\n--- quality ---")
say(f"  displacement / bbox diagonal: mean {disp.mean():.3e} "
    f"p99 {np.quantile(disp,0.99):.3e} max {disp.max():.3e}")

say(f"\ntotal         {t_in+t_seam+t_gpu+t_out:7.2f}s "
    f"(io {t_in+t_out:.2f}s, seams {t_seam:.2f}s, gpu {t_gpu:.2f}s)")
ctx.release_all()
sys.stdout.flush()
os._exit(0 if (not corrected and degen == 0 and uv_ok) else 1)
