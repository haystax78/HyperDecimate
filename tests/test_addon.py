"""Register the addon and drive the operator. PLAN.md milestone M5.

    blender --background --factory-startup --python tests/test_addon.py

Everything below the operator is covered elsewhere. What this checks is the part a
user actually touches: that the addon registers, that the operator polls and runs,
that settings reach the backend, that the result lands in the scene, and that UVs
and attributes survive the round trip.

Background mode has no event loop, so the operator's blocking `execute` path runs
here. The modal path is exercised by `tools/gui_smoke.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import hyper_decimate  # noqa: E402
from hyper_decimate.core import egress, ingest, seams  # noqa: E402
from hyper_decimate.tests.fixtures import grid, sphere  # noqa: E402

from hyper_decimate.tests.harness import check, finish, report  # noqa: E402


def clear_scene():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)
    for mesh in list(bpy.data.meshes):
        if mesh.users == 0:
            bpy.data.meshes.remove(mesh)


def add_object(name, positions, triangles, with_uvs=False):
    mesh = egress.build_mesh(name, positions, np.ascontiguousarray(
        triangles, dtype=np.int32))
    if with_uvs:
        layer = mesh.uv_layers.new(name="UVMap", do_init=False)
        corner_vert = ingest.read_corner_verts(mesh)
        pos = ingest.read_positions(mesh)
        # Split the UVs by face so there is a genuine seam to preserve.
        tris = np.asarray(triangles)
        side = pos[tris, 0].mean(axis=1) > 0.0
        uv = pos[corner_vert, :2].astype(np.float32).copy()
        uv[np.repeat(side, 3), 0] += 10.0
        layer.uv.foreach_set("vector", uv.reshape(-1))
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def test_metadata():
    """Both discovery paths must be intact, and their versions must agree.

    This exists because the add-on shipped without `bl_info` and was therefore
    invisible to script directories: Blender scans those for `bl_info` and never
    reads `blender_manifest.toml`, so the folder was silently skipped and nothing
    appeared in the Add-ons list. A script directory containing it found 17 other
    legacy add-ons and not this one, with no error anywhere.
    """
    print("")
    print("metadata", flush=True)
    info = getattr(hyper_decimate, "bl_info", None)
    check("bl_info exists, so script directories can find it", info is not None)
    if info is None:
        return
    for key in ("name", "version", "blender", "category"):
        check(f"bl_info has {key!r}", key in info)

    manifest = ROOT / "blender_manifest.toml"
    check("blender_manifest.toml exists, so it installs as an extension",
          manifest.exists())
    if not manifest.exists():
        return

    import tomllib
    with open(manifest, "rb") as fh:
        data = tomllib.load(fh)

    bl_version = ".".join(str(n) for n in info["version"])
    check("bl_info and manifest versions agree",
          bl_version == data["version"],
          f"bl_info {bl_version} vs manifest {data['version']}")
    check("module __version__ agrees too",
          hyper_decimate.__version__ == data["version"],
          f"{hyper_decimate.__version__} vs {data['version']}")
    check("manifest tagline is within Blender's 64-character limit",
          len(data["tagline"]) <= 64, f"{len(data['tagline'])} characters")
    bl_min = ".".join(str(n) for n in info["blender"])
    check("minimum Blender version agrees",
          bl_min == data["blender_version_min"],
          f"bl_info {bl_min} vs manifest {data['blender_version_min']}")


def test_registration():
    print("\nregistration", flush=True)
    hyper_decimate.register()
    check("operator is registered", hasattr(bpy.types, "HYPERDEC_OT_decimate"))
    check("panel is registered", hasattr(bpy.types, "HYPERDEC_PT_panel"))
    check("settings are attached to the scene",
          hasattr(bpy.context.scene, "hyper_decimate"))

    hyper_decimate.unregister()
    check("unregister removes the operator",
          not hasattr(bpy.types, "HYPERDEC_OT_decimate"))
    # Re-register for the rest of the run; a register/unregister/register cycle is
    # exactly what Blender does when a user toggles the addon off and on.
    hyper_decimate.register()
    check("re-registration works", hasattr(bpy.types, "HYPERDEC_OT_decimate"))


def test_poll():
    print("\npoll", flush=True)
    clear_scene()
    check("refuses with no object",
          not bpy.ops.hyperdec.decimate.poll())
    pos, tris = sphere(3)
    add_object("poll_subject", pos, tris)
    check("accepts a mesh object", bool(bpy.ops.hyperdec.decimate.poll()))


def test_ratio_run():
    print("\nratio mode, new object", flush=True)
    clear_scene()
    pos, tris = sphere(4)
    obj = add_object("subject", pos, tris)
    settings = bpy.context.scene.hyper_decimate
    settings.mode = 'RATIO'
    settings.ratio = 0.10
    settings.output = 'NEW'

    before = len(bpy.data.objects)
    result = bpy.ops.hyperdec.decimate()
    check("operator finished", result == {'FINISHED'}, str(result))
    check("a new object was added", len(bpy.data.objects) == before + 1,
          f"{len(bpy.data.objects)} objects")
    check("the original is untouched", len(obj.data.polygons) == tris.shape[0],
          f"{len(obj.data.polygons)} faces")

    out = next(o for o in bpy.data.objects if "_decimated_" in o.name)
    target = int(tris.shape[0] * 0.10)
    check("result hits the target", len(out.data.polygons) <= target,
          f"{len(out.data.polygons)} vs target {target}")
    check("result is a valid mesh", not out.data.validate(verbose=False))
    print(f"  report: {settings.last_report}", flush=True)
    check("a report was recorded", bool(settings.last_report))
    check("result is shaded flat by default",
          not any(p.use_smooth for p in out.data.polygons))


def test_shade_smooth():
    print("\nshade smooth option", flush=True)
    clear_scene()
    pos, tris = sphere(4)
    add_object("subject", pos, tris)
    settings = bpy.context.scene.hyper_decimate
    settings.mode = 'RATIO'
    settings.ratio = 0.10
    settings.output = 'NEW'
    settings.shade_smooth = True
    try:
        bpy.ops.hyperdec.decimate()
        out = next(o for o in bpy.data.objects if "_decimated_" in o.name)
        check("result is shaded smooth when asked",
              all(p.use_smooth for p in out.data.polygons))
    finally:
        settings.shade_smooth = False


def test_absolute_and_replace():
    print("\nabsolute mode, replace in place", flush=True)
    clear_scene()
    pos, tris = sphere(4)
    obj = add_object("subject", pos, tris)
    settings = bpy.context.scene.hyper_decimate
    settings.mode = 'ABSOLUTE'
    settings.target_faces = 1000
    settings.output = 'REPLACE'

    before = len(bpy.data.objects)
    bpy.ops.hyperdec.decimate()
    check("no new object was added", len(bpy.data.objects) == before,
          f"{len(bpy.data.objects)} objects")
    check("the object's own mesh was reduced",
          len(obj.data.polygons) <= 1000, f"{len(obj.data.polygons)} faces")
    check("replaced mesh is valid", not obj.data.validate(verbose=False))


def test_uvs_and_seams():
    print("\nUVs preserved and seams constrained", flush=True)
    clear_scene()
    pos, tris = grid(41)
    add_object("uv_subject", pos, tris, with_uvs=True)
    settings = bpy.context.scene.hyper_decimate
    settings.mode = 'RATIO'
    settings.ratio = 0.20
    settings.output = 'NEW'
    settings.keep_uvs = True

    bpy.ops.hyperdec.decimate()
    out = next(o for o in bpy.data.objects if "_decimated_" in o.name)
    check("UV layer survived", len(out.data.uv_layers) == 1,
          f"{len(out.data.uv_layers)} layers")
    check("result is valid", not out.data.validate(verbose=False))
    print(f"  report: {settings.last_report}", flush=True)
    # The seam constraint has to hold through the whole operator round trip,
    # since nothing else guards it.
    check("the run reported which path it took",
          bool(settings.last_report), settings.last_report)

    # The seam must survive the round trip through the operator, not just through
    # the transfer function. The fixture puts the two sides 10 units apart in UV,
    # so welding shows up as faces whose corners sit on both sides at once. See
    # tests/fixtures.py _check_uv_transfer for the exact version.
    n_loops = len(out.data.loops)
    out_uv = np.empty(n_loops * 2, dtype=np.float32)
    out.data.uv_layers[0].uv.foreach_get("vector", out_uv)
    out_uv = out_uv.reshape(n_loops, 2)
    side = (out_uv[:, 0] > 5.0).reshape(-1, 3)
    mixed = int((side.any(axis=1) != side.all(axis=1)).sum())
    check("no output face mixes UVs from both sides of the seam", mixed == 0,
          f"{mixed} of {side.shape[0]} faces straddle the seam in UV space")
    split = seams.seam_vertices_from_uvs(
        ingest.read_corner_verts(out.data), out_uv, len(out.data.vertices))
    check("the seam is still split after the round trip",
          int(split.sum()) > 0,
          f"{int(split.sum())} vertices carry two UVs")


def _edge_stats(mesh):
    """Median edge length and the count of edges far below it."""
    pos = ingest.read_positions(mesh)
    tris = ingest.read_triangles(mesh)[0].astype(np.int64)
    a = pos[tris[:, [0, 1, 2]]]
    b = pos[tris[:, [1, 2, 0]]]
    e = np.linalg.norm(b - a, axis=2).reshape(-1)
    e = e[e > 0]
    med = float(np.median(e))
    return med, int((e < med * 0.05).sum())


def test_exact_target_and_replay():
    """The collapse lands on the target itself, and a second target replays.

    There is no second decimator: the pass loop reaches the count it is asked
    for, and the collapse log it recorded is what a later target replays
    rather than reprocessing. Two runs on the same mesh -- the second inside
    the first's logged floor -- exercise both halves: the first processes,
    the second reports itself as a replay.
    """
    print("")
    print("exact target and the session cache", flush=True)
    clear_scene()
    pos, tris = sphere(5)
    add_object("exact_subject", pos, tris)
    settings = bpy.context.scene.hyper_decimate
    settings.mode = 'ABSOLUTE'
    settings.output = 'NEW'
    settings.cache_depth = 4

    results = {}
    for target in (1600, 400):
        for obj in [o for o in bpy.data.objects
                    if "_decimated_" in o.name]:
            bpy.data.objects.remove(obj, do_unlink=True)
        settings.target_faces = target
        bpy.context.view_layer.objects.active = \
            bpy.data.objects["exact_subject"]
        bpy.data.objects["exact_subject"].select_set(True)
        bpy.ops.hyperdec.decimate()
        out = next(o for o in bpy.data.objects
                   if "_decimated_" in o.name)
        _med, tiny = _edge_stats(out.data)
        results[target] = (len(out.data.polygons), tiny,
                           settings.last_report)
        check(f"result is valid at target {target}",
              not out.data.validate(verbose=False))
        check(f"the run hit the target {target} itself",
              abs(len(out.data.polygons) - target) <= target * 0.05,
              f"{len(out.data.polygons)} faces, asked for {target}")

    _faces, _tiny, second_report = results[400]
    check("the second target replayed the first run's log",
          "replayed" in second_report, second_report)


def test_result_is_selected():
    """A new object is what the user wants selected when the tool exits.

    Without this the run ends with the *source* still active, so the next thing
    the user does lands on the mesh they just decimated rather than the result.
    """
    print("")
    print("result selection", flush=True)
    clear_scene()
    pos, tris = sphere(4)
    src = add_object("select_subject", pos, tris)
    settings = bpy.context.scene.hyper_decimate
    settings.mode = 'RATIO'
    settings.ratio = 0.2
    settings.output = 'NEW'

    bpy.ops.hyperdec.decimate()
    out = next(o for o in bpy.data.objects if "_decimated_" in o.name)
    check("the new object is active",
          bpy.context.view_layer.objects.active is out,
          f"active is {getattr(bpy.context.view_layer.objects.active, 'name', None)!r}")
    check("the new object is selected", out.select_get())
    check("the source is no longer selected", not src.select_get())

    # Replacing in place keeps the original object, which is already active.
    clear_scene()
    pos, tris = sphere(4)
    src = add_object("replace_subject", pos, tris)
    settings.output = 'REPLACE'
    bpy.ops.hyperdec.decimate()
    check("replacing in place leaves the object active",
          bpy.context.view_layer.objects.active is src,
          f"active is {getattr(bpy.context.view_layer.objects.active, 'name', None)!r}")
    settings.output = 'NEW'


def test_no_op_target():
    print("\ntarget already met", flush=True)
    clear_scene()
    pos, tris = sphere(2)
    add_object("small", pos, tris)
    settings = bpy.context.scene.hyper_decimate
    settings.mode = 'ABSOLUTE'
    settings.target_faces = 10_000_000
    result = bpy.ops.hyperdec.decimate()
    check("declines gracefully rather than failing",
          result in ({'FINISHED'}, {'CANCELLED'}), str(result))
    print(f"  report: {settings.last_report}", flush=True)


def main():
    print("Hyper Decimate — addon tests", flush=True)
    test_metadata()
    test_registration()
    test_poll()
    test_ratio_run()
    test_shade_smooth()
    test_absolute_and_replace()
    test_uvs_and_seams()
    test_exact_target_and_replay()
    test_result_is_selected()
    test_no_op_target()

    hyper_decimate.unregister()

    return report()


if __name__ == "__main__":
    finish(main())
