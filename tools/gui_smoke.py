"""Drive the modal operator in a real Blender window. PLAN.md milestone M5.

    blender --factory-startup --gpu-backend vulkan --python tools/gui_smoke.py

Background mode has no event loop, so `tests/test_addon.py` only ever exercises
the operator's blocking path. The modal path is the one users actually hit, and it
is the one with the timer, the progress bar and the Escape key, so it needs a real
window to be tested at all.

Writes its verdict to the file named by HD_SMOKE_LOG and quits Blender.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))

import hyper_decimate  # noqa: E402
from hyper_decimate.core import egress, ingest  # noqa: E402
from hyper_decimate import ops, ui  # noqa: E402
from hyper_decimate.props import PROGRESS  # noqa: E402
from hyper_decimate.tests.fixtures import sphere  # noqa: E402

LOG = open(os.environ.get("HD_SMOKE_LOG", "gui_smoke.log"), "w")
STATE = {"phase": "init", "started": 0.0, "ticks": 0, "samples": [],
         "invoke_took": 0.0}


def say(msg):
    LOG.write(msg + "\n")
    LOG.flush()
    os.fsync(LOG.fileno())


class _Tee:
    """Mirror stderr into the log.

    The panel's `draw` runs in Blender's UI thread, and an exception there is
    printed and swallowed: the panel goes blank and the operator carries on
    reporting success. Without capturing stderr this test would pass with a
    progress bar that throws on every repaint.
    """

    def __init__(self, original):
        self.original = original
        self.seen = []

    def write(self, text):
        self.seen.append(text)
        return self.original.write(text)

    def flush(self):
        self.original.flush()


STDERR = _Tee(sys.stderr)
sys.stderr = STDERR


def show_sidebar():
    """Ask for the N-panel. Its region does not exist until the next redraw."""
    found = False
    for area in bpy.context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for space in area.spaces:
            if space.type == 'VIEW_3D':
                space.show_region_ui = True
                found = True
    return found


def focus_panel():
    """Switch the sidebar to this add-on's tab, and say whether it took.

    Has to be a tick after `show_sidebar`: the UI region is 1x1 and carries no
    category until the sidebar has actually been laid out once, so setting the
    category in the same tick silently does nothing and the panel never draws.
    That is exactly how the first version of this test reported a progress bar
    that had never been painted.
    """
    for area in bpy.context.screen.areas:
        if area.type != 'VIEW_3D':
            continue
        for region in area.regions:
            if region.type != 'UI':
                continue
            region.active_panel_category = "Hyper Decimate"
            region.tag_redraw()
            return region.active_panel_category
    return None


def record_progress():
    """Sample every label the operator puts on the bar, in both phases.

    Polling from this script's own timer races the run: a sub-second decimation
    caught two samples on one machine and five on the next, so the test passed or
    failed on timing. Wrapping the operator's own updates samples exactly what the
    panel is given to draw.

    Wrapping alone was still flaky, because the operator updates once per *time
    slice* and the whole run can fit in one or two of them. Shrinking the slice
    ties the update count to the number of passes instead, which is a property of
    the mesh and the ratio rather than of how fast this machine happens to be.

    Both phases are recorded because they are separate code paths: `_advance_setup`
    names the stages of getting a mesh ready, `_update_progress` reports the passes.
    """
    ops.SLICE_SECONDS = 0.001

    setup = ops.HYPERDEC_OT_decimate._advance_setup
    update = ops.HYPERDEC_OT_decimate._update_progress
    writeback = ops.HYPERDEC_OT_decimate._advance_writeback

    def recording_setup(self, context):
        out = setup(self, context)
        if PROGRESS.active:
            STATE["samples"].append(("setup", PROGRESS.fraction, PROGRESS.label))
        return out

    def recording_update(self, context):
        update(self, context)
        STATE["samples"].append(("pass", PROGRESS.fraction, PROGRESS.label))

    def recording_writeback(self, context):
        out = writeback(self, context)
        if PROGRESS.active:
            STATE["samples"].append(("write", PROGRESS.fraction, PROGRESS.label))
        return out

    ops.HYPERDEC_OT_decimate._advance_setup = recording_setup
    ops.HYPERDEC_OT_decimate._update_progress = recording_update
    ops.HYPERDEC_OT_decimate._advance_writeback = recording_writeback


def finish(code):
    say("SMOKE DONE")
    LOG.close()
    bpy.ops.wm.quit_blender()
    return None


def tick():
    STATE["ticks"] += 1

    if STATE["phase"] == "init":
        # Read settings only after registering; the property does not exist yet.
        hyper_decimate.register()
        settings = bpy.context.scene.hyper_decimate
        for obj in list(bpy.data.objects):
            bpy.data.objects.remove(obj, do_unlink=True)
        # Big enough that the modal loop runs over many timer ticks rather
        # than finishing inside one, which is the whole point of the test. A
        # smaller subject would still exercise the loop but would leave too
        # little per stage for progress to mean anything.
        pos, tris = sphere(8)
        mesh = egress.build_mesh("smoke", pos, tris.astype(np.int32))
        # With a real UV seam, so the seam-detection stage is exercised too.
        # Without UVs that stage is correctly skipped and never gets tested.
        layer = mesh.uv_layers.new(name="UVMap", do_init=False)
        corner_vert = ingest.read_corner_verts(mesh)
        side = pos[tris, 0].mean(axis=1) > 0.0
        uv = pos[corner_vert, :2].astype(np.float32).copy()
        uv[np.repeat(side, 3), 0] += 10.0
        layer.uv.foreach_set("vector", uv.reshape(-1))
        obj = bpy.data.objects.new("smoke", mesh)
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        say(f"subject: {len(mesh.polygons):,} faces")
        say(f"sidebar requested: {show_sidebar()}")
        settings.mode = 'RATIO'
        settings.ratio = 0.05
        settings.output = 'NEW'
        STATE["phase"] = "focus"
        return 0.3

    settings = bpy.context.scene.hyper_decimate

    if STATE["phase"] == "focus":
        category = focus_panel()
        say(f"sidebar tab is now {category!r}")
        if category != "Hyper Decimate":
            say("FAIL: could not put the add-on's panel on screen, so the "
                "progress bar could not be observed")
            return finish(1)
        record_progress()
        STATE["phase"] = "invoke"
        return 0.3

    if STATE["phase"] == "invoke":
        STATE["started"] = time.perf_counter()
        result = bpy.ops.hyperdec.decimate('INVOKE_DEFAULT')
        STATE["invoke_took"] = time.perf_counter() - STATE["started"]
        say(f"invoke returned {sorted(result)} in "
            f"{STATE['invoke_took'] * 1000:.0f} ms")
        if 'RUNNING_MODAL' not in result:
            say("FAIL: operator did not go modal")
            return finish(1)
        STATE["phase"] = "wait"
        return 0.1

    if STATE["phase"] == "wait":
        if settings.last_report:
            elapsed = time.perf_counter() - STATE["started"]
            say(f"report: {settings.last_report}")
            say(f"modal run took {elapsed:.2f}s over {STATE['ticks']} ticks")
            out = [o for o in bpy.data.objects if "_decimated_" in o.name]
            if not out:
                say("FAIL: no decimated object was produced")
                return finish(1)
            mesh = out[0].data
            bad = mesh.validate(verbose=False)
            say(f"result: {len(mesh.polygons):,} faces, validate_fixed={bad}")
            if bad:
                say("FAIL: result needed repair")
                return finish(1)
            # The modal path must have yielded, or it was really a blocking run
            # wearing a progress bar.
            if STATE["ticks"] < 4:
                say(f"FAIL: only {STATE['ticks']} ticks, modal did not yield")
                return finish(1)

            # Feedback has to be immediate. Every bit of getting a mesh ready
            # used to happen inside invoke, so Blender froze for seconds with
            # nothing on screen before the bar appeared at all. A blocking invoke
            # is the regression this guards, and it is measured rather than
            # inferred from the bar, because a slow invoke is the user's
            # complaint whatever the bar does afterwards.
            took = STATE["invoke_took"]
            if took > 0.1:
                say(f"FAIL: invoke blocked for {took * 1000:.0f} ms before "
                    "handing back to the event loop, so the UI was frozen with "
                    "nothing on screen")
                return finish(1)

            samples = STATE["samples"]
            setup = [s for s in samples if s[0] == "setup"]
            passes = [s for s in samples if s[0] == "pass"]
            writes = [s for s in samples if s[0] == "write"]
            fractions = [f for _, f, _ in samples]
            say(f"{len(setup)} setup labels then {len(passes)} pass updates, "
                f"fraction {min(fractions, default=-1):.2f} to "
                f"{max(fractions, default=-1):.2f}")
            shown = None
            for _, _, label in setup:
                # The last entry repeats: it is sampled after the generator has
                # finished, when the label still reads as the final stage.
                if label != shown:
                    say(f"  setup: {label}")
                    shown = label

            # Setup has to be named, not silent, and it has to come first.
            stages = {label for _, _, label in setup}
            if len(stages) < 3:
                say(f"FAIL: only {len(stages)} setup stage(s) named; a mesh with "
                    "UVs should report reading, seam detection and upload "
                    "separately rather than sitting on one label")
                return finish(1)
            if not any("seam" in label for label in stages):
                say("FAIL: the seam-detection stage was never named, and on a "
                    "big mesh it is the slowest part of getting started")
                return finish(1)
            if samples.index(setup[0]) > samples.index(passes[0]):
                say("FAIL: passes were reported before any setup stage")
                return finish(1)
            if len(passes) < 5:
                say(f"FAIL: only {len(passes)} pass updates, the bar was "
                    "never really observed moving")
                return finish(1)
            # Writing the result back is seconds of work on a big mesh, most
            # of it the second-stage Decimate. It has to be named on the bar,
            # not hidden behind whatever the last pass happened to report.
            for _, _, label in writes:
                say(f"  writeback: {label}")
            if len(writes) < 2:
                say(f"FAIL: only {len(writes)} writeback updates, so the bar "
                    "sits still through mesh building and the second stage")
                return finish(1)
            if not any("Decimate" in label for _, _, label in writes):
                say("FAIL: the second-stage Decimate was never named on the bar")
                return finish(1)
            if samples.index(writes[0]) < samples.index(passes[-1]):
                say("FAIL: writeback was reported before the passes finished")
                return finish(1)
            if ui.PROGRESS_WIDGET_OK is not True:
                say(f"FAIL: the panel never drew a real progress bar "
                    f"(PROGRESS_WIDGET_OK={ui.PROGRESS_WIDGET_OK!r}); either the "
                    "panel never drew, or UILayout.progress fell back to text")
                return finish(1)
            if max(fractions) <= min(fractions):
                say(f"FAIL: progress never moved, stuck at {fractions[0]:.2f}")
                return finish(1)
            if any(b < a - 1e-6 for a, b in zip(fractions, fractions[1:])):
                say("FAIL: progress went backwards")
                return finish(1)
            if PROGRESS.active:
                say("FAIL: progress still active after the run finished")
                return finish(1)
            say(f"last label before finishing: {samples[-1][2]!r}")

            blown = "".join(STDERR.seen)
            if "Traceback" in blown:
                say("FAIL: something raised while the UI was drawing:")
                for line in blown.splitlines()[-25:]:
                    say("    " + line)
                return finish(1)

            say("modal path completed, yielded, and drove the progress bar")
            STATE["phase"] = "break"
            return 0.2
        if time.perf_counter() - STATE["started"] > 120.0:
            say("FAIL: modal run did not finish within 120s")
            return finish(1)
        return 0.05

    if STATE["phase"] == "break":
        # A run that dies mid-flight must still put the progress bar away. If it
        # does not, the panel greys out the button the user would press to try
        # again and the add-on looks dead until it is toggled off and on.
        def boom(self, context):
            raise RuntimeError("gui_smoke: deliberate failure mid-run")

        ops.HYPERDEC_OT_decimate._update_progress = boom
        settings.last_report = ""
        obj = bpy.data.objects["smoke"]
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        result = bpy.ops.hyperdec.decimate('INVOKE_DEFAULT')
        say(f"deliberately broken run invoked: {sorted(result)}")
        STATE["started"] = time.perf_counter()
        STATE["phase"] = "recovered"
        return 0.4

    if STATE["phase"] == "recovered":
        if PROGRESS.active:
            if time.perf_counter() - STATE["started"] > 20.0:
                say("FAIL: a run that raised left the progress bar up, so the "
                    "panel stays greyed out and the add-on looks dead")
                return finish(1)
            return 0.2
        say("a run that raised cleared the progress bar, panel usable again")
        say("PASS: progress bar runs, moves, and recovers from a failed run")
        return finish(0)

    return finish(1)


bpy.app.timers.register(tick, first_interval=0.5)
