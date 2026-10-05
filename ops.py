"""The decimate operator. PLAN.md milestone M5.

Modal, and for a reason. A 13M-triangle sculpt is about six seconds of GPU work
spread over thirty passes. Running that in one blocking call freezes Blender with
no progress and no way out; stepping one pass per timer tick gives a real progress
bar and a working Escape key. `gpu_backend.simplify.Session` exists to make that
possible, and `core.dispatch.make_session` hands back the same interface, so
nothing here branches on internals.

**Redecimating.** A decimate output is tagged with the mesh it was made from
(`hd_src`), the object its masks were read from (`hd_srcobj`), and a content
hash of its own geometry (`hd_geo`). Running the operator on a tagged output
redirects the whole run at the *source*: the collapse, or the cached replay
of it, always works from the full-resolution mesh, never from an already
decimated result. Editing the output changes its hash and drops the tag, so
a sculpted-on result is treated as a new mesh. The source datablock is kept
alive for the session -- REPLACE orphans it rather than deleting it.
"""

from __future__ import annotations

import time

import bpy
import numpy as np

from .core import (
    cache as hd_cache, dispatch, egress, ingest, replay as rp,
    seams as seam_utils,
)
from .core.options import Options
from .props import PROGRESS

# How long to spend stepping before handing control back for a redraw. Long enough
# that dispatch overhead stays irrelevant, short enough to stay responsive.
SLICE_SECONDS = 0.05

# How much of the progress bar the GPU pass loop gets. The rest belongs to
# writing the result back, which on a big mesh is a second of mesh building and
# UV transfer. Reserving a tail for it is what stops the bar reaching 100% and
# then sitting there.
STAGE_ONE_SHARE = 0.85


def _mesh_targets(context, settings):
    if settings.apply_to == 'SELECTED':
        objs = [o for o in context.selected_objects if o.type == 'MESH']
        if context.active_object and context.active_object.type == 'MESH' \
                and context.active_object not in objs:
            objs.append(context.active_object)
        return objs
    obj = context.active_object
    return [obj] if obj and obj.type == 'MESH' else []


def _target_triangle_count(settings, face_count):
    if settings.mode == 'ABSOLUTE':
        return max(4, min(int(settings.target_faces), face_count))
    return max(4, int(face_count * settings.ratio))


def _read_density(obj, settings, vertex_count):
    """The density multiplier array, or None. Reports its own cost when slow."""
    if settings.density_source == 'NONE':
        return None, None
    if settings.density_source == 'VERTEX_GROUP':
        if not settings.density_group:
            return None, "no vertex group chosen, density ignored"
        weights = ingest.read_vertex_group_weights(obj, settings.density_group)
        if weights is None:
            return None, f"vertex group {settings.density_group!r} not found"
        note = None
        if vertex_count > 500_000:
            # Vertex groups have no bulk accessor, so this is a Python loop.
            note = (f"reading the vertex group over {vertex_count:,} vertices is "
                    "slow; a colour attribute is much faster")
        return ingest.density_from_weights(weights, settings.density_strength), note
    lum = ingest.read_color_attribute_luminance(obj.data,
                                                settings.density_attribute)
    if lum is None:
        return None, (f"colour attribute {settings.density_attribute!r} not found "
                      "on the point domain, density ignored")
    return ingest.density_from_weights(lum, settings.density_strength), None


def _compact(n):
    """13,040,640 -> '13.0M'. The panel's bar fits about 40 characters."""
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 10_000:
        return f"{n // 1_000:,}k"
    return f"{n:,}"


def _tag_panels(context):
    """Repaint the sidebar so the progress bar moves.

    A modal operator's timer ticks redraw nothing by themselves, so without this
    the bar is painted once at 0% and sits there until the run ends.

    The *region* is tagged, not the area. Tagging the area would redraw the 3D
    viewport too, and the viewport is currently displaying the multi-million
    triangle mesh being decimated, so that would spend more time drawing the
    progress than making any.
    """
    for window in context.window_manager.windows:
        for area in window.screen.areas:
            if area.type != 'VIEW_3D':
                continue
            for region in area.regions:
                if region.type == 'UI':
                    region.tag_redraw()


def _drop_tags(mesh):
    """Remove the preprocess tags from a mesh whose hash no longer matches.

    The tags say "this datablock is an unmodified decimate output". A mismatch
    means the user edited it, so it is its own mesh now and the pointer to a
    source would redirect a run at data the user no longer has on screen.
    """
    for key in ("hd_src", "hd_srcobj", "hd_geo"):
        if key in mesh:
            del mesh[key]


def _build_options(settings, density, seams, uvs=None):
    return Options(
        freeze_borders=settings.freeze_borders,
        max_normal_flip_deg=float(settings.max_normal_flip),
        optimal_placement=settings.optimal_placement,
        density=density,
        seams=seams,
        uvs=uvs,
    )


class _ReplaySession:
    """A cached run, replayed to a new target.

    Stands in for a live session: the operator steps it once and gets a
    finished result. The first decimation of a mesh recorded a collapse log
    down to a floor below its target; this target sits inside that range, so
    the whole answer is a prefix of the log, applied in the time the
    writeback takes anyway.
    """

    steppable = False
    passes = 0
    progress = 1.0

    def __init__(self, log, positions, triangles, target, uvs=None):
        self._log = log
        self._positions = positions
        self._triangles = triangles
        self._target = int(target)
        self._uvs = uvs
        self._result = None
        self.face_count = self._target

    def step(self):
        return False

    def finish(self):
        if self._result is None:
            self._result = rp.from_log(
                self._log, self._positions, self._triangles, self._target,
                uvs=self._uvs)
        return self._result

    def release(self):
        self._result = None


class HYPERDEC_OT_decimate(bpy.types.Operator):
    """Decimate the mesh on the GPU, preserving detail where it matters"""

    bl_idname = "hyperdec.decimate"
    bl_label = "Hyper Decimate"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        obj = context.active_object
        return obj is not None and obj.type == 'MESH' and obj.mode == 'OBJECT'

    # ------------------------------------------------------------ lifecycle

    def _reset(self, settings):
        self._settings = settings
        self._plan = dispatch.choose()
        self._notes = []
        self._done = []
        self._session = None
        self._current = None
        self._setup = None
        self._writeback = None
        self._writeback_step = 0
        self._created = []
        self._started = time.perf_counter()
        self._timer = None

    def invoke(self, context, event):
        if not self._prepare(context):
            return {'CANCELLED'}

        # The only reason to run straight through instead of modally:
        # background mode has no event loop to drive a modal operator.
        if bpy.app.background:
            return self._run_blocking(context)

        wm = context.window_manager
        wm.progress_begin(0.0, 1.0)
        # Only the modal path gets a panel progress bar. The blocking path never
        # returns to the event loop, so a bar it drew could never be repainted.
        PROGRESS.active = True
        PROGRESS.fraction = 0.0
        PROGRESS.label = ("starting" if len(self._queue) == 1
                          else f"{self._queue[0].name}: starting")
        _tag_panels(context)
        # Nothing is read here. The first timer tick starts the setup generator,
        # so the bar is on screen before any of the work begins.
        self._setup = self._setup_steps()
        self._timer = wm.event_timer_add(0.01, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def execute(self, context):
        """Blocking run, for scripts and for background mode."""
        if not self._prepare(context):
            return {'CANCELLED'}
        return self._run_blocking(context)

    def _prepare(self, context):
        """Build the object queue and reset per-run state.

        False when there is nothing to work on, with the error already reported.
        Shared by `invoke` and `execute` so the two entry points cannot disagree
        about what a run starts from.
        """
        settings = context.scene.hyper_decimate
        self._queue = _mesh_targets(context, settings)
        if not self._queue:
            self.report({'ERROR'}, "No mesh object selected")
            return False
        self._reset(settings)
        if not self._plan.is_gpu:
            detail = f" -- {self._plan.detail}" if self._plan.detail else ""
            self.report({'ERROR'},
                        f"Hyper Decimate needs a GPU: {self._plan.reason}"
                        f"{detail}")
            return False
        return True

    def _run_blocking(self, context):
        """Run every queued object to completion without yielding.

        Starting the first object is part of this rather than the caller's job:
        both entry points into the blocking path had their own copy of that
        check, which is two places to forget `_report_done` on an empty queue.
        """
        if not self._begin_next(context):
            return self._report_done(context)
        wm = context.window_manager
        wm.progress_begin(0.0, 1.0)
        try:
            while True:
                while self._session.step():
                    pass
                if not self._finish_current(context):
                    break
                if not self._begin_next(context):
                    break
        finally:
            wm.progress_end()
            dispatch.release()
        return self._report_done(context)

    def modal(self, context, event):
        """Fail-safe wrapper. The work is in `_step_modal`.

        An exception escaping here leaves the run's state behind: GPU textures
        unreleased and, worse for the user, the panel's progress flag still set,
        which greys out the very button they would press to try again. Blender
        prints the traceback and drops the handler, so without this the add-on
        would look dead until it was toggled off and on. The traceback is still
        re-raised; only the clearing up is added.
        """
        try:
            return self._step_modal(context, event)
        except Exception:
            self._cleanup(context)
            raise

    def _step_modal(self, context, event):
        if event.type in {'ESC', 'RIGHTMOUSE'}:
            self._notes.append("cancelled before finishing")
            return self._cleanup_and_report(context, cancelled=True)

        if event.type != 'TIMER':
            return {'RUNNING_MODAL'}

        # Starting an object is deferred to here rather than done in invoke, so
        # the panel is already showing what is happening while it happens.
        if self._setup is not None:
            return self._advance_setup(context)
        if self._writeback is not None:
            return self._advance_writeback(context)

        deadline = time.perf_counter() + SLICE_SECONDS
        while time.perf_counter() < deadline:
            if self._session.step():
                continue
            # This object is done. Writing it out is its own phase, stepped for
            # the same reason the pass loop is: on a big mesh it is seconds of
            # work, most of it building the result.
            if self._current is None:
                return self._cleanup_and_report(context)
            self._writeback = self._writeback_steps(context)
            self._writeback_step = 0
            return self._advance_writeback(context)

        self._update_progress(context)
        return {'RUNNING_MODAL'}

    def cancel(self, context):
        self._cleanup(context)

    # -------------------------------------------------------------- stages

    def _setup_steps(self):
        """Start the next queued object, a step at a time.

        A generator, and the yields are the point. Getting a 13M-triangle sculpt
        to the first pass is seconds of work before any pass has run: 1.1 s to
        read the mesh, 1.8 s to find its UV seams, then the upload. Done in one
        call that is Blender frozen with nothing on screen, which is exactly what
        it used to be.

        Each yield is the label for the work that comes *after* it, so the driver
        can put it on screen and let the panel repaint before that work starts.
        The user sees "finding UV seams" while the seams are being found, and
        Escape is live throughout, which on a mesh this size matters as much as
        the message does.

        Ends with a session in `self._session`, or with the queue exhausted and
        no session, which is how the caller tells the two apart.
        """
        from .gpu_backend.context import GPUUnavailable

        while self._queue:
            obj = self._queue.pop(0)
            mesh = obj.data
            if len(mesh.polygons) == 0:
                self._notes.append(f"{obj.name}: no faces, skipped")
                continue

            # Name the object only when there is more than one to tell apart.
            # The bar fits about 40 characters, and on a single-object run the
            # name is both redundant and enough to truncate the stage away.
            who = f"{obj.name}: " if (self._done or self._queue) else ""

            # The triangle count without reading the mesh: loops minus two per
            # face is exactly the fan count for triangles, quads and n-gons
            # alike. Worth having before the read, because a source small enough
            # to go straight to Blender should not be read in at all.
            tri_estimate = max(0, len(mesh.loops) - 2 * len(mesh.polygons))
            final_target = _target_triangle_count(self._settings, tri_estimate)
            tagged = "hd_src" in mesh
            if final_target >= tri_estimate and not tagged:
                self._notes.append(f"{obj.name}: already at or below target")
                continue

            yield f"{who}reading mesh"
            positions, tris, tri_loops, method = ingest.read_mesh(mesh)
            if tris.shape[0] == 0:
                self._notes.append(f"{obj.name}: no triangles, skipped")
                continue

            # A decimate output carries a pointer back to the mesh it was made
            # from. If this one does and it has not been edited since, the run
            # works from the source -- a replay of its cached log when the
            # target is in range, or a fresh preprocess of it when the settings
            # changed -- instead of decimating an already decimated mesh.
            src_mesh = mesh
            srcobj = obj
            if tagged:
                geo = hd_cache.mesh_key(positions, tris)
                src = bpy.data.meshes.get(mesh.get("hd_src", ""))
                if mesh.get("hd_geo") == geo and src is not None:
                    yield f"{who}reading preprocessed source"
                    positions, tris, tri_loops, method = ingest.read_mesh(src)
                    src_mesh = src
                    srcobj = bpy.data.objects.get(
                        mesh.get("hd_srcobj", ""), obj)
                    self._notes.append(
                        f"{obj.name}: redecimating the preprocessed source")
                    # The target was clamped against the decimated mesh's own
                    # size; on this path it means a count of the source's.
                    final_target = _target_triangle_count(
                        self._settings, tris.shape[0])
                else:
                    _drop_tags(mesh)

            if final_target >= tris.shape[0]:
                self._notes.append(f"{obj.name}: already at or below target")
                continue
            target = min(final_target, tris.shape[0])

            density = None
            if self._settings.density_source != 'NONE':
                yield f"{who}reading density map"
                density, note = _read_density(srcobj, self._settings,
                                              positions.shape[0])
                if note:
                    self._notes.append(f"{obj.name}: {note}")

            # Seams have to be constrained, not merely transferred afterwards.
            # Copying UVs onto a mesh whose seams were allowed to wander gives a
            # torn texture, so "Keep UVs" drives the constraint as well as the
            # transfer. See core/seams.py.
            #
            # The same goes for the UVs between the seams: the collapse refuses
            # to fold them over, which needs one UV per vertex. Read alongside
            # the seams, since both come from the same corner and UV arrays.
            seams = None
            uvs = None
            uv_layer = None
            if self._settings.keep_uvs and src_mesh.uv_layers:
                yield f"{who}finding UV seams"
                seams, uvs = seam_utils.read_seams_and_uvs(src_mesh)
                active = src_mesh.uv_layers.active or src_mesh.uv_layers[0]
                uv_layer = active.name
                if seams.any():
                    self._notes.append(
                        f"{obj.name}: {int(seams.sum()):,} UV seam vertices "
                        "constrained")

            opts = _build_options(self._settings, density, seams, uvs)

            # The session cache. A mesh's first decimation records a collapse
            # log to a floor below the target, and a later target inside that
            # range is a prefix of the same log -- applied in the time the
            # writeback takes anyway, with no passes at all. The key covers
            # the mesh content and every option that steers the sequence;
            # the target itself is deliberately outside it.
            depth = max(1, int(self._settings.cache_depth))
            ckey = (
                hd_cache.mesh_key(
                    positions, tris,
                    masks=(("density", density), ("seams", seams),
                           ("uvs", uvs))),
                hd_cache.options_key(opts),
            )
            log, why = hd_cache.lookup(ckey, final_target)
            if log is not None:
                self._notes.append(f"{obj.name}: {why}")
                self._session = _ReplaySession(log, positions, tris,
                                               final_target, uvs=uvs)
                self._current = {
                    "obj": obj, "mesh": src_mesh, "tri_loops": tri_loops,
                    "faces_in": tris.shape[0], "method": method,
                    "seams": seams, "uv_layer": uv_layer,
                    "final_target": final_target, "srcobj": srcobj.name, "ckey": ckey, "replayed": True,
                }
                return

            yield f"{who}uploading {_compact(tris.shape[0])} triangles"

            # How far below the target the log is recorded. Depth 1 is "only
            # the target itself", which is the old behaviour exactly; every
            # extra unit buys an octave of coarser targets that replay later
            # for the cost of a few cheap passes on an ever-smaller mesh.
            floor = max(4, target // depth)
            # `tris` goes across as the int32 the ingest read: the GPU tier
            # wants uint32 columns and widens them itself, so widening here
            # only built a copy twice the size of the mesh to discard.
            try:
                session, self._plan = dispatch.make_session(
                    positions, tris, target, opts, self._plan, floor=floor)
            except GPUUnavailable as exc:
                self._notes.append(f"{obj.name}: {exc}")
                continue
            self._session = session
            self._current = {
                "obj": obj, "mesh": src_mesh, "tri_loops": tri_loops,
                "faces_in": tris.shape[0], "method": method, "seams": seams,
                "uv_layer": uv_layer, "final_target": final_target,
                "srcobj": srcobj.name,
                "ckey": ckey, "replayed": False,
            }
            return

    def _begin_next(self, context):
        """Start the next object in one go. False when the queue is empty.

        The blocking path. Draining the generator runs every step back to back,
        which is what a script or a background run wants.
        """
        for _ in self._setup_steps():
            pass
        return self._session is not None

    def _advance_setup(self, context):
        """One step of starting an object, then back to the event loop."""
        try:
            label = next(self._setup)
        except StopIteration:
            self._setup = None
            if self._session is None:
                return self._cleanup_and_report(context)
            return {'RUNNING_MODAL'}

        PROGRESS.fraction = 0.0
        PROGRESS.label = label
        context.workspace.status_text_set(
            f"Hyper Decimate: {label}  (Esc to cancel)")
        _tag_panels(context)
        return {'RUNNING_MODAL'}

    def _writeback_steps(self, context):
        """Write the finished object back, a step at a time.

        A generator for the same reason `_setup_steps` is one. On a big sculpt
        this phase is seconds of work that used to happen in a single call:
        reading the result back, building the mesh, transferring UVs. The
        panel sat on whatever the last pass had reported for all of it. Each
        yield is the label for the work that follows, so the bar names the
        stage it is in.
        """
        cur = self._current
        if cur is None:
            return

        yield "reading back the result"
        result = self._session.finish()
        # What a live run recorded, for the cache. Taken before release: the
        # log is the one expensive thing the session holds that is not the
        # mesh itself, and it is a fraction of the mesh's size.
        session_log = getattr(self._session, "log", None)
        self._session.release()
        self._session = None

        obj, src_mesh = cur["obj"], cur["mesh"]
        yield "building the mesh"
        out = egress.build_mesh(
            f"{obj.data.name}_decimated_{result.triangles.shape[0]}",
            result.positions, result.triangles)
        # A new mesh with no `sharp_face` attribute reads as smooth, so flat
        # has to be asked for.
        egress.set_shading(out, self._settings.shade_smooth)
        egress.transfer_point_attributes(src_mesh, out, result.remap,
                                         result.survived)
        if self._settings.keep_uvs and src_mesh.uv_layers:
            yield "transferring UVs"
            # The seam mask is handed on, not recomputed: the UV transfer
            # needs to know which vertices carry more than one UV, and
            # finding that out costs a pass over every source corner.
            egress.transfer_uv_layers(src_mesh, out, cur["tri_loops"],
                                      result.face_origin, result.remap,
                                      result.survived,
                                      seam_mask=cur["seams"],
                                      tracked_uvs=result.uvs,
                                      tracked_layer=cur["uv_layer"])
        for material in src_mesh.materials:
            out.materials.append(material)

        # Keep what this run recorded under the mesh's cache key, so the next
        # target in its range replays instead of reprocessing.
        if session_log is not None and "ckey" in cur:
            hd_cache.store(cur["ckey"], session_log)

        # Tag the output with the mesh it was made from, so a later decimate
        # of this result replays (or reprocesses) the source. The geometry
        # hash is of the output as Blender will read it back -- float32
        # positions, int32 triangles -- because that is what the check runs on.
        out["hd_geo"] = hd_cache.mesh_key(
            np.asarray(result.positions, dtype=np.float32),
            np.asarray(result.triangles, dtype=np.int32))
        out["hd_src"] = src_mesh.name
        out["hd_srcobj"] = cur["srcobj"]

        if self._settings.output == 'REPLACE':
            replaced = obj.data
            obj.data = out
            # The datablock being replaced is removed when it has no other
            # users -- unless it is the preprocess source, which the tag chain
            # and the replay both still need.
            if replaced is not src_mesh and replaced.users == 0:
                bpy.data.meshes.remove(replaced)
        else:
            new_obj = bpy.data.objects.new(
                f"{obj.name}_decimated_{result.triangles.shape[0]}", out)
            new_obj.matrix_world = obj.matrix_world
            context.collection.objects.link(new_obj)
            self._created.append(new_obj)

        self._done.append((obj.name, cur["faces_in"], len(out.polygons),
                           cur["replayed"]))
        self._current = None

    def _finish_current(self, context):
        """Write the result back in one go. False when nothing is in flight.

        The blocking path, as `_begin_next` is for setup.
        """
        if self._current is None:
            return False
        for _ in self._writeback_steps(context):
            pass
        return True

    def _advance_writeback(self, context):
        """One step of writing a result back, then back to the event loop."""
        try:
            label = next(self._writeback)
        except StopIteration:
            self._writeback = None
            self._writeback_step = 0
            self._setup = self._setup_steps()
            return self._advance_setup(context)

        self._writeback_step += 1
        # The tail of the bar is reserved for this phase. Stage one maps onto
        # everything below STAGE_ONE_SHARE, so the bar cannot sit at 100% while
        # there is still work going on. How far through the writeback is is a
        # step count against a nominal four, which is a presentation choice: the
        # steps take wildly different times and none of them can be subdivided.
        fraction = STAGE_ONE_SHARE + (1.0 - STAGE_ONE_SHARE) * min(
            self._writeback_step / 4.0, 1.0)
        self._set_phase(context, label, fraction)
        return {'RUNNING_MODAL'}

    def _set_phase(self, context, label, fraction):
        """Put a named phase on the bar and repaint."""
        PROGRESS.fraction = fraction
        PROGRESS.label = label
        context.window_manager.progress_update(fraction)
        context.workspace.status_text_set(
            f"Hyper Decimate: {label}  (Esc to cancel)")
        _tag_panels(context)

    # ------------------------------------------------------------ reporting

    def _update_progress(self, context):
        wm = context.window_manager
        # Scaled into stage one's share of the bar before anything reads it,
        # so the cursor, the status bar and the panel all agree.
        fraction = (self._session.progress if self._session else 1.0)
        fraction *= STAGE_ONE_SHARE
        wm.progress_update(fraction)
        faces = self._session.face_count if self._session else 0
        passes = self._session.passes if self._session else 0
        label = f"{fraction * 100:.0f}%   {faces:,} tris   pass {passes}"
        if self._remaining:
            label += f"   ({self._remaining} more object"
            label += ")" if self._remaining == 1 else "s)"

        context.workspace.status_text_set(
            f"Hyper Decimate: {label}  (Esc to cancel)")

        # Only repaint when what is drawn would actually differ. The label
        # changes at whole percents, so this is around a hundred redraws across a
        # run instead of twenty a second.
        PROGRESS.fraction = fraction
        if label != PROGRESS.label:
            PROGRESS.label = label
            _tag_panels(context)

    @property
    def _remaining(self):
        """Objects still queued behind the one in flight."""
        return len(self._queue)

    def _cleanup(self, context):
        for name in ("_setup", "_writeback"):
            generator = getattr(self, name, None)
            if generator is not None:
                # Closing runs any `finally` inside the generator and drops its
                # references to the mesh arrays, which on a big sculpt is
                # hundreds of megabytes still reachable from a suspended frame.
                generator.close()
                setattr(self, name, None)
        if self._session is not None:
            self._session.release()
            self._session = None
        self._current = None
        if self._timer is not None:
            context.window_manager.event_timer_remove(self._timer)
            self._timer = None
        context.window_manager.progress_end()
        context.workspace.status_text_set(None)
        if PROGRESS.active:
            PROGRESS.reset()
            _tag_panels(context)
        dispatch.release()

    def _cleanup_and_report(self, context, cancelled=False):
        self._cleanup(context)
        return self._report_done(context, cancelled)

    def _select_results(self, context):
        """Leave the objects this run produced selected, with one active.

        Without this a run that outputs new objects finishes with the *source*
        still selected, so the next thing the user does lands on the mesh they
        just decimated rather than the result. Only the 'New Object' path needs
        it; replacing in place leaves the right object active already.
        """
        # Liveness is tested through `users_collection`, not through
        # `view_layer.objects`: the view layer has not been re-evaluated since
        # the objects were linked, so a freshly linked object is not in it yet
        # and filtering on it silently drops every result.
        alive = [o for o in self._created if o.users_collection]
        if not alive:
            return
        for obj in context.selected_objects:
            obj.select_set(False)
        for obj in alive:
            obj.select_set(True)
        context.view_layer.objects.active = alive[-1]

    def _report_done(self, context, cancelled=False):
        self._select_results(context)
        elapsed = time.perf_counter() - self._started
        settings = self._settings
        if not self._done:
            summary = "nothing decimated"
            settings.last_report = f"{summary}. " + "; ".join(self._notes)
            self.report({'WARNING'} if self._notes else {'INFO'}, summary)
            return {'CANCELLED'} if cancelled else {'FINISHED'}

        total_in = sum(row[1] for row in self._done)
        total_out = sum(row[2] for row in self._done)
        replayed = sum(1 for row in self._done if row[3])
        summary = (
            f"{total_in:,} to {total_out:,} "
            f"triangles ({total_out / total_in:.1%}) in {elapsed:.2f}s"
        )
        if len(self._done) > 1:
            summary += f" over {len(self._done)} objects"
        if replayed:
            summary += (", replayed" if len(self._done) == 1
                        else f", {replayed} replayed")
        if cancelled:
            summary = "cancelled, partial result. " + summary

        settings.last_report = summary + (
            ("  |  " + "; ".join(self._notes)) if self._notes else "")
        self.report({'INFO'}, summary)
        for note in self._notes:
            self.report({'WARNING'}, note)
        return {'FINISHED'}


class HYPERDEC_OT_clear_cache(bpy.types.Operator):
    """Drop the cached preprocess logs, freeing their memory"""

    bl_idname = "hyperdec.clear_cache"
    bl_label = "Clear Cache"
    bl_options = {'REGISTER', 'INTERNAL'}

    def execute(self, context):
        hd_cache.clear()
        self.report({'INFO'}, "Preprocess cache cleared")
        return {'FINISHED'}


CLASSES = (HYPERDEC_OT_decimate, HYPERDEC_OT_clear_cache)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
