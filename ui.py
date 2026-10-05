"""N-panel UI. PLAN.md 6."""

from __future__ import annotations

import os

import bpy
import bpy.utils.previews

from .props import PRESETS, PROGRESS

_LOGO_PATH = os.path.join(
    os.path.dirname(__file__), "icons", "hyper_dec_logo.png")

# Custom images do not go through a layout's `icon=` arguments -- those only
# name Blender's built-ins. A file on disk has to be loaded into a preview
# collection at register time, which hands back an entry whose icon_id
# `template_icon` can draw. The collection must outlive the registration, so it
# is kept here and removed in unregister().
_PREVIEWS = None

# The switches that belong behind the fold: things with a sensible default that
# most runs never touch. Order is deliberate, not alphabetical -- what a result
# keeps, then how it is built.
OPTION_PROPS = (
    "freeze_borders",
    "shade_smooth",
    "optimal_placement",
    "cache_depth",
    "max_normal_flip",
)


class HYPERDEC_PT_panel(bpy.types.Panel):
    bl_label = "Hyper Decimate"
    bl_idname = "HYPERDEC_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "Hyper Decimate"

    def draw(self, context):
        outer = self.layout
        _draw_logo(outer, context)
        settings = context.scene.hyper_decimate
        obj = context.active_object

        if PROGRESS.active:
            _draw_progress(outer)

        # Settings are read once when the operator starts, so changing them
        # mid-run would do nothing. Greying them out says so rather than letting
        # a user move a slider and wonder why it had no effect.
        live = not PROGRESS.active
        layout = outer.column()
        layout.enabled = live

        box = layout.box()
        col = box.column(align=True)
        # An expanded enum stacks vertically inside a column; a row puts the
        # two buttons side by side.
        col.row(align=True).prop(settings, "mode", expand=True)
        if settings.mode == 'RATIO':
            col.prop(settings, "ratio")
        else:
            col.prop(settings, "target_faces")

        title = box.row()
        title.alignment = 'CENTER'
        title.label(text="Presets")
        box.prop(settings, "preset", text="", slider=True)
        # Tick labels under the slider, the nearest preset in brackets.
        current = settings.preset
        ticks = box.row(align=True)
        ticks.scale_y = 0.7
        for i, count in enumerate(PRESETS):
            if count >= 1_000_000:
                label = f"{count // 1_000_000}M"
            else:
                label = f"{count // 1000}k" if count >= 1000 else str(count)
            cell = ticks.row()
            cell.alignment = 'CENTER'
            cell.label(text=f"[{label}]" if i == current else label)

        box = layout.box()
        box.prop(settings, "density_source")
        if settings.density_source == 'VERTEX_GROUP':
            if obj is not None:
                box.prop_search(settings, "density_group", obj, "vertex_groups",
                                text="Group")
            box.prop(settings, "density_strength")
        elif settings.density_source == 'COLOR':
            if obj is not None and obj.type == 'MESH':
                box.prop_search(settings, "density_attribute", obj.data,
                                "color_attributes", text="Attribute")
            box.prop(settings, "density_strength")

        col = layout.box().column(align=True)
        _choice_row(col, settings, "apply_to", "Apply To:")
        _choice_row(col, settings, "output", "Output:")
        _choice_row(col, settings, "keep_uvs_choice", "Keep UVs:")

        _draw_options(layout, settings, live)

        layout = outer.column()
        layout.enabled = live

        layout.operator("hyperdec.decimate", icon='MOD_DECIM')

        if settings.mode == 'RATIO' and settings.ratio < 0.02:
            note = layout.box()
            note.scale_y = 0.7
            note.label(text="Below 2%, detail falls off.", icon='INFO')
            note.label(text="Check the result before relying on it.")

        has_mesh = obj is not None and obj.type == 'MESH'
        if has_mesh or settings.last_report:
            box = layout.box()

        if has_mesh:
            tris = _triangle_estimate(obj.data)
            info = box.column(align=True)
            info.label(text=f"~{tris:,} tris")
            # A tagged output decimates from its source, so the target and
            # the count shown are of the source, not of this mesh.
            src = bpy.data.meshes.get(obj.data.get("hd_src", ""))
            if src is not None:
                src_tris = _triangle_estimate(src)
                info.label(
                    text=f"source: ~{src_tris:,} tris",
                    icon='LINKED')
                tris = src_tris
            target = (max(4, int(tris * settings.ratio))
                      if settings.mode == 'RATIO'
                      else min(int(settings.target_faces), tris))
            info.label(text=f"target ~{target:,} tris")

        if settings.last_report:
            if has_mesh:
                box.separator()
            box.label(text="Last run")
            for line in _wrap(settings.last_report, 34):
                box.label(text=line)


def _choice_row(layout, settings, prop, label):
    """A label with every choice of an enum as a button beside it.

    Quicker than a dropdown for two options. The split factor matches where a
    dropdown would start its value column.
    """
    split = layout.split(factor=0.27)
    split.label(text=label)
    buttons = split.row(align=True)
    for item in settings.bl_rna.properties[prop].enum_items:
        buttons.prop_enum(settings, prop, item.identifier)


def _draw_logo(layout, context):
    """The banner across the top of the panel, spanning the layout's width.

    `template_icon` takes a scale, not a width: the button it makes is `scale`
    UI units square, so a pixel width has to be converted back through the
    interface scale, minus the panel's own padding and the scrollbar, about a
    unit and a half all told. The last few percent are shaved off as well so a
    slightly off estimate clips nothing: the controls below fill the usable
    width exactly. The logo is drawn at 45% of that.

    The button is square and Blender sizes a preview by the button's height,
    so a wide logo only comes out as wide as the button is tall. Squashing the
    row vertically to the logo's aspect, as this once did, shrank the logo
    itself to a third of the panel; the row has to stay square, which leaves
    blank margin above and below the image.
    """
    if _PREVIEWS is None:
        return
    logo = _PREVIEWS.get("logo")
    if logo is None:
        return
    ui_scale = getattr(context.preferences.system, "ui_scale", 0) or 1.0
    units = context.region.width / (20.0 * ui_scale) - 1.5
    row = layout.row()
    row.alignment = 'CENTER'
    row.template_icon(icon_value=logo.icon_id, scale=max(1.0, 0.45 * units))

    # Imported here, not at the top: the package defines __version__ after it
    # imports this module, so only by draw time is it certain to exist.
    from . import __version__
    note = layout.row()
    note.alignment = 'CENTER'
    note.scale_y = 0.7
    note.label(text=f"v{__version__}")


def _draw_options(layout, settings, live):
    """The collapsible Options section: a flat toggle row, then a box.

    This used to be `UILayout.panel`, whose header Blender indents past the
    panel's left edge and which cannot be placed in a column or a box, so it
    could not line up with Apply To and Output. A plain boolean drawn without
    emboss collapses just as well and sits flush with them.
    """
    col = layout.column()
    col.enabled = live
    row = col.row()
    row.alignment = 'LEFT'
    row.prop(settings, "show_options", text="Options", emboss=False,
             icon='DOWNARROW_HLT' if settings.show_options
             else 'RIGHTARROW')
    if not settings.show_options:
        return
    target = col.box().column(align=True)
    for name in OPTION_PROPS:
        target.prop(settings, name)
    target.operator("hyperdec.clear_cache")


# Whether UILayout.progress worked the first time the panel reached for it. None
# until a running decimation has been drawn at least once. Exposed because
# "nothing raised" does not distinguish a drawn bar from a silent fallback, and
# tools/gui_smoke.py has to be able to tell the difference.
PROGRESS_WIDGET_OK = None


def _draw_progress(layout):
    """The running-decimation block: a real bar, the live numbers, and the way out.

    `UILayout.progress` does exist in 5.2, this add-on's minimum, but only as an
    RNA function: it is absent from `dir(UILayout)` and
    `hasattr(bpy.types.UILayout, "progress")` is False, so feature-detecting it
    that way reports it missing on a Blender that has it. Hence calling it and
    catching, with the numbers alone as the fallback so a future build that moves
    or renames it degrades to a readable panel rather than a broken one.
    """
    global PROGRESS_WIDGET_OK

    box = layout.box()
    col = box.column(align=True)
    try:
        col.progress(factor=PROGRESS.fraction, text=PROGRESS.label, type='BAR')
        if PROGRESS_WIDGET_OK is None:
            PROGRESS_WIDGET_OK = True
    except (AttributeError, TypeError) as exc:
        if PROGRESS_WIDGET_OK is None:
            PROGRESS_WIDGET_OK = False
            # Once, not every repaint: draw runs many times a second.
            print(f"Hyper Decimate: UILayout.progress is unavailable ({exc}); "
                  "showing progress as text instead")
        col.label(text=PROGRESS.label)
    col.separator()
    col.label(text="Esc or right-click to cancel", icon='CANCEL')


def _triangle_estimate(mesh):
    """Triangles without tessellating: loops minus two per face."""
    return max(0, len(mesh.loops) - 2 * len(mesh.polygons))


def _wrap(text, width):
    words, line, out = text.split(), "", []
    for word in words:
        if line and len(line) + 1 + len(word) > width:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return out[:6]


CLASSES = (HYPERDEC_PT_panel,)


def register():
    global _PREVIEWS
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    if os.path.exists(_LOGO_PATH):
        _PREVIEWS = bpy.utils.previews.new()
        _PREVIEWS.load("logo", _LOGO_PATH, 'IMAGE')


def unregister():
    global _PREVIEWS
    if _PREVIEWS is not None:
        bpy.utils.previews.remove(_PREVIEWS)
        _PREVIEWS = None
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
