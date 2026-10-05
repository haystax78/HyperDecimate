"""Per-scene settings. PLAN.md 6.

Kept on the scene rather than the object so a target survives switching between
sculpts, which is how the ZBrush workflow this is modelled on behaves.
"""

from __future__ import annotations

import bpy
from bpy.props import (
    BoolProperty, EnumProperty, FloatProperty, IntProperty, StringProperty,
)
from bpy.types import PropertyGroup

# One-click targets.
PRESETS = (1_000, 5_000, 10_000, 25_000, 50_000, 100_000, 500_000, 1_000_000)


class _Progress:
    """Live state of a running decimation, for the panel to draw.

    A module global rather than a scene property, on purpose. This is transient
    state belonging to one running operator, and a scene property is saved into
    the .blend: a crash or a forced quit mid-run would leave a file that opens
    for ever after showing a progress bar stuck at 40%. Nothing here outlives the
    process, and only one modal run can be in flight at a time, so there is
    nothing per-scene about it either.
    """

    __slots__ = ("active", "fraction", "label")

    def __init__(self):
        self.reset()

    def reset(self):
        self.active = False
        self.fraction = 0.0
        self.label = ""


PROGRESS = _Progress()


def _get_preset(self):
    """The preset nearest the current target, so the slider never goes stale."""
    return min(range(len(PRESETS)),
               key=lambda i: abs(PRESETS[i] - self.target_faces))


def _set_preset(self, index):
    self.mode = 'ABSOLUTE'
    self.target_faces = PRESETS[max(0, min(int(index), len(PRESETS) - 1))]


def _get_keep_uvs(self):
    return 0 if self.keep_uvs else 1


def _set_keep_uvs(self, value):
    self.keep_uvs = value == 0


class HyperDecimateSettings(PropertyGroup):
    mode: EnumProperty(
        name="Target",
        items=(
            ('RATIO', "Ratio", "Keep a fraction of the triangles"),
            ('ABSOLUTE', "Triangle Count", "Reduce to a triangle count"),
        ),
        default='RATIO',
    )
    ratio: FloatProperty(
        name="Ratio",
        description="Fraction of triangles to keep",
        default=0.1, min=0.0005, max=1.0, soft_min=0.01, subtype='FACTOR',
    )
    target_faces: IntProperty(
        name="Triangles",
        description="Triangle count to reduce to",
        default=100_000, min=4, soft_max=2_000_000,
    )
    preset: IntProperty(
        name="Preset",
        description="Snap the triangle target to a preset",
        min=0, max=len(PRESETS) - 1,
        get=_get_preset, set=_set_preset,
    )

    show_options: BoolProperty(
        name="Options",
        description="Show the options section",
        default=False,
    )
    freeze_borders: BoolProperty(
        name="Freeze Borders",
        description=(
            "Keep every boundary vertex exactly where it is. The border then "
            "cannot be simplified at all, which on an open mesh leaves a dense "
            "ring of vertices along the edge and puts a floor under the "
            "triangle count. Off, the border keeps its shape and path but is "
            "allowed to shorten along itself"
        ),
        default=False,
    )
    keep_uvs: BoolProperty(
        name="Keep UVs",
        description="Transfer UV layers onto the result",
        default=False,
    )
    keep_uvs_choice: EnumProperty(
        name="Keep UVs",
        description="Transfer UV layers onto the result",
        items=(
            ('YES', "Yes", "Transfer UV layers onto the result"),
            ('NO', "No", "Leave the result without UVs"),
        ),
        get=_get_keep_uvs, set=_set_keep_uvs,
    )
    shade_smooth: BoolProperty(
        name="Shade Smooth",
        description=(
            "Shade the result smooth. Off, it is shaded flat, which shows the "
            "decimated triangles as they are"
        ),
        default=False,
    )
    cache_depth: IntProperty(
        name="Cache Depth",
        description=(
            "How far below the requested target the collapse log is recorded, "
            "as a divisor. The first decimation of a mesh runs a little past "
            "its target to this floor, and any later target between the floor "
            "and the mesh's size replays from the log in a fraction of a "
            "second instead of processing again. 1 records only the target "
            "itself; 4 covers four times coarser targets"
        ),
        default=4, min=1, max=16,
    )
    optimal_placement: BoolProperty(
        name="Optimal Placement",
        description=(
            "Place each merged vertex at the point that best fits the surface "
            "the collapse replaced, rather than at the midpoint of the edge. "
            "Markedly more accurate. Turn it off if you need the result's "
            "vertices on the edge midpoints"
        ),
        default=True,
    )
    max_normal_flip: FloatProperty(
        name="Max Fold",
        description=(
            "Reject a collapse that would rotate any face normal by more than "
            "this. Lower is safer and decimates less"
        ),
        default=78.46304096718453, min=1.0, max=179.0,
    )

    density_source: EnumProperty(
        name="Density",
        description="Drive local resolution from a painted map",
        items=(
            ('NONE', "Uniform", "Same error budget everywhere"),
            ('VERTEX_GROUP', "Vertex Group", "Weight from a vertex group"),
            ('COLOR', "Color Attribute", "Brightness of a colour attribute"),
        ),
        default='NONE',
    )
    density_group: StringProperty(
        name="Group", description="Vertex group holding the weights")
    density_attribute: StringProperty(
        name="Attribute", description="Point-domain colour attribute")
    density_strength: FloatProperty(
        name="Strength",
        description=(
            "How much a fully weighted area resists collapsing, as a multiplier "
            "on its error"
        ),
        default=20.0, min=1.0, soft_max=1000.0,
    )

    apply_to: EnumProperty(
        name="Apply To",
        items=(
            ('ACTIVE', "Active", "The active object only"),
            ('SELECTED', "Selected", "Every selected mesh object"),
        ),
        default='ACTIVE',
    )
    output: EnumProperty(
        name="Output",
        items=(
            ('NEW', "New Object", "Add the result beside the original"),
            ('REPLACE', "Replace Mesh", "Replace the object's mesh data"),
        ),
        default='NEW',
    )


    # Filled in by the operator so the panel can show what happened.
    last_report: StringProperty(name="Last Run", default="")


CLASSES = (HyperDecimateSettings,)


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.hyper_decimate = bpy.props.PointerProperty(
        type=HyperDecimateSettings)


def unregister():
    # Disabling the add-on mid-run would otherwise leave the flag set, and the
    # panel would draw a bar for a run that no longer exists if it came back.
    PROGRESS.reset()
    del bpy.types.Scene.hyper_decimate
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
