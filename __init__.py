"""Hyper Decimate — GPU-accelerated mesh decimation for Blender 5.2.

The numeric core (core/) deliberately imports no bpy, so it can be
unit-tested with plain python plus numpy, which iterates far faster than
launching Blender. The GPU backend needs Blender's `gpu` module, so tests of
it run inside Blender itself. This module therefore tolerates bpy being
absent, and only wires up the Blender UI when it is present.
"""

from __future__ import annotations

# Blender finds an add-on two different ways, and it needs both entries to be
# findable both ways.
#
#   * As an **extension**, from `blender_manifest.toml`, when installed into an
#     extension repository. This is the shipping path.
#   * As a **legacy add-on**, from `bl_info`, when the containing folder sits in a
#     script directory's `addons/` subfolder. This is the development path, and it
#     is how anyone working on the code will actually load it.
#
# With only the manifest, the add-on is invisible to a script directory: Blender
# scans those for `bl_info` and never looks at the manifest, so the folder is
# silently skipped and nothing appears in the Add-ons list. Verified directly, a
# script directory holding this folder discovered 17 other legacy add-ons and not
# this one.
#
# Blender reads `bl_info` by parsing the file rather than importing it, so it has
# to stay a plain literal dict at module level. That means the version is repeated
# here and in the manifest; `tests/test_addon.py` asserts the two agree so they
# cannot drift.
bl_info = {
    "name": "Hyper Decimate",
    "author": "MattGPT",
    "version": (0, 4, 0),
    "blender": (5, 2, 0),
    "location": "View3D > Sidebar > Hyper Decimate",
    "description": "GPU quadric decimation for multi-million triangle sculpts",
    "category": "Mesh",
}

__version__ = "0.4.0"

try:
    import bpy  # noqa: F401
except ImportError:  # running under plain python, for tests
    bpy = None

_MODULES = ()

if bpy is not None:
    from . import ops, props, ui
    _MODULES = (props, ops, ui)


def register():
    if bpy is None:
        raise RuntimeError("Hyper Decimate requires Blender to register")
    for module in _MODULES:
        module.register()


def unregister():
    for module in reversed(_MODULES):
        module.unregister()
    # Shaders and textures belong to the GPU context that made them. Holding them
    # past teardown crashes Blender; see gpu_backend.context.release_all.
    from .core import cache, dispatch
    dispatch.release()
    # A cached collapse log is only valid against the code that produced it;
    # disabling the add-on drops them all so a stale log can never outlive it.
    cache.clear()
