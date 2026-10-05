"""Run a test script or module with `hyper_decimate` bound to this checkout.

Usage:
    blender --background --gpu-backend vulkan --python tools/run_suite.py \
        -- tests/test_gpu_simplify.py
    python tools/run_suite.py tests.test_addon

Importing `hyper_decimate` normally works when the checkout sits in a directory
of that name, as it does here, but a sibling add-on with the same import name
would shadow it. Blender ignores PYTHONPATH, so the package is preloaded under
the expected name with its real directory as the submodule search location
before the target runs.
"""

from __future__ import annotations

import importlib.util
import os
import runpy
import sys

_PACKAGE = "hyper_decimate"
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def bind_package():
    """Register this checkout as `hyper_decimate` in sys.modules."""
    spec = importlib.util.spec_from_file_location(
        _PACKAGE,
        os.path.join(_ROOT, "__init__.py"),
        submodule_search_locations=[_ROOT],
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_PACKAGE] = mod
    spec.loader.exec_module(mod)


def main():
    bind_package()
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv \
        else sys.argv[1:]
    target = argv[0]
    sys.argv = argv
    if target.endswith(".py") or os.sep in target or "/" in target:
        runpy.run_path(target, run_name="__main__")
    else:
        # A bare dotted name is a submodule of the bound package:
        # `tests.test_addon` means `hyper_decimate.tests.…`.
        runpy.run_module(f"{_PACKAGE}.{target}", run_name="__main__")


if __name__ == "__main__":
    main()
