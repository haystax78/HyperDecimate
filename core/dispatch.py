"""GPU dispatch. PLAN.md 3.

There is exactly one path: the GLSL compute backend collapsing the source
mesh directly, recording a log as it goes. `choose` probes the device once per
run and reports why it cannot run when it cannot -- with no second backend
behind it, an unavailable GPU is a user-visible error, not a silent fallback.
"""

from __future__ import annotations

import time


class Plan:
    """What will run, and why -- or what will not, and why not."""

    def __init__(self, ok, reason, detail=""):
        self.ok = ok
        self.reason = reason
        self.detail = detail

    @property
    def is_gpu(self):
        return self.ok

    @property
    def tier(self):
        return "GPU" if self.ok else "no GPU"

    def __repr__(self):
        return f"Plan({'GPU' if self.ok else 'unavailable'}, {self.reason!r})"


def choose():
    """Probe for the GPU compute path.

    A plan with `is_gpu` False carries the reason it cannot run; the operator
    reports it rather than pretending, because there is nothing to fall back
    to.
    """
    try:
        from ..gpu_backend import context as ctx
        info = ctx.probe()
    except Exception as exc:  # noqa: BLE001 - any failure means no GPU tier
        return Plan(False, "GPU unavailable", str(exc))

    return Plan(True, f"{info['backend']} on {info['renderer']}")


def make_session(positions, triangles, target_triangles, opts, plan=None,
                 floor=None):
    """A stepping session on the GPU. Returns (session, plan).

    The session exposes step, finish, release, progress and passes, and it is
    steppable: the pass loop yields between passes, which is what makes a
    progress bar and a cancel key mean anything.

    `floor`, below the target, is how deep the collapse log is recorded while
    the run happens; a later target inside the log's range replays from the
    session cache instead of processing again. See `core/replay.py` and
    `core/cache.py`.

    Raises `GPUUnavailable` when the mesh will not fit on the device -- the
    arrays have to fit the texture packing and the available VRAM, and a
    52M-triangle scan needs 156M adjacency entries where a 13M-triangle one
    needs 39M. The caller reports it; there is no slower path to quietly take.
    """
    plan = plan or choose()
    if not plan.is_gpu:
        from ..gpu_backend.context import GPUUnavailable
        raise GPUUnavailable(plan.detail or plan.reason)

    from ..gpu_backend.simplify import Session

    session = Session(positions, triangles, target_triangles, opts,
                      floor=floor)
    session.steppable = True
    return session, plan


def run(positions, triangles, target_triangles, opts, plan=None,
        on_progress=None, floor=None):
    """Decimate on the GPU, blocking.

    `on_progress(fraction, passes)` is called between passes.

    Returns (result, info) where info carries the pass count and timings.
    """
    plan = plan or choose()
    t0 = time.perf_counter()
    session, plan = make_session(
        positions, triangles, target_triangles, opts, plan, floor=floor)
    info = {"tier": plan.tier, "reason": plan.reason, "detail": plan.detail,
            "notes": []}
    try:
        if on_progress:
            on_progress(0.0, 0)
        while session.step():
            if on_progress:
                on_progress(session.progress, session.passes)
        result = session.finish()
        info["passes"] = session.passes
        info["schedule"] = getattr(session, "schedule", None)
        info["stats"] = getattr(session, "stats", None)
        info["log"] = getattr(session, "log", None)
    finally:
        session.release()

    info["seconds"] = time.perf_counter() - t0
    return result, info


def release():
    """Drop cached GPU objects. Safe to call when the GPU tier never ran."""
    try:
        from ..gpu_backend import context as ctx
        ctx.release_all()
    except Exception:  # noqa: BLE001
        pass
