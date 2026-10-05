"""Milestone 0 probe for Hyper Decimate.

Answers, empirically and per machine, every GPU API question that the Tier 1
design in PLAN.md depends on. Nothing in gpu_backend/ should be written before
this reports PASS for the atomics and layered-image sections.

Run inside Blender's text editor, or headless:

    blender --background --python tools/probe_gpu.py

Exit code is 0 if every required probe passed, 1 otherwise.
"""

import sys
import time
import traceback

import gpu
import numpy as np

RESULTS = []
W = 4096  # flat-index packing width, see PLAN.md 2.3

# Images bound to any single compute shader. OpenGL guarantees only 8 and
# reports exactly that; Vulkan reports a descriptor-pool ceiling in the
# millions, which is not a per-shader budget. Pick the honest floor and stay
# under it by packing multi-component arrays into layered images.
IMAGE_BUDGET = 8


def record(name, ok, detail="", required=True):
    RESULTS.append((name, ok, detail, required))
    mark = "PASS" if ok else ("FAIL" if required else "n/a ")
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail else ""))


def probe(name, required=True):
    """Decorator: run a probe, catch anything, record the outcome."""
    def wrap(fn):
        print(f"\n{name}")
        try:
            ok, detail = fn()
            record(name, ok, detail, required)
        except Exception as exc:  # noqa: BLE001 - a probe failing is data
            record(name, False, f"{type(exc).__name__}: {exc}", required)
            if "-v" in sys.argv:
                traceback.print_exc()
        return fn
    return wrap


# GLSL keywords that are legal-looking Python identifiers and so are easy to
# use as a resource name by accident. Blender reports the resulting failure as a
# generic "Shader Compile Error" pointing at a line in its own prelude, which is
# extremely misleading, so catch it here instead. Metal reserves more still
# (vertex, fragment, kernel, and the whole MSL type set).
GLSL_RESERVED = {
    "flat", "smooth", "noperspective", "centroid", "sample", "patch",
    "in", "out", "inout", "uniform", "buffer", "shared", "coherent",
    "volatile", "restrict", "readonly", "writeonly", "layout", "precision",
    "invariant", "discard", "filter", "image", "sampler", "texture",
    "vertex", "fragment", "kernel", "main",
}


def make_compute(source, images, local_size=(64, 1, 1), defines=None):
    """Build a compute shader. images is a list of (slot, format, type, name)."""
    info = gpu.types.GPUShaderCreateInfo()
    for slot, fmt, imgtype, name in images:
        if name in GLSL_RESERVED:
            raise ValueError(
                f"resource name {name!r} is a reserved GLSL keyword; "
                "the compile error it causes points at Blender's prelude, not here"
            )
        info.image(slot, fmt, imgtype, name, qualifiers={"READ", "WRITE"})
    for key, val in (defines or {}).items():
        info.define(key, str(val))
    # Always pass all three: see blender bug #145818, local_group_size defaults
    # to -1 rather than 1 when y/z are omitted.
    info.local_group_size(*local_size)
    info.compute_source(source)
    return gpu.shader.create_from_info(info)


def read_u32(tex):
    """Read a UINT texture back as a numpy array."""
    return np.asarray(tex.read()).astype(np.uint32, copy=False)


# ---------------------------------------------------------------- capabilities

def section_capabilities():
    print("\n=== Capabilities ===")
    caps = {}

    @probe("compute shader support")
    def _():
        caps["compute"] = gpu.capabilities.compute_shader_support_get()
        return caps["compute"], str(caps["compute"])

    @probe("image load/store support")
    def _():
        caps["imgls"] = gpu.capabilities.shader_image_load_store_support_get()
        return caps["imgls"], str(caps["imgls"])

    @probe("max image units >= design budget")
    def _():
        n = gpu.capabilities.max_images_get()
        caps["max_images"] = n
        return n >= IMAGE_BUDGET, f"reports {n:,}, budget {IMAGE_BUDGET}"

    @probe("max texture size")
    def _():
        n = gpu.capabilities.max_texture_size_get()
        caps["max_tex"] = n
        cap = W * n
        return n >= 4096, f"{n}px -> {cap:,} elements at W={W}"

    @probe("work group limits")
    def _():
        size = [gpu.capabilities.max_work_group_size_get(i) for i in range(3)]
        count = [gpu.capabilities.max_work_group_count_get(i) for i in range(3)]
        caps["wg_size"], caps["wg_count"] = size, count
        return size[0] >= 64, f"size={size} count={count}"

    print(f"\n  backend: {gpu.platform.backend_type_get()}")
    print(f"  device:  {gpu.platform.device_type_get()}")
    print(f"  renderer:{gpu.platform.renderer_get()}")
    print(f"  vendor:  {gpu.platform.vendor_get()}")
    return caps


# --------------------------------------------------------------- basic compute

def section_basic():
    print("\n=== Basic compute + image store ===")

    @probe("imageStore to R32UI, 1D-packed addressing")
    def _():
        n = 1 << 16
        h = n // W
        tex = gpu.types.GPUTexture((W, h), format="R32UI")
        tex.clear(format="UINT", value=(0,))
        sh = make_compute(
            """
            void main() {
              uint i = gl_GlobalInvocationID.x;
              if (i >= uint(N)) return;
              ivec2 c = ivec2(int(i) & (PACK_W - 1), int(i) >> PACK_SHIFT);
              imageStore(dst, c, uvec4(i * 3u + 1u, 0u, 0u, 0u));
            }""",
            [(0, "R32UI", "UINT_2D", "dst")],
            defines={"N": n, "PACK_W": W, "PACK_SHIFT": W.bit_length() - 1},
        )
        sh.image("dst", tex)
        gpu.compute.dispatch(sh, (n + 63) // 64, 1, 1)
        got = read_u32(tex).reshape(-1)[:n]
        want = np.arange(n, dtype=np.uint32) * 3 + 1
        bad = int(np.count_nonzero(got != want))
        return bad == 0, f"{n:,} texels, {bad} mismatched"


# ------------------------------------------------------------------- atomics

def section_atomics():
    print("\n=== Integer image atomics (gates Tier 1) ===")

    def atomic_case(op_src, clear_val, expect, threads=4096):
        tex = gpu.types.GPUTexture((16, 16), format="R32UI")
        tex.clear(format="UINT", value=(clear_val,))
        sh = make_compute(
            "void main() { uint i = gl_GlobalInvocationID.x;"
            " if (i >= uint(N)) return; ivec2 c = ivec2(0, 0); " + op_src + " }",
            [(0, "R32UI", "UINT_2D", "acc")],
            defines={"N": threads},
        )
        sh.image("acc", tex)
        gpu.compute.dispatch(sh, (threads + 63) // 64, 1, 1)
        got = int(read_u32(tex).reshape(-1)[0])
        return got == expect, f"got {got}, expected {expect}"

    @probe("imageAtomicAdd on r32ui")
    def _():
        # Kernels 3, 5 and 7 in PLAN.md 4.4 depend on this.
        return atomic_case("imageAtomicAdd(acc, c, 1u);", 0, 4096)

    @probe("imageAtomicMin on r32ui")
    def _():
        # Kernel 8, the collapse claim pass.
        return atomic_case("imageAtomicMin(acc, c, i + 1u);", 999999, 1)

    @probe("imageAtomicCompSwap on r32ui", required=False)
    def _():
        # Not required by the current design, probed in case we need a
        # compare-and-swap allocator later.
        return atomic_case("imageAtomicCompSwap(acc, c, 0u, 7u);", 0, 7)

    @probe("atomic scatter correctness under contention")
    def _():
        # Mimics kernel 5: many threads bumping per-row cursors.
        rows, per_row = 1024, 64
        n = rows * per_row
        tex = gpu.types.GPUTexture((W, 1), format="R32UI")
        tex.clear(format="UINT", value=(0,))
        sh = make_compute(
            """
            void main() {
              uint i = gl_GlobalInvocationID.x;
              if (i >= uint(N)) return;
              ivec2 c = ivec2(int(i % uint(ROWS)), 0);
              imageAtomicAdd(cursor, c, 1u);
            }""",
            [(0, "R32UI", "UINT_2D", "cursor")],
            defines={"N": n, "ROWS": rows},
        )
        sh.image("cursor", tex)
        gpu.compute.dispatch(sh, (n + 63) // 64, 1, 1)
        got = read_u32(tex).reshape(-1)[:rows]
        bad = int(np.count_nonzero(got != per_row))
        return bad == 0, f"{rows} counters, {bad} wrong"


# ------------------------------------------------------------- layered images

def section_layered():
    print("\n=== Layered images (needed to fit quadrics in 8 slots) ===")

    LAYER_WRITE = """
        void main() {
          uint i = gl_GlobalInvocationID.x;
          if (i >= 4096u) return;
          ivec2 c = ivec2(int(i) & 63, int(i) >> 6);
          for (int l = 0; l < LAYERS; l++)
            imageStore(arr, ivec3(c, l), uvec4(i + uint(l) * 1000u, 0u, 0u, 0u));
        }"""

    # Reads every layer back in-shader and flattens to a plain 2D image, which
    # is the only thing GPUTexture.read() reliably returns in full.
    LAYER_VERIFY = """
        void main() {
          uint i = gl_GlobalInvocationID.x;
          if (i >= 4096u) return;
          ivec2 c = ivec2(int(i) & 63, int(i) >> 6);
          uint bad = 0u;
          for (int l = 0; l < LAYERS; l++) {
            uint got = imageLoad(arr, ivec3(c, l)).r;
            if (got != i + uint(l) * 1000u) bad += 1u;
          }
          imageStore(flat_img, c, uvec4(bad, 0u, 0u, 0u));
        }"""

    @probe("UINT_2D_ARRAY read+write across all layers, in shader")
    def _():
        layers = 3
        arr = gpu.types.GPUTexture((64, 64), layers=layers, format="R32UI")
        arr.clear(format="UINT", value=(0,))
        flat_img = gpu.types.GPUTexture((64, 64), format="R32UI")
        flat_img.clear(format="UINT", value=(9,))

        wr = make_compute(
            LAYER_WRITE,
            [(0, "R32UI", "UINT_2D_ARRAY", "arr")],
            defines={"LAYERS": layers},
        )
        wr.image("arr", arr)
        gpu.compute.dispatch(wr, 64, 1, 1)

        ver = make_compute(
            LAYER_VERIFY,
            [(0, "R32UI", "UINT_2D_ARRAY", "arr"), (1, "R32UI", "UINT_2D", "flat_img")],
            defines={"LAYERS": layers},
        )
        ver.image("arr", arr)
        ver.image("flat_img", flat_img)
        gpu.compute.dispatch(ver, 64, 1, 1)

        bad = int(read_u32(flat_img).reshape(-1).sum())
        return bad == 0, f"{layers} layers x 4096 texels, {bad} wrong"

    @probe("GPUTexture.read() returns all layers", required=False)
    def _():
        # Informational. If this fails we simply never read a layered texture
        # directly; final results are flattened by a kernel first.
        layers = 3
        arr = gpu.types.GPUTexture((64, 64), layers=layers, format="R32UI")
        arr.clear(format="UINT", value=(0,))
        n = read_u32(arr).reshape(-1).size
        want = layers * 64 * 64
        return n == want, f"read() gave {n} of {want} texels"


# ------------------------------------------------------------- binding limits

def section_binding():
    print("\n=== Binding limits (max_images_get reported 8) ===")

    @probe("max_images_get() is a usable per-shader budget", required=False)
    def _():
        # On Vulkan this returns a descriptor-pool ceiling (observed: 1,048,576
        # on an RTX 4090), not a per-shader image limit. On OpenGL it returns
        # the real limit, 8. Never use it as a loop bound: the design assumes
        # IMAGE_BUDGET and treats anything larger as unverified.
        n = gpu.capabilities.max_images_get()
        return n <= 64, f"reports {n:,}; design budget is {IMAGE_BUDGET}"

    @probe(f"bind {IMAGE_BUDGET} images to one compute shader")
    def _():
        n = min(gpu.capabilities.max_images_get(), IMAGE_BUDGET)
        texs = [gpu.types.GPUTexture((64, 64), format="R32UI") for _ in range(n)]
        for t in texs:
            t.clear(format="UINT", value=(0,))
        decls = [(i, "R32UI", "UINT_2D", f"img{i}") for i in range(n)]
        body = "\n".join(
            f"  imageStore(img{i}, c, uvec4({i + 1}u, 0u, 0u, 0u));" for i in range(n)
        )
        sh = make_compute(
            "void main() {\n"
            "  uint i = gl_GlobalInvocationID.x;\n"
            "  if (i >= 4096u) return;\n"
            "  ivec2 c = ivec2(int(i) & 63, int(i) >> 6);\n" + body + "\n}",
            decls,
        )
        for i, t in enumerate(texs):
            sh.image(f"img{i}", t)
        gpu.compute.dispatch(sh, 64, 1, 1)
        got = [int(read_u32(t).reshape(-1)[0]) for t in texs]
        ok = got == list(range(1, n + 1))
        return ok, f"{n} slots bound, values {got}"


# -------------------------------------------------------------- host transfer

def section_transfer():
    print("\n=== Host transfer ===")

    @probe("np.asarray on gpu.types.Buffer is a zero-copy view")
    def _():
        buf = gpu.types.Buffer("FLOAT", 1024)
        arr = np.asarray(buf)
        arr[:] = np.arange(1024, dtype=np.float32)
        # If the view aliases the Buffer, mutating arr changes buf.
        return abs(buf[513] - 513.0) < 1e-4, f"buf[513]={buf[513]}"

    @probe("large texture allocation and upload", required=False)
    def _():
        # 4096 x 8192 RGBA32F = 512 MB, the scale a 25M-vertex mesh needs.
        h = 8192
        n = W * h * 4
        buf = gpu.types.Buffer("FLOAT", n)
        np.asarray(buf)[:] = 1.5
        t0 = time.perf_counter()
        tex = gpu.types.GPUTexture((W, h), format="RGBA32F", data=buf)
        up = time.perf_counter() - t0
        t0 = time.perf_counter()
        back = np.asarray(tex.read())
        down = time.perf_counter() - t0
        mb = n * 4 / 1e6
        ok = abs(float(back.reshape(-1)[0]) - 1.5) < 1e-5
        return ok, (
            f"{mb:.0f} MB up {up:.2f}s ({mb/up/1000:.1f} GB/s), "
            f"down {down:.2f}s ({mb/down/1000:.1f} GB/s)"
        )


# ------------------------------------------------------------------- reporting

def main():
    print("Hyper Decimate — Milestone 0 GPU probe")
    print("Blender GPU backend init: ", end="")
    try:
        gpu.init()  # required under --background, harmless otherwise
        print("ok")
    except Exception as exc:  # noqa: BLE001
        print(f"failed ({exc}) — continuing, may already be initialised")

    caps = section_capabilities()
    if not caps.get("compute"):
        print("\nCompute shaders unavailable. Tier 1 is impossible on this device.")
        print("Verdict: CPU only.")
        return 1

    section_basic()
    section_atomics()
    section_layered()
    section_binding()
    section_transfer()

    print("\n=== Verdict ===")
    required = [r for r in RESULTS if r[3]]
    failed = [r for r in required if not r[1]]
    optional_failed = [r for r in RESULTS if not r[3] and not r[1]]

    for name, _, detail, _ in optional_failed:
        print(f"  optional probe did not pass: {name} — {detail}")

    if not failed:
        print(f"  {len(required)}/{len(required)} required probes passed.")
        print("  Tier 1 (GPU compute) is viable on this device.")
        return 0

    print(f"  {len(failed)} of {len(required)} required probes failed:")
    for name, _, detail, _ in failed:
        print(f"    - {name}: {detail}")
    atomics_failed = any("atomic" in n.lower() for n, _, _, _ in failed)
    if atomics_failed:
        print(
            "\n  Integer image atomics are unavailable. Per PLAN.md 2.4, run\n"
            "  kernels 3, 5 and 8 on the CPU and keep the rest on GPU, or fall\n"
            "  back to Tier 2 entirely. Measure both before choosing."
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
