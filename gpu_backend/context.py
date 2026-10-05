"""Texture-as-array plumbing for the GLSL compute backend. PLAN.md 2.2, 2.3, 2.5.

Blender's Python GPU API has no storage buffers, so every array lives in a
texture and is addressed by hand. All of that arithmetic is confined to this
module so it appears exactly once, in one place, in both Python and GLSL.

Everything here is grounded in what `tools/probe_gpu.py` measured:

  * The per-shader image budget is 8. `max_images_get()` reports a
    descriptor-pool ceiling in the millions on Vulkan, so it is never trusted.
  * Multi-component arrays go in layered images, because a 3-layer array costs
    one image unit instead of three.
  * `GPUTexture.read()` returns layer 0 only, so a layered array is flattened by
    a kernel before it can come back to the host.
  * `np.asarray()` on a `gpu.types.Buffer` is a genuine zero-copy view.
"""

from __future__ import annotations

import numpy as np

# Flat index i maps to texel (i & (PACK_W - 1), i >> PACK_SHIFT).
#
# The width is a power of two chosen per mesh by `plan_packing`, not a fixed
# constant. A 2D image holds PACK_W * max_texture_size elements, so the width is
# what sets the ceiling on array length. At the 4096 of PLAN.md 2.3 a 32,768px
# device holds 134M elements, which covers the 25M-triangle target but not the
# 156M corners of a 52M-triangle scan: that mesh needed 38,205 rows and died
# inside GPUTexture with nothing but "unknown error. See console".
#
# The narrowest width that fits is used rather than simply the widest the device
# allows, because `clear` and `read` work on the whole padded texture, so a width
# wider than the data needs makes every small array's round trip -- the per-pass
# histogram above all -- proportionally more expensive.
PACK_W_MIN = 4096
PACK_W = PACK_W_MIN
PACK_SHIFT = PACK_W.bit_length() - 1

# max_texture_size_get() as of the last probe. Only `probe` writes it; everything
# that needs it goes through `max_texture_size()`, which probes on demand so the
# packing can be planned without the caller having probed first.
_MAX_TEX = 0

# Hard budget, not a queried value. See PLAN.md 2.5.
IMAGE_BUDGET = 8

# Names that look like ordinary identifiers but are GLSL keywords. Using one as a
# resource name fails with a generic "Shader Compile Error" pointing into
# Blender's own prelude, which is a genuinely awful place to start debugging.
GLSL_RESERVED = frozenset({
    "flat", "smooth", "noperspective", "centroid", "sample", "patch",
    "in", "out", "inout", "uniform", "buffer", "shared", "coherent",
    "volatile", "restrict", "readonly", "writeonly", "layout", "precision",
    "invariant", "discard", "filter", "image", "sampler", "texture",
    "vertex", "fragment", "kernel", "main", "common", "active", "partition",
})

# Included ahead of every kernel. Keeps the addressing identical everywhere.
#
# Rebuilt whenever the packing width changes, because the width is baked into
# IDX2/IDX3 as a define. `Kernel` reads the module global at compile time, so
# every kernel compiled after a width change sees the new addressing; the ones
# compiled before it are dropped from the cache by `set_pack_width`.
def _build_prelude():
    return f"""
#define PACK_W {PACK_W}
#define PACK_SHIFT {PACK_SHIFT}
#define IDX2(i) ivec2(int(i) & (PACK_W - 1), int(i) >> PACK_SHIFT)
#define IDX3(i, l) ivec3(int(i) & (PACK_W - 1), int(i) >> PACK_SHIFT, l)
#define GID uint(gl_GlobalInvocationID.x)

// lowbias32. The NumPy copy lives in tests/fixtures.py, where it is the
// oracle the hash test diffs this kernel against.
//
// `seed` is xored in before mixing, so a different seed gives an unrelated
// ordering of the same vertices. The pass loop reseeds every pass; see
// kernels/candidates.SET_PRIORITY for what that is for. Seed 0 is not a special
// case, it is simply the seed the first pass uses.
uint hd_hash_seeded(uint x, uint seed) {{
  x ^= seed;
  x ^= x >> 16; x *= 0x7feb352du;
  x ^= x >> 15; x *= 0x846ca68bu;
  x ^= x >> 16;
  return x;
}}

uint hd_hash(uint x) {{ return hd_hash_seeded(x, 0u); }}
"""


PRELUDE = _build_prelude()

# GPUTexture format -> (numpy dtype, components, Buffer type, clear format)
#
# Signed integer formats are deliberately absent. Measured on Blender 5.2.2 with
# an RTX 4090: a kernel writes ivec4 values into an R32I image, and
# GPUTexture.read() then hands back a float32 buffer whose contents are neither
# the values nor their bit patterns. R32UI round-trips exactly under the same
# test. Rather than carry a format that silently corrupts data, the backend uses
# unsigned throughout and encodes "no value" as NO_INDEX below. Every index in
# this algorithm is non-negative anyway.
FORMATS = {
    "R32UI": (np.uint32, 1, "INT", "UINT"),
    "R32F": (np.float32, 1, "FLOAT", "FLOAT"),
    "RGBA32UI": (np.uint32, 4, "INT", "UINT"),
    "RGBA32F": (np.float32, 4, "FLOAT", "FLOAT"),
}

UNSUPPORTED_FORMATS = {
    "R32I": "signed integer readback is broken in Blender 5.2; use R32UI",
    "RGBA32I": "signed integer readback is broken in Blender 5.2; use RGBA32UI",
}

# Sentinel for "no such index", replacing the -1 a signed array would use.
NO_INDEX = np.uint32(0xFFFFFFFF)

IMAGE_TYPES = {
    "R32UI": ("UINT_2D", "UINT_2D_ARRAY"),
    "RGBA32UI": ("UINT_2D", "UINT_2D_ARRAY"),
    "R32F": ("FLOAT_2D", "FLOAT_2D_ARRAY"),
    "RGBA32F": ("FLOAT_2D", "FLOAT_2D_ARRAY"),
}


class GPUUnavailable(RuntimeError):
    """Raised when this device cannot run the Tier 1 backend."""


def probe():
    """Return a dict describing the device, or raise GPUUnavailable.

    Deliberately does not check `max_images_get()` against the budget, because
    that number is not a per-shader limit on every backend.
    """
    import gpu

    try:
        gpu.init()
    except Exception:
        pass  # already initialised, or a GUI session

    global _MAX_TEX

    max_tex = gpu.capabilities.max_texture_size_get()
    if max_tex < PACK_W_MIN:
        raise GPUUnavailable(
            f"max texture size {max_tex} is below the minimum packing width "
            f"{PACK_W_MIN}"
        )
    _MAX_TEX = int(max_tex)
    return {
        "backend": gpu.platform.backend_type_get(),
        "renderer": gpu.platform.renderer_get(),
        "max_texture_size": max_tex,
        # The widest packing this device allows, which is the real ceiling on how
        # long an array can be. `capacity_at` gives the figure for a given width.
        "capacity": max_tex * max_tex,
        "reported_max_images": gpu.capabilities.max_images_get(),
    }


def max_texture_size():
    """The device's max texture size, probing once if that has not happened."""
    if not _MAX_TEX:
        probe()
    return _MAX_TEX


def capacity_at(width):
    """How many elements a single 2D image holds at `width`."""
    return int(width) * max_texture_size()


def width_for(count):
    """The narrowest packing width that holds `count` elements in one image.

    Powers of two only, since IDX2 masks rather than divides, and never narrower
    than PACK_W_MIN so small meshes keep the layout every measurement in PLAN.md
    was taken against.
    """
    limit = max_texture_size()
    width = PACK_W_MIN
    while rows_for(count, width) > limit:
        if width >= limit:
            raise GPUUnavailable(
                f"{int(count):,} elements exceed this device's capacity of "
                f"{capacity_at(limit):,} at the widest usable packing "
                f"({limit} x {limit} texels)"
            )
        width *= 2
    return width


def set_pack_width(width):
    """Set the packing width, rebuilding the prelude and dropping stale kernels.

    Changing the width invalidates every compiled kernel, because IDX2 and IDX3
    bake it in as a define, and every Array already allocated, because its texture
    was sized for the old width. Only `plan_packing` should call this, and only
    before any Array for the mesh exists.
    """
    global PACK_W, PACK_SHIFT, PRELUDE

    width = int(width)
    if width & (width - 1):
        raise ValueError(f"packing width {width} is not a power of two")
    if width < PACK_W_MIN:
        raise ValueError(f"packing width {width} is below {PACK_W_MIN}")
    if width == PACK_W:
        return PACK_W

    PACK_W = width
    PACK_SHIFT = width.bit_length() - 1
    PRELUDE = _build_prelude()
    release_all()
    return PACK_W


def plan_packing(count):
    """Widen the packing if `count` elements will not fit, and return the width.

    Raises GPUUnavailable when no width fits, so the caller can report it
    rather than discovering the problem as a texture allocation failure
    partway through building the mesh state.
    """
    return set_pack_width(width_for(count))


def rows_for(count, width=None):
    """Texture height needed to hold `count` elements at `width`."""
    width = PACK_W if width is None else int(width)
    return max(1, (int(count) + width - 1) // width)


def new_texture(name, size, **kwargs):
    """GPUTexture, with a failure message that says what could not be allocated.

    `gpu.texture.new` reports every refusal as "unknown error. See console",
    whether the size is past max_texture_size or the device is simply out of
    memory, and the console line it points at is not always there. Since this is
    the one place a large mesh fails first, it is worth naming the array, the
    shape and the size in bytes.
    """
    import gpu

    try:
        return gpu.types.GPUTexture(size, **kwargs)
    except Exception as exc:
        fmt = kwargs.get("format", "R32UI")
        comps = FORMATS.get(fmt, (None, 1))[1]
        layers = int(kwargs.get("layers", 1) or 1)
        mb = size[0] * size[1] * comps * max(1, layers) * 4 / (1 << 20)
        limit = _MAX_TEX or "unknown"
        why = (
            f"{size[1]:,} rows is past the device limit of {limit}"
            if _MAX_TEX and size[1] > _MAX_TEX
            else f"most likely out of GPU memory ({mb:,.0f} MB requested)"
        )
        raise GPUUnavailable(
            f"could not allocate {name!r} as a {size[0]}x{size[1]} {fmt} "
            f"texture with {layers} layer(s), {mb:,.0f} MB: {why}"
        ) from exc


class Array:
    """A flat array living in a GPUTexture, optionally layered.

    `layers=1` gives a plain 2D image; more gives a 2D array image where layer
    `l` holds component block `l`. Layers are how a quadric's eleven floats fit
    into a single image unit.
    """

    __slots__ = ("name", "count", "format", "layers", "texture", "width",
                 "_dtype", "_comps", "_buftype")

    def __init__(self, name, count, fmt="R32UI", layers=1, data=None):
        if name in GLSL_RESERVED:
            raise ValueError(
                f"array name {name!r} is a reserved GLSL keyword; the resulting "
                "compile error points into Blender's prelude, not your kernel"
            )
        if fmt in UNSUPPORTED_FORMATS:
            raise ValueError(f"{fmt}: {UNSUPPORTED_FORMATS[fmt]}")
        if fmt not in FORMATS:
            raise ValueError(f"unsupported format {fmt!r}")

        self.name = name
        self.count = int(count)
        self.format = fmt
        self.layers = int(layers)
        self._dtype, self._comps, self._buftype, _ = FORMATS[fmt]

        # The width this texture was sized for. A later `set_pack_width` leaves
        # this Array addressable only by kernels compiled at the same width, so
        # `Kernel.run` checks it rather than letting the mismatch show up as
        # silently wrong indices.
        self.width = PACK_W

        # Integer formats silently ignore uploaded data, so they get an empty
        # texture here and are filled by a kernel instead. See `upload`.
        self._allocate(data if self.is_float else None)
        if data is not None and not self.is_float:
            self.clear(0)
            self._upload_integer(data)

    # -- host transfer ----------------------------------------------------

    def _allocate(self, data=None):
        """Create or replace the texture, with `data` if the format accepts it.

        The one place a GPUTexture for an Array is made, because `upload` has to
        make a new one rather than writing into the old: there is no partial
        write in the Python API for a float format, only construction with data.
        """
        size = (self.width, rows_for(self.count, self.width))
        kwargs = {"format": self.format}
        if self.layers > 1:
            kwargs["layers"] = self.layers
        if data is not None:
            kwargs["data"] = self._to_buffer(data, size)
        self.texture = new_texture(self.name, size, **kwargs)

    def _elements(self, size):
        return size[0] * size[1] * self._comps * max(1, self.layers)

    @property
    def is_float(self):
        return self._dtype == np.float32

    def _to_buffer(self, array, size):
        """numpy float array -> gpu.types.Buffer, padded to the full texture.

        Float formats only. See `upload` for why integers cannot take this path.

        The assignment through `np.asarray` is a zero-copy write into
        GPU-visible memory, which is what makes upload cheap; M0 measured 7 GB/s
        on OpenGL and up to 12.9 GB/s on Vulkan.
        """
        import gpu

        flat = np.ascontiguousarray(array, dtype=np.float32).reshape(-1)
        total = self._elements(size)
        if flat.size > total:
            raise ValueError(f"{flat.size} elements exceed capacity {total}")

        buf = gpu.types.Buffer("FLOAT", total)
        raw = np.asarray(buf)
        raw[: flat.size] = flat
        if flat.size < total:
            raw[flat.size:] = 0
        return buf

    def upload(self, array):
        """Replace the contents.

        Two facts about Blender 5.2 force the shape of this, both measured:

        1. `GPUTexture(data=...)` accepts only a `FLOAT` Buffer. Anything else
           raises "Only Buffer of format `FLOAT` is currently supported".
        2. Passing a FLOAT Buffer to an *integer* format silently does nothing.
           The texture stays entirely zero, whether the buffer holds the values
           as floats or their bit patterns reinterpreted. No error is raised,
           which makes it a genuinely nasty trap.

        So integer arrays cannot be uploaded at all by the direct route. They go
        through a float staging texture holding the low and high 16 bits as exact
        float values, and a one-line kernel packs them. Splitting at 16 bits
        keeps every value under 65536, well inside float32's exact integer range,
        so all 32 bits survive; uploading the value as a single float would be
        exact only below 2**24 and would silently corrupt indices above 16.7M,
        which is under our 25M target.
        """
        if not self.is_float:
            self._upload_integer(array)
            return
        self._allocate(array)

    def _upload_integer(self, array):
        if self.layers > 1 or self._comps > 1:
            raise NotImplementedError(
                "integer upload is implemented for single-component, "
                "single-layer arrays only"
            )
        flat = np.ascontiguousarray(array).reshape(-1)
        if flat.size > self.count:
            raise ValueError(
                f"{flat.size} elements exceed capacity {self.count}"
            )
        as_u32 = flat.astype(np.uint32, copy=False)

        # One float per value, bit for bit, when the device round-trips the bits
        # (see `bitcast_ok`); otherwise the two-halves-as-floats fallback below.
        # The bitcast staging texture is a quarter the size of the RGBA32F one and
        # the host side is a single contiguous memcpy instead of two strided column
        # writes, which measured 5.3x faster on 13M values: 0.192s -> 0.036s.
        # Sized at the full count, not at what was handed in: `_to_buffer` pads the
        # tail with zeros and the kernel runs over the whole array, so a short
        # upload clears the elements it does not set rather than leaving whatever
        # was there before.
        if bitcast_ok():
            stage = Array(f"{self.name}_stage", self.count, fmt="R32F",
                          data=as_u32.view(np.float32))
            source, fmt = _PACK_BITS, "R32F"
        else:
            # Element i of an RGBA32F array occupies texels 4i..4i+3, so a plain
            # (n, 4) host array lands every half where IDX2 will look for it.
            staged = np.zeros((self.count, 4), dtype=np.float32)
            staged[: as_u32.size, 0] = (as_u32 & np.uint32(0xFFFF)).astype(np.float32)
            staged[: as_u32.size, 1] = (as_u32 >> np.uint32(16)).astype(np.float32)
            stage = Array(f"{self.name}_stage", self.count, fmt="RGBA32F",
                          data=staged)
            source, fmt = _PACK_INT, "RGBA32F"

        probe = probe_array()
        pack = cached_kernel(
            "hd_pack_int", source,
            [("stage_src", fmt), ("int_dst", self.format),
             ("probe_dst", self.format)],
            push_constants=(("INT", "n"),),
        )
        # `stage` must outlive the dispatch, and the result must be verified.
        #
        # Sizes travel as a push constant rather than a define specifically so
        # that one shader serves every array. When N was a define, each new array
        # size compiled a fresh shader, and on OpenGL that shader's *first*
        # dispatch was sometimes dropped. The upload then silently did nothing, so
        # a mesh's triangle arrays were all zeros, every face was (0,0,0), the pass
        # loop found no candidates and gave up after one pass, and the result was
        # 4,418 degenerate faces sharing a single vertex. It reproduced 4 times out
        # of 5 on OpenGL, never on Vulkan, and never when the same call ran without
        # other shaders having been compiled first.
        #
        # The retry is belt and braces on top of that: a silent upload failure
        # corrupts everything downstream, so it is worth one texel to catch.
        #
        # The check reads that texel from a one-element array the pack kernel
        # copies it into, not from this array: `GPUTexture.read()` has no partial
        # form, so verifying in place pulled the whole texture back over the bus to
        # look at one value -- 18 ms on 13M elements, 70 ms on 52M, per array.
        for _attempt in range(2):
            pack.run(self.count,
                     bind={"stage_src": stage, "int_dst": self,
                           "probe_dst": probe},
                     n=self.count)
            if as_u32.size == 0 or int(probe.download(1)[0]) == int(as_u32[0]):
                return
        raise RuntimeError(
            f"upload to {self.name!r} did not take effect after two attempts"
        )

    def download(self, count=None, layer_stride=None):
        """Read back as numpy. Layered arrays must be flattened first.

        `GPUTexture.read()` returns only layer 0 of a layered texture, so calling
        this on one is a bug rather than a slow path, and it says so.
        """
        if self.layers > 1:
            raise RuntimeError(
                f"{self.name!r} is layered; GPUTexture.read() returns layer 0 "
                "only. Flatten it with a kernel into a plain 2D Array first."
            )
        n = self.count if count is None else int(count)
        raw = np.asarray(self.texture.read()).reshape(-1)
        if raw.dtype != self._dtype:
            raw = raw.view(self._dtype)
        if self._comps > 1:
            out = raw.reshape(-1, self._comps)[:n]
        else:
            out = raw[:n]
        # A copy, not a view. `np.asarray` on a gpu.types.Buffer aliases GPU-side
        # memory, so returning a slice of it would hand the caller an array that
        # silently changes when the texture is read again, and would keep the
        # Buffer alive for as long as they hold it. Live GPU objects at
        # interpreter exit segfault Blender; see release_all.
        return np.array(out, copy=True)

    def clear(self, value=0):
        """Fill every texel with `value`.

        `GPUTexture.clear` marshals integers through a signed int, so anything
        above 2**31 - 1 fails with "one or more items could not be used as a
        int". That rules out clearing to the all-bits NO_INDEX sentinel, which is
        exactly what the claim pass needs. Such values go through a fill kernel
        instead, where imageStore has no such problem.
        """
        if not self.is_float and int(value) > 0x7FFFFFFF:
            self.fill(value)
            return
        fmt = FORMATS[self.format][3]
        vals = (value,) * self._comps
        self.texture.clear(format=fmt, value=vals)

    def fill(self, value):
        """clear() for integer values that exceed what clear() can marshal."""
        if self.is_float or self.layers > 1 or self._comps > 1:
            raise NotImplementedError(
                "fill is implemented for single-component unsigned arrays only"
            )
        hi = (int(value) >> 16) & 0xFFFF
        lo = int(value) & 0xFFFF
        kernel = cached_kernel(
            "hd_fill", _FILL, [("fill_dst", self.format)],
            push_constants=(("INT", "n"), ("INT", "hi"), ("INT", "lo")),
        )
        kernel.run(self.count, bind={"fill_dst": self},
                   n=self.count, hi=hi, lo=lo)

    # -- shader declaration ----------------------------------------------

    def declaration(self, slot, access=("READ", "WRITE")):
        """The tuple `Kernel` needs to bind this array to an image slot."""
        image_type = IMAGE_TYPES[self.format][1 if self.layers > 1 else 0]
        return (slot, self.format, image_type, self.name, set(access))

    def __repr__(self):
        return (f"Array({self.name!r}, count={self.count:,}, "
                f"{self.format}, layers={self.layers})")


class Kernel:
    """A compute shader with named image slots that any Array can be bound to.

    Declarations are names and formats, not Array instances, deliberately. A
    primitive like the prefix-sum scan runs over different arrays on every level
    of its recursion, and compiling a fresh shader each time would dominate its
    cost. Declaring `("src", "R32UI")` once and binding whichever Array is needed
    at dispatch keeps compilation to once per kernel for the whole session.

    Pass `images` either as `[(name, format), ...]`, as
    `[(name, format, layers), ...]`, or as a list of Arrays, in which case each
    Array's own name and format are used.
    """

    __slots__ = ("name", "shader", "slots", "width", "_consts", "_local",
                 "_defaults")

    def __init__(self, name, source, images, defines=None, local_size=64,
                 push_constants=()):
        import gpu

        # The width baked into this shader's IDX2/IDX3 by the prelude.
        self.width = PACK_W

        specs = [_image_spec(entry) for entry in images]
        if len(specs) > IMAGE_BUDGET:
            raise ValueError(
                f"kernel {name!r} binds {len(specs)} images, over the budget of "
                f"{IMAGE_BUDGET}. Pack components into layers instead."
            )

        info = gpu.types.GPUShaderCreateInfo()
        for slot, (res_name, fmt, layers, access) in enumerate(specs):
            if res_name in GLSL_RESERVED:
                raise ValueError(
                    f"image name {res_name!r} is a reserved GLSL keyword"
                )
            if fmt in UNSUPPORTED_FORMATS:
                raise ValueError(f"{fmt}: {UNSUPPORTED_FORMATS[fmt]}")
            image_type = IMAGE_TYPES[fmt][1 if layers > 1 else 0]
            info.image(slot, fmt, image_type, res_name, qualifiers=set(access))
        for ctype, cname in push_constants:
            info.push_constant(ctype, cname)
        for key, val in (defines or {}).items():
            info.define(str(key), str(val))
        # All three arguments, always: Blender bug #145818 makes the omitted
        # dimensions default to -1 rather than 1.
        info.local_group_size(int(local_size), 1, 1)
        info.compute_source(PRELUDE + source)

        self.name = name
        self.slots = [spec[0] for spec in specs]
        self._consts = {cname: ctype for ctype, cname in push_constants}
        self._local = int(local_size)
        # Arrays passed directly become the default bindings, which keeps simple
        # one-shot kernels readable without a bind dict.
        self._defaults = {
            e.name: e for e in images if isinstance(e, Array)
        }
        self.shader = gpu.shader.create_from_info(info)

    def run(self, threads, bind=None, **constants):
        """Bind arrays, set constants, and dispatch enough groups for `threads`.

        `bind` maps declared slot names to Arrays. Omit it to bind by matching
        names when the kernel was built from Arrays directly.
        """
        import gpu

        bound = dict(self._defaults)
        if bind:
            bound.update(bind)
        bind = bound
        missing = [s for s in self.slots if s not in bind]
        if missing:
            raise KeyError(f"kernel {self.name!r} missing bindings: {missing}")
        for slot in self.slots:
            array = bind[slot]
            # A width mismatch is not a slow path or a partial result: the shader
            # computes texel coordinates from one width and the texture is laid
            # out at another, so every read lands somewhere arbitrary and the
            # output looks like plausible garbage. Cheaper to refuse.
            if array.width != self.width:
                raise RuntimeError(
                    f"{array.name!r} is packed at width {array.width} but kernel "
                    f"{self.name!r} was compiled for {self.width}; the Array "
                    "predates a set_pack_width and must be rebuilt"
                )
            self.shader.image(slot, array.texture)

        for cname, value in constants.items():
            ctype = self._consts.get(cname)
            if ctype is None:
                raise KeyError(f"{self.name!r} has no push constant {cname!r}")
            # A one-element sequence is rejected ("sequence length is 1,
            # expected [2 - 16]"), so a scalar must be passed as a bare number.
            scalar = not isinstance(value, (list, tuple))
            if ctype in ("INT", "IVEC2", "IVEC3", "IVEC4", "UINT", "BOOL"):
                self.shader.uniform_int(
                    cname, int(value) if scalar else list(value))
            else:
                self.shader.uniform_float(
                    cname, float(value) if scalar else list(value))

        groups = (int(threads) + self._local - 1) // self._local
        gpu.compute.dispatch(self.shader, max(1, groups), 1, 1)

    def __repr__(self):
        return f"Kernel({self.name!r}, slots={self.slots})"


# Compiled kernels, keyed by everything that affects the generated shader.
#
# This cache is not only an optimisation, it is a correctness requirement. A
# Kernel holds a GPUShader, and a dispatch is queued rather than executed
# immediately. A Kernel created as a local variable is garbage collected as soon
# as the helper returns, which frees the shader while its dispatch may still be
# pending; the work is then silently dropped. The symptom is an array that reads
# back as whatever was in that texture memory before, which on a fresh allocation
# can look like plausible float data and sends you hunting for a conversion bug.
#
# Every transient kernel in this module goes through here, so the shader outlives
# any dispatch that references it and is compiled once per session rather than
# once per call.
_KERNEL_CACHE = {}


def release_all():
    """Drop every cached shader.

    A GPUShader and a GPUTexture belong to the GPU context that created them. If
    Python still holds them when Blender tears that context down, the objects are
    freed afterwards against a context that no longer exists, and the process
    dies with a segmentation fault after all the work has succeeded, which is a
    memorably confusing way to end a passing test run.

    Call this before a script exits, from an addon's `unregister`, and before a
    file load. Holding a `GPUState` past that point has the same hazard, so drop
    those references too.
    """
    global _PROBE

    _KERNEL_CACHE.clear()
    # The upload probe is a live texture and has the same teardown hazard. The
    # bitcast measurement is a property of the device, not of the context, so it
    # is kept.
    _PROBE = None


def cached_kernel(name, source, images, defines=None, local_size=64,
                  push_constants=()):
    """A Kernel that is compiled once and kept alive for the session."""
    key = (
        name, source, local_size, PACK_W,
        tuple(_image_spec(e)[:3] for e in images),
        tuple(sorted((str(k), str(v)) for k, v in (defines or {}).items())),
        tuple(push_constants),
    )
    kernel = _KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = Kernel(name, source, images, defines=defines,
                        local_size=local_size,
                        push_constants=push_constants)
        _KERNEL_CACHE[key] = kernel
    return kernel


def _image_spec(entry):
    """Normalise an images entry to (name, format, layers, access)."""
    if isinstance(entry, Array):
        return (entry.name, entry.format, entry.layers, ("READ", "WRITE"))
    if len(entry) == 2:
        return (entry[0], entry[1], 1, ("READ", "WRITE"))
    if len(entry) == 3:
        return (entry[0], entry[1], entry[2], ("READ", "WRITE"))
    return tuple(entry)


# Fills an unsigned array with a value clear() cannot marshal. Split into two
# 16-bit halves so the value travels as defines rather than a push constant.
_FILL = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  imageStore(fill_dst, IDX2(i), uvec4((uint(hi) << 16) | uint(lo)));
}
"""


# Both routes out of a float staging texture into a 32-bit integer array; see
# Array.upload for why the detour exists at all. They differ only in how the
# value is recovered from the texel, so the rest is written once: each also
# copies element 0 into a one-element probe array, which is what the upload check
# reads, and that line going missing from one of them would disable the check for
# that route alone, silently and only on the devices that use it.
# `imageLoad` returns a vec4 whatever the image's component count, so one load
# serves both: the split route reads .r and .g, the bitcast route only .r.
_PACK_BODY = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  vec4 t = imageLoad(stage_src, IDX2(i));
  uint v = %s;
  imageStore(int_dst, IDX2(i), uvec4(v));
  if (i == 0u) imageStore(probe_dst, IDX2(0u), uvec4(v));
}
"""

# The fallback: the value arrives as two 16-bit halves held as exact floats.
_PACK_INT = _PACK_BODY % "(uint(t.g) << 16) | uint(t.r)"

# The fast path: one float per value carrying the integer's bit pattern, undone
# with floatBitsToUint. A quarter the staging memory of the 16-bit split and a
# single contiguous host memcpy rather than two strided column writes.
#
# This is only safe where the bits survive the trip. Some of the values an index
# array holds are NaNs and infinities when read as float -- NO_INDEX is
# 0xFFFFFFFF, a quiet NaN -- and a driver is free to canonicalise a NaN payload,
# which would silently rewrite indices. `bitcast_ok` checks before this is used.
_PACK_BITS = _PACK_BODY % "floatBitsToUint(t.r)"


# A one-element scratch array the pack kernels copy element 0 into, so the upload
# check costs one texture row instead of reading the whole destination back.
_PROBE = None

# Whether this device round-trips a float32 bit pattern through an R32F image.
# None until measured; see `bitcast_ok`.
_BITCAST_OK = None

# Values that would be destroyed by NaN canonicalisation or by a float conversion
# anywhere in the chain: both infinities, a signalling and a quiet NaN, the
# sentinel, and the boundaries of float32's exact integer range.
_BITCAST_PROBES = np.array([
    0, 1, 0xFFFF, 0x10000, (1 << 24) - 1, 1 << 24, (1 << 24) + 1,
    25_000_000, (1 << 31) - 1, 1 << 31, 0x7F800000, 0x7FC00000, 0x7F800001,
    0xFF800000, 0xFFC00000, 0xFFFFFFFE, 0xFFFFFFFF,
], dtype=np.uint32)


def probe_array():
    global _PROBE
    if _PROBE is None or _PROBE.width != PACK_W:
        _PROBE = Array("hd_upload_probe", 1, fmt="R32UI")
    return _PROBE


def sync(written):
    """Block until the dispatch that wrote `written` has finished.

    `written` must be an Array that dispatch actually stored to. That is the
    whole point, and the obvious shortcut does not work: **reading a texture the
    dispatch did not write does not wait for it.** Measured on this machine,
    dispatching the position solve over 6.5M vertices and then reading one texel:

        from a one-element scratch array      0.3 ms (OpenGL), 4.1 ms (Vulkan)
        from the array the solve wrote       47.5 ms (OpenGL), 31.8 ms (Vulkan)

    and reading the scratch array first did not make the second read any cheaper.
    Drivers track dependencies per resource, so an unrelated read is serviced
    while the dispatch is still running. A one-texel readback is still the only
    synchronisation Blender's Python GPU API offers -- `Array.upload` leans on the
    same thing -- but it has to be a texel the dispatch produced. Where the real
    output cannot be read back, as with a layered `quad`, the kernel writes a
    sentinel into a scratch array and that is what gets read.

    Call this before dropping a texture a dispatch was reading. A dispatch is
    queued rather than executed, so freeing its inputs on the next Python line
    frees memory the GPU has not finished with, and the symptom is output that is
    silently wrong with nothing pointing at the free that caused it.
    """
    written.download(1)


def bitcast_ok():
    """True if an integer survives a trip through an R32F image as its own bits.

    Measured once per session on the values most likely to be mangled, because
    the gain is large -- 5.3x on a 13M-element upload -- and the failure mode is
    silent corruption of index data rather than an error. Nothing here assumes
    the answer: a device that canonicalises NaN payloads simply gets the 16-bit
    split instead.
    """
    global _BITCAST_OK
    if _BITCAST_OK is not None:
        return _BITCAST_OK

    # False while probing, so the staging upload inside this call takes the
    # fallback path and cannot recurse.
    _BITCAST_OK = False
    try:
        want = _BITCAST_PROBES
        stage = Array("hd_bitcast_probe_src", want.size, fmt="R32F",
                      data=want.view(np.float32))
        dst = Array("hd_bitcast_probe_dst", want.size, fmt="R32UI")
        dst.clear(0)
        kernel = cached_kernel(
            "hd_pack_int", _PACK_BITS,
            [("stage_src", "R32F"), ("int_dst", "R32UI"),
             ("probe_dst", "R32UI")],
            push_constants=(("INT", "n"),),
        )
        kernel.run(want.size,
                   bind={"stage_src": stage, "int_dst": dst,
                         "probe_dst": probe_array()},
                   n=want.size)
        _BITCAST_OK = bool(np.array_equal(dst.download(), want))
    except Exception:  # noqa: BLE001 - any failure means use the fallback
        _BITCAST_OK = False
    return _BITCAST_OK


# Flattens one layer of a layered array into a plain 2D array so it can be read
# back, because GPUTexture.read() ignores layers beyond the first. The resource
# names are substituted rather than fixed: a Kernel declares each image under its
# Array's own name, so hardcoding `src` and `dst` here compiles to undefined
# variables.
_FLATTEN = """
void main() {
  uint i = GID;
  if (i >= uint(n)) return;
  imageStore(flat_dst, IDX2(i), imageLoad(lay_src, IDX3(i, layer)));
}
"""


def download_layer(src, layer, scratch=None):
    """Read one layer of a layered Array back to the host.

    All components of the layer survive, not just the first: a quadric lives in
    three RGBA32F layers and needs every channel.
    """
    if src.layers <= 1:
        return src.download()
    # Same format as the source, so component count and type both carry over.
    dst = scratch or Array(f"{src.name}_flat", src.count, fmt=src.format)
    kernel = cached_kernel(
        "hd_flatten", _FLATTEN,
        [("lay_src", src.format, src.layers), ("flat_dst", dst.format)],
        push_constants=(("INT", "n"), ("INT", "layer")),
    )
    kernel.run(src.count, bind={"lay_src": src, "flat_dst": dst},
               n=src.count, layer=int(layer))
    return dst.download()
