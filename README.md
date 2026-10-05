# Hyper Decimate

GPU mesh decimation for Blender, built for meshes large enough that Blender's own
Decimate modifier becomes impractical. A 50-million-triangle scan preprocesses and decimates in
around 15 seconds on a 4090, and every target after that replays in a small fraction of that.

Requires Blender 5.2 or newer and a GPU Blender can run compute shaders on.
No dependencies to install: the pipeline uses Blender's own `gpu` module and the
bundled NumPy.

## Install

Download the latest release from https://github.com/haystax78/HyperDecimate/releases
and extract into your addons directory. Alternatively the zip should support being installed as
an extension but I haven't tested this.

## Use

Open the 3D viewport sidebar (**N**) and pick the **Hyper Decimate** tab.

Set a target, either as a **Ratio** of the current triangle count or as an
absolute **Triangle Count**, and press **Hyper Decimate**. The **Presets** slider
snaps the target to an absolute count: 1k, 5k, 10k, 25k, 50k, 100k, 500k or 1M.
The panel shows the active mesh's triangle estimate, the target, and, for a result
made by this add-on, the source it decimates from.

The run is cancellable with **Esc** or right-click, and reports progress on the
panel throughout. When it produces a new object, that object is left selected and
active.

After decimation, if the original mesh has been replaced, you don't need to undo
in order to decimate to a different triangle count as the original mesh is
still cached and preprocessed.

## Settings

| | |
| --- | --- |
| **Density** | Keep more detail where a vertex group or colour attribute is bright. **Strength** is how strongly a fully weighted area resists collapsing |
| **Apply To** | The active object, or every selected mesh object |
| **Output** | Add a new object beside the original, or replace the original's mesh |
| **Keep UVs** | Yes or No, No by default. Transfer UV layers to the result. Seams are detected as UV discontinuities and constrained, so the two sides of a seam are not welded together |

Under **Options**:

| | |
| --- | --- |
| **Freeze Borders** | Off by default. On, boundary vertices never move, which prevents an open border being simplified at all and puts a floor under the triangle count. Off, the border keeps its shape and path but may shorten along itself |
| **Shade Smooth** | Off by default, so the result is shaded flat and shows the decimated triangles as they are |
| **Optimal Placement** | On by default. Move each surviving vertex to the point that best fits the surface it replaced, rather than the edge midpoint. Markedly more accurate. Turn it off if you need result vertices on edge midpoints |
| **Cache Depth** | How far below the requested target the preprocess logs collapses. Any later target inside that range replays the log instead of reprocessing |
| **Max Fold** | Reject a collapse that would rotate a face normal by more than this. Lower is safer and decimates less |

**Clear Cache** drops the cached collapse logs to free memory.

## How it works

One stage: a GPU quadric-error simplifier running directly on the source
mesh. It collapses many non-interfering edges per pass rather than one at a
time, which is what makes it fast on meshes where a conventional
priority-queue simplifier is too slow to be usable. There is no clustering
stage and nothing welds vertices — which is why coincident shells in a scan
stay as separate shells instead of z-fighting.

Every collapse solves its pair of quadrics for the position that least distorts
the surface and places the survivor there — it is not restricted to one of the
two original endpoints — so a pass's flip and link tests are judged on the
geometry the collapse would actually produce. That is what lets the pass loop
land on the target itself rather than handing the endgame to a second
decimator.

While it collapses, the run logs every accepted edge — the pair, the placed
position, and the triangle count after each pass. The next run on the same mesh
replays that log instead of reprocessing: any target between the logged floor
and the source count is a read-and-replay, which is what makes iterating on a
target cheap. **Cache Depth** sets how far below the requested target the log
reaches; ask outside it and the run preprocesses again. The cache lives in
Blender's memory only, keyed by the mesh's content and the options that steer the
collapse order, and holds at most four meshes. Restarting Blender clears it.

## Measured

On an RTX 4090 under Vulkan, Blender 5.2.2, from a 13,040,640-triangle scan.
The first decimation is the preprocess: it collapses to the cache floor and
records the log. Every later target between the floor and the source count is
a replay instead.

| | Time |
| --- | --- |
| Preprocess, 13,040,640 to 2,500 triangles | ~4.6 s |
| Any target above the floor, replayed (5k, 100k, 1M, 2M) | 0.9 to 1.5 s |
| Blender's Decimate modifier to 5,000, for contrast | ~94 s |

Accuracy is measured as two-sided point-to-surface distance against the source,
compared with Blender's Decimate at an equal triangle count, over three generated
fixtures at three reduction ratios. Below 1.00 is better than Blender.

| | mean error | worst case |
| --- | --- | --- |
| Hyper Decimate, ratio to Blender's Decimate | 1.01× | 1.08× |

Triangle *shape* is a separate axis from surface error, and a thin sliver can score
perfectly on the second while ruining the shading on the first. On the 13M-triangle
head scan reduced to 2M, against Blender's Decimate at the same count:

| | ours | Blender |
| --- | --- | --- |
| mean shape quality | 0.706 | 0.764 |
| triangles with shape quality below 0.3 | 6.01% | 1.61% |
| triangles with a corner under 10° | 4.39% | 0.92% |

Reproduce with `tools/shape_quality.py`. The setting behind it,
`Options.optimal_placement`, is on by default: each survivor lands on its pair
quadric's optimum as the collapse commits.

Reproduce with `tools/quality.py`; run the test suites with `tools/run_tests.sh`.

## Limitations

- Metal is untested. There is no Apple hardware to measure on, and one kernel
  relies on integer image atomics that Metal may reject.
- A GPU is required; there is no fallback. The operator reports why when the
  device cannot run the shaders or fit the mesh.
- Below about 2% of the original triangle count, detail falls off sharply. Check
  the result before relying on it.
- UV seams are constrained per vertex rather than per wedge, which can
  over-constrain two separate seams that pass close together. The failure mode is
  a slightly stiffer seam, not a torn texture.

## Licence

GPL-3.0-or-later.
