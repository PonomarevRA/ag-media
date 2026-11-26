# Frame change regions

Enable **Frame change regions** in **Settings → Mods and models** and run scene
analysis. The existing FFmpeg pass keeps its cut/timestamp branch and sends a
320 × 320 color raster to a separate algorithm module. The worker compares
adjacent decoded frames, thresholds the maximum channel difference, and groups
active 16 × 16 tiles. Bounds come from the changed pixels in each group.
Hard cuts are excluded from local proposals. The fixed-threshold pass also
skips broad changes covering at least 32% of pixels.
This is a candidate signal; panning, lighting, compression and UI animation can
still produce real pixel changes unrelated to an object.

| Mode | What runs |
| --- | --- |
| `measure` | Save up to 8 changed frames per second, with up to 4 regions per frame. |
| `objects` | Also select up to 4 candidate frames and 2 extra crops per frame, using the full scene decode with its existing 1920 px cap. |
| `assist` | Use motion distribution and sustained split-screen geometry to propose UI crops. Add up to 2 compact object checks per second outside a verified panel. |

The mod is off by default. The 8-bit channel threshold defaults to 24 and can
be set from 8 to 96. Scene analysis still keeps the normal full-frame schedule
and OCR cadence. Crop results need a strong detector score and overlap with the
changed area. Observations are tagged `tile-difference` and associated by the
current object tracker. Consecutive crop observations up to 160 ms apart can be linked despite a
fast displacement when measured foreground and edges agree and the match is
unique in both directions. Motion alone does not merge IDs. If scanning
fails, basic scene analysis can continue with the optional layer marked failed.

In `assist`, a proposed split must pass layout/OCR or scroll verification before
it becomes a tracked panel. A repeated view can reference the earlier OCR frame
through `readingTimeMs` and `reading_from_frame`. A changed panel is eligible for
another read or the existing scroll mosaic workflow. The scene keeps its regular
detector schedule. Supplemental observations can extend a baseline track through
a unique, measured foreground and edge match across up to 160 ms in one shot.

Two optional settings extend the fixed-threshold pass:

- **Adaptive change threshold** checks otherwise empty frame pairs at a bounded
  lower threshold. It can correct small frame-wide exposure shifts before looking
  for local changes.
- **Camera motion compensation** aligns the preceding frame when widely spread
  Lucas–Kanade tracks support the motion. It recovers residual regions from pairs
  with no usable fixed-threshold region. Pure vertical shifts keep their
  original interpretation because they can be UI scrolling.

Fallback results use free slots in the existing measurement and crop budgets.
Their graph nodes have `data.supplemental: true`. Baseline object checks and
the frames used to verify baseline OCR crops retain their original schedule.
The `compensatedFrames` counter counts accepted alignments before cut filtering.

On a fresh short-source analysis, the monitor and MCP can receive the change
map before OCR and object detection finish. MCP reports partial scene coverage
until recognition completes. A saved complete scene is reused as one result.
Use **Change regions** next to the monitor's Technical mode switch. Dashed
outlines show only during measured source frames. The current state also applies
to MP4 export with markup; choosing a graph witness can produce an annotated
still via MCP. No new weights are downloaded.

The pilot uses [absolute difference](https://docs.opencv.org/4.x/d2/de8/group__core__array.html)
and [connected components](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).
Pixel differences can see an appearance change without displacement. Use this
layer as a bounded source of candidates for later OCR or tracking.

## Event sampling and OCR reuse pilot

Use **Frame selection and OCR** in the same mod. **Baseline** keeps the
current frame schedule. **Onset and peak** groups nearby pixel changes within
each shot before thinning. It retains the onset and strongest measured change
when they fit the existing frame budget. A short region that returns to its
previous appearance can submit one direct crop from a visible source frame.
The result carries `evidenceBasis: transient-frame` and an interval ending at
that frame. The normal evidence check still applies to other crops.

**Onset, peak and OCR reuse** selects the same frames. The local OCR worker can
reuse a complete crop-relative result during the same scene pass if the encoded
image, dimensions and OCR options match exactly. The caller projects the boxes
onto each crop's own coordinates. This is independent of existing panel and
scroll reuse.

For comparable runs, set **Scene OCR crop budget** to the same positive number
in all compared modes. For example, 256 allows up to 256 dispatched crops per scene
pass with 64 slots available only to frame-change recognition. Backend
fallbacks may attempt a crop again. `ocrPilot.engineCalls` counts those launches
and `ocrPilot.dispatched` counts the admitted crops. `requested`, `reused`,
`dropped`, and `ocrMs` describe the same pass. The baseline setting of 0 keeps
normal recognition limits. The event modes use 256 when the setting is 0.

## Stable regions, UI states and track continuation

**Events against a stable region** compares fixed tiles against the previous
frame and against their last settled appearance. Strong local changes emit
immediately. A decaying sum requires consecutive support before emitting weak
changes. Events retain their onset and favor a sharper follow-up at similar
extent. The preview uses the existing per-second cap and reserves half the
slots for time-distributed frames. The object and panel-motion passes read
their own adjacent-frame measurements.
[NoScope](https://people.eecs.berkeley.edu/~matei/papers/2017/vldb_noscope.pdf)
and [CUSUM](https://doi.org/10.1093/biomet/41.1-2.100) informed this approach.

**Interface state OCR** selects the sharpest observed crop for each stable
panel band or changed band state. It uses at most 48 crops per scene pass and
shares the existing crop ceiling. Short visual gaps between verified panel
views can add up to 96 sampled frames. The graph stores the exact recognition
frame, crop coordinates and the extent of measured support. Equal text with
unique registered geometry can link to an earlier observation. Rapid scroll
can still lead to multiple readings and incomplete intervals.
[You Only Recognize Once](https://arxiv.org/abs/1903.03299)
informed the frame choice.

**Small interface controls** adds visually supported button and icon
candidates from at most 16 verified source frames at original resolution.
It retains normalized coordinates, original pixel coordinates and a geometry
score. A shape alone has no semantic label. [UIED](https://arxiv.org/abs/2008.05132)
informed the two-pass structure.

**Continue tracks using weak detections** supplements confirmed tracks with
nearby detections of the same class. Ambiguity, an occupied box or a cut can
prevent a continuation. A weak sample cannot establish an identity. Disputed
matches can use sparse background alignment and, within a separate cap,
reciprocal DIS flow outside known object boxes.
[ByteTrack](https://arxiv.org/abs/2110.06864),
[BoT-SORT](https://arxiv.org/abs/2206.14651) and
[DIS](https://arxiv.org/abs/1603.03590) informed this pilot.

## Local speech evidence and scene scheduling

The active-speaker stage preserves a confirmed on-screen interval inside a longer
unresolved cue when one track has at least eight consecutive positive Light-ASD
frames. A cut or competing active face ends the interval. The technical overlay
can highlight the face during that interval while the transcript keeps its
audio speaker. The graph records the measured bounds and face with the cue.

The active-speaker worker can reuse the appearance and clothing image models
across windows of the same source. Each window still supplies its own frames
and runs the same models on them.

**Run OCR and object detection in parallel** overlaps the initial OCR batch
with object processing. The scene uses the same frame selection, crop order
and OCR budget in both modes. Later interface crops, mosaics and object
refinements keep their existing schedule. Parallel OCR is enabled by default
and works with frame-change detection disabled. Set `parallelPrimaryOcr` to
`false` in the **Frame Changes** mod settings to use sequential processing.
