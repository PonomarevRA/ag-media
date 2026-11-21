# Frame change regions

Enable **Frame change regions** in **Settings → Mods and models** and run scene
analysis. The existing FFmpeg pass keeps its cut/timestamp branch and sends a
320 × 320 color raster to a separate algorithm module. The worker compares
adjacent decoded frames, thresholds the maximum channel difference, and groups
active 16 × 16 tiles. Bounds come from the changed pixels in each group.
Hard cuts and broad changes (at least 32% of pixels) are not local proposals.
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

Use **Change regions** next to the monitor's Technical mode switch. Dashed
outlines show only during measured source frames. The current state also applies
to MP4 export with markup; choosing a graph witness can produce an annotated
still via MCP. No new weights are downloaded.

The pilot uses [absolute difference](https://docs.opencv.org/4.x/d2/de8/group__core__array.html)
and [connected components](https://docs.opencv.org/4.x/d3/dc0/group__imgproc__shape.html).
Pixel differences can see an appearance change without displacement. Global
camera compensation, [Lucas–Kanade](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)
and [codec motion vectors](https://ffmpeg.org/doxygen/trunk/extract__mvs_8c_source.html)
are separate candidates for displacement estimates. Use this layer as a bounded
source of candidate regions for later OCR or tracking work.
