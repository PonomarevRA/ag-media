# AG Media Studio icon

The original vector master is `web/brand/studio.svg`: the existing five-bar sound motif,
with a small amber recording indicator. The mark has no fonts, external references or scripts.
It is project artwork covered by the repository MIT license.

The 24-unit carrier uses rounded stroke caps, a 1.6-unit stroke and 1.4-unit gaps between
bars. Foreground bounds are x=5.2..19, y=5..18.8; the 22-unit background is an app-icon
plate, separate from the inner icon keyline. At 16 pixels the gaps are about 0.93 pixels;
review the small raster separately. Fixed dark plate and teal/amber foreground retain the
same identity in all six interface themes. This is a brand mark, not a new UI action.

On macOS, run `python3 packaging/build_icons.py` to regenerate committed PNG, ICO and ICNS
assets from that master. The Swift renderer uses AppKit, and iconutil builds the ICNS.
Other platforms build from the committed outputs, without Swift or an image dependency.
After reviewing regenerated binaries, update their exact hashes in
`tools/export_repository.py` (`REVIEWED_BRAND_ASSETS`). Only these exact brand files may
bypass the source exporter's text scanner; other binary files remain rejected.

Web favicon sizes: SVG, PNG 32, ICO 16/32/48/256, Apple touch 180. Native macOS: all ten
standard ICNS sizes through 1024; Windows executable/window/tray: ICO; Linux desktop:
hicolor SVG, window: PNG 512. System icon effects may differ from the flat web original.
