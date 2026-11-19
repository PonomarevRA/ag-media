"""Regenerate committed web/native icons from one SVG master on macOS; builds elsewhere use these assets."""

from pathlib import Path
import struct
import subprocess
import tempfile


def build_icons(root):
    """Render exact raster sizes, then assemble standard PNG-backed ICO and ten-size ICNS containers."""
    brand = root / "web/brand"
    native = root / "packaging/icons"
    native.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="ag-media-icons-") as folder:
        work = Path(folder)
        renderer = work / "render-icon"
        subprocess.run(["swiftc", "-module-cache-path", str(work / "swift-cache"),
                        str(root / "packaging/render_icon.swift"), "-o", str(renderer)], check=True)
        rasters = {}
        for pixels in (16, 32, 48, 64, 128, 180, 256, 512, 1024):
            raster = work / f"{pixels}.png"
            subprocess.run([str(renderer), str(brand / "studio.svg"), str(raster), str(pixels)], check=True)
            rasters[pixels] = raster.read_bytes()
        for pixels in (32, 180, 512):
            (brand / f"icon-{pixels}.png").write_bytes(rasters[pixels])
        iconset = work / "AgMediaStudio.iconset"
        iconset.mkdir()
        for size in (16, 32, 128, 256, 512):
            (iconset / f"icon_{size}x{size}.png").write_bytes(rasters[size])
            (iconset / f"icon_{size}x{size}@2x.png").write_bytes(rasters[size * 2])
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(native / "AgMediaStudio.icns")], check=True)
        sizes = (16, 32, 48, 256)
        offset = 6 + 16 * len(sizes)
        entries = []
        for size in sizes:
            entries.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(rasters[size]), offset))
            offset += len(rasters[size])
        ico = struct.pack("<HHH", 0, 1, len(sizes)) + b"".join(entries)
        for size in sizes:
            ico += rasters[size]
        (native / "AgMediaStudio.ico").write_bytes(ico)
        (brand / "favicon.ico").write_bytes(ico)


if __name__ == "__main__":
    build_icons(Path(__file__).resolve().parents[1])
