import AppKit

/// Rasterize the local vector master at the requested pixel size, without network or UI state.
func renderIcon(source: String, destination: String, pixels: Int) throws {
    guard pixels > 0, pixels <= 1024, let image = NSImage(contentsOfFile: source),
          let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels, pixelsHigh: pixels,
                                       bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                       isPlanar: false, colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0),
          let context = NSGraphicsContext(bitmapImageRep: bitmap) else {
        throw NSError(domain: "AGMediaIcon", code: 1, userInfo: [NSLocalizedDescriptionKey: "Cannot render local SVG"])
    }
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = context
    image.draw(in: NSRect(x: 0, y: 0, width: pixels, height: pixels))
    NSGraphicsContext.restoreGraphicsState()
    guard let png = bitmap.representation(using: .png, properties: [:]) else {
        throw NSError(domain: "AGMediaIcon", code: 2)
    }
    try png.write(to: URL(fileURLWithPath: destination))
}

guard CommandLine.arguments.count == 4, let pixels = Int(CommandLine.arguments[3]) else {
    fatalError("Usage: render_icon source.svg destination.png pixels")
}
try renderIcon(source: CommandLine.arguments[1], destination: CommandLine.arguments[2], pixels: pixels)
