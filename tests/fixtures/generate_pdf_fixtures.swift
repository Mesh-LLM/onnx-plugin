// Original fixtures for mesh-onnx acceptance tests. Run on macOS from this directory:
// swift generate_pdf_fixtures.swift
import Foundation
import CoreGraphics
import CoreText
import ImageIO
import PDFKit
import UniformTypeIdentifiers

func text(_ context: CGContext, _ value: String, _ x: CGFloat, _ y: CGFloat,
          _ size: CGFloat = 14, _ font: String = "Helvetica") {
    let string = NSAttributedString(string: value, attributes: [
        NSAttributedString.Key(kCTFontAttributeName as String): CTFontCreateWithName(font as CFString, size, nil),
        NSAttributedString.Key(kCTForegroundColorAttributeName as String): CGColor(gray: 0, alpha: 1)
    ])
    context.textPosition = CGPoint(x: x, y: y)
    CTLineDraw(CTLineCreateWithAttributedString(string), context)
}
func raster(_ width: Int, _ height: Int, _ draw: (CGContext) -> Void) -> CGImage {
    let context = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8,
        bytesPerRow: width, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: 0)!
    context.setFillColor(gray: 1, alpha: 1)
    context.fill(CGRect(x: 0, y: 0, width: width, height: height))
    draw(context)
    return context.makeImage()!
}
func pdf(_ file: String, _ draw: (CGContext) -> Void) {
    var bounds = CGRect(x: 0, y: 0, width: 612, height: 792)
    let context = CGContext(URL(fileURLWithPath: file) as CFURL, mediaBox: &bounds, nil)!
    context.beginPDFPage(nil)
    draw(context)
    context.endPDFPage()
    context.closePDF()
}
let words = raster(800, 140) { context in
    text(context, "IMAGE OCR WORDS", 24, 48, 64)
}
pdf("hybrid.pdf") { context in
    text(context, "Left column first line", 40, 730)
    text(context, "Left column second line", 40, 700)
    text(context, "Right column first line", 330, 730)
    text(context, "Right column second line", 330, 700)
    context.draw(words, in: CGRect(x: 40, y: 630, width: 260, height: 46))
}
pdf("text-with-image.pdf") { context in
    text(context, "A real PDF text layer with an embedded blank image.", 40, 730)
    let blank = raster(100, 100) { _ in }
    context.draw(blank, in: CGRect(x: 40, y: 500, width: 100, height: 100))
}
pdf("text-segments.pdf") { context in
    text(context, "Styled", 40, 730)
    text(context, " words", 80, 730, 14, "Helvetica-Bold")
    text(context, "Next line stays separate", 40, 700)
    text(context, "Other column stays separate", 330, 730)
}
let protected = PDFDocument(url: URL(fileURLWithPath: "text-with-image.pdf"))!
precondition(protected.write(to: URL(fileURLWithPath: "encrypted.pdf"), withOptions: [
    PDFDocumentWriteOption.userPasswordOption: "fixture-password",
    PDFDocumentWriteOption.ownerPasswordOption: "fixture-owner-password"
]))
let math = raster(600, 100) { context in
    text(context, "x", 20, 22, 64, "Times-Italic")
    text(context, "2", 50, 55, 36, "Times-Roman")
    text(context, "+", 94, 22, 64, "Times-Roman")
    text(context, "y", 160, 22, 64, "Times-Italic")
    text(context, "2", 192, 55, 36, "Times-Roman")
    text(context, "=", 240, 22, 64, "Times-Roman")
    text(context, "z", 306, 22, 64, "Times-Italic")
    text(context, "2", 337, 55, 36, "Times-Roman")
}
let destination = CGImageDestinationCreateWithURL(URL(fileURLWithPath: "math.png") as CFURL,
    UTType.png.identifier as CFString, 1, nil)!
CGImageDestinationAddImage(destination, math, nil)
precondition(CGImageDestinationFinalize(destination))
