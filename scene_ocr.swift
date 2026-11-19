import Foundation
import Vision

struct TextLine: Encodable {
    let text: String
    let score: Double
    let box: [Double]
}

// One Vision request is reused for a batch of full frames and sector crops.
func recognize(_ paths: [String]) throws -> [[TextLine]?] {
    let request = VNRecognizeTextRequest()
    request.recognitionLevel = .accurate
    request.usesLanguageCorrection = false
    let supported = try request.supportedRecognitionLanguages()
    request.recognitionLanguages = ["ru-RU", "en-US"].filter(supported.contains)
    return try paths.map { path in
        let image = VNImageRequestHandler(url: URL(fileURLWithPath: path))
        do {
            try image.perform([request])
        } catch {
            return nil
        }
        return (request.results ?? []).compactMap { observation in
            guard let candidate = observation.topCandidates(1).first else { return nil }
            let box = observation.boundingBox
            return TextLine(text: candidate.string, score: Double(candidate.confidence),
                            box: [box.minX, 1 - box.maxY, box.width, box.height])
        }
    }
}

do {
    let paths = try JSONDecoder().decode([String].self, from: FileHandle.standardInput.readDataToEndOfFile())
    let results = try recognize(paths)
    FileHandle.standardOutput.write(try JSONEncoder().encode(results))
} catch {
    FileHandle.standardError.write(Data("Vision OCR unavailable: \(error)\n".utf8))
    exit(1)
}
