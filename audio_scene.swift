import CoreMedia
import Foundation
import AVFoundation
import SoundAnalysis

/// Emit the classifier's alternatives on their original analysis clock.
final class Observer: NSObject, SNResultsObserving {
    private var rows: [[String: Any]] = []
    private var failure: Error?
    private let lock = NSLock()
    let completion = DispatchSemaphore(value: 0)

    func request(_ request: SNRequest, didProduce result: SNResult) {
        guard let result = result as? SNClassificationResult else { return }
        let scores = result.classifications.filter { $0.confidence >= 0.15 }.map {
            ["label": $0.identifier, "score": $0.confidence] as [String: Any]
        }
        if scores.isEmpty { return }
        lock.withLock {
            rows.append(["start": CMTimeGetSeconds(result.timeRange.start),
                         "end": CMTimeGetSeconds(CMTimeRangeGetEnd(result.timeRange)), "scores": scores])
        }
    }

    func request(_ request: SNRequest, didFailWithError error: Error) {
        lock.withLock { failure = error }
        completion.signal()
    }

    func requestDidComplete(_ request: SNRequest) { completion.signal() }

    func snapshot() throws -> [[String: Any]] {
        try lock.withLock {
            if let failure { throw failure }
            return rows
        }
    }
}

/// Feed consecutive blocks into one analyzer without resetting its window clock.
func analyzeStream(_ url: URL, chunkMs: Int, request: SNClassifySoundRequest, observer: Observer) throws {
    let audio = try AVAudioFile(forReading: url)
    let format = audio.processingFormat
    let block = AVAudioFrameCount((format.sampleRate * Double(chunkMs) / 1000).rounded())
    let analyzer = SNAudioStreamAnalyzer(format: format)
    try analyzer.add(request, withObserver: observer)
    while audio.framePosition < audio.length {
        let position = audio.framePosition
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: block) else {
            throw CocoaError(.fileReadCorruptFile)
        }
        try audio.read(into: buffer)
        if buffer.frameLength == 0 { break }
        analyzer.analyze(buffer, atAudioFramePosition: position)
    }
    analyzer.completeAnalysis()
    if observer.completion.wait(timeout: .now() + 30) == .timedOut {
        throw NSError(domain: "VoiceStudio.SoundAnalysis", code: 1)
    }
}

do {
    let arguments = CommandLine.arguments
    guard arguments.count == 2 || (arguments.count == 4 && arguments[1] == "--stream") else { exit(2) }
    let request = try SNClassifySoundRequest(classifierIdentifier: .version1)
    if arguments[1] == "--labels" {
        let data = try JSONSerialization.data(withJSONObject: request.knownClassifications)
        FileHandle.standardOutput.write(data)
    } else {
        request.overlapFactor = 0.5
        request.windowDuration = CMTime(seconds: 1.5, preferredTimescale: 1000)
        let observer = Observer()
        if arguments[1] == "--stream" {
            guard let chunk = Int(arguments[3]), (100...6000).contains(chunk) else { exit(2) }
            try analyzeStream(URL(fileURLWithPath: arguments[2]), chunkMs: chunk, request: request, observer: observer)
        } else {
            let analyzer = try SNAudioFileAnalyzer(url: URL(fileURLWithPath: arguments[1]))
            try analyzer.add(request, withObserver: observer)
            analyzer.analyze()
        }
        let data = try JSONSerialization.data(withJSONObject: observer.snapshot())
        FileHandle.standardOutput.write(data)
    }
} catch {
    FileHandle.standardError.write(Data("\(error)\n".utf8))
    exit(1)
}
