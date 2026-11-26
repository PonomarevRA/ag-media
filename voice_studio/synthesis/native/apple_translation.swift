import Foundation
import Translation

private struct Request: Decodable {
    let sourceLanguage: String
    let targetLanguage: String
    let texts: [String]
}

private struct Reply: Encodable {
    let status: String
    var translations: [String]? = nil
}

@main
private enum AppleTranslation {
    static func main() async throws {
        let reply: Reply
        if #available(macOS 26.0, *) {
            do {
                let request = try JSONDecoder().decode(Request.self, from: FileHandle.standardInput.readDataToEndOfFile())
                reply = try await translate(request)
            } catch TranslationError.notInstalled {
                reply = Reply(status: "needs-languages")
            } catch {
                reply = Reply(status: "failed")
            }
        } else {
            reply = Reply(status: "unavailable")
        }
        FileHandle.standardOutput.write(try JSONEncoder().encode(reply))
    }

    @available(macOS 26.0, *)
    private static func translate(_ request: Request) async throws -> Reply {
        let source = Locale.Language(identifier: request.sourceLanguage)
        let target = Locale.Language(identifier: request.targetLanguage)
        switch await LanguageAvailability().status(from: source, to: target) {
        case .unsupported: return Reply(status: "unsupported-pair")
        case .supported: return Reply(status: "needs-languages")
        case .installed: break
        @unknown default: return Reply(status: "unavailable")
        }
        if request.texts.isEmpty { return Reply(status: "ready") }

        // The direct initializer uses installed languages and cannot request downloads.
        let session = TranslationSession(installedSource: source, target: target)
        var translations: [String] = []
        for start in stride(from: 0, to: request.texts.count, by: 32) {
            let end = min(start + 32, request.texts.count)
            let batch = (start..<end).map {
                TranslationSession.Request(sourceText: request.texts[$0], clientIdentifier: String($0))
            }
            let responses = try await session.translations(from: batch)
            guard responses.count == batch.count else { return Reply(status: "failed") }
            var indexed: [Int: String] = [:]
            for response in responses {
                guard let id = response.clientIdentifier, let index = Int(id),
                      (start..<end).contains(index), indexed[index] == nil else {
                    return Reply(status: "failed")
                }
                indexed[index] = response.targetText
            }
            for index in start..<end {
                guard let text = indexed[index] else { return Reply(status: "failed") }
                translations.append(text)
            }
        }
        return Reply(status: "ready", translations: translations)
    }
}
