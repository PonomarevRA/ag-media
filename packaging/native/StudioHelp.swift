import AppKit

/// Provide native Help actions while preserving the active Studio document.
final class StudioHelp: NSObject {
    private let origin: URL
    private weak var window: NSWindow?
    private var checking = false

    init(origin: URL, window: NSWindow) {
        self.origin = origin
        self.window = window
    }

    /// Keep the current WebView and its analysis mounted.
    @objc func openDocumentation() {
        NSWorkspace.shared.open(origin.appendingPathComponent("wiki"))
    }

    /// Check through the owned local service, away from the AppKit event loop.
    @objc func checkForUpdates() {
        guard !checking else { return }
        checking = true
        let url = origin.appendingPathComponent("api").appendingPathComponent("desktop-update")
        var request = URLRequest(url: url, timeoutInterval: 7)
        request.httpMethod = "POST"
        request.setValue(origin.absoluteString.trimmingCharacters(in: CharacterSet(charactersIn: "/")), forHTTPHeaderField: "Origin")
        request.setValue("1", forHTTPHeaderField: "X-Voice-Studio")
        let config = URLSessionConfiguration.ephemeral
        config.connectionProxyDictionary = [:]
        config.timeoutIntervalForResource = 8
        let session = URLSession(configuration: config, delegate: NoUpdateRedirect(), delegateQueue: nil)
        session.dataTask(with: request) { [weak self] data, response, _ in
            let success = (response as? HTTPURLResponse)?.statusCode == 200
            let result = success && (data?.count ?? 0) <= 16_384
                ? data.flatMap { (try? JSONSerialization.jsonObject(with: $0)) as? [String: Any] } : nil
            DispatchQueue.main.async {
                self?.showResult(result)
            }
        }.resume()
        session.finishTasksAndInvalidate()
    }

    /// Offer a browser link after the user has seen the result.
    private func showResult(_ result: [String: Any]?) {
        guard let window = window, window.isVisible else {
            checking = false
            return
        }
        let alert = NSAlert()
        alert.messageText = "AG Media Studio Updates"
        alert.informativeText = result?["message"] as? String ?? "The update check could not reach this Studio."
        alert.addButton(withTitle: "OK")
        let releases = (result?["releasesUrl"] as? String).flatMap { URL(string: $0) }
        if releases?.scheme == "https" && releases?.host == "github.com" {
            alert.addButton(withTitle: "Open Releases")
        }
        alert.beginSheetModal(for: window) { [weak self] response in
            self?.checking = false
            if response == .alertSecondButtonReturn, let releases = releases {
                NSWorkspace.shared.open(releases)
            }
        }
    }
}

/// Keep checks on this owned instance if the response proposes another destination.
private final class NoUpdateRedirect: NSObject, URLSessionTaskDelegate {
    /// Decline redirects without issuing a second network request.
    func urlSession(_ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse,
                    newRequest request: URLRequest, completionHandler: @escaping (URLRequest?) -> Void) {
        completionHandler(nil)
    }
}
