import Foundation

/// Launch only this bundle's interpreter and service, with a private control pipe.
final class RuntimeProcess {
    private let process = Process()
    private let input = Pipe()
    private let output = Pipe()
    private let monitor = DispatchQueue(label: "ag-media-studio.process-owner")
    private var sampler: DispatchSourceTimer?
    private var ownership: OwnedProcesses?
    private var buffer = Data()
    private var ready = false
    private var stopping = false
    var onReady: (([String: Any]) -> Void)?
    var onFailure: ((String) -> Void)?
    var serverPID: Int32 { process.processIdentifier }

    /// Resolve immutable bundle resources; caller arguments are passed literally to Python.
    func start(data: URL, port: Int, demo: Bool) throws {
        guard let resources = Bundle.main.resourceURL else { throw RuntimeError.missingResources }
        let studio = resources.appendingPathComponent("studio", isDirectory: true)
        let python = resources.appendingPathComponent("runtime/bin/python3")
        guard FileManager.default.isExecutableFile(atPath: python.path) else { throw RuntimeError.missingResources }
        process.executableURL = python
        let bootstrap = "import runpy,sys; sys.path.insert(0,sys.argv.pop(1)); runpy.run_module('voice_studio.app.desktop_runtime',run_name='__main__')"
        process.arguments = ["-I", "-B", "-c", bootstrap, studio.path, "--data", data.path, "--port", String(port)] + (demo ? ["--demo"] : [])
        process.currentDirectoryURL = data
        var environment = ProcessInfo.processInfo.environment
        for key in ["PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PYTHONSTARTUP"] { environment.removeValue(forKey: key) }
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONNOUSERSITE"] = "1"
        environment["PATH"] = "/usr/local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin"
        environment["OMP_NUM_THREADS"] = environment["OMP_NUM_THREADS"] ?? "4"
        environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        process.environment = environment
        process.standardInput = input
        process.standardOutput = output
        // Preserve actionable startup errors without exposing them to a web document.
        let log = data.appendingPathComponent("desktop.log")
        FileManager.default.createFile(atPath: log.path, contents: nil)
        process.standardError = try FileHandle(forWritingTo: log)
        output.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let bytes = handle.availableData
            DispatchQueue.main.async { self?.receive(bytes) }
        }
        process.terminationHandler = { [weak self] child in
            DispatchQueue.main.async {
                guard let self = self, !self.stopping else { return }
                self.onFailure?("The owned Studio service stopped (status \(child.terminationStatus)). See desktop.log in the data folder.")
            }
        }
        try process.run()
        ownership = OwnedProcesses(pid: process.processIdentifier)
        let timer = DispatchSource.makeTimerSource(queue: monitor)
        timer.schedule(deadline: .now(), repeating: .milliseconds(200))
        timer.setEventHandler { [weak self] in self?.ownership?.sample() }
        sampler = timer
        timer.resume()
        DispatchQueue.main.asyncAfter(deadline: .now() + 30) { [weak self] in
            guard let self = self, !self.ready, !self.stopping else { return }
            self.onFailure?("Studio startup exceeded 30 seconds. See desktop.log in the data folder.")
        }
    }

    /// Consume only the bound service's small readiness envelope, never a URL from web content.
    private func receive(_ bytes: Data) {
        guard !stopping, !ready, !bytes.isEmpty else { return }
        buffer.append(bytes)
        guard buffer.count < 65536 else { onFailure?("Invalid Studio startup response."); return }
        while let newline = buffer.firstIndex(of: 10) {
            let line = buffer[..<newline]
            buffer.removeSubrange(...newline)
            guard let value = try? JSONSerialization.jsonObject(with: line) as? [String: Any],
                  value["schema"] as? String == "ag-media-desktop-ready-v1",
                  value["pid"] as? Int32 == serverPID,
                  let port = value["port"] as? Int, port > 0, port < 65536,
                  value["origin"] as? String == "http://127.0.0.1:\(port)" else { continue }
            ready = true
            onReady?(value)
        }
    }

    /// Drain off the UI thread, retaining exact child identities until the hard deadline passes.
    func stop(completion: @escaping () -> Void) {
        guard !stopping else { return }
        stopping = true
        output.fileHandleForReading.readabilityHandler = nil
        monitor.async { [self] in
            sampler?.cancel()
            ownership?.sample()
            try? input.fileHandleForWriting.write(contentsOf: Data("quit\n".utf8))
            ownership?.terminate()
            // A terminateLater event wait can run inside an AppKit callback. Schedule
            // through the native run loop rather than waiting for a reentrant GCD drain.
            RunLoop.main.perform(inModes: [.common], block: completion)
        }
    }
}

enum RuntimeError: Error {
    case missingResources
}
