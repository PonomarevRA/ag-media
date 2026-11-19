import AppKit

/// Keep AppKit lifecycle and the one owned service instance under a single native owner.
final class StudioApplication: NSObject, NSApplicationDelegate {
    private let runtime = RuntimeProcess()
    private var studio: StudioWindow?
    private var quitting = false
    private var terminationReady = false
    private var errorAlert: NSAlert?
    private var signalSources: [DispatchSourceSignal] = []
    private let data: URL
    private let port: Int
    private let demo: Bool

    /// Retain one explicit external workspace and the owned service launch options.
    init(data: URL, port: Int, demo: Bool) {
        self.data = data
        self.port = port
        self.demo = demo
        super.init()
    }

    /// Start the owned backend and bind OS termination signals before opening a window.
    func applicationDidFinishLaunching(_ notification: Notification) {
        for number in [SIGTERM, SIGINT] {
            signal(number, SIG_IGN)
            let source = DispatchSource.makeSignalSource(signal: number, queue: .main)
            source.setEventHandler { [weak self] in
                // AppKit may wait for terminateLater replies in its own event loop.
                // Return from GCD before invoking it, so completion cannot deadlock that queue.
                RunLoop.main.perform(inModes: [.common]) { [weak self] in
                    guard let self = self, !self.quitting else { return }
                    NSApp.terminate(nil)
                }
            }
            signalSources.append(source)
            source.resume()
        }
        runtime.onReady = { [weak self] status in
            guard let self = self, !self.quitting, let address = status["origin"] as? String, let origin = URL(string: address) else { return }
            let window = StudioWindow(origin: origin, data: self.data)
            self.studio = window
            installStudioMenus(window)
            window.show()
            if let path = ProcessInfo.processInfo.environment["VOICE_DESKTOP_STATUS_FILE"] {
                var value = status
                value["hostPid"] = ProcessInfo.processInfo.processIdentifier
                value["host"] = "AppKit/WKWebView"
                if let bytes = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys]) {
                    try? bytes.write(to: URL(fileURLWithPath: path), options: .atomic)
                }
            }
        }
        runtime.onFailure = { [weak self] message in
            guard let self = self, !self.quitting, self.errorAlert == nil else { return }
            let alert = NSAlert()
            alert.messageText = "ag-media-studio could not start"
            alert.informativeText = message
            let button = alert.addButton(withTitle: "Quit")
            button.target = NSApp
            button.action = #selector(NSApplication.terminate(_:))
            self.errorAlert = alert
            alert.layout()
            alert.window.makeKeyAndOrderFront(nil)
            NSApp.activate(ignoringOtherApps: true)
        }
        do { try runtime.start(data: data, port: port, demo: demo) }
        catch { runtime.onFailure?(String(describing: error)) }
    }

    /// Approve termination asynchronously after the one bounded owned-resource drain.
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if terminationReady { return .terminateNow }
        if quitting { return .terminateLater }
        quitting = true
        studio?.persist()
        studio?.window.orderOut(nil)
        errorAlert?.window.orderOut(nil)
        runtime.stop { [weak self] in
            self?.studio?.endBackgroundActivity()
            self?.terminationReady = true
            NSApp.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }

    /// Restore the retained document on Dock activation, unless deliberate termination is already draining.
    func applicationShouldHandleReopen(_ sender: NSApplication, hasVisibleWindows flag: Bool) -> Bool {
        guard !quitting else { return false }
        studio?.show()
        return true
    }

    /// Closing a presentation window never ends the application or its agents' service.
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}

/// Parse the small literal launch contract before AppKit takes control of the main thread.
func launchOptions() throws -> (URL, Int, Bool) {
    let arguments = Array(CommandLine.arguments.dropFirst())
    let environment = ProcessInfo.processInfo.environment
    let home = FileManager.default.homeDirectoryForCurrentUser
    var data = URL(fileURLWithPath: environment["VOICE_STUDIO_DATA"] ?? home.appendingPathComponent("Library/Application Support/Voice Studio/studio").path)
    var port = 0
    var demo = false
    var index = 0
    while index < arguments.count {
        let item = arguments[index]
        if item == "--demo" { demo = true }
        else if item == "--data", index + 1 < arguments.count {
            index += 1
            data = URL(fileURLWithPath: arguments[index])
        } else if item == "--port", index + 1 < arguments.count, let value = Int(arguments[index + 1]), (0...65535).contains(value) {
            index += 1
            port = value
        } else { throw NSError(domain: "ag-media-studio", code: 2, userInfo: [NSLocalizedDescriptionKey: "Unknown or incomplete launch option: \(item)"]) }
        index += 1
    }
    try FileManager.default.createDirectory(at: data, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
    return (data.standardizedFileURL, port, demo)
}

@main
struct Main {
    static func main() {
        if CommandLine.arguments.contains("--mcp") || CommandLine.arguments.contains("--print-config") {
            runConsole()
            return
        }
        do {
            let (data, port, demo) = try launchOptions()
            let application = NSApplication.shared
            application.setActivationPolicy(.regular)
            let delegate = StudioApplication(data: data, port: port, demo: demo)
            application.delegate = delegate
            withExtendedLifetime(delegate) { application.run() }
        } catch {
            fputs("ag-media-studio: \(error.localizedDescription)\n", stderr)
            exit(2)
        }
    }

    /// Attach stdio to the explicit native-instance helper without opening AppKit.
    static func runConsole() {
        guard let resources = Bundle.main.resourceURL else { exit(2) }
        let python = resources.appendingPathComponent("runtime/bin/python3").path
        let bootstrap = "import runpy,sys;sys.path.insert(0,sys.argv.pop(1));runpy.run_module('voice_studio.app.desktop_config',run_name='__main__')"
        let arguments = [python, "-I", "-B", "-c", bootstrap, resources.appendingPathComponent("studio").path] + Array(CommandLine.arguments.dropFirst())
        var pointers = arguments.map { strdup($0) }
        pointers.append(nil)
        execv(python, &pointers)
        for pointer in pointers { free(pointer) }
        fputs("Could not run bundled MCP/config helper.\n", stderr)
        exit(2)
    }

}
