import AppKit
import WebKit

/// Persist only presentation preferences beside the selected external workspace.
struct WindowPreferences: Codable {
    var frame: String?
    var sidebarCollapsed = false
    // Diagnostic presentation state; launch explicitly shows the window rather than restoring hidden state.
    var isVisible: Bool?

    static func read(_ data: URL) -> WindowPreferences {
        guard let bytes = try? Data(contentsOf: data.appendingPathComponent("desktop-window.json")),
              let value = try? JSONDecoder().decode(WindowPreferences.self, from: bytes) else { return WindowPreferences() }
        return value
    }

    func write(_ data: URL) {
        if let bytes = try? JSONEncoder().encode(self) { try? bytes.write(to: data.appendingPathComponent("desktop-window.json"), options: .atomic) }
    }
}

/// A standard AppKit window owns the toolbar; WKWebView owns only the existing human UI.
final class StudioWindow: NSObject, NSWindowDelegate, NSToolbarDelegate, WKNavigationDelegate, WKUIDelegate, WKScriptMessageHandler, WKDownloadDelegate {
    let window: NSWindow
    let webview: WKWebView
    let help: StudioHelp
    private let origin: URL
    private let data: URL
    private var preferences: WindowPreferences
    private var backgroundActivity: NSObjectProtocol?
    private let sidebarID = NSToolbarItem.Identifier("ag-media.sidebar")
    private let browserID = NSToolbarItem.Identifier("ag-media.browser")

    init(origin: URL, data: URL) {
        self.origin = origin
        self.data = data
        preferences = WindowPreferences.read(data)
        let config = WKWebViewConfiguration()
        config.websiteDataStore = .default()
        config.preferences.javaScriptCanOpenWindowsAutomatically = false
        // Keep the retained page's MCP long poll responsive while the native UI is hidden.
        if #available(macOS 14.0, *) { config.preferences.inactiveSchedulingPolicy = .none }
        let collapsed = preferences.sidebarCollapsed ? "1" : "0"
        let script = "try { localStorage.setItem('voice-studio:sidebar-collapsed','\(collapsed)'); } catch {} document.addEventListener('click', event => { if (event.target.closest('#nav-collapse')) setTimeout(() => window.webkit.messageHandlers.studioPresentation.postMessage({collapsed:document.documentElement.dataset.sidebarCollapsed === 'true'}), 0); });"
        config.userContentController.addUserScript(WKUserScript(source: script, injectionTime: .atDocumentStart, forMainFrameOnly: true))
        webview = WKWebView(frame: .zero, configuration: config)
        window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 840), styleMask: [.titled, .closable, .miniaturizable, .resizable], backing: .buffered, defer: false)
        help = StudioHelp(origin: origin, window: window)
        super.init()
        config.userContentController.add(self, name: "studioPresentation")
        window.title = "ag-media-studio"
        window.minSize = NSSize(width: 390, height: 560)
        window.contentView = webview
        window.delegate = self
        window.isReleasedWhenClosed = false
        window.toolbarStyle = .unifiedCompact
        let toolbar = NSToolbar(identifier: "ag-media-studio.toolbar")
        toolbar.delegate = self
        toolbar.displayMode = .iconOnly
        toolbar.allowsUserCustomization = false
        window.toolbar = toolbar
        if let frame = preferences.frame { window.setFrame(from: frame) } else { window.center() }
        webview.navigationDelegate = self
        webview.uiDelegate = self
        webview.load(URLRequest(url: origin))
    }

    /// Bring the actual native window forward after the owned service has become ready.
    @objc func show() {
        window.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
        endBackgroundActivity()
        persist()
    }

    /// Balance the hidden agent-work activity on restore or after deliberate shutdown finishes.
    func endBackgroundActivity() {
        if let activity = backgroundActivity {
            ProcessInfo.processInfo.endActivity(activity)
            backgroundActivity = nil
        }
    }

    /// Keep the browser control available when a capability is absent from WKWebView.
    @objc func openBrowser() { NSWorkspace.shared.open(origin) }
    @objc func reload() { webview.reload() }
    @objc func openData() { NSWorkspace.shared.open(data) }

    /// Open only the operator-configured model directory, outside the application bundle.
    @objc func openModels() {
        let fallback = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Voice Studio/Models")
        let folder = ProcessInfo.processInfo.environment["VOICE_MODELS_DIR"].map { URL(fileURLWithPath: $0) } ?? fallback
        try? FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        NSWorkspace.shared.open(folder)
    }

    /// Invoke one fixed human UI action; content can never supply a selector or program.
    @objc func toggleSidebar() {
        webview.evaluateJavaScript("document.getElementById('nav-collapse')?.click()")
    }

    /// Retain only a boolean from this owned top-level document's presentation channel.
    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard message.frameInfo.isMainFrame, sameOrigin(message.frameInfo.request.url),
              let value = message.body as? [String: Any], let collapsed = value["collapsed"] as? Bool else { return }
        preferences.sidebarCollapsed = collapsed
        persist()
    }

    private func sameOrigin(_ value: URL?) -> Bool {
        guard let value = value else { return false }
        return value.scheme == origin.scheme && value.host == origin.host && value.port == origin.port
    }

    /// Preserve frame and navigation state without storing user media or request bodies.
    func persist() {
        preferences.frame = window.frameDescriptor
        preferences.isVisible = window.isVisible
        preferences.write(data)
    }

    func windowDidResize(_ notification: Notification) { persist() }
    func windowDidMove(_ notification: Notification) { persist() }
    /// Hide without closing WKWebView, its current document or the owned background service.
    func windowShouldClose(_ sender: NSWindow) -> Bool {
        if backgroundActivity == nil {
            // Preserve user-requested agent work without disabling idle system or display sleep.
            backgroundActivity = ProcessInfo.processInfo.beginActivity(
                options: .userInitiatedAllowingIdleSystemSleep, reason: "Keep the user's hidden Studio available to local agents")
        }
        sender.orderOut(nil)
        persist()
        return false
    }

    func toolbarAllowedItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] { [sidebarID, .flexibleSpace, browserID] }
    func toolbarDefaultItemIdentifiers(_ toolbar: NSToolbar) -> [NSToolbarItem.Identifier] { [sidebarID, .flexibleSpace, browserID] }

    /// Native toolbar items retain platform focus, tooltips and accessible labels.
    func toolbar(_ toolbar: NSToolbar, itemForItemIdentifier identifier: NSToolbarItem.Identifier, willBeInsertedIntoToolbar flag: Bool) -> NSToolbarItem? {
        let item = NSToolbarItem(itemIdentifier: identifier)
        if identifier == sidebarID {
            item.label = "Toggle Sidebar"
            item.toolTip = "Toggle Sidebar (⌘⇧S)"
            item.image = NSImage(systemSymbolName: "sidebar.left", accessibilityDescription: "Toggle Sidebar")
            item.target = self
            item.action = #selector(toggleSidebar)
        } else if identifier == browserID {
            item.label = "Open in Browser"
            item.toolTip = "Open this Studio in your browser"
            item.image = NSImage(systemSymbolName: "safari", accessibilityDescription: "Open in Browser")
            item.target = self
            item.action = #selector(openBrowser)
        } else { return nil }
        return item
    }

    /// Keep service navigation local; explicit external links use the system browser.
    func webView(_ webView: WKWebView, decidePolicyFor navigationAction: WKNavigationAction, decisionHandler: @escaping (WKNavigationActionPolicy) -> Void) {
        guard let url = navigationAction.request.url else { decisionHandler(.cancel); return }
        if sameOrigin(url) || url.scheme == "blob" {
            decisionHandler(navigationAction.shouldPerformDownload ? .download : .allow)
            return
        }
        if navigationAction.navigationType == .linkActivated && ["https", "http"].contains(url.scheme ?? "") { NSWorkspace.shared.open(url) }
        decisionHandler(.cancel)
    }

    /// Uploads use the real operating-system file picker and never automatic file access.
    func webView(_ webView: WKWebView, runOpenPanelWith parameters: WKOpenPanelParameters, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping ([URL]?) -> Void) {
        let picker = NSOpenPanel()
        picker.allowsMultipleSelection = parameters.allowsMultipleSelection
        picker.canChooseDirectories = parameters.allowsDirectories
        picker.beginSheetModal(for: window) { response in completionHandler(response == .OK ? picker.urls : nil) }
    }

    /// Script confirmations retain a native, visible decision surface.
    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String, initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = "ag-media-studio"
        alert.informativeText = String(message.prefix(2000))
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        alert.beginSheetModal(for: window) { completionHandler($0 == .alertFirstButtonReturn) }
    }

    /// Download only after the web view has accepted a same-origin or generated blob action.
    func webView(_ webView: WKWebView, navigationAction: WKNavigationAction, didBecome download: WKDownload) {
        download.delegate = self
    }

    /// Every download requires an explicit native save destination.
    func download(_ download: WKDownload, decideDestinationUsing response: URLResponse, suggestedFilename: String, completionHandler: @escaping (URL?) -> Void) {
        let picker = NSSavePanel()
        picker.nameFieldStringValue = URL(fileURLWithPath: suggestedFilename).lastPathComponent
        picker.beginSheetModal(for: window) { result in completionHandler(result == .OK ? picker.url : nil) }
    }

}
