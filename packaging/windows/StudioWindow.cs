using System.Diagnostics;
using System.Drawing;
using System.Net.Http;
using System.Text.Json;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.WinForms;

namespace AgMediaStudio;

/// <summary>Real Windows system chrome, native menu and toolbar around the owned local application.</summary>
internal sealed class StudioWindow : Form
{
    private readonly OwnedBackend backend = new();
    private readonly WebView2 web = new() { Dock = DockStyle.Fill };
    private readonly ToolStripStatusLabel status = new("Starting local workspace…");
    private bool closing;
    private bool quitRequested;
    private bool startupRequested;
    private bool checkingUpdates;
    private readonly NotifyIcon tray = new() { Text = "AG Media Studio", Icon = SystemIcons.Application };
    private readonly CancellationTokenSource lifetime = new();
    private Task startup = Task.CompletedTask;

    /// <summary>Create native controls with standard keyboard access and no HTML window-control replicas.</summary>
    public StudioWindow()
    {
        Text = "AG Media Studio";
        Icon = Icon.ExtractAssociatedIcon(Application.ExecutablePath);
        tray.Icon = Icon ?? SystemIcons.Application;
        Width = 1440; Height = 960; MinimumSize = new Size(900, 600);
        StartPosition = FormStartPosition.CenterScreen;
        var menu = new MenuStrip();
        var file = new ToolStripMenuItem("&File");
        file.DropDownItems.Add("Open &workspace folder", null, (_, _) => OpenWorkspace());
        file.DropDownItems.Add("&Quit", null, (_, _) => RequestQuit());
        var view = new ToolStripMenuItem("&View");
        view.DropDownItems.Add("&Reload", null, (_, _) => Reload());
        view.DropDownItems.Add("Actual &size", null, (_, _) => web.ZoomFactor = 1.0);
        var help = new ToolStripMenuItem("&Help");
        help.DropDownItems.Add("Studio &Documentation", null, (_, _) => OpenDocumentation());
        help.DropDownItems.Add("Check for &Updates…", null, async (_, _) => await CheckForUpdatesAsync());
        menu.Items.Add(file); menu.Items.Add(view); menu.Items.Add(help);
        var toolbar = new ToolStrip { GripStyle = ToolStripGripStyle.Hidden };
        toolbar.Items.Add(new ToolStripLabel("AG Media Studio"));
        toolbar.Items.Add(new ToolStripSeparator());
        toolbar.Items.Add(new ToolStripButton("Home", null, (_, _) => NavigateHome()) { ToolTipText = "Open the workspace home" });
        toolbar.Items.Add(new ToolStripButton("Reload", null, (_, _) => Reload()) { ToolTipText = "Reload the local workspace" });
        toolbar.Items.Add(new ToolStripButton("Workspace", null, (_, _) => OpenWorkspace()) { ToolTipText = "Open the durable workspace folder" });
        var footer = new StatusStrip(); footer.Items.Add(status);
        Controls.Add(web); Controls.Add(footer); Controls.Add(toolbar); Controls.Add(menu);
        MainMenuStrip = menu;
        var trayMenu = new ContextMenuStrip();
        trayMenu.Items.Add("Show Studio", null, (_, _) => ShowStudio());
        trayMenu.Items.Add("Quit", null, (_, _) => RequestQuit());
        tray.ContextMenuStrip = trayMenu;
        tray.DoubleClick += (_, _) => ShowStudio();
        tray.Visible = true;
        Shown += (_, _) =>
        {
            if (startupRequested) return;
            startupRequested = true;
            startup = StartStudioAsync();
        };
        FormClosing += OnWindowClosing;
        FormClosed += (_, _) => { closing = true; lifetime.Cancel(); tray.Dispose(); trayMenu.Dispose(); Icon?.Dispose(); };
    }

    /// <summary>Start only the packaged backend; report a clear prerequisite failure if WebView2 is absent.</summary>
    private async Task StartStudioAsync()
    {
        try
        {
            await backend.StartAsync(lifetime.Token);
            lifetime.Token.ThrowIfCancellationRequested();
            string profile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "AG Media Studio", "WebView2");
            var environment = await CoreWebView2Environment.CreateAsync(null, profile).WaitAsync(lifetime.Token);
            await web.EnsureCoreWebView2Async(environment).WaitAsync(lifetime.Token);
            lifetime.Token.ThrowIfCancellationRequested();
            web.CoreWebView2.Settings.AreHostObjectsAllowed = false;
            web.CoreWebView2.Settings.IsWebMessageEnabled = false;
            web.CoreWebView2.Settings.AreDevToolsEnabled = false;
            web.CoreWebView2.NavigationStarting += (_, navigation) => navigation.Cancel = !IsOwnedUri(navigation.Uri);
            web.CoreWebView2.FrameNavigationStarting += (_, navigation) => navigation.Cancel = !IsOwnedUri(navigation.Uri);
            web.CoreWebView2.NewWindowRequested += (_, request) => request.Handled = true;
            web.CoreWebView2.PermissionRequested += OnPermissionRequested;
            web.CoreWebView2.ProcessFailed += (_, _) => status.Text = "WebView2 stopped. Close and reopen the application.";
            NavigateHome();
            status.Text = "Local workspace · models and optional engines are installed separately";
        }
        catch (Exception error)
        {
            if (closing) return;
            status.Text = "Startup failed";
            MessageBox.Show(this, error.Message + "\n\nThe Microsoft Edge WebView2 Evergreen Runtime is required. See WINDOWS-DISTRIBUTION.md in the package.", "AG Media Studio", MessageBoxButtons.OK, MessageBoxIcon.Error);
            RequestQuit();
        }
    }

    /// <summary>Retain the same WebView and backend on user close; OS shutdown and explicit Quit still exit.</summary>
    private void OnWindowClosing(object? sender, FormClosingEventArgs request)
    {
        if (request.CloseReason == CloseReason.UserClosing && !quitRequested)
        {
            request.Cancel = true;
            Hide();
        }
    }

    /// <summary>Restore the retained document from the tray without restarting its service or navigating home.</summary>
    private void ShowStudio()
    {
        if (closing || quitRequested) return;
        Show();
        if (WindowState == FormWindowState.Minimized) WindowState = FormWindowState.Normal;
        Activate();
    }

    /// <summary>Keep deliberate termination distinct from hiding, using the existing bounded cleanup path.</summary>
    private void RequestQuit()
    {
        quitRequested = true;
        Close();
    }

    /// <summary>Permit navigation only to the exact validated loopback server, including its effective port.</summary>
    private bool IsOwnedUri(string value)
    {
        return OriginPolicy.IsOwned(backend.Origin, value);
    }

    /// <summary>Ask the user before microphone capture from this owned workspace; deny every other origin and capability.</summary>
    private void OnPermissionRequested(object? sender, CoreWebView2PermissionRequestedEventArgs request)
    {
        request.State = CoreWebView2PermissionState.Deny;
        request.SavesInProfile = false;
        if (closing || !OriginPolicy.MayAskMicrophone(backend.Origin, request.Uri,
                request.PermissionKind == CoreWebView2PermissionKind.Microphone))
            return;
        var deferral = request.GetDeferral();
        // Post the native prompt after the WebView2 event returns; modal UI inside
        // its synchronous callback would create an unsupported nested message loop.
        BeginInvoke(() =>
        {
            try
            {
                if (closing || !IsOwnedUri(request.Uri)) return;
                DialogResult consent = MessageBox.Show(this,
                    "Allow this local AG Media Studio workspace to use the microphone? Audio recording starts only through the workspace controls.",
                    "Microphone access", MessageBoxButtons.YesNo, MessageBoxIcon.Question, MessageBoxDefaultButton.Button2);
                if (!closing && consent == DialogResult.Yes)
                    request.State = CoreWebView2PermissionState.Allow;
            }
            finally { deferral.Complete(); }
        });
    }

    private void NavigateHome() { if (backend.Origin is not null && web.CoreWebView2 is not null) web.CoreWebView2.Navigate(backend.Origin + "?desktop=windows"); }
    private void Reload() { if (web.CoreWebView2 is not null) web.Reload(); }

    /// <summary>Open the bundled Wiki beside the retained WebView document.</summary>
    private void OpenDocumentation()
    {
        if (backend.Origin is { } origin) OpenInBrowser(new Uri(origin, "/wiki").AbsoluteUri);
    }

    /// <summary>Request a bounded update check without blocking window rendering or media playback.</summary>
    private async Task CheckForUpdatesAsync()
    {
        if (checkingUpdates || closing || backend.Origin is not { } origin) return;
        checkingUpdates = true;
        string message = "The update check could not reach this Studio.";
        Uri? releases = null;
        try
        {
            using var deadline = CancellationTokenSource.CreateLinkedTokenSource(lifetime.Token);
            deadline.CancelAfter(TimeSpan.FromSeconds(8));
            using var client = new HttpClient(new HttpClientHandler { UseProxy = false, AllowAutoRedirect = false });
            using var request = new HttpRequestMessage(HttpMethod.Post, new Uri(origin, "/api/desktop-update"));
            request.Headers.Add("Origin", origin.GetLeftPart(UriPartial.Authority));
            request.Headers.Add("X-Voice-Studio", "1");
            using var response = await client.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, deadline.Token);
            if (response.IsSuccessStatusCode)
            {
                using var body = await response.Content.ReadAsStreamAsync(deadline.Token);
                byte[] buffer = new byte[16_385];
                int length = await body.ReadAtLeastAsync(buffer, buffer.Length, false, deadline.Token);
                if (length < buffer.Length)
                {
                    using var result = JsonDocument.Parse(buffer.AsMemory(0, length));
                    JsonElement root = result.RootElement;
                    if (root.TryGetProperty("message", out var text) && text.ValueKind == JsonValueKind.String)
                        message = text.GetString() ?? message;
                    if (root.TryGetProperty("releasesUrl", out var link) && link.ValueKind == JsonValueKind.String
                        && Uri.TryCreate(link.GetString(), UriKind.Absolute, out var url)
                        && url.Scheme == Uri.UriSchemeHttps && url.Host == "github.com")
                        releases = url;
                }
            }
        }
        catch (Exception error) when (error is HttpRequestException or IOException or OperationCanceledException or JsonException or InvalidOperationException) { }
        finally { checkingUpdates = false; }
        if (closing || !Visible) return;
        const string title = "AG Media Studio Updates";
        if (releases is null)
            MessageBox.Show(this, message, title, MessageBoxButtons.OK, MessageBoxIcon.Information);
        else if (MessageBox.Show(this, message + "\n\nOpen GitHub Releases?", title, MessageBoxButtons.YesNo,
                                 MessageBoxIcon.Information, MessageBoxDefaultButton.Button2) == DialogResult.Yes)
            OpenInBrowser(releases.AbsoluteUri);
    }

    /// <summary>Hand a trusted documentation or release URL to the user's browser.</summary>
    private static void OpenInBrowser(string uri)
    {
        Process.Start(new ProcessStartInfo(uri) { UseShellExecute = true });
    }

    /// <summary>Open one fixed local directory from a native user action, never a web-supplied command.</summary>
    private void OpenWorkspace()
    {
        Directory.CreateDirectory(backend.DataDirectory);
        Process.Start(new ProcessStartInfo("explorer.exe") { UseShellExecute = false, ArgumentList = { backend.DataDirectory } });
    }

    /// <summary>After the native window closes, cancel startup and finish owned process cleanup.</summary>
    public async Task ShutdownAsync()
    {
        closing = true;
        lifetime.Cancel();
        await startup;
        await backend.DisposeAsync();
        lifetime.Dispose();
    }
}
