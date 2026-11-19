# Native distributions

The desktop hosts start their own loopback service on an owned socket. They never
attach to the development service on port 8766. macOS uses real AppKit menus,
window controls and an NSToolbar around WKWebView. Linux uses GTK3 native window
controls and WebKit2GTK 4.1. Windows has a separate WinForms/WebView2 host; see
[WINDOWS-DISTRIBUTION.md](WINDOWS-DISTRIBUTION.md).

## What is included

The macOS app includes a relocatable CPython interpreter, installed core packages,
application source, web assets and the wiki. It runs from an unrelated working
directory without a development checkout, virtual environment or another AG app.
The Linux `.deb` vendors the core Python packages and declares its required
system Python 3.12+, GTK3, Python GI, WebKit2GTK 4.1 and xdg-utils dependencies.
Linux packages target the build runner's architecture and Python ABI; the package
dependency constrains the system interpreter to that minor version.

No model weights, recordings, caches, FFmpeg binaries or private engine environments
are bundled. The current Homebrew FFmpeg build is not redistributed. Optional
Whisper, synthesis, OCR and other model providers require separately provisioned,
compatible runtimes and weights. The native application opens without downloads.
To prepare the Whisper Base weights from a source checkout, run
`python -m voice_studio.app.first_run --models-only`. The same external model
directory is used by the source and desktop applications on the same platform.
Runtime readiness and the packaged `desktop-product.json` record this boundary.
An available file or command does not establish model quality or successful inference.
The browser control remains available for browser capabilities absent in a native web view.

## Install from a package

- **macOS.** Download the DMG for your processor and its adjacent `.sha256` file.
  Run `shasum -a 256 -c NAME.dmg.sha256` in the download directory using the
  actual filename. Open the DMG, drag the app to Applications and open it from
  Finder. Current local builds are ad-hoc signed. Use the release's signing and
  notarization metadata to decide whether your system policy permits the build.
- **Linux.** On a supported Debian system, check the `.deb` with
  `sha256sum -c NAME.deb.sha256`. Install with `sudo apt install ./NAME.deb`
  using the actual filename. Launch AG Media Studio from the app menu or run
  `ag-media-studio` from a terminal in the desktop session.
- **Windows.** Verify the x64 ZIP, install the WebView2 runtime and follow the
  [PowerShell install steps](WINDOWS-DISTRIBUTION.md#install-and-run).

In the opened app, **Help → Studio Documentation** gives the workflow and model
requirements. To test media analysis, use a short local clip and provision the
engine it requires.

## External state and launch

macOS stores the default workspace in `~/Library/Application Support/Voice Studio/studio`.
Linux uses `$XDG_DATA_HOME/voice-studio/studio` or `~/.local/share/voice-studio/studio`.
On these platforms, set `VOICE_STUDIO_DATA` or pass `--data` to select another
workspace. Windows defaults to `%LOCALAPPDATA%\AG Media Studio\workspace`.
`VOICE_MODELS_DIR`, `VOICE_DEPENDENCIES_DIR` and `VOICE_TEMP_DIR` configure the
separate engine and scratch directories.

For an isolated macOS launch:

```sh
'/path with spaces/ag-media-studio.app/Contents/MacOS/AgMediaStudio' \
  --data /absolute/temporary/native-data --port 0
```

The host reserves its own service port, writes private
`<workspace>/desktop-instance.json`. macOS and Linux also persist presentation
in `desktop-window.json`. Native hosts request an ephemeral port by default.
On macOS and Linux `VOICE_DESKTOP_STATUS_FILE` can record readiness for a QA runner.
Closing the native window now hides it, retaining its WebView document, UI MCP
Bridge session and the owned background service. On macOS restore it from the Dock
or **View → Show Studio** (⌘0); **Window → Close Window** (⌘W) hides it. On Windows
use the tray icon's **Show Studio** action or double-click the icon. On Linux launch
the same workspace again: GtkApplication activation presents the existing window
without starting another backend (a desktop D-Bus session is required).

Choose **Quit** in the native menu/tray to exit fully; macOS also supports ⌘Q.
Explicit Quit and OS termination still drain off the
UI thread with a bounded deadline and signal only their recorded owned process
identities, including workers with separate process groups. Unrelated servers are
never stopped. Abrupt OS/process crashes can bypass orderly cleanup.

On macOS14+ WKWebView uses the public `inactiveSchedulingPolicy = .none` so hiding
does not suspend agent requests. The retained document is not reloaded on restore.
While closed to the background, a balanced user-initiated ProcessInfo activity
also prevents App Nap of the host; it explicitly allows idle system/display
sleep and ends on restore or completed Quit. No timer or fake media is involved.
On macOS13 the window remains attached to its WebView; extended hidden-page
responsiveness still needs qualification on that OS. Windows does not call
WebView2 `TrySuspend`; Linux keeps its GTK application held while hidden. No
synthetic media, private WebKit API or periodic keep-awake loop is used.

## MCP

Start the native GUI before using HTTP-backed tools. On macOS or Linux, run its
executable with `--print-config --data /absolute/workspace`. On Windows, run the
[bundled Python command](WINDOWS-DISTRIBUTION.md#mcp). The helper validates the
workspace's live PID, exact loopback origin and instance nonce before printing
the stdio configuration. The client uses explicit workspace, graph and output
paths. Reconnect after the app restarts if the port has changed.

## Native Help and update checks

Use **Help → Studio Documentation** to open the bundled Wiki in the system
browser. The active Studio document remains open. On GTK, Help is inside the
native Studio menu.

**Help → Check for Updates** compares the installed version with the latest
public stable GitHub release. Set `releaseRepository` to the intended
`owner/repository` in `packaging/desktop-product.json` before building an
edition. The operator can override it with `VOICE_RELEASE_REPOSITORY` when
launching the application. With no repository configured the dialog reports the
installed version. A private GitHub repository requires a public release
channel for this check to see its metadata.

The check runs when selected in Help. A newer version is marked available when
its platform package and corresponding `.sha256` asset are listed. The dialog
can open the configured GitHub Releases page. Download and install a package
manually after verifying its checksum. The check itself only reads release metadata.

## Build

Use an empty output directory. Builds never replace an installed application.

```sh
# macOS: the base interpreter must be a relocatable standalone CPython build.
.venv/bin/python packaging/build_native.py \
  --python .venv/bin/python \
  --site-packages .venv/lib/python3.12/site-packages \
  --output /absolute/output/macos --work /absolute/temporary/build

# Debian 13/trixie Linux with Python 3.12+ and pip/dpkg-deb installed.
python3 packaging/build_linux.py --output /absolute/output/linux
```

macOS produces `.app`, a drag-install `.dmg`, SHA-256 and release metadata. It is
ad-hoc signed and **not notarized**. Linux produces an unsigned architecture-specific
`.deb` plus checksum/release metadata. The manifest's signing, bundled packages,
external dependencies and validation claims must describe the actual artifact.
The CI workflow builds platform packages; a successful build does not qualify a
GUI runtime on another operating system.

## Validation boundary

The focused tests exercise process-level workspace exclusion, occupied-port
fallback, live instance identity, rejected nonce tampering, external MCP paths and
metadata expiry. Native qualification separately checks actual OS controls,
keyboard navigation, window/sidebar restoration, bundled Python worker execution
and cleanup of an owned worker in a separate session. A synthetic test tone is
not a successful recognition or synthesis test. Keep platform runtime results
and unperformed checks explicit in release reports.
