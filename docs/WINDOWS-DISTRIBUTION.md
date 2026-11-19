# Windows distribution

The Windows x64 ZIP contains an actual native WinForms application with the normal Windows frame, a native menu and toolbar, WebView2 SDK, self-contained .NET runtime, CPython and the core Python packages. It launches its own loopback server and does not require a developer checkout, a separately installed Python, or a separately installed .NET runtime.

This package is unsigned. It is an installable ZIP with PowerShell scripts, not an MSI or MSIX installer. The initial release was cross-built on macOS; actual Windows GUI startup, WebView2 rendering, Job Object behaviour, and installation/uninstallation require the Windows runner/manual acceptance described below. A successful cross-build alone does not verify these behaviours.

## Install and run

1. Use Windows 11 x64, with the [Microsoft Edge WebView2 Evergreen Runtime](https://developer.microsoft.com/en-us/microsoft-edge/webview2/) installed. The runtime is an explicit external prerequisite and is not included in this ZIP. Windows 10 requires a supported Microsoft servicing arrangement and the same prerequisite; it has not been qualified here.
2. Verify the ZIP against its adjacent `.sha256` file, then extract the ZIP. Its internal `package-manifest.json` records every installed payload file's SHA-256 and size. These hashes detect corruption; the unsigned archive does not establish a publisher identity.
3. Review `install.ps1`, then run it in PowerShell: `powershell.exe -NoProfile -File .\install.ps1`. Execution policies may require an administrator-approved policy or an explicitly allowed local script. No policy is changed by the installer.
4. The script verifies all payload hashes before installing to `%LOCALAPPDATA%\Programs\AG Media Studio\<version>` and creates a current-user Start menu shortcut. It refuses to overwrite an existing version. To skip the shortcut, add `-NoShortcut`.
5. Start **AG Media Studio** from the Start menu, or run `AgMediaStudio.exe` from the extracted folder for a portable launch.

The default durable workspace is `%LOCALAPPDATA%\AG Media Studio\workspace`; WebView2's profile is `%LOCALAPPDATA%\AG Media Studio\WebView2`. Both are outside the installation. Models are not bundled, and the base application does not automatically download them. FFmpeg, ffprobe, and optional model engines are operator-provisioned capabilities; unavailable functions report missing dependencies. This is a core application package, not a bundle of every optional media/AI engine.

To remove one installed version, choose **Quit** in the app and run its `uninstall.ps1` (use `-WhatIf` to inspect the target). The script checks its installation marker, removes only that version and its matching shortcut, and preserves projects, workspace, model files and the browser profile.

## Native process and navigation boundaries

The host resolves all source/runtime paths relative to its executable, removes inherited Python module-path overrides and uses Python isolation. A private stdin gate prevents backend execution until the child is assigned to a Windows Job Object configured with `KILL_ON_JOB_CLOSE`. The readiness record must match the spawned PID, workspace, loopback address and nonce returned by `/api/desktop-instance` before WebView2 navigates. Busy ports are not attached to.

Navigation and child-frame navigation are limited to the validated exact loopback origin. New windows are denied. Web messages and host objects are disabled: page text cannot invoke arbitrary native commands. Microphone requests from the exact owned origin require an explicit native Yes/No consent dialog, defaulting to No and not saving a grant in the profile; other origins and permission kinds, including camera, are denied. The native Workspace button opens the external data directory. Closing the window hides it and keeps the Studio and backend running. To restore it, use **Show Studio** in the tray. Choose **Quit** to stop the owned backend and close its Job Object.

## MCP

Open the desktop app. From the directory containing `AgMediaStudio.exe`, run in PowerShell:

```powershell
$workspace = Join-Path $env:LOCALAPPDATA 'AG Media Studio\workspace'
.\runtime\python\python.exe -I -B -m voice_studio.app.desktop_config --print-config --data $workspace
```

Use the emitted stdio configuration for the client. The bundled interpreter
finds the packaged module through its isolated Python paths. The helper validates
the running desktop's PID, origin and nonce before returning its address.

## Reproduce the package

Use .NET SDK 10.0.102 and Python 3.13 with the build-time `packaging` module. In a dedicated build directory:

```powershell
dotnet publish packaging/windows/AGMediaStudio.csproj -c Release -r win-x64 --self-contained true -o C:\build\publish
python -m pip download --dest C:\build\wheels --platform win_amd64 --implementation cp --python-version 3.13 --only-binary=:all: -r requirements-core.txt
python -m pip download --dest C:\build\wheels --platform win_amd64 --implementation cp --python-version 3.13 --only-binary=:all: pywin32==312
python packaging/build_windows.py --source . --publish C:\build\publish --python-zip C:\build\python-3.13.11-embed-amd64.zip --wheels C:\build\wheels --nuget C:\build\nuget --output C:\build\release
```

Set `NUGET_PACKAGES=C:\build\nuget` before publishing so licenses are available at the explicit cache path. Download the [official CPython 3.13.11 embeddable x64 archive](https://www.python.org/ftp/python/3.13.11/python-3.13.11-embed-amd64.zip), SHA-256 `1ec066fb61ba5e8c73e29e048cd07c26850f74585e3a116005135b31b8004890`. The builder rejects another archive. It verifies Windows dependency markers because cross-platform pip resolution can omit Windows-only requirements such as pywin32. `python-wheels.json` pins all downloaded transitive versions and hashes for this artifact; reuse those exact wheels when reproducing it. Do not include optional models or private data in build inputs.

The output includes the versioned ZIP, adjacent SHA-256, `windows-release.json`, per-file `package-manifest.json`, Python wheel metadata/licenses, CPython's license, Microsoft's runtime/WebView2 redistribution licenses and project third-party notices. Project-owned source is MIT; bundled dependencies retain their original licenses. Runtime package versions here are .NET 10.0.2, WebView2 SDK 1.0.4314.35 and CPython 3.13.11. These are exact build inputs, not claims about latest upstream releases. The builder reads actual .NET/WebView2 versions from the published metadata. Deterministic compiler PathMap and embedded debug metadata normalize project paths to `/_/ag-media-studio/windows`; the package builder also rejects current build-account names and home paths in all staged payload files before emitting a ZIP.

## Windows acceptance

On a Windows runner, run `AgMediaStudio.exe --backend-smoke <fresh-external-directory>` and require exit code 0 plus `windows-smoke.json`. Confirm the matching loopback service exits after the command returns and no child workers remain. In an interactive Windows session, verify native frame/menu/toolbar, workspace loading, resize/focus, external-origin rejection, optional-engine diagnostics, close during startup, close while a worker is active, two launches against one workspace, and the install/uninstall scripts with workspace preservation. Keep this evidence separate from cross-build evidence. `windowsRuntimeVerified: false` remains truthful until that OS verification is recorded.

Official platform references: [WebView2 distribution](https://learn.microsoft.com/en-us/microsoft-edge/webview2/concepts/distribution), [WebView2 WinForms](https://learn.microsoft.com/en-us/microsoft-edge/webview2/get-started/winforms), [CPython embeddable distribution](https://docs.python.org/3.13/using/windows.html#the-embeddable-package), [self-contained .NET deployment](https://learn.microsoft.com/en-us/dotnet/core/deploying/).
