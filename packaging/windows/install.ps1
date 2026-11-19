# Install this exact verified package for the current user; never overwrite an existing version.
[CmdletBinding()]
param([switch]$NoShortcut)
$ErrorActionPreference = 'Stop'
$package = $PSScriptRoot
$manifest = Get-Content -LiteralPath (Join-Path $package 'package-manifest.json') -Raw | ConvertFrom-Json
if ($manifest.platform -ne 'windows' -or $manifest.architecture -ne 'x86_64' -or $manifest.appId -ne 'ag-media-studio') { throw 'Unexpected package identity.' }
if ([string]$manifest.version -notmatch '^\d+\.\d+\.\d+$') { throw 'Invalid package version.' }
if (-not [Environment]::Is64BitOperatingSystem) { throw 'Windows x64 is required.' }
# Every copied byte is checked before an installation directory or shortcut is created.
foreach ($entry in $manifest.files) {
    $relative = [string]$entry.path
    if ([IO.Path]::IsPathRooted($relative) -or $relative -match '(^|[\\/])\.\.([\\/]|$)') { throw 'Invalid manifest path.' }
    $path = Join-Path $package $relative
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant() -ne $entry.sha256) { throw "Integrity check failed: $relative" }
}
$base = Join-Path $env:LOCALAPPDATA 'Programs\AG Media Studio'
$target = Join-Path $base ([string]$manifest.version)
if (Test-Path -LiteralPath $target) { throw "This version already exists: $target. Use its uninstaller before reinstalling." }
$staging = Join-Path $base ('.install-' + [Guid]::NewGuid().ToString('N'))
try {
    New-Item -ItemType Directory -Path $staging -Force | Out-Null
    foreach ($entry in $manifest.files) {
        $destination = Join-Path $staging ([string]$entry.path)
        New-Item -ItemType Directory -Path (Split-Path -Parent $destination) -Force | Out-Null
        Copy-Item -LiteralPath (Join-Path $package ([string]$entry.path)) -Destination $destination
    }
    Copy-Item -LiteralPath (Join-Path $package 'package-manifest.json') -Destination $staging
    @{ appId='ag-media-studio'; version=$manifest.version; root=$target } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $staging '.ag-media-install.json') -Encoding UTF8
    Move-Item -LiteralPath $staging -Destination $target
} catch {
    if (Test-Path -LiteralPath $staging) { Remove-Item -LiteralPath $staging -Recurse -Force }
    throw
}
if (-not $NoShortcut) {
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut((Join-Path ([Environment]::GetFolderPath('Programs')) 'AG Media Studio.lnk'))
    $shortcut.TargetPath = Join-Path $target 'AgMediaStudio.exe'
    $shortcut.WorkingDirectory = $target
    $shortcut.Description = 'AG Media Studio — local media workspace'
    $shortcut.Save()
}
Write-Output "Installed $target. Requires Microsoft Edge WebView2 Evergreen Runtime. Workspace remains outside the installation."
