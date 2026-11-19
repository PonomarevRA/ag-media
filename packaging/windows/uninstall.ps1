# Remove only the exact current-user installation carrying this package's marker. Preserve all workspace data.
[CmdletBinding(SupportsShouldProcess=$true)]
param()
$ErrorActionPreference = 'Stop'
$target = [IO.Path]::GetFullPath($PSScriptRoot).TrimEnd('\')
$base = [IO.Path]::GetFullPath((Join-Path $env:LOCALAPPDATA 'Programs\AG Media Studio')).TrimEnd('\')
$markerPath = Join-Path $target '.ag-media-install.json'
if (-not (Test-Path -LiteralPath $markerPath)) { throw 'Not a marked AG Media Studio installation.' }
$marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
$expected = [IO.Path]::GetFullPath((Join-Path $base ([string]$marker.version))).TrimEnd('\')
if ($marker.appId -ne 'ag-media-studio' -or $target -ne $expected -or $marker.root -ne $target) { throw 'Installation marker or location does not match.' }
# Do not stop unrelated Python processes or remove a running application.
foreach ($process in Get-Process AgMediaStudio -ErrorAction SilentlyContinue) {
    if ($process.Path -eq (Join-Path $target 'AgMediaStudio.exe')) { throw 'Close this AG Media Studio instance before uninstalling.' }
}
if ($PSCmdlet.ShouldProcess($target, 'Remove this application version; preserve workspace and models')) {
    $shortcutPath = Join-Path ([Environment]::GetFolderPath('Programs')) 'AG Media Studio.lnk'
    if (Test-Path -LiteralPath $shortcutPath) {
        $shell = New-Object -ComObject WScript.Shell
        if ($shell.CreateShortcut($shortcutPath).TargetPath -eq (Join-Path $target 'AgMediaStudio.exe')) { Remove-Item -LiteralPath $shortcutPath }
    }
    Remove-Item -LiteralPath $target -Recurse -Force
    Write-Output 'Application removed. Workspace, projects, models and WebView2 profile were preserved.'
}
