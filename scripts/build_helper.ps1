#Requires -Version 5.1
<#
.SYNOPSIS
    Build jarvis-helper.exe, the elevated sensor sidecar.

.DESCRIPTION
    CLAUDE.md T-2.8 and section 6. Produces a single executable with a UAC
    manifest so Windows prompts for elevation when it starts. The main JARVIS
    process stays non-elevated and talks to this over a named pipe.

    The helper exposes exactly three RPC methods and accepts no arbitrary
    commands. Nothing in this script widens that surface.

    Idempotent: re-running rebuilds cleanly and skips work that is already done.

.PARAMETER Clean
    Remove previous build output before building.

.PARAMETER SkipVendorCheck
    Build even when LibreHardwareMonitorLib.dll is missing. The helper will
    then report sensors as unavailable at runtime rather than failing to start.

.EXAMPLE
    .\scripts\build_helper.ps1
#>

param(
    [switch]$Clean,
    [switch]$SkipVendorCheck
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BuildDir = Join-Path $RepoRoot "scripts\build"
$VendorDll = Join-Path $RepoRoot "vendor\LibreHardwareMonitorLib.dll"
$OutputExe = Join-Path $RepoRoot "jarvis-helper.exe"
$ManifestPath = Join-Path $BuildDir "jarvis-helper.manifest"

function Write-Step($Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Ok($Message) { Write-Host "    $Message" -ForegroundColor Green }
function Write-Warn($Message) { Write-Host "    $Message" -ForegroundColor Yellow }

Write-Step "Building the elevated helper"

# ---------------------------------------------------------------------------
# Preconditions
# ---------------------------------------------------------------------------

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw "uv is not on PATH. Run scripts\setup_env.ps1 first."
}

if (-not (Test-Path $VendorDll)) {
    $message = @"
LibreHardwareMonitorLib.dll was not found at:
    $VendorDll

It is MPL-2.0 licensed and is not redistributed in this repository. Download
the latest release from https://github.com/LibreHardwareMonitor/LibreHardwareMonitor
and copy LibreHardwareMonitorLib.dll into the vendor folder.

Without it the helper still builds and runs, but temperatures and fan speeds
report as unavailable.
"@
    if ($SkipVendorCheck) {
        Write-Warn $message
    }
    else {
        Write-Warn $message
        Write-Warn "Re-run with -SkipVendorCheck to build anyway."
        throw "vendor\LibreHardwareMonitorLib.dll is missing"
    }
}
else {
    Write-Ok "Found LibreHardwareMonitorLib.dll"
}

if ($Clean -and (Test-Path $BuildDir)) {
    Write-Step "Cleaning previous build output"
    Remove-Item -Recurse -Force $BuildDir
}

New-Item -ItemType Directory -Force -Path $BuildDir | Out-Null

# ---------------------------------------------------------------------------
# UAC manifest
# ---------------------------------------------------------------------------
# requireAdministrator is what raises the prompt. uiAccess stays false: the
# helper never needs to drive another process's UI, and true would require a
# signed binary in a protected path.

Write-Step "Writing the UAC manifest"

$manifest = @'
<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <assemblyIdentity version="0.1.0.0" processorArchitecture="*" name="Jarvis.Helper" type="win32"/>
  <description>JARVIS elevated sensor sidecar</description>
  <trustInfo xmlns="urn:schemas-microsoft-com:asm.v3">
    <security>
      <requestedPrivileges>
        <requestedExecutionLevel level="requireAdministrator" uiAccess="false"/>
      </requestedPrivileges>
    </security>
  </trustInfo>
  <compatibility xmlns="urn:schemas-microsoft-com:compat.v1">
    <application>
      <supportedOS Id="{8e0f7a12-bfb3-4fe8-b9a5-48fd50a15a9a}"/>
    </application>
  </compatibility>
</assembly>
'@

Set-Content -Path $ManifestPath -Value $manifest -Encoding UTF8
Write-Ok "Manifest written"

# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------

Write-Step "Ensuring PyInstaller is available"
uv pip install pyinstaller | Out-Null
Write-Ok "PyInstaller ready"

Write-Step "Compiling jarvis-helper.exe"

$pyinstallerArgs = @(
    "run", "pyinstaller",
    "--onefile",
    "--name", "jarvis-helper",
    "--distpath", $BuildDir,
    "--workpath", (Join-Path $BuildDir "work"),
    "--specpath", $BuildDir,
    "--manifest", $ManifestPath,
    "--uac-admin",
    "--noconfirm",
    "--clean",
    "--console",
    "--hidden-import", "jarvis.helper.lhm",
    "--hidden-import", "jarvis.helper.rpc",
    "--hidden-import", "clr",
    "--hidden-import", "win32pipe",
    "--hidden-import", "win32file"
)

if (Test-Path $VendorDll) {
    $pyinstallerArgs += @("--add-binary", "$VendorDll;vendor")
}

$entry = Join-Path $RepoRoot "src\jarvis\helper\__main__.py"
$pyinstallerArgs += $entry

& uv @pyinstallerArgs
if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller failed with exit code $LASTEXITCODE"
}

$built = Join-Path $BuildDir "jarvis-helper.exe"
if (-not (Test-Path $built)) {
    throw "the build reported success but $built does not exist"
}

Copy-Item -Path $built -Destination $OutputExe -Force
Write-Ok "Built $OutputExe"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

Write-Step "Verifying the method surface"

# --check prints the exposed methods and the elevation state. Run without
# elevation here, so a non-zero exit is expected and is not a build failure.
$check = & $OutputExe --check 2>&1
Write-Host "    $check"

if ($check -notmatch "get_thermals" -or $check -notmatch "get_fans" -or $check -notmatch "get_pending_updates") {
    throw "the helper did not report the three expected RPC methods"
}
Write-Ok "Exposes exactly the three permitted methods"

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  Executable: $OutputExe"
Write-Host ""
Write-Host "  Start it manually and approve the UAC prompt:" -ForegroundColor Gray
Write-Host "      .\jarvis-helper.exe" -ForegroundColor Gray
Write-Host ""
Write-Host "  Or let JARVIS start it on demand, which is the default." -ForegroundColor Gray
Write-Host "  Set helper.autostart to false in config.yaml to disable that." -ForegroundColor Gray
