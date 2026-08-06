<#
.SYNOPSIS
    Fetch the third-party binaries in vendor/ that cannot come from PyPI.

.DESCRIPTION
    Currently one item: LibreHardwareMonitorLib.dll, which the elevated helper
    loads through pythonnet to read CPU temperatures and fan speeds.

    The DLL is committed, so a fresh clone already has it and this script is
    only needed to re-fetch a corrupted copy or to move to a new version. It is
    idempotent: a second run verifies the hash and downloads nothing. Elevation
    is never required.

    The download comes from nuget.org rather than the GitHub release, because
    the NuGet package ships the library on its own, while the release ships the
    whole GUI application with the library inside it.

.PARAMETER Version
    Package version to fetch. Defaults to the version recorded in vendor/README.md.

.PARAMETER Force
    Re-download even when the file is present and its hash already matches.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_vendor.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\fetch_vendor.ps1 -Force
#>

[CmdletBinding()]
param(
    [string] $Version = '0.9.4',
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = Split-Path -Parent $PSScriptRoot
$VendorDir = Join-Path $RepoRoot 'vendor'
$Target = Join-Path $VendorDir 'LibreHardwareMonitorLib.dll'

# Pinned so a tampered or truncated download is caught rather than loaded into
# an elevated process. Update this together with -Version and vendor/README.md.
$ExpectedSha256 = @{
    '0.9.4' = 'A0F2728F1734C236A9D02D9E25A88BC4F8CB7BD1FAFF1770726BEB7AF06BF8DC'
}

# Framework build. See vendor/README.md for why this one and not netstandard2.0.
$EntryInPackage = 'lib/net472/LibreHardwareMonitorLib.dll'

function Write-Step { param([string] $Message) Write-Host "  $Message" }

function Get-FileSha256 {
    param([string] $Path)
    return (Get-FileHash -Path $Path -Algorithm SHA256).Hash
}

Write-Host ''
Write-Host 'Vendored binaries'
Write-Host '================='

if (-not (Test-Path $VendorDir)) {
    New-Item -ItemType Directory -Path $VendorDir -Force | Out-Null
}

if (-not $ExpectedSha256.ContainsKey($Version)) {
    Write-Warning "No pinned hash for version $Version. Add one to this script and to vendor/README.md before trusting the download."
    $expected = $null
} else {
    $expected = $ExpectedSha256[$Version]
}

if ((Test-Path $Target) -and -not $Force) {
    $actual = Get-FileSha256 -Path $Target
    if ($null -ne $expected -and $actual -eq $expected) {
        Write-Step "LibreHardwareMonitorLib.dll already present and verified ($Version)"
        Write-Host ''
        exit 0
    }
    if ($null -ne $expected) {
        Write-Warning "LibreHardwareMonitorLib.dll is present but its hash does not match $Version. Re-downloading."
    } else {
        Write-Step 'LibreHardwareMonitorLib.dll already present, hash not checked'
        Write-Host ''
        exit 0
    }
}

$Url = "https://www.nuget.org/api/v2/package/LibreHardwareMonitorLib/$Version"
$Temp = Join-Path ([System.IO.Path]::GetTempPath()) "lhmlib-$Version.nupkg"

Write-Step "Downloading LibreHardwareMonitorLib $Version from nuget.org"
try {
    Invoke-WebRequest -Uri $Url -OutFile $Temp -UseBasicParsing
} catch {
    Write-Error "Download failed: $($_.Exception.Message)"
    exit 1
}

Write-Step "Extracting $EntryInPackage"
Add-Type -AssemblyName System.IO.Compression.FileSystem
$archive = [System.IO.Compression.ZipFile]::OpenRead($Temp)
try {
    $entry = $archive.Entries | Where-Object { $_.FullName -eq $EntryInPackage }
    if ($null -eq $entry) {
        Write-Error "The package does not contain $EntryInPackage. Has the layout changed?"
        exit 1
    }
    [System.IO.Compression.ZipFileExtensions]::ExtractToFile($entry, $Target, $true)
} finally {
    $archive.Dispose()
    Remove-Item $Temp -Force -ErrorAction SilentlyContinue
}

$actual = Get-FileSha256 -Path $Target
if ($null -ne $expected -and $actual -ne $expected) {
    Remove-Item $Target -Force -ErrorAction SilentlyContinue
    Write-Error "Hash mismatch for LibreHardwareMonitorLib.dll. Expected $expected, got $actual. The file has been deleted rather than left in place for an elevated process to load."
    exit 1
}

Write-Step "Wrote vendor\LibreHardwareMonitorLib.dll ($Version, verified)"
Write-Host ''
