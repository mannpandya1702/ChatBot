<#
.SYNOPSIS
    Idempotent environment setup for JARVIS on Windows 11.

.DESCRIPTION
    Checks the prerequisites, syncs the uv environment with every extra JARVIS
    needs, seeds config/config.yaml from the committed template, resolves the
    hardware tier through scripts/verify_cuda.py, and creates the runtime
    directories.

    The script is idempotent. A second run re-verifies everything, reports
    "already present" for each step, changes nothing, and exits 0.

    It never requires elevation. The main JARVIS process runs non-elevated
    (CLAUDE.md section 6) and so does its environment. Nothing is installed
    system wide behind your back: a missing system package is reported with the
    exact command to install it, and the run stops if that package is essential.

.PARAMETER SkipModels
    Do not invoke scripts/pull_models.ps1 at the end. Useful on a metered
    connection or when the models are already on disk.

.PARAMETER SkipHud
    Do not check for Rust and Node, and do not prepare the Tauri HUD. The voice
    core runs headless without them.

.PARAMETER Force
    Redo the steps that are normally skipped once their result exists: hardware
    tier detection, and package metadata refresh during uv sync. Never
    overwrites an existing config/config.yaml.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_env.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\setup_env.ps1 -SkipModels -SkipHud
#>

[CmdletBinding()]
param(
    [switch] $SkipModels,
    [switch] $SkipHud,
    [switch] $Force
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ---------------------------------------------------------------------------
# Paths. Everything is derived from the script location so the repository can
# live anywhere. No absolute user path is ever baked into the tree, per
# CLAUDE.md section 0.6.
# ---------------------------------------------------------------------------

$RepoRoot      = Split-Path -Parent $PSScriptRoot
$ScriptsDir    = $PSScriptRoot
$ConfigDir     = Join-Path $RepoRoot 'config'
$ExampleConfig = Join-Path $ConfigDir 'config.example.yaml'
$UserConfig    = Join-Path $ConfigDir 'config.yaml'
$VerifyCuda    = Join-Path $ScriptsDir 'verify_cuda.py'
$PullModels    = Join-Path $ScriptsDir 'pull_models.ps1'
$HudDir        = Join-Path $RepoRoot 'src\jarvis\ui\app'

# Directories JARVIS writes to at runtime. Mirrors PathsConfig in
# src/jarvis/config.py.
$RuntimeDirs = @(
    (Join-Path $RepoRoot 'data'),
    (Join-Path $RepoRoot 'logs'),
    (Join-Path $RepoRoot 'models'),
    (Join-Path $RepoRoot 'bench')
)

# uv extras that make up a complete JARVIS install (see pyproject.toml).
$UvSyncArgs = @(
    'sync',
    '--extra', 'audio',
    '--extra', 'stt',
    '--extra', 'tts',
    '--extra', 'gpu',
    '--extra', 'windows',
    '--extra', 'vision',
    '--group', 'dev'
)

$script:Created  = [System.Collections.Generic.List[string]]::new()
$script:Present  = [System.Collections.Generic.List[string]]::new()
$script:Missing  = [System.Collections.Generic.List[object]]::new()
$script:Notes    = [System.Collections.Generic.List[string]]::new()

# ---------------------------------------------------------------------------
# Small output helpers. Plain ASCII only, no em dashes anywhere (CLAUDE.md
# section 5).
# ---------------------------------------------------------------------------

function Write-Head {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host ''
    Write-Host ('=' * 72) -ForegroundColor DarkCyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ('=' * 72) -ForegroundColor DarkCyan
}

function Write-Step {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host ''
    Write-Host "-- $Text" -ForegroundColor White
}

function Write-Ok {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host "   [ok]      $Text" -ForegroundColor Green
}

function Write-Skip {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host "   [present] $Text" -ForegroundColor DarkGray
}

function Write-Warn {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host "   [warn]    $Text" -ForegroundColor Yellow
}

function Write-Fail {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host "   [missing] $Text" -ForegroundColor Red
}

function Write-Hint {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host "             $Text" -ForegroundColor DarkYellow
}

function Test-CommandExists {
    <#
        .SYNOPSIS
            True when a command of that name is resolvable on PATH.
    #>
    param([Parameter(Mandatory = $true)][string] $Name)
    return [bool] (Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

function Get-CommandVersion {
    <#
        .SYNOPSIS
            First line of "<exe> <versionArg>", or an empty string on failure.
    #>
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [string] $VersionArg = '--version'
    )
    try {
        $output = & $Name $VersionArg 2>$null
        if ($LASTEXITCODE -ne 0 -or $null -eq $output) { return '' }
        return (($output | Select-Object -First 1) | Out-String).Trim()
    } catch {
        return ''
    }
}

function Add-MissingPrerequisite {
    <#
        .SYNOPSIS
            Record a missing prerequisite plus the exact command that installs it.
    #>
    param(
        [Parameter(Mandatory = $true)][string] $Name,
        [Parameter(Mandatory = $true)][string] $Purpose,
        [Parameter(Mandatory = $true)][string] $InstallCommand,
        [string] $Manual = '',
        [switch] $Fatal
    )
    $script:Missing.Add([pscustomobject]@{
        Name           = $Name
        Purpose        = $Purpose
        InstallCommand = $InstallCommand
        Manual         = $Manual
        Fatal          = [bool] $Fatal
    })
    Write-Fail "$Name is not on PATH ($Purpose)"
    Write-Hint "install: $InstallCommand"
    if ($Manual) {
        Write-Hint "or see:  $Manual"
    }
}

function Invoke-Native {
    <#
        .SYNOPSIS
            Run a native command and throw when it reports a non-zero exit code.

        .DESCRIPTION
            $ErrorActionPreference does not apply to native executables on
            Windows PowerShell 5.1, so the exit code is checked by hand.
    #>
    param(
        [Parameter(Mandatory = $true)][string] $Exe,
        [string[]] $Arguments = @()
    )
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE"
    }
}

function Test-Elevated {
    <#
        .SYNOPSIS
            True when this shell holds the local Administrators role.
    #>
    try {
        $identity  = [System.Security.Principal.WindowsIdentity]::GetCurrent()
        $principal = New-Object System.Security.Principal.WindowsPrincipal($identity)
        return $principal.IsInRole([System.Security.Principal.WindowsBuiltInRole]::Administrator)
    } catch {
        return $false
    }
}

function Get-ConfiguredTier {
    <#
        .SYNOPSIS
            Read hardware.tier out of a config.yaml without a YAML parser.

        .DESCRIPTION
            Returns the tier string, or an empty string when the file, the
            section, or the key is absent. Windows PowerShell ships no YAML
            reader, and adding one would violate the "no ad hoc installs" rule,
            so the hardware block is scanned directly. The block is written by
            jarvis.config.write_hardware and is always two-space indented.
    #>
    param([Parameter(Mandatory = $true)][string] $Path)

    if (-not (Test-Path -LiteralPath $Path)) { return '' }

    $inHardware = $false
    foreach ($line in (Get-Content -LiteralPath $Path)) {
        if ($line -match '^\s*#') { continue }
        if ($line -match '^hardware:\s*$') { $inHardware = $true; continue }
        if ($inHardware -and $line -match '^\S') { break }
        if ($inHardware -and $line -match '^\s+tier:\s*["'']?([A-Za-z0-9\-]+)') {
            return $Matches[1]
        }
    }
    return ''
}

# ---------------------------------------------------------------------------
# 0. Banner and elevation posture
# ---------------------------------------------------------------------------

Write-Head 'JARVIS environment setup'
Write-Host "   repository: $RepoRoot"
Write-Host "   powershell: $($PSVersionTable.PSVersion)"

if (Test-Elevated) {
    Write-Warn 'This shell is elevated. JARVIS never needs administrator rights,'
    Write-Warn 'and a virtual environment created as administrator can be awkward'
    Write-Warn 'to use later. Consider closing this window and re-running as your'
    Write-Warn 'normal user. Only jarvis-helper.exe is ever elevated.'
    $script:Notes.Add('Setup ran elevated. Only jarvis-helper.exe needs elevation.')
} else {
    Write-Ok 'Running non-elevated, which is what JARVIS wants.'
}

# ---------------------------------------------------------------------------
# 1. Prerequisites. Report, never install system wide.
# ---------------------------------------------------------------------------

Write-Step 'Checking prerequisites'

$hasWinget = Test-CommandExists -Name 'winget'
if ($hasWinget) {
    Write-Skip 'winget (Windows Package Manager)'
} else {
    Write-Warn 'winget is not available. Install commands below need to be run by hand.'
    Write-Hint 'winget ships with App Installer from the Microsoft Store.'
    $script:Notes.Add('winget missing, prerequisites must be installed manually.')
}

# uv is the only hard requirement. Nothing else in this script can run without it.
if (Test-CommandExists -Name 'uv') {
    Write-Skip "uv $(Get-CommandVersion -Name 'uv')"
} else {
    Add-MissingPrerequisite `
        -Name 'uv' `
        -Purpose 'python toolchain and dependency manager' `
        -InstallCommand 'winget install --id=astral-sh.uv -e --source winget' `
        -Manual 'https://docs.astral.sh/uv/getting-started/installation/' `
        -Fatal
}

# Ollama serves the LLM. Setup can finish without it, model pulls cannot.
if (Test-CommandExists -Name 'ollama') {
    Write-Skip "ollama $(Get-CommandVersion -Name 'ollama')"
} else {
    Add-MissingPrerequisite `
        -Name 'ollama' `
        -Purpose 'local LLM runtime on http://localhost:11434' `
        -InstallCommand 'winget install --id=Ollama.Ollama -e --source winget' `
        -Manual 'https://ollama.com/download/windows'
}

if (-not $SkipHud) {
    if (Test-CommandExists -Name 'cargo') {
        Write-Skip "rust $(Get-CommandVersion -Name 'cargo')"
    } else {
        Add-MissingPrerequisite `
            -Name 'rust' `
            -Purpose 'optional, builds the Tauri HUD' `
            -InstallCommand 'winget install --id=Rustlang.Rustup -e --source winget' `
            -Manual 'https://www.rust-lang.org/tools/install'
    }

    if (Test-CommandExists -Name 'node') {
        Write-Skip "node $(Get-CommandVersion -Name 'node')"
    } else {
        Add-MissingPrerequisite `
            -Name 'node' `
            -Purpose 'optional, builds the HUD front end' `
            -InstallCommand 'winget install --id=OpenJS.NodeJS.LTS -e --source winget' `
            -Manual 'https://nodejs.org/en/download'
    }
} else {
    Write-Skip 'HUD toolchain checks skipped (-SkipHud)'
}

$fatal = @($script:Missing | Where-Object { $_.Fatal })
if ($fatal.Count -gt 0) {
    Write-Host ''
    Write-Host 'Cannot continue. Install these, then re-run:' -ForegroundColor Red
    foreach ($item in $fatal) {
        Write-Host "  $($item.Name): $($item.InstallCommand)" -ForegroundColor Yellow
    }
    Write-Host ''
    exit 1
}

# Python 3.12 is provisioned by uv into its own user-local store, so it is
# reported rather than installed system wide. .python-version pins the value and
# uv sync fetches it automatically when absent.
Write-Step 'Checking the Python 3.12 toolchain'
$pythonList = ''
try {
    $pythonList = (& uv python list --only-installed 2>$null | Out-String)
} catch {
    $pythonList = ''
}
if ($pythonList -match '3\.12\.') {
    Write-Skip 'CPython 3.12 already provisioned for uv'
} else {
    Write-Warn 'uv has no CPython 3.12 yet. It will be fetched by uv sync below.'
    Write-Hint 'to pre-fetch it yourself: uv python install 3.12'
}

# ---------------------------------------------------------------------------
# 2. Dependencies
# ---------------------------------------------------------------------------

Write-Step 'Syncing dependencies with uv'
Push-Location $RepoRoot
try {
    $syncArgs = @($UvSyncArgs)
    if ($Force) {
        $syncArgs += '--refresh'
        Write-Host '   -Force given, refreshing cached package metadata.' -ForegroundColor DarkGray
    }
    Write-Host "   uv $($syncArgs -join ' ')" -ForegroundColor DarkGray
    Invoke-Native -Exe 'uv' -Arguments $syncArgs
    Write-Ok 'Environment in sync (.venv).'
    $script:Present.Add('.venv with extras audio stt tts gpu windows vision, plus group dev')
} finally {
    Pop-Location
}

# ---------------------------------------------------------------------------
# 3. Configuration. Seed once, never clobber.
# ---------------------------------------------------------------------------

Write-Step 'Preparing config/config.yaml'
if (-not (Test-Path -LiteralPath $ExampleConfig)) {
    throw "Template missing: $ExampleConfig. The repository checkout is incomplete."
}
if (Test-Path -LiteralPath $UserConfig) {
    Write-Skip 'config/config.yaml already exists, left untouched.'
    if ($Force) {
        Write-Hint '-Force never overwrites config.yaml. Delete it by hand to reseed.'
    }
} else {
    Copy-Item -LiteralPath $ExampleConfig -Destination $UserConfig
    Write-Ok 'Copied config.example.yaml to config.yaml (gitignored).'
    $script:Created.Add('config/config.yaml')
}

# ---------------------------------------------------------------------------
# 4. Runtime directories
# ---------------------------------------------------------------------------

Write-Step 'Creating runtime directories'
foreach ($dir in $RuntimeDirs) {
    $name = Split-Path -Leaf $dir
    if (Test-Path -LiteralPath $dir) {
        Write-Skip "$name/"
    } else {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Ok "created $name/"
        $script:Created.Add("$name/")
    }
}

# ---------------------------------------------------------------------------
# 5. Hardware tier. Persisted into config.yaml by verify_cuda.py.
# ---------------------------------------------------------------------------

Write-Step 'Resolving the hardware tier'
$tier = Get-ConfiguredTier -Path $UserConfig
if (-not (Test-Path -LiteralPath $VerifyCuda)) {
    Write-Warn "scripts/verify_cuda.py is missing, tier detection skipped."
    $script:Notes.Add('Hardware tier unresolved: scripts/verify_cuda.py not found.')
} elseif ($tier -and $tier -ne 'auto' -and -not $Force) {
    Write-Skip "hardware.tier is already '$tier' (re-run with -Force to redetect)."
} else {
    Push-Location $RepoRoot
    try {
        & uv run python 'scripts/verify_cuda.py'
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "verify_cuda.py exited with code $LASTEXITCODE."
            $script:Notes.Add('Tier detection failed. Set hardware.tier in config.yaml by hand.')
        }
    } finally {
        Pop-Location
    }
    $tier = Get-ConfiguredTier -Path $UserConfig
    if ($tier -and $tier -ne 'auto') {
        Write-Ok "hardware.tier resolved to '$tier'."
    } else {
        Write-Warn 'hardware.tier is still auto. It will be resolved again at first run.'
    }
}

# ---------------------------------------------------------------------------
# 6. Models
# ---------------------------------------------------------------------------

if ($SkipModels) {
    Write-Step 'Model downloads skipped (-SkipModels)'
    $script:Notes.Add('Models not downloaded. Run scripts\pull_models.ps1 before first use.')
} elseif (-not (Test-Path -LiteralPath $PullModels)) {
    Write-Step 'Model downloads'
    Write-Warn 'scripts/pull_models.ps1 is missing, nothing downloaded.'
} else {
    Write-Step 'Downloading models via scripts/pull_models.ps1'
    if ($Force) {
        & $PullModels -Force
    } else {
        & $PullModels
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Warn "pull_models.ps1 exited with code $LASTEXITCODE."
        $script:Notes.Add('Model download incomplete. Re-run scripts\pull_models.ps1.')
    }
}

# ---------------------------------------------------------------------------
# 7. HUD front end. Optional, never fatal.
# ---------------------------------------------------------------------------

if (-not $SkipHud) {
    Write-Step 'Preparing the HUD front end'
    $packageJson = Join-Path $HudDir 'package.json'
    $nodeModules = Join-Path $HudDir 'node_modules'
    if (-not (Test-Path -LiteralPath $packageJson)) {
        Write-Skip 'HUD project not scaffolded yet (Phase 3), nothing to do.'
    } elseif (-not (Test-CommandExists -Name 'npm')) {
        Write-Warn 'npm is not on PATH, HUD dependencies not installed.'
    } elseif (Test-Path -LiteralPath $nodeModules) {
        Write-Skip 'HUD node_modules already installed.'
    } else {
        Push-Location $HudDir
        try {
            Invoke-Native -Exe 'npm' -Arguments @('install')
            Write-Ok 'HUD dependencies installed.'
            $script:Created.Add('src/jarvis/ui/app/node_modules')
        } finally {
            Pop-Location
        }
    }
}

# ---------------------------------------------------------------------------
# 8. Summary
# ---------------------------------------------------------------------------

Write-Head 'Summary'

if ($script:Created.Count -eq 0) {
    Write-Host '  Nothing changed. The environment was already set up.' -ForegroundColor Green
} else {
    Write-Host '  Created:' -ForegroundColor Green
    foreach ($item in $script:Created) { Write-Host "    + $item" }
}

if ($script:Present.Count -gt 0) {
    Write-Host ''
    Write-Host '  Verified:' -ForegroundColor DarkGray
    foreach ($item in $script:Present) { Write-Host "    = $item" }
}

$optional = @($script:Missing | Where-Object { -not $_.Fatal })
if ($optional.Count -gt 0) {
    Write-Host ''
    Write-Host '  Missing optional components:' -ForegroundColor Yellow
    foreach ($item in $optional) {
        Write-Host "    - $($item.Name) ($($item.Purpose))"
        Write-Hint "install: $($item.InstallCommand)"
    }
}

if ($script:Notes.Count -gt 0) {
    Write-Host ''
    Write-Host '  Notes:' -ForegroundColor Yellow
    foreach ($note in $script:Notes) { Write-Host "    * $note" }
}

$tier = Get-ConfiguredTier -Path $UserConfig
if (-not $tier) { $tier = 'auto (unresolved)' }

Write-Host ''
Write-Host '  Next steps:' -ForegroundColor Cyan
Write-Host "    1. Review config\config.yaml. Hardware tier is '$tier'."
Write-Host '    2. Start the LLM runtime if it is not already running: ollama serve'
Write-Host '    3. Models: powershell -File scripts\pull_models.ps1 [-WithVision]'
Write-Host '    4. Verify the toolchain: uv run pytest -q'
Write-Host '    5. Talk to it: uv run python -m jarvis'
Write-Host ''
Write-Host '  Setup is idempotent. Re-run it any time.' -ForegroundColor DarkGray
Write-Host ''

exit 0
