<#
.SYNOPSIS
    Download every model JARVIS needs, sized to the resolved hardware tier.

.DESCRIPTION
    Fetches, in this order:
      1. the tier-appropriate Qwen3 model through ollama,
      2. the openWakeWord hey_jarvis model plus its shared feature extractors,
      3. the Silero VAD ONNX graph,
      4. the Kokoro-82M weights, config, and the bm_george voice.

    Every item is guarded by an existence check, so the script is idempotent:
    a second run downloads nothing and reports each item as already present.

    The tier is read from config/config.yaml. When it is missing or still
    "auto", scripts/verify_cuda.py is run once to probe the machine, and the cpu
    tier is assumed if that also fails. Elevation is never required.

.PARAMETER WithVision
    Also pull the vision model. Ignored with a spoken-style explanation on tiers
    that cannot host it, which is cpu and gpu-6 (CLAUDE.md section 2).

.PARAMETER Tier
    Override the tier instead of reading config/config.yaml. One of
    cpu, gpu-6, gpu-8, gpu-12, gpu-16, gpu-24.

.PARAMETER Force
    Re-download and re-pull everything even when it is already on disk.

.PARAMETER SkipLlm
    Do not touch ollama. Useful when the LLM is already served from elsewhere.

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\pull_models.ps1

.EXAMPLE
    powershell -NoProfile -ExecutionPolicy Bypass -File scripts\pull_models.ps1 -WithVision
#>

[CmdletBinding()]
param(
    [switch] $WithVision,
    [ValidateSet('cpu', 'gpu-6', 'gpu-8', 'gpu-12', 'gpu-16', 'gpu-24')]
    [string] $Tier = '',
    [switch] $Force,
    [switch] $SkipLlm
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
# Invoke-WebRequest is an order of magnitude slower with the progress bar on.
$ProgressPreference = 'SilentlyContinue'

# Older Windows PowerShell 5.1 builds still negotiate TLS 1.0 by default, which
# GitHub and Hugging Face both refuse. Opt in to TLS 1.2 before any download.
try {
    [Net.ServicePointManager]::SecurityProtocol =
        [Net.ServicePointManager]::SecurityProtocol -bor [Net.SecurityProtocolType]::Tls12
} catch {
    Write-Verbose 'Could not raise the TLS version, continuing with the default.'
}

# ---------------------------------------------------------------------------
# Paths, all derived from the script location. No absolute user path is ever
# written into the tree (CLAUDE.md section 0.6).
# ---------------------------------------------------------------------------

$RepoRoot   = Split-Path -Parent $PSScriptRoot
$ModelsDir  = Join-Path $RepoRoot 'models'
$UserConfig = Join-Path $RepoRoot 'config\config.yaml'
$VerifyCuda = Join-Path $PSScriptRoot 'verify_cuda.py'

# ---------------------------------------------------------------------------
# Tier to LLM model. This block MUST stay in sync with TIER_PROFILES in
# src/jarvis/config.py. tests/test_setup_scripts.py parses the literal below and
# compares it against the Python source, so keep the one-pair-per-line shape.
# ---------------------------------------------------------------------------

$TierModels = @{
    'cpu'    = 'qwen3:4b'
    'gpu-6'  = 'qwen3:4b'
    'gpu-8'  = 'qwen3:8b'
    'gpu-12' = 'qwen3:8b'
    'gpu-16' = 'qwen3:14b'
    'gpu-24' = 'qwen3:32b'
}

# Tiers with roughly 6 GB of VRAM to spare for the VLM. Mirrors
# TierProfile.supports_vision, and is checked by the same test.
$VisionTiers = @('gpu-8', 'gpu-12', 'gpu-16', 'gpu-24')

# Matches ToolsConfig.vision_model in src/jarvis/config.py.
$VisionModel = 'qwen3-vl:8b'

# ---------------------------------------------------------------------------
# File downloads. Each entry lists candidate URLs in preference order; the first
# one that yields a plausible file wins. MinimumBytes rejects HTML error pages
# served with a 200 status.
# ---------------------------------------------------------------------------

$OpenWakeWordRelease = 'https://github.com/dscripka/openWakeWord/releases/download/v0.5.1'
$SileroRepo          = 'https://raw.githubusercontent.com/snakers4/silero-vad'
$KokoroRepo          = 'https://huggingface.co/hexgrad/Kokoro-82M/resolve/main'

$FileDownloads = @(
    [pscustomobject]@{
        Label        = 'openWakeWord hey_jarvis'
        Relative     = 'openwakeword\hey_jarvis_v0.1.onnx'
        Urls         = @("$OpenWakeWordRelease/hey_jarvis_v0.1.onnx")
        MinimumBytes = 4096
        Hint         = 'Fallback: openwakeword.utils.download_models() fetches the same asset.'
    },
    [pscustomobject]@{
        Label        = 'openWakeWord melspectrogram'
        Relative     = 'openwakeword\melspectrogram.onnx'
        Urls         = @("$OpenWakeWordRelease/melspectrogram.onnx")
        MinimumBytes = 4096
        Hint         = 'Shared feature extractor, required by every wake word model.'
    },
    [pscustomobject]@{
        Label        = 'openWakeWord embedding model'
        Relative     = 'openwakeword\embedding_model.onnx'
        Urls         = @("$OpenWakeWordRelease/embedding_model.onnx")
        MinimumBytes = 4096
        Hint         = 'Shared feature extractor, required by every wake word model.'
    },
    [pscustomobject]@{
        Label        = 'Silero VAD'
        Relative     = 'silero\silero_vad.onnx'
        Urls         = @(
            "$SileroRepo/master/src/silero_vad/data/silero_vad.onnx",
            "$SileroRepo/v5.1/src/silero_vad/data/silero_vad.onnx"
        )
        MinimumBytes = 4096
        Hint         = 'MIT licensed endpointing model, about 2 MB.'
    },
    [pscustomobject]@{
        Label        = 'Kokoro-82M weights'
        Relative     = 'kokoro\kokoro-v1_0.pth'
        Urls         = @("$KokoroRepo/kokoro-v1_0.pth")
        MinimumBytes = 1048576
        Hint         = 'About 330 MB. Apache-2.0.'
    },
    [pscustomobject]@{
        Label        = 'Kokoro-82M config'
        Relative     = 'kokoro\config.json'
        Urls         = @("$KokoroRepo/config.json")
        MinimumBytes = 64
        Hint         = 'Voice pack metadata.'
    },
    [pscustomobject]@{
        Label        = 'Kokoro voice bm_george'
        Relative     = 'kokoro\voices\bm_george.pt'
        Urls         = @("$KokoroRepo/voices/bm_george.pt")
        MinimumBytes = 4096
        Hint         = 'The British butler voice named in CLAUDE.md section 9.'
    }
)

$script:Downloaded = [System.Collections.Generic.List[string]]::new()
$script:Skipped    = [System.Collections.Generic.List[string]]::new()
$script:Failed     = [System.Collections.Generic.List[string]]::new()

# ---------------------------------------------------------------------------
# Output helpers. Plain ASCII only, no em dashes (CLAUDE.md section 5).
# ---------------------------------------------------------------------------

function Write-Head {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host ''
    Write-Host ('=' * 72) -ForegroundColor DarkCyan
    Write-Host "  $Text" -ForegroundColor Cyan
    Write-Host ('=' * 72) -ForegroundColor DarkCyan
}

function Write-Ok {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host "   [got]     $Text" -ForegroundColor Green
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
    Write-Host "   [failed]  $Text" -ForegroundColor Red
}

function Write-Hint {
    param([Parameter(Mandatory = $true)][string] $Text)
    Write-Host "             $Text" -ForegroundColor DarkYellow
}

function Format-Size {
    <#
        .SYNOPSIS
            Human readable byte count, for example "327.4 MB".
    #>
    param([Parameter(Mandatory = $true)][long] $Bytes)
    if ($Bytes -ge 1073741824) { return ('{0:N1} GB' -f ($Bytes / 1073741824)) }
    if ($Bytes -ge 1048576)    { return ('{0:N1} MB' -f ($Bytes / 1048576)) }
    if ($Bytes -ge 1024)       { return ('{0:N1} KB' -f ($Bytes / 1024)) }
    return "$Bytes B"
}

function Test-CommandExists {
    <#
        .SYNOPSIS
            True when a command of that name is resolvable on PATH.
    #>
    param([Parameter(Mandatory = $true)][string] $Name)
    return [bool] (Get-Command -Name $Name -ErrorAction SilentlyContinue)
}

function Get-ConfiguredTier {
    <#
        .SYNOPSIS
            Read hardware.tier out of config.yaml without a YAML parser.

        .DESCRIPTION
            Windows PowerShell ships no YAML reader and adding one would mean an
            ad hoc install, so the hardware block is scanned directly. The block
            is written by jarvis.config.write_hardware and is two-space indented.
            Returns an empty string when the file, section, or key is absent.
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

function Resolve-Tier {
    <#
        .SYNOPSIS
            Decide which tier to download for.

        .DESCRIPTION
            Order of preference: the -Tier switch, then hardware.tier in
            config/config.yaml, then a probe via scripts/verify_cuda.py, then the
            cpu tier. The cpu tier is the safe default because its models fit
            everywhere.
    #>
    param([string] $Requested)

    if ($Requested) {
        Write-Host "   tier '$Requested' given on the command line."
        return $Requested
    }

    $configured = Get-ConfiguredTier -Path $UserConfig
    if ($configured -and $configured -ne 'auto') {
        Write-Host "   tier '$configured' read from config\config.yaml."
        return $configured
    }

    if (Test-Path -LiteralPath $VerifyCuda) {
        Write-Host '   tier not resolved yet, probing with scripts\verify_cuda.py.'
        Push-Location $RepoRoot
        try {
            # Out-Host keeps the probe's own output off this function's pipeline,
            # which would otherwise end up prepended to the tier string.
            & uv run python 'scripts/verify_cuda.py' | Out-Host
            if ($LASTEXITCODE -ne 0) {
                Write-Warn "verify_cuda.py exited with code $LASTEXITCODE."
            }
        } catch {
            Write-Warn "could not run verify_cuda.py: $($_.Exception.Message)"
        } finally {
            Pop-Location
        }
        $probed = Get-ConfiguredTier -Path $UserConfig
        if ($probed -and $probed -ne 'auto') {
            Write-Host "   probe resolved the tier to '$probed'."
            return $probed
        }
    } else {
        Write-Warn 'scripts\verify_cuda.py is missing, cannot probe the hardware.'
    }

    Write-Warn "falling back to the 'cpu' tier, whose models fit any machine."
    return 'cpu'
}

function Save-FileIfMissing {
    <#
        .SYNOPSIS
            Download one file unless a plausible copy is already on disk.

        .DESCRIPTION
            The existence check is the idempotency guarantee for this script. A
            file that exists but is smaller than MinimumBytes is treated as a
            truncated or error-page download and is fetched again. Bytes land in
            a .partial file first so an interrupted run never leaves a
            half-written model behind.

            The outcome is recorded in the script-scope Downloaded, Skipped, and
            Failed lists. Nothing is written to the pipeline, so a caller can
            never confuse progress output with a result.
    #>
    param(
        [Parameter(Mandatory = $true)][string] $Destination,
        [Parameter(Mandatory = $true)][string[]] $Urls,
        [Parameter(Mandatory = $true)][string] $Label,
        [long] $MinimumBytes = 4096,
        [string] $Hint = ''
    )

    if (Test-Path -LiteralPath $Destination) {
        $size = (Get-Item -LiteralPath $Destination).Length
        if ($Force) {
            Write-Host "   -Force given, refetching $Label."
        } elseif ($size -ge $MinimumBytes) {
            Write-Skip "$Label ($(Format-Size $size))"
            $script:Skipped.Add($Label)
            return
        } else {
            Write-Warn "$Label is only $size bytes, which looks truncated. Refetching."
        }
        Remove-Item -LiteralPath $Destination -Force
    }

    $parent = Split-Path -Parent $Destination
    if (-not (Test-Path -LiteralPath $parent)) {
        New-Item -ItemType Directory -Path $parent -Force | Out-Null
    }

    $partial = "$Destination.partial"
    foreach ($url in $Urls) {
        try {
            Write-Host "   fetching $Label from $url"
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
            Invoke-WebRequest -Uri $url -OutFile $partial -UseBasicParsing -MaximumRedirection 5
            $size = (Get-Item -LiteralPath $partial).Length
            if ($size -lt $MinimumBytes) {
                Write-Warn "got only $size bytes from $url, trying the next source."
                Remove-Item -LiteralPath $partial -Force
                continue
            }
            Move-Item -LiteralPath $partial -Destination $Destination -Force
            Write-Ok "$Label ($(Format-Size $size))"
            $script:Downloaded.Add($Label)
            return
        } catch {
            Write-Warn "$url failed: $($_.Exception.Message)"
            if (Test-Path -LiteralPath $partial) { Remove-Item -LiteralPath $partial -Force }
        }
    }

    Write-Fail "$Label could not be downloaded from any source."
    if ($Hint) { Write-Hint $Hint }
    $script:Failed.Add($Label)
}

function Test-OllamaModelPresent {
    <#
        .SYNOPSIS
            True when "ollama list" already shows the model.

        .DESCRIPTION
            Compares the first whitespace delimited field of each line, so
            qwen3:4b does not match qwen3:4b-instruct.
    #>
    param([Parameter(Mandatory = $true)][string] $Model)

    $listing = @()
    try {
        $listing = @(& ollama list 2>$null)
    } catch {
        return $false
    }
    if ($LASTEXITCODE -ne 0) { return $false }

    foreach ($line in $listing) {
        if (-not $line) { continue }
        $name = @($line -split '\s+') | Select-Object -First 1
        if (-not $name) { continue }
        if ($name -eq $Model -or $name -eq ($Model + ':latest')) { return $true }
    }
    return $false
}

function Invoke-OllamaPull {
    <#
        .SYNOPSIS
            Pull a model through ollama unless "ollama list" already has it.

        .DESCRIPTION
            Like Save-FileIfMissing, the outcome goes into the script-scope
            result lists rather than onto the pipeline, so the pull's own
            progress output reaches the console untouched.
    #>
    param(
        [Parameter(Mandatory = $true)][string] $Model,
        [Parameter(Mandatory = $true)][string] $Label
    )

    if (-not (Test-CommandExists -Name 'ollama')) {
        Write-Fail "ollama is not on PATH, cannot pull $Model."
        Write-Hint 'install: winget install --id=Ollama.Ollama -e --source winget'
        $script:Failed.Add("$Label $Model")
        return
    }

    if ((Test-OllamaModelPresent -Model $Model) -and (-not $Force)) {
        Write-Skip "$Label $Model"
        $script:Skipped.Add("$Label $Model")
        return
    }

    Write-Host "   ollama pull $Model"
    & ollama pull $Model
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "ollama pull $Model exited with code $LASTEXITCODE."
        Write-Hint 'is the ollama service running? try: ollama serve'
        $script:Failed.Add("$Label $Model")
        return
    }
    Write-Ok "$Label $Model"
    $script:Downloaded.Add("$Label $Model")
}

# ---------------------------------------------------------------------------
# 1. Which tier are we downloading for
# ---------------------------------------------------------------------------

Write-Head 'JARVIS model download'
Write-Host "   repository: $RepoRoot"
Write-Host "   models:     $ModelsDir"
Write-Host ''

$resolvedTier = Resolve-Tier -Requested $Tier
if (-not $TierModels.ContainsKey($resolvedTier)) {
    throw "Unknown hardware tier '$resolvedTier'. Expected one of: $($TierModels.Keys -join ', ')."
}
$llmModel = $TierModels[$resolvedTier]

if (-not (Test-Path -LiteralPath $ModelsDir)) {
    New-Item -ItemType Directory -Path $ModelsDir -Force | Out-Null
}

# ---------------------------------------------------------------------------
# 2. The language model
# ---------------------------------------------------------------------------

Write-Host ''
Write-Host "-- Language model for tier '$resolvedTier'" -ForegroundColor White
if ($SkipLlm) {
    Write-Skip 'LLM download skipped (-SkipLlm)'
} else {
    Invoke-OllamaPull -Model $llmModel -Label 'LLM'

    if ($WithVision) {
        if ($VisionTiers -contains $resolvedTier) {
            Invoke-OllamaPull -Model $VisionModel -Label 'vision model'
        } else {
            Write-Warn "tier '$resolvedTier' has under 6 GB of VRAM to spare, so the vision"
            Write-Warn 'model is not pulled. The vision tool disables itself on this machine.'
        }
    } else {
        Write-Skip 'vision model not requested (pass -WithVision to pull it)'
    }
}

# ---------------------------------------------------------------------------
# 3. Wake word, VAD, and TTS artifacts
# ---------------------------------------------------------------------------

Write-Host ''
Write-Host '-- Wake word, endpointing, and voice models' -ForegroundColor White
foreach ($item in $FileDownloads) {
    $destination = Join-Path $ModelsDir $item.Relative
    Save-FileIfMissing `
        -Destination $destination `
        -Urls $item.Urls `
        -Label $item.Label `
        -MinimumBytes $item.MinimumBytes `
        -Hint $item.Hint
}

# ---------------------------------------------------------------------------
# 4. Summary
# ---------------------------------------------------------------------------

Write-Head 'Summary'
Write-Host "  Tier:            $resolvedTier"
Write-Host "  LLM:             $llmModel"
Write-Host "  Downloaded:      $($script:Downloaded.Count)"
Write-Host "  Already present: $($script:Skipped.Count)"
Write-Host "  Failed:          $($script:Failed.Count)"

if ($script:Downloaded.Count -gt 0) {
    Write-Host ''
    Write-Host '  Downloaded this run:' -ForegroundColor Green
    foreach ($name in $script:Downloaded) { Write-Host "    + $name" }
}

if ($script:Skipped.Count -gt 0) {
    Write-Host ''
    Write-Host '  Skipped, already on disk:' -ForegroundColor DarkGray
    foreach ($name in $script:Skipped) { Write-Host "    = $name" }
}

if ($script:Failed.Count -gt 0) {
    Write-Host ''
    Write-Host '  Failed, re-run this script once the problem is fixed:' -ForegroundColor Red
    foreach ($name in $script:Failed) { Write-Host "    ! $name" }
    Write-Host ''
    exit 1
}

Write-Host ''
Write-Host '  All models are in place. A re-run downloads nothing.' -ForegroundColor Green
Write-Host ''
exit 0
