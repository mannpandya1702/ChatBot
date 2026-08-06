#Requires -Version 5.1
<#
.SYNOPSIS
    Install or remove a Task Scheduler entry that starts JARVIS at logon.

.DESCRIPTION
    CLAUDE.md T-5.4. Opt-in and non-elevated: the task runs as the current user
    with the lowest run level, matching section 6's privilege separation. The
    elevated helper is a separate artifact and is never autostarted with
    administrator rights by this task.

    Task Scheduler is used rather than a Run registry key because it supports a
    delayed start, restart on failure, and clean removal, and because section 6
    forbids registry edits from the assistant itself.

.PARAMETER Remove
    Remove the task instead of installing it.

.PARAMETER Headless
    Start without the HUD, using the --headless flag.

.PARAMETER DelaySeconds
    How long to wait after logon before starting. Default 30, which lets the
    desktop settle and Ollama come up first.

.PARAMETER TaskName
    Scheduled task name. Default JarvisAssistant.

.EXAMPLE
    .\scripts\install_autostart.ps1
    .\scripts\install_autostart.ps1 -Headless
    .\scripts\install_autostart.ps1 -Remove
#>

param(
    [switch]$Remove,
    [switch]$Headless,
    [int]$DelaySeconds = 30,
    [string]$TaskName = "JarvisAssistant"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$RepoRoot = Split-Path -Parent $PSScriptRoot

function Write-Step($Message) { Write-Host "==> $Message" -ForegroundColor Cyan }
function Write-Ok($Message) { Write-Host "    $Message" -ForegroundColor Green }
function Write-Warn($Message) { Write-Host "    $Message" -ForegroundColor Yellow }

function Test-TaskExists {
    param([string]$Name)
    $null -ne (Get-ScheduledTask -TaskName $Name -ErrorAction SilentlyContinue)
}

# ---------------------------------------------------------------------------
# Removal
# ---------------------------------------------------------------------------

if ($Remove) {
    Write-Step "Removing the autostart task"
    if (Test-TaskExists -Name $TaskName) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Ok "Removed scheduled task '$TaskName'"
    }
    else {
        Write-Ok "No task named '$TaskName' was installed, nothing to do"
    }
    exit 0
}

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------

Write-Step "Installing the autostart task"

$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    throw "uv is not on PATH. Run scripts\setup_env.ps1 first."
}

$configPath = Join-Path $RepoRoot "config\config.yaml"
if (-not (Test-Path $configPath)) {
    Write-Warn "config\config.yaml does not exist yet. Run scripts\setup_env.ps1 first."
}

# Idempotent: replace an existing task rather than failing or duplicating it.
if (Test-TaskExists -Name $TaskName) {
    Write-Warn "Task '$TaskName' already exists and will be replaced"
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

$arguments = "run python -m jarvis"
if ($Headless) {
    $arguments += " --headless"
}

$action = New-ScheduledTaskAction `
    -Execute $uv.Source `
    -Argument $arguments `
    -WorkingDirectory $RepoRoot

$trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$trigger.Delay = "PT${DelaySeconds}S"

# Limited run level is the point of this task. Elevating here would defeat the
# privilege separation the whole design rests on.
$principal = New-ScheduledTaskPrincipal `
    -UserId "$env:USERDOMAIN\$env:USERNAME" `
    -LogonType Interactive `
    -RunLevel Limited

$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 0)

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $action `
    -Trigger $trigger `
    -Principal $principal `
    -Settings $settings `
    -Description "Starts the JARVIS voice assistant at logon, non-elevated." | Out-Null

Write-Ok "Installed scheduled task '$TaskName'"

# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------

$task = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if (-not $task) {
    throw "the task was registered but cannot be read back"
}

if ($task.Principal.RunLevel -ne "Limited") {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    throw "the task was created elevated, which violates the privilege separation rule; removed it"
}

Write-Ok "Verified the task runs non-elevated"

Write-Host ""
Write-Host "Done." -ForegroundColor Green
Write-Host "  Task:        $TaskName"
Write-Host "  Command:     $($uv.Source) $arguments"
Write-Host "  Working dir: $RepoRoot"
Write-Host "  Delay:       $DelaySeconds seconds after logon"
Write-Host "  Run level:   Limited (non-elevated)"
Write-Host ""
Write-Host "  Start it now:      Start-ScheduledTask -TaskName $TaskName" -ForegroundColor Gray
Write-Host "  Remove it later:   .\scripts\install_autostart.ps1 -Remove" -ForegroundColor Gray
