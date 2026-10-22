param(
    [Parameter(Mandatory=$true)]
    [string]$PolicyPath,
    [string]$Event = "sprint-2m"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$challengesDir = "C:\GitHub\microduck_challenges"
$python = "$challengesDir\.venv\Scripts\python.exe"

if (-not (Test-Path $PolicyPath)) {
    Write-Error "Policy file not found at: $PolicyPath"
}

Write-Host "Scoring policy $PolicyPath for event: $Event" -ForegroundColor Cyan
& $python -m arena.cli score $PolicyPath --event $Event
