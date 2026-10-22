param(
    [string]$Challenge = "sprint-2m"
)

$ErrorActionPreference = "Stop"
$challengesDir = "C:\GitHub\microduck_challenges"
$python = "$challengesDir\.venv\Scripts\python.exe"

Write-Host "Launching MicroDuck Workshop ($Challenge) on Jupiter..." -ForegroundColor Cyan
Set-Location $challengesDir
& $python -m microduck_challenges.workshop.cli $Challenge
