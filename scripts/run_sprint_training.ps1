param(
    [int]$NumEnvs = 4096,
    [int]$MaxIterations = 3000,
    [int]$Seed = 1,
    [string]$RunName = "jupiter_sprint_01",
    [switch]$SmokeTest
)

$ErrorActionPreference = "Stop"
$challengesDir = "C:\GitHub\microduck_challenges"
$python = "$challengesDir\.venv\Scripts\python.exe"

if ($SmokeTest) {
    Write-Host "Running 5-iteration smoke test on RTX 4090..." -ForegroundColor Yellow
    $NumEnvs = 64
    $MaxIterations = 5
    $RunName = "smoke_test"
}

Write-Host "Starting MicroDuck 2m Sprint Training on RTX 4090 (Jupiter)..." -ForegroundColor Green
Write-Host "  Envs: $NumEnvs"
Write-Host "  Max Iterations: $MaxIterations"
Write-Host "  Run Name: $RunName"

Set-Location $challengesDir
& $python -m mjlab_microduck.train_cli Mjlab-Sprint2m-MicroDuck `
    --env.scene.num-envs $NumEnvs `
    --agent.max_iterations $MaxIterations `
    --agent.seed $Seed `
    --agent.logger tensorboard `
    --agent.run-name $RunName
