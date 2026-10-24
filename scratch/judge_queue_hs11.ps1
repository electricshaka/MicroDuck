# Judges every hs11 checkpoint as it lands: dry-run publish export, then headspin_check
# (seeds 1-3), check.py, arena score and the toprock probe, results in the run folder.
$ErrorActionPreference = 'Continue'
$env:PYTHONUTF8 = '1'
$root = 'C:\GitHub\microduck-private'
Set-Location $root
$runRel = 'logs/rsl_rl/got_a_talent/2026-10-24_05-08-48_hs11'
$run = Join-Path $root $runRel
$trainPid = 51376

foreach ($n in 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 2999) {
    $pt = Join-Path $run "model_$n.pt"
    $deadline = (Get-Date).AddHours(9)
    while ((Get-Date) -lt $deadline) {
        if (Test-Path $pt) {
            $age = ((Get-Date) - (Get-Item $pt).LastWriteTime).TotalSeconds
            if ($age -gt 20) { break }
        }
        if (-not (Get-Process -Id $trainPid -ErrorAction SilentlyContinue) -and -not (Test-Path $pt)) { break }
        Start-Sleep -Seconds 20
    }
    if (-not (Test-Path $pt)) { Write-Output "checkpoint ${n} never appeared, moving on"; continue }

    $jf = Join-Path $run "judge_$n.txt"
    if (Test-Path $jf) { Write-Output "checkpoint ${n} already judged"; continue }

    $pub = uv run --no-sync publish --run $runRel --checkpoint $n --repo local/microduck-eval --dry-run --allow-dirty 2>&1
    if ($LASTEXITCODE -ne 0) { Write-Output "checkpoint ${n} export FAILED"; $pub | Select-Object -Last 5 | Out-String | Set-Content $jf -Encoding UTF8; continue }
    $pol = Join-Path $root 'publish-eval/policy.onnx'
    $keep = Join-Path $run "model_$n.onnx"
    Copy-Item $pol $keep -Force

    $hs = uv run --no-sync --group workshop python -m microduck_challenges.got_a_talent.headspin_check $keep 2>&1
    $ck = uv run --no-sync --group workshop python -m microduck_challenges.got_a_talent.check $keep 2>&1
    $sc = uv run --no-sync --group workshop arena score $keep --event got-a-talent --timeline src/microduck_challenges/got_a_talent/timeline.json 2>&1
    $tp = uv run --no-sync --group workshop python scratch/toprock_probe.py $keep 2>&1
    (@('=== headspin_check (seeds 1-3) ===') + $hs + @('', '=== check.py ===') + $ck +
      @('', '=== arena score ===') + $sc + @('', '=== toprock probe (2.5 s stand, body[0] off/on) ===') + $tp) | Set-Content $jf -Encoding UTF8
    $mean = ($hs | Where-Object { $_ -like 'mean *' }) -join ''
    $stepLine = ($tp | Where-Object { $_ -match 'body\[0\]=\d MEAN' }) -join ' | '
    Write-Output "checkpoint $n judged: $mean | toprock: $stepLine"
}
Write-Output 'judge queue done'
