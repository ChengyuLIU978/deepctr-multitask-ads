param(
    [string]$RunName = "final_real_10pct_seed2026",
    [string[]]$Models = @("deepfm", "esmm", "mmoe")
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Data = Join-Path $ProjectRoot "data\processed\criteo_uniform_10pct.csv.gz"
$RunDir = Join-Path $ProjectRoot "artifacts\$RunName"

if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
if (-not (Test-Path $Data)) { throw "Prepare the 10% sample first; see README.md." }
if (-not (Test-Path $RunDir)) { throw "Run directory not found: $RunDir" }

foreach ($Model in $Models) {
    & $Python -m ctr_cvr.evaluate `
        --run-dir $RunDir `
        --data $Data `
        --model $Model `
        --batch-size 2048
}
