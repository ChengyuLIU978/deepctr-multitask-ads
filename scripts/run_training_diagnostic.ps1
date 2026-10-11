$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Data = Join-Path $ProjectRoot "data\processed\criteo_uniform_10pct.csv.gz"

if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
if (-not (Test-Path $Data)) { throw "Prepare the 10% sample first; see README.md." }

& $Python -m ctr_cvr.train `
    --data-mode real `
    --data $Data `
    --models deepfm,esmm,mmoe `
    --learning-rate 0.0003 `
    --epochs 12 `
    --patience 4 `
    --min-delta 0 `
    --batch-size 2048 `
    --threads 8 `
    --seed 2026 `
    --output-dir (Join-Path $ProjectRoot "artifacts\diagnostic_real_10pct_lr3e4_seed2026")
