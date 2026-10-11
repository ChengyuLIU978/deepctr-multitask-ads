$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Data = Join-Path $ProjectRoot "data\processed\criteo_uniform_1pct.csv.gz"

if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
if (-not (Test-Path $Data)) {
    throw "Run scripts/prepare_real_data.ps1 -SampleRate 0.01 -OutputName criteo_uniform_1pct.csv.gz first."
}

& $Python -m ctr_cvr.train `
    --data-mode real `
    --data $Data `
    --models deepfm,esmm,mmoe `
    --epochs 1 `
    --batch-size 1024 `
    --seed 2026

