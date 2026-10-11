$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Data = Join-Path $ProjectRoot "data\processed\criteo_uniform_10pct.csv.gz"

if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
if (-not (Test-Path $Data)) { throw "Prepare the 10% sample first; see README.md." }

foreach ($Experts in @(2, 5)) {
    & $Python -m ctr_cvr.train `
        --data-mode real `
        --data $Data `
        --models mmoe `
        --num-experts $Experts `
        --epochs 10 `
        --patience 2 `
        --batch-size 2048 `
        --threads 8 `
        --seed 2026 `
        --output-dir (Join-Path $ProjectRoot "artifacts\ablation_real_10pct_mmoe_experts$Experts")
}

& $Python -m ctr_cvr.train `
    --data-mode real `
    --data $Data `
    --models deepfm,esmm,mmoe `
    --feature-set sparse_only `
    --epochs 10 `
    --patience 2 `
    --batch-size 2048 `
    --threads 8 `
    --seed 2026 `
    --output-dir (Join-Path $ProjectRoot "artifacts\ablation_real_10pct_sparse_only")

