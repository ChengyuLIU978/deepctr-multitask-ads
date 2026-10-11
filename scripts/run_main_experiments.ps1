$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Data = Join-Path $ProjectRoot "data\processed\criteo_uniform_10pct.csv.gz"

if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
if (-not (Test-Path $Data)) { throw "Prepare the 10% sample first; see README.md." }

foreach ($Seed in @(2026, 2027, 2028)) {
    & $Python -m ctr_cvr.train `
        --data-mode real `
        --data $Data `
        --models deepfm,esmm,mmoe `
        --epochs 10 `
        --patience 2 `
        --batch-size 2048 `
        --threads 8 `
        --seed $Seed `
        --output-dir (Join-Path $ProjectRoot "artifacts\final_real_10pct_seed$Seed")
}

& $Python -m ctr_cvr.summarize `
    --run-dirs `
        (Join-Path $ProjectRoot "artifacts\final_real_10pct_seed2026") `
        (Join-Path $ProjectRoot "artifacts\final_real_10pct_seed2027") `
        (Join-Path $ProjectRoot "artifacts\final_real_10pct_seed2028") `
    --output (Join-Path $ProjectRoot "artifacts\final_real_10pct_summary.json")

