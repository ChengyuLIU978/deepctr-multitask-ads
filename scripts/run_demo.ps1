$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $Python)) {
    throw "Run scripts/setup.ps1 first."
}

& $Python -m ctr_cvr.train `
    --data-mode demo `
    --models deepfm,esmm,mmoe `
    --epochs 2 `
    --batch-size 512 `
    --seed 2026

