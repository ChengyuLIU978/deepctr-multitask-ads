param(
    [double]$SampleRate = 0.10,
    [int]$Seed = 2026,
    [string]$OutputName = "criteo_uniform_10pct.csv.gz"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Raw = Join-Path $ProjectRoot "data\raw\criteo_attribution_dataset.tsv.gz"
$Prepared = Join-Path $ProjectRoot "data\processed\$OutputName"

if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
if (-not (Test-Path $Raw)) { throw "Run scripts/download_criteo.ps1 -AcceptLicense first." }

& $Python -m ctr_cvr.prepare_criteo `
    --input $Raw `
    --output $Prepared `
    --sample-rate $SampleRate `
    --seed $Seed

