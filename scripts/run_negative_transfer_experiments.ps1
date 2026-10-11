$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Data = Join-Path $ProjectRoot "data\processed\criteo_uniform_10pct.csv.gz"

if (-not (Test-Path $Python)) { throw "Run scripts/setup.ps1 first." }
if (-not (Test-Path $Data)) { throw "Prepare the 10% sample first; see README.md." }

$Configurations = @(
    @{ Name = "sharedbottom_equal"; Model = "sharedbottom"; Weights = "1,1"; Experts = 3 },
    @{ Name = "mmoe_equal"; Model = "mmoe"; Weights = "1,1"; Experts = 3 },
    @{ Name = "mmoe_ctr_heavy"; Model = "mmoe"; Weights = "1.5,0.5"; Experts = 3 },
    @{ Name = "mmoe_ctcvr_heavy"; Model = "mmoe"; Weights = "0.5,1.5"; Experts = 3 },
    @{ Name = "mmoe_experts2"; Model = "mmoe"; Weights = "1,1"; Experts = 2 },
    @{ Name = "mmoe_experts5"; Model = "mmoe"; Weights = "1,1"; Experts = 5 }
)

foreach ($Configuration in $Configurations) {
    $RunDirectories = @()
    foreach ($Seed in @(2026, 2027, 2028)) {
        $RunDirectory = Join-Path $ProjectRoot "artifacts\negative_$($Configuration.Name)_seed$Seed"
        $RunDirectories += $RunDirectory
        & $Python -m ctr_cvr.train `
            --data-mode real `
            --data $Data `
            --models $Configuration.Model `
            --task-loss-weights $Configuration.Weights `
            --num-experts $Configuration.Experts `
            --epochs 10 `
            --patience 2 `
            --batch-size 2048 `
            --threads 8 `
            --seed $Seed `
            --output-dir $RunDirectory
    }

    & $Python -m ctr_cvr.summarize `
        --run-dirs $RunDirectories `
        --output (Join-Path $ProjectRoot "artifacts\negative_$($Configuration.Name)_summary.json")
}

& $Python -m ctr_cvr.analyze_negative_transfer
