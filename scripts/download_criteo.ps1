param(
    [switch]$AcceptLicense
)

$ErrorActionPreference = "Stop"
if (-not $AcceptLicense) {
    throw "Read docs/DATA_CARD.md, then rerun with -AcceptLicense. The dataset is CC BY-NC-SA 4.0 and is not for commercial use."
}

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Destination = Join-Path $ProjectRoot "data\raw\criteo_attribution_dataset.tsv.gz"
$Url = "https://huggingface.co/datasets/criteo/criteo-attribution-dataset/resolve/main/criteo_attribution_dataset.tsv.gz?download=true"
$ExpectedSha256 = "94ac7a465564349bc7ba008602211d5990a3c53cc133abc0aadef61ea2391a98"

New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
Invoke-WebRequest -Uri $Url -OutFile $Destination
$ActualSha256 = (Get-FileHash -Algorithm SHA256 $Destination).Hash.ToLowerInvariant()
if ($ActualSha256 -ne $ExpectedSha256) {
    throw "SHA256 mismatch. Expected $ExpectedSha256 but got $ActualSha256"
}
Write-Host "Verified download: $Destination"

