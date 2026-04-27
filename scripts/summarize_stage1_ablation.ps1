$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

$root = "./results/AVGuard/stage1_ablation"
$output = "./results/AVGuard/stage1_ablation/summary.csv"

python utils/summarize_stage1_ablation.py --root "$root" --output "$output"

if ($LASTEXITCODE -ne 0) {
    throw "Summary failed with exit code $LASTEXITCODE"
}

Write-Host "Wrote $output"
