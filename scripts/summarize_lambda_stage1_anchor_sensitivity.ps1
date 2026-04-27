$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

$root = "./results/AVGuard/lambda_stage1_anchor_sensitivity"
$output = "./results/AVGuard/lambda_stage1_anchor_sensitivity/summary.csv"

python utils/summarize_lambda_stage1_anchor_sensitivity.py --root "$root" --output "$output"

if ($LASTEXITCODE -ne 0) {
    throw "Summary failed with exit code $LASTEXITCODE"
}

Write-Host "Wrote $output"
