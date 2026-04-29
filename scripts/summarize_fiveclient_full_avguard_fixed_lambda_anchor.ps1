$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

$root = "./results/AVGuard/fiveclient_full_avguard_fixed_lambda_anchor"
$output = "./results/AVGuard/fiveclient_full_avguard_fixed_lambda_anchor/summary.csv"

python utils/summarize_fiveclient_full_avguard_fixed_lambda_anchor.py --root "$root" --output "$output"

if ($LASTEXITCODE -ne 0) {
    throw "Summary failed with exit code $LASTEXITCODE"
}

Write-Host "Wrote $output"
