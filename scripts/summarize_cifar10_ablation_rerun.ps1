$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

$resultRoot = "result/cifar10_ablation_rerun_scale16"
$configRoot = "configs/avguard/cifar10_ablation_rerun_scale16"

python utils/summarize_cifar10_ablation_rerun.py --result-root "$resultRoot" --config-root "$configRoot"

if ($LASTEXITCODE -ne 0) {
    throw "Summary failed with exit code $LASTEXITCODE"
}

Write-Host "Wrote summary files under $resultRoot"
