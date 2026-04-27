$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

function Invoke-Stage {
    param(
        [string]$Dataset,
        [double]$PoisonRate,
        [string]$Stage,
        [string]$ConfigPath
    )

    Write-Host "Dataset=$Dataset poison_rate=$PoisonRate stage=$Stage client_num=4 attack_client_num=3"
    python main.py --config "$ConfigPath"
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed for dataset=$Dataset stage=$Stage with exit code $LASTEXITCODE"
    }
}

Invoke-Stage -Dataset "PHISHING" -PoisonRate 0.25 -Stage "Stage1" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/phishing/stage1.json"
Invoke-Stage -Dataset "PHISHING" -PoisonRate 0.25 -Stage "Stage2" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/phishing/stage2.json"
Invoke-Stage -Dataset "PHISHING" -PoisonRate 0.25 -Stage "Stage3" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/phishing/stage3_test.json"

Invoke-Stage -Dataset "NUSWIDE" -PoisonRate 0.1 -Stage "Stage1" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/nuswide/stage1.json"
Invoke-Stage -Dataset "NUSWIDE" -PoisonRate 0.1 -Stage "Stage2" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/nuswide/stage2.json"
Invoke-Stage -Dataset "NUSWIDE" -PoisonRate 0.1 -Stage "Stage3" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/nuswide/stage3_test.json"

Invoke-Stage -Dataset "CIFAR10" -PoisonRate 0.05 -Stage "Stage1" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/cifar10/stage1.json"
Invoke-Stage -Dataset "CIFAR10" -PoisonRate 0.05 -Stage "Stage2" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/cifar10/stage2.json"
Invoke-Stage -Dataset "CIFAR10" -PoisonRate 0.05 -Stage "Stage3" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/cifar10/stage3_test.json"

Invoke-Stage -Dataset "IEEE_CIS_FRAUD" -PoisonRate 0.25 -Stage "Stage1" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/ieee_cis_fraud/stage1.json"
Invoke-Stage -Dataset "IEEE_CIS_FRAUD" -PoisonRate 0.25 -Stage "Stage2" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/ieee_cis_fraud/stage2.json"
Invoke-Stage -Dataset "IEEE_CIS_FRAUD" -PoisonRate 0.25 -Stage "Stage3" -ConfigPath "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/ieee_cis_fraud/stage3_test.json"
