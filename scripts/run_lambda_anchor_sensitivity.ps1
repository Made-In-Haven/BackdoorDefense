$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

$experiments = @(
    @{ Dataset = "PHISHING"; ConfigDir = "phishing"; PoisonRate = "0.25" },
    @{ Dataset = "NUSWIDE"; ConfigDir = "nuswide"; PoisonRate = "0.1" },
    @{ Dataset = "CIFAR10"; ConfigDir = "cifar10"; PoisonRate = "0.05" },
    @{ Dataset = "IEEE_CIS_FRAUD"; ConfigDir = "ieee_cis_fraud"; PoisonRate = "0.25" }
)

$lambdaValues = @("0.25", "0.5", "0.75")

function Invoke-AVGuardConfig {
    param(
        [string] $Dataset,
        [string] $PoisonRate,
        [string] $StageName,
        [string] $ConfigPath,
        [string] $LambdaAnchor = ""
    )

    if ($LambdaAnchor -eq "") {
        Write-Host "Running dataset=$Dataset poison_rate=$PoisonRate stage=$StageName"
    }
    else {
        Write-Host "Running dataset=$Dataset lambda_anchor=$LambdaAnchor poison_rate=$PoisonRate stage=$StageName"
    }

    python main.py --config "$ConfigPath"

    if ($LASTEXITCODE -ne 0) {
        if ($LambdaAnchor -eq "") {
            throw "Failed dataset=$Dataset poison_rate=$PoisonRate stage=$StageName with exit code $LASTEXITCODE"
        }
        throw "Failed dataset=$Dataset lambda_anchor=$LambdaAnchor poison_rate=$PoisonRate stage=$StageName with exit code $LASTEXITCODE"
    }
}

foreach ($experiment in $experiments) {
    $stage1Config = "configs/avguard/lambda_anchor_sensitivity/$($experiment.ConfigDir)/stage1.json"
    Invoke-AVGuardConfig `
        -Dataset $experiment.Dataset `
        -PoisonRate $experiment.PoisonRate `
        -StageName "stage1" `
        -ConfigPath $stage1Config

    foreach ($lambdaValue in $lambdaValues) {
        $configDir = "configs/avguard/lambda_anchor_sensitivity/$($experiment.ConfigDir)/lambda_anchor_$lambdaValue"

        Invoke-AVGuardConfig `
            -Dataset $experiment.Dataset `
            -PoisonRate $experiment.PoisonRate `
            -LambdaAnchor $lambdaValue `
            -StageName "stage2" `
            -ConfigPath "$configDir/stage2.json"

        Invoke-AVGuardConfig `
            -Dataset $experiment.Dataset `
            -PoisonRate $experiment.PoisonRate `
            -LambdaAnchor $lambdaValue `
            -StageName "stage3" `
            -ConfigPath "$configDir/stage3_test.json"
    }
}

Write-Host "All lambda_anchor sensitivity experiments completed."
