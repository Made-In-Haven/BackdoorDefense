$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

$experiments = @(
    @{ Dataset = "PHISHING"; ConfigDir = "phishing" },
    @{ Dataset = "NUSWIDE"; ConfigDir = "nuswide" },
    @{ Dataset = "CIFAR10"; ConfigDir = "cifar10" },
    @{ Dataset = "IEEE_CIS_FRAUD"; ConfigDir = "ieee_cis_fraud" }
)

$lambdaValues = @("0.3", "0.5", "0.7")
$stages = @(
    @{ Name = "stage1"; Config = "stage1.json" },
    @{ Name = "stage2"; Config = "stage2.json" },
    @{ Name = "stage3"; Config = "stage3_test.json" }
)

function Invoke-AVGuardStage {
    param(
        [string] $Dataset,
        [string] $LambdaValue,
        [string] $StageName,
        [string] $ConfigPath
    )

    Write-Host "Running dataset=$Dataset lambda_stage1_anchor=$LambdaValue stage=$StageName"
    python main.py --config "$ConfigPath"

    if ($LASTEXITCODE -ne 0) {
        throw "Failed dataset=$Dataset lambda_stage1_anchor=$LambdaValue stage=$StageName with exit code $LASTEXITCODE"
    }
}

foreach ($experiment in $experiments) {
    foreach ($lambdaValue in $lambdaValues) {
        foreach ($stage in $stages) {
            $configPath = "configs/avguard/lambda_stage1_anchor_sensitivity/$($experiment.ConfigDir)/lambda_$lambdaValue/$($stage.Config)"
            Invoke-AVGuardStage `
                -Dataset $experiment.Dataset `
                -LambdaValue $lambdaValue `
                -StageName $stage.Name `
                -ConfigPath $configPath
        }
    }
}

Write-Host "All lambda_stage1_anchor sensitivity experiments completed."
