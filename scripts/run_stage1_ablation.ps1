$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

$experiments = @(
    @{ Dataset = "PHISHING"; ConfigDir = "phishing"; LambdaStage1Anchor = "0.7" },
    @{ Dataset = "NUSWIDE"; ConfigDir = "nuswide"; LambdaStage1Anchor = "0.3" },
    @{ Dataset = "CIFAR10"; ConfigDir = "cifar10"; LambdaStage1Anchor = "0.5" },
    @{ Dataset = "IEEE_CIS_FRAUD"; ConfigDir = "ieee_cis_fraud"; LambdaStage1Anchor = "0.5" }
)

$variants = @(
    @{
        Name = "pretrained_anchor"
        Stages = @(
            @{ Name = "stage1"; Config = "stage1.json" },
            @{ Name = "stage2"; Config = "stage2.json" },
            @{ Name = "stage3"; Config = "stage3_test.json" }
        )
    },
    @{
        Name = "random_anchor"
        Stages = @(
            @{ Name = "stage2"; Config = "stage2.json" },
            @{ Name = "stage3"; Config = "stage3_test.json" }
        )
    }
)

function Invoke-AVGuardStage1AblationStage {
    param(
        [string] $Dataset,
        [string] $Variant,
        [string] $LambdaStage1Anchor,
        [string] $StageName,
        [string] $ConfigPath
    )

    Write-Host "Running dataset=$Dataset variant=$Variant lambda_stage1_anchor=$LambdaStage1Anchor stage=$StageName"
    python main.py --config "$ConfigPath"

    if ($LASTEXITCODE -ne 0) {
        throw "Failed dataset=$Dataset variant=$Variant lambda_stage1_anchor=$LambdaStage1Anchor stage=$StageName with exit code $LASTEXITCODE"
    }
}

foreach ($experiment in $experiments) {
    foreach ($variant in $variants) {
        foreach ($stage in $variant.Stages) {
            $configPath = "configs/avguard/stage1_ablation/$($experiment.ConfigDir)/$($variant.Name)/$($stage.Config)"
            Invoke-AVGuardStage1AblationStage `
                -Dataset $experiment.Dataset `
                -Variant $variant.Name `
                -LambdaStage1Anchor $experiment.LambdaStage1Anchor `
                -StageName $stage.Name `
                -ConfigPath $configPath
        }
    }
}

Write-Host "All Stage1 ablation experiments completed."
