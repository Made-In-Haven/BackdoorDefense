$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

$experiments = @(
    @{
        Dataset = "PHISHING"
        ConfigDir = "phishing"
        Runs = @(
            @{ Eta = "0.25"; PoisonRate = "0.125"; ComboDir = "eta_0.25_poison_0.125" },
            @{ Eta = "0.5"; PoisonRate = "0.25"; ComboDir = "eta_0.5_poison_0.25" },
            @{ Eta = "0.75"; PoisonRate = "0.375"; ComboDir = "eta_0.75_poison_0.375" },
            @{ Eta = "1.0"; PoisonRate = "0.5"; ComboDir = "eta_1.0_poison_0.5" }
        )
    },
    @{
        Dataset = "NUSWIDE"
        ConfigDir = "nuswide"
        Runs = @(
            @{ Eta = "0.25"; PoisonRate = "0.05"; ComboDir = "eta_0.25_poison_0.05" },
            @{ Eta = "0.5"; PoisonRate = "0.1"; ComboDir = "eta_0.5_poison_0.1" },
            @{ Eta = "0.75"; PoisonRate = "0.15"; ComboDir = "eta_0.75_poison_0.15" },
            @{ Eta = "1.0"; PoisonRate = "0.2"; ComboDir = "eta_1.0_poison_0.2" }
        )
    },
    @{
        Dataset = "CIFAR10"
        ConfigDir = "cifar10"
        Runs = @(
            @{ Eta = "0.25"; PoisonRate = "0.025"; ComboDir = "eta_0.25_poison_0.025" },
            @{ Eta = "0.5"; PoisonRate = "0.05"; ComboDir = "eta_0.5_poison_0.05" },
            @{ Eta = "0.75"; PoisonRate = "0.075"; ComboDir = "eta_0.75_poison_0.075" },
            @{ Eta = "1.0"; PoisonRate = "0.1"; ComboDir = "eta_1.0_poison_0.1" }
        )
    },
    @{
        Dataset = "IEEE_CIS_FRAUD"
        ConfigDir = "ieee_cis_fraud"
        Runs = @(
            @{ Eta = "0.25"; PoisonRate = "0.125"; ComboDir = "eta_0.25_poison_0.125" },
            @{ Eta = "0.5"; PoisonRate = "0.25"; ComboDir = "eta_0.5_poison_0.25" },
            @{ Eta = "0.75"; PoisonRate = "0.375"; ComboDir = "eta_0.75_poison_0.375" },
            @{ Eta = "1.0"; PoisonRate = "0.5"; ComboDir = "eta_1.0_poison_0.5" }
        )
    }
)

$stages = @(
    @{ Name = "stage1"; Config = "stage1.json" },
    @{ Name = "stage2"; Config = "stage2.json" },
    @{ Name = "stage3"; Config = "stage3_test.json" }
)

function Invoke-AVGuardStage {
    param(
        [string] $Dataset,
        [string] $Eta,
        [string] $PoisonRate,
        [string] $StageName,
        [string] $ConfigPath
    )

    Write-Host "Running dataset=$Dataset eta=$Eta poison_rate=$PoisonRate stage=$StageName"
    python main.py --config "$ConfigPath"

    if ($LASTEXITCODE -ne 0) {
        throw "Failed dataset=$Dataset eta=$Eta poison_rate=$PoisonRate stage=$StageName with exit code $LASTEXITCODE"
    }
}

foreach ($experiment in $experiments) {
    foreach ($run in $experiment.Runs) {
        foreach ($stage in $stages) {
            $configPath = "configs/avguard/relative_poison_strength_ablation/$($experiment.ConfigDir)/$($run.ComboDir)/$($stage.Config)"
            Invoke-AVGuardStage `
                -Dataset $experiment.Dataset `
                -Eta $run.Eta `
                -PoisonRate $run.PoisonRate `
                -StageName $stage.Name `
                -ConfigPath $configPath
        }
    }
}

Write-Host "All relative poison strength ablation experiments completed."
