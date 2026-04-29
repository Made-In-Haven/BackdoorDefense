param(
    [string] $Python = "python",
    [switch] $DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py" -PathType Leaf)) {
    throw "Please run this script from the project root directory; main.py was not found."
}

$configRoot = "configs/avguard/fiveclient_full_avguard_fixed_lambda_anchor"

$experiments = @(
    @{ Dataset = "PHISHING"; ConfigDir = "phishing" },
    @{ Dataset = "NUSWIDE"; ConfigDir = "nuswide" },
    @{ Dataset = "CIFAR10"; ConfigDir = "cifar10" },
    @{ Dataset = "IEEE_CIS_FRAUD"; ConfigDir = "ieee_cis_fraud" }
)

$stages = @(
    @{ Name = "Stage1"; ConfigFile = "stage1.json" },
    @{ Name = "Stage2"; ConfigFile = "stage2.json" },
    @{ Name = "Stage3"; ConfigFile = "stage3_test.json" }
)

function Invoke-AVGuardStage {
    param(
        [string] $Dataset,
        [string] $Stage,
        [string] $ConfigPath
    )

    if (-not (Test-Path $ConfigPath -PathType Leaf)) {
        throw "Missing config for dataset=$Dataset stage=$($Stage): $($ConfigPath)"
    }

    $config = Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
    Write-Host ""
    Write-Host "Dataset: $($config.dataset)"
    Write-Host "poison_rate: $($config.poison_rate)"
    Write-Host "Stage: $Stage"
    Write-Host "client_num=$($config.client_num)"
    Write-Host "attack_client_num=$($config.attack_client_num)"
    Write-Host "Config: $ConfigPath"

    $commandArgs = @("main.py", "--config", $ConfigPath)
    if ($DryRun) {
        Write-Host "$Python $($commandArgs -join ' ')"
        return
    }

    & $Python @commandArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Failed dataset=$($config.dataset) poison_rate=$($config.poison_rate) stage=$Stage client_num=$($config.client_num) attack_client_num=$($config.attack_client_num) exit_code=$LASTEXITCODE"
    }
}

foreach ($experiment in $experiments) {
    foreach ($stage in $stages) {
        $configPath = Join-Path $configRoot (Join-Path $experiment.ConfigDir $stage.ConfigFile)
        Invoke-AVGuardStage `
            -Dataset $experiment.Dataset `
            -Stage $stage.Name `
            -ConfigPath $configPath
    }
}

Write-Host ""
Write-Host "Five-client AVGuard fixed lambda-anchor run sequence completed."
