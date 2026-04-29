param(
    [switch] $DryRun,
    [string] $Python = "python",
    [switch] $Stage3Only,
    [switch] $ForcePrerequisites
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

$experimentRoot = "configs/avguard/threeclient_full_avguard_fixed_lambda_anchor_theta_supp_sensitivity"
$thetaSupps = @("0.05", "0.10", "0.15", "0.20", "0.25")
$datasets = @(
    @{ Name = "CIFAR10"; ConfigDir = "cifar10" },
    @{ Name = "NUSWIDE"; ConfigDir = "nuswide" },
    @{ Name = "PHISHING"; ConfigDir = "phishing" },
    @{ Name = "IEEE_CIS_FRAUD"; ConfigDir = "ieee_cis_fraud" }
)

function Read-JsonConfig {
    param([string] $ConfigPath)
    return Get-Content -Path $ConfigPath -Raw | ConvertFrom-Json
}

function Get-StageOutputPath {
    param(
        [string] $StageName,
        [object] $Config
    )

    if ($StageName -eq "stage1") {
        return $Config.anchor_bank_path
    }

    if ($StageName -eq "stage2") {
        return Join-Path $Config.results_dir "best_checkpoint.pth.tar"
    }

    return $null
}

function Invoke-AVGuardStage {
    param(
        [string] $Dataset,
        [string] $ThetaSupp,
        [string] $StageName,
        [string] $ConfigPath
    )

    if (-not (Test-Path $ConfigPath)) {
        throw "Missing config: $ConfigPath"
    }

    $config = Read-JsonConfig -ConfigPath $ConfigPath
    $expectedOutput = Get-StageOutputPath -StageName $StageName -Config $config

    if (
        -not $ForcePrerequisites `
        -and $StageName -ne "stage3" `
        -and $expectedOutput `
        -and (Test-Path $expectedOutput)
    ) {
        Write-Host "Skipping dataset=$Dataset theta_supp=$ThetaSupp stage=$StageName; found $expectedOutput"
        return
    }

    Write-Host "Running dataset=$Dataset theta_supp=$ThetaSupp stage=$StageName"
    Write-Host "$Python main.py --config `"$ConfigPath`""

    if ($DryRun) {
        return
    }

    & $Python "main.py" "--config" $ConfigPath

    if ($LASTEXITCODE -ne 0) {
        throw "Failed dataset=$Dataset theta_supp=$ThetaSupp stage=$StageName with exit code $LASTEXITCODE"
    }
}

foreach ($thetaSupp in $thetaSupps) {
    foreach ($dataset in $datasets) {
        $configDir = "$experimentRoot/theta_supp_$thetaSupp/$($dataset.ConfigDir)"

        if (-not $Stage3Only) {
            Invoke-AVGuardStage `
                -Dataset $dataset.Name `
                -ThetaSupp $thetaSupp `
                -StageName "stage1" `
                -ConfigPath "$configDir/stage1.json"

            Invoke-AVGuardStage `
                -Dataset $dataset.Name `
                -ThetaSupp $thetaSupp `
                -StageName "stage2" `
                -ConfigPath "$configDir/stage2.json"
        }

        Invoke-AVGuardStage `
            -Dataset $dataset.Name `
            -ThetaSupp $thetaSupp `
            -StageName "stage3" `
            -ConfigPath "$configDir/stage3_test.json"
    }
}

Write-Host "All three-client theta_supp sensitivity experiments completed."
