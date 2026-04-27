param(
    [switch] $DryRun,
    [switch] $Rerun,
    [double[]] $TauCorrs = @(0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30)
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

function Resolve-ExistingPath {
    param(
        [string[]] $Candidates,
        [string] $Description
    )

    foreach ($candidate in $Candidates) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Could not find $Description. Tried: $($Candidates -join ', ')"
}

function Test-CompletedExperiment {
    param(
        [string] $LogPath
    )

    if (-not (Test-Path $LogPath)) {
        return $false
    }

    $match = Select-String -Path $LogPath -Pattern "=> Test Epoch:" -Quiet
    return [bool] $match
}

function Invoke-TauCorrExperiment {
    param(
        [string] $Dataset,
        [string] $ConfigPath,
        [string] $CheckpointPath,
        [string] $OutputRoot,
        [double] $TauCorr
    )

    $tauLabel = "tau_{0:N2}" -f $TauCorr
    $outputDir = Join-Path $OutputRoot $tauLabel
    $logPath = Join-Path $outputDir "experiment.log"

    Write-Host "Dataset: $Dataset"
    Write-Host "tau_corr: $TauCorr"
    Write-Host "Config path: $ConfigPath"
    Write-Host "Checkpoint path: $CheckpointPath"
    Write-Host "Output directory: $outputDir"

    if ((-not $Rerun) -and (Test-CompletedExperiment -LogPath $logPath)) {
        Write-Host "Skip existing completed run: $Dataset $tauLabel"
        return
    }

    $commandArgs = @(
        "main.py",
        "--config", $ConfigPath,
        "--mode", "test",
        "--enable_conservative_correction", "true",
        "--tau_corr", ("{0:N2}" -f $TauCorr),
        "--test_checkpoint", $CheckpointPath,
        "--results_dir", $outputDir
    )

    if ($DryRun) {
        Write-Host "python $($commandArgs -join ' ')"
        return
    }

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    "python $($commandArgs -join ' ')" | Set-Content -Path (Join-Path $outputDir "run_command.txt") -Encoding UTF8

    python @commandArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed dataset=$Dataset tau_corr=$TauCorr with exit code $LASTEXITCODE"
    }
}

$experiments = @(
    @{
        Dataset = "CIFAR10"
        ConfigPath = Resolve-ExistingPath `
            -Description "CIFAR10 stage3_test.json" `
            -Candidates @(
                "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/cifar10/stage3_test.json",
                "dataset/configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/cifar10/stage3_test.json"
            )
        CheckpointPath = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/CIFAR10/stage2/best_checkpoint.pth.tar"
        OutputRoot = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/CIFAR10/stage3_tau_corr_sweep"
    },
    @{
        Dataset = "NUSWIDE"
        ConfigPath = Resolve-ExistingPath `
            -Description "NUSWIDE stage3_test.json" `
            -Candidates @(
                "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/nuswide/stage3_test.json",
                "dataset/configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/nuswide/stage3_test.json"
            )
        CheckpointPath = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/NUSWIDE/stage2/best_checkpoint.pth.tar"
        OutputRoot = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/NUSWIDE/stage3_tau_corr_sweep"
    },
    @{
        Dataset = "IEEE_CIS_FRAUD"
        ConfigPath = Resolve-ExistingPath `
            -Description "IEEE_CIS_FRAUD stage3_test.json" `
            -Candidates @(
                "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/ieee_cis_fraud/stage3_test.json",
                "dataset/configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/ieee_cis_fraud/stage3_test.json"
            )
        CheckpointPath = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/IEEE_CIS_FRAUD/stage2/best_checkpoint.pth.tar"
        OutputRoot = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/IEEE_CIS_FRAUD/stage3_tau_corr_sweep"
    },
    @{
        Dataset = "PHISHING"
        ConfigPath = Resolve-ExistingPath `
            -Description "PHISHING stage3_test.json" `
            -Candidates @(
                "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/phishing/stage3_test.json",
                "dataset/configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/phishing/stage3_test.json"
            )
        CheckpointPath = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/PHISHING/stage2/best_checkpoint.pth.tar"
        OutputRoot = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/PHISHING/stage3_tau_corr_sweep"
    }
)

foreach ($experiment in $experiments) {
    if (-not (Test-Path $experiment.CheckpointPath)) {
        throw "Missing checkpoint for dataset=$($experiment.Dataset): $($experiment.CheckpointPath)"
    }

    foreach ($tauCorr in $TauCorrs) {
        Invoke-TauCorrExperiment `
            -Dataset $experiment.Dataset `
            -ConfigPath $experiment.ConfigPath `
            -CheckpointPath $experiment.CheckpointPath `
            -OutputRoot $experiment.OutputRoot `
            -TauCorr $tauCorr
    }
}

if (-not $DryRun) {
    python utils/run_tau_corr_sweep.py --summarize-only --tau-corrs ($TauCorrs -join ",")

    if ($LASTEXITCODE -ne 0) {
        throw "Failed to summarize tau_corr sweep logs with exit code $LASTEXITCODE"
    }
}

Write-Host "Stage 3 tau_corr sweep script completed."
