param(
    [switch] $DryRun,
    [switch] $Rerun,
    [string[]] $Datasets = @("CIFAR10"),
    [double[]] $ThetaSupps = @(0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30),
    [string] $Python = "python"
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

    return [bool](Select-String -Path $LogPath -Pattern "=> Test Epoch:" -Quiet)
}

function ConvertTo-ThetaLabel {
    param(
        [double] $ThetaSupp
    )

    return "theta_{0:N2}" -f $ThetaSupp
}

function Get-ExperimentSpec {
    param(
        [string] $Dataset
    )

    switch ($Dataset.ToUpperInvariant()) {
        "CIFAR10" {
            return @{
                Dataset = "CIFAR10"
                ConfigDir = "cifar10"
                ResultDir = "CIFAR10"
            }
        }
        "NUSWIDE" {
            return @{
                Dataset = "NUSWIDE"
                ConfigDir = "nuswide"
                ResultDir = "NUSWIDE"
            }
        }
        "IEEE_CIS_FRAUD" {
            return @{
                Dataset = "IEEE_CIS_FRAUD"
                ConfigDir = "ieee_cis_fraud"
                ResultDir = "IEEE_CIS_FRAUD"
            }
        }
        "PHISHING" {
            return @{
                Dataset = "PHISHING"
                ConfigDir = "phishing"
                ResultDir = "PHISHING"
            }
        }
        default {
            throw "Unsupported dataset '$Dataset'. Supported values: CIFAR10, NUSWIDE, IEEE_CIS_FRAUD, PHISHING."
        }
    }
}

function New-ExperimentFromSpec {
    param(
        [hashtable] $Spec
    )

    $configPath = Resolve-ExistingPath `
        -Description "$($Spec.Dataset) stage3_test.json" `
        -Candidates @(
            "configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/$($Spec.ConfigDir)/stage3_test.json",
            "dataset/configs/avguard/fourclient_full_avguard_fixed_lambda_anchor/$($Spec.ConfigDir)/stage3_test.json"
        )

    return @{
        Dataset = $Spec.Dataset
        ConfigPath = $configPath
        CheckpointPath = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/$($Spec.ResultDir)/stage2/best_checkpoint.pth.tar"
        OutputRoot = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/$($Spec.ResultDir)/stage3_theta_supp_sweep"
    }
}

function Invoke-ThetaSuppExperiment {
    param(
        [string] $Dataset,
        [string] $ConfigPath,
        [string] $CheckpointPath,
        [string] $OutputRoot,
        [double] $ThetaSupp
    )

    $thetaLabel = ConvertTo-ThetaLabel -ThetaSupp $ThetaSupp
    $outputDir = Join-Path $OutputRoot $thetaLabel
    $logPath = Join-Path $outputDir "experiment.log"
    $thetaText = "{0:N2}" -f $ThetaSupp

    Write-Host "Dataset: $Dataset"
    Write-Host "theta_supp: $thetaText"
    Write-Host "Config path: $ConfigPath"
    Write-Host "Checkpoint path: $CheckpointPath"
    Write-Host "Output directory: $outputDir"

    if ((-not $Rerun) -and (Test-CompletedExperiment -LogPath $logPath)) {
        Write-Host "Skip existing completed run: $Dataset $thetaLabel"
        return
    }

    $commandArgs = @(
        "main.py",
        "--config", $ConfigPath,
        "--mode", "test",
        "--theta_supp", $thetaText,
        "--test_checkpoint", $CheckpointPath,
        "--results_dir", $outputDir
    )

    if ($DryRun) {
        Write-Host "$Python $($commandArgs -join ' ')"
        return
    }

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    "$Python $($commandArgs -join ' ')" | Set-Content -Path (Join-Path $outputDir "run_command.txt") -Encoding UTF8

    & $Python @commandArgs

    if ($LASTEXITCODE -ne 0) {
        throw "Failed dataset=$Dataset theta_supp=$thetaText with exit code $LASTEXITCODE"
    }
}

function Read-ThetaSuppMetrics {
    param(
        [string] $Dataset,
        [double] $ThetaSupp,
        [string] $ResultsDir
    )

    $logPath = Join-Path $ResultsDir "experiment.log"
    $row = [ordered]@{
        dataset = $Dataset
        theta_supp = "{0:N2}" -f $ThetaSupp
        status = "missing"
        clean_acc = ""
        asr = ""
        rac = ""
        stage3_final_acc = ""
        stage3_final_asr = ""
        detection_recall = ""
        detection_precision = ""
        detection_f1 = ""
        false_positive_rate = ""
        correction_rate = ""
        clean_suspicious = ""
        clean_total = ""
        poison_valid_suspicious = ""
        poison_valid_total = ""
        clean_support_avg = ""
        clean_suspicious_support_avg = ""
        clean_correction_support_avg = ""
        poison_valid_support_avg = ""
        poison_valid_suspicious_support_avg = ""
        poison_valid_correction_support_avg = ""
        attack_success_support_avg = ""
        attack_success_suspicious_support_avg = ""
        results_dir = $ResultsDir
        log_path = $logPath
    }

    if (-not (Test-Path $logPath)) {
        return [pscustomobject] $row
    }

    $number = "[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
    foreach ($line in Get-Content $logPath) {
        if ($line -match "=> Test Epoch: .* clean acc: ($number), Top-\d+: ($number), ASR: ($number), RAC: ($number), .* stage3 final accuracy: ($number), .* stage3 final asr: ($number),") {
            $row.status = "completed"
            $row.clean_acc = $Matches[1]
            $row.asr = $Matches[3]
            $row.rac = $Matches[4]
            $row.stage3_final_acc = $Matches[5]
            $row.stage3_final_asr = $Matches[6]
        }
        elseif ($line -match "=> Stage 3 Detection Summary: recall: ($number), precision: ($number), f1: ($number), false positive rate: ($number), correction rate: ($number)") {
            $row.detection_recall = $Matches[1]
            $row.detection_precision = $Matches[2]
            $row.detection_f1 = $Matches[3]
            $row.false_positive_rate = $Matches[4]
            $row.correction_rate = $Matches[5]
        }
        elseif ($line -match "=> Stage 3 Debug \(clean\): suspicious=(\d+)/(\d+),") {
            $row.clean_suspicious = $Matches[1]
            $row.clean_total = $Matches[2]
        }
        elseif ($line -match "=> Stage 3 Debug \(poison-valid\): suspicious=(\d+)/(\d+),") {
            $row.poison_valid_suspicious = $Matches[1]
            $row.poison_valid_total = $Matches[2]
        }
        elseif ($line -match "=> Stage 3 Effective Support Avg: clean\(all\)=($number), clean_suspicious\(false_positive\)=($number), clean_correction_applied=($number), poison_valid\(all\)=($number), poison_valid_suspicious=($number), poison_valid_correction_applied=($number), attack_success\(all\)=($number), attack_success_suspicious=($number)") {
            $row.clean_support_avg = $Matches[1]
            $row.clean_suspicious_support_avg = $Matches[2]
            $row.clean_correction_support_avg = $Matches[3]
            $row.poison_valid_support_avg = $Matches[4]
            $row.poison_valid_suspicious_support_avg = $Matches[5]
            $row.poison_valid_correction_support_avg = $Matches[6]
            $row.attack_success_support_avg = $Matches[7]
            $row.attack_success_suspicious_support_avg = $Matches[8]
        }
        elseif ($line -match "=> Stage 3 Effective Support Avg: clean_suspicious\(false_positive\)=($number), clean_correction_applied=($number), poison_valid\(all\)=($number), poison_valid_suspicious=($number), poison_valid_correction_applied=($number), attack_success\(all\)=($number), attack_success_suspicious=($number)") {
            $row.clean_suspicious_support_avg = $Matches[1]
            $row.clean_correction_support_avg = $Matches[2]
            $row.poison_valid_support_avg = $Matches[3]
            $row.poison_valid_suspicious_support_avg = $Matches[4]
            $row.poison_valid_correction_support_avg = $Matches[5]
            $row.attack_success_support_avg = $Matches[6]
            $row.attack_success_suspicious_support_avg = $Matches[7]
        }
    }

    return [pscustomobject] $row
}

$experiments = @()
foreach ($dataset in $Datasets) {
    $experiments += New-ExperimentFromSpec -Spec (Get-ExperimentSpec -Dataset $dataset)
}

$summaryRows = @()
foreach ($experiment in $experiments) {
    if (-not (Test-Path $experiment.CheckpointPath)) {
        throw "Missing checkpoint for dataset=$($experiment.Dataset): $($experiment.CheckpointPath)"
    }

    foreach ($thetaSupp in $ThetaSupps) {
        Invoke-ThetaSuppExperiment `
            -Dataset $experiment.Dataset `
            -ConfigPath $experiment.ConfigPath `
            -CheckpointPath $experiment.CheckpointPath `
            -OutputRoot $experiment.OutputRoot `
            -ThetaSupp $thetaSupp

        $thetaLabel = ConvertTo-ThetaLabel -ThetaSupp $thetaSupp
        $resultsDir = Join-Path $experiment.OutputRoot $thetaLabel
        $summaryRows += Read-ThetaSuppMetrics `
            -Dataset $experiment.Dataset `
            -ThetaSupp $thetaSupp `
            -ResultsDir $resultsDir
    }

    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $experiment.OutputRoot | Out-Null
        $summaryRows |
            Where-Object { $_.dataset -eq $experiment.Dataset } |
            Export-Csv -Path (Join-Path $experiment.OutputRoot "summary.csv") -NoTypeInformation -Encoding UTF8
    }
}

if (-not $DryRun) {
    $globalSummaryPath = "results/AVGuard/fourclient_full_avguard_fixed_lambda_anchor/stage3_theta_supp_sweep_summary.csv"
    $summaryRows | Export-Csv -Path $globalSummaryPath -NoTypeInformation -Encoding UTF8
    Write-Host "Wrote $globalSummaryPath"
}

Write-Host "Stage 3 theta_supp sweep script completed."
