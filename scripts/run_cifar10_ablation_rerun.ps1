$ErrorActionPreference = "Stop"

if (-not (Test-Path "main.py")) {
    throw "Please run this script from the project root directory."
}

function Invoke-Cifar10AblationRerun {
    param(
        [string] $Category,
        [string] $Dataset,
        [string] $ConfigPath,
        [string] $OutputDir
    )

    Write-Host "Experiment category: $Category"
    Write-Host "Dataset: $Dataset"
    Write-Host "Config path: $ConfigPath"
    Write-Host "Output directory: $OutputDir"

    python main.py --config "$ConfigPath"

    if ($LASTEXITCODE -ne 0) {
        throw "Failed category=$Category dataset=$Dataset config=$ConfigPath with exit code $LASTEXITCODE"
    }
}

$experiments = @(
    @{
        Category = "lambda_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/stage1.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/stage1"
    },
    @{
        Category = "lambda_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.25/stage2.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.25/stage2"
    },
    @{
        Category = "lambda_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.25/stage3_test.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.25/stage3"
    },
    @{
        Category = "lambda_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.5/stage2.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.5/stage2"
    },
    @{
        Category = "lambda_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.5/stage3_test.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.5/stage3"
    },
    @{
        Category = "lambda_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.75/stage2.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.75/stage2"
    },
    @{
        Category = "lambda_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.75/stage3_test.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_anchor_sensitivity/lambda_anchor_0.75/stage3"
    },
    @{
        Category = "lambda_stage1_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.3/stage1.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.3/stage1"
    },
    @{
        Category = "lambda_stage1_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.3/stage2.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.3/stage2"
    },
    @{
        Category = "lambda_stage1_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.3/stage3_test.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.3/stage3"
    },
    @{
        Category = "lambda_stage1_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.5/stage1.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.5/stage1"
    },
    @{
        Category = "lambda_stage1_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.5/stage2.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.5/stage2"
    },
    @{
        Category = "lambda_stage1_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.5/stage3_test.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.5/stage3"
    },
    @{
        Category = "lambda_stage1_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.7/stage1.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.7/stage1"
    },
    @{
        Category = "lambda_stage1_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.7/stage2.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.7/stage2"
    },
    @{
        Category = "lambda_stage1_anchor_sensitivity"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.7/stage3_test.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/lambda_stage1_anchor_sensitivity/lambda_0.7/stage3"
    },
    @{
        Category = "stage1_ablation"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/stage1_ablation/pretrained_anchor/stage1.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/stage1_ablation/pretrained_anchor/stage1"
    },
    @{
        Category = "stage1_ablation"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/stage1_ablation/pretrained_anchor/stage2.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/stage1_ablation/pretrained_anchor/stage2"
    },
    @{
        Category = "stage1_ablation"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/stage1_ablation/pretrained_anchor/stage3_test.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/stage1_ablation/pretrained_anchor/stage3"
    },
    @{
        Category = "stage1_ablation"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/stage1_ablation/random_anchor/stage2.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/stage1_ablation/random_anchor/stage2"
    },
    @{
        Category = "stage1_ablation"
        Dataset = "CIFAR10"
        ConfigPath = "configs/avguard/cifar10_ablation_rerun_scale16/stage1_ablation/random_anchor/stage3_test.json"
        OutputDir = "./results/AVGuard/cifar10_ablation_rerun_scale16/stage1_ablation/random_anchor/stage3"
    }
)

foreach ($experiment in $experiments) {
    Invoke-Cifar10AblationRerun `
        -Category $experiment.Category `
        -Dataset $experiment.Dataset `
        -ConfigPath $experiment.ConfigPath `
        -OutputDir $experiment.OutputDir
}

Write-Host "CIFAR10 ablation rerun script completed."
