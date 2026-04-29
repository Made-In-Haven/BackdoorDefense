import argparse
import copy
import json
import logging
import math
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from attack.runtime import create_attack_runtime
from dataset.dataset import CIFAR10_VFL, IEEE_CIS_FRAUD_VFL, NUSWIDE_VFL, PHISHING_VFL, UCIHAR_VFL
from defense.factory import load_defense_runtime_stats, normalize_defense_args, prepare_defense
from defense.anchor_utils import get_num_classes, get_stage1_dir
from dataset.utils import (
    get_attacker_feature_indices,
    describe_nuswide_client_partition,
    get_nuswide_local_output_dims,
    validate_nuswide_attack_client_num,
    validate_nuswide_client_num,
    validate_nuswide_total_dim,
)
from models.CIFAR10_models import GlobalModelForCifar10, LocalModelForCifar10
from models.IEEE_CIS_FRAUD_models import GlobalModelForIEEECISFRAUD, LocalModelForIEEECISFRAUD
from models.NUSWIDE_models import GlobalModelForNUSWIDE, LocalModelForNUSWIDE
from models.PHISHING_models import GlobalModelForPHISHING, LocalModelForPHISHING
from models.UCIHAR_models import GlobalModelForUCIHAR, LocalModelForUCIHAR
from utils.trainer import Trainer
from utils.utils import (
    LFBA_POISON_SOURCE_GRADIENT,
    canonicalize_dataset_name,
    load_torch_artifact,
    normalize_lfba_poison_source,
    raise_dataset_exception,
    set_seed,
)


def create_logger(results_dir):
    logger = logging.getLogger(__name__)
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    os.makedirs(results_dir, exist_ok=True)
    file_handler = logging.FileHandler(os.path.join(results_dir, "experiment.log"), encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def build_datasets(args, logger):
    logger.info("=> Preparing data...")
    if args.dataset == "CIFAR10":
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        transform_test = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
        train_data = CIFAR10_VFL(root=args.data_dir, train=True, download=True, transform=transform_train)
        test_data = CIFAR10_VFL(root=args.data_dir, train=False, download=True, transform=transform_test)
    elif args.dataset == "UCIHAR":
        train_data = UCIHAR_VFL(root=args.data_dir, train=True, transforms=None)
        test_data = UCIHAR_VFL(root=args.data_dir, train=False, transforms=None)
    elif args.dataset == "PHISHING":
        train_data = PHISHING_VFL(root=args.data_dir, train=True, transforms=None)
        test_data = PHISHING_VFL(root=args.data_dir, train=False, transforms=None)
        args.phishing_input_dim = train_data.data.shape[1]
        logger.info("=> PHISHING file: %s", train_data.source_path)
        logger.info(
            "=> PHISHING feature dim: %s, train samples: %s, test samples: %s",
            args.phishing_input_dim,
            len(train_data),
            len(test_data),
        )
    elif args.dataset == "IEEE_CIS_FRAUD":
        train_data = IEEE_CIS_FRAUD_VFL(root=args.data_dir, train=True, transforms=None)
        test_data = IEEE_CIS_FRAUD_VFL(root=args.data_dir, train=False, transforms=None)
        args.ieee_cis_fraud_input_dim = train_data.data.shape[1]
        args.ieeecis_input_dim = args.ieee_cis_fraud_input_dim
        args.ieeecis_num_classes = 2
        logger.info("=> IEEE-CIS-Fraud directory: %s", train_data.source_dir)
        logger.info(
            "=> IEEE-CIS-Fraud feature dim: %s, train samples: %s, test samples: %s",
            args.ieee_cis_fraud_input_dim,
            len(train_data),
            len(test_data),
        )
    elif args.dataset == "NUSWIDE":
        selected_labels = ["buildings", "grass", "animal", "water", "person"]
        train_data = NUSWIDE_VFL(root=args.data_dir, selected_labels=selected_labels, train=True, transforms=None)
        test_data = NUSWIDE_VFL(root=args.data_dir, selected_labels=selected_labels, train=False, transforms=None)
        args.nuswide_total_dim = train_data.data.shape[1]
        validate_nuswide_total_dim(args.nuswide_total_dim)
        args.nuswide_local_output_dims = get_nuswide_local_output_dims(args.client_num)
        args.nuswide_client_partition = describe_nuswide_client_partition(args.client_num)
        logger.info(
            "=> NUSWIDE total feature dim: %s, client_num: %s, local output dims: %s",
            args.nuswide_total_dim,
            args.client_num,
            args.nuswide_local_output_dims,
        )
        logger.info("=> NUSWIDE fixed feature order: [CH | CM55 | CORR | EDH | WT | Tags1k]")
        for partition_description in args.nuswide_client_partition:
            logger.info("=> NUSWIDE split: %s", partition_description)
        logger.info("=> NUSWIDE attacker client: client%s", args.attack_client_num)
    else:
        raise_dataset_exception()

    test_data_asr = copy.deepcopy(test_data)
    return train_data, test_data, test_data_asr


def build_models(args, device):
    if args.dataset == "CIFAR10":
        model_list = [GlobalModelForCifar10(args)] + [LocalModelForCifar10(args) for _ in range(args.client_num)]
    elif args.dataset == "UCIHAR":
        model_list = [GlobalModelForUCIHAR(args)] + [
            LocalModelForUCIHAR(args, client_id) for client_id in range(args.client_num)
        ]
    elif args.dataset == "PHISHING":
        model_list = [GlobalModelForPHISHING(args)] + [
            LocalModelForPHISHING(args, client_id) for client_id in range(args.client_num)
        ]
    elif args.dataset == "IEEE_CIS_FRAUD":
        model_list = [GlobalModelForIEEECISFRAUD(args)] + [
            LocalModelForIEEECISFRAUD(args, client_id) for client_id in range(args.client_num)
        ]
    elif args.dataset == "NUSWIDE":
        model_list = [GlobalModelForNUSWIDE(args)] + [
            LocalModelForNUSWIDE(args, client_id) for client_id in range(args.client_num)
        ]
    else:
        raise_dataset_exception()

    model_list = [model.to(device) for model in model_list]
    optimizer_list = [torch.optim.Adam(model.parameters(), lr=args.lr) for model in model_list]
    criterion = nn.CrossEntropyLoss().to(device)
    return model_list, optimizer_list, criterion


def load_checkpoint_if_available(args, device, logger, model_list, optimizer_list):
    checkpoint = None
    checkpoint_path = args.test_checkpoint if args.mode == "test" and args.test_checkpoint else args.pretrained_checkpoint
    if not checkpoint_path:
        return checkpoint
    if not os.path.isfile(checkpoint_path):
        logger.info("=> No checkpoint found at '%s'", checkpoint_path)
        return checkpoint

    logger.info("=> Loading checkpoint '%s'", checkpoint_path)
    checkpoint = load_torch_artifact(
        checkpoint_path,
        map_location=device,
        logger=logger,
        description="checkpoint",
    )
    for model, state_dict in zip(model_list, checkpoint["state_dict"]):
        model.load_state_dict(state_dict)
    if args.mode != "test":
        optimizer_states = checkpoint.get("optimizer")
        if optimizer_states:
            for optimizer, optimizer_state in zip(optimizer_list, optimizer_states):
                optimizer.load_state_dict(optimizer_state)
        args.start_epoch = checkpoint.get("epoch", 0)
        restore_rng_state_if_available(checkpoint, logger)
    logger.info(
        "=> Loaded checkpoint '%s' (epoch %s, best clean accuracy: %.4f)",
        checkpoint_path,
        checkpoint.get("epoch", "n/a"),
        checkpoint.get("best_clean_acc", checkpoint.get("best_acc", 0.0)),
    )
    if args.mode != "test":
        logger.info(
            "=> Resume training from epoch %s/%s using checkpoint '%s'",
            args.start_epoch,
            args.epoch,
            checkpoint_path,
        )
    return checkpoint


def restore_rng_state_if_available(checkpoint, logger):
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict):
        return
    try:
        python_state = rng_state.get("python")
        numpy_state = rng_state.get("numpy")
        torch_state = rng_state.get("torch")
        cuda_states = rng_state.get("cuda")
        if python_state is not None:
            random.setstate(python_state)
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        if torch_state is not None:
            torch.random.set_rng_state(torch_state.cpu())
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])
        logger.info("=> Restored RNG state from checkpoint for deterministic resume")
    except Exception as exc:
        logger.info("=> Skipped RNG state restore because it could not be applied: %s", exc)


def restore_attack_runtime_state_if_available(checkpoint, attack_runtime, logger):
    if checkpoint is None or attack_runtime is None or not hasattr(attack_runtime, "load_checkpoint_state"):
        return
    attack_state = checkpoint.get("attack_state")
    if attack_state is None:
        logger.info("=> No serialized attack runtime state found in checkpoint; falling back to compatibility resume path")
        return
    attack_runtime.load_checkpoint_state(attack_state)


def _extract_class_label(label_value):
    if torch.is_tensor(label_value):
        return int(label_value.item())
    return int(label_value)


def resolve_attack_target_label(args, logger, train_data):
    if args.attack == "LFBA":
        configured_target_label = getattr(args, "target_label", None)
        if configured_target_label is not None:
            logger.info(
                "=> Ignoring configured target_label=%s for LFBA; the attack target label is determined by anchor_idx=%s",
                configured_target_label,
                args.anchor_idx,
            )
        if args.anchor_idx < 0 or args.anchor_idx >= len(train_data):
            raise ValueError(
                "LFBA requires anchor_idx to point to a valid training sample. Got anchor_idx={} with train size {}.".format(
                    args.anchor_idx,
                    len(train_data),
                )
            )
        args.attack_target_label = _extract_class_label(train_data.targets[args.anchor_idx])
        logger.info(
            "=> Resolved LFBA attack target label=%s from training anchor_idx=%s",
            args.attack_target_label,
            args.anchor_idx,
        )
        return

    configured_target_label = getattr(args, "target_label", None)
    if configured_target_label is None:
        configured_target_label = 1 if args.dataset in {"PHISHING", "IEEE_CIS_FRAUD"} else 3
    num_classes = get_num_classes(args.dataset)
    args.attack_target_label = min(int(configured_target_label), num_classes - 1)
    if args.attack_target_label != int(configured_target_label):
        logger.info(
            "=> Clamped attack target label from %s to %s because dataset '%s' has %s classes",
            configured_target_label,
            args.attack_target_label,
            args.dataset,
            num_classes,
        )


def select_trigger_dimensions(args, train_data, logger=None):
    if args.dataset == "CIFAR10":
        return []
    if args.dataset == "UCIHAR":
        ranges = get_attacker_feature_indices(args, train_data.data.shape[1])
    elif args.dataset == "PHISHING":
        ranges = get_attacker_feature_indices(args, train_data.data.shape[1])
    elif args.dataset == "IEEE_CIS_FRAUD":
        ranges = get_attacker_feature_indices(args, train_data.data.shape[1])
    elif args.dataset == "NUSWIDE":
        ranges = get_attacker_feature_indices(args, train_data.data.shape[1])
    else:
        raise_dataset_exception()
    available_feature_count = int(len(ranges))
    if available_feature_count <= 0:
        raise ValueError(
            "Attack client {} has no available features for dataset '{}' with client_num={}.".format(
                args.attack_client_num,
                args.dataset,
                args.client_num,
            )
        )

    requested_poison_dimensions = int(args.poison_dimensions)
    effective_poison_dimensions = min(requested_poison_dimensions, available_feature_count)
    if logger is not None and effective_poison_dimensions != requested_poison_dimensions:
        logger.info(
            "=> Clamped poison_dimensions from %s to %s because attacker client %s has only %s available features under client_num=%s",
            requested_poison_dimensions,
            effective_poison_dimensions,
            args.attack_client_num,
            available_feature_count,
            args.client_num,
        )
    return np.random.choice(ranges, effective_poison_dimensions, replace=False)


def build_loaders(args, train_data, test_data, test_data_asr):
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    test_asr_loader = torch.utils.data.DataLoader(dataset=test_data_asr, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, test_asr_loader


def normalize_args(args):
    requested_dataset_name = str(args.dataset).upper()
    args.original_dataset = requested_dataset_name
    args.dataset = canonicalize_dataset_name(requested_dataset_name)
    args.attack_target_label = None
    args.lfba_poison_source = normalize_lfba_poison_source(
        getattr(args, "lfba_poison_source", LFBA_POISON_SOURCE_GRADIENT)
    )
    if args.dataset == "NUSWIDE":
        validate_nuswide_client_num(args.client_num)
        validate_nuswide_attack_client_num(args.client_num, args.attack_client_num)
    if args.mode == "pretrain_vflip":
        raise ValueError(
            "The pretrain_vflip mode has been removed from this project. "
            "Please use mode='pretrain_anchor', 'train', or 'test'."
        )
    if args.mode not in {"train", "test", "pretrain_anchor"}:
        raise ValueError("Unsupported mode '{}'.".format(args.mode))
    args.disable_stage3_eval = bool(getattr(args, "disable_stage3_eval", False))
    if args.disable_stage3_eval and args.mode != "test":
        raise ValueError("--disable_stage3_eval is only supported when mode='test'.")
    normalize_defense_args(args)
    args.top_k = max(1, int(getattr(args, "top_k", 1)))
    args.anchor_ema_update_freq = max(1, int(getattr(args, "anchor_ema_update_freq", 1)))
    args.anchor_ema_momentum = min(max(float(getattr(args, "anchor_ema_momentum", 0.995)), 0.0), 1.0)
    args.lambda_anchor = min(max(float(getattr(args, "lambda_anchor", 0.1)), 0.0), 1.0)
    args.anchor_pretrain_early_stop = max(0, int(getattr(args, "anchor_pretrain_early_stop", 5)))
    args.anchor_pretrain_min_delta = max(0.0, float(getattr(args, "anchor_pretrain_min_delta", 1e-4)))
    args.enable_stage1 = bool(getattr(args, "enable_stage1", True))
    args.gamma = max(float(getattr(args, "gamma", 2.0)), 1e-8)
    args.theta_supp = max(float(getattr(args, "theta_supp", 0.15)), 0.0)
    args.stage3_required_support_count = max(1, int(getattr(args, "stage3_required_support_count", 2)))
    args.stage3_enable_joint_weighted_voting = bool(
        getattr(args, "stage3_enable_joint_weighted_voting", True)
    )
    args.stage3_enable_static_reliability = bool(
        getattr(args, "stage3_enable_static_reliability", True)
    )
    args.enable_conservative_correction = bool(getattr(args, "enable_conservative_correction", False))
    args.tau_corr = max(float(getattr(args, "tau_corr", 0.0)), 0.0)
    args.anchor_margin_auto_adjusted = False
    args.stage3_debug_batches = max(0, int(getattr(args, "stage3_debug_batches", 0)))
    args.stage3_debug_max_samples = max(1, int(getattr(args, "stage3_debug_max_samples", 8)))
    args.requested_anchor_margin = args.anchor_margin
    if (
        args.dataset == "CIFAR10"
        and args.client_num > 2
        and args.anchor_margin > 0
    ):
        # Positive ArcFace margin is unstable for narrow CIFAR10 image slices in multi-client stage 1 pretraining.
        args.anchor_margin = 0.0
        args.anchor_margin_auto_adjusted = True

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.results_dir:
        args.results_dir = os.path.join("results", args.dataset, timestamp)
    if not args.anchor_stage1_dir:
        args.anchor_stage1_dir = get_stage1_dir(args)
    if not args.anchor_bank_path:
        args.anchor_bank_path = os.path.join(args.results_dir, "anchor_bank.pt")
    if args.resume_latest and not args.pretrained_checkpoint and args.mode in {"train", "pretrain_anchor"}:
        args.pretrained_checkpoint = os.path.join(args.results_dir, "latest_checkpoint.pth.tar")


def resolve_runtime_device(args):
    if not torch.cuda.is_available():
        return torch.device("cpu"), "CUDA is unavailable, falling back to CPU"

    requested_device = args.device
    available_gpu_count = torch.cuda.device_count()
    if requested_device < 0 or requested_device >= available_gpu_count:
        return (
            torch.device("cuda:0"),
            "Requested GPU index {} is unavailable, falling back to cuda:0".format(requested_device),
        )

    gpu_name = torch.cuda.get_device_name(requested_device)
    return torch.device(f"cuda:{requested_device}"), "Using GPU cuda:{} ({})".format(requested_device, gpu_name)


def main(args):
    normalize_args(args)
    device, device_message = resolve_runtime_device(args)
    logger = create_logger(args.results_dir)
    logger.info(args)
    if getattr(args, "original_dataset", args.dataset) != args.dataset:
        logger.info(
            "=> Canonicalized dataset name from '%s' to '%s'; legacy NUS-WIDE aliases now share the unified NUSWIDE entry",
            args.original_dataset,
            args.dataset,
        )
    logger.info("=> Defense scheme: %s", getattr(args, "defense_scheme", "AVGuard"))
    if args.attack == "LFBA":
        logger.info("=> LFBA poison-set source: %s", args.lfba_poison_source)
    if getattr(args, "anchor_margin_auto_adjusted", False):
        logger.info(
            "=> Adjusted anchor_margin from %.4f to %.4f for CIFAR10 anchor stage1 with %s clients to avoid ArcFace collapse on narrow image slices",
            args.requested_anchor_margin,
            args.anchor_margin,
            args.client_num,
        )
    logger.info("=> Python executable: %s", sys.executable)
    logger.info(
        "=> Torch version: %s, compiled CUDA: %s, cuda available: %s, visible GPU count: %s",
        torch.__version__,
        torch.version.cuda,
        torch.cuda.is_available(),
        torch.cuda.device_count(),
    )
    logger.info("=> %s", device_message)

    train_data, test_data, test_data_asr = build_datasets(args, logger)
    resolve_attack_target_label(args, logger, train_data)
    clean_train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    clean_test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    trigger_dimensions = select_trigger_dimensions(args, train_data, logger=logger)

    model_list, optimizer_list, criterion = build_models(args, device)
    checkpoint = load_checkpoint_if_available(args, device, logger, model_list, optimizer_list)
    if args.disable_stage3_eval:
        logger.info(
            "=> Stage 3 evaluation is disabled; reporting raw VFL clean accuracy and raw ASR from the loaded checkpoint"
        )
        defense_runtime = None
    else:
        defense_runtime = prepare_defense(
            args,
            device,
            logger,
            model_list,
            checkpoint,
            clean_train_loader,
            clean_test_loader,
            trigger_dimensions,
        )
        load_defense_runtime_stats(args, logger, defense_runtime)

    if args.mode == "pretrain_anchor":
        return
    if defense_runtime is None and not args.disable_stage3_eval:
        raise RuntimeError(
            "AVGuard requires anchor artifacts for '{}' mode. "
            "Please provide a stage1/stage2 checkpoint or a valid anchor_bank_path.".format(args.mode)
        )

    attack_runtime = create_attack_runtime(args=args, logger=logger, trigger_dimensions=trigger_dimensions, device=device)
    attack_runtime.apply_initial_dataset_poisoning(train_data, test_data_asr)
    train_loader, test_loader, test_asr_loader = build_loaders(args, train_data, test_data, test_data_asr)
    attack_runtime.attach_loaders(train_loader, test_asr_loader)
    restore_attack_runtime_state_if_available(checkpoint, attack_runtime, logger)

    trainer = Trainer(
        device=device,
        model_list=model_list,
        optimizer_list=optimizer_list,
        criterion=criterion,
        train_loader=train_loader,
        test_loader=test_loader,
        test_asr_loader=test_asr_loader,
        trigger_dimensions=trigger_dimensions,
        logger=logger,
        args=args,
        checkpoint=checkpoint,
        defense_runtime=defense_runtime,
        attack_runtime=attack_runtime,
    )

    if args.mode == "test":
        trainer.test((checkpoint.get("epoch", 1) - 1) if checkpoint else 0)
        return

    trainer.train()


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="optional JSON config file path")
    parser.add_argument("--data_dir", default="dataset/", help="data directory")
    parser.add_argument("--dataset", default="NUSWIDE", help="name of dataset")
    parser.add_argument("--device", default=0, type=int, help="GPU number")
    parser.add_argument("--results_dir", default="", help="directory used to save logs and checkpoints")
    parser.add_argument("--seed", default=100, type=int, help="random seed")
    parser.add_argument("--epoch", default=100, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="training batch size")
    parser.add_argument("--client_num", default=2, type=int, help="number of clients")
    parser.add_argument(
        "--pretrained_checkpoint",
        default=None,
        help="checkpoint used to resume training; latest_checkpoint.pth.tar is recommended for interrupted runs",
    )
    parser.add_argument("--test_checkpoint", default=None, help="checkpoint used for evaluation")
    parser.add_argument(
        "--resume_latest",
        default=False,
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="auto-resume from results_dir/latest_checkpoint.pth.tar when training",
    )
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting epoch index")
    parser.add_argument("--print_steps", default=10, type=int, help="logging interval")
    parser.add_argument("--early_stop", default=20, type=int, help="early stopping patience; set 0 to disable")
    parser.add_argument("--attack", default=None, help="attack method")
    parser.add_argument(
        "--target_label",
        default=None,
        type=int,
        help="optional target label for label-aware attacks; LFBA derives its target label from anchor_idx",
    )
    parser.add_argument("--poison_rate", default=0.1, type=float, help="ratio of poison samples")
    parser.add_argument("--poison_dimensions", default=5, type=int, help="number of poisoned feature dimensions")
    parser.add_argument("--trigger_feature_clip", default=1, type=float, help="feature trigger clip ratio")
    parser.add_argument("--attack_client_num", default=1, type=int, help="attacker client index")
    parser.add_argument("--feature_extractor", default="", help="reserved legacy argument")
    parser.add_argument("--select_rate", default=1, type=float, help="ratio of switched samples for LFBA")
    parser.add_argument(
        "--lfba_poison_source",
        default=LFBA_POISON_SOURCE_GRADIENT,
        help="LFBA poison-set source: 'gradient' for label-free inference or 'oracle_label' for same-label supervision",
    )
    parser.add_argument("--random_select", action="store_true")
    parser.add_argument("--poison_all", action="store_true")
    parser.add_argument("--anchor_idx", default=33930, type=int)
    parser.add_argument("--pretrain_stage", default=0, type=int, help="LFBA warmup stage")

    parser.add_argument("--mode", default="train", choices=["train", "test", "pretrain_anchor"])
    parser.add_argument("--defense", default="anchor", choices=["anchor"])
    parser.add_argument(
        "--defense_scheme",
        default="",
        help="named defense scheme used for experiment tracking, e.g. AVGuard",
    )
    parser.add_argument(
        "--enable_stage1",
        default=True,
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="whether to enable Stage 1 anchor pretraining/artifact reuse; when disabled, anchors are randomly initialized",
    )
    parser.add_argument(
        "--force_stage1_retrain",
        action="store_true",
        help="force rerunning stage 1 instead of reusing a saved stage1 artifact",
    )
    parser.add_argument(
        "--enable_anchor_loss",
        default=True,
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="whether to add anchor loss during stage 2 training",
    )
    parser.add_argument(
        "--lambda_anchor",
        default=0.1,
        type=float,
        help="stage2 interpolation weight in [0, 1] for total loss=(1-lambda_anchor)*CE + lambda_anchor*anchor_loss",
    )
    parser.add_argument(
        "--lambda_stage1_anchor",
        default=0.3,
        type=float,
        help="weight of anchor loss during stage1 anchor pretraining",
    )
    parser.add_argument("--anchor_pretrain_epochs", default=5, type=int, help="anchor pretraining epochs")
    parser.add_argument(
        "--anchor_pretrain_early_stop",
        default=5,
        type=int,
        help="stage1 early-stop patience on validation clean_acc; set 0 to disable",
    )
    parser.add_argument(
        "--anchor_pretrain_min_delta",
        default=1e-4,
        type=float,
        help="minimum clean_acc improvement required to reset stage1 early-stop patience",
    )
    parser.add_argument("--anchor_scale", default=16.0, type=float, help="ArcFace scale")
    parser.add_argument("--anchor_margin", default=0.2, type=float, help="ArcFace angular margin")
    parser.add_argument(
        "--anchor_bank_path",
        default="",
        help="path for loading or saving the serialized anchor defense artifact",
    )
    parser.add_argument(
        "--anchor_stage1_dir",
        default="",
        help="directory for saving/loading stage1 passive local models and anchor artifacts",
    )
    parser.add_argument(
        "--anchor_ema_momentum",
        default=0.995,
        type=float,
        help="EMA momentum used to calibrate the stage2 anchor bank",
    )
    parser.add_argument(
        "--anchor_ema_update_freq",
        default=1,
        type=int,
        help="number of stage2 epochs between anchor-bank EMA updates",
    )
    parser.add_argument(
        "--stage3_enable_joint_weighted_voting",
        default=True,
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="whether Stage 3 uses joint weighted voting; when disabled, majority voting is used directly",
    )
    parser.add_argument(
        "--stage3_enable_static_reliability",
        default=True,
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="whether Stage 3 weighting uses static client reliability r_i from Stage 1 metrics",
    )
    parser.add_argument(
        "--disable_stage3_eval",
        default=False,
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="test the loaded checkpoint without Stage 3 detection/correction and report raw clean acc/ASR",
    )
    parser.add_argument(
        "--gamma",
        default=2.0,
        type=float,
        help="stage3 gamma in the joint static-reliability and dynamic-confidence voting formula",
    )
    parser.add_argument(
        "--theta_supp",
        default=0.15,
        type=float,
        help="minimum local anchor confidence required for a client to count as valid support of the global prediction",
    )
    parser.add_argument(
        "--stage3_required_support_count",
        default=2,
        type=int,
        help="minimum number of valid supporting clients required to keep the global prediction non-suspicious",
    )
    parser.add_argument(
        "--enable_conservative_correction",
        default=False,
        type=lambda value: str(value).lower() in {"1", "true", "yes", "on"},
        help="whether to require the weighted-vote correction margin to exceed tau_corr before applying Stage 3 correction",
    )
    parser.add_argument(
        "--tau_corr",
        default=0.0,
        type=float,
        help="minimum Stage 3 weighted-vote correction margin required when conservative correction is enabled",
    )
    parser.add_argument(
        "--top_k",
        "--k",
        dest="top_k",
        default=4,
        type=int,
        help="top-k accuracy used in unified evaluation metrics",
    )
    parser.add_argument(
        "--stage3_debug_batches",
        default=0,
        type=int,
        help="number of Stage 3 evaluation batches to dump as detailed debug logs; 0 disables batch-level dumps",
    )
    parser.add_argument(
        "--stage3_debug_max_samples",
        default=8,
        type=int,
        help="maximum number of samples to print per Stage 3 debug batch",
    )
    return parser


def strip_json_line_comments(raw_text):
    # Allow config files to use inline // comments while keeping quoted strings intact.
    result = []
    in_string = False
    escape = False
    index = 0
    while index < len(raw_text):
        char = raw_text[index]
        next_char = raw_text[index + 1] if index + 1 < len(raw_text) else ""

        if char == '"' and not escape:
            in_string = not in_string

        if not in_string and char == "/" and next_char == "/":
            while index < len(raw_text) and raw_text[index] != "\n":
                index += 1
            continue

        result.append(char)
        escape = char == "\\" and not escape
        if char != "\\":
            escape = False
        index += 1
    return "".join(result)


def load_config_file(config_path):
    # Use utf-8-sig so Windows-saved JSON config files with BOM can still be parsed.
    with open(config_path, "r", encoding="utf-8-sig") as config_file:
        return json.loads(strip_json_line_comments(config_file.read()))


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = build_parser()
    if config_args.config:
        parser.set_defaults(**load_config_file(config_args.config))

    return parser.parse_args(remaining_argv)


if __name__ == "__main__":
    parsed_args = parse_args()
    set_seed(parsed_args.seed)
    main(parsed_args)
