import argparse
import copy
import json
import logging
import math
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms

from attack.runtime import create_attack_runtime
from dataset.dataset import CIFAR10_VFL, NUSWIDE_VFL, PHISHING_VFL, UCIHAR_VFL
from defense.factory import load_defense_runtime_stats, normalize_defense_args, prepare_defense
from defense.anchor_utils import get_num_classes, get_stage1_dir
from dataset.utils import get_attacker_feature_slice
from models.CIFAR10_models import GlobalModelForCifar10, LocalModelForCifar10
from models.NUSWIDE_models import GlobalModelForNUSWIDE, LocalModelForNUSWIDE
from models.PHISHING_models import GlobalModelForPHISHING, LocalModelForPHISHING
from models.UCIHAR_models import GlobalModelForUCIHAR, LocalModelForUCIHAR
from utils.trainer import Trainer
from utils.utils import raise_dataset_exception, set_seed


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
    elif args.dataset == "NUSWIDE":
        selected_labels = ["buildings", "grass", "animal", "water", "person"]
        train_data = NUSWIDE_VFL(root=args.data_dir, selected_labels=selected_labels, train=True, transforms=None)
        test_data = NUSWIDE_VFL(root=args.data_dir, selected_labels=selected_labels, train=False, transforms=None)
        args.nuswide_total_dim = train_data.data.shape[1]
        args.nuswide_local_output_dim = getattr(args, "nuswide_local_output_dim", 32)
        logger.info(
            "=> NUSWIDE total feature dim: %s, client_num: %s",
            args.nuswide_total_dim,
            args.client_num,
        )
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
    checkpoint = torch.load(checkpoint_path, map_location=device)
    for model, state_dict in zip(model_list, checkpoint["state_dict"]):
        model.load_state_dict(state_dict)
    if args.mode != "test":
        for optimizer, optimizer_state in zip(optimizer_list, checkpoint["optimizer"]):
            optimizer.load_state_dict(optimizer_state)
        args.start_epoch = checkpoint.get("epoch", 0)
    logger.info(
        "=> Loaded checkpoint '%s' (epoch %s, best accuracy: %.4f)",
        checkpoint_path,
        checkpoint.get("epoch", "n/a"),
        checkpoint.get("best_acc", 0.0),
    )
    checkpoint_target_label = checkpoint.get("target_label")
    if checkpoint_target_label is not None:
        args.target_label = int(checkpoint_target_label)
        logger.info("=> Restored target_label=%s from checkpoint", args.target_label)
    return checkpoint


def maybe_resolve_lfba_target_label(args, logger, checkpoint, train_data):
    if args.attack != "LFBA":
        return

    if checkpoint and checkpoint.get("target_label") is not None:
        return

    if args.anchor_idx < 0 or args.anchor_idx >= len(train_data):
        logger.info(
            "=> Unable to infer LFBA target label from anchor_idx=%s because it is outside the training split",
            args.anchor_idx,
        )
        return

    inferred_target_label = train_data.targets[args.anchor_idx]
    if torch.is_tensor(inferred_target_label):
        inferred_target_label = inferred_target_label.item()
    args.target_label = int(inferred_target_label)
    logger.info("=> Inferred LFBA target_label=%s from training anchor_idx=%s", args.target_label, args.anchor_idx)


def select_trigger_dimensions(args, train_data):
    if args.dataset == "CIFAR10":
        return []
    if args.dataset == "UCIHAR":
        attacker_start, attacker_end = get_attacker_feature_slice(args, train_data.data.shape[1])
        ranges = range(attacker_start, attacker_end)
    elif args.dataset == "PHISHING":
        attacker_start, attacker_end = get_attacker_feature_slice(args, train_data.data.shape[1])
        ranges = range(attacker_start, attacker_end)
    elif args.dataset == "NUSWIDE":
        if args.client_num == 2:
            ranges = range(634, 1634)
        else:
            attacker_start, attacker_end = get_attacker_feature_slice(args, train_data.data.shape[1])
            ranges = range(attacker_start, attacker_end)
    else:
        raise_dataset_exception()
    return np.random.choice(ranges, args.poison_dimensions, replace=False)


def build_loaders(args, train_data, test_data, test_data_asr):
    train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    test_asr_loader = torch.utils.data.DataLoader(dataset=test_data_asr, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, test_asr_loader


def normalize_args(args):
    args.dataset = args.dataset.upper()
    normalize_defense_args(args)
    if args.dataset == "PHISHING":
        args.target_label = 1 if args.target_label is None else args.target_label
    else:
        args.target_label = 3 if args.target_label is None else args.target_label

    args.anchor_margin_auto_adjusted = False
    args.requested_anchor_margin = args.anchor_margin
    if (
        args.dataset == "CIFAR10"
        and args.defense == "anchor"
        and args.client_num > 2
        and args.anchor_margin > 0
    ):
        # Positive ArcFace margin is unstable for narrow CIFAR10 image slices in multi-client stage 1 pretraining.
        args.anchor_margin = 0.0
        args.anchor_margin_auto_adjusted = True

    num_classes = get_num_classes(args.dataset)
    if args.target_label >= num_classes:
        args.target_label = num_classes - 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.results_dir:
        args.results_dir = os.path.join("results", args.dataset, timestamp)
    if args.defense == "anchor" and not args.anchor_stage1_dir:
        args.anchor_stage1_dir = get_stage1_dir(args)
    if args.defense == "anchor" and not args.anchor_bank_path:
        args.anchor_bank_path = os.path.join(args.results_dir, "anchor_bank.pt")


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
    logger.info("=> Defense scheme: %s", getattr(args, "defense_scheme", "none"))
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
    clean_train_loader = torch.utils.data.DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True)
    clean_test_loader = torch.utils.data.DataLoader(dataset=test_data, batch_size=args.batch_size, shuffle=False)
    trigger_dimensions = select_trigger_dimensions(args, train_data)

    model_list, optimizer_list, criterion = build_models(args, device)
    checkpoint = load_checkpoint_if_available(args, device, logger, model_list, optimizer_list)
    maybe_resolve_lfba_target_label(args, logger, checkpoint, train_data)
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

    attack_runtime = create_attack_runtime(args=args, logger=logger, trigger_dimensions=trigger_dimensions, device=device)
    attack_runtime.apply_initial_dataset_poisoning(train_data, test_data_asr)
    train_loader, test_loader, test_asr_loader = build_loaders(args, train_data, test_data, test_data_asr)
    attack_runtime.attach_loaders(train_loader, test_asr_loader)

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
        anchor_defense=defense_runtime,
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
    parser.add_argument("--pretrained_checkpoint", default=None, help="checkpoint used to resume training")
    parser.add_argument("--test_checkpoint", default=None, help="checkpoint used for evaluation")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting epoch index")
    parser.add_argument("--print_steps", default=10, type=int, help="logging interval")
    parser.add_argument("--early_stop", default=20, type=int, help="early stopping patience")
    parser.add_argument("--attack", default=None, help="attack method")
    parser.add_argument("--target_label", default=None, type=int, help="target label for the backdoor")
    parser.add_argument("--poison_rate", default=0.1, type=float, help="ratio of poison samples")
    parser.add_argument("--poison_dimensions", default=5, type=int, help="number of poisoned feature dimensions")
    parser.add_argument("--trigger_feature_clip", default=1, type=float, help="feature trigger clip ratio")
    parser.add_argument("--attack_client_num", default=1, type=int, help="attacker client index")
    parser.add_argument("--feature_extractor", default="", help="reserved legacy argument")
    parser.add_argument("--select_rate", default=1, type=float, help="ratio of switched samples for LFBA")
    parser.add_argument("--random_select", action="store_true")
    parser.add_argument("--select_replace", action="store_true")
    parser.add_argument("--poison_all", action="store_true")
    parser.add_argument("--anchor_idx", default=33930, type=int)
    parser.add_argument("--pretrain_stage", default=0, type=int, help="LFBA warmup stage")

    parser.add_argument("--mode", default="train", choices=["train", "test", "pretrain_anchor"])
    parser.add_argument("--defense", default="none", choices=["none", "anchor"])
    parser.add_argument(
        "--defense_scheme",
        default="",
        help="named defense scheme used for experiment tracking, e.g. AVGuard",
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
    parser.add_argument("--lambda_anchor", default=0.1, type=float, help="weight of anchor constraint loss")
    parser.add_argument(
        "--lambda_stage1_anchor",
        default=0.3,
        type=float,
        help="weight of anchor loss during stage1 anchor pretraining",
    )
    parser.add_argument("--anchor_pretrain_epochs", default=5, type=int, help="anchor pretraining epochs")
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
        "--detect_threshold",
        default=2.0,
        type=float,
        help="stage3 multiplicative threshold gamma",
    )
    parser.add_argument(
        "--detect_threshold_cap",
        default=1.5,
        type=float,
        help="stage3 distance-threshold upper cap; final threshold is min(gamma * client_anchor_loss, cap)",
    )
    parser.add_argument(
        "--majority_ratio",
        default=0.5,
        type=float,
        help="minimum inconsistent-client ratio required to flag a sample",
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
