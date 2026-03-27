import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from dataset.dataset import CIFAR10_VFL, NUSWIDE_VFL, PHISHING_VFL, UCIHAR_VFL
from models.CIFAR10_models import GlobalModelForCifar10, LocalModelForCifar10
from models.NUSWIDE_models import GlobalModelForNUSWIDE, LocalModelForNUSWIDE
from models.PHISHING_models import GlobalModelForPHISHING, LocalModelForPHISHING
from models.UCIHAR_models import GlobalModelForUCIHAR, LocalModelForUCIHAR
from standalone_lfba_vfl.attack_core import attack_lfba_test, select_trigger_dimensions
from standalone_lfba_vfl.trainer import PlainVFLTrainer
from utils.utils import raise_dataset_exception, set_seed


def create_logger(results_dir):
    logger = logging.getLogger("standalone_lfba_vfl")
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
    elif args.dataset == "NUSWIDE":
        selected_labels = ["buildings", "grass", "animal", "water", "person"]
        train_data = NUSWIDE_VFL(root=args.data_dir, selected_labels=selected_labels, train=True, transforms=None)
        test_data = NUSWIDE_VFL(root=args.data_dir, selected_labels=selected_labels, train=False, transforms=None)
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


def build_loaders(args, train_data, test_data, test_data_asr):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False)
    test_asr_loader = torch.utils.data.DataLoader(test_data_asr, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader, test_asr_loader


def normalize_args(args):
    args.dataset = args.dataset.upper()
    if args.dataset == "PHISHING":
        args.target_label = 1 if args.target_label is None else args.target_label
    elif args.dataset in {"CIFAR10", "UCIHAR", "NUSWIDE"}:
        args.target_label = 3 if args.target_label is None else args.target_label
    else:
        raise_dataset_exception()

    if args.dataset == "NUSWIDE" and args.client_num != 2:
        raise ValueError("NUSWIDE currently only supports client_num=2 in standalone_lfba_vfl.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if not args.results_dir:
        args.results_dir = os.path.join("results", "standalone_lfba_vfl", args.dataset, timestamp)


def resolve_runtime_device(args):
    if not torch.cuda.is_available():
        return torch.device("cpu"), "CUDA is unavailable, falling back to CPU"

    requested_device = args.device
    available_gpu_count = torch.cuda.device_count()
    if requested_device < 0 or requested_device >= available_gpu_count:
        return torch.device("cuda:0"), "Requested GPU index {} is unavailable, falling back to cuda:0".format(requested_device)

    gpu_name = torch.cuda.get_device_name(requested_device)
    return torch.device(f"cuda:{requested_device}"), "Using GPU cuda:{} ({})".format(requested_device, gpu_name)


def strip_json_line_comments(raw_text):
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
    with open(config_path, "r", encoding="utf-8-sig") as config_file:
        return json.loads(strip_json_line_comments(config_file.read()))


def load_checkpoint_if_available(args, device, logger, model_list, optimizer_list):
    if not args.test_checkpoint:
        return None
    if not os.path.isfile(args.test_checkpoint):
        logger.info("=> No checkpoint found at '%s'", args.test_checkpoint)
        return None

    logger.info("=> Loading checkpoint '%s'", args.test_checkpoint)
    checkpoint = torch.load(args.test_checkpoint, map_location=device)
    for model, state_dict in zip(model_list, checkpoint["state_dict"]):
        model.load_state_dict(state_dict)
    for optimizer, optimizer_state in zip(optimizer_list, checkpoint["optimizer"]):
        optimizer.load_state_dict(optimizer_state)
    return checkpoint


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="", help="optional JSON config file path")
    parser.add_argument("--data_dir", default="dataset/data_raw/", help="dataset root directory")
    parser.add_argument("--dataset", default="UCIHAR", choices=["CIFAR10", "UCIHAR", "PHISHING", "NUSWIDE"])
    parser.add_argument("--device", default=0, type=int, help="GPU number")
    parser.add_argument("--results_dir", default="", help="directory used to save logs and checkpoints")
    parser.add_argument("--seed", default=100, type=int, help="random seed")
    parser.add_argument("--epoch", default=60, type=int, help="number of training epochs")
    parser.add_argument("--batch_size", default=256, type=int, help="training batch size")
    parser.add_argument("--client_num", default=2, type=int, help="number of clients")
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--start_epoch", default=0, type=int, help="starting epoch index")
    parser.add_argument("--print_steps", default=10, type=int, help="logging interval")
    parser.add_argument("--early_stop", default=10, type=int, help="early stopping patience")
    parser.add_argument("--target_label", default=None, type=int, help="target label for the backdoor")
    parser.add_argument("--poison_rate", default=0.1, type=float, help="ratio of poison samples")
    parser.add_argument("--poison_dimensions", default=5, type=int, help="number of poisoned feature dimensions")
    parser.add_argument("--trigger_feature_clip", default=1.0, type=float, help="feature trigger clip ratio")
    parser.add_argument("--attack_client_num", default=1, type=int, help="attacker client index")
    parser.add_argument("--select_rate", default=1.0, type=float, help="ratio of switched samples for LFBA")
    parser.add_argument("--random_select", action="store_true")
    parser.add_argument("--poison_all", action="store_true")
    parser.add_argument("--anchor_idx", default=1000, type=int, help="anchor sample index for LFBA")
    parser.add_argument("--pretrain_stage", default=0, type=int, help="LFBA warmup stage")
    parser.add_argument("--mode", default="train", choices=["train", "test"])
    parser.add_argument("--test_checkpoint", default=None, help="checkpoint used for evaluation")
    return parser


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default="")
    config_args, remaining_argv = config_parser.parse_known_args()

    parser = build_parser()
    if config_args.config:
        parser.set_defaults(**load_config_file(config_args.config))
    return parser.parse_args(remaining_argv)


def main(args):
    normalize_args(args)
    device, device_message = resolve_runtime_device(args)
    logger = create_logger(args.results_dir)
    logger.info(args)
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
    trigger_dimensions = select_trigger_dimensions(args, train_data)
    test_data_asr.data = attack_lfba_test(args, logger, test_data_asr.data_p, test_data_asr.targets, trigger_dimensions, "test")

    model_list, optimizer_list, criterion = build_models(args, device)
    checkpoint = load_checkpoint_if_available(args, device, logger, model_list, optimizer_list)
    train_loader, test_loader, test_asr_loader = build_loaders(args, train_data, test_data, test_data_asr)

    trainer = PlainVFLTrainer(
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
    )

    if args.mode == "test":
        trainer.test((checkpoint.get("epoch", 1) - 1) if checkpoint else 0)
        return

    trainer.train()


if __name__ == "__main__":
    parsed_args = parse_args()
    set_seed(parsed_args.seed)
    main(parsed_args)
