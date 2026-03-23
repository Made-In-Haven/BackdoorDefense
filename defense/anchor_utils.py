import math
import os

import torch

from defense.anchor_losses import ArcFaceClassifier


def get_num_classes(dataset_name):
    mapping = {
        "CIFAR10": 10,
        "UCIHAR": 6,
        "PHISHING": 2,
        "NUSWIDE": 5,
        "NUSWIDET": 5,
        "NUSWIDEI": 5,
    }
    return mapping[dataset_name]


def get_local_models(model_list):
    return model_list[1:]


def get_local_feature_dims(model_list):
    return [model.output_dim for model in get_local_models(model_list)]


def build_anchor_heads(model_list, args, device):
    num_classes = get_num_classes(args.dataset)
    heads = []
    for feature_dim in get_local_feature_dims(model_list):
        heads.append(
            ArcFaceClassifier(
                feature_dim=feature_dim,
                num_classes=num_classes,
                scale=args.anchor_scale,
                margin=args.anchor_margin,
            ).to(device)
        )
    return heads


def ensure_parent_dir(path):
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def build_vote_threshold(client_num, majority_ratio):
    return max(1, math.ceil(client_num * majority_ratio))


def save_anchor_artifact(path, payload):
    ensure_parent_dir(path)
    torch.save(payload, path)


def get_stage1_dir(args):
    stage1_dir = getattr(args, "anchor_stage1_dir", "")
    if stage1_dir:
        return stage1_dir
    return os.path.join(args.results_dir, "anchor_stage1")
