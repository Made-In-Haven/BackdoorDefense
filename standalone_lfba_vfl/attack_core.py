import copy

import numpy as np
import torch

from dataset.utils import get_attacker_feature_slice, get_attacker_image_slice
from utils.utils import raise_dataset_exception


def select_trigger_dimensions(args, train_data):
    if args.dataset == "CIFAR10":
        return []
    if args.dataset in {"UCIHAR", "PHISHING", "NUSWIDE"}:
        attacker_start, attacker_end = get_attacker_feature_slice(args, train_data.data.shape[1])
        return np.random.choice(range(attacker_start, attacker_end), args.poison_dimensions, replace=False)
    raise_dataset_exception()


def add_trigger_to_data(args, logger, poison_indexes, new_data, trigger_dimensions, rate, mode):
    _mode_print(logger, mode)
    if args.dataset == "CIFAR10":
        return _add_triangle_pattern_trigger(args, logger, poison_indexes, new_data, rate)
    if args.dataset == "UCIHAR":
        return _add_feature_trigger(args, logger, poison_indexes, trigger_dimensions, new_data, rate)
    if args.dataset in {"PHISHING", "NUSWIDE"}:
        return _add_vector_trigger(args, logger, poison_indexes, trigger_dimensions, new_data, rate)
    raise_dataset_exception()


def add_trigger_to_data_replace(
    args,
    logger,
    replace_indexes_others,
    replace_indexes_target,
    poison_indexes,
    new_data,
    trigger_dimensions,
    rate,
    mode,
):
    _mode_print(logger, mode)
    if args.dataset == "CIFAR10":
        return _replace_triangle_pattern_trigger(
            args,
            logger,
            replace_indexes_others,
            replace_indexes_target,
            poison_indexes,
            new_data,
            rate,
        )
    if args.dataset == "UCIHAR":
        return _replace_feature_trigger(
            args,
            logger,
            replace_indexes_others,
            replace_indexes_target,
            poison_indexes,
            trigger_dimensions,
            new_data,
            rate,
        )
    if args.dataset in {"PHISHING", "NUSWIDE"}:
        return _replace_vector_trigger(
            args,
            logger,
            replace_indexes_others,
            replace_indexes_target,
            poison_indexes,
            trigger_dimensions,
            new_data,
            rate,
        )
    raise_dataset_exception()


def attack_lfba(args, logger, replace_indexes_others, replace_indexes_target, poison_indexes, data, trigger_dimensions, rate, mode):
    clean_data = copy.deepcopy(data)
    if args.poison_all:
        return add_trigger_to_data(args, logger, poison_indexes, clean_data, trigger_dimensions, rate, mode)
    return add_trigger_to_data_replace(
        args,
        logger,
        replace_indexes_others,
        replace_indexes_target,
        poison_indexes,
        clean_data,
        trigger_dimensions,
        rate,
        mode,
    )


def attack_lfba_test(args, logger, data, targets, trigger_dimensions, mode="test"):
    clean_data = copy.deepcopy(data)
    target_array = np.asarray(targets)
    attacked_indexes = np.where(target_array != args.target_label)[0]

    if len(attacked_indexes) == 0:
        logger.info("=> LFBA test attack skipped because no non-target samples were found for target label %s", args.target_label)
        return clean_data

    if args.poison_all:
        return add_trigger_to_data(args, logger, attacked_indexes, clean_data, trigger_dimensions, 1.0, mode)

    all_indexes = np.arange(len(clean_data))
    source_indexes = np.resize(np.roll(all_indexes, 1), len(attacked_indexes))
    return add_trigger_to_data_replace(
        args,
        logger,
        source_indexes,
        attacked_indexes,
        attacked_indexes,
        clean_data,
        trigger_dimensions,
        1.0,
        mode,
    )


def get_near_index(anchor_feature, train_features, num_poisons):
    anchor_feature_l1 = torch.norm(anchor_feature, p=1)
    train_features_l1 = torch.norm(train_features, p=1, dim=1)
    _, indices = torch.topk(
        torch.div((train_features @ anchor_feature), (train_features_l1 * anchor_feature_l1)),
        k=num_poisons,
        dim=0,
    )
    return indices


def _add_triangle_pattern_trigger(args, logger, poison_indexes, new_data, rate):
    height, width, channels = new_data.shape[1:]
    attacker_start, attacker_end = get_attacker_image_slice(args, width)
    patch_height = min(3, height)
    patch_width = min(3, max(1, attacker_end - attacker_start))
    row_start = height - patch_height
    col_start = attacker_end - patch_width

    for idx in poison_indexes:
        for channel in range(channels):
            new_data[idx, row_start:height, col_start:attacker_end, channel] = 0
            new_data[idx, row_start, attacker_end - 1, channel] = 255
            new_data[idx, height - 1, col_start, channel] = 255
            new_data[idx, height - 1, attacker_end - 1, channel] = 255
            if patch_height >= 2 and patch_width >= 2:
                center_row = row_start + patch_height // 2
                center_col = col_start + patch_width // 2
                new_data[idx, center_row, center_col, channel] = 255
    logger.info("Add Trigger to %d Poison Samples, %d Clean Samples (%.2f)", len(poison_indexes), len(new_data) - len(poison_indexes), rate)
    return new_data


def _add_feature_trigger(args, logger, poison_indexes, trigger_dimensions, new_data, rate):
    for idx in poison_indexes:
        new_data[idx][trigger_dimensions] = args.trigger_feature_clip
    logger.info("Add Trigger to %d Poison Samples, %d Clean Samples (%.2f)", len(poison_indexes), len(new_data) - len(poison_indexes), rate)
    return new_data


def _add_vector_trigger(args, logger, poison_indexes, trigger_dimensions, new_data, rate):
    for idx in poison_indexes:
        new_data[idx][trigger_dimensions] = 1
    logger.info("Add Trigger to %d Poison Samples, %d Clean Samples (%.2f)", len(poison_indexes), len(new_data) - len(poison_indexes), rate)
    return new_data


def _replace_triangle_pattern_trigger(args, logger, replace_indexes_others, replace_indexes_target, poison_indexes, new_data, rate):
    temp = copy.deepcopy(new_data)
    height, width, channels = new_data.shape[1:]
    attacker_start, attacker_end = get_attacker_image_slice(args, width)
    patch_height = min(3, height)
    patch_width = min(3, max(1, attacker_end - attacker_start))
    row_start = height - patch_height
    col_start = attacker_end - patch_width

    for idx in replace_indexes_others:
        for channel in range(channels):
            temp[idx, row_start:height, col_start:attacker_end, channel] = 0
            temp[idx, row_start, attacker_end - 1, channel] = 255
            temp[idx, height - 1, col_start, channel] = 255
            temp[idx, height - 1, attacker_end - 1, channel] = 255
            if patch_height >= 2 and patch_width >= 2:
                center_row = row_start + patch_height // 2
                center_col = col_start + patch_width // 2
                temp[idx, center_row, center_col, channel] = 255
    for source_index, target_index in zip(replace_indexes_others, replace_indexes_target):
        new_data[target_index, :, attacker_start:attacker_end, :] = temp[source_index, :, attacker_start:attacker_end, :]
    logger.info("Add Trigger to %d Poison Samples, %d Clean Samples (%.2f)", len(poison_indexes), len(new_data) - len(poison_indexes), rate)
    return new_data


def _replace_feature_trigger(
    args,
    logger,
    replace_indexes_others,
    replace_indexes_target,
    poison_indexes,
    trigger_dimensions,
    new_data,
    rate,
):
    temp = copy.deepcopy(new_data)
    attacker_start, attacker_end = get_attacker_feature_slice(args, new_data.shape[1])
    for source_index, target_index in zip(replace_indexes_others, replace_indexes_target):
        temp[source_index][trigger_dimensions] = args.trigger_feature_clip
        new_data[target_index][attacker_start:attacker_end] = temp[source_index][attacker_start:attacker_end]
    logger.info("Add Trigger to %d Poison Samples, %d Clean Samples (%.2f)", len(poison_indexes), len(new_data) - len(poison_indexes), rate)
    return new_data


def _replace_vector_trigger(
    args,
    logger,
    replace_indexes_others,
    replace_indexes_target,
    poison_indexes,
    trigger_dimensions,
    new_data,
    rate,
):
    temp = copy.deepcopy(new_data)
    attacker_start, attacker_end = get_attacker_feature_slice(args, new_data.shape[1])
    for source_index, target_index in zip(replace_indexes_others, replace_indexes_target):
        temp[source_index][trigger_dimensions] = 1
        new_data[target_index][attacker_start:attacker_end] = temp[source_index][attacker_start:attacker_end]
    logger.info("Add Trigger to %d Poison Samples, %d Clean Samples (%.2f)", len(poison_indexes), len(new_data) - len(poison_indexes), rate)
    return new_data


def _mode_print(logger, mode):
    if mode == "train":
        logger.info("=> Add Trigger to Train Data")
    else:
        logger.info("=> Add Trigger to Test Data")
