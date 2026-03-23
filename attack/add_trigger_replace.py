import copy

from dataset.utils import get_attacker_feature_slice
from utils.utils import *


def add_trigger_to_data_replace(args, logger, replace_indexes_others, replace_indexes_target, train_indexes,
                                poison_indexes, new_data, trigger_dimensions, new_targets, rate, mode,
                                replace_label):
    mode_print(logger, mode)
    if args.dataset == 'CIFAR10':
        new_data, new_targets = add_triangle_pattern_trigger(args, logger, replace_indexes_others,
                                                             replace_indexes_target, train_indexes, poison_indexes,
                                                             new_data,
                                                             new_targets, rate,
                                                             mode, replace_label)
        return new_data, new_targets
    elif args.dataset == 'UCIHAR':
        new_data, new_targets = add_feature_trigger(args, logger, replace_indexes_others, replace_indexes_target,
                                                    train_indexes, poison_indexes, trigger_dimensions, new_data,
                                                    new_targets, rate, mode,
                                                    replace_label)
        return new_data, new_targets
    elif args.dataset == 'PHISHING':
        new_data, new_targets = add_vector_replacement_trigger(args, logger, replace_indexes_others,
                                                               replace_indexes_target, train_indexes, poison_indexes,
                                                               trigger_dimensions, new_data,
                                                               new_targets, rate, mode,
                                                               replace_label)
        return new_data, new_targets
    elif args.dataset == 'NUSWIDE':
        new_data, new_targets = add_vector_replacement_trigger(args, logger, replace_indexes_others,
                                                               replace_indexes_target, train_indexes, poison_indexes,
                                                               trigger_dimensions, new_data, new_targets,
                                                               rate,
                                                               mode, replace_label)
        return new_data, new_targets


def add_triangle_pattern_trigger(args, logger, replace_indexes_others, replace_indexes_target, train_indexes,
                                 poison_indexes, new_data, new_targets, rate, mode,
                                 replace_label):
    height, width, channels = new_data.shape[1:]
    temp = copy.deepcopy(new_data)
    for i, idx in enumerate(replace_indexes_others):
        for c in range(channels):
            temp[idx, height - 3:, width - 3:, c] = 0
            temp[idx, height - 3, width - 1, c] = 255
            temp[idx, height - 1, width - 3, c] = 255
            temp[idx, height - 2, width - 2, c] = 255
            temp[idx, height - 1, width - 1, c] = 255
        new_data[replace_indexes_target[i], :, 16:, :] = temp[idx, :, 16:, :]
    logger.info(
        "Add Trigger to %d Poison Samples, %d Clean Samples (%.2f)" % (
            len(poison_indexes), len(new_data) - len(poison_indexes), rate))
    return new_data, new_targets


def add_feature_trigger(args, logger, replace_indexes_others, replace_indexes_target, train_indexes, poison_indexes,
                        trigger_dimensions, new_data, new_targets,
                        rate, mode,
                        replace_label=True):
    temp = copy.deepcopy(new_data)
    attacker_start, attacker_end = get_attacker_feature_slice(args, new_data.shape[1])
    for i, idx in enumerate(replace_indexes_others):
        temp[idx][trigger_dimensions] = args.trigger_feature_clip
        if args.dataset == 'UCIHAR':
            # Only copy the malicious client's feature slice when replacing samples.
            new_data[replace_indexes_target[i]][attacker_start:attacker_end] = temp[idx][attacker_start:attacker_end]
    logger.info(
        "Add Trigger to %d Bad Samples, %d Clean Samples (%.2f)" % (
            len(poison_indexes), len(new_data) - len(poison_indexes), rate))
    return new_data, new_targets


def add_vector_replacement_trigger(args, logger, replace_indexes_others, replace_indexes_target, train_indexes,
                                   poison_indexes, trigger_dimensions, new_data,
                                   new_targets, rate, mode, replace_label):
    temp = copy.deepcopy(new_data)
    attacker_start, attacker_end = get_attacker_feature_slice(args, new_data.shape[1])
    if args.dataset == 'PHISHING':
        for i, idx in enumerate(replace_indexes_others):
            temp[idx][trigger_dimensions] = 1
            new_data[replace_indexes_target[i]][attacker_start:attacker_end] = temp[idx][attacker_start:attacker_end]
    elif args.dataset == 'NUSWIDE':
        for i, idx in enumerate(replace_indexes_others):
            temp[idx][trigger_dimensions] = 1
            new_data[replace_indexes_target[i]][attacker_start:attacker_end] = temp[idx][attacker_start:attacker_end]
    logger.info(
        "Add Trigger to %d Bad Samples, %d Clean Samples (%.2f)" % (
            len(poison_indexes), len(new_data) - len(poison_indexes), rate))
    return new_data, new_targets


def mode_print(logger, mode):
    if mode == 'train':
        logger.info('=>Add Trigger to Train Data')
    else:
        logger.info('=>Add Trigger to Test Data')
