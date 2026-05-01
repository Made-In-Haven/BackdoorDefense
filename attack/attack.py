import copy

import numpy as np
import torch
from attack.add_trigger import add_trigger_to_data
from attack.add_trigger_replace import add_trigger_to_data_replace
from utils.utils import get_attack_target_label


def attack_lra(args, logger, data, trigger_dimensions, targets, rate, mode):
    new_data = copy.deepcopy(data)
    new_targets = copy.deepcopy(targets)
    poison_indexes = np.random.permutation(len(new_data))[0: int(len(new_data) * rate)]
    new_data, new_targets = add_trigger_to_data(args, logger, poison_indexes, new_data, trigger_dimensions, new_targets,
                                                rate, mode,
                                                replace_label=True)
    return new_data, new_targets


def attack_rsa(args, logger, data, trigger_dimensions, rate, mode):
    new_data = copy.deepcopy(data)
    poison_indexes = np.random.permutation(len(new_data))[0: int(len(new_data) * rate)]
    new_data, _ = add_trigger_to_data(args, logger, poison_indexes, data, trigger_dimensions, [], rate, mode,
                                      replace_label=False)
    return new_data


def attack_LFBA(args, logger, replace_indexes_others, replace_indexes_target, train_indexes, poison_indexes, data,
               target, trigger_dimensions, rate,
               mode):
    if args.poison_all:
        new_data, _ = add_trigger_to_data(args, logger, poison_indexes, data, trigger_dimensions, target, rate, mode,
                                          replace_label=False)
    else:
        new_data, _ = add_trigger_to_data_replace(args, logger, replace_indexes_others, replace_indexes_target,
                                                  train_indexes, poison_indexes, data, trigger_dimensions, target, rate,
                                                  mode,
                                                  replace_label=False)
    return new_data


def attack_lfba_test(args, logger, data, targets, trigger_dimensions, mode="test"):
    new_data = copy.deepcopy(data)
    new_targets = copy.deepcopy(targets)
    target_array = np.asarray(targets)
    attack_target_label = get_attack_target_label(args)
    attacked_indexes = np.where(target_array != attack_target_label)[0]

    if len(attacked_indexes) == 0:
        logger.info(
            "=> LFBA test attack skipped because no non-target samples were found for target label %s",
            attack_target_label,
        )
        return new_data

    # Rebuild the LFBA test set with the same attack family as training instead of falling back to RSA.
    if args.poison_all:
        new_data, _ = add_trigger_to_data(
            args,
            logger,
            attacked_indexes,
            new_data,
            trigger_dimensions,
            new_targets,
            1,
            mode,
            replace_label=False,
        )
        return new_data

    all_indexes = np.arange(len(new_data))
    source_indexes = np.resize(np.roll(all_indexes, 1), len(attacked_indexes))
    new_data, _ = add_trigger_to_data_replace(
        args,
        logger,
        source_indexes,
        attacked_indexes,
        all_indexes,
        attacked_indexes,
        new_data,
        trigger_dimensions,
        new_targets,
        1,
        mode,
        replace_label=False,
    )
    return new_data


def select_LFBA(train_features, num_poisons):
    anchor_idx = get_anchor_LFBA(
        train_features, num_poisons)
    anchor_feature = train_features[anchor_idx]

    poisoning_index = get_near_index(
        anchor_feature, train_features, num_poisons)
    poisoning_index = poisoning_index.cpu()

    return poisoning_index, anchor_idx


def get_anchor_LFBA(train_features, num_poisons):
    consistency = train_features @ train_features.T
    w = torch.cat((torch.ones((num_poisons)),
                   -torch.ones((num_poisons))), dim=0)
    top_con = torch.topk(consistency, 2 * num_poisons, dim=1)[0]
    mean_top_con = torch.matmul(top_con, w)
    idx = torch.argmax(mean_top_con)
    return idx


def get_near_index(anchor_feature, train_features, num_poisons):
    anchor_feature_l1 = torch.norm(anchor_feature, p=1)
    train_features_l1 = torch.norm(train_features, p=1, dim=1)
    vals, indices = torch.topk(torch.div((train_features @ anchor_feature), (train_features_l1 * anchor_feature_l1)), k=num_poisons, dim=0)
    return indices
