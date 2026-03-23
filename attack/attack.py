import copy

import numpy as np
import torch
from attack.add_trigger import add_trigger_to_data
from attack.add_trigger_replace import add_trigger_to_data_replace


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
