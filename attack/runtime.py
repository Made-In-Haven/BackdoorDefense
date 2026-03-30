import copy
import time
from random import sample

import numpy as np
import torch

from attack.attack import attack_LFBA, attack_lfba_test, attack_lra, attack_rsa, get_near_index


def build_stage1_private_poisoned_loaders(args, logger, train_loader, test_loader, trigger_dimensions):
    if args.attack is None:
        return None, None

    poisoned_train_dataset = copy.deepcopy(train_loader.dataset)
    poisoned_test_dataset = copy.deepcopy(test_loader.dataset)

    # Stage 1 only poisons the attacker's own branch, so we prepare a private poisoned view here.
    poisoned_train_dataset.data, poisoned_train_dataset.targets = attack_lra(
        args,
        logger,
        poisoned_train_dataset.data,
        trigger_dimensions,
        poisoned_train_dataset.targets,
        args.poison_rate,
        "train",
    )
    poisoned_test_dataset.data = attack_rsa(
        args,
        logger,
        poisoned_test_dataset.data,
        trigger_dimensions,
        1,
        "test",
    )

    poisoned_train_loader = torch.utils.data.DataLoader(
        dataset=poisoned_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    poisoned_test_loader = torch.utils.data.DataLoader(
        dataset=poisoned_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )
    return poisoned_train_loader, poisoned_test_loader


class BaseAttackRuntime:
    def __init__(self, args, logger, trigger_dimensions, device):
        self.args = args
        self.logger = logger
        self.trigger_dimensions = trigger_dimensions
        self.device = device
        self.train_loader = None
        self.test_asr_loader = None

    def attach_loaders(self, train_loader, test_asr_loader):
        self.train_loader = train_loader
        self.test_asr_loader = test_asr_loader

    def is_enabled(self):
        return self.args.attack is not None

    def apply_initial_dataset_poisoning(self, train_data, test_data_asr):
        if self.args.attack is None:
            test_data_asr.data = attack_rsa(self.args, self.logger, test_data_asr.data, self.trigger_dimensions, 1, "test")
        elif self.args.attack == "rsa":
            train_data.data = attack_rsa(
                self.args,
                self.logger,
                train_data.data,
                self.trigger_dimensions,
                self.args.poison_rate,
                "train",
            )
            test_data_asr.data = attack_rsa(
                self.args,
                self.logger,
                test_data_asr.data,
                self.trigger_dimensions,
                1,
                "test",
            )
        elif self.args.attack == "lra":
            train_data.data, train_data.targets = attack_lra(
                self.args,
                self.logger,
                train_data.data,
                self.trigger_dimensions,
                train_data.targets,
                self.args.poison_rate,
                "train",
            )
            test_data_asr.data, _ = attack_lra(
                self.args,
                self.logger,
                test_data_asr.data,
                self.trigger_dimensions,
                test_data_asr.targets,
                1,
                "test",
            )
        elif self.args.attack == "LFBA":
            # LFBA test samples are rebuilt during evaluation so they always match the current runtime target label.
            test_data_asr.data = attack_lfba_test(
                self.args,
                self.logger,
                test_data_asr.data_p,
                test_data_asr.targets,
                self.trigger_dimensions,
                "test",
            )
        else:
            raise ValueError("Unsupported attack method '{}'".format(self.args.attack))

    def on_epoch_start(self, epoch):
        del epoch

    def before_backward(self, local_output_list):
        del local_output_list

    def after_backward(self, local_output_list, labels, index):
        del local_output_list, labels, index

    def on_epoch_end(self):
        pass

    def prepare_test_asr_dataset(self):
        if self.args.attack == "LFBA" and self.test_asr_loader is not None:
            clean_test_asr_data = copy.deepcopy(self.test_asr_loader.dataset.data_p)
            self.test_asr_loader.dataset.data = attack_lfba_test(
                self.args,
                self.logger,
                clean_test_asr_data,
                self.test_asr_loader.dataset.targets,
                self.trigger_dimensions,
                "test",
            )


class LFBAAttackRuntime(BaseAttackRuntime):
    def __init__(self, args, logger, trigger_dimensions, device):
        super().__init__(args, logger, trigger_dimensions, device)
        self.train_features = None
        self.train_labels = None
        self.train_indexes = None
        self.grad_vec_epoch = []
        self.indexes_epoch = []
        self.target_epoch = []
        self.anchor_label = None
        self.poison_indexes = None
        self.total_time_gpc = 0.0
        self.total_time_hs = 0.0

    def _reset_epoch_cache(self):
        self.grad_vec_epoch = []
        self.indexes_epoch = []
        self.target_epoch = []

    def before_backward(self, local_output_list):
        local_output_list[self.args.attack_client_num].retain_grad()

    def after_backward(self, local_output_list, labels, index):
        self.grad_vec_epoch.append(
            local_output_list[self.args.attack_client_num].grad.detach().to(self.device)
        )
        self.indexes_epoch.append(index)
        self.target_epoch.append(labels.detach())

    def on_epoch_start(self, epoch):
        self._reset_epoch_cache()
        if epoch < 1 or self.train_features is None or self.train_labels is None or self.train_indexes is None:
            return

        dataset = self.train_loader.dataset
        self.train_features = self.train_features.cpu()
        self.train_labels = self.train_labels.cpu()
        self.train_indexes = self.train_indexes.cpu()

        num_poisons = int(self.args.poison_rate * len(dataset.data))
        num_select = int(num_poisons * self.args.select_rate)

        if epoch == 1:
            start_time = time.time()
            anchor_idx_t = torch.nonzero(self.train_indexes == self.args.anchor_idx).squeeze()
            neighbor_indexes = get_near_index(
                self.train_features[anchor_idx_t],
                self.train_features,
                num_poisons,
            )
            end_time = time.time()
            print("The poison set construction time: {}".format((end_time - start_time)))
            self.total_time_gpc += end_time - start_time

            self.poison_indexes = self.train_indexes[neighbor_indexes]
            self.anchor_label = int(self.train_labels[anchor_idx_t])
            poison_label_count = int((self.train_labels[neighbor_indexes] == self.anchor_label).sum())
            consistent_rate = float(poison_label_count / max(1, len(neighbor_indexes)))
            self.logger.info(
                "=> LFBA poison set summary: anchor label=%s, poison samples=%s, same-label poison samples=%s, consistent rate=%.4f",
                self.anchor_label,
                len(neighbor_indexes),
                poison_label_count,
                consistent_rate,
            )

        poison_membership = np.isin(self.train_indexes.numpy(), torch.tensor(self.poison_indexes).numpy())
        poison_positions = np.arange(len(self.train_indexes))[poison_membership]
        l2_norm_features = torch.norm(self.train_features[poison_positions], p=2, dim=1)

        start_time = time.time()
        _, select_indexes = l2_norm_features.topk(num_select, dim=0, largest=True, sorted=True)
        end_time = time.time()
        print("The hard-sample selection time: {}".format((end_time - start_time)))
        self.total_time_hs += end_time - start_time

        num_of_replace = int(len(self.poison_indexes) * self.args.select_rate)
        replace_all_list = list(
            set(self.train_indexes.numpy()).difference(set(torch.tensor(self.poison_indexes).numpy()))
        )
        replace_indexes_others = sample(replace_all_list, num_of_replace)
        random_indexes_target = sample(list(self.poison_indexes), num_of_replace)
        selected_indexes_target = self.train_indexes[poison_positions[select_indexes]]

        poisoning_labels = np.array(self.train_labels)[poison_membership]
        del poisoning_labels  # Keep the current LFBA data path identical while avoiding unused-variable warnings.

        self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
        self.args.target_label = self.anchor_label
        self.logger.info("Target label:%s", self.anchor_label)

        clean_data_p = copy.deepcopy(dataset.data_p)
        if self.args.poison_all:
            poison_indexes_t = self.poison_indexes
            if self.args.random_select:
                poison_indexes_t = sample(list(self.poison_indexes), num_select)
            dataset.data = attack_LFBA(
                self.args,
                self.logger,
                [],
                [],
                self.train_indexes,
                poison_indexes_t,
                clean_data_p,
                dataset.targets,
                self.trigger_dimensions,
                self.args.poison_rate,
                "train",
            )
            return

        replace_indexes_target = random_indexes_target if self.args.random_select else selected_indexes_target
        dataset.data = attack_LFBA(
            self.args,
            self.logger,
            replace_indexes_others,
            replace_indexes_target,
            self.train_indexes,
            self.poison_indexes,
            clean_data_p,
            dataset.targets,
            self.trigger_dimensions,
            self.args.poison_rate,
            "train",
        )

    def on_epoch_end(self):
        if not self.grad_vec_epoch:
            self.train_features = None
            self.train_labels = None
            self.train_indexes = None
            return
        self.train_features = torch.cat(self.grad_vec_epoch)
        self.train_indexes = torch.cat(self.indexes_epoch)
        self.train_labels = torch.cat(self.target_epoch)

    def get_timing_stats(self):
        return {
            "poison_set_construction_time": self.total_time_gpc,
            "hard_sample_selection_time": self.total_time_hs,
        }


def create_attack_runtime(args, logger, trigger_dimensions, device):
    if args.attack == "LFBA":
        return LFBAAttackRuntime(args=args, logger=logger, trigger_dimensions=trigger_dimensions, device=device)
    return BaseAttackRuntime(args=args, logger=logger, trigger_dimensions=trigger_dimensions, device=device)
