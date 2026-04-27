import copy
import time
from random import sample

import numpy as np
import torch

from attack.attack import attack_LFBA, attack_lfba_test, attack_lra, attack_rsa, get_near_index
from utils.utils import (
    LFBA_POISON_SOURCE_GRADIENT,
    LFBA_POISON_SOURCE_ORACLE_LABEL,
    get_attack_target_label,
)


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
            # LFBA test samples are rebuilt during evaluation so they always match the runtime label inferred from anchor_idx.
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

    def to_checkpoint_state(self):
        return None

    def load_checkpoint_state(self, state):
        del state


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

    def _uses_gradient_poison_source(self):
        return self.args.lfba_poison_source == LFBA_POISON_SOURCE_GRADIENT

    def _uses_oracle_label_poison_source(self):
        return self.args.lfba_poison_source == LFBA_POISON_SOURCE_ORACLE_LABEL

    def _reset_epoch_cache(self):
        self.grad_vec_epoch = []
        self.indexes_epoch = []
        self.target_epoch = []

    @staticmethod
    def _clone_optional_tensor(value):
        if value is None:
            return None
        return value.detach().cpu().clone()

    @staticmethod
    def _dataset_targets_to_tensor(dataset):
        targets = dataset.targets
        if torch.is_tensor(targets):
            return targets.detach().cpu().long().clone()
        return torch.as_tensor(targets, dtype=torch.long)

    @staticmethod
    def _dataset_indexes(dataset):
        return torch.arange(len(dataset), dtype=torch.long)

    @staticmethod
    def _sample_python_ints(values, sample_size):
        sample_size = max(0, min(int(sample_size), len(values)))
        if sample_size == 0:
            return []
        return sample(list(values), sample_size)

    @staticmethod
    def _empty_index_tensor():
        return torch.empty(0, dtype=torch.long)

    def _refresh_oracle_dataset_cache(self):
        if self.train_loader is None:
            return False
        dataset = self.train_loader.dataset
        self.train_indexes = self._dataset_indexes(dataset)
        self.train_labels = self._dataset_targets_to_tensor(dataset)
        self.anchor_label = int(self.train_labels[self.args.anchor_idx])
        return True

    def _build_oracle_label_poison_indexes(self):
        if not self._refresh_oracle_dataset_cache():
            return False
        same_label_mask = self.train_labels == self.anchor_label
        same_label_indexes = self.train_indexes[same_label_mask]
        total_same_label = int(len(same_label_indexes))
        if total_same_label == 0:
            self.logger.info(
                "=> Oracle LFBA poison-set construction skipped because no samples matched anchor label %s",
                self.anchor_label,
            )
            self.poison_indexes = self._empty_index_tensor()
            return True

        num_poisons = int(self.args.poison_rate * total_same_label)
        if self.args.poison_rate > 0 and num_poisons == 0:
            num_poisons = 1
        num_poisons = min(num_poisons, total_same_label)
        if num_poisons <= 0:
            self.poison_indexes = self._empty_index_tensor()
            self.logger.info(
                "=> Oracle LFBA poison set is empty because poison_rate=%.4f over %s same-label samples",
                self.args.poison_rate,
                total_same_label,
            )
            return True

        permutation = torch.randperm(total_same_label)[:num_poisons]
        self.poison_indexes = same_label_indexes[permutation].cpu()
        self.logger.info(
            "=> Built LFBA poison set from oracle labels: anchor label=%s, total same-label samples=%s, selected poison samples=%s, effective ratio=%.4f",
            self.anchor_label,
            total_same_label,
            len(self.poison_indexes),
            len(self.poison_indexes) / max(1, total_same_label),
        )
        return True

    def _rebuild_poison_indexes_from_cached_features(self):
        if self.train_features is None or self.train_labels is None or self.train_indexes is None:
            return False
        num_poisons = int(self.args.poison_rate * len(self.train_indexes))
        if num_poisons <= 0:
            self.poison_indexes = torch.empty(0, dtype=self.train_indexes.dtype)
            return True
        anchor_idx_t = torch.nonzero(self.train_indexes == self.args.anchor_idx).squeeze()
        if anchor_idx_t.numel() == 0:
            self.logger.info(
                "=> Unable to rebuild LFBA poison indexes because anchor_idx=%s is absent from cached train indexes",
                self.args.anchor_idx,
            )
            return False
        if anchor_idx_t.ndim > 0:
            anchor_idx_t = anchor_idx_t.reshape(-1)[0]
        neighbor_indexes = get_near_index(
            self.train_features[anchor_idx_t],
            self.train_features,
            num_poisons,
        )
        self.poison_indexes = self.train_indexes[neighbor_indexes]
        self.anchor_label = int(self.train_labels[anchor_idx_t])
        poison_label_count = int((self.train_labels[neighbor_indexes] == self.anchor_label).sum())
        consistent_rate = float(poison_label_count / max(1, len(neighbor_indexes)))
        self.logger.info(
            "=> Rebuilt LFBA poison set from cached gradients: anchor label=%s, poison samples=%s, same-label poison samples=%s, consistent rate=%.4f",
            self.anchor_label,
            len(neighbor_indexes),
            poison_label_count,
            consistent_rate,
        )
        return True

    def to_checkpoint_state(self):
        return {
            "attack_name": "LFBA",
            "train_features": self._clone_optional_tensor(self.train_features),
            "train_labels": self._clone_optional_tensor(self.train_labels),
            "train_indexes": self._clone_optional_tensor(self.train_indexes),
            "poison_indexes": self._clone_optional_tensor(self.poison_indexes),
            "anchor_label": None if self.anchor_label is None else int(self.anchor_label),
            "total_time_gpc": float(self.total_time_gpc),
            "total_time_hs": float(self.total_time_hs),
        }

    def load_checkpoint_state(self, state):
        if not isinstance(state, dict):
            return
        self.train_features = self._clone_optional_tensor(state.get("train_features"))
        self.train_labels = self._clone_optional_tensor(state.get("train_labels"))
        self.train_indexes = self._clone_optional_tensor(state.get("train_indexes"))
        self.poison_indexes = self._clone_optional_tensor(state.get("poison_indexes"))
        anchor_label = state.get("anchor_label")
        self.anchor_label = None if anchor_label is None else int(anchor_label)
        self.total_time_gpc = float(state.get("total_time_gpc", 0.0))
        self.total_time_hs = float(state.get("total_time_hs", 0.0))
        self.logger.info(
            "=> Restored LFBA runtime state: cached_features=%s, poison_indexes=%s, anchor_label=%s",
            0 if self.train_features is None else int(self.train_features.shape[0]),
            0 if self.poison_indexes is None else int(len(self.poison_indexes)),
            self.anchor_label,
        )

    def before_backward(self, local_output_list):
        if self._uses_gradient_poison_source():
            local_output_list[self.args.attack_client_num].retain_grad()

    def after_backward(self, local_output_list, labels, index):
        if not self._uses_gradient_poison_source():
            del local_output_list, labels, index
            return
        self.grad_vec_epoch.append(local_output_list[self.args.attack_client_num].grad.detach().to(self.device))
        self.indexes_epoch.append(index)
        self.target_epoch.append(labels.detach())

    def on_epoch_start(self, epoch):
        self._reset_epoch_cache()
        dataset = self.train_loader.dataset
        if self._uses_gradient_poison_source():
            if epoch < 1 or self.train_features is None or self.train_labels is None or self.train_indexes is None:
                return
            self.train_features = self.train_features.cpu()
            self.train_labels = self.train_labels.cpu()
            self.train_indexes = self.train_indexes.cpu()

            if epoch == 1 or self.poison_indexes is None:
                start_time = time.time()
                if not self._rebuild_poison_indexes_from_cached_features():
                    return
                end_time = time.time()
                print("The poison set construction time: {}".format((end_time - start_time)))
                self.total_time_gpc += end_time - start_time
        else:
            if epoch == 0 or self.poison_indexes is None:
                if not self._build_oracle_label_poison_indexes():
                    return
        if self.poison_indexes is None:
            self.logger.info("=> Skip LFBA epoch poisoning because poison_indexes are unavailable")
            return

        num_poisons = len(self.poison_indexes)
        num_select = int(num_poisons * self.args.select_rate)
        poison_indexes_np = self.poison_indexes.numpy()
        poison_membership = np.isin(self.train_indexes.numpy(), poison_indexes_np)
        poison_positions = np.arange(len(self.train_indexes))[poison_membership]
        selected_indexes_target = self._empty_index_tensor()
        if self._uses_gradient_poison_source() and num_select > 0 and len(poison_positions) > 0:
            l2_norm_features = torch.norm(self.train_features[poison_positions], p=2, dim=1)
            start_time = time.time()
            _, select_indexes = l2_norm_features.topk(
                min(num_select, len(poison_positions)),
                dim=0,
                largest=True,
                sorted=True,
            )
            end_time = time.time()
            print("The hard-sample selection time: {}".format((end_time - start_time)))
            self.total_time_hs += end_time - start_time
            selected_indexes_target = self.train_indexes[poison_positions[select_indexes]]
        elif self._uses_oracle_label_poison_source() and num_select > 0:
            self.logger.info(
                "=> Oracle-label LFBA skips gradient hard-sample ranking and samples replace targets directly from the known same-label poison set"
            )

        num_of_replace = min(int(len(self.poison_indexes) * self.args.select_rate), len(self.poison_indexes))
        replace_all_list = list(set(self.train_indexes.numpy()).difference(set(poison_indexes_np)))
        num_of_replace = min(num_of_replace, len(replace_all_list))
        replace_indexes_others = self._sample_python_ints(replace_all_list, num_of_replace)
        random_indexes_target = self._sample_python_ints(self.poison_indexes.tolist(), num_of_replace)
        if torch.is_tensor(selected_indexes_target):
            selected_indexes_target = selected_indexes_target[:num_of_replace]

        if self._uses_oracle_label_poison_source():
            self.anchor_label = int(self.train_labels[self.args.anchor_idx])
        else:
            self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
        self.logger.info("=> LFBA current anchor-derived target label: %s", get_attack_target_label(self.args))

        clean_data_p = copy.deepcopy(dataset.data_p)
        if self.args.poison_all:
            poison_indexes_t = self.poison_indexes
            if self.args.random_select:
                poison_indexes_t = self._sample_python_ints(self.poison_indexes.tolist(), num_select)
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
        if not self.args.random_select and self._uses_oracle_label_poison_source():
            replace_indexes_target = random_indexes_target
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
        if not self._uses_gradient_poison_source():
            self._refresh_oracle_dataset_cache()
            self.train_features = None
            return
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
