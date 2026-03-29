import copy
import os
import time
from random import sample

import numpy as np
import torch

from dataset.utils import split_vfl
from standalone_lfba_vfl.attack_core import attack_lfba, attack_lfba_test, get_near_index


class PlainVFLTrainer:
    def __init__(
        self,
        device,
        model_list,
        optimizer_list,
        criterion,
        train_loader,
        test_loader,
        test_asr_loader,
        trigger_dimensions,
        logger,
        args,
        checkpoint=None,
    ):
        self.device = device
        self.model_list = model_list
        self.optimizer_list = optimizer_list
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_asr_loader = test_asr_loader
        self.trigger_dimensions = trigger_dimensions
        self.logger = logger
        self.args = args
        self.checkpoint = checkpoint

    def adjust_learning_rate(self, epoch):
        lr = self.args.lr * (0.1 ** (epoch // 20))
        for optimizer in self.optimizer_list:
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

    def _forward_batch(self, x):
        x_split_list = split_vfl(x, self.args)
        local_output_list = [self.model_list[i + 1](x_split_list[i]) for i in range(self.args.client_num)]
        global_output = self.model_list[0](local_output_list)
        return local_output_list, global_output

    def train(self):
        self.logger.info("=> Start plain VFL training with LFBA...")
        best_acc = self.checkpoint["best_acc"] if self.checkpoint else 0.0
        best_epoch = self.checkpoint["epoch"] - 1 if self.checkpoint else 0
        best_metrics = self.checkpoint.get("metrics", {}) if self.checkpoint else {}
        no_change = 0
        start_time = time.time()

        for epoch in range(self.args.start_epoch, self.args.epoch):
            for model in self.model_list:
                model.train()

            if epoch >= 1:
                self._prepare_epoch_poison(epoch)

            self.logger.info("=> Epoch %s: training with current poisoned view", epoch + 1)
            batch_loss_list = []
            total = 0
            correct = 0
            self.grad_vec_epoch = []
            self.indexes_epoch = []
            self.target_epoch = []

            for step, (x_n, _x_p, y, index) in enumerate(self.train_loader):
                x = x_n.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, global_output = self._forward_batch(x)
                local_output_list[self.args.attack_client_num].retain_grad()

                loss = self.criterion(global_output, y)
                for optimizer in self.optimizer_list:
                    optimizer.zero_grad()
                loss.backward()

                self.grad_vec_epoch.append(local_output_list[self.args.attack_client_num].grad.detach().cpu())
                self.indexes_epoch.append(index)
                self.target_epoch.append(y.detach().cpu())

                for optimizer in self.optimizer_list:
                    optimizer.step()

                batch_loss_list.append(loss.item())
                _, predicted = global_output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                if step % self.args.print_steps == 0:
                    self.logger.info(
                        "Epoch: %s, %s/%s: loss: %.4f, train accuracy: %.4f",
                        epoch + 1,
                        step + 1,
                        len(self.train_loader),
                        sum(batch_loss_list) / len(batch_loss_list),
                        correct / max(1, total),
                    )

            self.grad_vec_epoch = torch.cat(self.grad_vec_epoch)
            self.indexes_epoch = torch.cat(self.indexes_epoch)
            self.target_epoch = torch.cat(self.target_epoch)

            self.logger.info(
                "=> Epoch %s summary: loss: %.4f, train accuracy: %.4f",
                epoch + 1,
                sum(batch_loss_list) / len(batch_loss_list),
                correct / max(1, total),
            )
            self.adjust_learning_rate(epoch + 1)
            metrics = self.test(epoch)

            if metrics["test_acc"] > best_acc:
                best_acc = metrics["test_acc"]
                best_metrics = metrics
                best_epoch = epoch
                no_change = 0
                self._save_checkpoint(epoch, metrics)
            else:
                if epoch > self.args.pretrain_stage:
                    no_change += 1

            self.logger.info(
                "=> End Epoch: %s, early stop epochs: %s, best epoch: %s, best main task accuracy: %.4f, best target accuracy: %.4f, best ASR: %.4f",
                epoch + 1,
                no_change,
                best_epoch + 1,
                best_acc,
                best_metrics.get("test_target", 0.0),
                best_metrics.get("test_asr", 0.0),
            )

            if no_change >= self.args.early_stop:
                break

        total_time = time.time() - start_time
        self.logger.info("=> Training finished in %.2fs", total_time)
        return best_metrics

    def test(self, epoch):
        self.logger.info("=> Test ASR...")
        for model in self.model_list:
            model.eval()

        clean_test_asr_data = copy.deepcopy(self.test_asr_loader.dataset.data_p)
        self.test_asr_loader.dataset.data = attack_lfba_test(
            self.args,
            self.logger,
            clean_test_asr_data,
            self.test_asr_loader.dataset.targets,
            self.trigger_dimensions,
            "test",
        )

        batch_loss_list = []
        total = 0
        correct = 0
        total_target = 0
        correct_target = 0

        with torch.no_grad():
            for x, _x_p, y, _index in self.test_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                _, global_output = self._forward_batch(x)
                loss = self.criterion(global_output, y)
                batch_loss_list.append(loss.item())

                _, predicted = global_output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                total_target += (y == self.args.target_label).float().sum().item()
                correct_target += predicted.eq(y)[y == self.args.target_label].float().sum().item()

        total_poison = 0
        correct_poison = 0
        total_asr = 0
        correct_asr = 0

        with torch.no_grad():
            for x, _x_p, y, _index in self.test_asr_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                _, global_output = self._forward_batch(x)
                _, predicted = global_output.max(1)

                total_poison += y.size(0)
                correct_poison += predicted.eq(y).sum().item()
                total_asr += (y != self.args.target_label).float().sum().item()
                correct_asr += (predicted[y != self.args.target_label] == self.args.target_label).float().sum().item()

        metrics = {
            "test_acc": correct / max(1, total),
            "test_poison_accuracy": correct_poison / max(1, total_poison),
            "test_target": correct_target / max(1, total_target),
            "test_asr": correct_asr / max(1, total_asr),
            "epoch_loss": sum(batch_loss_list) / len(batch_loss_list),
        }
        self.logger.info(
            "=> Test Epoch: %s, clean samples: %s, attack samples: %s, loss: %.4f, clean acc: %.4f, target acc: %.4f, ASR: %.4f",
            epoch + 1,
            len(self.test_loader.dataset),
            len(self.test_asr_loader.dataset),
            metrics["epoch_loss"],
            metrics["test_acc"],
            metrics["test_target"],
            metrics["test_asr"],
        )
        return metrics

    def _prepare_epoch_poison(self, epoch):
        train_features = self.grad_vec_epoch
        train_labels = self.target_epoch
        train_indexes = self.indexes_epoch
        num_poisons = min(len(train_indexes), int(self.args.poison_rate * len(self.train_loader.dataset.data)))
        num_select = min(num_poisons, max(1, int(num_poisons * self.args.select_rate)))

        if num_poisons <= 0:
            raise ValueError("poison_rate is too small for the current dataset size.")

        if epoch == 1:
            anchor_matches = torch.nonzero(train_indexes == self.args.anchor_idx).flatten()
            if len(anchor_matches) == 0:
                raise ValueError(
                    "anchor_idx {} was not found in the training split. Please choose a valid sample index.".format(
                        self.args.anchor_idx
                    )
                )
            anchor_position = anchor_matches[0]
            candidate_positions = get_near_index(train_features[anchor_position], train_features, num_poisons)
            self.poison_indexes = train_indexes[candidate_positions]
            self.anchor_label = int(train_labels[anchor_position])
            poison_label_count = int((train_labels[candidate_positions] == self.anchor_label).sum())
            consistency_rate = poison_label_count / max(1, len(candidate_positions))
            self.logger.info(
                "=> LFBA poison set summary: anchor label=%s, poison samples=%s, same-label poison samples=%s, consistent rate=%.4f",
                self.anchor_label,
                len(candidate_positions),
                poison_label_count,
                consistency_rate,
            )

        poison_mask = np.isin(train_indexes.numpy(), torch.tensor(self.poison_indexes).numpy())
        poison_positions = np.arange(len(train_indexes))[poison_mask]
        poison_feature_norm = torch.norm(train_features[poison_positions], p=2, dim=1)
        _, select_positions = poison_feature_norm.topk(num_select, dim=0, largest=True, sorted=True)
        selected_indexes_target = train_indexes[poison_positions[select_positions]]

        if self.args.poison_all:
            if self.args.random_select:
                active_poison_indexes = sample(self.poison_indexes.tolist(), num_select)
            else:
                active_poison_indexes = self.poison_indexes.tolist()

            self.args.target_label = self.anchor_label
            clean_data = copy.deepcopy(self.train_loader.dataset.data_p)
            self.train_loader.dataset.data = attack_lfba(
                self.args,
                self.logger,
                [],
                [],
                active_poison_indexes,
                clean_data,
                self.trigger_dimensions,
                self.args.poison_rate,
                "train",
            )
            self.logger.info("=> LFBA target label for epoch %s: %s", epoch + 1, self.args.target_label)
            return

        replace_count = len(selected_indexes_target)
        replace_all_list = list(set(train_indexes.numpy()).difference(set(torch.tensor(self.poison_indexes).numpy())))
        if len(replace_all_list) < replace_count:
            raise ValueError("Not enough clean candidate samples to build LFBA replacement pairs.")

        replace_indexes_others = sample(replace_all_list, replace_count)
        if self.args.random_select:
            replace_indexes_target = sample(list(self.poison_indexes), replace_count)
        else:
            replace_indexes_target = selected_indexes_target.tolist()

        self.args.target_label = self.anchor_label
        clean_data = copy.deepcopy(self.train_loader.dataset.data_p)
        self.train_loader.dataset.data = attack_lfba(
            self.args,
            self.logger,
            replace_indexes_others,
            replace_indexes_target,
            self.poison_indexes.tolist(),
            clean_data,
            self.trigger_dimensions,
            self.args.poison_rate,
            "train",
        )
        self.logger.info("=> LFBA target label for epoch %s: %s", epoch + 1, self.args.target_label)

    def _save_checkpoint(self, epoch, metrics):
        os.makedirs(self.args.results_dir, exist_ok=True)
        state = {
            "epoch": epoch + 1,
            "best_acc": metrics["test_acc"],
            "metrics": metrics,
            "state_dict": [model.state_dict() for model in self.model_list],
            "optimizer": [optimizer.state_dict() for optimizer in self.optimizer_list],
        }
        checkpoint_path = os.path.join(self.args.results_dir, "best_checkpoint.pth.tar")
        torch.save(state, checkpoint_path)
        self.logger.info("=> Saved best checkpoint to '%s'", checkpoint_path)
