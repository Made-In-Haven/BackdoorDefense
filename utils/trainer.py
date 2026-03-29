import copy
import os
import time
from random import sample

import numpy as np
import torch

from attack.attack import attack_LFBA, attack_lfba_test, get_near_index
from dataset.utils import split_vfl


class Trainer:
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
        args=None,
        checkpoint=None,
        anchor_defense=None,
    ):
        self.device = device
        self.model_list = model_list
        self.optimizer_list = optimizer_list
        self.criterion = criterion
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.test_asr_loader = test_asr_loader
        self.logger = logger
        self.args = args
        self.checkpoint = checkpoint
        self.trigger_dimensions = trigger_dimensions
        self.anchor_defense = anchor_defense
        self.enable_anchor_loss = getattr(args, "enable_anchor_loss", True)

    def adjust_learning_rate(self, epoch):
        lr = self.args.lr * (0.1) ** (epoch // 20)
        for opt in self.optimizer_list:
            for param_group in opt.param_groups:
                param_group['lr'] = lr

    def _forward_batch(self, x):
        x_split_list = split_vfl(x, self.args)
        local_output_list = [self.model_list[i + 1](x_split_list[i]) for i in range(self.args.client_num)]
        global_output = self.model_list[0](local_output_list)
        return local_output_list, global_output

    def train(self):
        start_time_train = time.time()
        if self.args.attack:
            self.logger.info("=> Start Training with {}...".format(self.args.attack))
        else:
            self.logger.info("=> Start Training Baseline...")
        if self.anchor_defense is not None:
            self.logger.info(
                "=> Enter Stage 2: joint VFL training starts from stage1 passive local backbones with frozen anchor heads"
            )
            self.logger.info("=> Stage 2 anchor loss enabled: %s", self.enable_anchor_loss)
        epoch_loss_list = []
        model_list = self.model_list
        best_acc = 0
        best_epoch = 0
        best_metrics = {}
        no_change = 0
        total_time_GPC = 0
        total_time_HS = 0
        if self.checkpoint:
            best_acc = self.checkpoint['best_acc']
        # train and update
        for ep in range(self.args.start_epoch, self.args.epoch):
            for model in model_list:
                model.train()
            batch_ce_loss_list = []
            batch_loss_list = []
            batch_anchor_loss_list = []
            total = 0
            correct = 0
            if ep >= 1 and self.args.attack == 'LFBA':
                self.train_features, self.train_labels, self.train_indexes = self.grad_vec_epoch, self.target_epoch, self.indexes_epoch
                self.train_features, self.train_labels, self.train_indexes = self.train_features.cpu(), self.train_labels.cpu(), self.train_indexes.cpu()
                self.num_poisons = int(self.args.poison_rate * len(self.train_loader.dataset.data))
                self.num_select = int(self.num_poisons * self.args.select_rate)

                # select sample set
                if ep == 1:
                    start_time = time.time()
                    self.anchor_idx_t = torch.nonzero(self.train_indexes == self.args.anchor_idx).squeeze()
                    self.indexes = get_near_index(self.train_features[self.anchor_idx_t], self.train_features,
                                                  self.num_poisons)
                    end_time = time.time()
                    print("The poison set construction time: {}".format((end_time - start_time)))
                    total_time_GPC += (end_time - start_time)
                    self.poison_indexes = self.train_indexes[self.indexes]
                    self.anchor_label = int(self.train_labels[self.anchor_idx_t])
                    self.poison_label_count = int((self.train_labels[self.indexes] == self.anchor_label).sum())
                    self.consistent_rate = float(
                        self.poison_label_count / len(self.indexes))
                    # LFBA poison-set purity matters a lot: a large poison_rate can easily mix in many non-target labels.
                    self.logger.info(
                        "=> LFBA poison set summary: anchor label=%s, poison samples=%s, same-label poison samples=%s, consistent rate=%.4f",
                        self.anchor_label,
                        len(self.indexes),
                        self.poison_label_count,
                        self.consistent_rate,
                    )

                # For replace poisoning
                self.indexes = np.isin(self.train_indexes.numpy(), torch.tensor(self.poison_indexes).numpy())
                temp = np.array(range(len(self.train_indexes)))
                self.indexes = temp[self.indexes]
                self.l2_norm_features = torch.norm(self.train_features[self.indexes], p=2, dim=1)
                start_time = time.time()
                self.poison_features, self.select_indexes = self.l2_norm_features.topk(self.num_select, dim=0,
                                                                                       largest=True,
                                                                                       sorted=True)
                end_time = time.time()
                print("The hard-sample selection time: {}".format((end_time - start_time)))
                total_time_HS += (end_time - start_time)
                num_of_replace = int(len(self.poison_indexes) * self.args.select_rate)
                replace_all_list = list(set(self.train_indexes.numpy()).difference(set(torch.tensor(self.poison_indexes).numpy())))
                replace_indexes_others = sample(replace_all_list, num_of_replace)
                random_indexes_target = sample(list(self.poison_indexes), num_of_replace)
                selected_indexes_target = self.train_indexes[self.indexes[self.select_indexes]]

                if self.args.poison_all:
                    if self.args.random_select:
                        self.poison_indexes_t = sample(list(self.poison_indexes), self.num_select)
                        self.indexes = np.isin(self.train_indexes.numpy(), torch.tensor(self.poison_indexes_t).numpy())
                    self.poisoning_labels = np.array(self.train_labels)[self.indexes]
                    self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
                    self.args.target_label = self.anchor_label
                    self.logger.info('Target label:{}'.format(self.anchor_label))
                    self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)
                    if self.args.random_select:
                        self.train_loader.dataset.data = attack_LFBA(self.args, self.logger, [],
                                                                    [], self.train_indexes,
                                                                    self.poison_indexes_t,
                                                                    self.clean_data_p, self.train_loader.dataset.targets,
                                                                    self.trigger_dimensions,
                                                                    self.args.poison_rate, 'train')
                    else:
                        self.train_loader.dataset.data = attack_LFBA(self.args, self.logger, [],
                                                                    [], self.train_indexes,
                                                                    self.poison_indexes,
                                                                    self.clean_data_p,
                                                                    self.train_loader.dataset.targets,
                                                                    self.trigger_dimensions,
                                                                    self.args.poison_rate, 'train')
                else:
                    if self.args.random_select:
                        replace_indexes_target = random_indexes_target
                    else:
                        replace_indexes_target = selected_indexes_target
                    self.poisoning_labels = np.array(self.train_labels)[self.indexes]
                    self.anchor_label = int(self.train_labels[self.train_indexes == self.args.anchor_idx])
                    self.clean_data_p = copy.deepcopy(self.train_loader.dataset.data_p)
                    self.train_loader.dataset.data = attack_LFBA(self.args, self.logger, replace_indexes_others,
                                                                replace_indexes_target, self.train_indexes,
                                                                self.poison_indexes,
                                                                self.clean_data_p,
                                                                self.train_loader.dataset.targets,
                                                                self.trigger_dimensions,
                                                                self.args.poison_rate, 'train')
                    self.args.target_label = self.anchor_label
                    self.logger.info('Target label:{}'.format(self.anchor_label))

            elif self.args.attack == 'rsa' or self.args.attack == 'lra' or self.args.attack is None:
                pass

            self.logger.info("=> Start Training for Injecting Backdoor...")

            self.grad_vec_epoch = []
            self.indexes_epoch = []
            self.target_epoch = []
            for step, (x_n, x_p, y, index) in enumerate(self.train_loader):
                x = x_n.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, global_output = self._forward_batch(x)
                if self.args.attack == 'LFBA':
                    local_output_list[self.args.attack_client_num].retain_grad()

                # global model backward
                ce_loss = self.criterion(global_output, y)
                anchor_loss = torch.zeros(1, device=self.device).squeeze()
                if self.anchor_defense is not None and self.enable_anchor_loss:
                    anchor_loss = self.anchor_defense.compute_anchor_loss(local_output_list, y)
                loss = ce_loss + self.args.lambda_anchor * anchor_loss
                for opt in self.optimizer_list:
                    opt.zero_grad()

                loss.backward()

                if self.args.attack == 'LFBA':
                    self.grad_vec_epoch.append(
                        local_output_list[self.args.attack_client_num].grad.detach().to(self.device)
                    )
                    self.indexes_epoch.append(index)
                    self.target_epoch.append(y.detach())

                for opt in self.optimizer_list:
                    opt.step()
                batch_ce_loss_list.append(ce_loss.item())
                batch_loss_list.append(loss.item())
                batch_anchor_loss_list.append(anchor_loss.item())

                # calculate the training accuracy
                _, predicted = global_output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                # train_acc
                train_acc = correct / total
                current_ce_loss = sum(batch_ce_loss_list) / len(batch_ce_loss_list)
                current_loss = sum(batch_loss_list) / len(batch_loss_list)
                current_anchor_loss = sum(batch_anchor_loss_list) / len(batch_anchor_loss_list)

                if step % self.args.print_steps == 0:
                    self.logger.info(
                        'Epoch: {}, {}/{}: VFL loss: {:.4f}, total loss: {:.4f}, anchor loss: {:.4f}, VFL train accuracy: {:.4f}'.format(
                            ep + 1,
                            step + 1,
                            len(self.train_loader),
                            current_ce_loss,
                            current_loss,
                            current_anchor_loss,
                            train_acc,
                        )
                    )
            if self.args.attack == 'LFBA':
                self.grad_vec_epoch = torch.cat(self.grad_vec_epoch)
                self.indexes_epoch = torch.cat(self.indexes_epoch)
                self.target_epoch = torch.cat(self.target_epoch)

            epoch_ce_loss = sum(batch_ce_loss_list) / len(batch_ce_loss_list)
            epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
            epoch_anchor_loss = sum(batch_anchor_loss_list) / len(batch_anchor_loss_list)
            epoch_train_acc = correct / max(1, total)
            epoch_loss_list.append(epoch_loss)
            self.logger.info(
                '=> Stage 2 Epoch {} Training Summary: VFL loss: {:.4f}, total loss: {:.4f}, anchor loss: {:.4f}, VFL train accuracy: {:.4f}'.format(
                    ep + 1,
                    epoch_ce_loss,
                    epoch_loss,
                    epoch_anchor_loss,
                    epoch_train_acc,
                )
            )
            self.adjust_learning_rate(ep + 1)
            metrics = self.test(ep)
            if metrics['test_acc'] > best_acc:
                # Save the checkpoint with the best clean main-task accuracy in stage 2.
                best_acc = metrics['test_acc']
                best_metrics = metrics
                no_change = 0
                best_epoch = ep
                # save model
                self.logger.info("=> Save best model...")
                state = {
                    'epoch': ep + 1,
                    'best_acc': best_acc,
                    'metrics': metrics,
                    'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                    'optimizer': [self.optimizer_list[i].state_dict() for i in range(len(self.optimizer_list))],
                    'anchor_state': self.anchor_defense.to_checkpoint_state() if self.anchor_defense else None,
                }
                filename = os.path.join(self.args.results_dir, 'best_checkpoint.pth.tar'.format(ep + 1))
                torch.save(state, filename)
            else:
                if ep > self.args.pretrain_stage:
                    no_change += 1
            self.logger.info(
                '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, best main task accuracy: {:.4f}, test target accuracy: {:.4f}, test asr: {:.4f}, anchor loss: {:.4f}'.format(
                    ep + 1,
                    no_change,
                    best_epoch + 1,
                    best_acc,
                    best_metrics.get('test_target', 0.0),
                    best_metrics.get('test_asr', 0.0),
                    best_metrics.get('anchor_loss', 0.0),
                )
            )
            if no_change == self.args.early_stop:
                end_time_train = time.time()
                print("The total training time: {}".format((end_time_train - start_time_train)))
                print("The average training time of each epoch: {}".format(((end_time_train - start_time_train)) / (ep + 1)))
                print("The poison set construction time: {}".format(total_time_GPC))
                print("The average hard-sample selection time: {}".format(total_time_HS / (ep + 1)))
                print("The total hard-sample selection time: {}".format(total_time_HS))
                return



    def test(self, ep):
        self.logger.info("=> Test ASR...")
        model_list = self.model_list
        model_list = [model.eval() for model in model_list]
        if self.args.attack == 'LFBA':
            # Rebuild the poisoned test set each epoch so LFBA evaluation matches the current training-time target label.
            clean_test_asr_data = copy.deepcopy(self.test_asr_loader.dataset.data_p)
            self.test_asr_loader.dataset.data = attack_lfba_test(
                self.args,
                self.logger,
                clean_test_asr_data,
                self.test_asr_loader.dataset.targets,
                self.trigger_dimensions,
                'test',
            )
        # test main task accuracy
        batch_loss_list = []
        batch_anchor_loss_list = []
        total = 0
        correct = 0
        total_target = 0
        correct_target = 0
        with torch.no_grad():
            for step, (x, x_p, y, index) in enumerate(self.test_loader):
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, global_output = self._forward_batch(x)

                loss = self.criterion(global_output, y)
                batch_loss_list.append(loss.item())
                if self.anchor_defense is not None and self.enable_anchor_loss:
                    batch_anchor_loss_list.append(self.anchor_defense.compute_anchor_loss(local_output_list, y).item())

                _, predicted = global_output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()
                total_target += (y == self.args.target_label).float().sum().item()
                correct_target += predicted.eq(y)[y == self.args.target_label].float().sum().item()

        # test poison accuracy and asr
        total_poison = 0
        correct_poison = 0
        total_asr = 0
        correct_asr = 0
        with torch.no_grad():
            for step, (x, x_p, y, index) in enumerate(self.test_asr_loader):
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, global_output = self._forward_batch(x)

                _, predicted = global_output.max(1)
                total_poison += y.size(0)
                correct_poison += predicted.eq(y).sum().item()
                total_asr += (y != self.args.target_label).float().sum().item()
                correct_asr += (predicted[y != self.args.target_label] == self.args.target_label).float().sum().item()

        # main task accuracy, poison_acc and asr
        test_acc = correct / max(1, total)
        test_poison_accuracy = correct_poison / max(1, total_poison)
        test_asr = correct_asr / max(1, total_asr)
        test_target = correct_target / max(1, total_target)
        epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        anchor_loss = sum(batch_anchor_loss_list) / max(1, len(batch_anchor_loss_list))
        # main task accuracy on target set
        self.logger.info(
            '=> Test Epoch: {}, main task samples: {}, attack samples: {}, test loss: {:.4f}, test main task '
            'accuracy: {:.4f}, test target accuracy: {:.4f}, test asr: {:.4f}, anchor loss: {:.4f}'.format(
                ep + 1,
                len(self.test_loader.dataset),
                len(self.test_asr_loader.dataset),
                epoch_loss,
                test_acc,
                test_target,
                test_asr,
                anchor_loss,
            )
        )

        return {
            'test_acc': test_acc,
            'test_poison_accuracy': test_poison_accuracy,
            'test_target': test_target,
            'test_asr': test_asr,
            'anchor_loss': anchor_loss,
            'epoch_loss': epoch_loss,
        }
