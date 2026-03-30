import json
import os
import time

import torch

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
        attack_runtime=None,
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
        self.attack_runtime = attack_runtime
        self.enable_anchor_loss = getattr(args, "enable_anchor_loss", True)

    @staticmethod
    def _format_client_anchor_losses(client_anchor_losses):
        return {
            int(client_id): round(float(client_anchor_losses[client_id]), 6)
            for client_id in sorted(client_anchor_losses)
        }

    def _record_final_epoch_client_anchor_losses(self, epoch, client_anchor_losses):
        if not client_anchor_losses:
            return
        self.anchor_defense.set_final_epoch_client_anchor_losses(client_anchor_losses)
        os.makedirs(self.args.results_dir, exist_ok=True)
        output_path = os.path.join(self.args.results_dir, 'stage2_final_epoch_client_anchor_losses.json')
        payload = {
            'epoch': epoch + 1,
            'client_anchor_losses': {
                str(client_id): float(client_anchor_losses[client_id])
                for client_id in sorted(client_anchor_losses)
            },
        }
        with open(output_path, 'w', encoding='utf-8') as output_file:
            json.dump(payload, output_file, indent=2)
        self.logger.info(
            "=> Recorded Stage 2 client anchor losses for epoch %s to '%s': %s",
            epoch + 1,
            output_path,
            self._format_client_anchor_losses(client_anchor_losses),
        )

    def _stage3_ready(self):
        return self.anchor_defense is not None and self.anchor_defense.has_stage3_stats()

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
        if self.checkpoint:
            best_acc = self.checkpoint['best_acc']
        # train and update
        for ep in range(self.args.start_epoch, self.args.epoch):
            for model in model_list:
                model.train()
            batch_ce_loss_list = []
            batch_loss_list = []
            batch_anchor_loss_list = []
            epoch_client_anchor_loss_sums = None
            epoch_client_anchor_sample_count = 0
            if self.anchor_defense is not None:
                epoch_client_anchor_loss_sums = {client_id: 0.0 for client_id in range(self.args.client_num)}
            total = 0
            correct = 0
            if self.attack_runtime is not None:
                self.attack_runtime.on_epoch_start(ep)
            if self.attack_runtime is not None and self.attack_runtime.is_enabled():
                self.logger.info("=> Start Training for Injecting Backdoor...")
            for step, (x_n, x_p, y, index) in enumerate(self.train_loader):
                x = x_n.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, global_output = self._forward_batch(x)
                if self.attack_runtime is not None:
                    self.attack_runtime.before_backward(local_output_list)

                # global model backward
                ce_loss = self.criterion(global_output, y)
                anchor_loss = torch.zeros(1, device=self.device).squeeze()
                client_anchor_loss_dict = {}
                if self.anchor_defense is not None:
                    anchor_loss, client_anchor_loss_dict = self.anchor_defense.compute_anchor_loss(
                        local_output_list,
                        y,
                        return_client_losses=True,
                    )
                    batch_size = y.size(0)
                    epoch_client_anchor_sample_count += batch_size
                    for client_id, client_anchor_loss in client_anchor_loss_dict.items():
                        epoch_client_anchor_loss_sums[client_id] += client_anchor_loss.item() * batch_size
                loss = ce_loss + self.args.lambda_anchor * anchor_loss if self.anchor_defense is not None and self.enable_anchor_loss else ce_loss
                for opt in self.optimizer_list:
                    opt.zero_grad()

                loss.backward()

                if self.attack_runtime is not None:
                    self.attack_runtime.after_backward(local_output_list, y, index)

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
            if self.attack_runtime is not None:
                self.attack_runtime.on_epoch_end()

            epoch_ce_loss = sum(batch_ce_loss_list) / len(batch_ce_loss_list)
            epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
            epoch_anchor_loss = sum(batch_anchor_loss_list) / len(batch_anchor_loss_list)
            epoch_client_anchor_losses = None
            if epoch_client_anchor_loss_sums is not None and epoch_client_anchor_sample_count > 0:
                epoch_client_anchor_losses = {
                    client_id: epoch_client_anchor_loss_sums[client_id] / epoch_client_anchor_sample_count
                    for client_id in sorted(epoch_client_anchor_loss_sums)
                }
                self._record_final_epoch_client_anchor_losses(ep, epoch_client_anchor_losses)
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
                    'target_label': int(self.args.target_label),
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
                '=> End Epoch: {}, early stop epochs: {}, best epoch: {}, best main task accuracy: {:.4f}, test target accuracy: {:.4f}, test asr: {:.4f}, stage3 final accuracy: {:.4f}, stage3 final asr: {:.4f}, anchor loss: {:.4f}'.format(
                    ep + 1,
                    no_change,
                    best_epoch + 1,
                    best_acc,
                    best_metrics.get('test_target', 0.0),
                    best_metrics.get('test_asr', 0.0),
                    best_metrics.get('stage3_final_acc', best_metrics.get('test_acc', 0.0)),
                    best_metrics.get('stage3_final_asr', best_metrics.get('test_asr', 0.0)),
                    best_metrics.get('anchor_loss', 0.0),
                )
            )
            self.logger.info(
                '=> Stage 3 Detection Summary: recall: {:.4f}, precision: {:.4f}, f1: {:.4f}, false positive rate: {:.4f}, correction rate: {:.4f}'.format(
                    best_metrics.get('detection_rate', 0.0),
                    best_metrics.get('detection_precision', 0.0),
                    best_metrics.get('detection_f1', 0.0),
                    best_metrics.get('false_positive_rate', 0.0),
                    best_metrics.get('correction_rate', 0.0),
                )
            )
            if no_change == self.args.early_stop:
                end_time_train = time.time()
                print("The total training time: {}".format((end_time_train - start_time_train)))
                print("The average training time of each epoch: {}".format(((end_time_train - start_time_train)) / (ep + 1)))
                if self.attack_runtime is not None and hasattr(self.attack_runtime, "get_timing_stats"):
                    timing_stats = self.attack_runtime.get_timing_stats()
                    print(
                        "The poison set construction time: {}".format(
                            timing_stats.get("poison_set_construction_time", 0.0)
                        )
                    )
                    print(
                        "The average hard-sample selection time: {}".format(
                            timing_stats.get("hard_sample_selection_time", 0.0) / (ep + 1)
                        )
                    )
                    print(
                        "The total hard-sample selection time: {}".format(
                            timing_stats.get("hard_sample_selection_time", 0.0)
                        )
                    )
                return



    def test(self, ep):
        self.logger.info("=> Test ASR...")
        model_list = self.model_list
        model_list = [model.eval() for model in model_list]
        stage3_enabled = self._stage3_ready()
        if self.anchor_defense is not None and not stage3_enabled:
            self.logger.info("=> Stage 3 detection is skipped because final epoch client anchor losses are unavailable")
        if self.attack_runtime is not None:
            self.attack_runtime.prepare_test_asr_dataset()
        # test main task accuracy
        batch_loss_list = []
        batch_anchor_loss_list = []
        total = 0
        correct = 0
        total_target = 0
        correct_target = 0
        stage3_correct = 0
        stage3_total_target = 0
        stage3_correct_target = 0
        false_positive = 0
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
                if stage3_enabled:
                    stage3_output = self.anchor_defense.run_stage3_detection(local_output_list, global_output)
                    final_predictions = stage3_output["final_predictions"]
                    suspicious_mask = stage3_output["suspicious_mask"]
                    false_positive += suspicious_mask.float().sum().item()
                    stage3_correct += final_predictions.eq(y).sum().item()
                    stage3_total_target += (y == self.args.target_label).float().sum().item()
                    stage3_correct_target += final_predictions.eq(y)[y == self.args.target_label].float().sum().item()

        # test poison accuracy and asr
        total_poison = 0
        correct_poison = 0
        total_asr = 0
        correct_asr = 0
        stage3_correct_poison = 0
        stage3_detected = 0
        stage3_correct_asr = 0
        stage3_original_attack_success = 0
        stage3_detected_attack_success = 0
        stage3_detected_attack_failed_poison = 0
        stage3_corrected_attack_success = 0
        with torch.no_grad():
            for step, (x, x_p, y, index) in enumerate(self.test_asr_loader):
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, global_output = self._forward_batch(x)

                _, predicted = global_output.max(1)
                total_poison += y.size(0)
                correct_poison += predicted.eq(y).sum().item()
                valid_mask = y != self.args.target_label
                total_asr += valid_mask.float().sum().item()
                correct_asr += (predicted[valid_mask] == self.args.target_label).float().sum().item()
                if stage3_enabled:
                    stage3_output = self.anchor_defense.run_stage3_detection(local_output_list, global_output)
                    final_predictions = stage3_output["final_predictions"]
                    suspicious_mask = stage3_output["suspicious_mask"]
                    stage3_correct_poison += final_predictions.eq(y).sum().item()
                    stage3_detected += suspicious_mask[valid_mask].float().sum().item()
                    stage3_correct_asr += (final_predictions[valid_mask] == self.args.target_label).float().sum().item()
                    original_attack_success_mask = valid_mask & predicted.eq(self.args.target_label)
                    original_attack_failed_mask = valid_mask & predicted.ne(self.args.target_label)
                    stage3_original_attack_success += original_attack_success_mask.float().sum().item()
                    stage3_detected_attack_success += suspicious_mask[original_attack_success_mask].float().sum().item()
                    stage3_detected_attack_failed_poison += suspicious_mask[original_attack_failed_mask].float().sum().item()
                    stage3_corrected_attack_success += final_predictions.eq(y)[original_attack_success_mask].float().sum().item()

        # main task accuracy, poison_acc and asr
        test_acc = correct / max(1, total)
        test_poison_accuracy = correct_poison / max(1, total_poison)
        test_asr = correct_asr / max(1, total_asr)
        test_target = correct_target / max(1, total_target)
        stage3_final_acc = stage3_correct / max(1, total) if stage3_enabled else test_acc
        stage3_final_target = stage3_correct_target / max(1, stage3_total_target) if stage3_enabled else test_target
        stage3_final_poison_accuracy = stage3_correct_poison / max(1, total_poison) if stage3_enabled else test_poison_accuracy
        stage3_final_asr = stage3_correct_asr / max(1, total_asr) if stage3_enabled else test_asr
        detection_rate = (
            stage3_detected_attack_success / max(1, stage3_original_attack_success)
            if stage3_enabled else 0.0
        )
        false_positive_rate = false_positive / max(1, total) if stage3_enabled else 0.0
        true_positive = stage3_detected_attack_success if stage3_enabled else 0.0
        false_positive_count = (
            false_positive + stage3_detected_attack_failed_poison
            if stage3_enabled else 0.0
        )
        detection_precision = (
            true_positive / max(1.0, true_positive + false_positive_count)
            if stage3_enabled else 0.0
        )
        detection_recall = detection_rate
        detection_f1 = (
            (2.0 * detection_precision * detection_recall) / max(1e-12, detection_precision + detection_recall)
            if stage3_enabled else 0.0
        )
        correction_rate = (
            stage3_corrected_attack_success / max(1, stage3_original_attack_success)
            if stage3_enabled else 0.0
        )
        epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        anchor_loss = sum(batch_anchor_loss_list) / max(1, len(batch_anchor_loss_list))
        # main task accuracy on target set
        self.logger.info(
            '=> Test Epoch: {}, main task samples: {}, attack samples: {}, test loss: {:.4f}, test main task '
            'accuracy: {:.4f}, test target accuracy: {:.4f}, test asr: {:.4f}, stage3 final accuracy: {:.4f}, '
            'stage3 final target accuracy: {:.4f}, stage3 final asr: {:.4f}, anchor loss: {:.4f}'.format(
                ep + 1,
                len(self.test_loader.dataset),
                len(self.test_asr_loader.dataset),
                epoch_loss,
                test_acc,
                test_target,
                test_asr,
                stage3_final_acc,
                stage3_final_target,
                stage3_final_asr,
                anchor_loss,
            )
        )
        self.logger.info(
            '=> Stage 3 Detection Summary: recall: {:.4f}, precision: {:.4f}, f1: {:.4f}, false positive rate: {:.4f}, correction rate: {:.4f}'.format(
                detection_recall,
                detection_precision,
                detection_f1,
                false_positive_rate,
                correction_rate,
            )
        )

        return {
            'test_acc': test_acc,
            'test_poison_accuracy': test_poison_accuracy,
            'test_target': test_target,
            'test_asr': test_asr,
            'stage3_final_acc': stage3_final_acc,
            'stage3_final_target': stage3_final_target,
            'stage3_final_poison_accuracy': stage3_final_poison_accuracy,
            'stage3_final_asr': stage3_final_asr,
            'detection_rate': detection_recall,
            'detection_precision': detection_precision,
            'detection_f1': detection_f1,
            'false_positive_rate': false_positive_rate,
            'correction_rate': correction_rate,
            'anchor_loss': anchor_loss,
            'epoch_loss': epoch_loss,
        }
