import json
import os
import random
import time

import numpy as np
import torch

from dataset.utils import split_vfl
from utils.utils import get_attack_target_label


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
        defense_runtime=None,
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
        self.attack_runtime = attack_runtime
        self.enable_anchor_loss = getattr(args, "enable_anchor_loss", True)
        self.anchor_defense = defense_runtime
        self.stage3_debug_batches = max(0, int(getattr(args, "stage3_debug_batches", 0)))
        self.stage3_debug_max_samples = max(1, int(getattr(args, "stage3_debug_max_samples", 8)))

    def _get_checkpoint_path(self, file_name):
        return os.path.join(self.args.results_dir, file_name)

    @staticmethod
    def _format_client_anchor_losses(client_anchor_losses):
        return {
            int(client_id): round(float(client_anchor_losses[client_id]), 6)
            for client_id in sorted(client_anchor_losses)
        }

    @staticmethod
    def _format_client_class_updates(client_class_updates):
        return {
            int(client_id): int(client_class_updates[client_id])
            for client_id in sorted(client_class_updates)
        }

    def _record_final_epoch_client_anchor_losses(self, epoch, client_anchor_losses):
        if not client_anchor_losses:
            return
        self.anchor_defense.set_final_epoch_client_anchor_losses(client_anchor_losses)
        os.makedirs(self.args.results_dir, exist_ok=True)
        output_path = os.path.join(self.args.results_dir, "stage2_final_epoch_client_anchor_losses.json")
        payload = {
            "epoch": epoch + 1,
            "client_anchor_losses": {
                str(client_id): float(client_anchor_losses[client_id])
                for client_id in sorted(client_anchor_losses)
            },
        }
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(payload, output_file, indent=2)
        self.logger.info(
            "=> Recorded Stage 2 client anchor losses for epoch %s to '%s': %s",
            epoch + 1,
            output_path,
            self._format_client_anchor_losses(client_anchor_losses),
        )

    def _stage3_ready(self):
        return self.anchor_defense is not None and self.anchor_defense.has_stage3_stats()

    def _topk_correct(self, logits, labels, k):
        k = max(1, min(int(k), logits.size(1)))
        topk_indices = logits.topk(k, dim=1, largest=True, sorted=True).indices
        return topk_indices.eq(labels.unsqueeze(1)).any(dim=1).float().sum().item()

    def _log_stage3_batch_debug(
        self,
        split_name,
        epoch,
        batch_index,
        sample_mask,
        stage3_output,
        predicted,
        final_predictions,
        y,
    ):
        if self.stage3_debug_batches <= 0 or not sample_mask.any():
            return False

        selected_indices = sample_mask.nonzero(as_tuple=False).view(-1)[: self.stage3_debug_max_samples]
        dynamic_vote_scores = stage3_output["dynamic_vote_scores"]
        dynamic_weight_matrix = stage3_output["dynamic_weight_matrix"]
        global_support_counts = stage3_output["global_support_counts"]
        valid_support_counts = stage3_output["valid_support_counts"]
        suspicious_mask = stage3_output["suspicious_mask"]
        correction_applied_mask = stage3_output["correction_applied_mask"]
        correction_labels = stage3_output.get("correction_labels", stage3_output["weighted_labels"])
        correction_tie_mask = stage3_output.get("correction_tie_mask", stage3_output["weighted_tie_mask"])
        top_vote_scores = stage3_output["top_vote_scores"]
        second_vote_scores = stage3_output["second_vote_scores"]
        correction_margin = stage3_output["correction_margin"]
        correction_margin_ok_mask = stage3_output["correction_margin_ok_mask"]
        client_prediction_dict = stage3_output["client_prediction_dict"]

        debug_rows = []
        for sample_index in selected_indices.tolist():
            debug_rows.append(
                {
                    "sample_index_in_batch": int(sample_index),
                    "y": int(y[sample_index].item()),
                    "predicted": int(predicted[sample_index].item()),
                    "correction_labels": int(correction_labels[sample_index].item()),
                    "final_predictions": int(final_predictions[sample_index].item()),
                    "suspicious": bool(suspicious_mask[sample_index].item()),
                    "correction_applied": bool(correction_applied_mask[sample_index].item()),
                    "correction_tie": bool(correction_tie_mask[sample_index].item()),
                    "top_vote_score": round(float(top_vote_scores[sample_index].item()), 6),
                    "second_vote_score": round(float(second_vote_scores[sample_index].item()), 6),
                    "correction_margin": round(float(correction_margin[sample_index].item()), 6),
                    "correction_margin_ok": bool(correction_margin_ok_mask[sample_index].item()),
                    "global_support_counts": int(global_support_counts[sample_index].item()),
                    "valid_support_counts": int(valid_support_counts[sample_index].item()),
                    "client_predictions": {
                        int(client_id): int(client_predictions[sample_index].item())
                        for client_id, client_predictions in sorted(client_prediction_dict.items())
                    },
                    "dynamic_weights": [round(float(value), 6) for value in dynamic_weight_matrix[sample_index].tolist()],
                    "dynamic_vote_scores": [round(float(value), 6) for value in dynamic_vote_scores[sample_index].tolist()],
                }
            )

        self.logger.info(
            "=> Stage 3 Batch Debug [%s] epoch=%s batch=%s samples=%s",
            split_name,
            epoch + 1,
            batch_index + 1,
            json.dumps(debug_rows, ensure_ascii=False),
        )
        return True

    @staticmethod
    def _capture_rng_state():
        rng_state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.random.get_rng_state().cpu(),
            "cuda": None,
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = [state.cpu() for state in torch.cuda.get_rng_state_all()]
        return rng_state

    @staticmethod
    def _safe_metric_value(metrics, key, default=0.0):
        try:
            value = float(metrics.get(key, default))
        except (TypeError, ValueError):
            return float(default)
        return value if np.isfinite(value) else float(default)

    def _compute_best_checkpoint_score(self, metrics):
        clean_acc = self._safe_metric_value(metrics, "clean_acc")
        detection_recall = self._safe_metric_value(
            metrics,
            "detection_recall",
            self._safe_metric_value(metrics, "detection_rate"),
        )
        correction_rate = self._safe_metric_value(metrics, "correction_rate")
        return (clean_acc + detection_recall + correction_rate) / 3.0

    def _build_checkpoint_state(
        self,
        epoch,
        metrics,
        best_clean_acc,
        best_epoch,
        best_metrics,
        best_checkpoint_score,
        no_change,
    ):
        return {
            "epoch": epoch + 1,
            "best_clean_acc": float(best_clean_acc),
            "best_acc": float(best_clean_acc),
            "best_epoch": int(best_epoch) + 1 if best_epoch >= 0 else 0,
            "best_metrics": dict(best_metrics),
            "best_checkpoint_score": float(best_checkpoint_score),
            "best_checkpoint_metric": "mean(clean_acc,detection_recall,correction_rate)",
            "metrics": metrics,
            "no_change": int(no_change),
            "state_dict": [model.state_dict() for model in self.model_list],
            "optimizer": [optimizer.state_dict() for optimizer in self.optimizer_list],
            "anchor_state": self.anchor_defense.to_checkpoint_state() if self.anchor_defense else None,
            "attack_state": (
                self.attack_runtime.to_checkpoint_state()
                if self.attack_runtime is not None and hasattr(self.attack_runtime, "to_checkpoint_state")
                else None
            ),
            "rng_state": self._capture_rng_state(),
        }

    def _save_checkpoint(
        self,
        file_name,
        epoch,
        metrics,
        best_clean_acc,
        best_epoch,
        best_metrics,
        best_checkpoint_score,
        no_change,
        label,
    ):
        os.makedirs(self.args.results_dir, exist_ok=True)
        checkpoint_path = self._get_checkpoint_path(file_name)
        torch.save(
            self._build_checkpoint_state(
                epoch=epoch,
                metrics=metrics,
                best_clean_acc=best_clean_acc,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                best_checkpoint_score=best_checkpoint_score,
                no_change=no_change,
            ),
            checkpoint_path,
        )

    def adjust_learning_rate(self, epoch):
        lr = self.args.lr * (0.1) ** (epoch // 20)
        for opt in self.optimizer_list:
            for param_group in opt.param_groups:
                param_group["lr"] = lr

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
            self.logger.info("=> Start Training...")
        if self.anchor_defense is not None:
            anchor_ema_momentum = getattr(self.args, "anchor_ema_momentum", 0.995)
            lambda_anchor = getattr(self.args, "lambda_anchor", 0.1)
            self.logger.info(
                "=> Enter Stage 2: joint VFL training starts from stage1 passive local backbones with frozen anchor heads"
            )
            self.logger.info("=> Stage 2 anchor loss enabled: %s", self.enable_anchor_loss)
            self.logger.info(
                "=> Stage 2 loss form: total_loss=(1-lambda_anchor)*CE + lambda_anchor*anchor_loss with lambda_anchor=%.4f",
                lambda_anchor,
            )
            if anchor_ema_momentum >= 1.0:
                self.logger.info(
                    "=> Stage 2 EMA anchor calibration disabled: momentum=%.4f makes anchor updates a no-op, update_freq=%s epoch(s)",
                    anchor_ema_momentum,
                    getattr(self.args, "anchor_ema_update_freq", 1),
                )
            else:
                self.logger.info(
                    "=> Stage 2 EMA anchor calibration: momentum=%.4f, update_freq=%s epoch(s)",
                    anchor_ema_momentum,
                    getattr(self.args, "anchor_ema_update_freq", 1),
                )
        self.logger.info("=> Stage 2 latest checkpoint path: '%s'", self._get_checkpoint_path("latest_checkpoint.pth.tar"))
        self.logger.info(
            "=> Stage 2 best checkpoint path: '%s'",
            self._get_checkpoint_path("best_checkpoint.pth.tar"),
        )
        self.logger.info(
            "=> Stage 2 best checkpoint compatibility alias path: '%s'",
            self._get_checkpoint_path("best_clean_checkpoint.pth.tar"),
        )
        self.logger.info(
            "=> Stage 2 best checkpoint metric: mean(clean_acc, detection_recall, correction_rate)"
        )

        epoch_loss_list = []
        model_list = self.model_list
        best_clean_acc = 0
        best_epoch = 0
        best_metrics = {}
        best_checkpoint_score = float("-inf")
        no_change = 0
        if self.checkpoint:
            best_clean_acc = self.checkpoint.get("best_clean_acc", self.checkpoint.get("best_acc", 0.0))
            best_epoch = max(0, int(self.checkpoint.get("best_epoch", self.checkpoint.get("epoch", 1))) - 1)
            best_metrics = dict(self.checkpoint.get("best_metrics", self.checkpoint.get("metrics", {})))
            best_checkpoint_score = float(
                self.checkpoint.get(
                    "best_checkpoint_score",
                    self._compute_best_checkpoint_score(best_metrics) if best_metrics else float("-inf"),
                )
            )
            no_change = int(self.checkpoint.get("no_change", 0))
            self.logger.info(
                "=> Resuming training state: start_epoch=%s, best_epoch=%s, best_checkpoint_score=%.4f, best_clean_acc=%.4f, no_change=%s",
                self.args.start_epoch,
                best_epoch + 1,
                best_checkpoint_score,
                best_clean_acc,
                no_change,
            )
        if self.args.start_epoch >= self.args.epoch:
            self.logger.info(
                "=> Resume checkpoint has already reached epoch %s, which is not smaller than the requested epoch cap %s. Nothing to train.",
                self.args.start_epoch,
                self.args.epoch,
            )
            return

        for ep in range(self.args.start_epoch, self.args.epoch):
            for model in model_list:
                model.train()

            batch_ce_loss_list = []
            batch_loss_list = []
            batch_anchor_loss_list = []
            epoch_client_anchor_loss_sums = None
            epoch_client_anchor_sample_count = 0
            epoch_anchor_statistics = None
            if self.anchor_defense is not None:
                epoch_client_anchor_loss_sums = {client_id: 0.0 for client_id in range(self.args.client_num)}
                epoch_anchor_statistics = self.anchor_defense.create_epoch_anchor_statistics()
            total = 0
            correct = 0
            if self.attack_runtime is not None:
                self.attack_runtime.on_epoch_start(ep)
            if self.attack_runtime is not None and self.attack_runtime.is_enabled():
                self.logger.info("=> Start Training for Injecting Backdoor...")

            for step, (x_n, _x_p, y, index) in enumerate(self.train_loader):
                x = x_n.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, global_output = self._forward_batch(x)
                if self.attack_runtime is not None:
                    self.attack_runtime.before_backward(local_output_list)

                ce_loss = self.criterion(global_output, y)
                anchor_loss = torch.zeros(1, device=self.device).squeeze()
                client_anchor_loss_dict = {}
                if self.anchor_defense is not None:
                    anchor_loss, client_anchor_loss_dict = self.anchor_defense.compute_anchor_loss(
                        local_output_list,
                        y,
                        return_client_losses=True,
                    )
                    self.anchor_defense.accumulate_epoch_anchor_statistics(
                        local_output_list,
                        y,
                        epoch_anchor_statistics,
                    )
                    batch_size = y.size(0)
                    epoch_client_anchor_sample_count += batch_size
                    for client_id, client_anchor_loss in client_anchor_loss_dict.items():
                        epoch_client_anchor_loss_sums[client_id] += client_anchor_loss.item() * batch_size

                if self.anchor_defense is not None and self.enable_anchor_loss:
                    loss = (1.0 - self.args.lambda_anchor) * ce_loss + self.args.lambda_anchor * anchor_loss
                else:
                    loss = ce_loss
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

                _, predicted = global_output.max(1)
                total += y.size(0)
                correct += predicted.eq(y).sum().item()

                train_acc = correct / total
                current_ce_loss = sum(batch_ce_loss_list) / len(batch_ce_loss_list)
                current_loss = sum(batch_loss_list) / len(batch_loss_list)
                current_anchor_loss = sum(batch_anchor_loss_list) / len(batch_anchor_loss_list)

                if step % self.args.print_steps == 0:
                    self.logger.info(
                        "Epoch: {}, {}/{}: VFL loss: {:.4f}, total loss: {:.4f}, anchor loss: {:.4f}, VFL train accuracy: {:.4f}".format(
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
            if self.anchor_defense is not None:
                if (ep + 1) % getattr(self.args, "anchor_ema_update_freq", 1) == 0:
                    anchor_ema_momentum = getattr(self.args, "anchor_ema_momentum", 0.995)
                    ema_update_summary = self.anchor_defense.apply_epoch_anchor_ema(epoch_anchor_statistics)
                    if anchor_ema_momentum >= 1.0:
                        self.logger.info(
                            "=> Stage 2 EMA anchor update skipped at epoch %s because momentum=%.4f disables EMA; eligible classes per client %s",
                            ep + 1,
                            anchor_ema_momentum,
                            self._format_client_class_updates(ema_update_summary),
                        )
                    else:
                        self.logger.info(
                            "=> Stage 2 EMA anchor update applied at epoch %s with momentum %.4f: updated classes per client %s",
                            ep + 1,
                            anchor_ema_momentum,
                            self._format_client_class_updates(ema_update_summary),
                        )
                else:
                    self.logger.info(
                        "=> Stage 2 EMA anchor update skipped at epoch %s because update_freq=%s",
                        ep + 1,
                        getattr(self.args, "anchor_ema_update_freq", 1),
                    )
            epoch_train_acc = correct / max(1, total)
            epoch_loss_list.append(epoch_loss)
            self.logger.info(
                "=> Stage 2 Epoch {} Training Summary: VFL loss: {:.4f}, total loss: {:.4f}, anchor loss: {:.4f}, VFL train accuracy: {:.4f}".format(
                    ep + 1,
                    epoch_ce_loss,
                    epoch_loss,
                    epoch_anchor_loss,
                    epoch_train_acc,
                )
            )
            self.adjust_learning_rate(ep + 1)
            metrics = self.test(ep)
            best_clean_acc = max(best_clean_acc, metrics["clean_acc"])
            checkpoint_score = self._compute_best_checkpoint_score(metrics)
            metrics["best_checkpoint_score"] = checkpoint_score
            self.logger.info(
                "=> Stage 2 checkpoint score: mean(clean_acc={:.4f}, recall={:.4f}, correction_rate={:.4f}) = {:.4f}".format(
                    metrics.get("clean_acc", 0.0),
                    metrics.get("detection_recall", metrics.get("detection_rate", 0.0)),
                    metrics.get("correction_rate", 0.0),
                    checkpoint_score,
                )
            )
            if checkpoint_score > best_checkpoint_score:
                best_checkpoint_score = checkpoint_score
                best_metrics = metrics
                no_change = 0
                best_epoch = ep
                self.logger.info("=> Save best checkpoint model...")
                for best_file_name in ("best_checkpoint.pth.tar", "best_clean_checkpoint.pth.tar"):
                    self._save_checkpoint(
                        file_name=best_file_name,
                        epoch=ep,
                        metrics=metrics,
                        best_clean_acc=best_clean_acc,
                        best_epoch=best_epoch,
                        best_metrics=best_metrics,
                        best_checkpoint_score=best_checkpoint_score,
                        no_change=no_change,
                        label="best",
                    )
            else:
                if ep > self.args.pretrain_stage:
                    no_change += 1
            self._save_checkpoint(
                file_name="latest_checkpoint.pth.tar",
                epoch=ep,
                metrics=metrics,
                best_clean_acc=best_clean_acc,
                best_epoch=best_epoch,
                best_metrics=best_metrics,
                best_checkpoint_score=best_checkpoint_score,
                no_change=no_change,
                label="latest",
            )
            self.logger.info(
                "=> End Epoch: {}, early stop epochs: {}, best epoch: {}, best clean acc: {:.4f}, best Top-{}: {:.4f}, best ASR: {:.4f}, best RAC: {:.4f}, anchor loss: {:.4f}, best checkpoint score: {:.4f}".format(
                    ep + 1,
                    no_change,
                    best_epoch + 1,
                    best_metrics.get("clean_acc", best_clean_acc),
                    max(1, min(int(self.args.top_k), self.model_list[0].num_classes)),
                    best_metrics.get("clean_topk", 0.0),
                    best_metrics.get("asr", 0.0),
                    best_metrics.get("rac", 0.0),
                    best_metrics.get("anchor_loss", 0.0),
                    best_checkpoint_score,
                )
            )
            self.logger.info(
                "=> Stage 3 Detection Summary: recall: {:.4f}, precision: {:.4f}, f1: {:.4f}, false positive rate: {:.4f}, correction rate: {:.4f}".format(
                    best_metrics.get("detection_rate", 0.0),
                    best_metrics.get("detection_precision", 0.0),
                    best_metrics.get("detection_f1", 0.0),
                    best_metrics.get("false_positive_rate", 0.0),
                    best_metrics.get("correction_rate", 0.0),
                )
            )
            if self.args.early_stop > 0 and no_change >= self.args.early_stop:
                end_time_train = time.time()
                print("The total training time: {}".format(end_time_train - start_time_train))
                print("The average training time of each epoch: {}".format((end_time_train - start_time_train) / (ep + 1)))
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
        detection_enabled = stage3_enabled
        remaining_stage3_debug_batches = self.stage3_debug_batches
        if self.anchor_defense is not None and not stage3_enabled:
            self.logger.info("=> Stage 3 detection is skipped because final epoch client anchor losses are unavailable")
        if self.attack_runtime is not None:
            self.attack_runtime.prepare_test_asr_dataset()
        attack_target_label = get_attack_target_label(self.args)

        batch_loss_list = []
        batch_anchor_loss_list = []
        total = 0
        correct = 0
        clean_topk_correct = 0
        total_target = 0
        correct_target = 0
        stage3_correct = 0
        stage3_total_target = 0
        stage3_correct_target = 0
        false_positive = 0
        stage3_debug_stats = {
            "clean_suspicious": 0,
            "clean_correction_applied": 0,
            "clean_prediction_changed": 0,
            "clean_skipped_tied_vote": 0,
            "clean_skipped_low_margin": 0,
            "clean_skipped_other": 0,
            "poison_valid_suspicious": 0,
            "poison_valid_correction_applied": 0,
            "poison_valid_prediction_changed": 0,
            "poison_valid_skipped_tied_vote": 0,
            "poison_valid_skipped_low_margin": 0,
            "poison_valid_skipped_other": 0,
            "attack_success_total": 0,
            "attack_success_suspicious": 0,
            "attack_success_correction_applied": 0,
            "attack_success_prediction_changed": 0,
        }
        with torch.no_grad():
            for batch_index, (x, _x_p, y, _index) in enumerate(self.test_loader):
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
                clean_topk_correct += self._topk_correct(global_output, y, self.args.top_k)
                total_target += (y == attack_target_label).float().sum().item()
                correct_target += predicted.eq(y)[y == attack_target_label].float().sum().item()
                if stage3_enabled:
                    stage3_output = self.anchor_defense.run_stage3_detection(local_output_list, global_output)
                    final_predictions = stage3_output["final_predictions"]
                    suspicious_mask = stage3_output["suspicious_mask"]
                    correction_applied_mask = stage3_output["correction_applied_mask"]
                    prediction_changed_mask = final_predictions.ne(predicted)
                    correction_tie_mask = stage3_output.get("correction_tie_mask", stage3_output["weighted_tie_mask"])
                    correction_margin_rejected_mask = stage3_output["correction_margin_rejected_mask"]
                    skipped_tied_vote_mask = suspicious_mask & ~correction_applied_mask & correction_tie_mask
                    skipped_low_margin_mask = correction_margin_rejected_mask
                    skipped_other_mask = (
                        suspicious_mask
                        & ~correction_applied_mask
                        & ~correction_tie_mask
                        & ~correction_margin_rejected_mask
                    )
                    false_positive += suspicious_mask.float().sum().item()
                    stage3_correct += final_predictions.eq(y).sum().item()
                    stage3_total_target += (y == attack_target_label).float().sum().item()
                    stage3_correct_target += final_predictions.eq(y)[y == attack_target_label].float().sum().item()
                    stage3_debug_stats["clean_suspicious"] += suspicious_mask.float().sum().item()
                    stage3_debug_stats["clean_correction_applied"] += correction_applied_mask.float().sum().item()
                    stage3_debug_stats["clean_prediction_changed"] += prediction_changed_mask.float().sum().item()
                    stage3_debug_stats["clean_skipped_tied_vote"] += skipped_tied_vote_mask.float().sum().item()
                    stage3_debug_stats["clean_skipped_low_margin"] += skipped_low_margin_mask.float().sum().item()
                    stage3_debug_stats["clean_skipped_other"] += skipped_other_mask.float().sum().item()
                    if remaining_stage3_debug_batches > 0:
                        clean_debug_mask = suspicious_mask
                        if self._log_stage3_batch_debug(
                            split_name="clean-suspicious",
                            epoch=ep,
                            batch_index=batch_index,
                            sample_mask=clean_debug_mask,
                            stage3_output=stage3_output,
                            predicted=predicted,
                            final_predictions=final_predictions,
                            y=y,
                        ):
                            remaining_stage3_debug_batches -= 1
        total_poison = 0
        correct_poison = 0
        total_asr = 0
        correct_asr = 0
        poison_topk_asr = 0
        stage3_correct_poison = 0
        stage3_detected = 0
        stage3_correct_asr = 0
        stage3_original_attack_success = 0
        stage3_detected_attack_success = 0
        stage3_detected_attack_failed_poison = 0
        stage3_corrected_attack_success = 0
        with torch.no_grad():
            for batch_index, (x, _x_p, y, _index) in enumerate(self.test_asr_loader):
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, global_output = self._forward_batch(x)

                _, predicted = global_output.max(1)
                total_poison += y.size(0)
                correct_poison += predicted.eq(y).sum().item()
                valid_mask = y != attack_target_label
                total_asr += valid_mask.float().sum().item()
                correct_asr += (predicted[valid_mask] == attack_target_label).float().sum().item()
                if valid_mask.any():
                    topk_indices = global_output[valid_mask].topk(
                        max(1, min(int(self.args.top_k), global_output.size(1))),
                        dim=1,
                        largest=True,
                        sorted=True,
                    ).indices
                    poison_topk_asr += topk_indices.eq(attack_target_label).any(dim=1).float().sum().item()
                if stage3_enabled:
                    stage3_output = self.anchor_defense.run_stage3_detection(local_output_list, global_output)
                    final_predictions = stage3_output["final_predictions"]
                    suspicious_mask = stage3_output["suspicious_mask"]
                    correction_applied_mask = stage3_output["correction_applied_mask"]
                    prediction_changed_mask = final_predictions.ne(predicted)
                    correction_tie_mask = stage3_output.get("correction_tie_mask", stage3_output["weighted_tie_mask"])
                    correction_margin_rejected_mask = stage3_output["correction_margin_rejected_mask"]
                    skipped_tied_vote_mask = suspicious_mask & ~correction_applied_mask & correction_tie_mask
                    skipped_low_margin_mask = correction_margin_rejected_mask
                    skipped_other_mask = (
                        suspicious_mask
                        & ~correction_applied_mask
                        & ~correction_tie_mask
                        & ~correction_margin_rejected_mask
                    )
                    stage3_correct_poison += final_predictions.eq(y).sum().item()
                    stage3_detected += suspicious_mask[valid_mask].float().sum().item()
                    stage3_correct_asr += (final_predictions[valid_mask] == attack_target_label).float().sum().item()
                    original_attack_success_mask = valid_mask & predicted.eq(attack_target_label)
                    original_attack_failed_mask = valid_mask & predicted.ne(attack_target_label)
                    stage3_original_attack_success += original_attack_success_mask.float().sum().item()
                    stage3_detected_attack_success += suspicious_mask[original_attack_success_mask].float().sum().item()
                    stage3_detected_attack_failed_poison += suspicious_mask[original_attack_failed_mask].float().sum().item()
                    stage3_corrected_attack_success += final_predictions.eq(y)[original_attack_success_mask].float().sum().item()
                    stage3_debug_stats["poison_valid_suspicious"] += suspicious_mask[valid_mask].float().sum().item()
                    stage3_debug_stats["poison_valid_correction_applied"] += (
                        correction_applied_mask[valid_mask].float().sum().item()
                    )
                    stage3_debug_stats["poison_valid_prediction_changed"] += (
                        prediction_changed_mask[valid_mask].float().sum().item()
                    )
                    stage3_debug_stats["poison_valid_skipped_tied_vote"] += (
                        skipped_tied_vote_mask[valid_mask].float().sum().item()
                    )
                    stage3_debug_stats["poison_valid_skipped_low_margin"] += (
                        skipped_low_margin_mask[valid_mask].float().sum().item()
                    )
                    stage3_debug_stats["poison_valid_skipped_other"] += (
                        skipped_other_mask[valid_mask].float().sum().item()
                    )
                    stage3_debug_stats["attack_success_total"] += original_attack_success_mask.float().sum().item()
                    stage3_debug_stats["attack_success_suspicious"] += (
                        suspicious_mask[original_attack_success_mask].float().sum().item()
                    )
                    stage3_debug_stats["attack_success_correction_applied"] += (
                        correction_applied_mask[original_attack_success_mask].float().sum().item()
                    )
                    stage3_debug_stats["attack_success_prediction_changed"] += (
                        prediction_changed_mask[original_attack_success_mask].float().sum().item()
                    )
                    if remaining_stage3_debug_batches > 0:
                        attack_success_debug_mask = original_attack_success_mask & suspicious_mask
                        if self._log_stage3_batch_debug(
                            split_name="attack-success-suspicious",
                            epoch=ep,
                            batch_index=batch_index,
                            sample_mask=attack_success_debug_mask,
                            stage3_output=stage3_output,
                            predicted=predicted,
                            final_predictions=final_predictions,
                            y=y,
                        ):
                            remaining_stage3_debug_batches -= 1
        pre_stage3_acc = correct / max(1, total)
        clean_topk = clean_topk_correct / max(1, total)
        test_poison_accuracy = correct_poison / max(1, total_poison)
        test_asr = correct_asr / max(1, total_asr)
        test_topk_asr = poison_topk_asr / max(1, total_asr)
        test_target = correct_target / max(1, total_target)
        stage3_final_acc = stage3_correct / max(1, total) if stage3_enabled else pre_stage3_acc
        stage3_final_target = stage3_correct_target / max(1, stage3_total_target) if stage3_enabled else test_target
        stage3_final_poison_accuracy = stage3_correct_poison / max(1, total_poison) if stage3_enabled else test_poison_accuracy
        stage3_final_asr = stage3_correct_asr / max(1, total_asr) if stage3_enabled else test_asr
        detection_rate = stage3_detected_attack_success / max(1, stage3_original_attack_success) if detection_enabled else 0.0
        false_positive_rate = false_positive / max(1, total) if detection_enabled else 0.0
        true_positive = stage3_detected_attack_success if detection_enabled else 0.0
        false_positive_count = false_positive + stage3_detected_attack_failed_poison if detection_enabled else 0.0
        detection_precision = (
            true_positive / max(1.0, true_positive + false_positive_count) if detection_enabled else 0.0
        )
        detection_recall = detection_rate
        detection_f1 = (
            (2.0 * detection_precision * detection_recall) / max(1e-12, detection_precision + detection_recall)
            if detection_enabled
            else 0.0
        )
        # CR is defined over successfully detected attack-success samples.
        correction_rate = (
            stage3_corrected_attack_success / max(1, stage3_detected_attack_success) if detection_enabled else 0.0
        )
        epoch_loss = sum(batch_loss_list) / len(batch_loss_list)
        anchor_loss = sum(batch_anchor_loss_list) / max(1, len(batch_anchor_loss_list))
        final_clean_acc = stage3_final_acc if stage3_enabled else pre_stage3_acc
        final_asr = stage3_final_asr if stage3_enabled else test_asr
        final_rac = stage3_final_poison_accuracy if stage3_enabled else test_poison_accuracy
        self.logger.info(
            "=> Test Epoch: {}, main task samples: {}, attack samples: {}, test loss: {:.4f}, clean acc: {:.4f}, Top-{}: {:.4f}, ASR: {:.4f}, RAC: {:.4f}, test target accuracy: {:.4f}, stage3 final accuracy: {:.4f}, stage3 final target accuracy: {:.4f}, stage3 final asr: {:.4f}, anchor loss: {:.4f}".format(
                ep + 1,
                len(self.test_loader.dataset),
                len(self.test_asr_loader.dataset),
                epoch_loss,
                final_clean_acc,
                max(1, min(int(self.args.top_k), self.model_list[0].num_classes)),
                clean_topk,
                final_asr,
                final_rac,
                test_target,
                stage3_final_acc,
                stage3_final_target,
                stage3_final_asr,
                anchor_loss,
            )
        )
        self.logger.info(
            "=> Stage 3 Detection Summary: recall: {:.4f}, precision: {:.4f}, f1: {:.4f}, false positive rate: {:.4f}, correction rate: {:.4f}".format(
                detection_recall,
                detection_precision,
                detection_f1,
                false_positive_rate,
                correction_rate,
            )
        )
        if stage3_enabled:
            self.logger.info(
                "=> Stage 3 Debug (clean): suspicious=%s/%s, correction_applied=%s, prediction_changed=%s, skipped_tied_vote=%s, skipped_low_margin=%s, skipped_other=%s",
                int(stage3_debug_stats["clean_suspicious"]),
                int(total),
                int(stage3_debug_stats["clean_correction_applied"]),
                int(stage3_debug_stats["clean_prediction_changed"]),
                int(stage3_debug_stats["clean_skipped_tied_vote"]),
                int(stage3_debug_stats["clean_skipped_low_margin"]),
                int(stage3_debug_stats["clean_skipped_other"]),
            )
            self.logger.info(
                "=> Stage 3 Debug (poison-valid): suspicious=%s/%s, correction_applied=%s, prediction_changed=%s, skipped_tied_vote=%s, skipped_low_margin=%s, skipped_other=%s",
                int(stage3_debug_stats["poison_valid_suspicious"]),
                int(total_asr),
                int(stage3_debug_stats["poison_valid_correction_applied"]),
                int(stage3_debug_stats["poison_valid_prediction_changed"]),
                int(stage3_debug_stats["poison_valid_skipped_tied_vote"]),
                int(stage3_debug_stats["poison_valid_skipped_low_margin"]),
                int(stage3_debug_stats["poison_valid_skipped_other"]),
            )
            self.logger.info(
                "=> Stage 3 Debug (attack-success): suspicious=%s/%s, correction_applied=%s, prediction_changed=%s, corrected_to_true=%s",
                int(stage3_debug_stats["attack_success_suspicious"]),
                int(stage3_debug_stats["attack_success_total"]),
                int(stage3_debug_stats["attack_success_correction_applied"]),
                int(stage3_debug_stats["attack_success_prediction_changed"]),
                int(stage3_corrected_attack_success),
            )

        return {
            "pre_stage3_acc": pre_stage3_acc,
            "clean_acc": final_clean_acc,
            "clean_topk": clean_topk,
            "test_poison_accuracy": test_poison_accuracy,
            "rac": final_rac,
            "test_target": test_target,
            "test_asr": test_asr,
            "asr": final_asr,
            "test_topk_asr": test_topk_asr,
            "stage3_final_acc": stage3_final_acc,
            "stage3_final_target": stage3_final_target,
            "stage3_final_poison_accuracy": stage3_final_poison_accuracy,
            "stage3_final_asr": stage3_final_asr,
            "detection_rate": detection_recall,
            "detection_recall": detection_recall,
            "detection_precision": detection_precision,
            "detection_f1": detection_f1,
            "false_positive_rate": false_positive_rate,
            "correction_rate": correction_rate,
            "anchor_loss": anchor_loss,
            "epoch_loss": epoch_loss,
        }
