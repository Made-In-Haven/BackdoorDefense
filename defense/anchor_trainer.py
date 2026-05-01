import copy
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from attack.runtime import build_stage1_private_poisoned_loaders
from dataset.utils import split_vfl
from defense.anchor_defense import AnchorDefense
from defense.anchor_utils import build_anchor_heads, get_stage1_dir
from defense.anchor_losses import compute_single_client_anchor_loss
from utils.utils import get_attack_target_label


class AnchorPretrainer:
    def __init__(self, device, args, logger):
        self.device = device
        self.args = args
        self.logger = logger

    def _get_stage1_latest_checkpoint_path(self):
        return os.path.join(self.args.results_dir, "latest_checkpoint.pth.tar")

    def _build_stage1_attacker_loaders(self, train_loader, test_loader, trigger_dimensions):
        return build_stage1_private_poisoned_loaders(
            self.args,
            self.logger,
            train_loader,
            test_loader,
            trigger_dimensions,
        )

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

    def _move_optimizer_state_to_device(self, optimizer):
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(self.device)

    def _save_stage1_latest_checkpoint(
        self,
        model_list,
        heads,
        next_client_id,
        next_epoch,
        optimizer_state,
        stage1_metrics,
        client_progress=None,
    ):
        os.makedirs(self.args.results_dir, exist_ok=True)
        checkpoint_path = self._get_stage1_latest_checkpoint_path()
        torch.save(
            {
                "epoch": int(next_epoch),
                "best_acc": 0.0,
                "metrics": {
                    "stage1_next_client_id": int(next_client_id),
                    "stage1_next_epoch": int(next_epoch),
                },
                "state_dict": [model.state_dict() for model in model_list],
                "optimizer": [],
                "stage1_state": {
                    "next_client_id": int(next_client_id),
                    "next_epoch": int(next_epoch),
                    "heads": [head.state_dict() for head in heads],
                    "optimizer_state": optimizer_state,
                    "stage1_metrics": dict(stage1_metrics),
                    "client_progress": dict(client_progress or {}),
                },
                "rng_state": self._capture_rng_state(),
            },
            checkpoint_path,
        )

    def _load_stage1_resume_state(self, checkpoint, heads, client_count):
        stage1_state = checkpoint.get("stage1_state") if isinstance(checkpoint, dict) else None
        if not isinstance(stage1_state, dict):
            return 0, 0, None, {}, {}

        head_state_list = stage1_state.get("heads")
        if isinstance(head_state_list, list):
            for head, state_dict in zip(heads, head_state_list):
                head.load_state_dict(state_dict)

        next_client_id = max(0, int(stage1_state.get("next_client_id", 0)))
        next_epoch = max(0, int(stage1_state.get("next_epoch", 0)))
        optimizer_state = stage1_state.get("optimizer_state")
        raw_stage1_metrics = stage1_state.get("stage1_metrics", {})
        stage1_metrics = {
            int(client_id): client_metrics
            for client_id, client_metrics in raw_stage1_metrics.items()
        }
        raw_client_progress = stage1_state.get("client_progress", {})
        client_progress = {
            int(client_id): progress
            for client_id, progress in raw_client_progress.items()
        }

        if next_client_id > client_count:
            next_client_id = client_count
            next_epoch = 0
            optimizer_state = None

        return next_client_id, next_epoch, optimizer_state, stage1_metrics, client_progress

    @staticmethod
    def _is_better_stage1_metric(clean_metrics, best_progress, min_delta):
        best_clean_acc = float(best_progress.get("best_clean_acc", float("-inf")))
        current_clean_acc = float(clean_metrics["clean_acc"])
        if current_clean_acc > best_clean_acc + min_delta:
            return True
        if abs(current_clean_acc - best_clean_acc) <= min_delta:
            best_anchor_loss = float(best_progress.get("best_anchor_loss", float("inf")))
            return float(clean_metrics["anchor_loss"]) < best_anchor_loss - 1e-12
        return False

    @staticmethod
    def _snapshot_state_dict(module):
        return copy.deepcopy(module.state_dict())

    def _build_default_client_progress(self, local_model, head):
        return {
            "best_clean_acc": float("-inf"),
            "best_anchor_loss": float("inf"),
            "best_backdoor_acc": 0.0,
            "best_epoch": -1,
            "no_improve_count": 0,
            "best_local_model_state": self._snapshot_state_dict(local_model),
            "best_head_state": self._snapshot_state_dict(head),
        }

    def _evaluate_clean_metrics(self, local_model, head, data_loader, client_id):
        local_model.eval()
        head.eval()

        total = 0
        correct = 0
        anchor_loss_total = 0.0

        with torch.no_grad():
            for x_n, _, y, _ in data_loader:
                x = x_n.to(self.device).float()
                y = y.to(self.device).long()
                x_split_list = split_vfl(x, self.args)
                features = local_model(x_split_list[client_id])
                logits, _ = head(features)
                anchor_loss = compute_single_client_anchor_loss(head, features, y)

                total += y.size(0)
                correct += logits.argmax(dim=1).eq(y).sum().item()
                anchor_loss_total += anchor_loss.item() * y.size(0)

        return {
            "clean_acc": correct / max(1, total),
            "anchor_loss": anchor_loss_total / max(1, total),
        }

    def _evaluate_backdoor_asr(self, local_model, head, data_loader, client_id):
        if data_loader is None:
            return 0.0

        local_model.eval()
        head.eval()
        total = 0
        success = 0
        attack_target_label = get_attack_target_label(self.args)

        with torch.no_grad():
            for x_n, _, y, _ in data_loader:
                x = x_n.to(self.device).float()
                y = y.to(self.device).long()
                x_split_list = split_vfl(x, self.args)
                features = local_model(x_split_list[client_id])
                logits, _ = head(features)
                predicted = logits.argmax(dim=1)
                non_target_mask = y != attack_target_label
                if non_target_mask.any():
                    total += non_target_mask.float().sum().item()
                    success += (predicted[non_target_mask] == attack_target_label).float().sum().item()

        return success / max(1, total)

    def pretrain(self, model_list, train_loader, test_loader, trigger_dimensions, checkpoint=None):
        self.logger.info(
            "=> Enter Stage 1: passive local backbones are trained with active-party private two-layer MLP heads"
        )
        self.logger.info("=> Stage 1 latest checkpoint path: '%s'", self._get_stage1_latest_checkpoint_path())
        local_models = model_list[1:]
        # Active party owns one private two-layer head for each passive branch.
        heads = build_anchor_heads(model_list, self.args, self.device)
        attacker_train_loader, attacker_test_loader = self._build_stage1_attacker_loaders(
            train_loader,
            test_loader,
            trigger_dimensions,
        )
        (
            resume_client_id,
            resume_epoch,
            resume_optimizer_state,
            stage1_metrics,
            client_progress_state,
        ) = self._load_stage1_resume_state(
            checkpoint,
            heads,
            len(local_models),
        )
        if resume_client_id < len(local_models) or resume_epoch > 0:
            if resume_client_id > 0 or resume_epoch > 0:
                self.logger.info(
                    "=> Resuming Stage 1 from passive client %s, epoch %s/%s",
                    resume_client_id,
                    resume_epoch + 1,
                    self.args.anchor_pretrain_epochs,
                )
        else:
            self.logger.info("=> Stage 1 checkpoint already reached the end of all passive clients; finalizing artifacts")

        for client_id, (local_model, head) in enumerate(zip(local_models, heads)):
            if client_id < resume_client_id:
                self.logger.info("=> Stage 1 passive client %s already completed in checkpoint, skipping", client_id)
                continue

            optimizer = torch.optim.Adam(list(local_model.parameters()) + list(head.parameters()), lr=self.args.lr)
            start_epoch = 0
            if client_id == resume_client_id:
                start_epoch = resume_epoch
                if resume_optimizer_state is not None and start_epoch > 0:
                    optimizer.load_state_dict(resume_optimizer_state)
                    self._move_optimizer_state_to_device(optimizer)
            current_client_progress = client_progress_state.get(client_id)
            if not isinstance(current_client_progress, dict):
                current_client_progress = self._build_default_client_progress(local_model, head)
            else:
                current_client_progress.setdefault("best_clean_acc", float("-inf"))
                current_client_progress.setdefault("best_anchor_loss", float("inf"))
                current_client_progress.setdefault("best_backdoor_acc", 0.0)
                current_client_progress.setdefault("best_epoch", -1)
                current_client_progress.setdefault("no_improve_count", 0)
                current_client_progress.setdefault(
                    "best_local_model_state",
                    self._snapshot_state_dict(local_model),
                )
                current_client_progress.setdefault(
                    "best_head_state",
                    self._snapshot_state_dict(head),
                )
            resume_optimizer_state = None
            local_model.train()
            head.train()
            client_train_loader = train_loader
            if attacker_train_loader is not None and client_id == self.args.attack_client_num:
                # Only the attacker branch uses poisoned inputs in stage 1.
                client_train_loader = attacker_train_loader

            self.logger.info("=> Stage 1 start for passive client %s", client_id)
            if start_epoch > 0:
                self.logger.info(
                    "=> Continuing passive client %s from epoch %s/%s",
                    client_id,
                    start_epoch + 1,
                    self.args.anchor_pretrain_epochs,
                )
            stage1_patience = max(0, int(getattr(self.args, "anchor_pretrain_early_stop", 5)))
            stage1_min_delta = max(0.0, float(getattr(self.args, "anchor_pretrain_min_delta", 1e-4)))
            if stage1_patience > 0:
                self.logger.info(
                    "=> Stage 1 early stop for passive client %s: patience=%s, min_delta=%.6f",
                    client_id,
                    stage1_patience,
                    stage1_min_delta,
                )

            for epoch in range(start_epoch, self.args.anchor_pretrain_epochs):
                epoch_ce_loss = 0.0
                epoch_anchor_loss = 0.0
                epoch_total_loss = 0.0
                total = 0
                for x_n, _, y, _ in client_train_loader:
                    x = x_n.to(self.device).float()
                    y = y.to(self.device).long()
                    x_split_list = split_vfl(x, self.args)
                    features = local_model(x_split_list[client_id])
                    logits, _ = head(features, y)
                    # Stage 1 optimizes the local branch using its active-side private head.
                    ce_loss = F.cross_entropy(logits, y)
                    anchor_loss = compute_single_client_anchor_loss(head, features, y)
                    total_loss = ce_loss + self.args.lambda_stage1_anchor * anchor_loss

                    optimizer.zero_grad()
                    total_loss.backward()
                    optimizer.step()

                    epoch_ce_loss += ce_loss.item() * y.size(0)
                    epoch_anchor_loss += anchor_loss.item() * y.size(0)
                    epoch_total_loss += total_loss.item() * y.size(0)
                    total += y.size(0)

                clean_metrics = self._evaluate_clean_metrics(local_model, head, test_loader, client_id)
                backdoor_acc = 0.0
                if attacker_test_loader is not None and client_id == self.args.attack_client_num:
                    backdoor_acc = self._evaluate_backdoor_asr(local_model, head, attacker_test_loader, client_id)

                self.logger.info(
                    "=> Stage 1 client %s epoch %s/%s, ce loss: %.4f, train anchor loss: %.4f, total loss: %.4f, clean acc: %.4f, eval anchor loss: %.4f, backdoor acc: %.4f",
                    client_id,
                    epoch + 1,
                    self.args.anchor_pretrain_epochs,
                    epoch_ce_loss / max(1, total),
                    epoch_anchor_loss / max(1, total),
                    epoch_total_loss / max(1, total),
                    clean_metrics["clean_acc"],
                    clean_metrics["anchor_loss"],
                    backdoor_acc,
                )

                improved = self._is_better_stage1_metric(clean_metrics, current_client_progress, stage1_min_delta)
                if improved:
                    current_client_progress["best_clean_acc"] = float(clean_metrics["clean_acc"])
                    current_client_progress["best_anchor_loss"] = float(clean_metrics["anchor_loss"])
                    current_client_progress["best_backdoor_acc"] = float(backdoor_acc)
                    current_client_progress["best_epoch"] = int(epoch)
                    current_client_progress["no_improve_count"] = 0
                    current_client_progress["best_local_model_state"] = self._snapshot_state_dict(local_model)
                    current_client_progress["best_head_state"] = self._snapshot_state_dict(head)
                else:
                    current_client_progress["no_improve_count"] = int(current_client_progress["no_improve_count"]) + 1

                stage1_metrics[client_id] = {
                    "clean_acc": clean_metrics["clean_acc"],
                    "anchor_loss": clean_metrics["anchor_loss"],
                    "backdoor_acc": backdoor_acc,
                }
                client_progress_state[client_id] = current_client_progress
                next_client_id = client_id
                next_epoch = epoch + 1
                optimizer_state = optimizer.state_dict()
                if next_epoch >= self.args.anchor_pretrain_epochs:
                    next_client_id = client_id + 1
                    next_epoch = 0
                    optimizer_state = None
                self._save_stage1_latest_checkpoint(
                    model_list=model_list,
                    heads=heads,
                    next_client_id=next_client_id,
                    next_epoch=next_epoch,
                    optimizer_state=optimizer_state,
                    stage1_metrics=stage1_metrics,
                    client_progress=client_progress_state,
                )

                if stage1_patience > 0 and int(current_client_progress["no_improve_count"]) >= stage1_patience:
                    self.logger.info(
                        "=> Stage 1 early stop triggered for passive client %s at epoch %s/%s; best clean acc: %.4f at epoch %s",
                        client_id,
                        epoch + 1,
                        self.args.anchor_pretrain_epochs,
                        float(current_client_progress["best_clean_acc"]),
                        int(current_client_progress["best_epoch"]) + 1,
                    )
                    break

            best_local_model_state = current_client_progress.get("best_local_model_state")
            best_head_state = current_client_progress.get("best_head_state")
            if best_local_model_state is not None:
                local_model.load_state_dict(best_local_model_state)
            if best_head_state is not None:
                head.load_state_dict(best_head_state)
            stage1_metrics[client_id] = {
                "clean_acc": float(current_client_progress.get("best_clean_acc", 0.0)),
                "anchor_loss": float(current_client_progress.get("best_anchor_loss", 0.0)),
                "backdoor_acc": float(current_client_progress.get("best_backdoor_acc", 0.0)),
                "best_epoch": int(current_client_progress.get("best_epoch", -1)) + 1,
            }
            client_progress_state.pop(client_id, None)
            self._save_stage1_latest_checkpoint(
                model_list=model_list,
                heads=heads,
                next_client_id=client_id + 1,
                next_epoch=0,
                optimizer_state=None,
                stage1_metrics=stage1_metrics,
                client_progress=client_progress_state,
            )

            # Print an explicit per-passive-party stage 1 summary so the final accuracy is easy to find.
            self.logger.info(
                "=> Stage 1 final summary for passive client %s: best epoch: %s, clean acc: %.4f, anchor loss: %.4f, backdoor acc: %.4f",
                client_id,
                stage1_metrics[client_id]["best_epoch"],
                stage1_metrics[client_id]["clean_acc"],
                stage1_metrics[client_id]["anchor_loss"],
                stage1_metrics[client_id]["backdoor_acc"],
            )

        anchor_defense = AnchorDefense.from_heads(
            heads=heads,
            model_list=model_list,
            device=self.device,
            args=self.args,
            logger=self.logger,
            stage1_metrics=stage1_metrics,
        )
        stage1_dir = get_stage1_dir(self.args)
        anchor_defense.save_stage1_artifacts(model_list, stage1_dir)
        with open(os.path.join(stage1_dir, "stage1_metrics.json"), "w", encoding="utf-8") as metric_file:
            json.dump(stage1_metrics, metric_file, indent=2)
        self.logger.info("=> Stage 1 metrics were also saved to '%s'", os.path.join(stage1_dir, "stage1_metrics.json"))
        self.logger.info("=> Saved stage1 passive-party local models and anchors to '%s'", stage1_dir)
        return anchor_defense
