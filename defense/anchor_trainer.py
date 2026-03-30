import json
import os

import torch
import torch.nn.functional as F

from attack.runtime import build_stage1_private_poisoned_loaders
from dataset.utils import split_vfl
from defense.anchor_defense import AnchorDefense
from defense.anchor_utils import build_anchor_heads, get_stage1_dir
from defense.anchor_losses import compute_single_client_anchor_loss


class AnchorPretrainer:
    def __init__(self, device, args, logger):
        self.device = device
        self.args = args
        self.logger = logger

    def _build_stage1_attacker_loaders(self, train_loader, test_loader, trigger_dimensions):
        return build_stage1_private_poisoned_loaders(
            self.args,
            self.logger,
            train_loader,
            test_loader,
            trigger_dimensions,
        )

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

        with torch.no_grad():
            for x_n, _, y, _ in data_loader:
                x = x_n.to(self.device).float()
                y = y.to(self.device).long()
                x_split_list = split_vfl(x, self.args)
                features = local_model(x_split_list[client_id])
                logits, _ = head(features)
                predicted = logits.argmax(dim=1)
                non_target_mask = y != self.args.target_label
                if non_target_mask.any():
                    total += non_target_mask.float().sum().item()
                    success += (predicted[non_target_mask] == self.args.target_label).float().sum().item()

        return success / max(1, total)

    def pretrain(self, model_list, train_loader, test_loader, trigger_dimensions):
        self.logger.info(
            "=> Enter Stage 1: passive local backbones are trained with active-party private two-layer MLP heads"
        )
        local_models = model_list[1:]
        # Active party owns one private two-layer head for each passive branch.
        heads = build_anchor_heads(model_list, self.args, self.device)
        attacker_train_loader, attacker_test_loader = self._build_stage1_attacker_loaders(
            train_loader,
            test_loader,
            trigger_dimensions,
        )
        stage1_metrics = {}

        for client_id, (local_model, head) in enumerate(zip(local_models, heads)):
            optimizer = torch.optim.Adam(list(local_model.parameters()) + list(head.parameters()), lr=self.args.lr)
            local_model.train()
            head.train()
            client_train_loader = train_loader
            if attacker_train_loader is not None and client_id == self.args.attack_client_num:
                # Only the attacker branch uses poisoned inputs in stage 1.
                client_train_loader = attacker_train_loader

            self.logger.info("=> Stage 1 start for passive client %s", client_id)

            for epoch in range(self.args.anchor_pretrain_epochs):
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

                stage1_metrics[client_id] = {
                    "clean_acc": clean_metrics["clean_acc"],
                    "anchor_loss": clean_metrics["anchor_loss"],
                    "backdoor_acc": backdoor_acc,
                }

            # Print an explicit per-passive-party stage 1 summary so the final accuracy is easy to find.
            self.logger.info(
                "=> Stage 1 final summary for passive client %s: clean acc: %.4f, anchor loss: %.4f, backdoor acc: %.4f",
                client_id,
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
        )
        stage1_dir = get_stage1_dir(self.args)
        anchor_defense.save_stage1_artifacts(model_list, stage1_dir)
        with open(os.path.join(stage1_dir, "stage1_metrics.json"), "w", encoding="utf-8") as metric_file:
            json.dump(stage1_metrics, metric_file, indent=2)
        self.logger.info("=> Stage 1 metrics were also saved to '%s'", os.path.join(stage1_dir, "stage1_metrics.json"))
        self.logger.info("=> Saved stage1 passive-party local models and anchors to '%s'", stage1_dir)
        return anchor_defense
