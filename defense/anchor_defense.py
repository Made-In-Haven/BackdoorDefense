import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset.utils import split_vfl
from defense.anchor_losses import compute_anchor_constraint_loss
from defense.anchor_utils import build_anchor_heads, ensure_dir, get_stage1_dir, save_anchor_artifact
from defense.detector import MultiClientDetector
from utils.utils import load_torch_artifact

STAGE1_ARTIFACT_VERSION = 3
STAGE1_ARTIFACT_STRATEGY = "active_party_private_heads_with_attacker_only_stage1_poison"
STAGE1_ARTIFACT_TYPE = "stage1_anchor_artifact"
ANCHOR_STATE_VERSION = 5
ANCHOR_ARTIFACT_VERSION = 1
ANCHOR_ARTIFACT_TYPE = "anchor_state_artifact"


class AnchorDefense(nn.Module):
    def __init__(
        self,
        heads,
        model_list,
        device,
        args,
        logger,
        anchor_bank=None,
        detector_state=None,
        final_epoch_client_anchor_losses=None,
        stage1_metrics=None,
        client_reliability=None,
    ):
        super(AnchorDefense, self).__init__()
        self.device = device
        self.args = args
        self.logger = logger
        self.heads = nn.ModuleList(heads)
        self.projectors = nn.ModuleDict({str(client_id): head.projector for client_id, head in enumerate(self.heads)})
        self.anchor_bank = self._build_anchor_bank(heads, device, anchor_bank)
        self.final_epoch_client_anchor_losses = self._normalize_final_epoch_client_anchor_losses(
            final_epoch_client_anchor_losses
        )
        self.stage1_metrics = self._normalize_stage1_metrics(stage1_metrics)
        self.client_reliability = self._normalize_client_reliability(client_reliability)
        if self.client_reliability is None:
            self.client_reliability = self._build_client_reliability(
                self.stage1_metrics,
                client_num=len(model_list) - 1,
            )
        self.detector = MultiClientDetector(
            client_num=len(model_list) - 1,
            gamma=getattr(args, "gamma", 2.0),
            theta_supp=getattr(args, "theta_supp", 0.15),
            required_support_count=getattr(args, "stage3_required_support_count", 2),
            enable_joint_weighted_voting=getattr(args, "stage3_enable_joint_weighted_voting", True),
            enable_static_reliability=getattr(args, "stage3_enable_static_reliability", True),
            enable_conservative_correction=getattr(args, "enable_conservative_correction", False),
            tau_corr=getattr(args, "tau_corr", 0.0),
            client_reliability=self.client_reliability,
        )
        self.detector.load_state_dict(detector_state)
        self.detector.gamma = float(getattr(args, "gamma", 2.0))
        self.detector.theta_supp = float(getattr(args, "theta_supp", 0.15))
        self.detector.required_support_count = self.detector._normalize_required_support_count(
            getattr(args, "stage3_required_support_count", 2)
        )
        self.detector.enable_joint_weighted_voting = bool(
            getattr(args, "stage3_enable_joint_weighted_voting", True)
        )
        self.detector.enable_static_reliability = bool(
            getattr(args, "stage3_enable_static_reliability", True)
        )
        self.detector.enable_conservative_correction = bool(
            getattr(args, "enable_conservative_correction", False)
        )
        self.detector.tau_corr = float(getattr(args, "tau_corr", 0.0))
        self.detector.set_client_reliability(self.client_reliability)

        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @staticmethod
    def _build_anchor_bank(heads, device, anchor_bank=None):
        if anchor_bank is not None:
            return {
                int(client_id): F.normalize(anchor.detach().clone().to(device), dim=1)
                for client_id, anchor in anchor_bank.items()
            }
        return {
            client_id: F.normalize(head.classifier.weight.detach().clone(), dim=1).to(device)
            for client_id, head in enumerate(heads)
        }

    @staticmethod
    def _build_random_anchor_bank(heads, device):
        return {
            client_id: F.normalize(torch.randn_like(head.classifier.weight.detach()), dim=1).to(device)
            for client_id, head in enumerate(heads)
        }

    @staticmethod
    def _normalize_final_epoch_client_anchor_losses(final_epoch_client_anchor_losses):
        if final_epoch_client_anchor_losses is None:
            return None
        return {
            int(client_id): float(client_loss.item() if torch.is_tensor(client_loss) else client_loss)
            for client_id, client_loss in final_epoch_client_anchor_losses.items()
        }

    @staticmethod
    def _normalize_stage1_metrics(stage1_metrics):
        if stage1_metrics is None:
            return None
        normalized_metrics = {}
        for client_id, metrics in stage1_metrics.items():
            normalized_metrics[int(client_id)] = {
                metric_name: float(metric_value.item() if torch.is_tensor(metric_value) else metric_value)
                for metric_name, metric_value in metrics.items()
            }
        return normalized_metrics

    @staticmethod
    def _normalize_client_reliability(client_reliability):
        if client_reliability is None:
            return None

        if isinstance(client_reliability, dict):
            normalized = {
                int(client_id): max(0.0, float(weight.item() if torch.is_tensor(weight) else weight))
                for client_id, weight in client_reliability.items()
            }
        else:
            normalized = {
                client_id: max(0.0, float(weight))
                for client_id, weight in enumerate(client_reliability)
            }

        total = sum(normalized.values())
        if total <= 0.0:
            return None
        return {
            client_id: weight / total
            for client_id, weight in sorted(normalized.items())
        }

    @staticmethod
    def _build_client_reliability(stage1_metrics, client_num):
        if client_num <= 0:
            return {}

        if not stage1_metrics:
            uniform_weight = 1.0 / client_num
            return {client_id: uniform_weight for client_id in range(client_num)}

        clean_acc_by_client = {}
        for client_id in range(client_num):
            client_metrics = stage1_metrics.get(client_id, {})
            clean_acc_by_client[client_id] = max(0.0, float(client_metrics.get("clean_acc", 0.0)))

        total_clean_acc = sum(clean_acc_by_client.values())
        if total_clean_acc <= 0.0:
            uniform_weight = 1.0 / client_num
            return {client_id: uniform_weight for client_id in range(client_num)}

        return {
            client_id: clean_acc_by_client[client_id] / total_clean_acc
            for client_id in range(client_num)
        }

    def _export_anchor_bank(self):
        return {
            client_id: anchor.detach().cpu()
            for client_id, anchor in self.anchor_bank.items()
        }

    def _build_empty_anchor_statistics(self):
        return {
            "embedding_sums": {
                client_id: torch.zeros_like(anchor)
                for client_id, anchor in self.anchor_bank.items()
            },
            "class_counts": {
                client_id: torch.zeros(anchor.size(0), device=self.device)
                for client_id, anchor in self.anchor_bank.items()
            },
        }

    def has_stage3_stats(self):
        return True

    def _load_stage1_metrics_from_dir_if_available(self, stage1_dir):
        if self.stage1_metrics is not None and self.client_reliability is not None:
            return

        metrics_path = os.path.join(stage1_dir, "stage1_metrics.json")
        if not os.path.isfile(metrics_path):
            return

        with open(metrics_path, "r", encoding="utf-8") as metric_file:
            raw_stage1_metrics = json.load(metric_file)

        self.set_stage1_metrics(raw_stage1_metrics)
        self.logger.info(
            "=> Loaded Stage 1 client metrics from '%s'; static reliability weights: %s",
            metrics_path,
            {
                client_id: round(weight, 6)
                for client_id, weight in sorted(self.client_reliability.items())
            },
        )

    @staticmethod
    def _normalize_anchor_state_payload(payload):
        if payload is None:
            return None
        if "anchor_state" in payload:
            return payload["anchor_state"]
        return payload

    @staticmethod
    def _extract_head_state_list(anchor_state):
        if "heads" in anchor_state:
            return anchor_state["heads"]
        return anchor_state["anchor_heads"]

    @staticmethod
    def _load_local_models_from_payload(payload, model_list):
        if "local_models" not in payload:
            return
        for local_model, state_dict in zip(model_list[1:], payload["local_models"]):
            local_model.load_state_dict(state_dict)

    def _build_anchor_artifact_payload(self):
        return {
            "artifact_type": ANCHOR_ARTIFACT_TYPE,
            "artifact_version": ANCHOR_ARTIFACT_VERSION,
            "anchor_state": self.to_checkpoint_state(),
        }

    def _build_stage1_artifact_payload(self, model_list):
        return {
            "artifact_type": STAGE1_ARTIFACT_TYPE,
            "artifact_version": STAGE1_ARTIFACT_VERSION,
            "stage1_artifact_version": STAGE1_ARTIFACT_VERSION,
            "stage1_artifact_strategy": STAGE1_ARTIFACT_STRATEGY,
            "local_models": [local_model.state_dict() for local_model in model_list[1:]],
            "anchor_state": self.to_checkpoint_state(),
        }

    @classmethod
    def from_heads(
        cls,
        heads,
        model_list,
        device,
        args,
        logger,
        anchor_bank=None,
        detector_state=None,
        final_epoch_client_anchor_losses=None,
        stage1_metrics=None,
        client_reliability=None,
    ):
        return cls(
            heads=heads,
            model_list=model_list,
            device=device,
            args=args,
            logger=logger,
            anchor_bank=anchor_bank,
            detector_state=detector_state,
            final_epoch_client_anchor_losses=final_epoch_client_anchor_losses,
            stage1_metrics=stage1_metrics,
            client_reliability=client_reliability,
        )

    @classmethod
    def create_with_random_anchors(cls, model_list, device, args, logger):
        heads = build_anchor_heads(model_list, args, device)
        anchor_bank = cls._build_random_anchor_bank(heads, device)
        logger.info(
            "=> Created Stage 2/3 anchor runtime from random initialization: one normalized anchor vector per client and class"
        )
        return cls.from_heads(
            heads=heads,
            model_list=model_list,
            device=device,
            args=args,
            logger=logger,
            anchor_bank=anchor_bank,
        )

    @classmethod
    def load_from_artifact(cls, artifact_path, model_list, device, args, logger):
        payload = load_torch_artifact(
            artifact_path,
            map_location=device,
            logger=logger,
            description="anchor artifact",
        )
        cls._load_local_models_from_payload(payload, model_list)
        anchor_state = cls._normalize_anchor_state_payload(payload)
        heads = build_anchor_heads(model_list, args, device)
        head_state_list = cls._extract_head_state_list(anchor_state)
        for head, state_dict in zip(heads, head_state_list):
            head.load_state_dict(state_dict)
        return cls.from_heads(
            heads=heads,
            model_list=model_list,
            device=device,
            args=args,
            logger=logger,
            anchor_bank=anchor_state.get("anchor_bank"),
            detector_state=anchor_state.get("detector_state"),
            final_epoch_client_anchor_losses=anchor_state.get("final_epoch_client_anchor_losses"),
            stage1_metrics=anchor_state.get("stage1_metrics"),
            client_reliability=anchor_state.get("client_reliability"),
        )

    @classmethod
    def load_from_checkpoint_state(cls, checkpoint_state, model_list, device, args, logger):
        anchor_state = cls._normalize_anchor_state_payload(checkpoint_state)
        heads = build_anchor_heads(model_list, args, device)
        for head, state_dict in zip(heads, cls._extract_head_state_list(anchor_state)):
            head.load_state_dict(state_dict)
        return cls.from_heads(
            heads=heads,
            model_list=model_list,
            device=device,
            args=args,
            logger=logger,
            anchor_bank=anchor_state.get("anchor_bank"),
            detector_state=anchor_state.get("detector_state"),
            final_epoch_client_anchor_losses=anchor_state.get("final_epoch_client_anchor_losses"),
            stage1_metrics=anchor_state.get("stage1_metrics"),
            client_reliability=anchor_state.get("client_reliability"),
        )

    def to_checkpoint_state(self):
        return {
            "state_type": "frozen_anchor_state",
            "anchor_state_version": ANCHOR_STATE_VERSION,
            "heads": [head.state_dict() for head in self.heads],
            "anchor_bank": self._export_anchor_bank(),
            "detector_state": self.detector.state_dict(),
            "anchor_ema_momentum": float(getattr(self.args, "anchor_ema_momentum", 0.995)),
            "anchor_ema_update_freq": int(getattr(self.args, "anchor_ema_update_freq", 1)),
            "final_epoch_client_anchor_losses": self.final_epoch_client_anchor_losses,
            "stage1_metrics": self.stage1_metrics,
            "client_reliability": self.client_reliability,
        }

    def set_final_epoch_client_anchor_losses(self, final_epoch_client_anchor_losses):
        self.final_epoch_client_anchor_losses = self._normalize_final_epoch_client_anchor_losses(
            final_epoch_client_anchor_losses
        )

    def set_stage1_metrics(self, stage1_metrics):
        self.stage1_metrics = self._normalize_stage1_metrics(stage1_metrics)
        self.client_reliability = self._build_client_reliability(
            self.stage1_metrics,
            client_num=len(self.heads),
        )
        self.detector.set_client_reliability(self.client_reliability)

    def save(self, artifact_path):
        save_anchor_artifact(artifact_path, self._build_anchor_artifact_payload())

    def save_stage1_artifacts(self, model_list, stage1_dir):
        ensure_dir(stage1_dir)
        # Stage 1 artifacts contain passive local models plus the active-party private heads.
        payload = self._build_stage1_artifact_payload(model_list)
        save_anchor_artifact(f"{stage1_dir}/stage1_artifact.pt", payload)
        for client_id, local_model in enumerate(model_list[1:]):
            torch.save(local_model.state_dict(), f"{stage1_dir}/client_{client_id}_local_model.pt")
            torch.save(self.heads[client_id].state_dict(), f"{stage1_dir}/client_{client_id}_anchor_head.pt")
            torch.save(self.anchor_bank[client_id].detach().cpu(), f"{stage1_dir}/client_{client_id}_anchors.pt")

    @classmethod
    def load_stage1_artifacts(cls, model_list, device, args, logger):
        stage1_dir = get_stage1_dir(args)
        artifact_path = f"{stage1_dir}/stage1_artifact.pt"
        if not os.path.isfile(artifact_path):
            return None
        payload = load_torch_artifact(
            artifact_path,
            map_location=device,
            logger=logger,
            description="stage1 anchor artifact",
        )
        artifact_version = payload.get("stage1_artifact_version", payload.get("artifact_version"))
        if artifact_version not in {2, STAGE1_ARTIFACT_VERSION}:
            logger.info("=> Existing stage1 artifact is outdated, stage 1 will be re-run")
            return None
        logger.info("=> Loading stage1 local models and anchors from '%s'", artifact_path)
        instance = cls.load_from_artifact(artifact_path, model_list, device, args, logger)
        instance._load_stage1_metrics_from_dir_if_available(stage1_dir)
        return instance

    def compute_anchor_loss(self, local_output_list, labels, return_client_losses=False):
        output = compute_anchor_constraint_loss(
            local_output_list,
            labels,
            self.anchor_bank,
            self.projectors,
            return_client_losses=return_client_losses,
        )
        if return_client_losses:
            loss, _, client_loss_dict = output
            return loss, client_loss_dict
        loss, _ = output
        return loss

    def create_epoch_anchor_statistics(self):
        return self._build_empty_anchor_statistics()

    def accumulate_epoch_anchor_statistics(self, local_output_list, labels, anchor_statistics):
        if anchor_statistics is None:
            return

        with torch.no_grad():
            detached_outputs = [local_output.detach() for local_output in local_output_list]
            embedding_dict = self._compute_embeddings(detached_outputs)
            one_hot_labels = F.one_hot(labels, num_classes=self.anchor_bank[0].size(0)).to(self.device).float()

            for client_id, embeddings in embedding_dict.items():
                anchor_statistics["embedding_sums"][client_id] += one_hot_labels.transpose(0, 1).matmul(embeddings)
                anchor_statistics["class_counts"][client_id] += one_hot_labels.sum(dim=0)

    def apply_epoch_anchor_ema(self, anchor_statistics):
        if anchor_statistics is None:
            return {}

        momentum = float(getattr(self.args, "anchor_ema_momentum", 0.995))
        disable_ema = momentum >= 1.0
        updated_class_summary = {}
        with torch.no_grad():
            for client_id, anchor in self.anchor_bank.items():
                class_counts = anchor_statistics["class_counts"][client_id]
                updated_mask = class_counts > 0
                updated_class_summary[client_id] = int(updated_mask.sum().item())
                if not updated_mask.any() or disable_ema:
                    continue

                class_means = anchor_statistics["embedding_sums"][client_id][updated_mask] / class_counts[
                    updated_mask
                ].unsqueeze(1)
                blended_anchor = momentum * anchor[updated_mask] + (1.0 - momentum) * class_means
                anchor[updated_mask] = F.normalize(blended_anchor, dim=1)

        return updated_class_summary

    def _compute_embeddings(self, local_output_list):
        embedding_dict = {}
        for client_id, local_output in enumerate(local_output_list):
            embeddings = F.normalize(self.projectors[str(client_id)](local_output), dim=1)
            embedding_dict[client_id] = embeddings
        return embedding_dict

    def _compute_distances_for_labels(self, embedding_dict, labels):
        distance_dict = {}
        for client_id, embeddings in embedding_dict.items():
            anchors = self.anchor_bank[client_id].index_select(0, labels)
            distance_dict[client_id] = ((embeddings - anchors) ** 2).sum(dim=1)
        return distance_dict

    def _compute_all_class_distances(self, embedding_dict):
        distance_dict = {}
        for client_id, embeddings in embedding_dict.items():
            anchors = self.anchor_bank[client_id]
            distance_dict[client_id] = ((embeddings.unsqueeze(1) - anchors.unsqueeze(0)) ** 2).sum(dim=2)
        return distance_dict

    def _predict_labels_from_anchors(self, all_class_distance_dict):
        return {
            client_id: client_distances.argmin(dim=1)
            for client_id, client_distances in all_class_distance_dict.items()
        }

    def run_stage3_detection(self, local_output_list, global_output):
        predicted_labels = global_output.argmax(dim=1)
        embedding_dict = self._compute_embeddings(local_output_list)
        all_class_distance_dict = self._compute_all_class_distances(embedding_dict)
        client_prediction_dict = self._predict_labels_from_anchors(all_class_distance_dict)
        detector_output = self.detector.detect_from_client_predictions(
            client_prediction_dict,
            predicted_labels,
            all_class_distance_dict=all_class_distance_dict,
        )
        suspicious_mask = detector_output["suspicious_mask"]
        majority_labels = detector_output["majority_labels"]
        majority_vote_counts = detector_output["majority_vote_counts"]
        corrected_predictions = predicted_labels.clone()
        correction_applied_mask = detector_output["correction_applied_mask"]
        correction_labels = detector_output["correction_labels"]
        corrected_predictions[correction_applied_mask] = correction_labels[correction_applied_mask]
        return {
            "predicted_labels": predicted_labels,
            "final_predictions": corrected_predictions,
            "suspicious_mask": suspicious_mask,
            "majority_labels": majority_labels,
            "majority_vote_counts": majority_vote_counts,
            "global_support_counts": detector_output["global_support_counts"],
            "valid_support_counts": detector_output["valid_support_counts"],
            "strict_majority_votes": detector_output["strict_majority_votes"],
            "client_prediction_dict": client_prediction_dict,
            "client_agreement_mask_dict": detector_output["client_agreement_mask_dict"],
            "client_valid_support_mask_dict": detector_output["client_valid_support_mask_dict"],
            "remaining_client_mask_dict": detector_output["remaining_client_mask_dict"],
            "remaining_client_counts": detector_output["remaining_client_counts"],
            "dynamic_vote_scores": detector_output["dynamic_vote_scores"],
            "dynamic_weight_matrix": detector_output["dynamic_weight_matrix"],
            "top_vote_scores": detector_output["top_vote_scores"],
            "second_vote_scores": detector_output["second_vote_scores"],
            "correction_margin": detector_output["correction_margin"],
            "correction_margin_ok_mask": detector_output["correction_margin_ok_mask"],
            "correction_margin_rejected_mask": detector_output["correction_margin_rejected_mask"],
            "client_confidence_matrix": detector_output["client_confidence_matrix"],
            "client_reliability_vector": detector_output["client_reliability_vector"],
            "correction_labels": correction_labels,
            "correction_tie_mask": detector_output["correction_tie_mask"],
            "weighted_labels": detector_output["weighted_labels"],
            "weighted_tie_mask": detector_output["weighted_tie_mask"],
            "correction_applied_mask": correction_applied_mask,
            "all_class_distance_dict": all_class_distance_dict,
        }

    def _compute_batch_distances(self, local_output_list, predicted_labels):
        _, embedding_dict = compute_anchor_constraint_loss(
            local_output_list,
            predicted_labels,
            self.anchor_bank,
            self.projectors,
        )
        distance_dict = {}
        for client_id, embeddings in embedding_dict.items():
            anchors = self.anchor_bank[client_id].index_select(0, predicted_labels)
            distance_dict[client_id] = ((embeddings - anchors) ** 2).sum(dim=1)
        return distance_dict

    def _forward_for_detection(self, model_list, x):
        x_split_list = split_vfl(x, self.args)
        local_output_list = [model_list[i + 1](x_split_list[i]) for i in range(self.args.client_num)]
        global_output = model_list[0](local_output_list)
        predicted = global_output.argmax(dim=1)
        return local_output_list, global_output, predicted

    def calibrate(self, model_list, clean_loader):
        del model_list, clean_loader
        self.logger.info(
            "=> Stage 3 detector does not require clean-data calibration; it uses effective-support filtering plus static-reliability and dynamic-confidence voting on clients that disagree with the global prediction"
        )

    def evaluate_detection(self, model_list, data_loader, exclude_target_label=None):
        total = 0
        detected = 0
        with torch.no_grad():
            for x, _, y, _ in data_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, global_output, predicted = self._forward_for_detection(model_list, x)
                stage3_output = self.run_stage3_detection(local_output_list, global_output)
                flags = stage3_output["suspicious_mask"]

                if exclude_target_label is not None:
                    valid_mask = y != exclude_target_label
                    if valid_mask.any():
                        total += valid_mask.float().sum().item()
                        detected += flags[valid_mask].float().sum().item()
                else:
                    total += y.size(0)
                    detected += flags.float().sum().item()
        return detected / max(1, total)
