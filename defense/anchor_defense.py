import os

import torch
import torch.nn as nn

from dataset.utils import split_vfl
from defense.anchor_losses import compute_anchor_constraint_loss
from defense.anchor_utils import build_anchor_heads, ensure_dir, get_stage1_dir, save_anchor_artifact
from defense.detector import MultiClientDetector

STAGE1_ARTIFACT_VERSION = 3
STAGE1_ARTIFACT_STRATEGY = "active_party_private_heads_with_attacker_only_stage1_poison"
STAGE1_ARTIFACT_TYPE = "stage1_anchor_artifact"
ANCHOR_STATE_VERSION = 1
ANCHOR_ARTIFACT_VERSION = 1
ANCHOR_ARTIFACT_TYPE = "anchor_state_artifact"


class AnchorDefense(nn.Module):
    def __init__(self, heads, model_list, device, args, logger, anchor_bank=None, detector_state=None):
        super(AnchorDefense, self).__init__()
        self.device = device
        self.args = args
        self.logger = logger
        self.heads = nn.ModuleList(heads)
        self.projectors = nn.ModuleDict({str(client_id): head.projector for client_id, head in enumerate(self.heads)})
        self.anchor_bank = self._build_anchor_bank(heads, device, anchor_bank)
        self.detector = MultiClientDetector(
            client_num=len(model_list) - 1,
            detect_threshold=args.detect_threshold,
            majority_ratio=args.majority_ratio,
        )
        self.detector.load_state_dict(detector_state)

        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()

    @staticmethod
    def _build_anchor_bank(heads, device, anchor_bank=None):
        if anchor_bank is not None:
            return {
                int(client_id): anchor.detach().clone().to(device)
                for client_id, anchor in anchor_bank.items()
            }
        return {
            client_id: torch.nn.functional.normalize(head.classifier.weight.detach().clone(), dim=1).to(device)
            for client_id, head in enumerate(heads)
        }

    def _export_anchor_bank(self):
        return {
            client_id: anchor.detach().cpu()
            for client_id, anchor in self.anchor_bank.items()
        }

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
    def from_heads(cls, heads, model_list, device, args, logger, anchor_bank=None, detector_state=None):
        return cls(
            heads=heads,
            model_list=model_list,
            device=device,
            args=args,
            logger=logger,
            anchor_bank=anchor_bank,
            detector_state=detector_state,
        )

    @classmethod
    def load_from_artifact(cls, artifact_path, model_list, device, args, logger):
        payload = torch.load(artifact_path, map_location=device)
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
        )

    def to_checkpoint_state(self):
        return {
            "state_type": "frozen_anchor_state",
            "anchor_state_version": ANCHOR_STATE_VERSION,
            "heads": [head.state_dict() for head in self.heads],
            "anchor_bank": self._export_anchor_bank(),
            "detector_state": self.detector.state_dict(),
        }

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
        payload = torch.load(artifact_path, map_location=device)
        artifact_version = payload.get("stage1_artifact_version", payload.get("artifact_version"))
        if artifact_version not in {2, STAGE1_ARTIFACT_VERSION}:
            logger.info("=> Existing stage1 artifact is outdated, stage 1 will be re-run")
            return None
        logger.info("=> Loading stage1 local models and anchors from '%s'", artifact_path)
        return cls.load_from_artifact(artifact_path, model_list, device, args, logger)

    def compute_anchor_loss(self, local_output_list, labels):
        loss, _ = compute_anchor_constraint_loss(local_output_list, labels, self.anchor_bank, self.projectors)
        return loss

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
        return local_output_list, predicted

    def calibrate(self, model_list, clean_loader):
        distance_dict = {client_id: [] for client_id in range(self.args.client_num)}
        with torch.no_grad():
            for x, _, _, _ in clean_loader:
                x = x.to(self.device).float()
                local_output_list, predicted = self._forward_for_detection(model_list, x)
                batch_distances = self._compute_batch_distances(local_output_list, predicted)
                for client_id, distances in batch_distances.items():
                    distance_dict[client_id].append(distances.detach().cpu())
        thresholds = self.detector.fit(distance_dict)
        self.logger.info(
            "=> Detector thresholds: %s",
            {client_id: round(value.item(), 6) for client_id, value in thresholds.items()},
        )

    def evaluate_detection(self, model_list, data_loader, exclude_target_label=None):
        total = 0
        detected = 0
        with torch.no_grad():
            for x, _, y, _ in data_loader:
                x = x.to(self.device).float()
                y = y.to(self.device).long()
                local_output_list, predicted = self._forward_for_detection(model_list, x)
                batch_distances = self._compute_batch_distances(local_output_list, predicted)
                flags = self.detector.predict(batch_distances)

                if exclude_target_label is not None:
                    valid_mask = y != exclude_target_label
                    if valid_mask.any():
                        total += valid_mask.float().sum().item()
                        detected += flags[valid_mask].float().sum().item()
                else:
                    total += y.size(0)
                    detected += flags.float().sum().item()
        return detected / max(1, total)
