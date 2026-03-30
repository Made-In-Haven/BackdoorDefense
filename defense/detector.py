import torch

from defense.anchor_utils import build_vote_threshold


class MultiClientDetector:
    def __init__(self, client_num, detect_threshold, majority_ratio, detect_threshold_cap=None):
        self.client_num = client_num
        self.detect_threshold = detect_threshold
        self.majority_ratio = majority_ratio
        self.detect_threshold_cap = detect_threshold_cap
        self.epsilon = 1e-8

    def state_dict(self):
        return {
            "client_num": self.client_num,
            "detect_threshold": self.detect_threshold,
            "majority_ratio": self.majority_ratio,
            "detect_threshold_cap": self.detect_threshold_cap,
            "epsilon": self.epsilon,
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        self.detect_threshold = state_dict.get("detect_threshold", self.detect_threshold)
        self.majority_ratio = state_dict.get("majority_ratio", self.majority_ratio)
        self.detect_threshold_cap = state_dict.get("detect_threshold_cap", self.detect_threshold_cap)
        self.epsilon = state_dict.get("epsilon", self.epsilon)

    def detect_from_client_predictions(self, client_prediction_dict, global_predictions, all_class_distance_dict=None):
        if not client_prediction_dict:
            raise RuntimeError("Client anchor predictions are unavailable for stage 3 detection.")

        first_client_id = next(iter(client_prediction_dict))
        num_samples = client_prediction_dict[first_client_id].shape[0]
        if all_class_distance_dict is not None:
            num_classes = all_class_distance_dict[first_client_id].shape[1]
            aggregate_distances = torch.zeros_like(all_class_distance_dict[first_client_id])
        else:
            num_classes = int(global_predictions.max().item()) + 1
            aggregate_distances = None

        vote_counts = torch.zeros((num_samples, num_classes), device=global_predictions.device, dtype=torch.long)
        client_agreement_mask_dict = {}

        for client_id, client_predictions in client_prediction_dict.items():
            vote_counts.scatter_add_(
                1,
                client_predictions.view(-1, 1),
                torch.ones((num_samples, 1), device=global_predictions.device, dtype=torch.long),
            )
            client_agreement_mask_dict[int(client_id)] = client_predictions.eq(global_predictions)
            if aggregate_distances is not None:
                aggregate_distances += all_class_distance_dict[client_id]

        max_votes = vote_counts.max(dim=1).values
        candidate_mask = vote_counts.eq(max_votes.unsqueeze(1))
        if aggregate_distances is not None:
            inf = torch.full_like(aggregate_distances, float("inf"))
            candidate_distances = torch.where(candidate_mask, aggregate_distances, inf)
            majority_labels = candidate_distances.argmin(dim=1)
        else:
            majority_labels = vote_counts.argmax(dim=1)

        required_votes = build_vote_threshold(self.client_num, self.majority_ratio)
        suspicious_mask = majority_labels.ne(global_predictions) | max_votes.lt(required_votes)
        return suspicious_mask, majority_labels, max_votes, client_agreement_mask_dict
