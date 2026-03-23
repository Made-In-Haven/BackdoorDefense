import torch

from defense.anchor_utils import build_vote_threshold


class MultiClientDetector:
    def __init__(self, client_num, detect_threshold, majority_ratio):
        self.client_num = client_num
        self.detect_threshold = detect_threshold
        self.majority_ratio = majority_ratio
        self.thresholds = None

    def state_dict(self):
        if self.thresholds is None:
            thresholds = None
        else:
            thresholds = {
                client_id: threshold.detach().cpu()
                for client_id, threshold in self.thresholds.items()
            }
        return {
            "client_num": self.client_num,
            "detect_threshold": self.detect_threshold,
            "majority_ratio": self.majority_ratio,
            "thresholds": thresholds,
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            self.thresholds = None
            return
        thresholds = state_dict.get("thresholds")
        if thresholds is None:
            self.thresholds = None
            return
        self.thresholds = {
            int(client_id): threshold.detach().clone()
            for client_id, threshold in thresholds.items()
        }

    def fit(self, distance_dict):
        thresholds = {}
        for client_id, distances in distance_dict.items():
            if not distances:
                thresholds[client_id] = torch.tensor(0.0)
                continue
            concat_distances = torch.cat(distances)
            mean = concat_distances.mean()
            std = concat_distances.std(unbiased=False)
            thresholds[client_id] = mean + self.detect_threshold * std
        self.thresholds = thresholds
        return thresholds

    def predict(self, distance_dict):
        if self.thresholds is None:
            raise RuntimeError("Detector thresholds have not been fitted.")

        abnormal_votes = None
        for client_id, distances in distance_dict.items():
            votes = (distances > self.thresholds[client_id].to(distances.device)).long()
            abnormal_votes = votes if abnormal_votes is None else abnormal_votes + votes

        required_votes = build_vote_threshold(self.client_num, self.majority_ratio)
        return abnormal_votes >= required_votes
