import torch


class MultiClientDetector:
    def __init__(
        self,
        client_num,
        gamma=2.0,
        theta_supp=0.15,
        enable_joint_weighted_voting=True,
        enable_static_reliability=True,
        enable_conservative_correction=False,
        tau_corr=0.0,
        client_reliability=None,
    ):
        self.client_num = client_num
        self.gamma = float(gamma)
        self.theta_supp = float(theta_supp)
        self.enable_joint_weighted_voting = bool(enable_joint_weighted_voting)
        self.enable_static_reliability = bool(enable_static_reliability)
        self.enable_conservative_correction = bool(enable_conservative_correction)
        self.tau_corr = float(tau_corr)
        self.epsilon = 1e-8
        self.reliability_log_epsilon = 1e-12
        self.client_reliability = None
        self.set_client_reliability(client_reliability)

    def state_dict(self):
        return {
            "client_num": self.client_num,
            "gamma": self.gamma,
            "theta_supp": self.theta_supp,
            "enable_joint_weighted_voting": self.enable_joint_weighted_voting,
            "enable_static_reliability": self.enable_static_reliability,
            "enable_conservative_correction": self.enable_conservative_correction,
            "tau_corr": self.tau_corr,
            "epsilon": self.epsilon,
            "reliability_log_epsilon": self.reliability_log_epsilon,
            "client_reliability": self.client_reliability.cpu().tolist() if self.client_reliability is not None else None,
        }

    def load_state_dict(self, state_dict):
        if not state_dict:
            return
        self.gamma = float(state_dict.get("gamma", state_dict.get("vote_temperature", self.gamma)))
        self.theta_supp = float(state_dict.get("theta_supp", state_dict.get("support_threshold", self.theta_supp)))
        self.enable_joint_weighted_voting = bool(
            state_dict.get("enable_joint_weighted_voting", self.enable_joint_weighted_voting)
        )
        self.enable_static_reliability = bool(
            state_dict.get("enable_static_reliability", self.enable_static_reliability)
        )
        self.enable_conservative_correction = bool(
            state_dict.get("enable_conservative_correction", self.enable_conservative_correction)
        )
        self.tau_corr = float(state_dict.get("tau_corr", self.tau_corr))
        self.epsilon = state_dict.get("epsilon", self.epsilon)
        self.reliability_log_epsilon = state_dict.get("reliability_log_epsilon", self.reliability_log_epsilon)
        self.set_client_reliability(
            state_dict.get("client_reliability", state_dict.get("static_client_weights"))
        )

    def set_client_reliability(self, client_reliability):
        if client_reliability is None:
            self.client_reliability = None
            return

        if isinstance(client_reliability, dict):
            normalized_reliability = {
                int(client_id): float(weight)
                for client_id, weight in client_reliability.items()
            }
            sorted_client_ids = sorted(normalized_reliability)
            reliability_tensor = torch.tensor(
                [normalized_reliability[client_id] for client_id in sorted_client_ids],
                dtype=torch.float32,
            )
        else:
            reliability_tensor = torch.tensor(client_reliability, dtype=torch.float32)

        reliability_tensor = reliability_tensor.clamp_min(0.0)
        total = reliability_tensor.sum()
        if reliability_tensor.numel() == 0 or total.item() <= 0.0:
            self.client_reliability = None
            return

        self.client_reliability = reliability_tensor / total

    @staticmethod
    def _get_uniform_client_vector(sorted_client_ids, device):
        if not sorted_client_ids:
            return torch.zeros(0, device=device)
        return torch.full(
            (len(sorted_client_ids),),
            1.0 / max(1, len(sorted_client_ids)),
            device=device,
        )

    def _get_client_reliability_vector(self, sorted_client_ids, device):
        if not sorted_client_ids:
            return torch.zeros(0, device=device)
        if self.client_reliability is None or self.client_reliability.numel() <= max(sorted_client_ids):
            return torch.full(
                (len(sorted_client_ids),),
                1.0 / max(1, len(sorted_client_ids)),
                device=device,
            )

        client_index_tensor = torch.tensor(sorted_client_ids, dtype=torch.long, device=self.client_reliability.device)
        reliability_vector = self.client_reliability.index_select(0, client_index_tensor).to(device)
        total = reliability_vector.sum()
        if total.item() <= 0.0:
            return torch.full(
                (len(sorted_client_ids),),
                1.0 / max(1, len(sorted_client_ids)),
                device=device,
            )
        return reliability_vector / total

    @staticmethod
    def _select_majority_labels(vote_counts, aggregate_distances=None):
        max_votes = vote_counts.max(dim=1).values
        candidate_mask = vote_counts.eq(max_votes.unsqueeze(1))
        if aggregate_distances is not None:
            inf = torch.full_like(aggregate_distances, float("inf"))
            candidate_distances = torch.where(candidate_mask, aggregate_distances, inf)
            majority_labels = candidate_distances.argmin(dim=1)
        else:
            majority_labels = vote_counts.argmax(dim=1)
        return majority_labels, max_votes

    def _compute_client_confidences(self, sorted_client_ids, all_class_distance_dict, num_samples, device):
        confidence_matrix = torch.zeros((num_samples, len(sorted_client_ids)), device=device)
        if all_class_distance_dict is None:
            return confidence_matrix

        for column_id, client_id in enumerate(sorted_client_ids):
            distances = all_class_distance_dict[client_id]
            if distances.size(1) == 1:
                nearest_distances = distances[:, 0]
                second_nearest_distances = distances[:, 0]
            else:
                top2_distances = distances.topk(k=2, dim=1, largest=False).values
                nearest_distances = top2_distances[:, 0]
                second_nearest_distances = top2_distances[:, 1]
            confidence_matrix[:, column_id] = (
                second_nearest_distances - nearest_distances
            ) / (nearest_distances + self.epsilon)
        return confidence_matrix

    def detect_from_client_predictions(self, client_prediction_dict, global_predictions, all_class_distance_dict=None):
        if not client_prediction_dict:
            raise RuntimeError("Client anchor predictions are unavailable for stage 3 detection.")

        sorted_client_ids = sorted(int(client_id) for client_id in client_prediction_dict)
        first_client_id = sorted_client_ids[0]
        num_samples = client_prediction_dict[first_client_id].shape[0]
        if all_class_distance_dict is not None:
            num_classes = all_class_distance_dict[first_client_id].shape[1]
            aggregate_distances = torch.zeros_like(all_class_distance_dict[first_client_id])
        else:
            num_classes = int(global_predictions.max().item()) + 1
            aggregate_distances = None

        stacked_predictions = []
        vote_counts = torch.zeros((num_samples, num_classes), device=global_predictions.device, dtype=torch.long)
        client_agreement_mask_dict = {}
        remaining_client_mask_dict = {}

        for client_id in sorted_client_ids:
            client_predictions = client_prediction_dict[client_id]
            stacked_predictions.append(client_predictions)
            vote_counts.scatter_add_(
                1,
                client_predictions.view(-1, 1),
                torch.ones((num_samples, 1), device=global_predictions.device, dtype=torch.long),
            )
            client_agreement_mask_dict[client_id] = client_predictions.eq(global_predictions)
            remaining_client_mask_dict[client_id] = client_predictions.ne(global_predictions)
            if aggregate_distances is not None:
                aggregate_distances += all_class_distance_dict[client_id]

        stacked_predictions = torch.stack(stacked_predictions, dim=1)
        majority_labels, max_votes = self._select_majority_labels(vote_counts, aggregate_distances)
        strict_majority_votes = (self.client_num // 2) + 1
        global_support_counts = stacked_predictions.eq(global_predictions.unsqueeze(1)).sum(dim=1)

        remaining_mask = stacked_predictions.ne(global_predictions.unsqueeze(1))
        remaining_counts = remaining_mask.sum(dim=1)

        confidence_matrix = self._compute_client_confidences(
            sorted_client_ids,
            all_class_distance_dict,
            num_samples,
            global_predictions.device,
        )
        confidence_matrix = torch.nan_to_num(confidence_matrix, nan=0.0, posinf=1e4, neginf=0.0)
        valid_support_mask = stacked_predictions.eq(global_predictions.unsqueeze(1)) & confidence_matrix.ge(
            self.theta_supp
        )
        valid_support_counts = valid_support_mask.sum(dim=1)
        suspicious_mask = valid_support_counts.lt(strict_majority_votes)

        if self.enable_static_reliability:
            static_reliability_vector = self._get_client_reliability_vector(
                sorted_client_ids,
                global_predictions.device,
            )
        else:
            static_reliability_vector = self._get_uniform_client_vector(sorted_client_ids, global_predictions.device)

        dynamic_weight_matrix = torch.zeros_like(confidence_matrix)
        valid_weight_rows = remaining_mask.any(dim=1)
        if self.enable_joint_weighted_voting:
            if valid_weight_rows.any():
                valid_remaining_mask = remaining_mask[valid_weight_rows]
                joint_scores = self.gamma * confidence_matrix[valid_weight_rows]
                if self.enable_static_reliability:
                    log_reliability = torch.log(
                        static_reliability_vector.clamp_min(self.reliability_log_epsilon)
                    ).unsqueeze(0)
                    joint_scores = log_reliability + joint_scores
                joint_scores = torch.nan_to_num(joint_scores, nan=-1e4, posinf=1e4, neginf=-1e4)

                masked_joint_scores = torch.where(
                    valid_remaining_mask,
                    joint_scores,
                    torch.full_like(joint_scores, -1e4),
                )
                shifted_scores = masked_joint_scores - masked_joint_scores.max(dim=1, keepdim=True).values
                exp_scores = torch.exp(shifted_scores) * valid_remaining_mask.float()
                weight_denominator = exp_scores.sum(dim=1, keepdim=True).clamp_min(self.epsilon)
                dynamic_weight_matrix[valid_weight_rows] = exp_scores / weight_denominator

            dynamic_vote_scores = torch.zeros(
                (num_samples, num_classes),
                device=global_predictions.device,
                dtype=dynamic_weight_matrix.dtype,
            )
            dynamic_vote_scores.scatter_add_(1, stacked_predictions, dynamic_weight_matrix)
            correction_labels = dynamic_vote_scores.argmax(dim=1)
            top2_vote_scores = dynamic_vote_scores.topk(k=min(2, num_classes), dim=1, largest=True, sorted=True).values
            top_vote_scores = top2_vote_scores[:, 0]
            second_vote_scores = (
                top2_vote_scores[:, 1]
                if top2_vote_scores.size(1) > 1
                else torch.zeros_like(top_vote_scores)
            )
            correction_margin = top_vote_scores - second_vote_scores
            correction_tie_mask = valid_weight_rows & dynamic_vote_scores.eq(top_vote_scores.unsqueeze(1)).sum(dim=1).gt(1)
            correction_margin_ok_mask = correction_margin.ge(self.tau_corr)
            if self.enable_conservative_correction:
                correction_applied_mask = (
                    suspicious_mask & valid_weight_rows & ~correction_tie_mask & correction_margin_ok_mask
                )
            else:
                correction_applied_mask = suspicious_mask & valid_weight_rows & ~correction_tie_mask
            correction_margin_rejected_mask = (
                suspicious_mask & valid_weight_rows & ~correction_tie_mask & ~correction_margin_ok_mask
            )
        else:
            dynamic_vote_scores = vote_counts.to(dtype=confidence_matrix.dtype)
            correction_labels = majority_labels
            top2_vote_scores = dynamic_vote_scores.topk(k=min(2, num_classes), dim=1, largest=True, sorted=True).values
            top_vote_scores = top2_vote_scores[:, 0]
            second_vote_scores = (
                top2_vote_scores[:, 1]
                if top2_vote_scores.size(1) > 1
                else torch.zeros_like(top_vote_scores)
            )
            correction_margin = top_vote_scores - second_vote_scores
            correction_tie_mask = torch.zeros_like(suspicious_mask, dtype=torch.bool)
            correction_margin_ok_mask = torch.ones_like(suspicious_mask, dtype=torch.bool)
            correction_applied_mask = suspicious_mask
            correction_margin_rejected_mask = torch.zeros_like(suspicious_mask, dtype=torch.bool)

        return {
            "suspicious_mask": suspicious_mask,
            "majority_labels": majority_labels,
            "majority_vote_counts": max_votes,
            "global_support_counts": global_support_counts,
            "valid_support_counts": valid_support_counts,
            "strict_majority_votes": strict_majority_votes,
            "client_agreement_mask_dict": client_agreement_mask_dict,
            "client_valid_support_mask_dict": {
                client_id: valid_support_mask[:, column_id]
                for column_id, client_id in enumerate(sorted_client_ids)
            },
            "remaining_client_mask_dict": remaining_client_mask_dict,
            "remaining_client_counts": remaining_counts,
            "dynamic_vote_scores": dynamic_vote_scores,
            "dynamic_weight_matrix": dynamic_weight_matrix,
            "top_vote_scores": top_vote_scores,
            "second_vote_scores": second_vote_scores,
            "correction_margin": correction_margin,
            "correction_margin_ok_mask": correction_margin_ok_mask,
            "correction_margin_rejected_mask": correction_margin_rejected_mask,
            "client_confidence_matrix": confidence_matrix,
            "client_reliability_vector": static_reliability_vector,
            "correction_labels": correction_labels,
            "correction_tie_mask": correction_tie_mask,
            "weighted_labels": correction_labels,
            "weighted_tie_mask": correction_tie_mask,
            "correction_applied_mask": correction_applied_mask,
        }
