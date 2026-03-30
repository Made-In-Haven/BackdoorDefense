import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceClassifier(nn.Module):
    def __init__(self, feature_dim, num_classes, scale=16.0, margin=0.2):
        super(ArcFaceClassifier, self).__init__()
        # This two-layer MLP head is owned by the active party in stage 1.
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(feature_dim, num_classes, bias=False)
        self.scale = scale
        self.margin = margin

    def embed(self, features):
        projected = self.projector(features)
        return F.normalize(projected, dim=1)

    def forward(self, features, labels=None):
        embeddings = self.embed(features)
        weights = F.normalize(self.classifier.weight, dim=1)
        cosine = F.linear(embeddings, weights).clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        if labels is None:
            return cosine * self.scale, embeddings

        target_logits = cosine.gather(1, labels.view(-1, 1))
        sine = torch.sqrt(1.0 - target_logits.pow(2))
        margin_logits = target_logits * math.cos(self.margin) - sine * math.sin(self.margin)
        logits = cosine.scatter(1, labels.view(-1, 1), margin_logits)
        return logits * self.scale, embeddings


def compute_anchor_constraint_loss(local_output_list, labels, anchor_bank, projector_dict, return_client_losses=False):
    loss_list = []
    embedding_dict = {}
    client_loss_dict = {}

    for client_id, local_output in enumerate(local_output_list):
        embeddings = F.normalize(projector_dict[str(client_id)](local_output), dim=1)
        anchors = anchor_bank[client_id].index_select(0, labels)
        embedding_dict[client_id] = embeddings
        client_loss = ((embeddings - anchors) ** 2).sum(dim=1).mean()
        client_loss_dict[client_id] = client_loss
        loss_list.append(client_loss)

    total_loss = torch.zeros(1, device=labels.device).squeeze()
    if not loss_list:
        if return_client_losses:
            return total_loss, embedding_dict, client_loss_dict
        return total_loss, embedding_dict

    total_loss = sum(loss_list) / len(loss_list)
    if return_client_losses:
        return total_loss, embedding_dict, client_loss_dict
    return total_loss, embedding_dict


def compute_single_client_anchor_loss(head, features, labels):
    embeddings = head.embed(features)
    anchors = F.normalize(head.classifier.weight, dim=1).index_select(0, labels)
    return ((embeddings - anchors) ** 2).sum(dim=1).mean()
