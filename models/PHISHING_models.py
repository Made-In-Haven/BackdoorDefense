import torch
import torch.nn as nn

from dataset.utils import get_client_input_dim


class GlobalModelForPHISHING(nn.Module):
    def __init__(self, args):
        super(GlobalModelForPHISHING, self).__init__()
        local_feature_dim = 8
        self.linear1 = nn.Linear(local_feature_dim * args.client_num, 16)
        self.classifier = nn.Linear(16, 2)
        self.args = args
        self.num_classes = 2

    def forward(self, input_list):
        tensor_t = torch.cat(input_list, dim=1)

        # forward
        x = tensor_t
        x = self.linear1(x)
        x = self.classifier(x)
        return x


class LocalModelForPHISHING(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForPHISHING, self).__init__()
        self.args = args
        total_dim = getattr(args, "phishing_input_dim", 111)
        input_dim = get_client_input_dim(args, total_dim=total_dim, client_id=client_number)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
        )
        self.output_dim = 8

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.extract_features(x)
