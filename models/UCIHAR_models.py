import torch
import torch.nn as nn

from dataset.utils import get_client_input_dim


class GlobalModelForUCIHAR(nn.Module):
    def __init__(self, args):
        super(GlobalModelForUCIHAR, self).__init__()
        local_feature_dim = 16
        self.linear1 = nn.Linear(local_feature_dim * args.client_num, 32)
        self.linear2 = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, 6)
        self.args = args
        self.num_classes = 6

    def forward(self, input_list):
        tensor_t = torch.cat(input_list, dim=1)

        # forward
        x = tensor_t
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.classifier(x)
        return x


class LocalModelForUCIHAR(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForUCIHAR, self).__init__()
        self.args = args
        input_dim = get_client_input_dim(args, total_dim=561, client_id=client_number)
        # Each client backbone is built from its own feature slice width.
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 140),
            nn.ReLU(),
            nn.Linear(140, 70),
            nn.ReLU(),
            nn.Linear(70, 35),
            nn.ReLU(),
            nn.Linear(35, 16),
            nn.ReLU()
        )
        self.output_dim = 16

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.extract_features(x)
