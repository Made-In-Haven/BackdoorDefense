import torch
import torch.nn as nn

from dataset.utils import get_client_input_dim


class GlobalModelForPHISHING(nn.Module):
    def __init__(self, args):
        super(GlobalModelForPHISHING, self).__init__()
        local_feature_dim = 4
        self.linear1 = nn.Linear(local_feature_dim * args.client_num, 4)
        self.classifier = nn.Linear(4, 2)
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
        input_dim = get_client_input_dim(args, total_dim=30, client_id=client_number)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 4)
        )
        self.output_dim = 4

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.extract_features(x)
