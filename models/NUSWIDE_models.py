import torch
import torch.nn as nn

from dataset.utils import (
    NUSWIDE_TOTAL_DIM,
    get_nuswide_client_layout,
    validate_nuswide_total_dim,
)


class GlobalModelForNUSWIDE(nn.Module):
    def __init__(self, args):
        super(GlobalModelForNUSWIDE, self).__init__()
        total_feature_dim = sum(
            client_layout["output_dim"] for client_layout in get_nuswide_client_layout(args.client_num)
        )
        self.linear1 = nn.Linear(total_feature_dim, 100)
        self.linear2 = nn.Linear(100, 50)
        self.classifier = nn.Linear(50, 5)
        self.args = args
        self.num_classes = 5

    def forward(self, input_list):
        tensor_t = torch.cat(input_list, dim=1)
        x = self.linear1(tensor_t)
        x = self.linear2(x)
        x = self.classifier(x)
        return x


class LocalModelForNUSWIDE(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForNUSWIDE, self).__init__()
        self.args = args
        total_dim = getattr(args, "nuswide_total_dim", NUSWIDE_TOTAL_DIM)
        validate_nuswide_total_dim(total_dim)
        client_layout = get_nuswide_client_layout(args.client_num)[client_number]
        input_dim = client_layout["input_dim"]
        self.output_dim = client_layout["output_dim"]

        if client_number == 0:
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, self.output_dim),
                nn.ReLU(),
            )
        else:
            hidden_dim = max(64, min(256, max(input_dim // 2, self.output_dim * 4)))
            middle_dim = max(32, max(hidden_dim // 2, self.output_dim * 2))
            self.backbone = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, middle_dim),
                nn.ReLU(),
                nn.Linear(middle_dim, self.output_dim),
                nn.ReLU(),
            )

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.extract_features(x)
