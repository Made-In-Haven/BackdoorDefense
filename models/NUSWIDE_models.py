import torch
import torch.nn as nn

from dataset.utils import get_client_input_dim


class GlobalModelForNUSWIDE(nn.Module):
    def __init__(self, args):
        super(GlobalModelForNUSWIDE, self).__init__()
        total_feature_dim = 40 + 60 * (args.client_num - 1)
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
        if client_number == 0:
            self.backbone = nn.Sequential(
                nn.Linear(634, 320),
                nn.ReLU(),
                nn.Linear(320, 160),
                nn.ReLU(),
                nn.Linear(160, 80),
                nn.ReLU(),
                nn.Linear(80, 40),
                nn.ReLU(),
            )
            self.output_dim = 40
        else:
            total_dim = getattr(args, "nuswide_total_dim", 1634)
            input_dim = get_client_input_dim(args, total_dim=total_dim, client_id=client_number)
            hidden_dim = max(125, min(500, input_dim))
            middle_dim = max(64, hidden_dim // 2)
            self.output_dim = 60
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
