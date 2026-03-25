import torch
import torch.nn as nn
from torchvision import models


class GlobalModelForCifar10(nn.Module):
    def __init__(self, args):
        super(GlobalModelForCifar10, self).__init__()
        local_feature_dim = 128
        self.linear1 = nn.Linear(local_feature_dim * args.client_num, 256)
        self.linear2 = nn.Linear(256, 128)
        self.classifier = nn.Linear(128, 10)
        self.args = args
        self.num_classes = 10

    def forward(self, input_list):
        tensor_t = torch.cat(input_list, dim=1)

        # forward
        x = tensor_t
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.classifier(x)
        return x


class LocalModelForCifar10(nn.Module):
    def __init__(self, args):
        super(LocalModelForCifar10, self).__init__()
        self.args = args
        self.backbone = models.resnet18(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        # Keep the local encoder output width fixed so the global model only needs to adapt to client count.
        self.backbone.fc = nn.Linear(num_ftrs, 128)
        self.output_dim = 128

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.extract_features(x)


class SingleModelForCifar10(nn.Module):
    def __init__(self, args):
        super(SingleModelForCifar10, self).__init__()
        self.args = args
        self.backbone = models.resnet18(pretrained=False)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.backbone(x)
        return x
