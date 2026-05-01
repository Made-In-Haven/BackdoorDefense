import torch
import torch.nn as nn

from utils.utils import get_client_input_sizes, get_local_output_dims


class GlobalModelForIEEECIS_FRAUD(nn.Module):
    def __init__(self, args):
        super(GlobalModelForIEEECIS_FRAUD, self).__init__()
        fused_dim = sum(get_local_output_dims(args))
        hidden_dim_1 = max(32, fused_dim * 2)
        hidden_dim_2 = max(16, hidden_dim_1 // 2)
        num_classes = getattr(args, "ieeecis_num_classes", 2)
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, num_classes),
        )
        self.args = args
        self.num_classes = num_classes

    def forward(self, input_list):
        tensor_t = torch.cat(input_list, dim=1)
        return self.classifier(tensor_t)

    def forward_from_concat(self, concat_embeddings):
        return self.classifier(concat_embeddings)


class LocalModelForIEEECIS_FRAUD(nn.Module):
    def __init__(self, args, client_number):
        super(LocalModelForIEEECIS_FRAUD, self).__init__()
        self.args = args
        input_dim = get_client_input_sizes(args)[client_number]
        output_dim = get_local_output_dims(args)[client_number]
        hidden_dim_1 = max(128, min(512, input_dim))
        hidden_dim_2 = max(64, hidden_dim_1 // 2)
        hidden_dim_3 = max(32, hidden_dim_2 // 2)
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim_1),
            nn.ReLU(),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(),
            nn.Linear(hidden_dim_2, hidden_dim_3),
            nn.ReLU(),
            nn.Linear(hidden_dim_3, output_dim),
        )
        self.output_dim = output_dim

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        return self.extract_features(x)


GlobalModelForIEEECISFRAUD = GlobalModelForIEEECIS_FRAUD
LocalModelForIEEECISFRAUD = LocalModelForIEEECIS_FRAUD
