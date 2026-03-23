import os
import random

import numpy as np
import torch


def raise_dataset_exception():
    raise Exception('Unknown dataset, please implement it.')


def raise_split_exception():
    raise Exception('Unknown split, please implement it.')


def raise_attack_exception():
    raise Exception('Unknown attack, please complement it.')


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
