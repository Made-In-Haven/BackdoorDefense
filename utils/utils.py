import os
import pickle
import random

import numpy as np
import torch

DATASET_NAME_ALIASES = {
    "NUSWIDET": "NUSWIDE",
    "NUSWIDEI": "NUSWIDE",
    "IEEE-CIS-FRAUD": "IEEE_CIS_FRAUD",
    "IEEECISFRAUD": "IEEE_CIS_FRAUD",
    "IEEE-CIS_FRAUD": "IEEE_CIS_FRAUD",
}

LFBA_POISON_SOURCE_GRADIENT = "gradient"
LFBA_POISON_SOURCE_ORACLE_LABEL = "oracle_label"
LFBA_POISON_SOURCE_CHOICES = {
    LFBA_POISON_SOURCE_GRADIENT,
    LFBA_POISON_SOURCE_ORACLE_LABEL,
}


def raise_dataset_exception():
    raise Exception('Unknown dataset, please implement it.')


def raise_split_exception():
    raise Exception('Unknown split, please implement it.')


def raise_attack_exception():
    raise Exception('Unknown attack, please complement it.')


def canonicalize_dataset_name(dataset_name):
    normalized_name = str(dataset_name).upper()
    return DATASET_NAME_ALIASES.get(normalized_name, normalized_name)


def _split_sizes_evenly(total_dim, client_num):
    total_dim = int(total_dim)
    client_num = int(client_num)
    base_dim = total_dim // client_num
    remainder = total_dim % client_num
    return [base_dim + (1 if client_id < remainder else 0) for client_id in range(client_num)]


def get_client_input_sizes(args):
    dataset_name = canonicalize_dataset_name(getattr(args, "dataset", ""))
    client_num = int(args.client_num)

    if dataset_name == "IEEE_CIS_FRAUD":
        total_dim = int(
            getattr(
                args,
                "ieeecis_input_dim",
                getattr(args, "ieee_cis_fraud_input_dim", 432),
            )
        )
        return _split_sizes_evenly(total_dim, client_num)

    raise ValueError(
        "get_client_input_sizes currently supports IEEE_CIS_FRAUD only, got '{}'.".format(dataset_name)
    )


def get_local_output_dims(args):
    dataset_name = canonicalize_dataset_name(getattr(args, "dataset", ""))
    client_num = int(args.client_num)

    if dataset_name == "IEEE_CIS_FRAUD":
        configured_dims = getattr(
            args,
            "ieeecis_local_output_dims",
            getattr(args, "ieee_cis_fraud_local_output_dims", None),
        )
        if configured_dims is None:
            return [16 for _ in range(client_num)]
        if len(configured_dims) != client_num:
            raise ValueError(
                "IEEE_CIS_FRAUD local output dims length mismatch: expected {}, got {}.".format(
                    client_num,
                    len(configured_dims),
                )
            )
        return [int(output_dim) for output_dim in configured_dims]

    raise ValueError(
        "get_local_output_dims currently supports IEEE_CIS_FRAUD only, got '{}'.".format(dataset_name)
    )


def get_attack_target_label(args):
    attack_target_label = getattr(args, "attack_target_label", None)
    if attack_target_label is not None:
        return int(attack_target_label)

    fallback_target_label = getattr(args, "target_label", None)
    if fallback_target_label is not None:
        return int(fallback_target_label)

    raise ValueError("Attack target label is unavailable. Resolve it before running attack-dependent logic.")


def normalize_lfba_poison_source(poison_source):
    normalized_source = str(poison_source).strip().lower()
    if normalized_source not in LFBA_POISON_SOURCE_CHOICES:
        raise ValueError(
            "Unsupported lfba_poison_source '{}'. Expected one of {}.".format(
                poison_source,
                sorted(LFBA_POISON_SOURCE_CHOICES),
            )
        )
    return normalized_source


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_torch_artifact(path, map_location=None, logger=None, description="artifact", **kwargs):
    try:
        return torch.load(path, map_location=map_location, weights_only=True, **kwargs)
    except TypeError:
        return torch.load(path, map_location=map_location, **kwargs)
    except pickle.UnpicklingError as exc:
        if "Weights only load failed" not in str(exc):
            raise
        if logger is not None:
            logger.info(
                "=> %s '%s' requires full pickle loading; falling back to weights_only=False",
                description,
                path,
            )
        return torch.load(path, map_location=map_location, weights_only=False, **kwargs)
