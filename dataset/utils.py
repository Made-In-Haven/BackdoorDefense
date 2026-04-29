import os

import numpy as np
import pandas as pd

from utils.utils import raise_dataset_exception

NUSWIDE_IMAGE_VIEW_DIMS = (
    ("CH", 64),
    ("CM55", 225),
    ("CORR", 144),
    ("EDH", 73),
    ("WT", 128),
)
NUSWIDE_TEXT_OUTPUT_DIM = 60
NUSWIDE_IMAGE_VIEW_OUTPUT_DIM = 8
NUSWIDE_TEXT_DIM = 1000
NUSWIDE_IMAGE_DIM = sum(view_dim for _, view_dim in NUSWIDE_IMAGE_VIEW_DIMS)
NUSWIDE_TOTAL_DIM = NUSWIDE_IMAGE_DIM + NUSWIDE_TEXT_DIM


def _build_nuswide_view_slices():
    view_slices = {}
    start = 0
    for view_name, view_dim in NUSWIDE_IMAGE_VIEW_DIMS:
        end = start + view_dim
        view_slices[view_name] = (start, end)
        start = end
    return view_slices


NUSWIDE_IMAGE_VIEW_SLICES = _build_nuswide_view_slices()
NUSWIDE_CLIENT_VIEW_GROUPS = {
    2: [("CH", "CM55", "CORR", "EDH", "WT")],
    3: [("CH", "CM55", "EDH"), ("CORR", "WT")],
    4: [("CH", "CM55"), ("CORR",), ("EDH", "WT")],
    5: [("CH", "CM55"), ("CORR",), ("WT",), ("EDH",)],
}
NUSWIDE_CLIENT_VIEW_OUTPUT_DIMS = {
    5: (24, 16, 16, 16),
}
NUSWIDE_SUPPORTED_CLIENT_NUMS = tuple(sorted(NUSWIDE_CLIENT_VIEW_GROUPS))


def _get_nuswide_image_client_output_dim(client_num, image_client_index, view_group):
    output_dims = NUSWIDE_CLIENT_VIEW_OUTPUT_DIMS.get(client_num)
    if output_dims is not None:
        return output_dims[image_client_index]
    return len(view_group) * NUSWIDE_IMAGE_VIEW_OUTPUT_DIM


def _build_nuswide_local_output_dims():
    local_output_dims = {}
    for client_num, view_groups in NUSWIDE_CLIENT_VIEW_GROUPS.items():
        local_output_dims[client_num] = [NUSWIDE_TEXT_OUTPUT_DIM] + [
            _get_nuswide_image_client_output_dim(client_num, index, view_group)
            for index, view_group in enumerate(view_groups)
        ]
    return local_output_dims


NUSWIDE_LOCAL_OUTPUT_DIMS = _build_nuswide_local_output_dims()


def _merge_adjacent_ranges(ranges):
    if not ranges:
        return tuple()
    merged_ranges = [list(ranges[0])]
    for start, end in ranges[1:]:
        if merged_ranges[-1][1] == start:
            merged_ranges[-1][1] = end
        else:
            merged_ranges.append([start, end])
    return tuple((start, end) for start, end in merged_ranges)


def _get_nuswide_view_group_ranges(view_group):
    ranges = [NUSWIDE_IMAGE_VIEW_SLICES[view_name] for view_name in view_group]
    return _merge_adjacent_ranges(ranges)


def _expand_feature_ranges(feature_ranges):
    if not feature_ranges:
        return np.array([], dtype=np.int64)
    expanded = [np.arange(start, end, dtype=np.int64) for start, end in feature_ranges]
    return np.concatenate(expanded, axis=0)


def validate_nuswide_client_num(client_num):
    if client_num not in NUSWIDE_CLIENT_VIEW_GROUPS:
        raise ValueError(
            "NUSWIDE only supports client_num in {}.".format(NUSWIDE_SUPPORTED_CLIENT_NUMS)
        )


def validate_nuswide_total_dim(total_dim):
    if int(total_dim) != NUSWIDE_TOTAL_DIM:
        raise ValueError(
            "NUSWIDE expects concatenated feature dim {} ([image:{}] + [text:{}]), got {}.".format(
                NUSWIDE_TOTAL_DIM,
                NUSWIDE_IMAGE_DIM,
                NUSWIDE_TEXT_DIM,
                total_dim,
            )
        )


def validate_nuswide_attack_client_num(client_num, attack_client_num):
    validate_nuswide_client_num(client_num)
    if attack_client_num < 0 or attack_client_num >= client_num:
        raise ValueError(
            "NUSWIDE attack_client_num must be in [0, {}], got {}.".format(client_num - 1, attack_client_num)
        )


def get_nuswide_feature_file_name(dtype, view_name):
    return "{}_Normalized_{}.dat".format(dtype, view_name)


def get_nuswide_client_layout(client_num):
    validate_nuswide_client_num(client_num)
    layout = [
        {
            "client_id": 0,
            "modality": "text",
            "views": ("Tags1k",),
            "slice": (NUSWIDE_IMAGE_DIM, NUSWIDE_TOTAL_DIM),
            "feature_ranges": ((NUSWIDE_IMAGE_DIM, NUSWIDE_TOTAL_DIM),),
            "input_dim": NUSWIDE_TEXT_DIM,
            "output_dim": NUSWIDE_TEXT_OUTPUT_DIM,
            "description": "client0 = TEXT[{}]".format(NUSWIDE_TEXT_DIM),
        }
    ]
    for offset, view_group in enumerate(NUSWIDE_CLIENT_VIEW_GROUPS[client_num], start=1):
        feature_ranges = _get_nuswide_view_group_ranges(view_group)
        input_dim = sum(end - start for start, end in feature_ranges)
        layout.append(
            {
                "client_id": offset,
                "modality": "image",
                "views": tuple(view_group),
                "slice": feature_ranges[0] if len(feature_ranges) == 1 else None,
                "feature_ranges": feature_ranges,
                "input_dim": input_dim,
                "output_dim": _get_nuswide_image_client_output_dim(client_num, offset - 1, view_group),
                "description": "client{} = {} [{}]".format(offset, " + ".join(view_group), input_dim),
            }
        )
    return layout


def describe_nuswide_client_partition(client_num):
    return [client_layout["description"] for client_layout in get_nuswide_client_layout(client_num)]


def get_labeled_data(data_dir, selected_label, n_samples, dtype="Train"):
    # get labels
    data_path = "Groundtruth/TrainTestLabels/"
    dfs = []
    for label in selected_label:
        file = os.path.join(data_dir, data_path, "_".join(["Labels", label, dtype]) + ".txt")
        print("Loading {}.".format(file))
        df = pd.read_csv(file, header=None, engine="c")
        df.columns = [label]
        dfs.append(df)
    data_labels = pd.concat(dfs, axis=1)
    if len(selected_label) > 1:
        selected = data_labels[data_labels.sum(axis=1) == 1]
    else:
        selected = data_labels
    # get XA, which are image low level features
    features_path = "Low_Level_Features"
    dfs = []
    features_dir = os.path.join(data_dir, features_path)
    for view_name, expected_dim in NUSWIDE_IMAGE_VIEW_DIMS:
        file_name = get_nuswide_feature_file_name(dtype, view_name)
        file_path = os.path.join(features_dir, file_name)
        if not os.path.isfile(file_path):
            raise FileNotFoundError(
                "Missing NUSWIDE feature file '{}'. Expected fixed image view order: {}.".format(
                    file_path,
                    [view for view, _ in NUSWIDE_IMAGE_VIEW_DIMS],
                )
            )
        print("Loading {}.".format(file_path))
        df = pd.read_csv(file_path, header=None, sep=" ", engine="c")
        df.dropna(axis=1, inplace=True)
        if df.shape[1] != expected_dim:
            raise ValueError(
                "Unexpected NUSWIDE view dim for {}: expected {}, got {}.".format(
                    view_name,
                    expected_dim,
                    df.shape[1],
                )
            )
        dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    if data_XA.shape[1] != NUSWIDE_IMAGE_DIM:
        raise ValueError(
            "Unexpected NUSWIDE image dim after ordered concatenation: expected {}, got {}.".format(
                NUSWIDE_IMAGE_DIM,
                data_XA.shape[1],
            )
        )
    data_X_image_selected = data_XA.loc[selected.index]
    # get XB, which are tags
    tag_path = "NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    print("Loading {}.".format(file))
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t", engine="c")
    tagsdf.dropna(axis=1, inplace=True)
    if tagsdf.shape[1] != NUSWIDE_TEXT_DIM:
        raise ValueError(
            "Unexpected NUSWIDE text dim for Tags1k: expected {}, got {}.".format(
                NUSWIDE_TEXT_DIM,
                tagsdf.shape[1],
            )
        )
    data_X_text_selected = tagsdf.loc[selected.index]
    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], np.argmax(selected.values[:], 1)
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], np.argmax(
        selected.values[:n_samples], 1)


def get_feature_slices(total_dim, client_num):
    base_dim = total_dim // client_num
    remainder = total_dim % client_num
    feature_slices = []
    start = 0
    for client_id in range(client_num):
        current_dim = base_dim + (1 if client_id < remainder else 0)
        end = start + current_dim
        feature_slices.append((start, end))
        start = end
    return feature_slices


def get_nuswide_feature_slices(client_num):
    validate_nuswide_client_num(client_num)
    return [client_layout["feature_ranges"] for client_layout in get_nuswide_client_layout(client_num)]


def get_nuswide_feature_indices(client_num):
    validate_nuswide_client_num(client_num)
    return [
        _expand_feature_ranges(client_layout["feature_ranges"])
        for client_layout in get_nuswide_client_layout(client_num)
    ]


def get_nuswide_local_output_dims(client_num):
    validate_nuswide_client_num(client_num)
    return list(NUSWIDE_LOCAL_OUTPUT_DIMS[client_num])


def get_nuswide_client_output_dim(client_num, client_id):
    return get_nuswide_local_output_dims(client_num)[client_id]


def get_dataset_feature_slices(args, total_dim):
    if args.dataset == "NUSWIDE":
        validate_nuswide_total_dim(total_dim)
        return get_nuswide_feature_slices(args.client_num)
    return get_feature_slices(total_dim, args.client_num)


def get_dataset_feature_indices(args, total_dim):
    if args.dataset == "NUSWIDE":
        validate_nuswide_total_dim(total_dim)
        return get_nuswide_feature_indices(args.client_num)
    return [np.arange(start, end, dtype=np.int64) for start, end in get_feature_slices(total_dim, args.client_num)]


def get_client_feature_slice(args, total_dim, client_id):
    feature_indices = get_dataset_feature_indices(args, total_dim)[client_id]
    if feature_indices.size == 0:
        return 0, 0
    if not np.array_equal(feature_indices, np.arange(feature_indices[0], feature_indices[-1] + 1, dtype=np.int64)):
        raise ValueError(
            "Dataset '{}' client{} uses non-contiguous feature indices; use get_client_feature_indices instead.".format(
                args.dataset,
                client_id,
            )
        )
    return int(feature_indices[0]), int(feature_indices[-1] + 1)


def get_client_feature_indices(args, total_dim, client_id):
    return get_dataset_feature_indices(args, total_dim)[client_id]


def get_client_input_dim(args, total_dim, client_id):
    return int(len(get_client_feature_indices(args, total_dim, client_id)))


def get_attacker_feature_slice(args, total_dim):
    return get_client_feature_slice(args, total_dim, args.attack_client_num)


def get_attacker_feature_indices(args, total_dim):
    return get_client_feature_indices(args, total_dim, args.attack_client_num)


def get_image_slices(total_width, client_num):
    return get_feature_slices(total_width, client_num)


def get_client_image_slice(args, total_width, client_id):
    return get_image_slices(total_width, args.client_num)[client_id]


def get_attacker_image_slice(args, total_width):
    return get_client_image_slice(args, total_width, args.attack_client_num)


def split_vector_vfl(data, client_num):
    feature_slices = get_feature_slices(data.shape[1], client_num)
    return [data[:, start:end] for start, end in feature_slices]


def split_dataset_vector_vfl(data, args):
    feature_indices = get_dataset_feature_indices(args, data.shape[1])
    return [data[:, indexes] for indexes in feature_indices]


def split_image_vfl(data, client_num):
    image_slices = get_image_slices(data.shape[-1], client_num)
    return [data[:, :, :, start:end] for start, end in image_slices]


def split_vfl(data, args):
    if args.dataset == 'CIFAR10':
        # Split the image into contiguous width slices so CIFAR10 can support different client counts.
        return split_image_vfl(data, args.client_num)
    elif args.dataset == 'UCIHAR':
        # Vector features are split into contiguous client-specific segments.
        return split_vector_vfl(data, args.client_num)
    elif args.dataset == 'PHISHING':
        return split_vector_vfl(data, args.client_num)
    elif args.dataset == 'IEEE_CIS_FRAUD':
        return split_vector_vfl(data, args.client_num)
    elif args.dataset == 'NUSWIDE':
        # NUSWIDE uses a fixed multimodal split: client 0 keeps full text and the other clients receive grouped image views.
        return split_dataset_vector_vfl(data, args)
    else:
        raise_dataset_exception()
