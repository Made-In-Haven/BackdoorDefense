import os

import numpy as np
import pandas as pd

from utils.utils import raise_dataset_exception

NUSWIDE_IMAGE_DIM = 634
NUSWIDE_TEXT_DIM = 1000


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
    for file in os.listdir(os.path.join(data_dir, features_path)):
        if file.startswith("_".join([dtype, "Normalized"])):
            print("Loading {}.".format(os.path.join(data_dir, features_path, file)))
            df = pd.read_csv(os.path.join(data_dir, features_path, file), header=None, sep=" ", engine="c")
            df.dropna(axis=1, inplace=True)
            dfs.append(df)
    data_XA = pd.concat(dfs, axis=1)
    data_X_image_selected = data_XA.loc[selected.index]
    # get XB, which are tags
    tag_path = "NUS_WID_Tags/"
    file = "_".join([dtype, "Tags1k"]) + ".dat"
    print("Loading {}.".format(file))
    tagsdf = pd.read_csv(os.path.join(data_dir, tag_path, file), header=None, sep="\t", engine="c")
    tagsdf.dropna(axis=1, inplace=True)
    data_X_text_selected = tagsdf.loc[selected.index]
    if n_samples is None:
        return data_X_image_selected.values[:], data_X_text_selected.values[:], np.argmax(selected.values[:], 1)
    return data_X_image_selected.values[:n_samples], data_X_text_selected.values[:n_samples], np.argmax(
        selected.values[:n_samples])


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
    if client_num == 2:
        return [(0, NUSWIDE_IMAGE_DIM), (NUSWIDE_IMAGE_DIM, NUSWIDE_IMAGE_DIM + NUSWIDE_TEXT_DIM)]

    # Preserve the full image modality on client 0, then split only the text modality across the remaining clients.
    text_slices = get_feature_slices(NUSWIDE_TEXT_DIM, client_num - 1)
    feature_slices = [(0, NUSWIDE_IMAGE_DIM)]
    for start, end in text_slices:
        feature_slices.append((NUSWIDE_IMAGE_DIM + start, NUSWIDE_IMAGE_DIM + end))
    return feature_slices


def get_dataset_feature_slices(args, total_dim):
    if args.dataset in {"NUSWIDE", "NUSWIDET"}:
        return get_nuswide_feature_slices(args.client_num)
    if args.dataset == "NUSWIDEI" and args.client_num == 2:
        return [(NUSWIDE_IMAGE_DIM, total_dim), (0, NUSWIDE_IMAGE_DIM)]
    return get_feature_slices(total_dim, args.client_num)


def get_client_feature_slice(args, total_dim, client_id):
    return get_dataset_feature_slices(args, total_dim)[client_id]


def get_client_input_dim(args, total_dim, client_id):
    start, end = get_client_feature_slice(args, total_dim, client_id)
    return end - start


def get_attacker_feature_slice(args, total_dim):
    return get_client_feature_slice(args, total_dim, args.attack_client_num)


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
    feature_slices = get_dataset_feature_slices(args, data.shape[1])
    return [data[:, start:end] for start, end in feature_slices]


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
    elif args.dataset == 'NUSWIDE' or args.dataset == 'NUSWIDET':
        # Preserve the original image/text split for 2 clients; otherwise split the concatenated vector contiguously.
        return split_dataset_vector_vfl(data, args)
    elif args.dataset == 'NUSWIDEI':
        return split_dataset_vector_vfl(data, args)
    else:
        raise_dataset_exception()
