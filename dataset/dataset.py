import copy
import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10
from .utils import get_labeled_data
from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
from typing import Any, Callable, Optional, Tuple


PHISHING_FILENAME = "PHISHING_full.csv"
PHISHING_SPLIT_SEED = 100
PHISHING_TEST_RATIO = 0.2


class CIFAR10_VFL(CIFAR10):
    def __init__(
            self,
            root: str,
            train: bool = True,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:

        super(CIFAR10, self).__init__(root, transform=transform,
                                      target_transform=target_transform)

        self.train = train  # training set or test set

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data: Any = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

        self.data_p = copy.deepcopy(self.data)

    def __getitem__(self, index):
        img, img_p, target = self.data[index], self.data_p[index], self.targets[index]

        img = Image.fromarray(img)

        img_poisoned = Image.fromarray(img_p)

        if self.transform is not None:
            img = self.transform(img)
            img_poisoned = self.transform(img_poisoned)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, img_poisoned, target, index


class UCIHAR_VFL(Dataset):
    def __init__(self, root, train, transforms):
        if train:
            self.data = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/train/X_train.txt')
            self.data_p = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/train/X_train.txt')
            self.targets = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/train/y_train.txt') - 1
        else:
            self.data = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/test/X_test.txt')
            self.data_p = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/test/X_test.txt')
            self.targets = np.loadtxt(root + '/UCI-HAR/UCI HAR Dataset/test/y_test.txt') - 1

    def __getitem__(self, index):
        x = self.data[index]
        x_poisoned = self.data_p[index]
        y = self.targets[index]
        return x, x_poisoned, y, index

    def __len__(self):
        return len(self.data)


class NUSWIDE_VFL(Dataset):
    def __init__(self, root, selected_labels, train, transforms):
        if train:
            X_image, X_text, Y = get_labeled_data(root + 'NUS_WIDE', selected_labels, None, 'Train')
            self.data = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)
            self.data_p = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)
            self.targets = torch.tensor(Y)
        else:
            X_image, X_text, Y = get_labeled_data(root + 'NUS_WIDE', selected_labels, None, 'Test')
            self.data = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)
            self.data_p = torch.cat((torch.tensor(X_image), torch.tensor(X_text)), dim=1)
            self.targets = torch.tensor(Y)

    def __getitem__(self, index):
        x = self.data[index]
        x_poisoned = self.data_p[index]
        y = self.targets[index]
        return x, x_poisoned, y, index

    def __len__(self):
        return len(self.data)


def _resolve_phishing_csv_path(root):
    candidate_paths = [
        os.path.join(root, "Phishing", PHISHING_FILENAME),
        os.path.join(root, "data_raw", "Phishing", PHISHING_FILENAME),
        os.path.join(root, PHISHING_FILENAME),
    ]
    for candidate_path in candidate_paths:
        if os.path.isfile(candidate_path):
            return candidate_path
    raise FileNotFoundError(
        "PHISHING dataset file was not found. Expected '{}' under one of: {}".format(
            PHISHING_FILENAME,
            ", ".join(candidate_paths),
        )
    )


def _get_phishing_label_column(dataframe):
    if "phishing" in dataframe.columns:
        return "phishing"
    if "Result" in dataframe.columns:
        return "Result"
    raise KeyError("PHISHING dataset must contain either a 'phishing' or 'Result' label column.")


class PHISHING_VFL(Dataset):
    def __init__(self, root, train, transforms):
        self.source_path = _resolve_phishing_csv_path(root)
        data = pd.read_csv(self.source_path)
        label_column = _get_phishing_label_column(data)
        feature_frame = data.drop(columns=[label_column])
        labels = data[label_column].astype(np.int64).to_numpy()

        train_features_raw, test_features_raw, train_labels, test_labels = train_test_split(
            feature_frame.to_numpy(dtype=np.float32),
            labels,
            test_size=PHISHING_TEST_RATIO,
            random_state=PHISHING_SPLIT_SEED,
            stratify=labels,
            shuffle=True,
        )

        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features_raw)
        test_features = scaler.transform(test_features_raw)

        train_data = torch.tensor(train_features, dtype=torch.float32)
        test_data = torch.tensor(test_features, dtype=torch.float32)
        train_target = torch.tensor(train_labels, dtype=torch.long)
        test_target = torch.tensor(test_labels, dtype=torch.long)
        self.feature_names = list(feature_frame.columns)
        self.input_dim = len(self.feature_names)

        if train:
            self.data = train_data
            self.data_p = copy.deepcopy(train_data)
            self.targets = train_target
        else:
            self.data = test_data
            self.data_p = copy.deepcopy(test_data)
            self.targets = test_target

    def __getitem__(self, index):
        x = self.data[index]
        x_poisoned = self.data_p[index]
        y = self.targets[index]
        return x, x_poisoned, y, index

    def __len__(self):
        return len(self.data)
