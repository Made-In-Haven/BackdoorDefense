from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


LABEL_COLUMN = "isFraud"
KEY_COLUMN = "TransactionID"
DEFAULT_RANDOM_SEED = 42
DEFAULT_NEGATIVE_TO_POSITIVE_RATIO = 2


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    default_input_dir = repo_root / "dataset" / "data_raw" / "IEEE-CIS-Fraud"
    default_output_dir = repo_root / "dataset" / "data_raw" / "processed" / "IEEE-CIS-Fraud"

    parser = argparse.ArgumentParser(
        description="Preprocess the IEEE-CIS Fraud Detection training data."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input_dir,
        help="Directory containing train_transaction.csv and train_identity.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory used to save the preprocessed outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--negative-positive-ratio",
        type=int,
        default=DEFAULT_NEGATIVE_TO_POSITIVE_RATIO,
        help="Target ratio for negative:positive samples after balancing.",
    )
    return parser.parse_args()


def validate_input_files(input_dir: Path) -> Tuple[Path, Path]:
    transaction_path = input_dir / "train_transaction.csv"
    identity_path = input_dir / "train_identity.csv"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not transaction_path.exists():
        raise FileNotFoundError(f"Missing file: {transaction_path}")
    if not identity_path.exists():
        raise FileNotFoundError(f"Missing file: {identity_path}")

    return transaction_path, identity_path


def load_and_merge_data(input_dir: Path) -> pd.DataFrame:
    transaction_path, identity_path = validate_input_files(input_dir)

    transaction_df = pd.read_csv(transaction_path, low_memory=False)
    identity_df = pd.read_csv(identity_path, low_memory=False)

    if KEY_COLUMN not in transaction_df.columns:
        raise KeyError(f"Primary key column '{KEY_COLUMN}' not found in {transaction_path.name}.")
    if KEY_COLUMN not in identity_df.columns:
        raise KeyError(f"Primary key column '{KEY_COLUMN}' not found in {identity_path.name}.")
    if LABEL_COLUMN not in transaction_df.columns:
        raise KeyError(f"Label column '{LABEL_COLUMN}' not found in {transaction_path.name}.")

    merged_df = transaction_df.merge(identity_df, on=KEY_COLUMN, how="left")
    return merged_df


def split_features_and_target(merged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if LABEL_COLUMN not in merged_df.columns:
        raise KeyError(f"Label column '{LABEL_COLUMN}' is missing after merging.")
    if KEY_COLUMN not in merged_df.columns:
        raise KeyError(f"Primary key column '{KEY_COLUMN}' is missing after merging.")

    y = merged_df[LABEL_COLUMN].copy()
    X = merged_df.drop(columns=[LABEL_COLUMN, KEY_COLUMN]).copy()

    if LABEL_COLUMN in X.columns:
        raise ValueError(f"Feature matrix still contains label column '{LABEL_COLUMN}'.")

    return X, y


def identify_column_types(features: pd.DataFrame) -> Tuple[List[str], List[str]]:
    categorical_columns = features.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric_columns = [column for column in features.columns if column not in categorical_columns]
    return categorical_columns, numeric_columns


def fill_missing_values(
    features: pd.DataFrame,
    numeric_columns: List[str],
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    filled_features = features.copy()
    numeric_fill_values: Dict[str, float] = {}

    if numeric_columns:
        numeric_medians = filled_features[numeric_columns].median(numeric_only=True)
        # If a numeric column is entirely missing, its median is NaN. We fall back to 0.0
        # so the column can be preserved without dropping it.
        numeric_medians = numeric_medians.fillna(0.0)
        filled_features[numeric_columns] = filled_features[numeric_columns].fillna(numeric_medians)
        numeric_fill_values = {column: float(value) for column, value in numeric_medians.items()}

    if categorical_columns:
        filled_features[categorical_columns] = filled_features[categorical_columns].fillna("missing")

    return filled_features, numeric_fill_values


def encode_categorical_features(
    features: pd.DataFrame,
    categorical_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
    encoded_features = features.copy()
    categorical_mappings: Dict[str, Dict[str, int]] = {}

    for column in categorical_columns:
        encoder = LabelEncoder()
        column_values = encoded_features[column].astype(str)
        encoder.fit(column_values)
        encoded_features[column] = encoder.transform(column_values).astype(np.int32)
        categorical_mappings[column] = {
            category: int(index) for index, category in enumerate(encoder.classes_.tolist())
        }

    return encoded_features, categorical_mappings


def standardize_numeric_features(
    features: pd.DataFrame,
    numeric_columns: List[str],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    standardized_features = features.copy()
    scaling_statistics: Dict[str, Dict[str, float]] = {}

    if not numeric_columns:
        return standardized_features, scaling_statistics

    numeric_frame = standardized_features[numeric_columns].astype(np.float32)
    means = numeric_frame.mean(axis=0)
    stds = numeric_frame.std(axis=0, ddof=0)

    safe_stds = stds.replace(0.0, 1.0)
    standardized_numeric = (numeric_frame - means) / safe_stds

    zero_std_columns = stds[stds == 0.0].index.tolist()
    if zero_std_columns:
        standardized_numeric[zero_std_columns] = 0.0

    standardized_features[numeric_columns] = standardized_numeric.astype(np.float32)
    scaling_statistics = {
        column: {
            "mean": float(means[column]),
            "std": float(stds[column]),
        }
        for column in numeric_columns
    }

    return standardized_features, scaling_statistics


def validate_binary_labels(labels: pd.Series) -> None:
    unique_labels = set(pd.unique(labels.dropna()))
    if not unique_labels:
        raise ValueError("Label column is empty.")
    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            f"Label column '{LABEL_COLUMN}' must be binary with values 0/1, got: {sorted(unique_labels)}"
        )


def balance_dataset(
    features: pd.DataFrame,
    labels: pd.Series,
    negative_to_positive_ratio: int,
    seed: int,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, int], Dict[str, int]]:
    if negative_to_positive_ratio <= 0:
        raise ValueError("negative_to_positive_ratio must be a positive integer.")

    validate_binary_labels(labels)

    positive_indices = labels[labels == 1].index.to_numpy()
    negative_indices = labels[labels == 0].index.to_numpy()

    original_counts = {
        "negative": int(len(negative_indices)),
        "positive": int(len(positive_indices)),
    }

    if original_counts["positive"] == 0:
        raise ValueError("No positive samples found. Cannot perform class balancing.")
    if original_counts["negative"] == 0:
        raise ValueError("No negative samples found. Cannot perform class balancing.")

    target_negative_count = negative_to_positive_ratio * original_counts["positive"]
    if target_negative_count > original_counts["negative"]:
        raise ValueError(
            "The requested negative:positive ratio cannot be satisfied with random downsampling. "
            f"Required negatives: {target_negative_count}, available negatives: {original_counts['negative']}."
        )

    rng = np.random.default_rng(seed)
    sampled_negative_indices = rng.choice(
        negative_indices,
        size=target_negative_count,
        replace=False,
    )

    balanced_indices = np.concatenate([positive_indices, sampled_negative_indices])
    rng.shuffle(balanced_indices)

    balanced_features = features.loc[balanced_indices].reset_index(drop=True)
    balanced_labels = labels.loc[balanced_indices].reset_index(drop=True)

    balanced_counts = {
        "negative": int((balanced_labels == 0).sum()),
        "positive": int((balanced_labels == 1).sum()),
    }

    return balanced_features, balanced_labels, original_counts, balanced_counts


def save_outputs(
    output_dir: Path,
    balanced_features: pd.DataFrame,
    balanced_labels: pd.Series,
    categorical_columns: List[str],
    numeric_columns: List[str],
    categorical_mappings: Dict[str, Dict[str, int]],
    numeric_fill_values: Dict[str, float],
    scaling_statistics: Dict[str, Dict[str, float]],
    seed: int,
    negative_to_positive_ratio: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "X_balanced.npy", balanced_features.to_numpy(dtype=np.float32, copy=True))
    np.save(output_dir / "y_balanced.npy", balanced_labels.to_numpy(dtype=np.int64, copy=True))

    pd.Series(balanced_features.columns, name="feature_name").to_csv(
        output_dir / "feature_columns.csv",
        index=False,
    )

    with open(output_dir / "categorical_mappings.json", "w", encoding="utf-8") as file:
        json.dump(categorical_mappings, file, ensure_ascii=False, indent=2)

    preprocessing_metadata = {
        "label_column": LABEL_COLUMN,
        "primary_key_column": KEY_COLUMN,
        "seed": seed,
        "negative_to_positive_ratio": negative_to_positive_ratio,
        "categorical_columns": categorical_columns,
        "numeric_columns": numeric_columns,
        "numeric_fill_values": numeric_fill_values,
        "scaling_statistics": scaling_statistics,
        "output_files": {
            "X_balanced": "X_balanced.npy",
            "y_balanced": "y_balanced.npy",
            "feature_columns": "feature_columns.csv",
            "categorical_mappings": "categorical_mappings.json",
        },
    }

    with open(output_dir / "preprocessing_metadata.json", "w", encoding="utf-8") as file:
        json.dump(preprocessing_metadata, file, ensure_ascii=False, indent=2)


def print_summary(
    merged_shape: Tuple[int, int],
    categorical_count: int,
    numeric_count: int,
    original_counts: Dict[str, int],
    balanced_counts: Dict[str, int],
    balanced_feature_shape: Tuple[int, int],
) -> None:
    print(f"Merged data shape: {merged_shape}")
    print(f"Categorical columns: {categorical_count}")
    print(f"Numeric columns: {numeric_count}")
    print(
        "Original label counts: "
        f"negative={original_counts['negative']}, positive={original_counts['positive']}"
    )
    print(
        "Balanced label counts: "
        f"negative={balanced_counts['negative']}, positive={balanced_counts['positive']}"
    )
    print(f"Final feature matrix shape: {balanced_feature_shape}")


def main() -> int:
    args = parse_args()

    try:
        merged_df = load_and_merge_data(args.input_dir)
        merged_shape = merged_df.shape

        features, labels = split_features_and_target(merged_df)
        categorical_columns, numeric_columns = identify_column_types(features)

        features, numeric_fill_values = fill_missing_values(
            features=features,
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
        )
        features, categorical_mappings = encode_categorical_features(
            features=features,
            categorical_columns=categorical_columns,
        )
        features, scaling_statistics = standardize_numeric_features(
            features=features,
            numeric_columns=numeric_columns,
        )

        balanced_features, balanced_labels, original_counts, balanced_counts = balance_dataset(
            features=features,
            labels=labels,
            negative_to_positive_ratio=args.negative_positive_ratio,
            seed=args.seed,
        )

        save_outputs(
            output_dir=args.output_dir,
            balanced_features=balanced_features,
            balanced_labels=balanced_labels,
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
            categorical_mappings=categorical_mappings,
            numeric_fill_values=numeric_fill_values,
            scaling_statistics=scaling_statistics,
            seed=args.seed,
            negative_to_positive_ratio=args.negative_positive_ratio,
        )

        print_summary(
            merged_shape=merged_shape,
            categorical_count=len(categorical_columns),
            numeric_count=len(numeric_columns),
            original_counts=original_counts,
            balanced_counts=balanced_counts,
            balanced_feature_shape=balanced_features.shape,
        )
        print(f"Saved outputs to: {args.output_dir.resolve()}")
        return 0
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


"""
Output files:
- X_balanced.npy: preprocessed feature matrix after class balancing.
- y_balanced.npy: labels aligned with X_balanced.npy.
- feature_columns.csv: ordered feature names for the saved matrix.
- categorical_mappings.json: per-column label encoding mappings.
- preprocessing_metadata.json: preprocessing settings and numeric statistics.
"""
