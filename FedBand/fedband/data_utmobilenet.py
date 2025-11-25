"""
UTMobileNetTraffic2021 data utilities for FedBand.

Dataset:
    UTMobileNetTraffic2021
    Wild Test Data CSV files for 14 mobile apps.

To download the dataset, visit:
    https://utexas.box.com/s/okrimcsz1mn9ec4j667kbb00d9gt16ii

In this implementation we use the CSV traffic files from the
"Wild Test Data" folder and build:
    - non-IID per-client train/validation loaders
    - a balanced global test loader
"""

import os
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline


# 1. Load & preprocess the raw UTMobileNet traffic CSV files


UTM_FILENAMES = [
    "facebook_manual_2019-03-31_11-36-11_d56097ed.csv",
    "gmail_manual_2019-03-31_12-08-30_d56097ed.csv",
    "google-drive_manual_2019-03-31_12-23-47_d56097ed.csv",
    "google-maps_manual_2019-03-31_22-15-35_5bd0c615.csv",
    "hangout_manual_2019-03-31_22-10-35_5bd0c615.csv",
    "hulu_manual_2019-03-31_13-19-08_d56097ed.csv",
    "instagram_manual_2019-03-31_13-03-02_d56097ed.csv",
    "messenger_manual_2019-03-31_12-00-40_d56097ed.csv",
    "netflix_manual_2019-03-31_11-51-55_d56097ed.csv",
    "pinterest_manual_2019-03-31_13-11-15_d56097ed.csv",
    "reddit_manual_2019-03-31_12-55-53_d56097ed.csv",
    "spotify_manual_2019-03-31_11-44-08_d56097ed.csv",
    "twitter_manual_2019-03-31_22-00-35_5bd0c615.csv",
    "youtube_manual_2019-03-31_22-05-35_5bd0c615.csv",
]

UTM_APPS = [
    "Facebook",
    "Gmail",
    "Google Drive",
    "Google Maps",
    "Hangouts",
    "Hulu",
    "Instagram",
    "Messenger",
    "Netflix",
    "Pinterest",
    "Reddit",
    "Spotify",
    "Twitter",
    "Youtube",
]

# Columns often dropped in UTMobileNet pipelines
DEFAULT_DROP_COLUMNS = [
    "udp.srcport",
    "udp.dstport",
    "udp.length",
    "udp.checksum",
    "gquic.puflags.rsv",
    "gquic.packet_number",
    "frame.time",
    "frame.number",
    "location",
]


def load_utmobilenet_dataframe(
    base_path: str = ".",
    drop_columns=None,
    verbose: bool = True,
):
    """
    Load and preprocess UTMobileNetTraffic2021 CSV files.

    Steps:
        * Read 14 application CSV files.
        * Add 'Application' label column.
        * Drop unused columns.
        * Impute missing values.
        * Label-encode 'Application'.
        * Convert other columns to numeric.
        * Min-max normalize numeric features.
        * Drop zero-variance columns.

    Returns:
        df              : pandas DataFrame with features + 'Application'
        label_encoder   : fitted LabelEncoder for the application labels
        label_mapping   : dict[int -> str] mapping label id to app name
    """
    if drop_columns is None:
        drop_columns = DEFAULT_DROP_COLUMNS

    dfs = []
    for fname, app in zip(UTM_FILENAMES, UTM_APPS):
        path = os.path.join(base_path, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(f"UTMobileNet file not found: {path}")
        df_i = pd.read_csv(path)
        df_i["Application"] = app
        dfs.append(df_i)

    df = pd.concat(dfs, ignore_index=True)

    # Drop unused columns if present
    df = df.drop(columns=drop_columns, errors="ignore")

    # Impute missing values
    imputer = SimpleImputer(strategy="most_frequent")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Label encode Application
    le = LabelEncoder()
    df["Application"] = le.fit_transform(df["Application"])

    # Convert all non-label columns to numeric
    for col in df.columns:
        if col != "Application":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Min-max normalize numeric features (excluding Application)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = numeric_cols.drop("Application")
    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # Drop columns with zero standard deviation
    std = df.std(numeric_only=True)
    std_cols = std[std > 0].index
    df = df[std_cols]

    label_mapping = {idx: cls for idx, cls in enumerate(le.classes_)}

    if verbose:
        print("UTMobileNet: label mapping (id -> app):")
        for k, v in label_mapping.items():
            print(f"  {k}: {v}")
        print("\nSamples per class:")
        print(df["Application"].value_counts())

    return df, le, label_mapping


# 2. Helpers: balancing test set & non-IID distribution

def balance_test_set(X_test: np.ndarray, y_test: np.ndarray, seed: int = 0):
    """
    Balance the test set so each class has the same number of samples
    (replicating classes with fewer samples).
    """
    rng = np.random.default_rng(seed)
    counter = Counter(y_test)
    avg_samples = int(sum(counter.values()) / len(counter))

    resampled_data = []
    resampled_labels = []

    for clazz in counter:
        idx = np.where(y_test == clazz)[0]
        # sample with replacement to reach avg_samples per class
        chosen = rng.choice(idx, size=avg_samples, replace=True)
        resampled_data.append(X_test[chosen])
        resampled_labels.append(y_test[chosen])

    X_bal = np.concatenate(resampled_data, axis=0)
    y_bal = np.concatenate(resampled_labels, axis=0)
    return X_bal, y_bal


def non_iid_data_distribution(
    y_train_val: np.ndarray,
    num_clients: int,
    alpha: float,
    min_samples_per_client: int = 1,
    max_samples_per_client: int = 1000,
    seed: int = 0,
):
    """
    Build a non-IID client index allocation using a Dirichlet prior
    over the 14 application classes (similar to your original code).

    For each client:
        - sample total #samples in [min_samples_per_client, max_samples_per_client]
        - sample Dirichlet(alpha) over classes
        - allocate samples according to those proportions
    """
    rng = np.random.default_rng(seed)
    classes = np.unique(y_train_val)
    num_classes = len(classes)

    client_data_indices = []

    for _ in range(num_clients):
        client_indices = []

        samples_per_client = int(
            rng.integers(min_samples_per_client, max_samples_per_client + 1)
        )

        # Dirichlet proportions
        class_proportions = rng.dirichlet([alpha] * num_classes)
        num_samples_each = (class_proportions * samples_per_client).astype(int)

        for c, n_samples in zip(classes, num_samples_each):
            if n_samples <= 0:
                continue
            class_indices = np.where(y_train_val == c)[0]
            if class_indices.size == 0:
                continue
            chosen = rng.choice(
                class_indices,
                size=min(n_samples, class_indices.size),
                replace=True,
            )
            client_indices.extend(chosen.tolist())

        # Ensure at least one sample
        if len(client_indices) == 0:
            fallback = int(rng.integers(0, len(y_train_val)))
            client_indices = [fallback]

        # Shuffle indices
        client_indices = rng.permutation(client_indices)
        client_data_indices.append(client_indices)

    return client_data_indices


def print_class_counts(train_loaders, val_loaders, global_test_loader):
    """
    Utility for debugging per-client class distributions.
    """
    print("\nPer-client class counts:")
    for client in train_loaders.keys():
        train_counts = Counter()
        val_counts = Counter()

        for _, labels in train_loaders[client]:
            for l in labels.tolist():
                train_counts[int(l)] += 1

        for _, labels in val_loaders[client]:
            for l in labels.tolist():
                val_counts[int(l)] += 1

        print(f"Client {client}:")
        print(f"  Train: {dict(train_counts)}")
        print(f"  Val  : {dict(val_counts)}")

    test_counts = Counter()
    for _, labels in global_test_loader:
        for l in labels.tolist():
            test_counts[int(l)] += 1

    print("\nGlobal test class counts:")
    print(dict(test_counts))


# 3. build client loaders + global test loader

def build_utmobilenet_clients(
    num_clients: int,
    batch_size: int,
    alpha: float = 0.4,
    val_size: float = 0.1,
    test_size: float = 0.1,
    base_path: str = ".",
    seed: int = 0,
    resample_target_per_class: int = 30_000,
    verbose: bool = True,
):
    """
    High-level function to construct UTMobileNet loaders for FedBand.

    Returns:
        train_loaders       : dict[cid] -> DataLoader (client train)
        val_loaders         : dict[cid] -> DataLoader (client val)
        global_test_loader  : DataLoader (balanced global test set)
        num_features        : int, feature dimension for ResNet1D
        num_classes         : int, number of application classes
        label_mapping       : dict[int -> str], label id -> app name
    """
    # 1) Load + preprocess full dataframe
    df, label_encoder, label_mapping = load_utmobilenet_dataframe(
        base_path=base_path, verbose=verbose
    )
    num_classes = len(label_encoder.classes_)

    # Separate features and labels
    X = df.drop("Application", axis=1)
    y = df["Application"]

    # 2) Split into train/val pool vs global test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # Balance test set across classes
    X_test_bal, y_test_bal = balance_test_set(
        X_test.values, y_test.values, seed=seed
    )

    # Build global test loader
    X_test_tensor = torch.tensor(X_test_bal, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test_bal, dtype=torch.long)
    global_test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    global_test_loader = DataLoader(
        global_test_dataset, batch_size=batch_size, shuffle=False
    )

    # 3) Balance training part with SMOTE + RandomUnderSampler
    oversample_classes = {
        cls: resample_target_per_class
        for cls, cnt in Counter(y_train_val).items()
        if cnt < resample_target_per_class
    }
    undersample_classes = {
        cls: resample_target_per_class
        for cls, cnt in Counter(y_train_val).items()
        if cnt > resample_target_per_class
    }

    resampling = Pipeline(
        steps=[
            ("over", SMOTE(sampling_strategy=oversample_classes)),
            ("under", RandomUnderSampler(sampling_strategy=undersample_classes)),
        ]
    )

    X_resampled, y_resampled = resampling.fit_resample(X_train_val, y_train_val)

    # Back to pandas objects for easier indexing
    X_resampled = pd.DataFrame(X_resampled, columns=X_train_val.columns)
    y_resampled = pd.Series(y_resampled)

    if verbose:
        print("\nClass counts BEFORE resampling:", Counter(y_train_val))
        print("Class counts AFTER  resampling:", Counter(y_resampled))

    # 4) Non-IID client index allocation on resampled training set
    client_indices_list = non_iid_data_distribution(
        y_train_val=y_resampled.values,
        num_clients=num_clients,
        alpha=alpha,
        min_samples_per_client=1,
        max_samples_per_client=1000,
        seed=seed,
    )

    # 5) Per-client train/val DataLoaders
    train_loaders = {}
    val_loaders = {}

    for cid, indices in enumerate(client_indices_list):
        client_X = X_resampled.iloc[indices, :]
        client_y = y_resampled.iloc[indices]

        X_tr, X_va, y_tr, y_va = train_test_split(
            client_X,
            client_y,
            test_size=val_size,
            random_state=seed,
        )

        train_dataset = TensorDataset(
            torch.tensor(X_tr.values, dtype=torch.float32),
            torch.tensor(y_tr.values, dtype=torch.long),
        )
        val_dataset = TensorDataset(
            torch.tensor(X_va.values, dtype=torch.float32),
            torch.tensor(y_va.values, dtype=torch.long),
        )

        train_loaders[cid] = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        val_loaders[cid] = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False
        )

    num_features = X.shape[1]

    return (
        train_loaders,
        val_loaders,
        global_test_loader,
        num_features,
        num_classes,
        label_mapping,
    )

