import torch
from collections import Counter
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms as T

from .config import VALIDATION_SPLIT, DIRICHLET_ALPHA, BATCH_SIZE, NUM_CLIENTS, SEED


def get_cifar10(val_split=0.1, test_split=0.1):
    """
    Download CIFAR-10 and split the original train set into:
    - train
    - val
    - local test (per-client)
    Returns:
        x_train, y_train,
        x_val, y_val,
        x_local_test, y_local_test,
        x_test, y_test
    """
    data_train = datasets.CIFAR10("./data", train=True, download=True)
    data_test = datasets.CIFAR10("./data", train=False, download=True)

    x_train_full = data_train.data                  # [N, 32, 32, 3] uint8
    y_train_full = np.array(data_train.targets)     # [N]
    x_test = data_test.data
    y_test = np.array(data_test.targets)

    train_len = int((1 - val_split - test_split) * len(x_train_full))
    val_len = int(val_split * len(x_train_full))

    x_train = x_train_full[:train_len]
    x_val = x_train_full[train_len:train_len + val_len]
    x_local_test = x_train_full[train_len + val_len:]

    y_train = y_train_full[:train_len]
    y_val = y_train_full[train_len:train_len + val_len]
    y_local_test = y_train_full[train_len + val_len:]

    return (
        x_train,
        y_train,
        x_val,
        y_val,
        x_local_test,
        y_local_test,
        x_test,
        y_test,
    )


def dirichlet_split_indices(
    labels,
    num_clients,
    alpha=0.4,
    size_alpha=None,
    seed=0,
    ensure_nonempty=True,
):
    """
    Dirichlet partition:

    For each class:
      - Draw shares ~ Dir(alpha_vec) across clients
      - Allocate samples via Multinomial(shares)

    Optionally, use 'size_alpha' to vary expected client sizes via another Dirichlet.
    Returns:
        List of index lists per client, covering all samples with no overlap.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    classes = np.unique(labels)

    # indices per class (shuffled)
    class_bins = [rng.permutation(np.where(labels == c)[0]).tolist() for c in classes]

    # Dirichlet parameter vector over clients
    if size_alpha is None:
        alpha_vec = np.full(num_clients, alpha, dtype=np.float64)
    else:
        w = rng.dirichlet([size_alpha] * num_clients)
        alpha_vec = (alpha * num_clients) * (w / w.sum())

    client_indices = [[] for _ in range(num_clients)]

    for pool in class_bins:
        m = len(pool)
        if m == 0:
            continue
        shares = rng.dirichlet(alpha_vec)
        counts = rng.multinomial(m, shares)
        off = 0
        for k, take in enumerate(counts):
            if take:
                client_indices[k].extend(pool[off: off + take])
                off += take

    # Ensure no client is empty
    if ensure_nonempty:
        sizes = np.array([len(ix) for ix in client_indices])
        empties = np.where(sizes == 0)[0].tolist()
        for e in empties:
            donor = int(sizes.argmax())
            if sizes[donor] <= 1:
                break
            client_indices[e].append(client_indices[donor].pop())
            sizes[donor] -= 1
            sizes[e] += 1

    return client_indices


def ensure_min_per_client(index_lists, min_per=1, seed=0):
    """
    Steal random samples from largest clients until every client
    has at least 'min_per' samples.
    """
    rng = np.random.default_rng(seed)
    sizes = [len(x) for x in index_lists]
    need = [i for i, s in enumerate(sizes) if s < min_per]
    if not need:
        return index_lists

    donors = sorted(
        [(i, s) for i, s in enumerate(sizes)], key=lambda t: -t[1]
    )
    donors = [i for i, s in donors if s > min_per]

    for cid in need:
        while len(index_lists[cid]) < min_per and donors:
            d = donors[0]
            if len(index_lists[d]) <= min_per:
                donors.pop(0)
                continue
            j = int(rng.integers(len(index_lists[d])))
            index_lists[cid].append(index_lists[d].pop(j))
    return index_lists


class CustomImageDataset(Dataset):
    """
    Wraps CIFAR arrays (HWC, uint8) and applies torchvision transforms.
    """

    def __init__(self, inputs_np_uint8, labels_np_int64, transforms=None):
        self.inputs = inputs_np_uint8
        self.labels = labels_np_int64.astype(np.int64)
        self.transforms = transforms

    def __getitem__(self, index):
        img = self.inputs[index]           # ndarray HWC uint8
        lbl = int(self.labels[index])
        if self.transforms is not None:
            img = self.transforms(img)
        return img, torch.tensor(lbl, dtype=torch.long)

    def __len__(self):
        return self.inputs.shape[0]


def get_default_data_transforms(verbose=True):
    """
    Standard CIFAR-10 augmentation & normalization.
    """
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)

    cifar_train = T.Compose(
        [
            T.ToPILImage(),
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    cifar_eval = T.Compose(
        [
            T.ToPILImage(),
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    if verbose:
        print("\nData preprocessing (CIFAR-10):")
        for t in cifar_train.transforms:
            print(" -", t)
    return cifar_train, cifar_eval


def build_clients_from_cifar10_split(
    num_clients,
    batch_size,
    val_split=0.1,
    test_split=0.1,
    *,
    alpha=None,
    size_alpha=None,
    seed=0,
    drop_last_train=False,
):
    """
    Create per-client CIFAR-10 Datasets and DataLoaders using Dirichlet splits
    """
    (
        x_train,
        y_train,
        x_val,
        y_val,
        x_local_test,
        y_local_test,
        x_test,
        y_test,
    ) = get_cifar10(val_split=val_split, test_split=test_split)

    t_train, t_eval = get_default_data_transforms(verbose=False)
    a = float(DIRICHLET_ALPHA) if alpha is None else float(alpha)

    train_lists = dirichlet_split_indices(
        y_train, num_clients, alpha=a, size_alpha=size_alpha, seed=seed + 0
    )
    val_lists = dirichlet_split_indices(
        y_val, num_clients, alpha=a, size_alpha=size_alpha, seed=seed + 1
    )
    test_lists = dirichlet_split_indices(
        y_local_test, num_clients, alpha=a, size_alpha=size_alpha, seed=seed + 2
    )

    train_lists = ensure_min_per_client(train_lists, min_per=1, seed=seed + 100)

    clients_datasets = {}
    client_train_loaders = {}
    client_val_loaders = {}
    client_test_loaders = {}

    for cid in range(num_clients):
        tr_idx = np.array(train_lists[cid], dtype=int)
        va_idx = np.array(val_lists[cid], dtype=int)
        te_idx = np.array(test_lists[cid], dtype=int)

        if tr_idx.size == 0:
            tr_idx = np.array([np.random.randint(0, len(x_train))], dtype=int)

        ds_tr = CustomImageDataset(x_train[tr_idx], y_train[tr_idx], transforms=t_train)
        ds_va = CustomImageDataset(x_val[va_idx], y_val[va_idx], transforms=t_eval)
        ds_te = CustomImageDataset(
            x_local_test[te_idx], y_local_test[te_idx], transforms=t_eval
        )

        clients_datasets[cid] = {"train": ds_tr, "val": ds_va, "test": ds_te}
        client_train_loaders[cid] = DataLoader(
            ds_tr, batch_size=batch_size, shuffle=True, drop_last=drop_last_train
        )
        client_val_loaders[cid] = DataLoader(
            ds_va, batch_size=batch_size, shuffle=False
        )
        client_test_loaders[cid] = DataLoader(
            ds_te, batch_size=batch_size, shuffle=False
        )

    return (
        clients_datasets,
        client_train_loaders,
        client_val_loaders,
        client_test_loaders,
        (x_test, y_test),
    )


def get_global_test_loader(batch_size):
    """
    Global CIFAR-10 test loader (server-side).
    """
    *_unused, x_te, y_te = get_cifar10(val_split=VALIDATION_SPLIT, test_split=0.1)
    _, t_eval = get_default_data_transforms(verbose=False)
    ds_global_te = CustomImageDataset(x_te, y_te, transforms=t_eval)
    return DataLoader(ds_global_te, batch_size=batch_size, shuffle=False)


def print_client_class_dist(clients_datasets):
    """
    Quick per-client label histogram for sanity checking heterogeneity.
    """
    print("\nClient Class Distribution:")
    for cid, splits in clients_datasets.items():
        train_ds = splits["train"]

        labels = []
        for _, lbl in train_ds:
            if isinstance(lbl, torch.Tensor):
                labels.append(int(lbl.item()))
            else:
                labels.append(int(lbl))

        cnt = Counter(labels)
        total = sum(cnt.values())
        print(f"Client {cid}: {len(cnt)} classes, {total} samples")
        for cls in sorted(cnt.keys()):
            print(f"  Class {cls}: {cnt[cls]} samples")


