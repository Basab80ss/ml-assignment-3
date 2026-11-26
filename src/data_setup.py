import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from sklearn.model_selection import StratifiedShuffleSplit


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


def load_raw_cifar10(data_root: str = "data"):
    """Load CIFAR-10 train (50k) and test (10k) arrays as float32 in [0, 1], shape (N, 3, 32, 32)."""
    train_ds = datasets.CIFAR10(root=data_root, train=True, download=False)
    test_ds = datasets.CIFAR10(root=data_root, train=False, download=False)

    x_train_full = np.transpose(train_ds.data, (0, 3, 1, 2)).astype(np.float32) / 255.0
    y_train_full = np.array(train_ds.targets, dtype=np.int64)

    x_test = np.transpose(test_ds.data, (0, 3, 1, 2)).astype(np.float32) / 255.0
    y_test = np.array(test_ds.targets, dtype=np.int64)
    return x_train_full, y_train_full, x_test, y_test


def get_or_create_split(y_train_full: np.ndarray, x_train_full: np.ndarray,
                        split_path: str, seed: int = 26002):
    """Create or load the stratified 45,000 train / 5,000 validation split and train channel stats."""
    if os.path.exists(split_path):
        data = np.load(split_path)
        return (
            data["train_indices"],
            data["val_indices"],
            data["train_mean"],
            data["train_std"],
        )

    os.makedirs(os.path.dirname(split_path), exist_ok=True)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=seed)
    train_idx, val_idx = next(splitter.split(np.zeros(len(y_train_full)), y_train_full))
    train_idx = np.sort(train_idx)
    val_idx = np.sort(val_idx)

    # Compute channel-wise mean and std strictly from the 45,000 training images
    x_tr = x_train_full[train_idx]
    train_mean = x_tr.mean(axis=(0, 2, 3), keepdims=True).astype(np.float32)
    train_std = x_tr.std(axis=(0, 2, 3), keepdims=True).astype(np.float32)

    np.savez(
        split_path,
        train_indices=train_idx,
        val_indices=val_idx,
        train_mean=train_mean,
        train_std=train_std,
        seed=np.array([seed], dtype=np.int64),
    )
    return train_idx, val_idx, train_mean, train_std


def get_or_create_q4_subset(y_train: np.ndarray, subset_path: str,
                            subset_size: int = 10000, seed: int = 26002):
    """Create or load the fixed stratified training subset for Q4 kernel SVMs."""
    if os.path.exists(subset_path):
        return np.load(subset_path)["subset_indices"]

    os.makedirs(os.path.dirname(subset_path), exist_ok=True)
    splitter = StratifiedShuffleSplit(n_splits=1, train_size=subset_size, random_state=seed)
    sub_idx, _ = next(splitter.split(np.zeros(len(y_train)), y_train))
    sub_idx = np.sort(sub_idx)
    np.savez(subset_path, subset_indices=sub_idx, seed=np.array([seed], dtype=np.int64))
    return sub_idx


def load_cifar10_splits(cfg: dict):
    """Load normalized and unnormalized [0,1] splits for train (45k), val (5k), and test (10k)."""
    seed = cfg["student"]["seed"]
    data_root = cfg["data"]["root"]
    split_path = cfg["data"]["split_file"]

    x_full, y_full, x_test_01, y_test = load_raw_cifar10(data_root)
    train_idx, val_idx, mean, std = get_or_create_split(y_full, x_full, split_path, seed=seed)

    x_train_01, y_train = x_full[train_idx], y_full[train_idx]
    x_val_01, y_val = x_full[val_idx], y_full[val_idx]

    return {
        "x_train_01": x_train_01,
        "y_train": y_train,
        "x_val_01": x_val_01,
        "y_val": y_val,
        "x_test_01": x_test_01,
        "y_test": y_test,
        "mean": mean,
        "std": std,
        "train_idx": train_idx,
        "val_idx": val_idx,
    }


def standardize_images(x_01: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Standardize [0, 1] images using training-set channel mean and std."""
    return ((x_01 - mean) / std).astype(np.float32)


def permute_pixels(x: np.ndarray, seed: int = 26002) -> np.ndarray:
    """Apply one fixed random permutation of the 1024 spatial (H, W) locations across all images."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(32 * 32)
    n, c, h, w = x.shape
    flat = x.reshape(n, c, h * w)
    return flat[:, :, perm].reshape(n, c, h, w).copy()


def permute_patches(x: np.ndarray, patch_size: int = 8, seed: int = 26003) -> np.ndarray:
    """Apply one fixed random permutation of non-overlapping (patch_size x patch_size) patches."""
    rng = np.random.default_rng(seed)
    n, c, h, w = x.shape
    grid_h, grid_w = h // patch_size, w // patch_size
    num_patches = grid_h * grid_w
    perm = rng.permutation(num_patches)

    # Reshape (N, C, gh, p, gw, p) -> (N, C, gh*gw, p, p)
    patches = x.reshape(n, c, grid_h, patch_size, grid_w, patch_size)
    patches = patches.transpose(0, 1, 2, 4, 3, 5).reshape(n, c, num_patches, patch_size, patch_size)
    patches = patches[:, :, perm, :, :]
    out = patches.reshape(n, c, grid_h, grid_w, patch_size, patch_size).transpose(0, 1, 2, 4, 3, 5)
    return out.reshape(n, c, h, w).copy()


def add_gaussian_noise(x_01: np.ndarray, sigma: float = 0.25, seed: int = 26002) -> np.ndarray:
    """Add deterministic Gaussian noise to [0, 1] images and clip to [0, 1]."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, size=x_01.shape).astype(np.float32)
    return np.clip(x_01 + noise, 0.0, 1.0)


class TensorImageDataset(Dataset):
    """Fast in-memory dataset with optional random crop + horizontal flip for shift augmentation."""

    def __init__(self, x: np.ndarray, y: np.ndarray, shift_augment: bool = False):
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.shift_augment = shift_augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img = self.x[idx]
        if self.shift_augment:
            # Random horizontal flip + random crop with 4-pixel zero padding
            if torch.rand(1).item() < 0.5:
                img = torch.flip(img, dims=[2])
            padded = torch.nn.functional.pad(img, (4, 4, 4, 4), mode="constant", value=0.0)
            top = int(torch.randint(0, 9, (1,)).item())
            left = int(torch.randint(0, 9, (1,)).item())
            img = padded[:, top:top + 32, left:left + 32]
        return img, self.y[idx]
