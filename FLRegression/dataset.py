import numpy as np
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from pathlib import Path

# Data Configuration
NUM_CLASSES = 10
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Dirichlet non-IID concentration parameter (lower = more non-IID)
DIRICHLET_ALPHA = 0.1

# Global cache for data
_data_cache = None


def reset_data_cache():
    """Reset the data cache to force reloading."""
    global _data_cache
    _data_cache = None


def _get_data_dir():
    """Get the directory to store/load CIFAR-10 data."""
    import os
    data_path = os.getenv("DATA_PATH")
    if data_path:
        return Path(data_path)

    current_dir = Path(__file__).parent
    data_dir = current_dir.parent / "data" / "cifar10"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def _load_and_preprocess_data():
    """Load and preprocess CIFAR-10 dataset once."""
    global _data_cache

    if _data_cache is not None:
        return _data_cache

    data_dir = _get_data_dir()

    # Standard CIFAR-10 normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Download and load CIFAR-10
    trainset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=transform
    )

    # Extract all training labels for partitioning
    train_targets = np.array(trainset.targets)

    _data_cache = {
        "trainset": trainset,
        "testset": testset,
        "train_targets": train_targets,
        "num_classes": NUM_CLASSES,
    }

    return _data_cache


def get_input_dim():
    """Get the input dimension info (for compatibility).
    Returns number of classes for CIFAR-10.
    """
    return NUM_CLASSES


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition of CIFAR-10 data with EXTREME non-IID partitioning.

    Non-IID Strategy (Dirichlet-based):
    1. Dirichlet distribution (alpha=0.1) for label skew across clients
    2. Quantity skew: uneven data distribution across clients
    3. Some clients predominantly see only 1-3 classes
    """
    data_cache = _load_and_preprocess_data()

    trainset = data_cache["trainset"]
    train_targets = data_cache["train_targets"]

    np.random.seed(42 + partition_id)

    # === Strategy 1: Dirichlet-based label distribution ===
    # Allocate data indices to each partition using Dirichlet distribution
    np.random.seed(42)  # Same seed for consistent partitioning across calls
    num_samples = len(train_targets)

    # Group indices by class
    class_indices = {}
    for cls in range(NUM_CLASSES):
        class_indices[cls] = np.where(train_targets == cls)[0]

    # Use Dirichlet distribution to assign proportions of each class to each client
    client_indices = [[] for _ in range(num_partitions)]

    for cls in range(NUM_CLASSES):
        indices = class_indices[cls].copy()
        np.random.shuffle(indices)

        # Dirichlet distribution for this class across clients
        proportions = np.random.dirichlet(
            np.repeat(DIRICHLET_ALPHA, num_partitions)
        )

        # Convert proportions to actual counts
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)
        splits = np.split(indices, proportions[:-1])

        for client_id in range(num_partitions):
            if client_id < len(splits):
                client_indices[client_id].extend(splits[client_id].tolist())

    # Get this partition's indices
    partition_indices = np.array(client_indices[partition_id])

    # === Strategy 2: Quantity skew ===
    # Some clients get more data, some get less
    np.random.seed(42 + partition_id)
    quantity_factor = 0.5 + (partition_id % 5) * 0.2  # 0.5 to 1.3

    max_samples = int(len(partition_indices) * quantity_factor)
    if max_samples < len(partition_indices):
        np.random.shuffle(partition_indices)
        partition_indices = partition_indices[:max_samples]

    # Ensure minimum data
    if len(partition_indices) < 50:
        all_indices = np.arange(num_samples)
        remaining = np.setdiff1d(all_indices, partition_indices)
        extra = np.random.choice(remaining, min(200, len(remaining)), replace=False)
        partition_indices = np.concatenate([partition_indices, extra])

    np.random.shuffle(partition_indices)

    # Split into train/test (80/20)
    split_idx = int(len(partition_indices) * 0.8)
    train_indices = partition_indices[:split_idx]
    test_indices = partition_indices[split_idx:]

    # Create subsets
    train_subset = Subset(trainset, train_indices)
    test_subset = Subset(trainset, test_indices)

    trainloader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(test_subset, batch_size=batch_size)

    return trainloader, testloader


def load_centralized_dataset():
    """Load entire CIFAR-10 test set for centralized evaluation."""
    data_cache = _load_and_preprocess_data()
    testset = data_cache["testset"]
    return DataLoader(testset, batch_size=128, shuffle=False)
