"""pytorchexample: Model, data loading, and training for disposable income prediction."""

import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

SEED = 2023


def set_seed(seed: int = SEED):
    #Seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Path to dataset
DATA_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "data", "indianPersonalFinanceAndSpendingHabits.csv",
)

TARGET_COL = "Disposable_Income"
CATEGORICAL_COLS = ["Occupation", "City_Tier"]

_cached_data = None 


class Net(nn.Module):
    """MLP Model for Personal Finance Prediction (Regression)."""

    def __init__(self, input_dim: int = 26):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 1)  # Single output for regression

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x).squeeze(-1)


def _load_and_preprocess():
    """Load CSV, one-hot encode categoricals, standardize, build partition keys."""
    global _cached_data
    if _cached_data is not None:
        return _cached_data

    df = pd.read_csv(DATA_PATH)

    # Target — standardize so the model trains in unit-scale space
    y_raw = df[TARGET_COL].values.astype(np.float32)
    y_mean = float(y_raw.mean())
    y_std = float(y_raw.std()) + 1e-8
    y = (y_raw - y_mean) / y_std

    # Build non-IID partition keys BEFORE dropping categoricals
    income_brackets = pd.qcut(df["Income"], q=3, labels=["low", "mid", "high"])
    partition_keys = (
        df["Occupation"].astype(str)
        + "_" + df["City_Tier"].astype(str)
        + "_" + income_brackets.astype(str)
    ).values

    # Features: drop target, one-hot encode categoricals
    X = df.drop(columns=[TARGET_COL])
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS, drop_first=True)

    # Standardize features
    X_vals = X.values.astype(np.float32)
    means = X_vals.mean(axis=0)
    stds = X_vals.std(axis=0) + 1e-8
    X_scaled = (X_vals - means) / stds

    input_dim = X_scaled.shape[1]
    _cached_data = (X_scaled, y, partition_keys, input_dim, y_mean, y_std)
    return _cached_data


def get_input_dim() -> int:
    """Return the number of input features after preprocessing."""
    _, _, _, input_dim, _, _ = _load_and_preprocess()
    return input_dim


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    #Load non-IID partitioned data for a specific client.
    X, y, partition_keys, _, _, _ = _load_and_preprocess()

    # Deterministic non-IID partitioning
    unique_keys = sorted(set(partition_keys))
    rng = np.random.RandomState(42)
    rng.shuffle(unique_keys)

    # Round-robin assignment of demographic groups to clients
    key_to_partition = {k: i % num_partitions for i, k in enumerate(unique_keys)}

    # Collect sample indices for this partition
    indices = np.array(
        [i for i, k in enumerate(partition_keys) if key_to_partition[k] == partition_id]
    )

    # Fallback if partition got no data
    if len(indices) < 2:
        rng_fb = np.random.RandomState(partition_id)
        indices = rng_fb.choice(len(y), size=min(200, len(y)), replace=False)

    # Shuffle and 80/20 split
    rng_part = np.random.RandomState(42 + partition_id)
    rng_part.shuffle(indices)
    split = max(1, int(0.8 * len(indices)))
    train_idx, test_idx = indices[:split], indices[split:]

    train_ds = TensorDataset(torch.tensor(X[train_idx]), torch.tensor(y[train_idx]))
    test_ds = TensorDataset(torch.tensor(X[test_idx]), torch.tensor(y[test_idx]))

    g = torch.Generator()
    g.manual_seed(SEED + partition_id)
    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, generator=g)
    testloader = DataLoader(test_ds, batch_size=batch_size)
    return trainloader, testloader


def load_centralized_test():
    """Load a centralized test set (last 20% of data)."""
    X, y, _, _, _, _ = _load_and_preprocess()
    split = int(0.8 * len(y))
    test_ds = TensorDataset(torch.tensor(X[split:]), torch.tensor(y[split:]))
    return DataLoader(test_ds, batch_size=128)


def load_centralized_train():
    """Load the centralized training set (first 80% of data)."""
    X, y, _, _, _, _ = _load_and_preprocess()
    split = int(0.8 * len(y))
    train_ds = TensorDataset(torch.tensor(X[:split]), torch.tensor(y[:split]))
    return DataLoader(train_ds, batch_size=128)


def train(net, trainloader, epochs, lr, device, mu=0.0, global_params=None, grad_clip=0.0):
    #Train model with optional FedProx proximal term and gradient clipping.
    net.to(device)
    net.train()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    running_loss = 0.0

    for _ in range(epochs):
        for features, targets in trainloader:
            features, targets = features.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(net(features), targets)

            # FedProx proximal term: (mu/2) * ||w - w_global||^2
            if mu > 0.0 and global_params is not None:
                prox = sum(
                    ((lp - gp) ** 2).sum()
                    for lp, gp in zip(net.parameters(), global_params)
                )
                loss = loss + (mu / 2.0) * prox

            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
            optimizer.step()
            running_loss += loss.item()

    return running_loss / max(1, epochs * len(trainloader))


def test(net, testloader, device):
    """Evaluate model. Returns (mse_loss, r2_score)."""
    net.to(device)
    net.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []

    with torch.no_grad():
        for features, targets in testloader:
            features, targets = features.to(device), targets.to(device)
            preds = net(features)
            total_loss += nn.MSELoss()(preds, targets).item()
            all_preds.append(preds.cpu())
            all_targets.append(targets.cpu())

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)

    # R-squared
    ss_res = ((all_targets - all_preds) ** 2).sum().item()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum().item()
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)

    return total_loss / max(1, len(testloader)), r2
