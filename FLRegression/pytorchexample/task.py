"""pytorchexample: A Flower / PyTorch app for Personal Finance Prediction."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================
TARGET_COLUMN = "Disposable_Income"
PARTITION_COLUMN = "City_Tier"  # Non-IID partitioning by city tier

# Feature columns (excluding target and partition column from features)
CATEGORICAL_COLUMNS = ["Occupation", "City_Tier"]
NUMERICAL_COLUMNS = [
    "Income", "Age", "Dependents", "Rent", "Loan_Repayment", "Insurance",
    "Groceries", "Transport", "Eating_Out", "Entertainment", "Utilities",
    "Healthcare", "Education", "Miscellaneous", "Desired_Savings_Percentage",
    "Desired_Savings", "Potential_Savings_Groceries", "Potential_Savings_Transport",
    "Potential_Savings_Eating_Out", "Potential_Savings_Entertainment",
    "Potential_Savings_Utilities", "Potential_Savings_Healthcare",
    "Potential_Savings_Education", "Potential_Savings_Miscellaneous"
]

# Global cache for data and preprocessors
_data_cache = None
_preprocessors = None


def reset_data_cache():
    """Reset the data cache to force reloading."""
    global _data_cache, _preprocessors
    _data_cache = None
    _preprocessors = None


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
        return self.fc4(x)


def _get_data_path():
    """Get the path to the CSV data file."""
    # Look for the CSV file relative to the project root
    current_dir = Path(__file__).parent.parent
    return current_dir / "indianPersonalFinanceAndSpendingHabits.csv"


def _load_and_preprocess_data():
    """Load and preprocess the entire dataset once."""
    global _data_cache, _preprocessors
    
    if _data_cache is not None:
        return _data_cache, _preprocessors
    
    # Load CSV
    data_path = _get_data_path()
    df = pd.read_csv(data_path)
    
    # Initialize preprocessors
    label_encoders = {}
    scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Encode categorical columns
    df_processed = df.copy()
    for col in CATEGORICAL_COLUMNS:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    # Get feature columns (all numerical + encoded categorical, excluding target)
    feature_cols = NUMERICAL_COLUMNS + CATEGORICAL_COLUMNS
    feature_cols = [c for c in feature_cols if c != TARGET_COLUMN]
    
    # Prepare features and target
    X = df_processed[feature_cols].values.astype(np.float32)
    y = df_processed[TARGET_COLUMN].values.astype(np.float32).reshape(-1, 1)
    
    # Scale features
    X_scaled = scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)
    
    # Store partitioning columns (before encoding) for extreme non-IID
    city_tiers = df[PARTITION_COLUMN].values
    occupations = df["Occupation"].values
    incomes = df["Income"].values
    
    # Create combined partition key: Occupation + City_Tier + Income_Bracket
    # This creates ~12+ unique groups for more heterogeneity
    income_brackets = pd.qcut(incomes, q=3, labels=["Low", "Medium", "High"]).astype(str)
    combined_keys = np.array([f"{o}_{c}_{i}" for o, c, i in zip(occupations, city_tiers, income_brackets)])
    
    _preprocessors = {
        "label_encoders": label_encoders,
        "scaler": scaler,
        "target_scaler": target_scaler,
        "feature_cols": feature_cols,
        "input_dim": X_scaled.shape[1],
    }
    
    _data_cache = {
        "X": X_scaled,
        "y": y_scaled,
        "y_raw": df[TARGET_COLUMN].values,  # For income-based skew
        "city_tiers": city_tiers,
        "occupations": occupations,
        "incomes": incomes,
        "combined_keys": combined_keys,
        "unique_tiers": np.unique(city_tiers),
        "unique_keys": np.unique(combined_keys),
    }
    
    return _data_cache, _preprocessors


def get_input_dim():
    """Get the input dimension for the model."""
    _, preprocessors = _load_and_preprocess_data()
    return preprocessors["input_dim"]


def load_data(partition_id: int, num_partitions: int, batch_size: int):
    """Load partition of Personal Finance data with EXTREME non-IID partitioning.
    
    Non-IID Strategy:
    1. Primary split by Occupation + City_Tier + Income_Bracket (36 possible groups)
    2. Label skew: Some clients only see high/low disposable income samples
    3. Quantity skew: Uneven data distribution across clients
    """
    data_cache, preprocessors = _load_and_preprocess_data()
    
    X = data_cache["X"]
    y = data_cache["y"]
    combined_keys = data_cache["combined_keys"]
    unique_keys = data_cache["unique_keys"]
    incomes = data_cache["incomes"]
    
    # =========================================================================
    # EXTREME NON-IID: Multi-dimensional partitioning
    # =========================================================================
    
    np.random.seed(42 + partition_id)  # Reproducible but different per client
    
    # Strategy 1: Assign clients to specific combined keys (Occupation+City+Income)
    # Each client primarily gets data from 1-2 specific demographic groups
    num_keys = len(unique_keys)
    
    # Determine which keys this client primarily handles
    primary_key_idx = partition_id % num_keys
    secondary_key_idx = (partition_id + num_keys // 2) % num_keys
    
    primary_key = unique_keys[primary_key_idx]
    secondary_key = unique_keys[secondary_key_idx]
    
    # Get indices for primary key (70%) and secondary key (30%)
    primary_indices = np.where(combined_keys == primary_key)[0]
    secondary_indices = np.where(combined_keys == secondary_key)[0]
    
    # =========================================================================
    # Strategy 2: Income-based label skew
    # Odd partition_ids get high-income bias, even get low-income bias
    # =========================================================================
    income_percentile = np.percentile(incomes, [25, 75])
    
    if partition_id % 2 == 0:
        # Low income bias - prefer samples below 25th percentile
        income_mask_primary = incomes[primary_indices] < income_percentile[1]
        income_mask_secondary = incomes[secondary_indices] < income_percentile[1]
    else:
        # High income bias - prefer samples above 25th percentile  
        income_mask_primary = incomes[primary_indices] > income_percentile[0]
        income_mask_secondary = incomes[secondary_indices] > income_percentile[0]
    
    # Apply income filter (keep at least 50% of data even if filter is too strict)
    if income_mask_primary.sum() > len(primary_indices) * 0.3:
        primary_indices = primary_indices[income_mask_primary]
    if income_mask_secondary.sum() > len(secondary_indices) * 0.3:
        secondary_indices = secondary_indices[income_mask_secondary]
    
    # =========================================================================
    # Strategy 3: Quantity skew - uneven data distribution
    # =========================================================================
    # Some clients get more data, some get less
    quantity_factor = 0.5 + (partition_id % 5) * 0.2  # Ranges from 0.5 to 1.3
    
    # Sample from primary (70%) and secondary (30%) with quantity skew
    n_primary = min(len(primary_indices), int(800 * quantity_factor * 0.7))
    n_secondary = min(len(secondary_indices), int(800 * quantity_factor * 0.3))
    
    if len(primary_indices) > n_primary:
        np.random.shuffle(primary_indices)
        primary_indices = primary_indices[:n_primary]
    
    if len(secondary_indices) > n_secondary:
        np.random.shuffle(secondary_indices)
        secondary_indices = secondary_indices[:n_secondary]
    
    # Combine indices
    partition_indices = np.concatenate([primary_indices, secondary_indices])
    np.random.shuffle(partition_indices)
    
    # Ensure minimum data
    if len(partition_indices) < 50:
        # Fallback: add random samples
        all_indices = np.arange(len(X))
        remaining = np.setdiff1d(all_indices, partition_indices)
        extra = np.random.choice(remaining, min(100, len(remaining)), replace=False)
        partition_indices = np.concatenate([partition_indices, extra])
    
    # Get partition data
    X_partition = X[partition_indices]
    y_partition = y[partition_indices]
    
    # Split into train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_partition, y_partition, test_size=0.2, random_state=42
    )
    
    # Create PyTorch datasets
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return trainloader, testloader


def load_centralized_dataset():
    """Load entire test set for centralized evaluation."""
    data_cache, _ = _load_and_preprocess_data()
    
    X = data_cache["X"]
    y = data_cache["y"]
    
    # Use 20% of all data as test set
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32)
    )
    
    return DataLoader(test_dataset, batch_size=128)


def get_target_scaler():
    """Get the target scaler for inverse transforming predictions."""
    _, preprocessors = _load_and_preprocess_data()
    return preprocessors["target_scaler"]


def compute_model_divergence(local_params, global_params):
    """Compute L2 divergence between local and global model parameters.

    Returns:
        divergence: L2 norm of (local - global) parameters.
    """
    divergence = 0.0
    for local_p, global_p in zip(local_params, global_params):
        divergence += ((local_p - global_p) ** 2).sum().item()
    return divergence ** 0.5


def compute_adaptive_mu(
    base_mu: float,
    historical_divergence: float,
    global_avg_divergence: float,
    local_epochs: int,
    mu_min: float = 0.001,
    mu_max: float = 1.0,
) -> float:
    """Compute adaptive proximal coefficient based on client behavior.

    Adaptive strategy:
    - Higher historical divergence compared to global average → higher μ
    - More local epochs → higher μ (prevent excessive drift)
    - Clients that diverge more need tighter regularization

    Args:
        base_mu: Base proximal coefficient from config.
        historical_divergence: This client's historical divergence (from previous rounds).
        global_avg_divergence: Average divergence across all clients.
        local_epochs: Number of local training epochs.
        mu_min: Minimum allowed μ value.
        mu_max: Maximum allowed μ value.

    Returns:
        Adaptive μ value clamped between mu_min and mu_max.
    """
    # Factor 1: Divergence-based scaling
    # If client's historical divergence is higher than global average, increase μ
    if global_avg_divergence > 0 and historical_divergence > 0:
        # Scale μ based on how much this client diverges vs average
        divergence_ratio = historical_divergence / (global_avg_divergence + 1e-8)
        # Smooth the ratio to prevent extreme values
        divergence_factor = 1.0 + 0.5 * (divergence_ratio - 1.0)  # Dampened scaling
        divergence_factor = max(0.5, min(2.0, divergence_factor))  # Clamp to [0.5, 2.0]
    else:
        divergence_factor = 1.0

    # Factor 2: Local epochs scaling (more epochs = higher μ to prevent drift)
    epoch_factor = 1.0 + 0.1 * (local_epochs - 1)  # Scale up for >1 epoch

    # Combine factors
    adaptive_mu = base_mu * divergence_factor * epoch_factor

    # Clamp to valid range
    return max(mu_min, min(mu_max, adaptive_mu))


def train(net, trainloader, epochs, lr, device, proximal_mu=0.0, adaptive_mu_config=None):
    """Train the model on the training set using FedProx with optional adaptive μ.

    Args:
        net: The neural network model.
        trainloader: DataLoader for training data.
        epochs: Number of local epochs.
        lr: Learning rate.
        device: Device to train on (CPU/GPU).
        proximal_mu: Proximal term coefficient (0.0 = FedAvg, >0 = FedProx).
        adaptive_mu_config: Optional dict with adaptive μ settings:
            - enabled: bool, whether to use adaptive μ
            - historical_divergence: float, client's historical divergence
            - global_avg_divergence: float, average divergence across all clients
            - mu_min: float, minimum μ
            - mu_max: float, maximum μ

    Returns:
        dict with:
            - train_loss: Average training loss.
            - divergence: Post-training divergence from global model.
            - effective_mu: The μ value actually used (may be adaptive).
    """
    net.to(device)  # move model to GPU if available
    net.train()

    # Store global model parameters for proximal term (before training)
    global_params = [p.clone().detach().to(device) for p in net.parameters()]

    # Compute adaptive μ if enabled (using historical data, not pre-training divergence)
    effective_mu = proximal_mu
    if adaptive_mu_config and adaptive_mu_config.get("enabled", False):
        effective_mu = compute_adaptive_mu(
            base_mu=proximal_mu,
            historical_divergence=adaptive_mu_config.get("historical_divergence", 0.0),
            global_avg_divergence=adaptive_mu_config.get("global_avg_divergence", 0.0),
            local_epochs=epochs,
            mu_min=adaptive_mu_config.get("mu_min", 0.001),
            mu_max=adaptive_mu_config.get("mu_max", 1.0),
        )

    # Use MSELoss for regression task
    criterion = torch.nn.MSELoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    
    running_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for batch in trainloader:
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()

            # Forward pass
            predictions = net(features)
            
            # Standard MSE loss for regression
            loss = criterion(predictions, targets)

            # Add proximal term: (mu/2) * ||w - w^t||^2
            if effective_mu > 0.0:
                proximal_term = 0.0
                for local_param, global_param in zip(net.parameters(), global_params):
                    proximal_term += ((local_param - global_param) ** 2).sum()
                loss += (effective_mu / 2) * proximal_term

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

    avg_trainloss = running_loss / max(num_batches, 1)

    # Compute post-training divergence
    post_divergence = compute_model_divergence(list(net.parameters()), global_params)

    return {
        "train_loss": avg_trainloss,
        "divergence": post_divergence,
        "effective_mu": effective_mu,
    }


def test(net, testloader, device):
    """Validate the model on the test set (Regression)."""
    net.to(device)
    net.eval()
    criterion = torch.nn.MSELoss()
    
    total_loss = 0.0
    total_samples = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for batch in testloader:
            features, targets = batch
            features = features.to(device)
            targets = targets.to(device)
            
            predictions = net(features)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)
            
            all_predictions.append(predictions.cpu())
            all_targets.append(targets.cpu())
    
    # Calculate metrics
    avg_loss = total_loss / max(total_samples, 1)
    
    # Calculate R² score (coefficient of determination)
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    ss_res = ((all_targets - all_predictions) ** 2).sum().item()
    ss_tot = ((all_targets - all_targets.mean()) ** 2).sum().item()
    r2_score = 1 - (ss_res / max(ss_tot, 1e-8))
    
    # Calculate RMSE
    rmse = (total_loss / max(total_samples, 1)) ** 0.5
    
    return avg_loss, r2_score
