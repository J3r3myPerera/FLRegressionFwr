"""
Dataset preparation and loading for Indian Personal Finance data.
Handles data loading, preprocessing, and extreme non-IID partitioning.
"""

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

# ============================================================================
# Data Configuration
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

# ============================================================================
# Data Loading and Preprocessing
# ============================================================================
def reset_data_cache():
    """Reset the data cache to force reloading."""
    global _data_cache, _preprocessors
    _data_cache = None
    _preprocessors = None


def _get_data_path():
    """Get the path to the CSV data file."""
    # CSV file is located in the data directory at the workspace root
    return Path("/Users/dinukaperera/FLRegressionFlwr/data/indianPersonalFinanceAndSpendingHabits.csv")


def _load_and_preprocess_data():
    """Load and preprocess the entire dataset once."""
    global _data_cache, _preprocessors
    
    if _data_cache is not None:
        return _data_cache, _preprocessors
        
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
