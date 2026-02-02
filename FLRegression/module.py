
#Shared configuration and constants for federated learning simulation.
#Contains model definition and training code.


import torch
import torch.nn as nn
import torch.nn.functional as F

# Import dataset functions
from dataset import (
    get_input_dim,
    load_data,
    load_centralized_dataset,
    _load_and_preprocess_data,
    reset_data_cache,
)

# Simulation Configuration
NUM_ROUNDS = 17
NUM_CLIENTS = 10
FRACTION_FIT = 0.5
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Strategy configurations
STRATEGIES = {
    "FedAvg": {
        "proximal_mu": 0.0,
        "adaptive_mu_enabled": False,
        "selection_strategy": "random",
        "description": "Baseline FedAvg (μ=0)"
    },
    "FedProx": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": False,
        "selection_strategy": "random",
        "description": "FedProx (μ=0.1, random selection)"
    },
    "SmartFedProx": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "hybrid",
        "description": "SmartFedProx (adaptive μ, hybrid selection)"
    }
}

# Model Definition
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

# Training and Evaluation Functions
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
