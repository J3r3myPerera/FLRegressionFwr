
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
    NUM_CLASSES,
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
    #CNN Model for CIFAR-10 Classification

    def __init__(self, num_classes: int = NUM_CLASSES):
        super(Net, self).__init__()
        # Conv Block 1: 3x32x32 -> 32x16x16
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)

        # Conv Block 2: 32x16x16 -> 64x8x8
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)

        # Conv Block 3: 64x8x8 -> 128x4x4
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.4)

        # Fully connected: 128*4*4 -> 128 -> num_classes
        self.fc1 = nn.Linear(128 * 4 * 4, 128)
        self.bn_fc = nn.BatchNorm1d(128)
        self.dropout_fc = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout1(self.pool1(x))

        # Block 2
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.dropout2(self.pool2(x))

        # Block 3
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout3(self.pool3(x))

        # Classifier
        x = x.view(x.size(0), -1)
        x = self.dropout_fc(F.relu(self.bn_fc(self.fc1(x))))
        return self.fc2(x)


def compute_model_divergence(local_params, global_params):
    #Compute L2 divergence between local and global model parameters.

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

    # Factor 2: Local epochs scaling
    epoch_factor = 1.0 + 0.1 * (local_epochs - 1)  # Scale up for >1 epoch

    # Combine factors
    adaptive_mu = base_mu * divergence_factor * epoch_factor

    # Clamp to valid range
    return max(mu_min, min(mu_max, adaptive_mu))


def train(net, trainloader, epochs, lr, device, proximal_mu=0.0, adaptive_mu_config=None):
    #Train the model on the training set using FedProx with optional adaptive μ.
    net.to(device)
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

    # Use CrossEntropyLoss for classification task
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)

    running_loss = 0.0
    num_batches = 0
    for _ in range(epochs):
        for batch in trainloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            # Forward pass
            outputs = net(images)

            # Standard CrossEntropy loss for classification
            loss = criterion(outputs, labels)

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
    """Validate the model on the test set (Classification)."""
    net.to(device)
    net.eval()
    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total_samples = 0

    with torch.no_grad():
        for batch in testloader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)

            outputs = net(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total_samples += images.size(0)

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = correct / max(total_samples, 1)

    return avg_loss, accuracy
