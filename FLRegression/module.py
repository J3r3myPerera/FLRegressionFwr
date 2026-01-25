"""
Shared configuration and constants for federated learning simulation.
"""

import torch
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from pytorchexample.task import (
    Net, get_input_dim, load_data, load_centralized_dataset, 
    train, test, _load_and_preprocess_data, reset_data_cache
)

# ============================================================================
# Configuration
# ============================================================================
NUM_ROUNDS = 20
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
