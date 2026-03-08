"""
Basic tests for the federated learning implementation (CIFAR-10).
"""

import sys
from pathlib import Path

# Add FLRegression to path
sys.path.insert(0, str(Path(__file__).parent.parent / "FLRegression"))

import torch
import pytest
from dataset import (
    _get_data_dir,
    get_input_dim,
    load_data,
    load_centralized_dataset,
    reset_data_cache,
    NUM_CLASSES,
)
from module import Net, train, test, compute_model_divergence, compute_adaptive_mu
from client import SimulatedClient
from server import FederatedSimulator
from module import NUM_CLIENTS, BATCH_SIZE, DEVICE, STRATEGIES


class TestDataLoading:
    """Test data loading and preprocessing."""

    def test_get_input_dim(self):
        """Test getting input dimension (num classes)."""
        reset_data_cache()
        dim = get_input_dim()
        assert dim == NUM_CLASSES, f"Input dim should be {NUM_CLASSES}, got {dim}"

    def test_load_data(self):
        """Test loading client data partition."""
        reset_data_cache()
        trainloader, testloader = load_data(0, NUM_CLIENTS, BATCH_SIZE)
        assert trainloader is not None, "Trainloader should not be None"
        assert testloader is not None, "Testloader should not be None"

        # Check that we can iterate through data
        batch = next(iter(trainloader))
        assert len(batch) == 2, "Batch should contain images and labels"
        images, labels = batch
        assert images.shape[1] == 3, "Images should have 3 channels"
        assert images.shape[2] == 32, "Images should be 32x32"
        assert images.shape[3] == 32, "Images should be 32x32"

    def test_load_centralized_dataset(self):
        """Test loading centralized test dataset."""
        reset_data_cache()
        testloader = load_centralized_dataset()
        assert testloader is not None, "Testloader should not be None"

        batch = next(iter(testloader))
        assert len(batch) == 2, "Batch should contain images and labels"


class TestModel:
    """Test model creation and operations."""

    def test_model_creation(self):
        """Test creating a model instance."""
        model = Net()

        assert model is not None, "Model should be created"
        assert sum(p.numel() for p in model.parameters()) > 0, "Model should have parameters"

    def test_model_forward(self):
        """Test model forward pass."""
        model = Net()
        model.eval()

        # Create dummy CIFAR-10 input (batch_size, 3, 32, 32)
        batch_size = 10
        dummy_input = torch.randn(batch_size, 3, 32, 32)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, NUM_CLASSES), f"Output shape should be ({batch_size}, {NUM_CLASSES}), got {output.shape}"

    def test_compute_model_divergence(self):
        """Test divergence computation."""
        model1 = Net()
        model2 = Net()

        # Initialize with different weights
        for param in model2.parameters():
            param.data += 0.1

        params1 = list(model1.parameters())
        params2 = list(model2.parameters())

        divergence = compute_model_divergence(params1, params2)
        assert divergence > 0, "Divergence should be positive for different models"
        assert isinstance(divergence, float), "Divergence should be a float"

    def test_compute_adaptive_mu(self):
        """Test adaptive μ computation."""
        base_mu = 0.1
        historical_divergence = 0.5
        global_avg_divergence = 0.3
        local_epochs = 3

        adaptive_mu = compute_adaptive_mu(
            base_mu=base_mu,
            historical_divergence=historical_divergence,
            global_avg_divergence=global_avg_divergence,
            local_epochs=local_epochs,
        )

        assert adaptive_mu > 0, "Adaptive μ should be positive"
        assert adaptive_mu >= 0.001, "Adaptive μ should be at least mu_min"
        assert adaptive_mu <= 1.0, "Adaptive μ should be at most mu_max"


class TestClient:
    """Test client functionality."""

    def test_client_initialization(self):
        """Test client initialization."""
        client = SimulatedClient(0, NUM_CLIENTS, BATCH_SIZE)

        assert client.client_id == 0, "Client ID should be 0"
        assert client.num_examples > 0, "Client should have data"
        assert client.historical_divergence == 0.0, "Initial divergence should be 0"

    def test_client_train(self):
        """Test client training."""
        reset_data_cache()
        client = SimulatedClient(0, NUM_CLIENTS, BATCH_SIZE)

        model = Net()
        model_state_dict = model.state_dict()

        config = {
            "local_epochs": 1,  # Use 1 epoch for quick test
            "lr": 0.001,
            "proximal_mu": 0.1,
            "adaptive_mu_enabled": False,
        }

        result = client.train(model_state_dict, config, global_avg_divergence=0.0)

        assert "state_dict" in result, "Result should contain state_dict"
        assert "num_examples" in result, "Result should contain num_examples"
        assert "train_loss" in result, "Result should contain train_loss"
        assert "divergence" in result, "Result should contain divergence"
        assert result["train_loss"] >= 0, "Training loss should be non-negative"

    def test_client_evaluate(self):
        """Test client evaluation."""
        reset_data_cache()
        client = SimulatedClient(0, NUM_CLIENTS, BATCH_SIZE)

        model = Net()
        model_state_dict = model.state_dict()

        result = client.evaluate(model_state_dict)

        assert "loss" in result, "Result should contain loss"
        assert "accuracy" in result, "Result should contain accuracy"
        assert "num_examples" in result, "Result should contain num_examples"
        assert result["loss"] >= 0, "Loss should be non-negative"
        assert 0.0 <= result["accuracy"] <= 1.0, "Accuracy should be between 0 and 1"


class TestServer:
    """Test server/simulator functionality."""

    def test_simulator_initialization(self):
        """Test simulator initialization."""
        config = STRATEGIES["FedAvg"]
        simulator = FederatedSimulator("FedAvg", config)

        assert simulator.strategy_name == "FedAvg", "Strategy name should match"
        assert len(simulator.clients) == NUM_CLIENTS, f"Should have {NUM_CLIENTS} clients"

    def test_client_selection_random(self):
        """Test random client selection."""
        config = STRATEGIES["FedAvg"]
        simulator = FederatedSimulator("FedAvg", config)

        selected = simulator.select_clients(round_num=1)
        assert len(selected) > 0, "Should select at least one client"
        assert all(0 <= c < NUM_CLIENTS for c in selected), "All selected clients should be valid"

    def test_aggregation(self):
        """Test model aggregation."""
        reset_data_cache()

        config = STRATEGIES["FedAvg"]
        simulator = FederatedSimulator("FedAvg", config)

        model = Net()

        # Create dummy client results
        client_results = []
        for i in range(3):
            client = simulator.clients[i]
            result = client.train(model.state_dict(), {
                "local_epochs": 1,
                "lr": 0.001,
                "proximal_mu": 0.0,
                "adaptive_mu_enabled": False,
            })
            client_results.append(result)

        aggregated = simulator.aggregate(client_results)

        assert aggregated is not None, "Aggregated model should not be None"
        assert len(aggregated) > 0, "Aggregated state should have parameters"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
