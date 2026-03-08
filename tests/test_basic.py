"""
Basic tests for the Flower-based federated learning implementation.
"""

import sys
from pathlib import Path

# Add FLRegression to path
sys.path.insert(0, str(Path(__file__).parent.parent / "FLRegression"))

import torch
import pytest
from dataset import (
    _get_data_path,
    get_input_dim,
    load_data,
    load_centralized_dataset,
    reset_data_cache,
)
from module import (
    Net, train, test, compute_model_divergence, compute_adaptive_mu,
    get_model_parameters, set_model_parameters,
)
from client import FlowerClient
from server import CustomFedStrategy
from module import NUM_CLIENTS, BATCH_SIZE, FRACTION_FIT, DEVICE, STRATEGIES


class TestDataLoading:
    """Test data loading and preprocessing."""

    def test_data_path_exists(self):
        """Test that data file path is valid."""
        path = _get_data_path()
        assert path.exists(), f"Data file not found at {path}"

    def test_get_input_dim(self):
        """Test getting input dimension."""
        reset_data_cache()
        dim = get_input_dim()
        assert dim > 0, "Input dimension should be positive"
        assert isinstance(dim, int), "Input dimension should be integer"

    def test_load_data(self):
        """Test loading client data partition."""
        reset_data_cache()
        trainloader, testloader = load_data(0, NUM_CLIENTS, BATCH_SIZE)
        assert trainloader is not None, "Trainloader should not be None"
        assert testloader is not None, "Testloader should not be None"

        # Check that we can iterate through data
        batch = next(iter(trainloader))
        assert len(batch) == 2, "Batch should contain features and targets"
        assert batch[0].shape[0] > 0, "Batch should have samples"

    def test_load_centralized_dataset(self):
        """Test loading centralized test dataset."""
        reset_data_cache()
        testloader = load_centralized_dataset()
        assert testloader is not None, "Testloader should not be None"

        batch = next(iter(testloader))
        assert len(batch) == 2, "Batch should contain features and targets"


class TestModel:
    """Test model creation and operations."""

    def test_model_creation(self):
        """Test creating a model instance."""
        reset_data_cache()
        input_dim = get_input_dim()
        model = Net(input_dim=input_dim)

        assert model is not None, "Model should be created"
        assert sum(p.numel() for p in model.parameters()) > 0, "Model should have parameters"

    def test_model_forward(self):
        """Test model forward pass."""
        reset_data_cache()
        input_dim = get_input_dim()
        model = Net(input_dim=input_dim)
        model.eval()

        # Create dummy input
        batch_size = 10
        dummy_input = torch.randn(batch_size, input_dim)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (batch_size, 1), f"Output shape should be ({batch_size}, 1), got {output.shape}"

    def test_compute_model_divergence(self):
        """Test divergence computation."""
        reset_data_cache()
        input_dim = get_input_dim()

        model1 = Net(input_dim=input_dim)
        model2 = Net(input_dim=input_dim)

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

    def test_get_set_model_parameters(self):
        """Test parameter conversion helpers."""
        reset_data_cache()
        input_dim = get_input_dim()
        model = Net(input_dim=input_dim)

        # Get parameters as numpy arrays
        params = get_model_parameters(model)
        assert len(params) > 0, "Should have parameters"
        assert all(hasattr(p, 'shape') for p in params), "Parameters should be numpy arrays"

        # Set parameters back
        model2 = Net(input_dim=input_dim)
        set_model_parameters(model2, params)

        # Verify parameters match
        params2 = get_model_parameters(model2)
        for p1, p2 in zip(params, params2):
            assert (p1 == p2).all(), "Parameters should match after set"


class TestClient:
    """Test Flower client functionality."""

    def test_client_initialization(self):
        """Test FlowerClient initialization."""
        reset_data_cache()
        trainloader, testloader = load_data(0, NUM_CLIENTS, BATCH_SIZE)
        client = FlowerClient(0, trainloader, testloader)

        assert client.client_id == 0, "Client ID should be 0"
        assert client.num_examples > 0, "Client should have data"

    def test_client_fit(self):
        """Test client training via Flower's fit interface."""
        reset_data_cache()
        trainloader, testloader = load_data(0, NUM_CLIENTS, BATCH_SIZE)
        client = FlowerClient(0, trainloader, testloader)

        input_dim = get_input_dim()
        model = Net(input_dim=input_dim)
        parameters = get_model_parameters(model)

        config = {
            "local_epochs": 1,
            "lr": 0.001,
            "proximal_mu": 0.1,
            "adaptive_mu_enabled": False,
            "global_avg_divergence": 0.0,
            "historical_divergence": 0.0,
        }

        updated_params, num_examples, metrics = client.fit(parameters, config)

        assert len(updated_params) > 0, "Should return updated parameters"
        assert num_examples > 0, "Should return num_examples"
        assert "train_loss" in metrics, "Result should contain train_loss"
        assert "divergence" in metrics, "Result should contain divergence"
        assert "effective_mu" in metrics, "Result should contain effective_mu"
        assert "client_id" in metrics, "Result should contain client_id"
        assert metrics["train_loss"] >= 0, "Training loss should be non-negative"

    def test_client_evaluate(self):
        """Test client evaluation via Flower's evaluate interface."""
        reset_data_cache()
        trainloader, testloader = load_data(0, NUM_CLIENTS, BATCH_SIZE)
        client = FlowerClient(0, trainloader, testloader)

        input_dim = get_input_dim()
        model = Net(input_dim=input_dim)
        parameters = get_model_parameters(model)

        loss, num_examples, metrics = client.evaluate(parameters, {})

        assert loss >= 0, "Loss should be non-negative"
        assert num_examples > 0, "Should return num_examples"
        assert "r2" in metrics, "Result should contain r2"


class TestServer:
    """Test Flower server strategy functionality."""

    def _create_strategy(self, strategy_name="FedAvg"):
        """Helper to create a CustomFedStrategy."""
        config = STRATEGIES[strategy_name]
        num_to_select = max(2, int(NUM_CLIENTS * FRACTION_FIT))
        return CustomFedStrategy(
            strategy_name=strategy_name,
            strategy_config=config,
            num_clients=NUM_CLIENTS,
            fraction_fit_val=FRACTION_FIT,
            fraction_fit=float(num_to_select) / NUM_CLIENTS,
            fraction_evaluate=0.0,
            min_fit_clients=num_to_select,
            min_available_clients=NUM_CLIENTS,
        )

    def test_strategy_initialization(self):
        """Test strategy initialization."""
        strategy = self._create_strategy("FedAvg")

        assert strategy.strategy_name == "FedAvg", "Strategy name should match"
        assert strategy._num_clients == NUM_CLIENTS, f"Should have {NUM_CLIENTS} clients"
        assert strategy.global_avg_divergence == 0.0, "Initial divergence should be 0"

    def test_hybrid_selection_cold_start(self):
        """Test hybrid client selection during cold start (random)."""
        strategy = self._create_strategy("SmartFedProx")

        # Cold start (round <= 3) should return random selection
        selected = strategy._hybrid_select(round_num=1)
        num_to_select = max(2, int(NUM_CLIENTS * FRACTION_FIT))
        assert len(selected) == num_to_select, f"Should select {num_to_select} clients"
        assert all(0 <= c < NUM_CLIENTS for c in selected), "All selected clients should be valid"

    def test_hybrid_selection_with_history(self):
        """Test hybrid client selection with divergence history."""
        strategy = self._create_strategy("SmartFedProx")

        # Populate divergence history for all clients
        for i in range(NUM_CLIENTS):
            strategy.client_stats[i]["divergences"] = [float(i) * 0.1, float(i) * 0.15]

        # Round > 3 should use divergence-based selection
        selected = strategy._hybrid_select(round_num=5)
        num_to_select = max(2, int(NUM_CLIENTS * FRACTION_FIT))
        assert len(selected) <= num_to_select, f"Should select at most {num_to_select} clients"
        assert all(0 <= c < NUM_CLIENTS for c in selected), "All selected clients should be valid"

    def test_strategy_metrics_structure(self):
        """Test that strategy metrics dict has correct structure."""
        strategy = self._create_strategy("FedAvg")

        expected_keys = {"rounds", "r2_scores", "mse_losses", "avg_train_loss",
                         "avg_divergence", "avg_effective_mu"}
        assert set(strategy.metrics.keys()) == expected_keys, "Metrics should have all expected keys"
        for key in expected_keys:
            assert isinstance(strategy.metrics[key], list), f"{key} should be a list"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
