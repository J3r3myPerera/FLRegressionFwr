# Flower Server Strategy and Simulation Runner

import logging
import numpy as np
import flwr as fl
from collections import defaultdict
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays, FitIns
from module import (
    Net, get_input_dim, load_centralized_dataset, load_data, test,
    get_model_parameters, set_model_parameters,
    NUM_CLIENTS, BATCH_SIZE, FRACTION_FIT, LOCAL_EPOCHS, LEARNING_RATE, DEVICE,
    NUM_ROUNDS,
)
from client import FlowerClient


class CustomFedStrategy(fl.server.strategy.FedAvg):
    """Custom Flower strategy supporting FedAvg, FedProx, and SmartFedProx.

    Preserves all original logic:
    - Weighted FedAvg aggregation (by num_examples)
    - Hybrid client selection (divergence-based with cold start and exploration)
    - Adaptive proximal coefficient (mu) tracking
    - Centralized evaluation on test set
    """

    def __init__(self, strategy_name, strategy_config, num_clients, fraction_fit_val, **kwargs):
        super().__init__(**kwargs)
        self.strategy_name = strategy_name
        self.strategy_config = strategy_config
        self._num_clients = num_clients
        self._fraction_fit_val = fraction_fit_val
        self.client_stats = defaultdict(lambda: {
            "divergences": [],
            "participation": 0,
            "historical_divergence": 0.0,
        })
        self.global_avg_divergence = 0.0
        self.metrics = {
            "rounds": [],
            "r2_scores": [],
            "mse_losses": [],
            "avg_train_loss": [],
            "avg_divergence": [],
            "avg_effective_mu": [],
        }

    def configure_fit(self, server_round, parameters, client_manager):
        """Select clients and configure per-client training parameters."""
        sample_size = max(2, int(self._num_clients * self._fraction_fit_val))

        # Client selection based on strategy
        if self.strategy_config["selection_strategy"] == "hybrid":
            selected_ids = self._hybrid_select(server_round)
            # Get all available clients and pick specific ones
            all_clients = client_manager.sample(
                num_clients=client_manager.num_available(),
                min_num_clients=sample_size,
            )
            client_map = {int(c.cid): c for c in all_clients}
            selected_clients = [client_map[cid] for cid in selected_ids if cid in client_map]
            print(f"    Selected clients (hybrid): {selected_ids}")
        else:
            selected_clients = client_manager.sample(
                num_clients=sample_size,
                min_num_clients=sample_size,
            )
            print(f"    Selected clients (random): {[int(c.cid) for c in selected_clients]}")

        # Create per-client config (each client gets its own historical_divergence)
        fit_configs = []
        for client in selected_clients:
            cid = int(client.cid)
            config = {
                "proximal_mu": float(self.strategy_config["proximal_mu"]),
                "adaptive_mu_enabled": bool(self.strategy_config["adaptive_mu_enabled"]),
                "local_epochs": int(LOCAL_EPOCHS),
                "lr": float(LEARNING_RATE),
                "global_avg_divergence": float(self.global_avg_divergence),
                "historical_divergence": float(
                    self.client_stats[cid]["historical_divergence"]
                ),
            }
            fit_configs.append((client, FitIns(parameters, config)))

        return fit_configs

    def aggregate_fit(self, server_round, results, failures):
        """Aggregate client results and track custom metrics."""
        if not results:
            return None, {}

        # Extract custom metrics from client results
        train_losses = []
        divergences = []
        effective_mus = []

        for _, fit_res in results:
            metrics = fit_res.metrics
            client_id = int(metrics.get("client_id", -1))

            train_losses.append(metrics.get("train_loss", 0.0))
            divergences.append(metrics.get("divergence", 0.0))
            effective_mus.append(metrics.get("effective_mu", 0.0))

            # Track client stats for hybrid selection and adaptive mu
            if client_id >= 0:
                divergence = metrics.get("divergence", 0.0)
                self.client_stats[client_id]["divergences"].append(divergence)
                self.client_stats[client_id]["participation"] += 1

                # Update historical divergence with EMA (same alpha=0.3 as original)
                old_div = self.client_stats[client_id]["historical_divergence"]
                alpha = 0.3
                self.client_stats[client_id]["historical_divergence"] = (
                    alpha * divergence + (1 - alpha) * old_div
                )

        # Update global average divergence (EMA, same as original)
        avg_divergence = float(np.mean(divergences)) if divergences else 0.0
        if self.global_avg_divergence == 0:
            self.global_avg_divergence = avg_divergence
        else:
            self.global_avg_divergence = 0.7 * self.global_avg_divergence + 0.3 * avg_divergence

        # Store round metrics
        self.metrics["rounds"].append(server_round)
        self.metrics["avg_train_loss"].append(
            float(np.mean(train_losses)) if train_losses else 0.0
        )
        self.metrics["avg_divergence"].append(avg_divergence)
        self.metrics["avg_effective_mu"].append(
            float(np.mean(effective_mus)) if effective_mus else 0.0
        )

        # Standard weighted aggregation via parent (FedAvg weighted by num_examples)
        return super().aggregate_fit(server_round, results, failures)

    def evaluate(self, server_round, parameters):
        """Centralized evaluation on test set after each round."""
        param_ndarrays = parameters_to_ndarrays(parameters)
        model = Net(input_dim=get_input_dim())
        set_model_parameters(model, param_ndarrays)

        test_loader = load_centralized_dataset()
        loss, r2 = test(model, test_loader, DEVICE)

        # Only store metrics for rounds after training (skip initial round 0 eval)
        if server_round > 0:
            self.metrics["r2_scores"].append(float(r2))
            self.metrics["mse_losses"].append(float(loss))
            avg_mu = self.metrics["avg_effective_mu"][-1] if self.metrics["avg_effective_mu"] else 0.0
            print(f"    R² = {r2:.4f}, MSE = {loss:.4f}, Avg μ = {avg_mu:.4f}")

        return float(loss), {"r2_score": float(r2)}

    def _hybrid_select(self, round_num):
        """Hybrid client selection strategy (preserved from original implementation).

        - Cold start (rounds 1-3): Random selection
        - 15% exploration rate: Random selection
        - Otherwise: 30% high-divergence + 50% middle + 20% low-divergence
        """
        num_to_select = max(2, int(self._num_clients * self._fraction_fit_val))

        # Cold start: random for first 3 rounds to build better history
        if round_num <= 3:
            return list(np.random.choice(self._num_clients, num_to_select, replace=False))

        # Exploration: 15% chance of random selection to prevent lock-in
        if np.random.random() < 0.15:
            return list(np.random.choice(self._num_clients, num_to_select, replace=False))

        # Get clients with divergence history using smoothed average
        clients_with_history = []
        for i in range(self._num_clients):
            divs = self.client_stats[i]["divergences"]
            if len(divs) >= 2:
                if len(divs) > 2:
                    avg_div = 0.5 * divs[-1] + 0.3 * divs[-2] + 0.2 * np.mean(divs[:-2])
                else:
                    avg_div = 0.6 * divs[-1] + 0.4 * divs[-2]
            elif len(divs) == 1:
                avg_div = divs[-1]
            else:
                avg_div = 0
            clients_with_history.append((i, avg_div))

        # Sort by divergence
        sorted_clients = sorted(clients_with_history, key=lambda x: x[1], reverse=True)

        # Balanced selection: 30% high, 50% middle, 20% low
        num_high = max(1, num_to_select * 3 // 10)
        num_low = max(1, num_to_select * 2 // 10)
        num_mid = num_to_select - num_high - num_low

        high_div = [c[0] for c in sorted_clients[:num_high]]
        low_div = [c[0] for c in sorted_clients[-num_low:] if c[0] not in high_div]

        # Middle clients for stability
        mid_start = num_high
        mid_end = len(sorted_clients) - num_low
        mid_candidates = [c[0] for c in sorted_clients[mid_start:mid_end]
                          if c[0] not in high_div + low_div]
        if len(mid_candidates) > num_mid:
            mid_div = list(np.random.choice(mid_candidates, num_mid, replace=False))
        else:
            mid_div = mid_candidates

        selected = high_div + mid_div + low_div

        # Fill remaining slots randomly if needed
        while len(selected) < num_to_select:
            remaining = [i for i in range(self._num_clients) if i not in selected]
            if remaining:
                selected.append(np.random.choice(remaining))
            else:
                break

        return selected[:num_to_select]


def run_flower_simulation(strategy_name, config, num_rounds=None):
    """Run a Flower-based federated learning simulation.

    Returns a metrics dict with the same structure as the original implementation:
    {rounds, r2_scores, mse_losses, avg_train_loss, avg_divergence, avg_effective_mu}
    """
    if num_rounds is None:
        num_rounds = NUM_ROUNDS

    # Suppress Flower's verbose logging
    logging.getLogger("flwr").setLevel(logging.WARNING)

    # Client state store (data loaders cached via closure, avoids reloading each round)
    client_data_cache = {}

    def client_fn(cid):
        """Create a FlowerClient for the given client ID."""
        client_id = int(cid)
        if client_id not in client_data_cache:
            trainloader, testloader = load_data(client_id, NUM_CLIENTS, BATCH_SIZE)
            client_data_cache[client_id] = (trainloader, testloader)
        trainloader, testloader = client_data_cache[client_id]
        return FlowerClient(client_id, trainloader, testloader)

    # Initialize model for initial parameters
    initial_model = Net(input_dim=get_input_dim())
    initial_params = ndarrays_to_parameters(get_model_parameters(initial_model))

    num_to_select = max(2, int(NUM_CLIENTS * FRACTION_FIT))

    # Create strategy
    strategy = CustomFedStrategy(
        strategy_name=strategy_name,
        strategy_config=config,
        num_clients=NUM_CLIENTS,
        fraction_fit_val=FRACTION_FIT,
        # FedAvg parent parameters
        fraction_fit=float(num_to_select) / NUM_CLIENTS,
        fraction_evaluate=0.0,  # Disable distributed eval (we use centralized)
        min_fit_clients=num_to_select,
        min_evaluate_clients=0,
        min_available_clients=NUM_CLIENTS,
        initial_parameters=initial_params,
        fit_metrics_aggregation_fn=lambda metrics: {},
    )

    print(f"\n{'='*60}")
    print(f"Running: {strategy_name}")
    print(f"Config: {config['description']}")
    print(f"{'='*60}")

    # Run Flower simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        ray_init_args={"include_dashboard": False, "log_to_driver": False},
    )

    print(f"\n  Final R²: {strategy.metrics['r2_scores'][-1]:.4f}")
    return strategy.metrics
