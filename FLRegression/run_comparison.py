"""
Baseline Comparison: FedAvg vs FedProx vs SmartFedProx

This script runs three federated learning simulations and compares their performance
on the Personal Finance dataset (predicting Disposable_Income).

Strategies compared:
1. FedAvg: No proximal term (μ=0), random client selection
2. FedProx: Proximal term (μ=0.1), random client selection  
3. SmartFedProx: Proximal term (μ=0.1), hybrid client selection with adaptive μ
"""

import os
import sys
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

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


class SimulatedClient:
    """Simulates a federated learning client."""
    
    def __init__(self, client_id: int, num_clients: int, batch_size: int):
        self.client_id = client_id
        self.trainloader, self.testloader = load_data(client_id, num_clients, batch_size)
        self.historical_divergence = 0.0
        self.num_examples = len(self.trainloader.dataset)
    
    def train(self, model_state_dict, config, global_avg_divergence: float = 0.0):
        """Train local model and return updated weights + metrics."""
        input_dim = get_input_dim()
        model = Net(input_dim=input_dim)
        model.load_state_dict(model_state_dict)
        
        adaptive_mu_config = None
        if config.get("adaptive_mu_enabled", False):
            adaptive_mu_config = {
                "enabled": True,
                "historical_divergence": self.historical_divergence,
                "global_avg_divergence": global_avg_divergence,
                "mu_min": 0.001,
                "mu_max": 1.0,
            }
        
        result = train(
            model,
            self.trainloader,
            epochs=config["local_epochs"],
            lr=config["lr"],
            device=DEVICE,
            proximal_mu=config["proximal_mu"],
            adaptive_mu_config=adaptive_mu_config,
        )
        
        # Update historical divergence with EMA
        alpha = 0.3
        self.historical_divergence = alpha * result["divergence"] + (1 - alpha) * self.historical_divergence
        
        return {
            "state_dict": model.state_dict(),
            "num_examples": self.num_examples,
            "train_loss": result["train_loss"],
            "divergence": result["divergence"],
            "effective_mu": result["effective_mu"],
        }
    
    def evaluate(self, model_state_dict):
        """Evaluate model on local test data."""
        input_dim = get_input_dim()
        model = Net(input_dim=input_dim)
        model.load_state_dict(model_state_dict)
        
        loss, r2 = test(model, self.testloader, DEVICE)
        return {"loss": loss, "r2": r2, "num_examples": len(self.testloader.dataset)}


class FederatedSimulator:
    """Simulates federated learning with different strategies."""
    
    def __init__(self, strategy_name: str, config: dict):
        self.strategy_name = strategy_name
        self.config = config
        self.clients = []
        self.client_stats = defaultdict(lambda: {"divergences": [], "participation": 0})
        
        # Initialize clients
        print(f"\n  Initializing {NUM_CLIENTS} clients...")
        for i in range(NUM_CLIENTS):
            self.clients.append(SimulatedClient(i, NUM_CLIENTS, BATCH_SIZE))
    
    def select_clients(self, round_num: int) -> list:
        """Select clients based on strategy."""
        num_to_select = max(2, int(NUM_CLIENTS * FRACTION_FIT))
        
        if self.config["selection_strategy"] == "random":
            return list(np.random.choice(NUM_CLIENTS, num_to_select, replace=False))
        
        elif self.config["selection_strategy"] == "hybrid":
            # Cold start: random for first 3 rounds to build better history
            if round_num <= 3:
                return list(np.random.choice(NUM_CLIENTS, num_to_select, replace=False))
            
            # Exploration: 15% chance of random selection to prevent lock-in
            if np.random.random() < 0.15:
                return list(np.random.choice(NUM_CLIENTS, num_to_select, replace=False))
            
            # Get clients with divergence history (use smoothed average)
            clients_with_history = []
            for i in range(NUM_CLIENTS):
                divs = self.client_stats[i]["divergences"]
                if len(divs) >= 2:
                    # Weighted average favoring recent but not too much
                    avg_div = 0.5 * divs[-1] + 0.3 * divs[-2] + 0.2 * np.mean(divs[:-2]) if len(divs) > 2 else 0.6 * divs[-1] + 0.4 * divs[-2]
                elif len(divs) == 1:
                    avg_div = divs[-1]
                else:
                    avg_div = 0
                clients_with_history.append((i, avg_div))
            
            # Sort by divergence
            sorted_clients = sorted(clients_with_history, key=lambda x: x[1], reverse=True)
            
            # Balanced selection: prioritize middle-divergence clients for stability
            # with some high and low for adaptation
            # 30% high, 50% middle, 20% low
            num_high = max(1, num_to_select * 3 // 10)
            num_low = max(1, num_to_select * 2 // 10)
            num_mid = num_to_select - num_high - num_low
            
            high_div = [c[0] for c in sorted_clients[:num_high]]
            low_div = [c[0] for c in sorted_clients[-num_low:] if c[0] not in high_div]
            
            # Middle clients for stability
            mid_start = num_high
            mid_end = len(sorted_clients) - num_low
            mid_candidates = [c[0] for c in sorted_clients[mid_start:mid_end] if c[0] not in high_div + low_div]
            if len(mid_candidates) > num_mid:
                mid_div = list(np.random.choice(mid_candidates, num_mid, replace=False))
            else:
                mid_div = mid_candidates
            
            selected = high_div + mid_div + low_div
            
            # Fill remaining slots randomly if needed
            while len(selected) < num_to_select:
                remaining = [i for i in range(NUM_CLIENTS) if i not in selected]
                if remaining:
                    selected.append(np.random.choice(remaining))
                else:
                    break
            
            return selected[:num_to_select]
        
        return list(range(num_to_select))
    
    def aggregate(self, client_results: list) -> dict:
        """FedAvg aggregation."""
        total_examples = sum(r["num_examples"] for r in client_results)
        
        # Weighted average of model parameters
        aggregated_state = {}
        for key in client_results[0]["state_dict"].keys():
            weighted_sum = torch.zeros_like(client_results[0]["state_dict"][key], dtype=torch.float32)
            for result in client_results:
                weight = result["num_examples"] / total_examples
                weighted_sum += result["state_dict"][key].float() * weight
            aggregated_state[key] = weighted_sum
        
        return aggregated_state
    
    def evaluate_global(self, model_state_dict) -> tuple:
        """Evaluate global model on centralized test set."""
        input_dim = get_input_dim()
        model = Net(input_dim=input_dim)
        model.load_state_dict(model_state_dict)
        
        test_dataloader = load_centralized_dataset()
        loss, r2 = test(model, test_dataloader, DEVICE)
        return loss, r2
    
    def run(self, num_rounds: int) -> dict:
        """Run federated learning simulation."""
        print(f"\n{'='*60}")
        print(f"Running: {self.strategy_name}")
        print(f"Config: {self.config['description']}")
        print(f"{'='*60}")
        
        # Initialize global model
        input_dim = get_input_dim()
        global_model = Net(input_dim=input_dim)
        global_state = global_model.state_dict()
        
        # Metrics storage
        metrics = {
            "rounds": [],
            "r2_scores": [],
            "mse_losses": [],
            "avg_train_loss": [],
            "avg_divergence": [],
            "avg_effective_mu": [],
        }
        
        # Training config
        train_config = {
            "local_epochs": LOCAL_EPOCHS,
            "lr": LEARNING_RATE,
            "proximal_mu": self.config["proximal_mu"],
            "adaptive_mu_enabled": self.config["adaptive_mu_enabled"],
        }
        
        # Track global average divergence across rounds
        global_avg_divergence = 0.0
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n  Round {round_num}/{num_rounds}")
            
            # Select clients
            selected_ids = self.select_clients(round_num)
            print(f"    Selected clients: {selected_ids}")
            
            # Train on selected clients
            client_results = []
            for client_id in selected_ids:
                result = self.clients[client_id].train(global_state, train_config, global_avg_divergence)
                client_results.append(result)
                
                # Track stats
                self.client_stats[client_id]["divergences"].append(result["divergence"])
                self.client_stats[client_id]["participation"] += 1
            
            # Aggregate
            global_state = self.aggregate(client_results)
            
            # Evaluate global model
            loss, r2 = self.evaluate_global(global_state)
            
            # Compute round metrics
            avg_train_loss = np.mean([r["train_loss"] for r in client_results])
            avg_divergence = np.mean([r["divergence"] for r in client_results])
            avg_mu = np.mean([r["effective_mu"] for r in client_results])
            
            # Update global average divergence for next round (EMA)
            if global_avg_divergence == 0:
                global_avg_divergence = avg_divergence
            else:
                global_avg_divergence = 0.7 * global_avg_divergence + 0.3 * avg_divergence
            
            # Store metrics
            metrics["rounds"].append(round_num)
            metrics["r2_scores"].append(r2)
            metrics["mse_losses"].append(loss)
            metrics["avg_train_loss"].append(avg_train_loss)
            metrics["avg_divergence"].append(avg_divergence)
            metrics["avg_effective_mu"].append(avg_mu)
            
            print(f"    R² = {r2:.4f}, MSE = {loss:.4f}, Avg μ = {avg_mu:.4f}")
        
        print(f"\n  Final R²: {metrics['r2_scores'][-1]:.4f}")
        return metrics


def plot_comparison(all_results: dict, save_path: str = "comparison_results.png"):
    """Create comprehensive comparison plots."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Federated Learning Strategy Comparison\n(Personal Finance - Disposable Income Prediction)", 
                 fontsize=14, fontweight='bold')
    
    colors = {"FedAvg": "#e74c3c", "FedProx": "#3498db", "SmartFedProx": "#2ecc71"}
    markers = {"FedAvg": "o", "FedProx": "s", "SmartFedProx": "^"}
    
    # Plot 1: R² Score
    ax = axes[0, 0]
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["r2_scores"], 
                color=colors[name], marker=markers[name], 
                linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_title("R² Score Progression", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: MSE Loss
    ax = axes[0, 1]
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["mse_losses"], 
                color=colors[name], marker=markers[name], 
                linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("MSE Loss Progression", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Training Loss
    ax = axes[0, 2]
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["avg_train_loss"], 
                color=colors[name], marker=markers[name], 
                linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Avg Training Loss", fontsize=11)
    ax.set_title("Average Training Loss", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Model Divergence
    ax = axes[1, 0]
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["avg_divergence"], 
                color=colors[name], marker=markers[name], 
                linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Avg Divergence", fontsize=11)
    ax.set_title("Average Model Divergence", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Effective μ
    ax = axes[1, 1]
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["avg_effective_mu"], 
                color=colors[name], marker=markers[name], 
                linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Avg Effective μ", fontsize=11)
    ax.set_title("Average Proximal Coefficient (μ)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Final Comparison Bar Chart
    ax = axes[1, 2]
    names = list(all_results.keys())
    final_r2 = [all_results[n]["r2_scores"][-1] for n in names]
    final_mse = [all_results[n]["mse_losses"][-1] for n in names]
    
    x = np.arange(len(names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, final_r2, width, label='Final R²', color=[colors[n] for n in names], alpha=0.8)
    ax.set_ylabel("R² Score", fontsize=11)
    ax.set_title("Final Performance Comparison", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars1, final_r2):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)
    
    # Secondary y-axis for MSE
    ax2 = ax.twinx()
    ax2.plot(x, final_mse, 'ko--', linewidth=2, markersize=10, label='Final MSE')
    ax2.set_ylabel("MSE Loss", fontsize=11)
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Comparison plot saved to '{save_path}'")
    
    # Also save individual metric plots
    save_individual_plots(all_results, colors, markers)


def save_individual_plots(all_results: dict, colors: dict, markers: dict):
    """Save individual plots for each metric."""
    
    # R² Score only
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["r2_scores"], 
                color=colors[name], marker=markers[name], 
                linewidth=2.5, markersize=10, label=f'{name}')
        # Add final value annotation
        final_r2 = metrics["r2_scores"][-1]
        ax.annotate(f'{final_r2:.4f}', 
                    xy=(metrics["rounds"][-1], final_r2),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("R² Score Comparison: FedAvg vs FedProx vs SmartFedProx", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("r2_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ R² comparison plot saved to 'r2_comparison.png'")
    
    # MSE Loss only
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["mse_losses"], 
                color=colors[name], marker=markers[name], 
                linewidth=2.5, markersize=10, label=f'{name}')
    
    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("MSE Loss Comparison: FedAvg vs FedProx vs SmartFedProx", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("mse_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ MSE comparison plot saved to 'mse_comparison.png'")


def print_summary(all_results: dict):
    """Print summary table of results."""
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"{'Strategy':<20} {'Final R²':>12} {'Final MSE':>12} {'Best R²':>12} {'Lowest MSE':>12}")
    print("-"*70)
    
    for name, metrics in all_results.items():
        final_r2 = metrics["r2_scores"][-1]
        final_mse = metrics["mse_losses"][-1]
        best_r2 = max(metrics["r2_scores"])
        lowest_mse = min(metrics["mse_losses"])
        print(f"{name:<20} {final_r2:>12.4f} {final_mse:>12.4f} {best_r2:>12.4f} {lowest_mse:>12.4f}")
    
    print("-"*70)
    
    # Determine winner
    final_r2_scores = {name: metrics["r2_scores"][-1] for name, metrics in all_results.items()}
    winner = max(final_r2_scores, key=final_r2_scores.get)
    print(f"\n🏆 Best performing strategy: {winner} (R² = {final_r2_scores[winner]:.4f})")
    print("="*70)


def main():
    """Main entry point."""
    NUM_TRIALS = 3  # Run multiple trials for statistical significance
    
    print("\n" + "="*70)
    print("FEDERATED LEARNING STRATEGY COMPARISON")
    print("Dataset: Indian Personal Finance (Disposable Income Prediction)")
    print("Non-IID: EXTREME (Occupation + City_Tier + Income stratification)")
    print(f"Device: {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}, Fraction Fit: {FRACTION_FIT}")
    print(f"Rounds: {NUM_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
    print(f"Trials: {NUM_TRIALS}")
    print("="*70)
    
    # Reset cache and preload data with new extreme non-IID partitioning
    print("\nResetting data cache and loading with EXTREME non-IID partitioning...")
    reset_data_cache()
    _load_and_preprocess_data()
    print(f"Input dimension: {get_input_dim()}")
    
    # Storage for all trial results
    all_trial_results = {name: [] for name in STRATEGIES.keys()}
    
    # Use time-based base seed for different results each run
    import time
    base_seed = int(time.time()) % 10000
    print(f"Base seed for this run: {base_seed}")
    
    for trial in range(NUM_TRIALS):
        print(f"\n{'#'*70}")
        print(f"# TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'#'*70}")
        
        # Use different seed for each trial (but consistent within trial for fair comparison)
        trial_seed = base_seed + trial * 100
        
        for strategy_name, config in STRATEGIES.items():
            # Reset random seed for fair comparison WITHIN a trial
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            
            simulator = FederatedSimulator(strategy_name, config)
            metrics = simulator.run(NUM_ROUNDS)
            all_trial_results[strategy_name].append(metrics)
    
    # Aggregate results across trials
    print("\n" + "="*70)
    print("AGGREGATED RESULTS ACROSS ALL TRIALS")
    print("="*70)
    
    aggregated_results = {}
    for strategy_name in STRATEGIES.keys():
        trials = all_trial_results[strategy_name]
        
        # Average across trials
        avg_final_r2 = np.mean([t["r2_scores"][-1] for t in trials])
        std_final_r2 = np.std([t["r2_scores"][-1] for t in trials])
        avg_final_mse = np.mean([t["mse_losses"][-1] for t in trials])
        avg_best_r2 = np.mean([max(t["r2_scores"]) for t in trials])
        
        aggregated_results[strategy_name] = {
            "avg_final_r2": avg_final_r2,
            "std_final_r2": std_final_r2,
            "avg_final_mse": avg_final_mse,
            "avg_best_r2": avg_best_r2,
            # Use first trial for plotting (representative)
            "rounds": trials[0]["rounds"],
            "r2_scores": [np.mean([t["r2_scores"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "mse_losses": [np.mean([t["mse_losses"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "avg_train_loss": [np.mean([t["avg_train_loss"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "avg_divergence": [np.mean([t["avg_divergence"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "avg_effective_mu": [np.mean([t["avg_effective_mu"][i] for t in trials]) for i in range(NUM_ROUNDS)],
        }
        
        print(f"{strategy_name}:")
        print(f"  Final R²: {avg_final_r2:.4f} ± {std_final_r2:.4f}")
        print(f"  Final MSE: {avg_final_mse:.4f}")
        print(f"  Best R² (avg): {avg_best_r2:.4f}")
    
    # Determine winner
    winner = max(aggregated_results.keys(), key=lambda x: aggregated_results[x]["avg_final_r2"])
    print(f"\n🏆 Best performing strategy: {winner} (R² = {aggregated_results[winner]['avg_final_r2']:.4f} ± {aggregated_results[winner]['std_final_r2']:.4f})")
    print("="*70)
    
    # Generate plots with averaged results
    print("\nGenerating comparison plots (averaged across trials)...")
    plot_comparison(aggregated_results)
    
    print("\n✅ All simulations complete!")
    print("Generated files:")
    print("  - comparison_results.png (comprehensive comparison)")
    print("  - r2_comparison.png (R² score only)")
    print("  - mse_comparison.png (MSE loss only)")


if __name__ == "__main__":
    main()
