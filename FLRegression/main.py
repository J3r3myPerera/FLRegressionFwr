#Strategies compared:
#1. FedAvg: Random client selection
#2. FedProx: Proximal term (μ=0.1), random client selection  
#3. SmartFedProx: Proximal term (μ=0.1), hybrid client selection with adaptive μ

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import wandb

from module import (
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS, 
    DEVICE, STRATEGIES, get_input_dim, _load_and_preprocess_data, 
    reset_data_cache
)
from server import FederatedSimulator


def plot_comparison(all_results: dict, save_path: str = "resultOutput/comparison_results.png"):
    """Create comprehensive comparison plots."""
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

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
    plt.savefig("resultOutput/r2_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("R² comparison plot saved to 'resultOutput/r2_comparison.png'")
    
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
    plt.savefig("resultOutput/mse_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("MSE comparison plot saved to 'resultOutput/mse_comparison.png'")


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
    print(f"Best performing strategy: {winner} (R² = {final_r2_scores[winner]:.4f})")
    print("="*70)


def main():
    NUM_TRIALS = 3 
    FIXED_SEED = 2023
    
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
    
    # Storage for all trial results  and final models
    all_trial_results = {name: [] for name in STRATEGIES.keys()}
    final_models = {}
    
   
    if FIXED_SEED is not None:
        base_seed = FIXED_SEED
        print(f"Using fixed seed: {base_seed} (for reproducibility)")
    else:
        base_seed = int(time.time()) % 10000
        print(f"Using time-based seed: {base_seed} (results will vary each run)")
    
    for trial in range(NUM_TRIALS):
        print(f"\n{'#'*70}")
        print(f"# TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'#'*70}")
        
        # Use different seed for each trial (but consistent within trial for fair comparison)
        trial_seed = base_seed + trial * 100
        
        for strategy_name, config in STRATEGIES.items():

            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)

            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)

            wandb.init(
                entity="dinuka-20210433-university-of-westminster",
                project="fl-regression",
                group=strategy_name,
                name=f"{strategy_name}_trial_{trial + 1}",
                config={
                    "strategy": strategy_name,
                    "num_rounds": NUM_ROUNDS,
                    "num_clients": NUM_CLIENTS,
                    "fraction_fit": FRACTION_FIT,
                    "local_epochs": LOCAL_EPOCHS,
                    **config,
                },
                reinit="finish_previous",
            )

            simulator = FederatedSimulator(strategy_name, config)
            metrics, model_state = simulator.run(NUM_ROUNDS)
            all_trial_results[strategy_name].append(metrics)
            final_models[strategy_name] = model_state

            wandb.finish()
    
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
    print(f"\n Best performing strategy: {winner} (R² = {aggregated_results[winner]['avg_final_r2']:.4f} ± {aggregated_results[winner]['std_final_r2']:.4f})")
    
    # Check if results are statistically significant
    all_final_r2 = [aggregated_results[name]["avg_final_r2"] for name in STRATEGIES.keys()]
    max_r2 = max(all_final_r2)
    min_r2 = min(all_final_r2)
    difference = max_r2 - min_r2
    avg_std = np.mean([aggregated_results[name]["std_final_r2"] for name in STRATEGIES.keys()])
    
    if difference < avg_std:
        print(f"\n Note: Strategy differences ({difference:.4f}) are smaller than average std dev ({avg_std:.4f})")
        print("   Results may vary between runs. Consider running more trials for better statistical power.")
    print("="*70)
    
    # Generate plots with averaged results
    print("\nGenerating comparison plots (averaged across trials)...")
    plot_comparison(aggregated_results)

    # Save final models
    os.makedirs("resultOutput", exist_ok=True)
    for strategy_name, model_state in final_models.items():
        model_path = f"resultOutput/{strategy_name}_final_model.pt"
        torch.save(model_state, model_path)
        print(f"Model saved to '{model_path}'")
    
    print("\n All simulations complete!")
    print("Generated files (in resultOutput/):")
    print("  - resultOutput/comparison_results.png (comprehensive comparison)")
    print("  - resultOutput/r2_comparison.png (R² score only)")
    print("  - resultOutput/mse_comparison.png (MSE loss only)")
    for strategy_name in STRATEGIES.keys():
        print(f"  - resultOutput/{strategy_name}_final_model.pt (final model)")


if __name__ == "__main__":
    main()
