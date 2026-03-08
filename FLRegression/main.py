#Strategies compared:
#1. FedAvg: Random client selection
#2. FedProx: Proximal term (μ=0.1), random client selection
#3. SmartFedProx: Proximal term (μ=0.1), hybrid client selection with adaptive μ


import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from module import (
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS,
    DEVICE, STRATEGIES, _load_and_preprocess_data,
    reset_data_cache
)
from server import FederatedSimulator


def plot_comparison(all_results: dict, save_path: str = "comparison_results.png"):
    """Create comprehensive comparison plots."""

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Federated Learning Strategy Comparison\n(CIFAR-10 Image Classification)",
                 fontsize=14, fontweight='bold')

    colors = {"FedAvg": "#e74c3c", "FedProx": "#3498db", "SmartFedProx": "#2ecc71"}
    markers = {"FedAvg": "o", "FedProx": "s", "SmartFedProx": "^"}

    # Plot 1: Accuracy
    ax = axes[0, 0]
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["accuracies"],
                color=colors[name], marker=markers[name],
                linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Accuracy Progression", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Loss
    ax = axes[0, 1]
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["losses"],
                color=colors[name], marker=markers[name],
                linewidth=2, markersize=8, label=name)
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax.set_title("Loss Progression", fontsize=12, fontweight='bold')
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
    final_acc = [all_results[n]["accuracies"][-1] for n in names]
    final_loss = [all_results[n]["losses"][-1] for n in names]

    x = np.arange(len(names))
    width = 0.35

    bars1 = ax.bar(x - width/2, final_acc, width, label='Final Accuracy', color=[colors[n] for n in names], alpha=0.8)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_title("Final Performance Comparison", fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, val in zip(bars1, final_acc):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9)

    # Secondary y-axis for Loss
    ax2 = ax.twinx()
    ax2.plot(x, final_loss, 'ko--', linewidth=2, markersize=10, label='Final Loss')
    ax2.set_ylabel("CrossEntropy Loss", fontsize=11)
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n✓ Comparison plot saved to '{save_path}'")

    # Also save individual metric plots
    save_individual_plots(all_results, colors, markers)


def save_individual_plots(all_results: dict, colors: dict, markers: dict):
    """Save individual plots for each metric."""

    # Accuracy only
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["accuracies"],
                color=colors[name], marker=markers[name],
                linewidth=2.5, markersize=10, label=f'{name}')
        # Add final value annotation
        final_acc = metrics["accuracies"][-1]
        ax.annotate(f'{final_acc:.4f}',
                    xy=(metrics["rounds"][-1], final_acc),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=10, fontweight='bold')

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Accuracy Comparison: FedAvg vs FedProx vs SmartFedProx", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("accuracy_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ Accuracy comparison plot saved to 'accuracy_comparison.png'")

    # Loss only
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, metrics in all_results.items():
        ax.plot(metrics["rounds"], metrics["losses"],
                color=colors[name], marker=markers[name],
                linewidth=2.5, markersize=10, label=f'{name}')

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("CrossEntropy Loss", fontsize=12)
    ax.set_title("Loss Comparison: FedAvg vs FedProx vs SmartFedProx", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("loss_comparison.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("✓ Loss comparison plot saved to 'loss_comparison.png'")


def print_summary(all_results: dict):
    """Print summary table of results."""
    print("\n" + "="*70)
    print("SUMMARY OF RESULTS")
    print("="*70)
    print(f"{'Strategy':<20} {'Final Acc':>12} {'Final Loss':>12} {'Best Acc':>12} {'Low Loss':>12}")
    print("-"*70)

    for name, metrics in all_results.items():
        final_acc = metrics["accuracies"][-1]
        final_loss = metrics["losses"][-1]
        best_acc = max(metrics["accuracies"])
        lowest_loss = min(metrics["losses"])
        print(f"{name:<20} {final_acc:>12.4f} {final_loss:>12.4f} {best_acc:>12.4f} {lowest_loss:>12.4f}")

    print("-"*70)

    # Determine winner
    final_accuracies = {name: metrics["accuracies"][-1] for name, metrics in all_results.items()}
    winner = max(final_accuracies, key=final_accuracies.get)
    print(f"\n🏆 Best performing strategy: {winner} (Accuracy = {final_accuracies[winner]:.4f})")
    print("="*70)


def main():
    NUM_TRIALS = 3
    FIXED_SEED = 2023

    print("\n" + "="*70)
    print("FEDERATED LEARNING STRATEGY COMPARISON")
    print("Dataset: CIFAR-10 (Image Classification)")
    print("Non-IID: EXTREME (Dirichlet α=0.1 label skew + quantity skew)")
    print(f"Device: {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}, Fraction Fit: {FRACTION_FIT}")
    print(f"Rounds: {NUM_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
    print(f"Trials: {NUM_TRIALS}")
    print("="*70)

    # Reset cache and preload data with extreme non-IID partitioning
    print("\nResetting data cache and loading CIFAR-10 with EXTREME non-IID partitioning...")
    reset_data_cache()
    _load_and_preprocess_data()
    print("CIFAR-10 dataset loaded successfully.")

    # Storage for all trial results
    all_trial_results = {name: [] for name in STRATEGIES.keys()}

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
        avg_final_acc = np.mean([t["accuracies"][-1] for t in trials])
        std_final_acc = np.std([t["accuracies"][-1] for t in trials])
        avg_final_loss = np.mean([t["losses"][-1] for t in trials])
        avg_best_acc = np.mean([max(t["accuracies"]) for t in trials])

        aggregated_results[strategy_name] = {
            "avg_final_acc": avg_final_acc,
            "std_final_acc": std_final_acc,
            "avg_final_loss": avg_final_loss,
            "avg_best_acc": avg_best_acc,
            # Use first trial for plotting (representative)
            "rounds": trials[0]["rounds"],
            "accuracies": [np.mean([t["accuracies"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "losses": [np.mean([t["losses"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "avg_train_loss": [np.mean([t["avg_train_loss"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "avg_divergence": [np.mean([t["avg_divergence"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "avg_effective_mu": [np.mean([t["avg_effective_mu"][i] for t in trials]) for i in range(NUM_ROUNDS)],
        }

        print(f"{strategy_name}:")
        print(f"  Final Accuracy: {avg_final_acc:.4f} ± {std_final_acc:.4f}")
        print(f"  Final Loss: {avg_final_loss:.4f}")
        print(f"  Best Accuracy (avg): {avg_best_acc:.4f}")

    # Determine winner
    winner = max(aggregated_results.keys(), key=lambda x: aggregated_results[x]["avg_final_acc"])
    print(f"\n🏆 Best performing strategy: {winner} (Accuracy = {aggregated_results[winner]['avg_final_acc']:.4f} ± {aggregated_results[winner]['std_final_acc']:.4f})")

    # Check if results are statistically significant
    all_final_acc = [aggregated_results[name]["avg_final_acc"] for name in STRATEGIES.keys()]
    max_acc = max(all_final_acc)
    min_acc = min(all_final_acc)
    difference = max_acc - min_acc
    avg_std = np.mean([aggregated_results[name]["std_final_acc"] for name in STRATEGIES.keys()])

    if difference < avg_std:
        print(f"\n⚠️  Note: Strategy differences ({difference:.4f}) are smaller than average std dev ({avg_std:.4f})")
        print("   Results may vary between runs. Consider running more trials for better statistical power.")

    print("="*70)

    # Generate plots with averaged results
    print("\nGenerating comparison plots (averaged across trials)...")
    plot_comparison(aggregated_results)

    print("\n✅ All simulations complete!")
    print("Generated files:")
    print("  - comparison_results.png (comprehensive comparison)")
    print("  - accuracy_comparison.png (accuracy only)")
    print("  - loss_comparison.png (loss only)")


if __name__ == "__main__":
    main()
