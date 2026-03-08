"""
Ablation Study: Client Selection Split Ratios

Tests different high/mid/low divergence split ratios for the SmartFedProx
hybrid client selection strategy to validate the chosen 30/50/20 split.

Configurations tested:
  1. Current (30/50/20)  — baseline design
  2. High-heavy (50/30/20) — more correction focus
  3. Low-heavy (20/30/50)  — more stability focus
  4. Uniform (33/34/33)    — no preference
  5. Random baseline       — pure random selection (no divergence-based split)
"""

import os
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from module import (
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS,
    DEVICE, get_input_dim, _load_and_preprocess_data,
    reset_data_cache,
)
from server import FederatedSimulator
from metrics_table import generate_per_round_table


# Ablation configurations — all use SmartFedProx base with adaptive mu + hybrid selection
# except "Random" which uses random selection as a control
ABLATION_CONFIGS = {
    "30/50/20 (Current)": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "hybrid",
        "high_ratio": 0.3,
        "low_ratio": 0.2,
        "description": "SmartFedProx — 30% high, 50% mid, 20% low (current design)",
    },
    "50/30/20 (High-heavy)": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "hybrid",
        "high_ratio": 0.5,
        "low_ratio": 0.2,
        "description": "SmartFedProx — 50% high, 30% mid, 20% low",
    },
    "20/30/50 (Low-heavy)": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "hybrid",
        "high_ratio": 0.2,
        "low_ratio": 0.5,
        "description": "SmartFedProx — 20% high, 30% mid, 50% low",
    },
    "33/34/33 (Uniform)": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "hybrid",
        "high_ratio": 0.33,
        "low_ratio": 0.33,
        "description": "SmartFedProx — 33% high, 34% mid, 33% low (uniform)",
    },
    "Random (Control)": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "random",
        "description": "FedProx with adaptive mu — random selection (no split)",
    },
}

OUTPUT_DIR = "outputs/ablation_split"


def plot_ablation_comparison(aggregated_results: dict, save_dir: str):
    """Create comparison plots for the ablation study."""

    colors = {
        "30/50/20 (Current)":     "#2ecc71",
        "50/30/20 (High-heavy)":  "#e74c3c",
        "20/30/50 (Low-heavy)":   "#3498db",
        "33/34/33 (Uniform)":     "#f39c12",
        "Random (Control)":       "#9b59b6",
    }
    markers = {
        "30/50/20 (Current)":     "^",
        "50/30/20 (High-heavy)":  "v",
        "20/30/50 (Low-heavy)":   "s",
        "33/34/33 (Uniform)":     "D",
        "Random (Control)":       "o",
    }

    # --- Multi-panel figure ---
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Ablation Study: Hybrid Client Selection Split Ratios\n"
        "(SmartFedProx — Personal Finance, Extreme Non-IID)",
        fontsize=14, fontweight='bold',
    )

    metric_specs = [
        ("r2_scores",      "R\u00b2 Score",         axes[0, 0]),
        ("mse_losses",     "MSE Loss",              axes[0, 1]),
        ("rmse_scores",    "RMSE",                  axes[0, 2]),
        ("mae_scores",     "MAE",                   axes[1, 0]),
        ("avg_divergence", "Avg Model Divergence",  axes[1, 1]),
        ("avg_train_loss", "Avg Training Loss",     axes[1, 2]),
    ]

    for key, ylabel, ax in metric_specs:
        for name, data in aggregated_results.items():
            rounds = data["rounds"]
            values = data[key]
            stds = data.get(f"{key}_std", None)

            ax.plot(rounds, values,
                    color=colors[name], marker=markers[name],
                    linewidth=2, markersize=6, label=name)

            if stds is not None:
                values_arr = np.array(values)
                stds_arr = np.array(stds)
                ax.fill_between(rounds,
                                values_arr - stds_arr,
                                values_arr + stds_arr,
                                color=colors[name], alpha=0.1)

        ax.set_xlabel("Federated Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_split_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved multi-panel plot -> {path}")

    # --- Dedicated R2 plot ---
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in aggregated_results.items():
        rounds = data["rounds"]
        values = data["r2_scores"]
        stds = data.get("r2_scores_std", None)
        ax.plot(rounds, values,
                color=colors[name], marker=markers[name],
                linewidth=2.5, markersize=8, label=name)
        if stds is not None:
            values_arr = np.array(values)
            stds_arr = np.array(stds)
            ax.fill_between(rounds,
                            values_arr - stds_arr,
                            values_arr + stds_arr,
                            color=colors[name], alpha=0.12)
        final_r2 = values[-1]
        ax.annotate(f'{final_r2:.4f}',
                    xy=(rounds[-1], final_r2),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=colors[name])

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("R\u00b2 Score", fontsize=12)
    ax.set_title("Ablation: R\u00b2 Convergence by Split Ratio", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_r2_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved R2 comparison plot -> {path}")

    # --- Final bar chart ---
    fig, ax = plt.subplots(figsize=(10, 6))
    names = list(aggregated_results.keys())
    final_r2 = [aggregated_results[n]["avg_final_r2"] for n in names]
    final_std = [aggregated_results[n]["std_final_r2"] for n in names]
    bar_colors = [colors[n] for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, final_r2, yerr=final_std, capsize=5,
                  color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

    for bar, val, std in zip(bars, final_r2, final_std):
        ax.annotate(f'{val:.4f}\n+/-{std:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')

    ax.set_ylabel("Final R\u00b2 Score (mean +/- std)", fontsize=12)
    ax.set_title("Ablation: Final R\u00b2 by Split Ratio", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_final_r2_bar.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved final R2 bar chart -> {path}")


def print_ablation_summary(aggregated_results: dict):
    """Print summary comparison table."""
    print("\n" + "=" * 110)
    print("ABLATION STUDY: SPLIT RATIO COMPARISON")
    print("=" * 110)
    print(f"{'Configuration':<25} {'Split (H/M/L)':<16} {'Final R2':>12} {'Std':>8} {'Final MSE':>11} {'Final RMSE':>12} {'Best R2':>10}")
    print("-" * 110)

    for name, data in aggregated_results.items():
        config = ABLATION_CONFIGS[name]
        if config["selection_strategy"] == "random":
            split_str = "Random"
        else:
            h = int(config["high_ratio"] * 100)
            l = int(config["low_ratio"] * 100)
            m = 100 - h - l
            split_str = f"{h}/{m}/{l}"

        print(f"{name:<25} {split_str:<16} "
              f"{data['avg_final_r2']:>12.4f} {data['std_final_r2']:>8.4f} "
              f"{data['avg_final_mse']:>11.4f} "
              f"{data['avg_final_rmse']:>12.4f} "
              f"{data['avg_best_r2']:>10.4f}")

    print("-" * 110)

    # Determine winner
    winner = max(aggregated_results.keys(),
                 key=lambda x: aggregated_results[x]["avg_final_r2"])
    w = aggregated_results[winner]
    print(f"\nBest performing split: {winner}")
    print(f"  R2 = {w['avg_final_r2']:.4f} +/- {w['std_final_r2']:.4f}")
    print("=" * 110)


def save_summary_txt(aggregated_results: dict, save_dir: str):
    """Save summary table as a text file."""
    lines = []
    lines.append("=" * 110)
    lines.append("ABLATION STUDY: CLIENT SELECTION SPLIT RATIO COMPARISON")
    lines.append("=" * 110)
    lines.append(f"{'Configuration':<25} {'Split (H/M/L)':<16} {'Final R2':>12} {'Std':>8} {'Final MSE':>11} {'Final RMSE':>12} {'Best R2':>10}")
    lines.append("-" * 110)

    for name, data in aggregated_results.items():
        config = ABLATION_CONFIGS[name]
        if config["selection_strategy"] == "random":
            split_str = "Random"
        else:
            h = int(config["high_ratio"] * 100)
            l = int(config["low_ratio"] * 100)
            m = 100 - h - l
            split_str = f"{h}/{m}/{l}"

        lines.append(f"{name:<25} {split_str:<16} "
                     f"{data['avg_final_r2']:>12.4f} {data['std_final_r2']:>8.4f} "
                     f"{data['avg_final_mse']:>11.4f} "
                     f"{data['avg_final_rmse']:>12.4f} "
                     f"{data['avg_best_r2']:>10.4f}")

    lines.append("-" * 110)
    winner = max(aggregated_results.keys(),
                 key=lambda x: aggregated_results[x]["avg_final_r2"])
    w = aggregated_results[winner]
    lines.append(f"\nBest performing split: {winner}")
    lines.append(f"  R2 = {w['avg_final_r2']:.4f} +/- {w['std_final_r2']:.4f}")
    lines.append("=" * 110)

    path = os.path.join(save_dir, "ablation_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved summary table -> {path}")


def main():
    NUM_TRIALS = 3
    FIXED_SEED = 2023

    print("\n" + "=" * 70)
    print("ABLATION STUDY: CLIENT SELECTION SPLIT RATIOS")
    print("Dataset: Indian Personal Finance (Disposable Income Prediction)")
    print("Non-IID: EXTREME (Occupation + City_Tier + Income stratification)")
    print(f"Device: {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}, Fraction Fit: {FRACTION_FIT}")
    print(f"Rounds: {NUM_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
    print(f"Trials: {NUM_TRIALS}")
    print("=" * 70)

    # Reset cache and preload data
    print("\nResetting data cache and loading with EXTREME non-IID partitioning...")
    reset_data_cache()
    _load_and_preprocess_data()
    print(f"Input dimension: {get_input_dim()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Storage for all trial results
    all_trial_results = {name: [] for name in ABLATION_CONFIGS.keys()}

    base_seed = FIXED_SEED
    print(f"Using fixed seed: {base_seed} (for reproducibility)")

    for trial in range(NUM_TRIALS):
        print(f"\n{'#' * 70}")
        print(f"# TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'#' * 70}")

        trial_seed = base_seed + trial * 100

        for config_name, config in ABLATION_CONFIGS.items():
            # Reset RNG for fair comparison within each trial
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)

            simulator = FederatedSimulator(config_name, config)
            metrics = simulator.run(NUM_ROUNDS)
            all_trial_results[config_name].append(metrics)

    # Aggregate results across trials
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS ACROSS ALL TRIALS")
    print("=" * 70)

    aggregated_results = {}
    metric_keys = ["r2_scores", "mse_losses", "rmse_scores", "mae_scores",
                   "avg_train_loss", "avg_divergence", "avg_effective_mu"]

    for config_name in ABLATION_CONFIGS.keys():
        trials = all_trial_results[config_name]

        agg = {
            "avg_final_r2":   np.mean([t["r2_scores"][-1] for t in trials]),
            "std_final_r2":   np.std([t["r2_scores"][-1] for t in trials]),
            "avg_final_mse":  np.mean([t["mse_losses"][-1] for t in trials]),
            "avg_final_rmse": np.mean([t["rmse_scores"][-1] for t in trials]),
            "avg_best_r2":    np.mean([max(t["r2_scores"]) for t in trials]),
            "rounds":         trials[0]["rounds"],
        }

        # Per-round mean and std
        for key in metric_keys:
            agg[key] = [np.mean([t[key][i] for t in trials]) for i in range(NUM_ROUNDS)]
            agg[f"{key}_std"] = [np.std([t[key][i] for t in trials]) for i in range(NUM_ROUNDS)]

        aggregated_results[config_name] = agg

        print(f"{config_name}:")
        print(f"  Final R2: {agg['avg_final_r2']:.4f} +/- {agg['std_final_r2']:.4f}")
        print(f"  Final MSE: {agg['avg_final_mse']:.4f}")
        print(f"  Best R2 (avg): {agg['avg_best_r2']:.4f}")

    # Print summary
    print_ablation_summary(aggregated_results)

    # Generate per-round tables (sanitize names for file paths)
    print("\nGenerating per-round metrics tables...")
    safe_trial_results = {}
    for name, trials in all_trial_results.items():
        safe_name = name.replace("/", "-").replace(" ", "_").replace("(", "").replace(")", "")
        safe_trial_results[safe_name] = trials
    generate_per_round_table(
        all_trial_results=safe_trial_results,
        num_rounds=NUM_ROUNDS,
        output_dir=os.path.join(OUTPUT_DIR, "metrics_tables"),
        save_txt=True,
        save_latex=True,
    )

    # Generate plots
    print("\nGenerating comparison plots...")
    plot_ablation_comparison(aggregated_results, OUTPUT_DIR)

    # Save summary
    save_summary_txt(aggregated_results, OUTPUT_DIR)

    print("\nAblation study complete!")
    print("Generated files:")
    print(f"  - {OUTPUT_DIR}/ablation_split_comparison.png  (multi-panel comparison)")
    print(f"  - {OUTPUT_DIR}/ablation_r2_comparison.png     (R2 convergence)")
    print(f"  - {OUTPUT_DIR}/ablation_final_r2_bar.png      (final R2 bar chart)")
    print(f"  - {OUTPUT_DIR}/ablation_summary.txt           (summary table)")
    print(f"  - {OUTPUT_DIR}/metrics_tables/                (per-round tables)")


if __name__ == "__main__":
    main()
