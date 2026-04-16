"""
Component Ablation Study: Client Selection vs Adaptive Mu

Isolates the individual and combined contributions of the two SmartFedProx
components using a 2x2 factorial design:

  1. Random + Fixed mu     — baseline FedProx (neither component)
  2. Random + Adaptive mu  — adaptive mu only
  3. Hybrid + Fixed mu     — client selection only
  4. Hybrid + Adaptive mu  — full SmartFedProx (both components)
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure FLRegression is on the path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "FLRegression"))
sys.path.insert(0, _SCRIPT_DIR)

from module import (
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS,
    DEVICE, get_input_dim, _load_and_preprocess_data,
    reset_data_cache,
)
from server import FederatedSimulator
from metrics_table import generate_per_round_table


COMPONENT_CONFIGS = {
    "Random + Fixed mu": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": False,
        "selection_strategy": "random",
        "description": "FedProx baseline (random selection, fixed mu=0.1)",
    },
    "Random + Adaptive mu": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "random",
        "description": "Adaptive mu only (random selection, adaptive mu)",
    },
    "Hybrid + Fixed mu": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": False,
        "selection_strategy": "hybrid",
        "description": "Client selection only (hybrid selection, fixed mu=0.1)",
    },
    "Hybrid + Adaptive mu": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "hybrid",
        "description": "Full SmartFedProx (hybrid selection, adaptive mu)",
    },
}

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "testScripts", "results")

COLORS = {
    "Random + Fixed mu":    "#95a5a6",
    "Random + Adaptive mu": "#e67e22",
    "Hybrid + Fixed mu":    "#3498db",
    "Hybrid + Adaptive mu": "#2ecc71",
}
MARKERS = {
    "Random + Fixed mu":    "o",
    "Random + Adaptive mu": "D",
    "Hybrid + Fixed mu":    "s",
    "Hybrid + Adaptive mu": "^",
}


def plot_component_comparison(aggregated_results: dict, save_dir: str):
    """Multi-panel convergence plots."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Component Ablation: Client Selection vs Adaptive Mu\n"
        "(SmartFedProx — Personal Finance, Extreme Non-IID)",
        fontsize=14, fontweight='bold',
    )

    metric_specs = [
        ("r2_scores",      "R\u00b2 Score",        axes[0, 0]),
        ("mse_losses",     "MSE Loss",             axes[0, 1]),
        ("rmse_scores",    "RMSE",                 axes[0, 2]),
        ("mae_scores",     "MAE",                  axes[1, 0]),
        ("avg_divergence", "Avg Model Divergence", axes[1, 1]),
        ("avg_train_loss", "Avg Training Loss",    axes[1, 2]),
    ]

    for key, ylabel, ax in metric_specs:
        for name, data in aggregated_results.items():
            rounds = data["rounds"]
            values = data[key]
            stds = data.get(f"{key}_std")

            ax.plot(rounds, values,
                    color=COLORS[name], marker=MARKERS[name],
                    linewidth=2, markersize=6, label=name)
            if stds is not None:
                v = np.array(values)
                s = np.array(stds)
                ax.fill_between(rounds, v - s, v + s,
                                color=COLORS[name], alpha=0.1)

        ax.set_xlabel("Federated Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "component_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved multi-panel plot -> {path}")


def plot_r2_comparison(aggregated_results: dict, save_dir: str):
    """Dedicated R2 convergence plot with confidence bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in aggregated_results.items():
        rounds = data["rounds"]
        values = data["r2_scores"]
        stds = data.get("r2_scores_std")

        ax.plot(rounds, values,
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2.5, markersize=8, label=name)
        if stds is not None:
            v = np.array(values)
            s = np.array(stds)
            ax.fill_between(rounds, v - s, v + s,
                            color=COLORS[name], alpha=0.12)
        ax.annotate(f'{values[-1]:.4f}',
                    xy=(rounds[-1], values[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=COLORS[name])

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("R\u00b2 Score", fontsize=12)
    ax.set_title("Component Ablation: R\u00b2 Convergence", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "component_r2_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved R2 comparison plot -> {path}")


def plot_contribution_bar(aggregated_results: dict, save_dir: str):
    """Bar chart showing final R2 + individual contribution breakdown."""
    names = list(COMPONENT_CONFIGS.keys())
    final_r2 = [aggregated_results[n]["avg_final_r2"] for n in names]
    final_std = [aggregated_results[n]["std_final_r2"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: Final R2 bar chart ---
    ax = axes[0]
    x = np.arange(len(names))
    bars = ax.bar(x, final_r2, yerr=final_std, capsize=5,
                  color=[COLORS[n] for n in names], alpha=0.85,
                  edgecolor='black', linewidth=0.5)
    for bar, val, std in zip(bars, final_r2, final_std):
        ax.annotate(f'{val:.4f}\n+/-{std:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel("Final R\u00b2 Score", fontsize=12)
    ax.set_title("Final R\u00b2 by Component Configuration", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # --- Panel 2: Contribution breakdown ---
    ax = axes[1]
    baseline = aggregated_results["Random + Fixed mu"]["avg_final_r2"]
    selection_only = aggregated_results["Hybrid + Fixed mu"]["avg_final_r2"] - baseline
    adaptive_only = aggregated_results["Random + Adaptive mu"]["avg_final_r2"] - baseline
    full_gain = aggregated_results["Hybrid + Adaptive mu"]["avg_final_r2"] - baseline
    interaction = full_gain - selection_only - adaptive_only

    contributions = [selection_only, adaptive_only, interaction]
    labels = ["Hybrid Selection\n(alone)", "Adaptive mu\n(alone)", "Interaction\n(synergy)"]
    bar_colors = [COLORS["Hybrid + Fixed mu"], COLORS["Random + Adaptive mu"], "#8e44ad"]

    bars = ax.bar(range(len(contributions)), contributions,
                  color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, contributions):
        sign = "+" if val >= 0 else ""
        ax.annotate(f'{sign}{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() if val >= 0 else bar.get_y()),
                    xytext=(0, 8 if val >= 0 else -15),
                    textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel("R\u00b2 Gain over Baseline", fontsize=12)
    ax.set_title("Individual Component Contributions", fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(contributions)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(save_dir, "component_contributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved contribution breakdown -> {path}")

    return baseline, selection_only, adaptive_only, interaction, full_gain


def print_component_summary(aggregated_results: dict):
    """Print summary table and contribution analysis."""
    print("\n" + "=" * 115)
    print("COMPONENT ABLATION: CLIENT SELECTION vs ADAPTIVE MU")
    print("=" * 115)
    print(f"{'Configuration':<26} {'Selection':<10} {'Mu Type':<12} {'Final R2':>10} {'Std':>8} {'Final MSE':>11} {'Final RMSE':>12} {'Best R2':>10}")
    print("-" * 115)

    meta = {
        "Random + Fixed mu":    ("Random", "Fixed"),
        "Random + Adaptive mu": ("Random", "Adaptive"),
        "Hybrid + Fixed mu":    ("Hybrid", "Fixed"),
        "Hybrid + Adaptive mu": ("Hybrid", "Adaptive"),
    }

    for name, data in aggregated_results.items():
        sel, mu = meta[name]
        print(f"{name:<26} {sel:<10} {mu:<12} "
              f"{data['avg_final_r2']:>10.4f} {data['std_final_r2']:>8.4f} "
              f"{data['avg_final_mse']:>11.4f} "
              f"{data['avg_final_rmse']:>12.4f} "
              f"{data['avg_best_r2']:>10.4f}")

    print("-" * 115)

    # Contribution analysis
    baseline = aggregated_results["Random + Fixed mu"]["avg_final_r2"]
    selection_only = aggregated_results["Hybrid + Fixed mu"]["avg_final_r2"] - baseline
    adaptive_only = aggregated_results["Random + Adaptive mu"]["avg_final_r2"] - baseline
    full_gain = aggregated_results["Hybrid + Adaptive mu"]["avg_final_r2"] - baseline
    interaction = full_gain - selection_only - adaptive_only

    print(f"\nCONTRIBUTION ANALYSIS (R2 gain over baseline {baseline:.4f}):")
    print(f"  Hybrid client selection alone: {selection_only:+.4f}")
    print(f"  Adaptive mu alone:             {adaptive_only:+.4f}")
    print(f"  Interaction (synergy):         {interaction:+.4f}")
    print(f"  Total (full SmartFedProx):     {full_gain:+.4f}")

    if abs(selection_only) > abs(adaptive_only):
        print(f"\n  -> Client selection contributes MORE than adaptive mu")
    elif abs(adaptive_only) > abs(selection_only):
        print(f"\n  -> Adaptive mu contributes MORE than client selection")
    else:
        print(f"\n  -> Both components contribute equally")

    print("=" * 115)


def save_component_summary(aggregated_results: dict, save_dir: str):
    """Save summary as text file."""
    baseline = aggregated_results["Random + Fixed mu"]["avg_final_r2"]
    selection_only = aggregated_results["Hybrid + Fixed mu"]["avg_final_r2"] - baseline
    adaptive_only = aggregated_results["Random + Adaptive mu"]["avg_final_r2"] - baseline
    full_gain = aggregated_results["Hybrid + Adaptive mu"]["avg_final_r2"] - baseline
    interaction = full_gain - selection_only - adaptive_only

    meta = {
        "Random + Fixed mu":    ("Random", "Fixed"),
        "Random + Adaptive mu": ("Random", "Adaptive"),
        "Hybrid + Fixed mu":    ("Hybrid", "Fixed"),
        "Hybrid + Adaptive mu": ("Hybrid", "Adaptive"),
    }

    lines = []
    lines.append("=" * 115)
    lines.append("COMPONENT ABLATION: CLIENT SELECTION vs ADAPTIVE MU")
    lines.append("=" * 115)
    lines.append(f"{'Configuration':<26} {'Selection':<10} {'Mu Type':<12} {'Final R2':>10} {'Std':>8} {'Final MSE':>11} {'Final RMSE':>12} {'Best R2':>10}")
    lines.append("-" * 115)

    for name, data in aggregated_results.items():
        sel, mu = meta[name]
        lines.append(f"{name:<26} {sel:<10} {mu:<12} "
                     f"{data['avg_final_r2']:>10.4f} {data['std_final_r2']:>8.4f} "
                     f"{data['avg_final_mse']:>11.4f} "
                     f"{data['avg_final_rmse']:>12.4f} "
                     f"{data['avg_best_r2']:>10.4f}")

    lines.append("-" * 115)
    lines.append(f"\nCONTRIBUTION ANALYSIS (R2 gain over baseline {baseline:.4f}):")
    lines.append(f"  Hybrid client selection alone: {selection_only:+.4f}")
    lines.append(f"  Adaptive mu alone:             {adaptive_only:+.4f}")
    lines.append(f"  Interaction (synergy):         {interaction:+.4f}")
    lines.append(f"  Total (full SmartFedProx):     {full_gain:+.4f}")
    lines.append("=" * 115)

    path = os.path.join(save_dir, "component_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved summary -> {path}")


def main():
    NUM_TRIALS = 3
    FIXED_SEED = 2023

    print("\n" + "=" * 70)
    print("COMPONENT ABLATION: CLIENT SELECTION vs ADAPTIVE MU")
    print("Dataset: Indian Personal Finance (Disposable Income Prediction)")
    print("Non-IID: EXTREME (Occupation + City_Tier + Income stratification)")
    print(f"Device: {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}, Fraction Fit: {FRACTION_FIT}")
    print(f"Rounds: {NUM_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
    print(f"Trials: {NUM_TRIALS}")
    print("=" * 70)

    print("\nResetting data cache and loading with EXTREME non-IID partitioning...")
    reset_data_cache()
    _load_and_preprocess_data()
    print(f"Input dimension: {get_input_dim()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_trial_results = {name: [] for name in COMPONENT_CONFIGS.keys()}
    base_seed = FIXED_SEED
    print(f"Using fixed seed: {base_seed} (for reproducibility)")

    for trial in range(NUM_TRIALS):
        print(f"\n{'#' * 70}")
        print(f"# TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'#' * 70}")

        trial_seed = base_seed + trial * 100

        for config_name, config in COMPONENT_CONFIGS.items():
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)

            simulator = FederatedSimulator(config_name, config)
            metrics, _model_state = simulator.run(NUM_ROUNDS)

            # Derive RMSE (exact) and MAE (estimated) from MSE
            metrics["rmse_scores"] = [mse ** 0.5 for mse in metrics["mse_losses"]]
            metrics["mae_scores"] = [rmse * 0.7979 for rmse in metrics["rmse_scores"]]

            all_trial_results[config_name].append(metrics)

    # Aggregate
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS ACROSS ALL TRIALS")
    print("=" * 70)

    aggregated_results = {}
    metric_keys = ["r2_scores", "mse_losses", "rmse_scores", "mae_scores",
                   "avg_train_loss", "avg_divergence", "avg_effective_mu"]

    for config_name in COMPONENT_CONFIGS.keys():
        trials = all_trial_results[config_name]
        agg = {
            "avg_final_r2":   np.mean([t["r2_scores"][-1] for t in trials]),
            "std_final_r2":   np.std([t["r2_scores"][-1] for t in trials]),
            "avg_final_mse":  np.mean([t["mse_losses"][-1] for t in trials]),
            "avg_final_rmse": np.mean([t["rmse_scores"][-1] for t in trials]),
            "avg_best_r2":    np.mean([max(t["r2_scores"]) for t in trials]),
            "rounds":         trials[0]["rounds"],
        }
        for key in metric_keys:
            agg[key] = [np.mean([t[key][i] for t in trials]) for i in range(NUM_ROUNDS)]
            agg[f"{key}_std"] = [np.std([t[key][i] for t in trials]) for i in range(NUM_ROUNDS)]

        aggregated_results[config_name] = agg
        print(f"{config_name}:")
        print(f"  Final R2: {agg['avg_final_r2']:.4f} +/- {agg['std_final_r2']:.4f}")
        print(f"  Final MSE: {agg['avg_final_mse']:.4f}")
        print(f"  Best R2 (avg): {agg['avg_best_r2']:.4f}")

    # Summary
    print_component_summary(aggregated_results)

    # Per-round tables
    print("\nGenerating per-round metrics tables...")
    safe_trial_results = {}
    for name, trials in all_trial_results.items():
        safe_name = name.replace("+", "plus").replace(" ", "_")
        safe_trial_results[safe_name] = trials
    generate_per_round_table(
        all_trial_results=safe_trial_results,
        num_rounds=NUM_ROUNDS,
        output_dir=os.path.join(OUTPUT_DIR, "metrics_tables"),
        save_txt=True,
        save_latex=True,
    )

    # Plots
    print("\nGenerating comparison plots...")
    plot_component_comparison(aggregated_results, OUTPUT_DIR)
    plot_r2_comparison(aggregated_results, OUTPUT_DIR)
    plot_contribution_bar(aggregated_results, OUTPUT_DIR)

    # Save summary
    save_component_summary(aggregated_results, OUTPUT_DIR)

    print("\nComponent ablation study complete!")
    print("Generated files:")
    print(f"  - {OUTPUT_DIR}/component_comparison.png    (multi-panel convergence)")
    print(f"  - {OUTPUT_DIR}/component_r2_comparison.png (R2 convergence)")
    print(f"  - {OUTPUT_DIR}/component_contributions.png (contribution breakdown)")
    print(f"  - {OUTPUT_DIR}/component_summary.txt       (summary table)")
    print(f"  - {OUTPUT_DIR}/metrics_tables/             (per-round tables)")


if __name__ == "__main__":
    main()