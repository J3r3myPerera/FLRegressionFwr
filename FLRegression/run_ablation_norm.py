"""
Ablation study: Normalization layer comparison in Federated Learning.

Compares BatchNorm, LayerNorm, and GroupNorm under SmartFedProx
to evaluate the impact of normalization choice on non-IID federated training.
Saves results to outputs/ablation_norm/.
"""

import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Ensure FLRegression is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import module
import client
import server
from module import (
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS,
    DEVICE, get_input_dim, _load_and_preprocess_data,
    reset_data_cache,
)
from server import FederatedSimulator


OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "outputs", "ablation_norm"
)

NUM_TRIALS = 3
FIXED_SEED = 2023

# -------------------- Model variants --------------------

class NetBatchNorm(nn.Module):
    """MLP with BatchNorm1d."""

    def __init__(self, input_dim: int = 26):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        return self.fc4(x)


class NetLayerNorm(nn.Module):
    """MLP with LayerNorm."""

    def __init__(self, input_dim: int = 26):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.ln2 = nn.LayerNorm(64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 32)
        self.ln3 = nn.LayerNorm(32)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.ln1(self.fc1(x))))
        x = self.dropout2(F.relu(self.ln2(self.fc2(x))))
        x = F.relu(self.ln3(self.fc3(x)))
        return self.fc4(x)


class NetGroupNorm(nn.Module):
    """MLP with GroupNorm (8 groups)."""

    def __init__(self, input_dim: int = 26):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.gn1 = nn.GroupNorm(8, 128)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(128, 64)
        self.gn2 = nn.GroupNorm(8, 64)
        self.dropout2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(64, 32)
        self.gn3 = nn.GroupNorm(8, 32)

        self.fc4 = nn.Linear(32, 1)

    def forward(self, x):
        x = self.dropout1(F.relu(self.gn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.gn2(self.fc2(x))))
        x = F.relu(self.gn3(self.fc3(x)))
        return self.fc4(x)


# Map config names to model classes
NORM_MODELS = {
    "BatchNorm": NetBatchNorm,
    "LayerNorm": NetLayerNorm,
    "GroupNorm": NetGroupNorm,
}

# All configs use SmartFedProx to isolate normalization as the only variable
ABLATION_CONFIGS = {
    name: {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "hybrid",
        "description": f"SmartFedProx with {name}",
    }
    for name in NORM_MODELS
}


def _set_net_class(cls):
    """Monkey-patch Net in all modules so FederatedSimulator uses the given variant."""
    module.Net = cls
    client.Net = cls
    server.Net = cls


# ----------------------------- output helpers -----------------------------------

def save_metrics_tables(all_trial_results: dict, num_rounds: int):
    """Save per-round .txt and .tex tables for every configuration."""
    tables_dir = os.path.join(OUTPUT_DIR, "metrics_tables")
    os.makedirs(tables_dir, exist_ok=True)

    col = 24
    row_sep = "+" + "-" * 7 + ("+" + "-" * col) * 6 + "+"
    head_sep = "+" + "=" * 7 + ("+" + "=" * col) * 6 + "+"

    for config_name, trials in all_trial_results.items():
        num_trials = len(trials)

        # ---- .txt file ----
        txt_path = os.path.join(tables_dir, f"table_{config_name}.txt")
        with open(txt_path, "w") as f:
            f.write(
                f"Normalization: {config_name}  |  Per-Round Metrics "
                f"(mean, min-max across {num_trials} trial(s))\n"
            )
            f.write("=" * (7 + col * 6 + 7) + "\n")
            f.write(row_sep + "\n")
            f.write(
                f"| {'Round':^5} |"
                f" {'R² (min-max)':^{col-2}} |"
                f" {'MSE (min-max)':^{col-2}} |"
                f" {'RMSE (min-max)':^{col-2}} |"
                f" {'MAE (min-max)':^{col-2}} |"
                f" {'Train Loss (min-max)':^{col-2}} |"
                f" {'Avg μ (min-max)':^{col-2}} |\n"
            )
            f.write(head_sep + "\n")

            for i in range(num_rounds):
                round_num = trials[0]["rounds"][i]

                def fmt(key):
                    vals = [t[key][i] for t in trials]
                    mean, lo, hi = np.mean(vals), min(vals), max(vals)
                    return f"{mean:.4f} ({lo:.4f}-{hi:.4f})"

                f.write(
                    f"| {round_num:^5} |"
                    f" {fmt('r2_scores'):^{col-2}} |"
                    f" {fmt('mse_losses'):^{col-2}} |"
                    f" {fmt('rmse_scores'):^{col-2}} |"
                    f" {fmt('mae_scores'):^{col-2}} |"
                    f" {fmt('avg_train_loss'):^{col-2}} |"
                    f" {fmt('avg_effective_mu'):^{col-2}} |\n"
                )
                f.write(row_sep + "\n")

        # ---- .tex file ----
        tex_path = os.path.join(tables_dir, f"table_{config_name}.tex")
        label_safe = config_name.lower().replace("-", "_").replace(" ", "_")
        with open(tex_path, "w") as f:
            f.write("% Auto-generated by run_ablation_norm.py\n")
            f.write("\\begin{table}[ht]\n\\centering\n")
            f.write(
                f"\\caption{{Per-Round Metrics — {config_name} "
                f"(mean, min--max across {num_trials} trial(s))}}\n"
            )
            f.write(f"\\label{{tab:norm_{label_safe}}}\n")
            f.write("\\begin{tabular}{|c|c|c|c|c|c|c|}\n\\hline\n")
            f.write(
                "\\textbf{Round} & \\textbf{R² (min-max)} & "
                "\\textbf{MSE (min-max)} & \\textbf{RMSE (min-max)} & "
                "\\textbf{MAE (min-max)} & \\textbf{Train Loss (min-max)} & "
                "\\textbf{Avg μ (min-max)} \\\\\n\\hline\n"
            )

            for i in range(num_rounds):
                round_num = trials[0]["rounds"][i]

                def tex_fmt(key):
                    vals = [t[key][i] for t in trials]
                    mean, lo, hi = np.mean(vals), min(vals), max(vals)
                    return f"{mean:.4f} ({lo:.4f}--{hi:.4f})"

                f.write(
                    f"{round_num} & {tex_fmt('r2_scores')} & "
                    f"{tex_fmt('mse_losses')} & {tex_fmt('rmse_scores')} & "
                    f"{tex_fmt('mae_scores')} & {tex_fmt('avg_train_loss')} & "
                    f"{tex_fmt('avg_effective_mu')} \\\\\n\\hline\n"
                )

            f.write("\\end{tabular}\n\\end{table}\n")

        print(f"  - {config_name}  (.txt + .tex)")


def save_summary(aggregated: dict):
    """Write ablation_norm_summary.txt."""
    path = os.path.join(OUTPUT_DIR, "ablation_norm_summary.txt")
    width = 100

    best_name = max(aggregated, key=lambda k: aggregated[k]["avg_final_r2"])

    with open(path, "w") as f:
        f.write("=" * width + "\n")
        f.write("ABLATION STUDY: NORMALIZATION LAYER COMPARISON\n")
        f.write("Strategy: SmartFedProx (adaptive mu, hybrid selection)\n")
        f.write("=" * width + "\n")
        f.write(
            f"{'Normalization':<20} {'Final R2':>10} {'Std':>8} "
            f"{'Final MSE':>11} {'Final RMSE':>12} {'Best R2':>10}\n"
        )
        f.write("-" * width + "\n")

        for name, agg in aggregated.items():
            marker = " << BEST" if name == best_name else ""
            f.write(
                f"{name:<20} "
                f"{agg['avg_final_r2']:>10.4f} "
                f"{agg['std_final_r2']:>8.4f} "
                f"{agg['avg_final_mse']:>11.4f} "
                f"{agg['avg_final_rmse']:>12.4f} "
                f"{agg['avg_best_r2']:>10.4f}"
                f"{marker}\n"
            )

        f.write("-" * width + "\n\n")
        f.write(f"Best performing normalization: {best_name}\n")
        f.write(
            f"  R2 = {aggregated[best_name]['avg_final_r2']:.4f} "
            f"+/- {aggregated[best_name]['std_final_r2']:.4f}\n"
        )
        f.write("=" * width + "\n")

    print(f"\n  Summary saved to {path}")


# ----------------------------- plotting -----------------------------------

COLORS = {"BatchNorm": "#e74c3c", "LayerNorm": "#3498db", "GroupNorm": "#2ecc71"}
MARKERS = {"BatchNorm": "o", "LayerNorm": "s", "GroupNorm": "^"}


def plot_r2_comparison(aggregated: dict):
    """R² progression for all normalization variants."""
    fig, ax = plt.subplots(figsize=(12, 7))

    for name, agg in aggregated.items():
        ax.plot(
            agg["rounds"], agg["r2_scores"],
            color=COLORS[name], marker=MARKERS[name],
            linewidth=2, markersize=6, label=name,
        )
        final = agg["r2_scores"][-1]
        ax.annotate(
            f"{final:.4f}",
            xy=(agg["rounds"][-1], final),
            xytext=(5, 0), textcoords="offset points",
            fontsize=9, fontweight="bold",
        )

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title(
        "Ablation: R² Score by Normalization Layer\n(SmartFedProx, Extreme Non-IID)",
        fontsize=14, fontweight="bold",
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "norm_r2_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  norm_r2_comparison.png saved")


def plot_comprehensive(aggregated: dict):
    """2x3 comprehensive comparison grid."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Ablation Study: Normalization Layer Comparison\n"
        "(SmartFedProx — Adaptive mu + Hybrid Selection, Extreme Non-IID)",
        fontsize=14, fontweight="bold",
    )

    # Panel 1 - R²
    ax = axes[0, 0]
    for name, agg in aggregated.items():
        ax.plot(agg["rounds"], agg["r2_scores"],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=5, label=name)
    ax.set_xlabel("Round"); ax.set_ylabel("R²")
    ax.set_title("R² Score Progression"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 2 - MSE
    ax = axes[0, 1]
    for name, agg in aggregated.items():
        ax.plot(agg["rounds"], agg["mse_losses"],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=5, label=name)
    ax.set_xlabel("Round"); ax.set_ylabel("MSE")
    ax.set_title("MSE Loss Progression"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 3 - Training Loss
    ax = axes[0, 2]
    for name, agg in aggregated.items():
        ax.plot(agg["rounds"], agg["avg_train_loss"],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=5, label=name)
    ax.set_xlabel("Round"); ax.set_ylabel("Avg Train Loss")
    ax.set_title("Average Training Loss"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 4 - Divergence
    ax = axes[1, 0]
    for name, agg in aggregated.items():
        ax.plot(agg["rounds"], agg["avg_divergence"],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=5, label=name)
    ax.set_xlabel("Round"); ax.set_ylabel("Avg Divergence")
    ax.set_title("Model Divergence"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 5 - Effective mu
    ax = axes[1, 1]
    for name, agg in aggregated.items():
        ax.plot(agg["rounds"], agg["avg_effective_mu"],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=5, label=name)
    ax.set_xlabel("Round"); ax.set_ylabel("Avg Effective μ")
    ax.set_title("Average Proximal Coefficient (μ)"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 6 - Final R² bar chart with error bars
    ax = axes[1, 2]
    names = list(aggregated.keys())
    final_r2 = [aggregated[n]["avg_final_r2"] for n in names]
    stds = [aggregated[n]["std_final_r2"] for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, final_r2, yerr=stds, capsize=6,
                  color=[COLORS[n] for n in names], alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.set_ylabel("Final R²")
    ax.set_title("Final R² Comparison")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val, std in zip(bars, final_r2, stds):
        ax.annotate(
            f"{val:.4f}\n+/-{std:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=9, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "norm_comprehensive_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  norm_comprehensive_comparison.png saved")


def plot_final_bar(aggregated: dict):
    """Standalone final R² bar chart."""
    fig, ax = plt.subplots(figsize=(9, 6))

    names = list(aggregated.keys())
    final_r2 = [aggregated[n]["avg_final_r2"] for n in names]
    stds = [aggregated[n]["std_final_r2"] for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, final_r2, yerr=stds, capsize=6,
                  color=[COLORS[n] for n in names], alpha=0.85,
                  edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel("Final R² Score", fontsize=12)
    ax.set_title(
        "Ablation: Final R² by Normalization Layer\n(SmartFedProx, 3 Trials)",
        fontsize=14, fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="y")

    for bar, val, std in zip(bars, final_r2, stds):
        ax.annotate(
            f"{val:.4f}\n+/-{std:.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 6), textcoords="offset points",
            ha="center", fontsize=10, fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "norm_final_r2_bar.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  norm_final_r2_bar.png saved")


# -------------------------------- main -----------------------------------

def main():
    print("\n" + "=" * 70)
    print("ABLATION STUDY: NORMALIZATION LAYER COMPARISON")
    print("Variants: BatchNorm vs LayerNorm vs GroupNorm")
    print("Strategy: SmartFedProx (adaptive mu, hybrid selection)")
    print(f"Device: {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}, Fraction Fit: {FRACTION_FIT}")
    print(f"Rounds: {NUM_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
    print(f"Trials: {NUM_TRIALS}, Seed: {FIXED_SEED}")
    print("=" * 70)

    # Prepare data
    print("\nResetting data cache and loading with EXTREME non-IID partitioning...")
    reset_data_cache()
    _load_and_preprocess_data()
    print(f"Input dimension: {get_input_dim()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Storage per config
    all_trial_results = {name: [] for name in NORM_MODELS}

    base_seed = FIXED_SEED

    for trial in range(NUM_TRIALS):
        print(f"\n{'#' * 70}")
        print(f"# TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'#' * 70}")

        trial_seed = base_seed + trial * 100

        for norm_name, net_cls in NORM_MODELS.items():
            # Seed identically for fair comparison
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)

            # Swap the Net class used by the entire pipeline
            _set_net_class(net_cls)

            config = ABLATION_CONFIGS[norm_name]
            simulator = FederatedSimulator(norm_name, config)
            metrics = simulator.run(NUM_ROUNDS)
            all_trial_results[norm_name].append(metrics)

    # ---- aggregate across trials ----
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    aggregated = {}
    for norm_name in NORM_MODELS:
        trials = all_trial_results[norm_name]
        avg_final_r2 = np.mean([t["r2_scores"][-1] for t in trials])
        std_final_r2 = np.std([t["r2_scores"][-1] for t in trials])
        avg_final_mse = np.mean([t["mse_losses"][-1] for t in trials])
        avg_final_rmse = np.mean([t["rmse_scores"][-1] for t in trials])
        avg_best_r2 = np.mean([max(t["r2_scores"]) for t in trials])

        aggregated[norm_name] = {
            "avg_final_r2": avg_final_r2,
            "std_final_r2": std_final_r2,
            "avg_final_mse": avg_final_mse,
            "avg_final_rmse": avg_final_rmse,
            "avg_best_r2": avg_best_r2,
            "rounds": trials[0]["rounds"],
            "r2_scores": [np.mean([t["r2_scores"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "mse_losses": [np.mean([t["mse_losses"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "rmse_scores": [np.mean([t["rmse_scores"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "avg_train_loss": [np.mean([t["avg_train_loss"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "avg_divergence": [np.mean([t["avg_divergence"][i] for t in trials]) for i in range(NUM_ROUNDS)],
            "avg_effective_mu": [np.mean([t["avg_effective_mu"][i] for t in trials]) for i in range(NUM_ROUNDS)],
        }

        print(f"{norm_name}:")
        print(f"  Final R²: {avg_final_r2:.4f} +/- {std_final_r2:.4f}")
        print(f"  Final MSE: {avg_final_mse:.4f}")
        print(f"  Best R² (avg): {avg_best_r2:.4f}")

    best = max(aggregated, key=lambda k: aggregated[k]["avg_final_r2"])
    print(f"\nBest performing normalization: {best} (R² = {aggregated[best]['avg_final_r2']:.4f})")

    # ---- save outputs ----
    print("\nSaving outputs...")
    save_summary(aggregated)
    save_metrics_tables(all_trial_results, NUM_ROUNDS)
    plot_r2_comparison(aggregated)
    plot_comprehensive(aggregated)
    plot_final_bar(aggregated)

    print("\nAll ablation outputs saved to:", OUTPUT_DIR)
    print("  - ablation_norm_summary.txt")
    print("  - norm_r2_comparison.png")
    print("  - norm_comprehensive_comparison.png")
    print("  - norm_final_r2_bar.png")
    print("  - metrics_tables/  (per-config .txt + .tex)")


if __name__ == "__main__":
    main()
