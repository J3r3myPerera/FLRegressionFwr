"""
Extended Component Ablation Study: Fine-grained SmartFedProx Decomposition

Isolates each individual mechanism of SmartFedProx using a 5-configuration design:

  1. Baseline (FedProx)         — random selection, fixed mu (no components)
  2. Adaptive Client Mu Only    — per-client divergence-scaled mu, no server adaptation
  3. Server Coefficient Only    — loss-based server-side mu adjustment only
  4. Client Selection Only      — hybrid divergence-aware selection, fixed mu
  5. Full SmartFedProx          — all three components combined

Unlike ablation_components.py (which treats client+server mu as one "adaptive mu" component),
this study separates the client-level coefficient (compute_adaptive_mu) from the server-level
coefficient (loss-trend adjustment in FederatedSimulator.run), giving a finer-grained view
of each mechanism's individual contribution.
"""

import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "FLRegression"))
sys.path.insert(0, _SCRIPT_DIR)

from module import (
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS, LEARNING_RATE,
    DEVICE, get_input_dim, _load_and_preprocess_data, reset_data_cache,
    Net,
)
from server import FederatedSimulator
from module import load_centralized_dataset, test, BATCH_SIZE
from client import SimulatedClient
from metrics_table import generate_per_round_table

import wandb


# ---------------------------------------------------------------------------
# Extended simulator — separates client-level vs server-level mu adaptation
# ---------------------------------------------------------------------------

class ExtendedFederatedSimulator(FederatedSimulator):
    """
    Extends FederatedSimulator to independently gate:
      - adaptive_client_mu_enabled : per-client compute_adaptive_mu scaling
      - adaptive_server_mu_enabled : server-side loss-trend mu adjustment
    """

    def run(self, num_rounds: int) -> tuple:
        print(f"\n{'='*60}")
        print(f"Running: {self.strategy_name}")
        print(f"Config: {self.config['description']}")
        print(f"{'='*60}")

        input_dim = get_input_dim()
        global_model = Net(input_dim=input_dim)
        global_state = global_model.state_dict()

        metrics = {
            "rounds": [],
            "r2_scores": [],
            "mse_losses": [],
            "avg_train_loss": [],
            "avg_divergence": [],
            "avg_effective_mu": [],
            "server_mu": [],
        }

        server_mu = self.config["proximal_mu"]
        mu_min = self.config.get("mu_min", 0.001)
        mu_max = self.config.get("mu_max", 1.0)
        prev_loss = None
        best_loss = None

        client_mu_enabled = self.config.get("adaptive_client_mu_enabled", False)
        server_mu_enabled = self.config.get("adaptive_server_mu_enabled", False)

        train_config = {
            "local_epochs": LOCAL_EPOCHS,
            "lr": LEARNING_RATE,
            "proximal_mu": server_mu,
            # client.train() reads this key to decide whether to call compute_adaptive_mu
            "adaptive_mu_enabled": client_mu_enabled,
        }

        global_avg_divergence = 0.0

        for round_num in range(1, num_rounds + 1):
            print(f"\n  Round {round_num}/{num_rounds}")

            selected_ids = self.select_clients(round_num)
            print(f"    Selected clients: {selected_ids}")

            client_results = []
            for client_id in selected_ids:
                result = self.clients[client_id].train(global_state, train_config, global_avg_divergence)
                client_results.append(result)
                self.client_stats[client_id]["divergences"].append(result["divergence"])
                self.client_stats[client_id]["participation"] += 1

            global_state = self.aggregate(client_results)
            loss, r2 = self.evaluate_global(global_state)

            avg_train_loss = np.mean([r["train_loss"] for r in client_results])
            avg_divergence = np.mean([r["divergence"] for r in client_results])
            avg_mu = np.mean([r["effective_mu"] for r in client_results])

            # Server-side mu adjustment — only when server_mu_enabled
            if server_mu_enabled and prev_loss is not None:
                if loss < prev_loss:
                    server_mu = max(server_mu * 0.97, mu_min)
                elif loss > best_loss * 1.05:
                    server_mu = min(server_mu * 1.5, mu_max)
                else:
                    server_mu = min(server_mu * 1.1, mu_max)
                train_config["proximal_mu"] = server_mu

            if best_loss is None or loss < best_loss:
                best_loss = loss
            prev_loss = loss

            if global_avg_divergence == 0:
                global_avg_divergence = avg_divergence
            else:
                global_avg_divergence = 0.7 * global_avg_divergence + 0.3 * avg_divergence

            metrics["rounds"].append(round_num)
            metrics["r2_scores"].append(r2)
            metrics["mse_losses"].append(loss)
            metrics["avg_train_loss"].append(avg_train_loss)
            metrics["avg_divergence"].append(avg_divergence)
            metrics["avg_effective_mu"].append(avg_mu)
            metrics["server_mu"].append(server_mu)

            if wandb.run is not None:
                wandb.log({
                    "r2_score": r2, "mse_loss": loss,
                    "avg_train_loss": avg_train_loss,
                    "avg_divergence": avg_divergence,
                    "avg_effective_mu": avg_mu,
                    "server_mu": server_mu,
                }, step=round_num)

            print(f"    R² = {r2:.4f}, MSE = {loss:.4f}, Avg μ = {avg_mu:.4f}, Server μ = {server_mu:.4f}")

        print(f"\n  Final R²: {metrics['r2_scores'][-1]:.4f}")
        return metrics, global_state


# ---------------------------------------------------------------------------
# Experiment configurations
# ---------------------------------------------------------------------------

EXTENDED_CONFIGS = {
    "Baseline (FedProx)": {
        "proximal_mu": 0.1,
        "adaptive_client_mu_enabled": False,
        "adaptive_server_mu_enabled": False,
        "selection_strategy": "random",
        "description": "FedProx baseline — random selection, fixed mu, no adaptation",
    },
    "Adaptive Client Mu Only": {
        "proximal_mu": 0.1,
        "adaptive_client_mu_enabled": True,
        "adaptive_server_mu_enabled": False,
        "selection_strategy": "random",
        "description": "Client-level adaptive mu only — per-client divergence scaling",
    },
    "Server Coefficient Only": {
        "proximal_mu": 0.1,
        "adaptive_client_mu_enabled": False,
        "adaptive_server_mu_enabled": True,
        "selection_strategy": "random",
        "description": "Server-side mu adaptation only — loss-trend adjustment",
    },
    "Client Selection Only": {
        "proximal_mu": 0.1,
        "adaptive_client_mu_enabled": False,
        "adaptive_server_mu_enabled": False,
        "selection_strategy": "hybrid",
        "description": "Hybrid client selection only — fixed mu, no mu adaptation",
    },
    "Full SmartFedProx": {
        "proximal_mu": 0.1,
        "adaptive_client_mu_enabled": True,
        "adaptive_server_mu_enabled": True,
        "selection_strategy": "hybrid",
        "description": "Full SmartFedProx — all three components combined",
    },
}

OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "testScripts", "results_extended")

COLORS = {
    "Baseline (FedProx)":      "#95a5a6",
    "Adaptive Client Mu Only": "#e67e22",
    "Server Coefficient Only": "#9b59b6",
    "Client Selection Only":   "#3498db",
    "Full SmartFedProx":       "#2ecc71",
}
MARKERS = {
    "Baseline (FedProx)":      "o",
    "Adaptive Client Mu Only": "D",
    "Server Coefficient Only": "P",
    "Client Selection Only":   "s",
    "Full SmartFedProx":       "^",
}


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_multi_panel(aggregated_results: dict, save_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(21, 12))
    fig.suptitle(
        "Extended Component Ablation: Fine-grained SmartFedProx Decomposition\n"
        "(Personal Finance Dataset — Extreme Non-IID)",
        fontsize=14, fontweight='bold',
    )

    metric_specs = [
        ("r2_scores",      "R² Score",        axes[0, 0]),
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
                    linewidth=2, markersize=6, label=name, markevery=2)
            if stds is not None:
                v = np.array(values)
                s = np.array(stds)
                ax.fill_between(rounds, v - s, v + s, color=COLORS[name], alpha=0.1)

        ax.set_xlabel("Federated Round", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(ylabel, fontsize=12, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "extended_multi_panel.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved multi-panel plot -> {path}")


def plot_r2_convergence(aggregated_results: dict, save_dir: str):
    fig, ax = plt.subplots(figsize=(11, 6))

    for name, data in aggregated_results.items():
        rounds = data["rounds"]
        values = data["r2_scores"]
        stds = data.get("r2_scores_std")

        ax.plot(rounds, values,
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2.5, markersize=8, label=name, markevery=2)
        if stds is not None:
            v = np.array(values)
            s = np.array(stds)
            ax.fill_between(rounds, v - s, v + s, color=COLORS[name], alpha=0.12)
        ax.annotate(f'{values[-1]:.4f}',
                    xy=(rounds[-1], values[-1]),
                    xytext=(5, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=COLORS[name])

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("Extended Ablation: R² Convergence by Component", fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "extended_r2_convergence.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved R² convergence plot -> {path}")


def plot_server_mu_trajectory(aggregated_results: dict, save_dir: str):
    """Show how server mu evolves for configs that have server adaptation."""
    fig, ax = plt.subplots(figsize=(10, 5))

    configs_with_server_mu = ["Server Coefficient Only", "Full SmartFedProx"]
    for name in configs_with_server_mu:
        data = aggregated_results[name]
        rounds = data["rounds"]
        values = data["server_mu"]
        ax.plot(rounds, values,
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2.5, markersize=7, label=name, markevery=2)

    # Baseline reference line
    baseline_mu = EXTENDED_CONFIGS["Baseline (FedProx)"]["proximal_mu"]
    ax.axhline(y=baseline_mu, color=COLORS["Baseline (FedProx)"],
               linestyle='--', linewidth=1.5, label=f"Fixed mu = {baseline_mu}")

    ax.set_xlabel("Federated Round", fontsize=12)
    ax.set_ylabel("Server Proximal μ", fontsize=12)
    ax.set_title("Server Coefficient Trajectory (Loss-Trend Adaptation)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "extended_server_mu_trajectory.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved server mu trajectory -> {path}")


def plot_contribution_bars(aggregated_results: dict, save_dir: str):
    """Bar charts: final R2 per config + individual component gain breakdown."""
    names = list(EXTENDED_CONFIGS.keys())
    final_r2 = [aggregated_results[n]["avg_final_r2"] for n in names]
    final_std = [aggregated_results[n]["std_final_r2"] for n in names]

    baseline = aggregated_results["Baseline (FedProx)"]["avg_final_r2"]
    client_mu_gain = aggregated_results["Adaptive Client Mu Only"]["avg_final_r2"] - baseline
    server_mu_gain = aggregated_results["Server Coefficient Only"]["avg_final_r2"] - baseline
    selection_gain = aggregated_results["Client Selection Only"]["avg_final_r2"] - baseline
    full_gain      = aggregated_results["Full SmartFedProx"]["avg_final_r2"] - baseline
    # Interaction = gain not explained by individual components alone
    interaction = full_gain - client_mu_gain - server_mu_gain - selection_gain

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel 1: Final R2 per config
    ax = axes[0]
    x = np.arange(len(names))
    bar_colors = [COLORS[n] for n in names]
    bars = ax.bar(x, final_r2, yerr=final_std, capsize=5,
                  color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, val, std in zip(bars, final_r2, final_std):
        ax.annotate(f'{val:.4f}\n±{std:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 8), textcoords='offset points',
                    ha='center', fontsize=8.5, fontweight='bold')
    ax.set_ylabel("Final R² Score", fontsize=12)
    ax.set_title("Final R² by Configuration", fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=20, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 2: Individual component gains over baseline
    ax = axes[1]
    component_names = [
        "Adaptive\nClient Mu",
        "Server\nCoefficient",
        "Client\nSelection",
        "Interaction\n(synergy)",
    ]
    gains = [client_mu_gain, server_mu_gain, selection_gain, interaction]
    component_colors = [
        COLORS["Adaptive Client Mu Only"],
        COLORS["Server Coefficient Only"],
        COLORS["Client Selection Only"],
        "#c0392b",
    ]

    bars = ax.bar(range(len(gains)), gains,
                  color=component_colors, alpha=0.85, edgecolor='black', linewidth=0.5)
    for bar, val in zip(bars, gains):
        sign = "+" if val >= 0 else ""
        y_pos = bar.get_height() if val >= 0 else bar.get_y()
        offset = 8 if val >= 0 else -15
        ax.annotate(f'{sign}{val:.4f}',
                    xy=(bar.get_x() + bar.get_width() / 2, y_pos),
                    xytext=(0, offset), textcoords='offset points',
                    ha='center', fontsize=11, fontweight='bold')

    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.set_ylabel(f"R² Gain over Baseline ({baseline:.4f})", fontsize=12)
    ax.set_title("Individual Component Contributions", fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(gains)))
    ax.set_xticklabels(component_names, fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    path = os.path.join(save_dir, "extended_contributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved contribution breakdown -> {path}")

    return baseline, client_mu_gain, server_mu_gain, selection_gain, interaction, full_gain


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def _contribution_block(baseline, client_mu_gain, server_mu_gain, selection_gain, interaction, full_gain):
    lines = [
        f"\nCONTRIBUTION ANALYSIS (R² gain over baseline {baseline:.4f}):",
        f"  Adaptive client mu (alone):    {client_mu_gain:+.4f}",
        f"  Server coefficient (alone):    {server_mu_gain:+.4f}",
        f"  Client selection (alone):      {selection_gain:+.4f}",
        f"  Interaction (synergy/cancel):  {interaction:+.4f}",
        f"  Total (Full SmartFedProx):     {full_gain:+.4f}",
    ]
    ranked = sorted(
        [("Adaptive client mu", client_mu_gain),
         ("Server coefficient", server_mu_gain),
         ("Client selection",   selection_gain)],
        key=lambda x: abs(x[1]), reverse=True,
    )
    lines.append(f"\n  Ranking by absolute R² contribution:")
    for rank, (comp, gain) in enumerate(ranked, 1):
        lines.append(f"    {rank}. {comp}: {gain:+.4f}")
    return lines


def print_extended_summary(aggregated_results: dict):
    print("\n" + "=" * 120)
    print("EXTENDED COMPONENT ABLATION: ADAPTIVE CLIENT MU vs SERVER COEFFICIENT vs CLIENT SELECTION")
    print("=" * 120)
    header = (f"{'Configuration':<28} {'Selection':<10} {'Client Mu':<12} {'Server Mu':<12} "
              f"{'Final R2':>10} {'Std':>8} {'Final MSE':>11} {'Final RMSE':>12} {'Best R2':>10}")
    print(header)
    print("-" * 120)

    meta = {
        "Baseline (FedProx)":      ("Random",  "Fixed",    "Fixed"),
        "Adaptive Client Mu Only": ("Random",  "Adaptive", "Fixed"),
        "Server Coefficient Only": ("Random",  "Fixed",    "Adaptive"),
        "Client Selection Only":   ("Hybrid",  "Fixed",    "Fixed"),
        "Full SmartFedProx":       ("Hybrid",  "Adaptive", "Adaptive"),
    }

    for name, data in aggregated_results.items():
        sel, cmu, smu = meta[name]
        print(f"{name:<28} {sel:<10} {cmu:<12} {smu:<12} "
              f"{data['avg_final_r2']:>10.4f} {data['std_final_r2']:>8.4f} "
              f"{data['avg_final_mse']:>11.4f} {data['avg_final_rmse']:>12.4f} "
              f"{data['avg_best_r2']:>10.4f}")

    print("-" * 120)

    baseline    = aggregated_results["Baseline (FedProx)"]["avg_final_r2"]
    client_gain = aggregated_results["Adaptive Client Mu Only"]["avg_final_r2"] - baseline
    server_gain = aggregated_results["Server Coefficient Only"]["avg_final_r2"] - baseline
    select_gain = aggregated_results["Client Selection Only"]["avg_final_r2"] - baseline
    full_gain   = aggregated_results["Full SmartFedProx"]["avg_final_r2"] - baseline
    interaction = full_gain - client_gain - server_gain - select_gain

    for line in _contribution_block(baseline, client_gain, server_gain, select_gain, interaction, full_gain):
        print(line)
    print("=" * 120)


def save_extended_summary(aggregated_results: dict, save_dir: str):
    baseline    = aggregated_results["Baseline (FedProx)"]["avg_final_r2"]
    client_gain = aggregated_results["Adaptive Client Mu Only"]["avg_final_r2"] - baseline
    server_gain = aggregated_results["Server Coefficient Only"]["avg_final_r2"] - baseline
    select_gain = aggregated_results["Client Selection Only"]["avg_final_r2"] - baseline
    full_gain   = aggregated_results["Full SmartFedProx"]["avg_final_r2"] - baseline
    interaction = full_gain - client_gain - server_gain - select_gain

    meta = {
        "Baseline (FedProx)":      ("Random",  "Fixed",    "Fixed"),
        "Adaptive Client Mu Only": ("Random",  "Adaptive", "Fixed"),
        "Server Coefficient Only": ("Random",  "Fixed",    "Adaptive"),
        "Client Selection Only":   ("Hybrid",  "Fixed",    "Fixed"),
        "Full SmartFedProx":       ("Hybrid",  "Adaptive", "Adaptive"),
    }

    lines = ["=" * 120,
             "EXTENDED COMPONENT ABLATION: ADAPTIVE CLIENT MU vs SERVER COEFFICIENT vs CLIENT SELECTION",
             "=" * 120]
    lines.append(f"{'Configuration':<28} {'Selection':<10} {'Client Mu':<12} {'Server Mu':<12} "
                 f"{'Final R2':>10} {'Std':>8} {'Final MSE':>11} {'Final RMSE':>12} {'Best R2':>10}")
    lines.append("-" * 120)

    for name, data in aggregated_results.items():
        sel, cmu, smu = meta[name]
        lines.append(f"{name:<28} {sel:<10} {cmu:<12} {smu:<12} "
                     f"{data['avg_final_r2']:>10.4f} {data['std_final_r2']:>8.4f} "
                     f"{data['avg_final_mse']:>11.4f} {data['avg_final_rmse']:>12.4f} "
                     f"{data['avg_best_r2']:>10.4f}")
    lines.append("-" * 120)
    lines.extend(_contribution_block(baseline, client_gain, server_gain, select_gain, interaction, full_gain))
    lines.append("=" * 120)

    path = os.path.join(save_dir, "extended_summary.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Saved summary -> {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    NUM_TRIALS = 3
    FIXED_SEED = 2023

    print("\n" + "=" * 70)
    print("EXTENDED COMPONENT ABLATION")
    print("Configurations: Baseline | Client Mu | Server Coeff | Selection | Full")
    print("Dataset: Indian Personal Finance (Disposable Income Prediction)")
    print("Non-IID: EXTREME (Occupation + City_Tier + Income stratification)")
    print(f"Device:  {DEVICE}")
    print(f"Clients: {NUM_CLIENTS}, Fraction Fit: {FRACTION_FIT}")
    print(f"Rounds:  {NUM_ROUNDS}, Local Epochs: {LOCAL_EPOCHS}")
    print(f"Trials:  {NUM_TRIALS}")
    print("=" * 70)

    print("\nResetting data cache and loading dataset...")
    reset_data_cache()
    _load_and_preprocess_data()
    print(f"Input dimension: {get_input_dim()}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_trial_results = {name: [] for name in EXTENDED_CONFIGS}

    for trial in range(NUM_TRIALS):
        print(f"\n{'#' * 70}")
        print(f"# TRIAL {trial + 1}/{NUM_TRIALS}")
        print(f"{'#' * 70}")

        trial_seed = FIXED_SEED + trial * 100

        for config_name, config in EXTENDED_CONFIGS.items():
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)

            simulator = ExtendedFederatedSimulator(config_name, config)
            metrics, _model_state = simulator.run(NUM_ROUNDS)

            metrics["rmse_scores"] = [mse ** 0.5 for mse in metrics["mse_losses"]]
            metrics["mae_scores"]  = [rmse * 0.7979 for rmse in metrics["rmse_scores"]]

            all_trial_results[config_name].append(metrics)

    # Aggregate across trials
    print("\n" + "=" * 70)
    print("AGGREGATING RESULTS")
    print("=" * 70)

    aggregated_results = {}
    metric_keys = ["r2_scores", "mse_losses", "rmse_scores", "mae_scores",
                   "avg_train_loss", "avg_divergence", "avg_effective_mu", "server_mu"]

    for config_name in EXTENDED_CONFIGS:
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
            agg[key]              = [np.mean([t[key][i] for t in trials]) for i in range(NUM_ROUNDS)]
            agg[f"{key}_std"]     = [np.std([t[key][i] for t in trials]) for i in range(NUM_ROUNDS)]

        aggregated_results[config_name] = agg
        print(f"{config_name}:")
        print(f"  Final R2: {agg['avg_final_r2']:.4f} +/- {agg['std_final_r2']:.4f}")
        print(f"  Final MSE: {agg['avg_final_mse']:.4f}  Best R2: {agg['avg_best_r2']:.4f}")

    print_extended_summary(aggregated_results)

    # Per-round tables
    print("\nGenerating per-round metrics tables...")
    safe_trial_results = {}
    for name, trials in all_trial_results.items():
        safe_name = name.replace("+", "plus").replace(" ", "_").replace("(", "").replace(")", "")
        safe_trial_results[safe_name] = trials
    generate_per_round_table(
        all_trial_results=safe_trial_results,
        num_rounds=NUM_ROUNDS,
        output_dir=os.path.join(OUTPUT_DIR, "metrics_tables"),
        save_txt=True,
        save_latex=True,
    )

    # Plots
    print("\nGenerating plots...")
    plot_multi_panel(aggregated_results, OUTPUT_DIR)
    plot_r2_convergence(aggregated_results, OUTPUT_DIR)
    plot_server_mu_trajectory(aggregated_results, OUTPUT_DIR)
    plot_contribution_bars(aggregated_results, OUTPUT_DIR)

    save_extended_summary(aggregated_results, OUTPUT_DIR)

    print("\nExtended ablation study complete!")
    print("Generated files:")
    print(f"  - {OUTPUT_DIR}/extended_multi_panel.png         (6-metric convergence)")
    print(f"  - {OUTPUT_DIR}/extended_r2_convergence.png      (R² convergence)")
    print(f"  - {OUTPUT_DIR}/extended_server_mu_trajectory.png (server mu over rounds)")
    print(f"  - {OUTPUT_DIR}/extended_contributions.png        (final R² + gain breakdown)")
    print(f"  - {OUTPUT_DIR}/extended_summary.txt              (summary table)")
    print(f"  - {OUTPUT_DIR}/metrics_tables/                   (per-round tables)")


if __name__ == "__main__":
    main()
