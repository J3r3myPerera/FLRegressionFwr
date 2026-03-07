"""
Federated vs Centralized Training Comparison.

Trains two centralized baselines and compares against FedAvg, FedProx, and
SmartFedProx under the same compute budget:

1. Centralized (IID)     — all data pooled, homogeneous distribution
2. Centralized (Non-IID) — only the same non-IID partitioned data the
                           federated clients see, concatenated and trained
                           sequentially partition-by-partition

Usage:
    cd FLRegressionFlwr/FLRegression
    python centralized_vs_federated.py
"""

import time
import math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from module import (
    Net, test, get_input_dim,
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS,
    LEARNING_RATE, BATCH_SIZE, DEVICE, STRATEGIES,
    reset_data_cache, _load_and_preprocess_data,
)
from dataset import load_data, load_centralized_dataset
from server import FederatedSimulator

# Output directory
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


# ---------------------------------------------------------------------------
# Centralized Training
# ---------------------------------------------------------------------------

def train_centralized(num_epochs, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=2023):
    """Train on ALL data combined (no federation).

    Uses the same Net architecture, optimizer, and test set as the federated
    approaches so the comparison is fair.

    Returns a metrics dict with per-epoch R2 and MSE on the centralized test set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_cache, preprocessors = _load_and_preprocess_data()
    X = data_cache["X"]
    y = data_cache["y"]

    # 80/20 split – same random_state as load_centralized_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_dataset, batch_size=128)

    input_dim = preprocessors["input_dim"]
    model = Net(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    metrics = {
        "epochs": [],
        "r2_scores": [],
        "mse_losses": [],
        "train_losses": [],
    }

    print(f"\n{'='*60}")
    print("Running: Centralized (all data, no federation)")
    print(f"Epochs: {num_epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for features, targets in trainloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(num_batches, 1)

        # Evaluate on test set
        mse, r2, _rmse, _mae = test(model, testloader, DEVICE)

        metrics["epochs"].append(epoch)
        metrics["r2_scores"].append(r2)
        metrics["mse_losses"].append(mse)
        metrics["train_losses"].append(avg_train_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  Epoch {epoch}/{num_epochs}  R2={r2:.4f}  MSE={mse:.4f}  TrainLoss={avg_train_loss:.4f}")

    print(f"\n  Final R2: {metrics['r2_scores'][-1]:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Centralized Training on Non-IID Partitioned Data (Sequential)
# ---------------------------------------------------------------------------

def train_centralized_noniid(num_epochs, lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=2023):
    """Train centrally on the SAME non-IID partitioned data the federated clients use.

    Collects every client's local training data, concatenates it into one pool,
    then trains a single model on it.  The data is still heterogeneous (biased
    by occupation, city tier, income) — the only difference vs federated is that
    there is no communication overhead or aggregation; one model sees everything
    in sequence.

    Each epoch cycles through all client partitions sequentially (partition 0,
    then partition 1, ..., partition N-1) to simulate the effect of
    heterogeneous data ordering — a key stress test for centralized training on
    non-IID sources.

    Returns a metrics dict with per-epoch R2 and MSE on the centralized test set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Collect all client train data (same partitions used in federated)
    all_X_train = []
    all_y_train = []
    partition_boundaries = []  # track where each partition starts/ends

    for cid in range(NUM_CLIENTS):
        trainloader, _ = load_data(cid, NUM_CLIENTS, batch_size)
        for features, targets in trainloader:
            all_X_train.append(features)
            all_y_train.append(targets)
        partition_boundaries.append(len(all_X_train))

    all_X = torch.cat(all_X_train, dim=0)
    all_y = torch.cat(all_y_train, dim=0)

    total_samples = all_X.shape[0]

    # Build a loader that preserves partition ordering (sequential non-IID)
    # No shuffle — data arrives partition-by-partition each epoch
    noniid_dataset = TensorDataset(all_X, all_y)
    noniid_loader = DataLoader(noniid_dataset, batch_size=batch_size, shuffle=False)

    # Centralized test set (same as federated global eval)
    testloader = load_centralized_dataset()

    input_dim = get_input_dim()
    model = Net(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    metrics = {
        "epochs": [],
        "r2_scores": [],
        "mse_losses": [],
        "train_losses": [],
    }

    print(f"\n{'='*60}")
    print("Running: Centralized (Non-IID — sequential client partitions)")
    print(f"Epochs: {num_epochs}, LR: {lr}, Batch: {batch_size}")
    print(f"Total train samples (from {NUM_CLIENTS} client partitions): {total_samples}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0

        for features, targets in noniid_loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(num_batches, 1)

        # Evaluate on the same centralized test set
        mse, r2, _rmse, _mae = test(model, testloader, DEVICE)

        metrics["epochs"].append(epoch)
        metrics["r2_scores"].append(r2)
        metrics["mse_losses"].append(mse)
        metrics["train_losses"].append(avg_train_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  Epoch {epoch}/{num_epochs}  R2={r2:.4f}  MSE={mse:.4f}  TrainLoss={avg_train_loss:.4f}")

    print(f"\n  Final R2: {metrics['r2_scores'][-1]:.4f}")
    return metrics


# ---------------------------------------------------------------------------
# Run Federated Strategies
# ---------------------------------------------------------------------------

def run_federated_strategies(seed=2023):
    """Run all 3 federated strategies and return their metrics."""
    all_results = {}

    for strategy_name, config in STRATEGIES.items():
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        simulator = FederatedSimulator(strategy_name, config)
        metrics = simulator.run(NUM_ROUNDS)
        all_results[strategy_name] = metrics

    return all_results


# ---------------------------------------------------------------------------
# Compute effective computation units
# ---------------------------------------------------------------------------

def compute_effective_units_federated(num_rounds, local_epochs, fraction_fit):
    """Each federated round does local_epochs * fraction_fit effective work."""
    per_round = local_epochs * fraction_fit
    return [per_round * r for r in range(1, num_rounds + 1)]


def compute_effective_units_centralized(num_epochs):
    """Each centralized epoch = 1 effective computation unit."""
    return list(range(1, num_epochs + 1))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

COLORS = {
    "Centralized": "#f39c12",
    "Centralized (Non-IID)": "#9b59b6",
    "FedAvg": "#e74c3c",
    "FedProx": "#3498db",
    "SmartFedProx": "#2ecc71",
}
MARKERS = {
    "Centralized": "D",
    "Centralized (Non-IID)": "P",
    "FedAvg": "o",
    "FedProx": "s",
    "SmartFedProx": "^",
}


def _plot_metric_vs_compute(ax, centralized, federated_all, metric_key,
                            ylabel, title, fed_compute_units, cent_compute_units,
                            centralized_noniid=None, cent_noniid_cu=None):
    """Helper: plot one metric for all approaches against effective compute."""
    # Centralized (IID)
    ax.plot(cent_compute_units, centralized[metric_key],
            color=COLORS["Centralized"], marker=MARKERS["Centralized"],
            linewidth=2.5, markersize=7, label="Centralized",
            markevery=max(1, len(cent_compute_units)//12))

    # Centralized (Non-IID)
    if centralized_noniid is not None and cent_noniid_cu is not None:
        ax.plot(cent_noniid_cu, centralized_noniid[metric_key],
                color=COLORS["Centralized (Non-IID)"], marker=MARKERS["Centralized (Non-IID)"],
                linewidth=2.5, markersize=7, label="Centralized (Non-IID)",
                linestyle='--', markevery=max(1, len(cent_noniid_cu)//12))

    # Federated strategies
    for name, metrics in federated_all.items():
        ax.plot(fed_compute_units, metrics[metric_key],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=7, label=name,
                markevery=max(1, len(fed_compute_units)//12))

    ax.set_xlabel("Effective Computation Units", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_r2_comparison(centralized, federated_all, fed_cu, cent_cu, save_path,
                       centralized_noniid=None, cent_noniid_cu=None):
    """Plot 1: R2 Score vs Effective Computation."""
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_metric_vs_compute(ax, centralized, federated_all,
                            "r2_scores", "R2 Score",
                            "R2 Score: Federated vs Centralized", fed_cu, cent_cu,
                            centralized_noniid, cent_noniid_cu)

    # Annotate final values
    all_to_annotate = {"Centralized": (centralized, cent_cu)}
    if centralized_noniid is not None:
        all_to_annotate["Centralized (Non-IID)"] = (centralized_noniid, cent_noniid_cu)
    for name, metrics in federated_all.items():
        all_to_annotate[name] = (metrics, fed_cu)

    for name, (metrics, cu) in all_to_annotate.items():
        final_r2 = metrics["r2_scores"][-1]
        ax.annotate(f'{final_r2:.4f}', xy=(cu[-1], final_r2),
                    xytext=(8, 0), textcoords='offset points',
                    fontsize=9, fontweight='bold', color=COLORS[name])

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_mse_comparison(centralized, federated_all, fed_cu, cent_cu, save_path,
                        centralized_noniid=None, cent_noniid_cu=None):
    """Plot 2: MSE Loss vs Effective Computation."""
    fig, ax = plt.subplots(figsize=(12, 6))
    _plot_metric_vs_compute(ax, centralized, federated_all,
                            "mse_losses", "MSE Loss",
                            "MSE Loss: Federated vs Centralized", fed_cu, cent_cu,
                            centralized_noniid, cent_noniid_cu)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_final_performance_bar(centralized, federated_all, save_path,
                               centralized_noniid=None):
    """Plot 3: Final Performance Bar Chart (R2 and MSE)."""
    names = ["Centralized"]
    final_r2 = [centralized["r2_scores"][-1]]
    final_mse = [centralized["mse_losses"][-1]]
    if centralized_noniid is not None:
        names.append("Centralized (Non-IID)")
        final_r2.append(centralized_noniid["r2_scores"][-1])
        final_mse.append(centralized_noniid["mse_losses"][-1])
    names += list(federated_all.keys())
    final_r2 += [m["r2_scores"][-1] for m in federated_all.values()]
    final_mse += [m["mse_losses"][-1] for m in federated_all.values()]
    colors = [COLORS[n] for n in names]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(names))
    width = 0.35

    bars = ax.bar(x - width / 2, final_r2, width, color=colors, alpha=0.85, label="Final R2")
    for bar, val in zip(bars, final_r2):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')

    ax.set_ylabel("R2 Score", fontsize=12)
    ax.set_title("Final Performance: Federated vs Centralized", fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha='right')
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Secondary y-axis for MSE
    ax2 = ax.twinx()
    ax2.plot(x, final_mse, 'ko--', linewidth=2, markersize=10, label='Final MSE')
    for i, val in enumerate(final_mse):
        ax2.annotate(f'{val:.4f}', xy=(x[i], val), xytext=(0, 8),
                     textcoords='offset points', ha='center', fontsize=9)
    ax2.set_ylabel("MSE Loss", fontsize=12)
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_convergence_speed(centralized, federated_all, fed_cu, cent_cu, save_path,
                           centralized_noniid=None, cent_noniid_cu=None):
    """Plot 4: Convergence speed — compute units to reach 90% of centralized final R2."""
    target_r2 = 0.90 * centralized["r2_scores"][-1]

    def _first_cu_above(r2_list, cu_list, threshold):
        for r2, cu in zip(r2_list, cu_list):
            if r2 >= threshold:
                return cu
        return None  # never reached

    convergence = {}
    convergence["Centralized"] = _first_cu_above(centralized["r2_scores"], cent_cu, target_r2)
    if centralized_noniid is not None:
        convergence["Centralized (Non-IID)"] = _first_cu_above(
            centralized_noniid["r2_scores"], cent_noniid_cu, target_r2)
    for name, metrics in federated_all.items():
        convergence[name] = _first_cu_above(metrics["r2_scores"], fed_cu, target_r2)

    fig, ax = plt.subplots(figsize=(11, 6))
    names = list(convergence.keys())
    values = [convergence[n] if convergence[n] is not None else 0 for n in names]
    reached = [convergence[n] is not None for n in names]
    colors = [COLORS[n] for n in names]
    bar_alphas = [0.85 if r else 0.35 for r in reached]

    bars = ax.bar(names, values, color=colors, alpha=0.85)
    for bar, val, r, a in zip(bars, values, reached, bar_alphas):
        bar.set_alpha(a)
        label = f'{val:.1f}' if r else 'Not reached'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center',
                    fontsize=11, fontweight='bold')

    ax.set_ylabel("Effective Computation Units", fontsize=12)
    ax.set_title(f"Convergence Speed (to 90% of Centralized Final R2 = {target_r2:.4f})",
                 fontsize=13, fontweight='bold')
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_communication_cost(federated_all, save_path, include_noniid=False):
    """Plot 5: Communication cost (total parameter exchanges)."""
    input_dim = get_input_dim()
    model = Net(input_dim=input_dim)
    total_params = sum(p.numel() for p in model.parameters())
    num_selected = max(2, int(NUM_CLIENTS * FRACTION_FIT))

    names = ["Centralized"]
    costs = [0]
    if include_noniid:
        names.append("Centralized (Non-IID)")
        costs.append(0)  # also zero communication
    for name in federated_all.keys():
        names.append(name)
        total_transfers = NUM_ROUNDS * 2 * num_selected * total_params
        costs.append(total_transfers)

    fig, ax = plt.subplots(figsize=(11, 6))
    colors = [COLORS[n] for n in names]
    bars = ax.bar(names, [c / 1e6 for c in costs], color=colors, alpha=0.85)

    for bar, val in zip(bars, costs):
        label = f'{val/1e6:.1f}M' if val > 0 else '0'
        ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center',
                    fontsize=11, fontweight='bold')

    ax.set_ylabel("Parameters Transferred (Millions)", fontsize=12)
    ax.set_title("Communication Cost: Federated vs Centralized", fontsize=14, fontweight='bold')
    ax.set_xticklabels(names, fontsize=10, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_comprehensive_dashboard(centralized, federated_all, fed_cu, cent_cu, save_path,
                                 centralized_noniid=None, cent_noniid_cu=None):
    """Plot 6: 6-panel comprehensive dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    fig.suptitle("Federated vs Centralized: Comprehensive Comparison\n"
                 "(Indian Personal Finance — Disposable Income Prediction)",
                 fontsize=15, fontweight='bold')

    # Panel 1: R2 Score
    _plot_metric_vs_compute(axes[0, 0], centralized, federated_all,
                            "r2_scores", "R2 Score",
                            "R2 Score Progression", fed_cu, cent_cu,
                            centralized_noniid, cent_noniid_cu)

    # Panel 2: MSE Loss
    _plot_metric_vs_compute(axes[0, 1], centralized, federated_all,
                            "mse_losses", "MSE Loss",
                            "MSE Loss Progression", fed_cu, cent_cu,
                            centralized_noniid, cent_noniid_cu)

    # Panel 3: Training Loss
    ax = axes[0, 2]
    ax.plot(cent_cu, centralized["train_losses"],
            color=COLORS["Centralized"], marker=MARKERS["Centralized"],
            linewidth=2.5, markersize=7, label="Centralized",
            markevery=max(1, len(cent_cu)//12))
    if centralized_noniid is not None:
        ax.plot(cent_noniid_cu, centralized_noniid["train_losses"],
                color=COLORS["Centralized (Non-IID)"], marker=MARKERS["Centralized (Non-IID)"],
                linewidth=2.5, markersize=7, label="Centralized (Non-IID)",
                linestyle='--', markevery=max(1, len(cent_noniid_cu)//12))
    for name, metrics in federated_all.items():
        ax.plot(fed_cu, metrics["avg_train_loss"],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=7, label=name,
                markevery=max(1, len(fed_cu)//12))
    ax.set_xlabel("Effective Computation Units", fontsize=11)
    ax.set_ylabel("Training Loss", fontsize=11)
    ax.set_title("Training Loss", fontsize=12, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Final R2 Bar Chart
    ax = axes[1, 0]
    names = ["Centralized"]
    final_r2 = [centralized["r2_scores"][-1]]
    if centralized_noniid is not None:
        names.append("Cent. (Non-IID)")
        final_r2.append(centralized_noniid["r2_scores"][-1])
    names += list(federated_all.keys())
    final_r2 += [m["r2_scores"][-1] for m in federated_all.values()]
    bar_colors = [COLORS.get(n, COLORS.get("Centralized (Non-IID)")) for n in names]
    # Fix color lookup for abbreviated name
    bar_colors = []
    for n in names:
        if n == "Cent. (Non-IID)":
            bar_colors.append(COLORS["Centralized (Non-IID)"])
        else:
            bar_colors.append(COLORS[n])
    bars = ax.bar(names, final_r2, color=bar_colors, alpha=0.85)
    for bar, val in zip(bars, final_r2):
        ax.annotate(f'{val:.4f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords='offset points', ha='center', fontsize=8, fontweight='bold')
    ax.set_ylabel("R2 Score", fontsize=11)
    ax.set_title("Final R2 Comparison", fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9, rotation=15)
    ax.grid(True, alpha=0.3, axis='y')

    # Panel 5: Model Divergence (federated only)
    ax = axes[1, 1]
    for name, metrics in federated_all.items():
        rounds = metrics["rounds"]
        ax.plot(rounds, metrics["avg_divergence"],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=7, label=name,
                markevery=max(1, len(rounds)//12))
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Avg Model Divergence", fontsize=11)
    ax.set_title("Model Divergence (Federated Only)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 6: Effective Mu (federated only)
    ax = axes[1, 2]
    for name, metrics in federated_all.items():
        rounds = metrics["rounds"]
        ax.plot(rounds, metrics["avg_effective_mu"],
                color=COLORS[name], marker=MARKERS[name],
                linewidth=2, markersize=7, label=name,
                markevery=max(1, len(rounds)//12))
    ax.set_xlabel("Federated Round", fontsize=11)
    ax.set_ylabel("Avg Effective mu", fontsize=11)
    ax.set_title("Proximal Coefficient mu (Federated Only)", fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def print_summary(centralized, federated_all, fed_cu, cent_cu,
                  centralized_noniid=None, cent_noniid_cu=None):
    """Print a comprehensive summary table."""
    target_r2 = 0.90 * centralized["r2_scores"][-1]

    def _first_cu(r2_list, cu_list, threshold):
        for r2, cu in zip(r2_list, cu_list):
            if r2 >= threshold:
                return cu
        return None

    input_dim = get_input_dim()
    model = Net(input_dim=input_dim)
    total_params = sum(p.numel() for p in model.parameters())
    num_selected = max(2, int(NUM_CLIENTS * FRACTION_FIT))

    print("\n" + "=" * 100)
    print("FEDERATED vs CENTRALIZED — SUMMARY")
    print("=" * 100)
    print(f"{'Approach':<25} {'Final R2':>10} {'Best R2':>10} {'Final MSE':>10} "
          f"{'Conv. CU':>10} {'Comm. Cost':>14}")
    print("-" * 100)

    # Centralized (IID)
    c_conv = _first_cu(centralized["r2_scores"], cent_cu, target_r2)
    print(f"{'Centralized':<25} {centralized['r2_scores'][-1]:>10.4f} "
          f"{max(centralized['r2_scores']):>10.4f} "
          f"{centralized['mse_losses'][-1]:>10.4f} "
          f"{(str(round(c_conv, 1)) if c_conv else 'N/A'):>10} "
          f"{'0':>14}")

    # Centralized (Non-IID)
    if centralized_noniid is not None:
        cn_conv = _first_cu(centralized_noniid["r2_scores"], cent_noniid_cu, target_r2)
        print(f"{'Centralized (Non-IID)':<25} {centralized_noniid['r2_scores'][-1]:>10.4f} "
              f"{max(centralized_noniid['r2_scores']):>10.4f} "
              f"{centralized_noniid['mse_losses'][-1]:>10.4f} "
              f"{(str(round(cn_conv, 1)) if cn_conv else 'N/A'):>10} "
              f"{'0':>14}")

    # Federated
    for name, metrics in federated_all.items():
        conv = _first_cu(metrics["r2_scores"], fed_cu, target_r2)
        comm = NUM_ROUNDS * 2 * num_selected * total_params
        print(f"{name:<25} {metrics['r2_scores'][-1]:>10.4f} "
              f"{max(metrics['r2_scores']):>10.4f} "
              f"{metrics['mse_losses'][-1]:>10.4f} "
              f"{(str(round(conv, 1)) if conv else 'N/A'):>10} "
              f"{comm/1e6:>12.1f}M")

    print("-" * 100)

    # Privacy gap analysis
    cent_final = centralized["r2_scores"][-1]
    print(f"\nCentralized (IID) Final R2 (upper bound): {cent_final:.4f}")
    print(f"90% Target R2: {target_r2:.4f}")
    print()

    # Gap for non-IID centralized
    if centralized_noniid is not None:
        cn_final = centralized_noniid["r2_scores"][-1]
        gap = cent_final - cn_final
        pct = (gap / abs(cent_final)) * 100 if cent_final != 0 else 0
        print(f"  Centralized (Non-IID): R2 gap = {gap:+.4f} ({pct:.1f}% below IID centralized)")

    for name, metrics in federated_all.items():
        gap = cent_final - metrics["r2_scores"][-1]
        pct = (gap / abs(cent_final)) * 100 if cent_final != 0 else 0
        print(f"  {name}: R2 gap = {gap:+.4f} ({pct:.1f}% below IID centralized)")

    # Decompose the gap: non-IID effect vs federation effect
    if centralized_noniid is not None:
        print(f"\n--- Gap Decomposition ---")
        noniid_gap = cent_final - centralized_noniid["r2_scores"][-1]
        print(f"  Non-IID data effect (IID vs Non-IID centralized): {noniid_gap:+.4f}")
        for name, metrics in federated_all.items():
            fed_gap = centralized_noniid["r2_scores"][-1] - metrics["r2_scores"][-1]
            total_gap = cent_final - metrics["r2_scores"][-1]
            print(f"  {name}:")
            print(f"    Total gap from IID centralized:     {total_gap:+.4f}")
            print(f"    Due to non-IID data:                {noniid_gap:+.4f} ({noniid_gap/max(total_gap,1e-8)*100:.0f}%)")
            print(f"    Due to federation overhead:          {fed_gap:+.4f} ({fed_gap/max(total_gap,1e-8)*100:.0f}%)")

    # Winner
    all_approaches = {"Centralized": centralized["r2_scores"][-1]}
    if centralized_noniid is not None:
        all_approaches["Centralized (Non-IID)"] = centralized_noniid["r2_scores"][-1]
    all_approaches.update({n: m["r2_scores"][-1] for n, m in federated_all.items()})
    fed_winner = max(
        {n: r2 for n, r2 in all_approaches.items() if "Centralized" not in n},
        key=lambda n: all_approaches[n]
    )
    print(f"\nBest federated strategy: {fed_winner} (R2 = {all_approaches[fed_winner]:.4f})")
    print("=" * 100)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    SEED = 2023

    print("\n" + "=" * 70)
    print("FEDERATED vs CENTRALIZED TRAINING COMPARISON")
    print("Dataset: Indian Personal Finance (Disposable Income Prediction)")
    print(f"Device: {DEVICE}")
    print(f"Federated: {NUM_CLIENTS} clients, Fraction Fit: {FRACTION_FIT}, "
          f"{NUM_ROUNDS} rounds, {LOCAL_EPOCHS} local epochs")
    print("=" * 70)

    # Reset and preload data
    reset_data_cache()
    _load_and_preprocess_data()
    print(f"Input dimension: {get_input_dim()}")

    # Fair compute budget: federated effective work = rounds * local_epochs * fraction_fit
    effective_epochs = math.ceil(NUM_ROUNDS * LOCAL_EPOCHS * FRACTION_FIT)
    print(f"\nEffective compute budget: {NUM_ROUNDS} rounds x {LOCAL_EPOCHS} epochs x "
          f"{FRACTION_FIT} fraction = {effective_epochs} centralized epochs")

    # 1. Centralized training (IID — all data, homogeneous)
    start = time.time()
    centralized_metrics = train_centralized(num_epochs=effective_epochs, seed=SEED)
    cent_time = time.time() - start
    print(f"  Centralized (IID) training time: {cent_time:.1f}s")

    # 2. Centralized training (Non-IID — sequential client partitions)
    start = time.time()
    centralized_noniid_metrics = train_centralized_noniid(
        num_epochs=effective_epochs, seed=SEED)
    cent_noniid_time = time.time() - start
    print(f"  Centralized (Non-IID) training time: {cent_noniid_time:.1f}s")

    # 3. Federated training (all 3 strategies)
    start = time.time()
    federated_results = run_federated_strategies(seed=SEED)
    fed_time = time.time() - start
    print(f"\n  Federated training time (all strategies): {fed_time:.1f}s")

    # 4. Compute effective computation units for x-axis
    fed_cu = compute_effective_units_federated(NUM_ROUNDS, LOCAL_EPOCHS, FRACTION_FIT)
    cent_cu = compute_effective_units_centralized(effective_epochs)
    cent_noniid_cu = compute_effective_units_centralized(effective_epochs)

    # 5. Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 6. Generate all plots
    print("\nGenerating comparison plots...")
    plot_r2_comparison(centralized_metrics, federated_results, fed_cu, cent_cu,
                       OUTPUT_DIR / "fed_vs_cent_r2.png",
                       centralized_noniid_metrics, cent_noniid_cu)
    plot_mse_comparison(centralized_metrics, federated_results, fed_cu, cent_cu,
                        OUTPUT_DIR / "fed_vs_cent_mse.png",
                        centralized_noniid_metrics, cent_noniid_cu)
    plot_final_performance_bar(centralized_metrics, federated_results,
                               OUTPUT_DIR / "fed_vs_cent_final_bar.png",
                               centralized_noniid_metrics)
    plot_convergence_speed(centralized_metrics, federated_results, fed_cu, cent_cu,
                           OUTPUT_DIR / "fed_vs_cent_convergence.png",
                           centralized_noniid_metrics, cent_noniid_cu)
    plot_communication_cost(federated_results,
                            OUTPUT_DIR / "fed_vs_cent_comm_cost.png",
                            include_noniid=True)
    plot_comprehensive_dashboard(centralized_metrics, federated_results, fed_cu, cent_cu,
                                 OUTPUT_DIR / "fed_vs_cent_dashboard.png",
                                 centralized_noniid_metrics, cent_noniid_cu)

    # 7. Summary
    print_summary(centralized_metrics, federated_results, fed_cu, cent_cu,
                  centralized_noniid_metrics, cent_noniid_cu)

    print("\nAll plots saved to:", OUTPUT_DIR.resolve())
    print("  1. fed_vs_cent_r2.png           — R2 Score vs Effective Computation")
    print("  2. fed_vs_cent_mse.png          — MSE Loss vs Effective Computation")
    print("  3. fed_vs_cent_final_bar.png    — Final Performance Bar Chart")
    print("  4. fed_vs_cent_convergence.png  — Convergence Speed")
    print("  5. fed_vs_cent_comm_cost.png    — Communication Cost")
    print("  6. fed_vs_cent_dashboard.png    — Comprehensive 6-Panel Dashboard")


if __name__ == "__main__":
    main()
