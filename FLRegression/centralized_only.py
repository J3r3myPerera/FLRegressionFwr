"""
Centralized Model Comparison: IID vs Non-IID Data.

Trains and compares two centralized baselines at 35 epochs:

1. Centralized (IID)     — all data pooled, homogeneous 80/20 split
2. Centralized (Non-IID) — same non-IID partitioned data that federated clients
                           use, concatenated and trained sequentially
                           (partition 0 → partition 1 → ... → partition N-1)
                           to expose the model to the same data heterogeneity.

The Non-IID variant uses exactly the same data loading path as the FL
simulation (load_data with the same partition_id / num_partitions / seed),
so results are directly comparable.

Usage:
    cd FLRegressionFlwr/FLRegression
    python centralized_only.py
"""

import time
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
    NUM_CLIENTS, LEARNING_RATE, BATCH_SIZE, DEVICE,
    reset_data_cache, _load_and_preprocess_data,
)
from dataset import load_data, load_centralized_dataset

# ── Configuration ────────────────────────────────────────────────────────────
NUM_EPOCHS = 35
SEED = 2023

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "centralized_only"

COLORS = {
    "Centralized (IID)":     "#f39c12",
    "Centralized (Non-IID)": "#9b59b6",
}
MARKERS = {
    "Centralized (IID)":     "D",
    "Centralized (Non-IID)": "P",
}


# ── IID Training ─────────────────────────────────────────────────────────────

def train_centralized_iid(num_epochs: int, lr: float = LEARNING_RATE,
                          batch_size: int = BATCH_SIZE, seed: int = SEED) -> dict:
    """Train a single model on the full pooled dataset (IID).

    Uses an 80/20 train/test split with the same random state as
    load_centralized_dataset() so the test set is identical to the one used
    during FL global evaluation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_cache, preprocessors = _load_and_preprocess_data()
    X = data_cache["X"]
    y = data_cache["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trainloader = DataLoader(
        TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32),
        ),
        batch_size=batch_size, shuffle=True,
    )
    testloader = DataLoader(
        TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.float32),
        ),
        batch_size=128,
    )

    input_dim = preprocessors["input_dim"]
    model = Net(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    metrics = {"epochs": [], "r2_scores": [], "mse_losses": [],
               "rmse_scores": [], "mae_scores": [], "train_losses": []}

    print(f"\n{'='*60}")
    print("Running: Centralized (IID — full pooled dataset)")
    print(f"  Epochs: {num_epochs} | LR: {lr} | Batch: {batch_size}")
    print(f"  Train samples: {len(X_train)} | Test samples: {len(X_test)}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, num_batches = 0.0, 0

        for features, targets in trainloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(num_batches, 1)
        mse, r2, rmse, mae = test(model, testloader, DEVICE)

        metrics["epochs"].append(epoch)
        metrics["r2_scores"].append(r2)
        metrics["mse_losses"].append(mse)
        metrics["rmse_scores"].append(rmse)
        metrics["mae_scores"].append(mae)
        metrics["train_losses"].append(avg_train_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  Epoch {epoch:>3}/{num_epochs}  "
                  f"R2={r2:.4f}  MSE={mse:.4f}  "
                  f"RMSE={rmse:.4f}  MAE={mae:.4f}  "
                  f"TrainLoss={avg_train_loss:.4f}")

    print(f"\n  Final R2: {metrics['r2_scores'][-1]:.4f}  "
          f"Final MSE: {metrics['mse_losses'][-1]:.4f}")
    return metrics


# ── Non-IID Training ──────────────────────────────────────────────────────────

def train_centralized_noniid(num_epochs: int, lr: float = LEARNING_RATE,
                             batch_size: int = BATCH_SIZE, seed: int = SEED) -> dict:
    """Train a single model on the same non-IID partitioned data as the FL clients.

    Collects every client partition (same load_data() call used by FL clients),
    concatenates them, and trains sequentially without shuffling so that the
    model experiences the same data ordering heterogeneity as federated rounds.

    The evaluation uses the identical centralized test set as FL global eval.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    # ── Collect all client partitions (exact same data as FL) ────────────────
    print(f"\n{'='*60}")
    print("Collecting client partitions (same as FL simulation)...")

    all_X_train, all_y_train = [], []
    for cid in range(NUM_CLIENTS):
        trainloader, _ = load_data(cid, NUM_CLIENTS, batch_size)
        for features, targets in trainloader:
            all_X_train.append(features)
            all_y_train.append(targets)
        print(f"  Client {cid:>2}: loaded partition")

    all_X = torch.cat(all_X_train, dim=0)   # shape: (N_total, input_dim)
    all_y = torch.cat(all_y_train, dim=0)   # shape: (N_total, 1)
    total_samples = all_X.shape[0]

    # No shuffle —  preserves non-IID partition ordering across the full dataset
    noniid_loader = DataLoader(
        TensorDataset(all_X, all_y),
        batch_size=batch_size, shuffle=False,
    )

    # Same test set as FL global evaluation
    testloader = load_centralized_dataset()

    input_dim = get_input_dim()
    model = Net(input_dim=input_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()

    metrics = {"epochs": [], "r2_scores": [], "mse_losses": [],
               "rmse_scores": [], "mae_scores": [], "train_losses": []}

    print(f"\n{'='*60}")
    print("Running: Centralized (Non-IID — sequential FL partitions)")
    print(f"  Epochs: {num_epochs} | LR: {lr} | Batch: {batch_size}")
    print(f"  Total train samples ({NUM_CLIENTS} partitions): {total_samples}")
    print(f"{'='*60}")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, num_batches = 0.0, 0

        for features, targets in noniid_loader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(features), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            num_batches += 1

        avg_train_loss = running_loss / max(num_batches, 1)
        mse, r2, rmse, mae = test(model, testloader, DEVICE)

        metrics["epochs"].append(epoch)
        metrics["r2_scores"].append(r2)
        metrics["mse_losses"].append(mse)
        metrics["rmse_scores"].append(rmse)
        metrics["mae_scores"].append(mae)
        metrics["train_losses"].append(avg_train_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == num_epochs:
            print(f"  Epoch {epoch:>3}/{num_epochs}  "
                  f"R2={r2:.4f}  MSE={mse:.4f}  "
                  f"RMSE={rmse:.4f}  MAE={mae:.4f}  "
                  f"TrainLoss={avg_train_loss:.4f}")

    print(f"\n  Final R2: {metrics['r2_scores'][-1]:.4f}  "
          f"Final MSE: {metrics['mse_losses'][-1]:.4f}")
    return metrics


# ── Plotting ──────────────────────────────────────────────────────────────────

def _annotate_final(ax, epochs, values, color, offset=(8, 0)):
    ax.annotate(f'{values[-1]:.4f}',
                xy=(epochs[-1], values[-1]),
                xytext=offset, textcoords='offset points',
                fontsize=9, fontweight='bold', color=color)


def plot_r2(iid: dict, noniid: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, m in [("Centralized (IID)", iid), ("Centralized (Non-IID)", noniid)]:
        ax.plot(m["epochs"], m["r2_scores"],
                color=COLORS[label], marker=MARKERS[label],
                linewidth=2.5, markersize=8, label=label,
                markevery=max(1, NUM_EPOCHS // 12))
        _annotate_final(ax, m["epochs"], m["r2_scores"], COLORS[label])

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("R² Score", fontsize=12)
    ax.set_title("R² Score: Centralized IID vs Non-IID", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_mse(iid: dict, noniid: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, m in [("Centralized (IID)", iid), ("Centralized (Non-IID)", noniid)]:
        ax.plot(m["epochs"], m["mse_losses"],
                color=COLORS[label], marker=MARKERS[label],
                linewidth=2.5, markersize=8, label=label,
                markevery=max(1, NUM_EPOCHS // 12))
        _annotate_final(ax, m["epochs"], m["mse_losses"], COLORS[label])

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("MSE Loss", fontsize=12)
    ax.set_title("MSE Loss: Centralized IID vs Non-IID", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_train_loss(iid: dict, noniid: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, m in [("Centralized (IID)", iid), ("Centralized (Non-IID)", noniid)]:
        ax.plot(m["epochs"], m["train_losses"],
                color=COLORS[label], marker=MARKERS[label],
                linewidth=2.5, markersize=8, label=label,
                markevery=max(1, NUM_EPOCHS // 12))

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Training Loss", fontsize=12)
    ax.set_title("Training Loss: Centralized IID vs Non-IID", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_rmse(iid: dict, noniid: dict, save_path: Path):
    fig, ax = plt.subplots(figsize=(11, 6))
    for label, m in [("Centralized (IID)", iid), ("Centralized (Non-IID)", noniid)]:
        ax.plot(m["epochs"], m["rmse_scores"],
                color=COLORS[label], marker=MARKERS[label],
                linewidth=2.5, markersize=8, label=label,
                markevery=max(1, NUM_EPOCHS // 12))
        _annotate_final(ax, m["epochs"], m["rmse_scores"], COLORS[label])

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title("RMSE: Centralized IID vs Non-IID", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_final_bar(iid: dict, noniid: dict, save_path: Path):
    """Bar chart comparing final R2, MSE, RMSE and MAE side-by-side."""
    labels = ["Centralized (IID)", "Centralized (Non-IID)"]
    metrics_to_compare = {
        "Final R²":  [iid["r2_scores"][-1],  noniid["r2_scores"][-1]],
        "Final MSE": [iid["mse_losses"][-1], noniid["mse_losses"][-1]],
        "Final RMSE":[iid["rmse_scores"][-1],noniid["rmse_scores"][-1]],
        "Final MAE": [iid["mae_scores"][-1], noniid["mae_scores"][-1]],
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle("Final Performance: Centralized IID vs Non-IID\n"
                 "(Indian Personal Finance — Disposable Income Prediction)",
                 fontsize=13, fontweight='bold')

    for ax, (metric_name, vals) in zip(axes, metrics_to_compare.items()):
        colors = [COLORS[l] for l in labels]
        bars = ax.bar(labels, vals, color=colors, alpha=0.85, width=0.5)
        for bar, val in zip(bars, vals):
            ax.annotate(f'{val:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords='offset points',
                        ha='center', fontsize=11, fontweight='bold')
        ax.set_title(metric_name, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name.split()[-1], fontsize=11)
        ax.tick_params(axis='x', labelsize=9, rotation=15)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


def plot_dashboard(iid: dict, noniid: dict, save_path: Path):
    """4-panel dashboard: R2, MSE, RMSE, Training Loss."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Centralized Model: IID vs Non-IID Data\n"
                 "(Indian Personal Finance — Disposable Income Prediction, "
                 f"{NUM_EPOCHS} epochs)",
                 fontsize=14, fontweight='bold')

    panels = [
        (axes[0, 0], "r2_scores",   "R² Score",      "R² Score Progression"),
        (axes[0, 1], "mse_losses",  "MSE Loss",       "MSE Loss Progression"),
        (axes[1, 0], "rmse_scores", "RMSE",           "RMSE Progression"),
        (axes[1, 1], "train_losses","Training Loss",  "Training Loss Progression"),
    ]

    for ax, key, ylabel, title in panels:
        for label, m in [("Centralized (IID)", iid), ("Centralized (Non-IID)", noniid)]:
            ax.plot(m["epochs"], m[key],
                    color=COLORS[label], marker=MARKERS[label],
                    linewidth=2.5, markersize=7, label=label,
                    markevery=max(1, NUM_EPOCHS // 12))
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {save_path.name}")


# ── Summary ───────────────────────────────────────────────────────────────────

def print_summary(iid: dict, noniid: dict):
    cent_final_r2 = iid["r2_scores"][-1]
    noniid_final_r2 = noniid["r2_scores"][-1]
    gap = cent_final_r2 - noniid_final_r2
    gap_pct = (gap / abs(cent_final_r2)) * 100 if cent_final_r2 != 0 else 0

    print("\n" + "=" * 80)
    print("CENTRALIZED MODEL: IID vs NON-IID — SUMMARY")
    print(f"  Epochs: {NUM_EPOCHS} | Seed: {SEED}")
    print("=" * 80)
    print(f"{'Approach':<28} {'Final R²':>10} {'Best R²':>10} "
          f"{'Final MSE':>11} {'Final RMSE':>12} {'Final MAE':>11}")
    print("-" * 80)

    for label, m in [("Centralized (IID)", iid), ("Centralized (Non-IID)", noniid)]:
        print(f"{label:<28} "
              f"{m['r2_scores'][-1]:>10.4f} "
              f"{max(m['r2_scores']):>10.4f} "
              f"{m['mse_losses'][-1]:>11.4f} "
              f"{m['rmse_scores'][-1]:>12.4f} "
              f"{m['mae_scores'][-1]:>11.4f}")

    print("-" * 80)
    best_epoch_iid    = iid["r2_scores"].index(max(iid["r2_scores"])) + 1
    best_epoch_noniid = noniid["r2_scores"].index(max(noniid["r2_scores"])) + 1
    print(f"\n  Best R² epoch — IID: {best_epoch_iid}  |  Non-IID: {best_epoch_noniid}")
    print(f"\n  Non-IID Penalty (Final R² gap): {gap:+.4f} ({gap_pct:.1f}% below IID)")
    if gap > 0:
        print(f"  → Data heterogeneity reduces final R² by {gap_pct:.1f}%")
    else:
        print(f"  → Non-IID data did NOT hurt performance (gap = {gap:+.4f})")
    print("=" * 80)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("CENTRALIZED MODEL COMPARISON: IID vs NON-IID DATA")
    print("Dataset: Indian Personal Finance (Disposable Income Prediction)")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {NUM_EPOCHS} | Clients (for Non-IID partitioning): {NUM_CLIENTS}")
    print("=" * 70)

    # Reset and preload data
    reset_data_cache()
    _load_and_preprocess_data()
    print(f"Input dimension: {get_input_dim()}")

    # 1. IID centralized training
    start = time.time()
    iid_metrics = train_centralized_iid(num_epochs=NUM_EPOCHS, seed=SEED)
    print(f"  IID training time: {time.time() - start:.1f}s")

    # 2. Non-IID centralized training (exact FL partition data)
    start = time.time()
    noniid_metrics = train_centralized_noniid(num_epochs=NUM_EPOCHS, seed=SEED)
    print(f"  Non-IID training time: {time.time() - start:.1f}s")

    # 3. Save plots
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\nGenerating plots...")

    plot_r2(iid_metrics, noniid_metrics,
            OUTPUT_DIR / "cent_r2.png")
    plot_mse(iid_metrics, noniid_metrics,
             OUTPUT_DIR / "cent_mse.png")
    plot_rmse(iid_metrics, noniid_metrics,
              OUTPUT_DIR / "cent_rmse.png")
    plot_train_loss(iid_metrics, noniid_metrics,
                    OUTPUT_DIR / "cent_train_loss.png")
    plot_final_bar(iid_metrics, noniid_metrics,
                   OUTPUT_DIR / "cent_final_bar.png")
    plot_dashboard(iid_metrics, noniid_metrics,
                   OUTPUT_DIR / "cent_dashboard.png")

    # 4. Summary
    print_summary(iid_metrics, noniid_metrics)

    print(f"\nAll plots saved to: {OUTPUT_DIR.resolve()}")
    print("  1. cent_r2.png          — R² Score per epoch")
    print("  2. cent_mse.png         — MSE Loss per epoch")
    print("  3. cent_rmse.png        — RMSE per epoch")
    print("  4. cent_train_loss.png  — Training Loss per epoch")
    print("  5. cent_final_bar.png   — Final R², MSE, RMSE, MAE comparison bars")
    print("  6. cent_dashboard.png   — 4-panel comprehensive dashboard")


if __name__ == "__main__":
    main()
