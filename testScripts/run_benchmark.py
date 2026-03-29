#!/usr/bin/env python3
"""
Centralized vs Federated Learning Benchmark
============================================
Trains centralized models (IID and Non-IID) and runs federated strategies
(FedAvg, FedProx, SmartFedProx), then generates comparison visualizations.

Usage:
    python testScripts/run_benchmark.py
    python testScripts/run_benchmark.py --epochs 35 --rounds 20 --trials 3 --seed 2023

Outputs saved to: testScripts/results/
"""

import sys
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Path setup — allow imports from FLRegression/
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FL_DIR = PROJECT_ROOT / "FLRegression"
sys.path.insert(0, str(FL_DIR))

from module import (
    Net, test, get_input_dim,
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS,
    LEARNING_RATE, BATCH_SIZE, DEVICE, STRATEGIES,
    _load_and_preprocess_data, reset_data_cache,
)
from dataset import load_centralized_dataset
from server import FederatedSimulator

RESULTS_DIR = Path(__file__).resolve().parent / "results"

def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Centralized Training
def train_centralized(mode: str = "iid", epochs: int = 35,
                      lr: float = 0.001, batch_size: int = 64,
                      seed: int = 2023) -> tuple:
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, TensorDataset

    set_seed(seed)

    data_cache, preprocessors = _load_and_preprocess_data()
    X = data_cache["X"]
    y = data_cache["y"]
    input_dim = preprocessors["input_dim"]

    if mode == "iid":
        # Standard IID: random 80/20 split on the full dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    else:
        # Non-IID: only use data from a biased subset of composite keys
        # (Occupation + City_Tier + Income_Bracket), plus income skew.
        combined_keys = data_cache["combined_keys"]
        unique_keys = data_cache["unique_keys"]
        incomes = data_cache["incomes"]

        np.random.seed(42)
        # Keep roughly half the demographic groups
        selected_keys = np.random.choice(
            unique_keys, size=len(unique_keys) // 2, replace=False
        )
        key_mask = np.isin(combined_keys, selected_keys)

        # Apply additional income skew (bias towards lower income)
        income_p75 = np.percentile(incomes, 75)
        income_mask = incomes < income_p75
        biased_mask = key_mask & income_mask

        # Fallback if too few samples survive the double filter
        if biased_mask.sum() < len(X) * 0.25:
            biased_mask = key_mask

        train_indices = np.where(biased_mask)[0]
        X_train = X[train_indices]
        y_train = y[train_indices]

        # Test on the same global test set used by FL for fair comparison
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_ds = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    trainloader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(test_ds, batch_size=batch_size)

    model = Net(input_dim=input_dim).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    metrics = {
        "steps": [],
        "r2_scores": [],
        "mse_losses": [],
        "rmse_losses": [],
        "mae_losses": [],
        "train_losses": [],
    }

    label = f"Centralized ({mode.upper()})"
    print(f"\n{'='*60}")
    print(f"  {label}  —  {epochs} epochs")
    print(f"  Train: {len(train_ds)} samples, Test: {len(test_ds)} samples")
    print(f"{'='*60}")

    for epoch in range(1, epochs + 1):
        # --- train ---
        model.train()
        running_loss, n_batches = 0.0, 0
        for features, targets in trainloader:
            features, targets = features.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            preds = model(features)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n_batches += 1
        avg_train_loss = running_loss / max(n_batches, 1)

        # --- evaluate ---
        model.eval()
        total_loss, total_n = 0.0, 0
        all_preds, all_targs = [], []
        with torch.no_grad():
            for features, targets in testloader:
                features, targets = features.to(DEVICE), targets.to(DEVICE)
                preds = model(features)
                total_loss += criterion(preds, targets).item() * features.size(0)
                total_n += features.size(0)
                all_preds.append(preds.cpu())
                all_targs.append(targets.cpu())

        mse = total_loss / max(total_n, 1)
        rmse = mse ** 0.5
        all_preds = torch.cat(all_preds)
        all_targs = torch.cat(all_targs)
        mae = (all_targs - all_preds).abs().mean().item()

        ss_res = ((all_targs - all_preds) ** 2).sum().item()
        ss_tot = ((all_targs - all_targs.mean()) ** 2).sum().item()
        r2 = 1.0 - (ss_res / max(ss_tot, 1e-8))

        metrics["steps"].append(epoch)
        metrics["r2_scores"].append(r2)
        metrics["mse_losses"].append(mse)
        metrics["rmse_losses"].append(rmse)
        metrics["mae_losses"].append(mae)
        metrics["train_losses"].append(avg_train_loss)

        if epoch % 5 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}:  R²={r2:.4f}  MSE={mse:.4f}  "
                  f"RMSE={rmse:.4f}  MAE={mae:.4f}  TrainLoss={avg_train_loss:.4f}")

    print(f"  Final:  R²={metrics['r2_scores'][-1]:.4f}  "
          f"MSE={metrics['mse_losses'][-1]:.4f}")

    return metrics, model.state_dict()

# Federated Strategy Runner
def run_federated_strategies(num_rounds: int = 20, num_trials: int = 3,
                             seed: int = 2023) -> dict:
    """
    Run FedAvg / FedProx / SmartFedProx and return trial-averaged metrics.
    """
    all_trial_results = {name: [] for name in STRATEGIES}

    for trial in range(num_trials):
        trial_seed = seed + trial * 100
        print(f"\n{'#'*60}")
        print(f"  FL TRIAL {trial + 1}/{num_trials}  (seed={trial_seed})")
        print(f"{'#'*60}")

        for strategy_name, config in STRATEGIES.items():
            set_seed(trial_seed)
            simulator = FederatedSimulator(strategy_name, config)
            metrics, _ = simulator.run(num_rounds)
            all_trial_results[strategy_name].append(metrics)

    # Average across trials
    aggregated = {}
    for name in STRATEGIES:
        trials = all_trial_results[name]
        n = len(trials[0]["rounds"])
        aggregated[name] = {
            "steps": trials[0]["rounds"],
            "r2_scores": [np.mean([t["r2_scores"][i] for t in trials]) for i in range(n)],
            "mse_losses": [np.mean([t["mse_losses"][i] for t in trials]) for i in range(n)],
            "train_losses": [np.mean([t["avg_train_loss"][i] for t in trials]) for i in range(n)],
            "avg_divergence": [np.mean([t["avg_divergence"][i] for t in trials]) for i in range(n)],
            "avg_effective_mu": [np.mean([t["avg_effective_mu"][i] for t in trials]) for i in range(n)],
            "final_r2_std": np.std([t["r2_scores"][-1] for t in trials]),
        }

    return aggregated

# Plotting
CENT_COLORS = {
    "Centralized (IID)": "#FF8C00",
    "Centralized (Non-IID)": "#8B5CF6",
}
FL_COLORS = {
    "FedAvg": "#e74c3c",
    "FedProx": "#3498db",
    "SmartFedProx": "#2ecc71",
}
MARKERS = {
    "Centralized (IID)": "o",
    "Centralized (Non-IID)": "D",
    "FedAvg": "s",
    "FedProx": "^",
    "SmartFedProx": "v",
}


def _plot_metric(ax, all_results: dict, metric_key: str, ylabel: str, title: str):
    for name, m in all_results.items():
        color = {**CENT_COLORS, **FL_COLORS}.get(name, "#999999")
        marker = MARKERS.get(name, "o")
        ax.plot(m["steps"], m[metric_key],
                color=color, marker=marker, linewidth=2,
                markersize=6, label=name, markevery=max(1, len(m["steps"]) // 10))
        # Annotate final value
        final_val = m[metric_key][-1]
        ax.annotate(f"{final_val:.4f}",
                    xy=(m["steps"][-1], final_val),
                    xytext=(5, 0), textcoords="offset points",
                    fontsize=8, fontweight="bold", color=color)
    ax.set_xlabel("Epoch / FL Round", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_dashboard(cent_results: dict, fl_results: dict, save_dir: Path):
    all_r = {**cent_results, **fl_results}

    fig, axes = plt.subplots(2, 3, figsize=(20, 11))
    fig.suptitle(
        "Benchmark: Centralized vs Federated Learning\n"
        "(Indian Personal Finance — Disposable Income Prediction)",
        fontsize=15, fontweight="bold",
    )

    _plot_metric(axes[0, 0], all_r, "r2_scores", "R² Score", "R² Score Progression")
    _plot_metric(axes[0, 1], all_r, "mse_losses", "MSE Loss", "MSE Loss Progression")
    _plot_metric(axes[0, 2], all_r, "train_losses", "Training Loss", "Training Loss Progression")

    # RMSE — only centralized has it pre-computed; compute for FL
    rmse_results = {}
    for name, m in all_r.items():
        rmse_results[name] = dict(m)
        if "rmse_losses" not in m:
            rmse_results[name]["rmse_losses"] = [v ** 0.5 for v in m["mse_losses"]]
    _plot_metric(axes[1, 0], rmse_results, "rmse_losses", "RMSE", "RMSE Progression")

    # MAE — only centralized has it; skip FL approaches for this panel
    mae_results = {k: v for k, v in cent_results.items() if "mae_losses" in v}
    if mae_results:
        _plot_metric(axes[1, 1], mae_results, "mae_losses", "MAE", "MAE Progression (Centralized)")
    else:
        axes[1, 1].set_visible(False)

    # Final bar chart comparison
    ax = axes[1, 2]
    names = list(all_r.keys())
    final_r2 = [all_r[n]["r2_scores"][-1] for n in names]
    colors = [{**CENT_COLORS, **FL_COLORS}[n] for n in names]
    x = np.arange(len(names))

    bars = ax.bar(x, final_r2, color=colors, alpha=0.85, edgecolor="white", linewidth=0.8)
    for bar, val, name in zip(bars, final_r2, names):
        ax.annotate(f"{val:.4f}",
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("Centralized ", "Cent.\n") for n in names],
                       fontsize=9, rotation=0)
    ax.set_ylabel("Final R² Score", fontsize=11)
    ax.set_title("Final R² Comparison", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = save_dir / "benchmark_dashboard.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved: {path}")


def plot_individual(cent_results: dict, fl_results: dict, save_dir: Path):
    """Save individual metric comparison plots."""
    all_r = {**cent_results, **fl_results}

    for metric_key, ylabel, title, fname in [
        ("r2_scores", "R² Score", "R² Score: Centralized vs Federated", "benchmark_r2.png"),
        ("mse_losses", "MSE Loss", "MSE Loss: Centralized vs Federated", "benchmark_mse.png"),
        ("train_losses", "Training Loss", "Training Loss: Centralized vs Federated", "benchmark_train_loss.png"),
    ]:
        fig, ax = plt.subplots(figsize=(11, 6))
        _plot_metric(ax, all_r, metric_key, ylabel, title)
        plt.tight_layout()
        path = save_dir / fname
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # RMSE plot (compute for FL)
    rmse_r = {}
    for name, m in all_r.items():
        rmse_r[name] = dict(m)
        if "rmse_losses" not in m:
            rmse_r[name]["rmse_losses"] = [v ** 0.5 for v in m["mse_losses"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    _plot_metric(ax, rmse_r, "rmse_losses", "RMSE", "RMSE: Centralized vs Federated")
    plt.tight_layout()
    path = save_dir / "benchmark_rmse.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_final_bars(cent_results: dict, fl_results: dict, save_dir: Path):
    """Bar chart of final metrics across all approaches."""
    all_r = {**cent_results, **fl_results}
    names = list(all_r.keys())
    colors = [{**CENT_COLORS, **FL_COLORS}[n] for n in names]

    final_r2 = [all_r[n]["r2_scores"][-1] for n in names]
    final_mse = [all_r[n]["mse_losses"][-1] for n in names]
    final_rmse = [v ** 0.5 for v in final_mse]

    # MAE only available for centralized; use None for FL
    final_mae = []
    for n in names:
        m = all_r[n]
        if "mae_losses" in m:
            final_mae.append(m["mae_losses"][-1])
        else:
            final_mae.append(None)

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        "Final Performance: Centralized vs Federated\n"
        "(Indian Personal Finance — Disposable Income Prediction)",
        fontsize=14, fontweight="bold",
    )

    x = np.arange(len(names))
    short_names = [n.replace("Centralized ", "Cent.\n") for n in names]

    for ax, vals, ylabel, title in [
        (axes[0], final_r2, "R²", "Final R²"),
        (axes[1], final_mse, "MSE", "Final MSE"),
        (axes[2], final_rmse, "RMSE", "Final RMSE"),
    ]:
        bars = ax.bar(x, vals, color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            ax.annotate(f"{val:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(short_names, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    # MAE bar — only centralized entries
    ax = axes[3]
    mae_names, mae_vals, mae_colors = [], [], []
    for i, n in enumerate(names):
        if final_mae[i] is not None:
            mae_names.append(short_names[i])
            mae_vals.append(final_mae[i])
            mae_colors.append(colors[i])
    if mae_vals:
        xm = np.arange(len(mae_names))
        bars = ax.bar(xm, mae_vals, color=mae_colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, mae_vals):
            ax.annotate(f"{val:.4f}",
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 4), textcoords="offset points",
                        ha="center", fontsize=10, fontweight="bold")
        ax.set_xticks(xm)
        ax.set_xticklabels(mae_names, fontsize=9)
    ax.set_ylabel("MAE", fontsize=11)
    ax.set_title("Final MAE", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = save_dir / "benchmark_final_bars.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")

# Summary Table
def print_summary(cent_results: dict, fl_results: dict):
    all_r = {**cent_results, **fl_results}

    print("\n" + "=" * 85)
    print("BENCHMARK SUMMARY")
    print("=" * 85)
    print(f"{'Approach':<25} {'Final R²':>10} {'Best R²':>10} "
          f"{'Final MSE':>10} {'Final RMSE':>11} {'Final MAE':>10}")
    print("-" * 85)

    for name, m in all_r.items():
        final_r2 = m["r2_scores"][-1]
        best_r2 = max(m["r2_scores"])
        final_mse = m["mse_losses"][-1]
        final_rmse = final_mse ** 0.5
        mae_str = f"{m['mae_losses'][-1]:>10.4f}" if "mae_losses" in m else f"{'N/A':>10}"
        print(f"{name:<25} {final_r2:>10.4f} {best_r2:>10.4f} "
              f"{final_mse:>10.4f} {final_rmse:>11.4f} {mae_str}")

    print("-" * 85)

    # Determine overall winner
    winner = max(all_r, key=lambda n: all_r[n]["r2_scores"][-1])
    print(f"  Best overall: {winner} (R² = {all_r[winner]['r2_scores'][-1]:.4f})")

    # Best federated strategy
    fl_only = {k: v for k, v in fl_results.items()}
    if fl_only:
        fl_winner = max(fl_only, key=lambda n: fl_only[n]["r2_scores"][-1])
        cent_iid_r2 = cent_results.get("Centralized (IID)", {}).get("r2_scores", [0])[-1]
        fl_winner_r2 = fl_only[fl_winner]["r2_scores"][-1]
        gap = cent_iid_r2 - fl_winner_r2
        print(f"  Best FL strategy: {fl_winner} (R² = {fl_winner_r2:.4f})")
        print(f"  Gap to centralized IID: {gap:.4f}")

    print("=" * 85)

# Main
def main():
    parser = argparse.ArgumentParser(
        description="Benchmark centralized vs federated learning strategies."
    )
    parser.add_argument("--epochs", type=int, default=35,
                        help="Centralized training epochs (default: 35)")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS,
                        help=f"Federated rounds (default: {NUM_ROUNDS})")
    parser.add_argument("--trials", type=int, default=3,
                        help="Number of FL trials to average (default: 3)")
    parser.add_argument("--seed", type=int, default=2023,
                        help="Random seed (default: 2023)")
    parser.add_argument("--skip-fl", action="store_true",
                        help="Skip federated strategies (centralized only)")
    parser.add_argument("--skip-centralized", action="store_true",
                        help="Skip centralized training (FL only)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  CENTRALIZED vs FEDERATED BENCHMARK")
    print("  Dataset: Indian Personal Finance (Disposable Income)")
    print(f"  Device: {DEVICE}")
    print(f"  Centralized epochs: {args.epochs}")
    print(f"  FL rounds: {args.rounds}  |  Trials: {args.trials}")
    print(f"  FL clients: {NUM_CLIENTS}  |  Fraction fit: {FRACTION_FIT}")
    print(f"  Local epochs: {LOCAL_EPOCHS}  |  LR: {LEARNING_RATE}")
    print(f"  Seed: {args.seed}")
    print("=" * 60)

    # Preload dataset
    reset_data_cache()
    _load_and_preprocess_data()
    print(f"  Input dimension: {get_input_dim()}")

    start = time.time()

    # ── Centralized Training ──────────────────────────────────
    cent_results = {}
    if not args.skip_centralized:
        iid_metrics, iid_state = train_centralized(
            mode="iid", epochs=args.epochs,
            lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=args.seed,
        )
        cent_results["Centralized (IID)"] = iid_metrics

        noniid_metrics, noniid_state = train_centralized(
            mode="non_iid", epochs=args.epochs,
            lr=LEARNING_RATE, batch_size=BATCH_SIZE, seed=args.seed,
        )
        cent_results["Centralized (Non-IID)"] = noniid_metrics

        # Save model checkpoints
        torch.save(iid_state, RESULTS_DIR / "centralized_iid_model.pt")
        torch.save(noniid_state, RESULTS_DIR / "centralized_noniid_model.pt")

    # ── Federated Strategies ──────────────────────────────────
    fl_results = {}
    if not args.skip_fl:
        fl_results = run_federated_strategies(
            num_rounds=args.rounds, num_trials=args.trials, seed=args.seed,
        )

    elapsed = time.time() - start
    print(f"\n  Total runtime: {elapsed:.1f}s")

    # ── Results ───────────────────────────────────────────────
    if cent_results or fl_results:
        print_summary(cent_results, fl_results)

        print("\nGenerating plots...")
        plot_dashboard(cent_results, fl_results, RESULTS_DIR)
        plot_individual(cent_results, fl_results, RESULTS_DIR)
        plot_final_bars(cent_results, fl_results, RESULTS_DIR)

        print(f"\n  All outputs saved to: {RESULTS_DIR}/")
        print("  Files:")
        for f in sorted(RESULTS_DIR.iterdir()):
            print(f"    - {f.name}")
    else:
        print("\n  Nothing to plot (both centralized and FL were skipped).")

    print("\n  Benchmark complete!")


if __name__ == "__main__":
    main()
