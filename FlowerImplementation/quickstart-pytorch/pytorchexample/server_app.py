"""Flower server comparing FedAvg, FedProx, and SmartFedProx."""

import math
import random
from logging import INFO

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.common import MessageType, log
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import (
    Net,
    get_input_dim,
    load_centralized_test,
    load_centralized_train,
    set_seed,
    test,
)

set_seed()
app = ServerApp()

# Adaptive mu computation
def compute_adaptive_mu(
    base_mu: float,
    historical_divergence: float,
    global_avg_divergence: float,
    local_epochs: int,
    mu_min: float = 0.001,
    mu_max: float = 1.0,
) -> float:
    # Factor 1: Divergence-based scaling
    if global_avg_divergence > 0 and historical_divergence > 0:
        divergence_ratio = historical_divergence / (global_avg_divergence + 1e-8)
        divergence_factor = 1.0 + 0.5 * (divergence_ratio - 1.0)   # dampened
        divergence_factor = max(0.5, min(2.0, divergence_factor))   # clamp [0.5, 2]
    else:
        divergence_factor = 1.0

    # Factor 2: Local epochs scaling
    epoch_factor = 1.0 + 0.1 * (local_epochs - 1)

    adaptive_mu = base_mu * divergence_factor * epoch_factor
    return max(mu_min, min(mu_max, adaptive_mu))

# SmartFedAvg — custom strategy with balanced client selection

class SmartFedAvg(FedAvg):
    """FedAvg subclass that adds two SmartFedProx-specific behaviours"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.divergence_history: dict = {}   # {node_id: float}
        self.global_avg_divergence: float = 0.0
        # Set by _run_smart_fedprox before each round call
        self.base_mu: float = 0.1
        self.round_epochs: int = 1

    # Balanced selection
    def _balanced_select(self, all_node_ids: list) -> list:
        """Return node IDs split 30 % high / 50 % mid / 20 % low divergence."""
        n = len(all_node_ids)
        target = max(self.min_train_nodes, int(n * self.fraction_train))

        if not self.divergence_history:
            # No history yet — use all available nodes
            return list(all_node_ids)

        sorted_ids = sorted(
            all_node_ids,
            key=lambda nid: self.divergence_history.get(nid, self.global_avg_divergence),
        )

        t1, t2 = n // 3, 2 * n // 3
        low_tier  = sorted_ids[:t1]
        mid_tier  = sorted_ids[t1:t2]
        high_tier = sorted_ids[t2:]

        n_high = max(1, round(target * 0.30))
        n_mid  = max(1, round(target * 0.50))
        n_low  = max(1, round(target * 0.20))

        rng = random.Random(42)
        selected = (
            rng.sample(high_tier, min(n_high, len(high_tier)))
            + rng.sample(mid_tier,  min(n_mid,  len(mid_tier)))
            + rng.sample(low_tier,  min(n_low,  len(low_tier)))
        )
        return selected
    
    # Override: balanced selection + per-node adaptive mu
    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> list:
        all_node_ids = list(grid.get_node_ids())
        selected_ids = self._balanced_select(all_node_ids)

        messages = []
        for node_id in selected_ids:
            hist_div = self.divergence_history.get(node_id, self.global_avg_divergence)
            per_node_mu = compute_adaptive_mu(
                base_mu=self.base_mu,
                historical_divergence=hist_div,
                global_avg_divergence=self.global_avg_divergence,
                local_epochs=self.round_epochs,
            )

            # Build a fresh ConfigRecord for this node
            node_config = ConfigRecord({"server-round": server_round})
            node_config["lr"] = config["lr"]
            node_config["mu"] = per_node_mu
            for key in ("grad_clip", "epochs"):
                try:
                    node_config[key] = config[key]
                except KeyError:
                    pass

            record = RecordDict({
                self.arrayrecord_key:    arrays,
                self.configrecord_key:   node_config,
            })
            messages.append(Message(
                content=record,
                message_type=MessageType.TRAIN,
                dst_node_id=node_id,
            ))

        log(
            INFO,
            "[SmartFedAvg] Round %d: selected %d / %d nodes  "
            "(30%% high / 50%% mid / 20%% low divergence)",
            server_round, len(selected_ids), len(all_node_ids),
        )
        return messages

    # Override: capture per-node divergences for next round's selection
    def aggregate_train(self, server_round: int, replies):
        # Materialise so we can read metrics then pass to super
        replies = list(replies)
        for msg in replies:
            if not msg.has_error():
                node_id = msg.metadata.src_node_id
                try:
                    div = float(msg.content["metrics"]["divergence"])
                    self.divergence_history[node_id] = div
                except (KeyError, TypeError):
                    pass

        if self.divergence_history:
            self.global_avg_divergence = (
                sum(self.divergence_history.values()) / len(self.divergence_history)
            )

        return super().aggregate_train(server_round, replies)

# Evaluation helpers
def _evaluate_model(arrays: ArrayRecord, dataloader):
    """Evaluate global model on a dataloader. Returns (mse, r2)."""
    input_dim = get_input_dim()
    model = Net(input_dim)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return test(model, dataloader, device)


def _make_evaluate_fn():
    """Create the evaluate_fn callback required by strategy.start()."""
    def evaluate_fn(server_round: int, arrays: ArrayRecord) -> MetricRecord:
        testloader = load_centralized_test()
        mse, r2 = _evaluate_model(arrays, testloader)
        return MetricRecord({"mse_loss": mse, "r2_score": r2})
    return evaluate_fn

# Round-by-round helpers
def _snapshot(arrays):
    """Clone current model state for divergence computation."""
    return {k: v.clone() for k, v in arrays.to_torch_state_dict().items()}


def _divergence(prev_state, arrays):
    """L2 norm of model parameter change between rounds."""
    new_state = arrays.to_torch_state_dict()
    return sum(
        ((new_state[k] - prev_state[k]) ** 2).sum().item()
        for k in prev_state
    ) ** 0.5


def _one_round(grid, arrays, lr, mu, fraction_evaluate,
               epochs=None, grad_clip=0.0, strategy=None):
    """Execute one FL round and return updated arrays.
    Passing in a pre-built strategy instance (e.g. SmartFedAvg) to preserve
    state across rounds.  Omit it to get a fresh FedAvg each time.
    """
    config = {"lr": lr, "mu": mu}
    if epochs is not None:
        config["epochs"] = float(epochs)
    if grad_clip > 0:
        config["grad_clip"] = grad_clip
    strat = strategy if strategy is not None else FedAvg(fraction_evaluate=fraction_evaluate)
    result = strat.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(config),
        num_rounds=1,
        evaluate_fn=_make_evaluate_fn(),
    )
    return result.arrays


def _collect_round_metrics(arrays, prev_state, mu, testloader, trainloader):
    #Evaluate model and compute all metrics for one round.
    div = _divergence(prev_state, arrays)
    test_mse, r2 = _evaluate_model(arrays, testloader)
    train_mse, _ = _evaluate_model(arrays, trainloader)
    return {
        "mse_loss": test_mse,
        "r2_score": r2,
        "divergence": div,
        "avg_loss": train_mse,
        "mu": mu,
    }


def _new_history():
    return {"mse_loss": [], "r2_score": [], "divergence": [], "avg_loss": [], "mu": []}


def _append(history, m):
    for k in history:
        history[k].append(m[k])

# Strategy runners — all return (final_arrays, history_dict)
def _run_fedavg(grid, arrays, lr, num_rounds, fraction_evaluate, testloader, trainloader, **_):
    """FedAvg baseline: no proximal term (mu=0), no extras."""
    print("\n[FedAvg] mu=0.0 — baseline")
    history = _new_history()

    for rnd in range(1, num_rounds + 1):
        prev = _snapshot(arrays)
        arrays = _one_round(grid, arrays, lr, 0.0, fraction_evaluate)
        m = _collect_round_metrics(arrays, prev, 0.0, testloader, trainloader)
        _append(history, m)
        print(f"  Round {rnd}: R²={m['r2_score']:.4f}  MSE={m['mse_loss']:.4f}  Div={m['divergence']:.4f}")

    return arrays, history


def _run_fedprox(grid, arrays, lr, num_rounds, fraction_evaluate, testloader, trainloader, **_):
    #FedProx with fixed mu=0.1
    print("\n[FedProx] mu=0.1 — fixed proximal term")
    history = _new_history()

    for rnd in range(1, num_rounds + 1):
        prev = _snapshot(arrays)
        arrays = _one_round(grid, arrays, lr, 0.1, fraction_evaluate)
        m = _collect_round_metrics(arrays, prev, 0.1, testloader, trainloader)
        _append(history, m)
        print(f"  Round {rnd}: R²={m['r2_score']:.4f}  MSE={m['mse_loss']:.4f}  Div={m['divergence']:.4f}")

    return arrays, history


def _run_smart_fedprox(grid, arrays, lr, num_rounds, fraction_evaluate,
                        testloader, trainloader, base_epochs=1,
                        mu_init=0.1, mu_min=0.06, mu_max=0.2):
    #SmartFedProx — starts from FedProx's proven baseline
    print("\n[SmartFedProx] Balanced selection + adaptive mu + boosted LR + grad clipping")
    history = _new_history()

    # Persistent strategy — divergence_history survives across rounds
    smart_strategy = SmartFedAvg(fraction_evaluate=fraction_evaluate)

    # Start from configured mu. Decay slowly so we stay competitive.
    mu = mu_init
    prev_loss = float("inf")
    best_loss = float("inf")

    # 1.5x peak LR — safe because grad clipping is active
    peak_lr = lr * 1.5
    # Keep peak LR for first 60% of rounds, then cosine decay
    decay_start = max(1, int(num_rounds * 0.6))

    for rnd in range(1, num_rounds + 1):
        # LR schedule: flat peak then cosine decay
        if rnd <= decay_start:
            current_lr = peak_lr
        else:
            progress = (rnd - decay_start) / max(1, num_rounds - decay_start)
            current_lr = peak_lr * (0.2 + 0.8 * 0.5 * (1.0 + math.cos(math.pi * progress)))

        # Adaptive local epochs: exploit strong proximal term 
        round_epochs = base_epochs + 1 if mu >= 0.07 else None

        # Update strategy state for this round 
        smart_strategy.base_mu = mu
        smart_strategy.round_epochs = (base_epochs + 1) if mu >= 0.07 else base_epochs

        # Run one federated round via SmartFedAvg 
        prev = _snapshot(arrays)
        arrays = _one_round(
            grid, arrays, current_lr, mu, fraction_evaluate,
            grad_clip=5.0,
            epochs=round_epochs,
            strategy=smart_strategy,
        )

        # Collect metrics
        m = _collect_round_metrics(arrays, prev, mu, testloader, trainloader)
        _append(history, m)

        cur_loss = m["mse_loss"]
        best_loss = min(best_loss, cur_loss)

        if cur_loss < prev_loss:
            # Improving — relax proximal SLOWLY (stays near 0.1 for many rounds)
            mu = max(mu * 0.97, mu_min)
        elif cur_loss > best_loss * 1.05:
            # Noticeably worse than best — strong correction
            mu = min(mu * 1.5, mu_max)
        else:
            # Slight worsening or plateau — gentle tightening
            mu = min(mu * 1.1, mu_max)

        prev_loss = cur_loss

        print(
            f"  Round {rnd}: R²={m['r2_score']:.4f}  MSE={cur_loss:.4f}  "
            f"Div={m['divergence']:.4f}  mu={mu:.4f}  lr={current_lr:.5f}  "
            f"epochs={round_epochs or base_epochs}"
        )

    return arrays, history


# Plotting
def _plot_comparison(all_metrics, num_rounds):
    rounds = list(range(1, num_rounds + 1))
    strategies = list(all_metrics.keys())
    colors = {"FedAvg": "#1f77b4", "FedProx": "#ff7f0e", "SmartFedProx": "#2ca02c"}

    #Main 6-panel figure
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))

    panels = [
        ("r2_score",   "R² Score",                     "R²",         axes[0, 0]),
        ("mse_loss",   "MSE Loss (Test)",               "MSE",        axes[0, 1]),
        ("avg_loss",   "Average Training Loss",         "Loss",       axes[1, 0]),
        ("divergence", "Model Divergence (L2)",         "Divergence", axes[1, 1]),
        ("mu",         "Effective μ (Proximal Coeff.)", "μ",          axes[2, 0]),
    ]

    for key, title, ylabel, ax in panels:
        for s in strategies:
            ax.plot(rounds, all_metrics[s][key], label=s, color=colors[s], linewidth=2)
        ax.set_title(title, fontsize=13)
        ax.set_xlabel("Round")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Summary table in the last panel
    ax = axes[2, 1]
    ax.axis("off")
    header = ["Strategy", "Final R²", "Final MSE", "Final Loss"]
    rows = []
    for s in strategies:
        h = all_metrics[s]
        rows.append([
            s,
            f"{h['r2_score'][-1]:.4f}",
            f"{h['mse_loss'][-1]:.4f}",
            f"{h['avg_loss'][-1]:.4f}",
        ])
    table = ax.table(
        cellText=rows, colLabels=header, loc="center",
        cellLoc="center", colColours=["#f0f0f0"] * 4,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)
    ax.set_title("Final Round Summary", fontsize=13)

    fig.suptitle(
        "Federated Learning Strategy Comparison\n(Non-IID Personal Finance Data)",
        fontsize=15, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig("comparison_results.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\nSaved comparison_results.png")

    # Individual metric plots
    for key, ylabel, title in [
        ("r2_score", "R²", "R² Score Comparison"),
        ("mse_loss", "MSE", "MSE Loss Comparison"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for s in strategies:
            ax.plot(
                rounds, all_metrics[s][key], label=s,
                color=colors[s], linewidth=2, marker="o", markersize=4,
            )
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Round", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = f"{key}_comparison.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved {filename}")


# Main entry point
@app.main()
def main(grid: Grid, context: Context) -> None:
    """Run all three strategies and generate comparison plots."""
    num_rounds = int(context.run_config["num-server-rounds"])
    lr = float(context.run_config["learning-rate"])
    fraction_evaluate = float(context.run_config["fraction-evaluate"])
    try:
        base_epochs = int(context.run_config["local-epochs"])
    except (KeyError, TypeError):
        base_epochs = 1

    input_dim = get_input_dim()

    # Preload dataloaders once (shared across all evaluations)
    testloader = load_centralized_test()
    trainloader = load_centralized_train()

    print("=" * 60)
    print("Federated Learning Strategy Comparison")
    print(f"Rounds: {num_rounds} | LR: {lr} | Clients: 10")
    print(f"Model: Net ({input_dim} -> 128 -> 64 -> 32 -> 1)")
    print("Strategies: FedAvg, FedProx, SmartFedProx")
    print("=" * 60)

    all_metrics = {}
    runners = [
        ("FedAvg",        _run_fedavg),
        ("FedProx",       _run_fedprox),
        ("SmartFedProx",  _run_smart_fedprox),
    ]

    final_arrays = None
    for name, runner in runners:
        # Fresh model for each strategy
        model = Net(input_dim)
        arrays = ArrayRecord(model.state_dict())
        extra = {
            "base_epochs": base_epochs,
            "mu_init": float(context.run_config.get("smart-mu-init", 0.1)),
            "mu_min": float(context.run_config.get("smart-mu-min", 0.06)),
            "mu_max": float(context.run_config.get("smart-mu-max", 0.2)),
        } if name == "SmartFedProx" else {}
        arrays, history = runner(grid, arrays, lr, num_rounds, fraction_evaluate, testloader, trainloader, **extra)
        all_metrics[name] = history
        final_arrays = arrays

    # Generate comparison plots
    _plot_comparison(all_metrics, num_rounds)

    # Save final SmartFedProx model
    print("\nSaving final SmartFedProx model to final_model.pt")
    torch.save(final_arrays.to_torch_state_dict(), "final_model.pt")

    # Print final summary
    print("\n" + "=" * 60)
    print("Final Results (last round):")
    print("-" * 60)
    for name in all_metrics:
        h = all_metrics[name]
        print(
            f"  {name:15s}: R²={h['r2_score'][-1]:.4f}  "
            f"MSE={h['mse_loss'][-1]:.4f}  Loss={h['avg_loss'][-1]:.4f}"
        )
    print("=" * 60)
