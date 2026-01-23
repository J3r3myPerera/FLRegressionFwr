"""pytorchexample: A Flower / PyTorch app with smart client selection for Personal Finance."""

import torch
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp

from pytorchexample.strategy import SelectionStrategy, SmartFedProx
from pytorchexample.task import Net, get_input_dim, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()

# Global storage for tracking metrics across rounds
_round_metrics = {"rounds": [], "r2_scores": [], "mse_losses": []}


def plot_training_progress():
    """Plot R² score progression across federated learning rounds."""
    if not _round_metrics["rounds"]:
        print("No metrics to plot.")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    rounds = _round_metrics["rounds"]
    r2_scores = _round_metrics["r2_scores"]
    mse_losses = _round_metrics["mse_losses"]
    
    # Plot R² Score
    ax1.plot(rounds, r2_scores, 'b-o', linewidth=2, markersize=8, label='R² Score')
    ax1.set_xlabel('Federated Round', fontsize=12)
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Global Model R² Score vs. Federated Rounds', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([min(0, min(r2_scores) - 0.1), max(1, max(r2_scores) + 0.1)])
    
    # Add value annotations
    for i, (r, score) in enumerate(zip(rounds, r2_scores)):
        ax1.annotate(f'{score:.4f}', (r, score), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9)
    
    # Plot MSE Loss
    ax2.plot(rounds, mse_losses, 'r-s', linewidth=2, markersize=8, label='MSE Loss')
    ax2.set_xlabel('Federated Round', fontsize=12)
    ax2.set_ylabel('MSE Loss', fontsize=12)
    ax2.set_title('Global Model MSE Loss vs. Federated Rounds', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add value annotations
    for i, (r, loss) in enumerate(zip(rounds, mse_losses)):
        ax2.annotate(f'{loss:.4f}', (r, loss), textcoords="offset points", 
                     xytext=(0, 10), ha='center', fontsize=9)
    
    plt.tight_layout()
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    print("\n✓ Training progress plot saved to 'training_progress.png'")


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""
    
    # Reset metrics for new run
    _round_metrics["rounds"].clear()
    _round_metrics["r2_scores"].clear()
    _round_metrics["mse_losses"].clear()

    # Read run config
    fraction_fit: float = context.run_config["fraction-fit"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]
    proximal_mu: float = context.run_config["proximal-mu"]

    # Adaptive μ configuration
    adaptive_mu_enabled: bool = context.run_config.get("adaptive-mu-enabled", False)
    mu_min: float = context.run_config.get("mu-min", 0.001)
    mu_max: float = context.run_config.get("mu-max", 1.0)

    # Smart client selection configuration
    selection_strategy: str = context.run_config.get(
        "selection-strategy", SelectionStrategy.RANDOM
    )
    selection_temperature: float = context.run_config.get("selection-temperature", 1.0)
    hybrid_high_ratio: float = context.run_config.get("hybrid-high-ratio", 0.5)
    cold_start_rounds: int = context.run_config.get("cold-start-rounds", 2)
    exploration_rate: float = context.run_config.get("exploration-rate", 0.1)

    # Load global model with correct input dimension
    input_dim = get_input_dim()
    global_model = Net(input_dim=input_dim)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize SmartFedProx strategy with intelligent client selection
    strategy = SmartFedProx(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        proximal_mu=proximal_mu,
        selection_strategy=selection_strategy,
        selection_temperature=selection_temperature,
        hybrid_high_ratio=hybrid_high_ratio,
        cold_start_rounds=cold_start_rounds,
        exploration_rate=exploration_rate,
    )

    # Build train config with adaptive μ settings
    train_config = ConfigRecord({
        "lr": lr,
        "proximal_mu": proximal_mu,
        "adaptive_mu_enabled": adaptive_mu_enabled,
        "mu_min": mu_min,
        "mu_max": mu_max,
    })

    # Start strategy, run FedProx for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=train_config,
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    # Print client selection statistics
    print("\n--- Client Selection Statistics ---")
    for node_id, stats in strategy.get_client_stats().items():
        print(
            f"  Node {node_id}: participated {stats['participation_count']}x, "
            f"avg_div={stats['avg_divergence']:.4f}, "
            f"latest_div={stats['latest_divergence']:.4f}"
        )

    # Save final model to disk
    print("\nSaving final model to disk...")
    state_dict = result.arrays.to_torch_state_dict()
    torch.save(state_dict, "final_model.pt")
    
    # Plot training progress
    plot_training_progress()


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    input_dim = get_input_dim()
    model = Net(input_dim=input_dim)
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_r2 = test(model, test_dataloader, device)
    
    # Store metrics for plotting
    _round_metrics["rounds"].append(server_round)
    _round_metrics["r2_scores"].append(test_r2)
    _round_metrics["mse_losses"].append(test_loss)
    
    print(f"  Round {server_round}: R² = {test_r2:.4f}, MSE = {test_loss:.4f}")

    # Return the evaluation metrics (R² for regression instead of accuracy)
    return MetricRecord({"r2_score": test_r2, "mse_loss": test_loss})
