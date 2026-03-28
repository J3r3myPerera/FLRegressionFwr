# Federated Learning for Personal Finance Prediction (Flower)

This project implements a **Federated Learning** system using the [Flower (flwr)](https://flower.ai) framework to predict disposable income from the Indian Personal Finance dataset. It compares three federated learning strategies: **FedAvg**, **FedProx**, and **SmartFedProx**.

## Project Structure

```
FlowerImplementation/
├── data/
│   └── indianPersonalFinanceAndSpendingHabits.csv   # Dataset
├── quickstart-pytorch/
│   ├── pyproject.toml                               # Flower app config & dependencies
│   └── pytorchexample/
│       ├── __init__.py
│       ├── task.py                                  # Model, data loading, train/test functions
│       ├── client_app.py                            # Flower ClientApp (train + evaluate handlers)
│       └── server_app.py                            # Flower ServerApp (FedAvg, FedProx, SmartFedProx)
└── requirements.txt
```

## Dataset

**File:** `data/indianPersonalFinanceAndSpendingHabits.csv`

**Target Variable:** `Disposable_Income` (regression — standardized during training)

**Features:** Income, Age, Dependents, Rent, Loan_Repayment, Insurance, spending categories, and potential savings. Categorical columns (`Occupation`, `City_Tier`) are one-hot encoded.

**Non-IID Partitioning:** Data is partitioned using a demographic key of `Occupation + City_Tier + Income_Bracket`. Keys are shuffled deterministically and assigned to clients via round-robin, producing realistic label and quantity skew across clients.

## Installation

Install dependencies from the app's `pyproject.toml` using pip:

```bash
cd quickstart-pytorch
pip install -e .
```

Or install the pinned dependencies manually:

```bash
pip install "flwr[simulation]>=1.20.0" "torch==2.2.2" "pandas>=2.0.0" "numpy<2" "matplotlib>=3.7.0"
```

## Running the Simulation

All commands must be run from inside the `quickstart-pytorch/` directory:

```bash
cd quickstart-pytorch
flwr run .
```

This launches a **local Flower simulation** with 12 supernodes (clients), runs all three strategies sequentially, and saves the output plots and model to the `quickstart-pytorch/` directory.

## Configuration

All parameters are defined in `quickstart-pytorch/pyproject.toml` under `[tool.flwr.app.config]`:

| Parameter | Default | Description |
|---|---|---|
| `num-server-rounds` | `10` | Number of federated rounds per strategy |
| `fraction-evaluate` | `0.7` | Fraction of clients used for evaluation |
| `local-epochs` | `3` | Local training epochs per client per round |
| `learning-rate` | `0.001` | Adam optimizer learning rate |
| `batch-size` | `64` | Batch size for local training |
| `strategy` | `"compare"` | Runs all three strategies in sequence |
| `smart-mu-init` | `0.1` | SmartFedProx: initial proximal coefficient μ |
| `smart-mu-min` | `0.001` | SmartFedProx: minimum μ |
| `smart-mu-max` | `1.0` | SmartFedProx: maximum μ |

**Simulation resources** (`[tool.flwr.federations.local-simulation]`):
- `num-supernodes = 12`
- `num-cpus = 2` per client

## Model Architecture

Defined in `pytorchexample/task.py` as `Net` — a Multi-Layer Perceptron for regression:

```
Input (input_dim features)
  → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(64)  → BatchNorm1d → ReLU → Dropout(0.2)
  → Linear(32)  → BatchNorm1d → ReLU
  → Linear(1)   → scalar output (standardized disposable income)
```

The actual `input_dim` is derived at runtime from the preprocessed dataset.

## Strategies

### FedAvg

Standard federated averaging. No proximal term (`mu=0`). Clients are selected randomly each round.

### FedProx

FedProx with a **fixed** proximal coefficient `mu=0.1`. Adds a regularization term `(mu/2) * ||w - w_global||²` to the client loss to limit drift from the global model.

### SmartFedProx

An enhanced FedProx variant implemented as `SmartFedAvg` (a `FedAvg` subclass):

- **Adaptive μ:** Per-node μ is computed each round based on that client's historical divergence relative to the global average divergence, scaled by local epochs. μ is further adjusted round-to-round based on whether test loss is improving.
- **Balanced client selection:** Clients are ranked by divergence and selected with a 30% high / 50% mid / 20% low divergence split to balance stability and convergence.
- **Boosted learning rate:** Peak LR is `1.5x` the base LR, held flat for the first 60% of rounds then decayed via cosine schedule.
- **Adaptive local epochs:** Clients run `base_epochs + 1` when `mu >= 0.07`.
- **Gradient clipping:** Gradients clipped to L2 norm of 5.0 to prevent non-IID gradient explosion.

## Client & Server Apps

**ClientApp** (`pytorchexample/client_app.py`):
- `@app.train()`: Receives global model weights and config (including `mu`, `lr`, `epochs`, `grad_clip`) from the server. Trains locally, computes model divergence (L2 norm of param diff), and returns updated weights and metrics.
- `@app.evaluate()`: Evaluates the global model on the client's local test split. Returns `eval_loss` (MSE) and `eval_r2` (R²).

**ServerApp** (`pytorchexample/server_app.py`):
- `@app.main()`: Orchestrates all three strategy runs sequentially. Each strategy gets a fresh model. Preloads centralized train/test dataloaders once and shares them across all evaluations.
- Centralized evaluation is performed after each round using the last 20% of the dataset.

## Output Files

After running, the following files are saved in `quickstart-pytorch/`:

| File | Description |
|---|---|
| `comparison_results.png` | 6-panel plot: R², MSE (test), training loss, model divergence (L2), effective μ, and a final summary table |
| `r2_score_comparison.png` | R² score progression per round for all three strategies |
| `mse_loss_comparison.png` | MSE loss progression per round for all three strategies |
| `final_model.pt` | Saved PyTorch state dict of the final SmartFedProx model |

## Metrics Tracked Per Round

- **R² Score** — coefficient of determination on centralized test set
- **MSE Loss** — mean squared error on centralized test set
- **Training Loss** — MSE on centralized training set
- **Model Divergence** — L2 norm of parameter change between rounds
- **Effective μ** — proximal coefficient used (0 for FedAvg, 0.1 for FedProx, adaptive for SmartFedProx)
