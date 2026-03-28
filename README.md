# Federated Learning for Personal Finance Prediction

This project implements a **Federated Learning** system for predicting disposable income from the Indian Personal Finance dataset. It contains two separate implementations:

1. **FLRegression** — a custom simulation framework comparing FedAvg, FedProx, and SmartFedProx
2. **FlowerImplementation** — the same three strategies rebuilt on the [Flower (flwr)](https://flower.ai) federated learning framework

---

## Project Structure

```
FLRegressionFlwr/
├── FLRegression/
│   ├── dataset.py                    # Dataset loading and non-IID partitioning
│   ├── module.py                     # Model definition, training, configuration
│   ├── client.py                     # SimulatedClient class for local training
│   ├── server.py                     # FederatedSimulator with WandB integration
│   ├── main.py                       # Entry point — runs 3 trials, logs to WandB
│   └── resultOutput/                 # Generated plots and saved models (gitignored)
├── FlowerImplementation/
│   ├── data/
│   │   └── indianPersonalFinanceAndSpendingHabits.csv
│   ├── quickstart-pytorch/
│   │   ├── pyproject.toml            # Flower app config and dependencies
│   │   └── pytorchexample/
│   │       ├── task.py               # Model, data loading, train/test functions
│   │       ├── client_app.py         # Flower ClientApp
│   │       └── server_app.py         # Flower ServerApp (all three strategies)
│   └── README.md                     # Flower implementation details
├── data/
│   ├── indianPersonalFinanceAndSpendingHabits.csv
│   ├── DFA.ipynb                     # Data feature analysis notebook
│   └── visualize_clients.py          # Client data distribution visualizer
├── model/
│   └── ModelTesting.ipynb            # Centralized model architecture comparison
├── api.py                            # FastAPI backend for running simulations
├── static/
│   ├── index.html                    # Web dashboard
│   ├── app.js                        # Interactive Plotly frontend
│   └── styles.css                    # Dark-themed UI styling
├── outputs/                          # Generated output PNGs (gitignored)
├── tests/
│   └── test_basic.py
├── scripts/
│   └── run_tests.sh
├── requirements.txt
└── .github/workflows/
    ├── ci.yml
    └── test.yml
```

---

## Dataset

**File:** `data/indianPersonalFinanceAndSpendingHabits.csv`

**Target Variable:** `Disposable_Income` (regression task — standardized during training)

**Features:** Income, Age, Dependents, Rent, Loan_Repayment, Insurance, spending categories, and potential savings. Categorical columns (`Occupation`, `City_Tier`) are one-hot encoded.

**Non-IID Partitioning:** Data is partitioned by a composite key of `Occupation + City_Tier + Income_Bracket`, producing realistic label and quantity skew across clients.

---

## FLRegression — Custom Simulation

### Running the Simulation

```bash
cd FLRegression
python main.py
```

This runs **3 trials** of each strategy, averages the results, logs all metrics to **WandB**, and saves plots and models to `FLRegression/resultOutput/`.

### Configuration

Defined in `FLRegression/module.py`:

| Parameter | Value | Description |
|---|---|---|
| `NUM_ROUNDS` | `20` | Federated rounds per strategy |
| `NUM_CLIENTS` | `12` | Total number of simulated clients |
| `FRACTION_FIT` | `0.7` | Fraction of clients selected per round |
| `LOCAL_EPOCHS` | `3` | Local training epochs per client |
| `LEARNING_RATE` | `0.001` | Adam optimizer learning rate |
| `BATCH_SIZE` | `64` | Batch size for local training |

### Output Files

Saved to `FLRegression/resultOutput/` (gitignored):

| File | Description |
|---|---|
| `comparison_results.png` | 6-panel plot: R², MSE, training loss, divergence, μ, final bar chart |
| `r2_comparison.png` | R² score progression per round |
| `mse_comparison.png` | MSE loss progression per round |
| `FedAvg_final_model.pt` | Final FedAvg model weights |
| `FedProx_final_model.pt` | Final FedProx model weights |
| `SmartFedProx_final_model.pt` | Final SmartFedProx model weights |

### WandB Integration

Each strategy run across each trial is logged to WandB under project `fl-regression`. Metrics logged per round: R² score, MSE loss, training loss, model divergence, and effective μ.

---

## FlowerImplementation — Flower Framework

### Running the Simulation

```bash
cd FlowerImplementation/quickstart-pytorch
flwr run .
```

Launches a local Flower simulation with 12 supernodes, runs all three strategies sequentially, and saves plots to the `quickstart-pytorch/` directory.

See [FlowerImplementation/README.md](FlowerImplementation/README.md) for full details on configuration and output.

---

## Strategies

### FedAvg

Baseline federated averaging. No proximal term (`mu=0`), random client selection each round.

### FedProx

FedProx with a fixed proximal coefficient `mu=0.1`. Adds `(mu/2) * ||w - w_global||²` to the client loss to limit drift from the global model. Random client selection.

### SmartFedProx

An enhanced FedProx variant with three key additions:

- **Adaptive μ (client-side):** Per-client μ is computed each round based on that client's historical divergence relative to the global average, scaled by local epochs. μ is clamped to `[0.001, 1.0]`.
- **Hybrid client selection:** After a 3-round random cold start and excluding 15% random exploration rounds, clients are ranked by smoothed divergence history and selected with a **30% high / 50% mid / 20% low** divergence split.
- **Server-side μ adaptation:** After each round the server adjusts the global μ: relaxed (×0.97) if loss is improving, spiked (×1.5) if divergence is detected, or gently tightened (×1.1) on plateau.

---

## Model Architecture

Both implementations share the same MLP (`Net`) for regression:

```
Input (dynamic input_dim)
  → Linear(128) → BatchNorm1d → ReLU → Dropout(0.3)
  → Linear(64)  → BatchNorm1d → ReLU → Dropout(0.2)
  → Linear(32)  → BatchNorm1d → ReLU
  → Linear(1)   → scalar output (disposable income)
```

---

## FastAPI Web Dashboard

```bash
uvicorn api:app --reload --port 8000
```

Open `http://localhost:8000` in your browser.

**API Endpoints:**

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web dashboard |
| `/api/config` | GET | Server configuration |
| `/api/simulate` | POST | Run simulation with custom parameters |
| `/api/results/{run_id}` | GET | Fetch results for a run |
| `/docs` | GET | Interactive OpenAPI docs |

The dashboard provides a dark-themed UI with real-time Plotly charts for R², MSE, training loss, divergence, and μ across all three strategies.

---

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `torch`, `pandas`, `scikit-learn`, `numpy`, `matplotlib`, `wandb`, `plotly`, `fastapi`, `uvicorn`.

For the Flower implementation, install separately:

```bash
cd FlowerImplementation/quickstart-pytorch
pip install -e .
```

---

## CI/CD Pipeline

GitHub Actions runs on every push and pull request:

- **Code quality:** flake8 linting, Black and isort formatting checks
- **Tests:** pytest with coverage reporting
- **Multi-Python:** Python 3.10 and 3.11

### Running Tests Locally

```bash
pip install pytest pytest-cov flake8
pytest tests/ -v
./scripts/run_tests.sh
```

---

## AWS EC2 Deployment

See `EC2_GUIDE.txt` for the full setup guide. Recommended instance: `c5.xlarge` (4 vCPUs, 8 GB RAM) on Ubuntu 24.04 LTS.

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```
