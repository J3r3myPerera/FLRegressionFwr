# Federated Learning for Personal Finance Prediction

This project implements a **Federated Learning** system for predicting disposable income using the Indian Personal Finance dataset. It compares three federated learning strategies: **FedAvg**, **FedProx**, and **SmartFedProx** with adaptive μ and hybrid client selection.

## Project Structure

```
FLRegressionFlwr/
├── api.py                                  # FastAPI backend for running simulations
├── requirements.txt                        # Project dependencies
├── EC2_GUIDE.txt                           # AWS EC2 deployment guide
├── centralized_model_comparison.ipynb      # Model selection & comparison notebook
├── pytest.ini                              # Test configuration
├── README.md                               # This file
├── FLRegression/
│   ├── dataset.py                          # Dataset loading and preprocessing
│   ├── module.py                           # Model definition, training functions, and configuration
│   ├── client.py                           # SimulatedClient class for local training
│   ├── server.py                           # FederatedSimulator with server-side adaptive μ
│   ├── main.py                             # Main entry point for running simulations
│   └── run_comparison.py                   # Comparison script for all three strategies
├── static/
│   ├── index.html                          # Web interface
│   ├── app.js                              # Interactive frontend with Plotly charts
│   └── styles.css                          # Dark-themed UI styling
├── data/
│   └── indianPersonalFinanceAndSpendingHabits.csv
├── tests/
│   └── test_basic.py                       # Unit tests
├── scripts/
│   └── run_tests.sh                        # Test runner script
└── .github/workflows/
    ├── ci.yml                              # Main CI pipeline
    └── test.yml                            # Unit test pipeline
```

## Dataset

The dataset (`indianPersonalFinanceAndSpendingHabits.csv`) is located in the `data/` directory.

**Target Variable:** `Disposable_Income` (regression task)

**Features:** Income, Age, Dependents, Rent, Loan_Repayment, Insurance, various spending categories, and potential savings.

**Non-IID Partitioning:** The data is partitioned using extreme non-IID strategy:
- Primary split by Occupation + City_Tier + Income_Bracket
- Label skew: Some clients only see high/low disposable income samples
- Quantity skew: Uneven data distribution across clients

## Installation

### Prerequisites

- Python 3.10+
- PyTorch

### Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install torch pandas scikit-learn numpy matplotlib plotly fastapi "uvicorn[standard]"
```

## Running the Project

### Run Main Simulation

Run the main federated learning simulation comparing all three strategies:

```bash
cd FLRegression
python main.py
```

This will:
- Run simulations for FedAvg, FedProx, and SmartFedProx
- Generate comparison plots (R² score, MSE loss, training loss, divergence, effective μ)
- Save results to `comparison_results.png`, `r2_comparison.png`, and `mse_comparison.png`

### Run Comparison Script

Run the detailed comparison script with multiple trials:

```bash
python run_comparison.py
```

This runs multiple trials for statistical significance and generates comprehensive comparison plots.

### Run FastAPI Web Interface

Launch the API and web dashboard:

```bash
uvicorn api:app --reload --port 8000
```

Then open `http://localhost:8000` in your browser.

**API Endpoints:**
| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Serve the web dashboard |
| `/api/config` | GET | Get server configuration |
| `/api/simulate` | POST | Run a simulation with custom parameters |
| `/api/results/{run_id}` | GET | Fetch results for a specific run |
| `/docs` | GET | Interactive OpenAPI documentation |

**Configurable Simulation Parameters:**
- Number of rounds (5–50)
- Number of trials (1–5)
- Number of clients (2–50)
- Fraction fit (0.1–1.0)
- Local epochs (1–20)
- Batch size (8–512)
- Learning rate (0.00001–0.1)

## Web Dashboard

The project includes a modern, interactive web interface built with vanilla JS and Plotly:

- **Dark-themed UI** with color-coded strategy cards (FedAvg in red, FedProx in blue, SmartFedProx in green)
- **Real-time simulation tracking** with progress updates
- **Interactive Plotly charts** for visualizing R² scores, MSE loss, training loss, divergence, and μ progression
- **Configurable parameters** via the dashboard before launching simulations

## Strategies Compared

1. **FedAvg**: Baseline federated averaging (μ=0, random client selection)
2. **FedProx**: FedProx with fixed μ=0.1 and random client selection
3. **SmartFedProx**: FedProx with adaptive μ and hybrid client selection
   - Cold start: 3 rounds of random selection before switching to hybrid
   - 15% exploration rate for diversity
   - Hybrid selection: 30% high-divergence, 50% middle, 20% low-divergence clients

## Server-Side Adaptive μ Computation

The server dynamically adjusts the proximal coefficient (μ) each round based on global performance:

- **Loss improving** → μ relaxed (×0.97) to allow faster convergence
- **Divergence detected** → μ spiked (×1.5) to pull clients back toward the global model
- **Convergence plateau** → μ gently tightened (×1.1) to refine the model

Global divergence is tracked using an Exponential Moving Average (EMA) across all participating clients, giving the server a smooth signal to react to.

## Configuration

Key configuration parameters are defined in `FLRegression/module.py`:

- `NUM_ROUNDS = 20`: Number of federated learning rounds
- `NUM_CLIENTS = 10`: Number of clients
- `FRACTION_FIT = 0.5`: Fraction of clients selected per round
- `LOCAL_EPOCHS = 3`: Local training epochs per client
- `LEARNING_RATE = 0.001`: Learning rate for Adam optimizer
- `BATCH_SIZE = 64`: Batch size for training

## Model Architecture

The model is a Multi-Layer Perceptron (MLP) for regression:
- Input: 26 features (after preprocessing)
- Hidden layers: 128 → 64 → 32 neurons
- Output: 1 neuron (disposable income prediction)
- Activation: ReLU with BatchNorm and Dropout (0.3, 0.2)

A centralized model comparison notebook (`centralized_model_comparison.ipynb`) is included for evaluating and comparing different model architectures on the dataset.

## Key Features

- **Extreme Non-IID Data Partitioning**: Realistic heterogeneous data distribution
- **Server-Side Adaptive μ**: Server dynamically adjusts the proximal coefficient based on global loss and divergence trends
- **Client-Side Adaptive μ**: Clients further adjust μ based on local divergence
- **Hybrid Client Selection**: Balances high, middle, and low divergence clients for stability
- **FastAPI REST API**: Production-ready backend with configurable simulation parameters
- **Interactive Web Dashboard**: Real-time dark-themed UI with Plotly visualizations
- **AWS EC2 Deployment**: Ready-to-deploy on `c5.xlarge` instances (~$0.17/hr)
- **Comprehensive Metrics**: Tracks R² score, MSE loss, training loss, model divergence, and effective μ

## AWS EC2 Deployment

The project can be deployed on AWS EC2 for remote access. See `EC2_GUIDE.txt` for the full setup guide.

**Quick start:**
```bash
# SSH into your instance
ssh -i your-key.pem ubuntu@<your-ec2-ip>

# Clone and install
git clone <repo-url> && cd FLRegressionFlwr
pip install -r requirements.txt

# Launch the server
uvicorn api:app --host 0.0.0.0 --port 8000
```

**Recommended instance:** `c5.xlarge` (4 vCPUs, 8GB RAM) on Ubuntu 24.04 LTS.

## CI/CD Pipeline

This project includes a comprehensive CI/CD pipeline using GitHub Actions:

- **Automated Testing**: Runs on every push and pull request
- **Code Quality Checks**: Linting with flake8, formatting checks with Black and isort
- **Simulation Validation**: Quick and full simulation tests
- **Multi-Python Support**: Tests on Python 3.10 and 3.11
- **Coverage Reporting**: pytest with Codecov integration

### Running Tests Locally

```bash
# Install test dependencies
pip install -r requirements.txt
pip install pytest pytest-cov flake8

# Run all tests
pytest tests/ -v

# Run quick CI simulation
./scripts/run_tests.sh
```

## Output Files

After running simulations, the following files are generated:

- `comparison_results.png`: Comprehensive 6-panel comparison plot
- `r2_comparison.png`: R² score progression comparison
- `mse_comparison.png`: MSE loss progression comparison

## For More Details

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed documentation on:
- Client selection strategies
- FedProx algorithm implementation
- Adaptive μ computation
- Data flow and architecture
