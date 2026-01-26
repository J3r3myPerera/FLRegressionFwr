# Federated Learning for Personal Finance Prediction

This project implements a **Federated Learning** system for predicting disposable income using the Indian Personal Finance dataset. It compares three federated learning strategies: **FedAvg**, **FedProx**, and **SmartFedProx** with adaptive μ and hybrid client selection.

## Project Structure

```
FLRegression/
├── dataset.py              # Dataset loading and preprocessing
├── module.py               # Model definition, training functions, and configuration
├── client.py               # SimulatedClient class for local training
├── server.py               # FederatedSimulator class for orchestration
├── main.py                 # Main entry point for running simulations
├── run_comparison.py       # Comparison script for all three strategies
├── README.md               # This file
└── IMPLEMENTATION_PLAN.md  # Detailed implementation documentation
```

## Dataset

The dataset (`indianPersonalFinanceAndSpendingHabits.csv`) is located at:
```
/Users/dinukaperera/FLRegressionFlwr/data/indianPersonalFinanceAndSpendingHabits.csv
```

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
- NumPy, Pandas, scikit-learn
- Matplotlib

### Install Dependencies

```bash
pip install torch pandas scikit-learn numpy matplotlib
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

## Strategies Compared

1. **FedAvg**: Baseline federated averaging (μ=0, random client selection)
2. **FedProx**: FedProx with fixed μ=0.1 and random client selection
3. **SmartFedProx**: FedProx with adaptive μ and hybrid client selection

## Configuration

Key configuration parameters are defined in `module.py`:

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
- Activation: ReLU with BatchNorm and Dropout

## Key Features

- **Extreme Non-IID Data Partitioning**: Realistic heterogeneous data distribution
- **Adaptive Proximal Coefficient (μ)**: Dynamically adjusts based on client divergence
- **Hybrid Client Selection**: Balances high and low divergence clients for stability
- **Comprehensive Metrics**: Tracks R² score, MSE loss, training loss, model divergence, and effective μ

## Output Files

After running simulations, the following files are generated:

- `comparison_results.png`: Comprehensive 6-panel comparison plot
- `r2_comparison.png`: R² score progression comparison
- `mse_comparison.png`: MSE loss progression comparison

## CI/CD Pipeline

This project includes a comprehensive CI/CD pipeline using GitHub Actions:

- **Automated Testing**: Runs on every push and pull request
- **Code Quality Checks**: Linting with flake8, formatting checks with Black and isort
- **Simulation Validation**: Quick and full simulation tests
- **Multi-Python Support**: Tests on Python 3.10 and 3.11

See [.github/workflows/README.md](../.github/workflows/README.md) for detailed CI/CD documentation.

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

## For More Details

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for detailed documentation on:
- Client selection strategies
- FedProx algorithm implementation
- Adaptive μ computation
- Data flow and architecture
