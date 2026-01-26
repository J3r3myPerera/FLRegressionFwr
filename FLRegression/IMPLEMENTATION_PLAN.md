# FedProx with Smart Client Selection - Implementation Plan

## Overview

This project implements a **Federated Learning** system using **FedProx** algorithm with **intelligent client selection** based on divergence metrics. The implementation uses standalone Python scripts (no Flower framework) to simulate federated learning with custom strategies for handling non-IID data distributions common in real-world federated settings.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                    SERVER (FederatedSimulator)                   │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    SmartFedProx Strategy                     ││
│  │  ┌──────────────────┐  ┌────────────────┐  ┌──────────────┐ ││
│  │  │ Client Selection │  │ Model Aggreg.  │  │ Client Stats │ ││
│  │  │ • Random         │  │ • FedAvg base  │  │ • Divergence │ ││
│  │  │ • Hybrid         │  │ • Proximal μ   │  │ • Loss hist. │ ││
│  │  └──────────────────┘  └────────────────┘  └──────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Global Model + Config
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    CLIENT NODES (SimulatedClient)            │
    │  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐     │
    │  │  Client 1   │  │  Client 2   │  ...  │  Client N   │     │
    │  │ • Partition │  │ • Partition │       │ • Partition │     │
    │  │ • Train     │  │ • Train     │       │ • Train     │     │
    │  │ • Adaptive μ│  │ • Adaptive μ│       │ • Adaptive μ│     │
    │  └─────────────┘  └─────────────┘       └─────────────┘     │
    └─────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
FLRegression/
├── dataset.py              # Dataset loading, preprocessing, and partitioning
├── module.py               # Model (Net), training/test functions, configuration
├── client.py               # SimulatedClient class for local training
├── server.py               # FederatedSimulator class for orchestration
├── main.py                 # Main entry point for running simulations
└── run_comparison.py       # Comparison script for all three strategies
```

---

## 1. Client Selection Strategies

### 1.1 Selection Strategy Types

The system supports three client selection strategies:

| Strategy | Description | Best For |
|----------|-------------|----------|
| **RANDOM** | Uniform random sampling | Baseline, stable convergence |
| **HYBRID** | Balanced mix of high/middle/low divergence | Best of both worlds (default for SmartFedProx) |

### 1.2 Implementation Details

#### Location: [server.py](server.py) - `FederatedSimulator.select_clients()`

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selection_strategy` | `"random"` | Which selection algorithm to use |
| `cold_start_rounds` | `3` | Random selection rounds before using divergence data |
| `exploration_rate` | `0.15` | Probability of random selection for exploration |

### 1.3 Selection Algorithm Flow

```
┌─────────────────────────────────────┐
│         select_clients()             │
│   Called at start of each round     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│    Cold Start? (round ≤ 3)          │
│    YES → Random Selection           │
│    NO  → Continue                   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│    Exploration? (15% chance)         │
│    YES → Random Selection           │
│    NO  → Continue                   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│    Apply Hybrid Strategy             │
│    • 30% high-divergence clients    │
│    • 50% middle-divergence clients  │
│    • 20% low-divergence clients     │
└─────────────────────────────────────┘
```

### 1.4 Hybrid Selection

Balances convergence stability with diversity:

1. Sort clients by smoothed divergence (weighted average: 50% latest + 30% previous + 20% mean of rest)
2. Select 30% from high-divergence group
3. Select 50% from middle-divergence group
4. Select 20% from low-divergence group
5. Fill any gaps with random sampling

**Use Case:** Production systems where you need both stability and adaptation.

---

## 2. FedProx Algorithm Implementation

### 2.1 Core Concept

FedProx adds a **proximal term** to the local objective function:

$$
\min_w F_k(w) + \frac{\mu}{2} \|w - w^t\|^2
$$

Where:
- $F_k(w)$ = Local loss function (MSE for regression)
- $w^t$ = Global model weights (from server)
- $\mu$ = Proximal coefficient (regularization strength)

### 2.2 Training with Proximal Term

#### Location: [module.py](module.py) - `train()` function

```python
# Store global model before training
global_params = [p.clone().detach().to(device) for p in net.parameters()]

# In training loop:
loss = criterion(predictions, targets)

# Add proximal term: (μ/2) * ||w - w^t||²
if effective_mu > 0.0:
    proximal_term = 0.0
    for local_param, global_param in zip(net.parameters(), global_params):
        proximal_term += ((local_param - global_param) ** 2).sum()
    loss += (effective_mu / 2) * proximal_term
```

### 2.3 Divergence Computation

Measures how much local model has drifted from global:

#### Location: [module.py](module.py) - `compute_model_divergence()`

```python
def compute_model_divergence(local_params, global_params):
    """Compute L2 norm: ||w_local - w_global||"""
    divergence = 0.0
    for local_p, global_p in zip(local_params, global_params):
        divergence += ((local_p - global_p) ** 2).sum().item()
    return divergence ** 0.5
```

---

## 3. Adaptive μ (Proximal Coefficient)

### 3.1 Motivation

Fixed μ is suboptimal because:
- Some clients have more non-IID data (need higher μ)
- Some clients train more epochs (need drift prevention)
- Historical behavior indicates client "personality"

### 3.2 Adaptive μ Formula

#### Location: [module.py](module.py) - `compute_adaptive_mu()`

```python
# Factor 1: Divergence ratio
if global_avg_divergence > 0 and historical_divergence > 0:
    divergence_ratio = historical_divergence / (global_avg_divergence + 1e-8)
    divergence_factor = 1.0 + 0.5 * (divergence_ratio - 1.0)  # Dampened scaling
    divergence_factor = max(0.5, min(2.0, divergence_factor))  # Clamp to [0.5, 2.0]
else:
    divergence_factor = 1.0

# Factor 2: Epoch scaling
epoch_factor = 1.0 + 0.1 * (local_epochs - 1)

# Combine
adaptive_mu = base_mu * divergence_factor * epoch_factor

# Clamp to [mu_min, mu_max]
return max(mu_min, min(mu_max, adaptive_mu))
```

### 3.3 Historical Divergence Tracking

Exponential Moving Average (EMA) maintains client history:

#### Location: [client.py](client.py) - `SimulatedClient.train()`

```python
# Update historical divergence with EMA
alpha = 0.3  # EMA smoothing factor
self.historical_divergence = alpha * result["divergence"] + (1 - alpha) * self.historical_divergence
```

---

## 4. Dataset and Preprocessing

### 4.1 Dataset Location

The dataset is located at:
```
/Users/dinukaperera/FLRegressionFlwr/data/indianPersonalFinanceAndSpendingHabits.csv
```

### 4.2 Data Preprocessing

#### Location: [dataset.py](dataset.py) - `_load_and_preprocess_data()`

1. **Load CSV**: Read Indian Personal Finance dataset
2. **Encode Categorical**: Label encode `Occupation` and `City_Tier`
3. **Scale Features**: StandardScaler for numerical features
4. **Scale Target**: StandardScaler for `Disposable_Income`
5. **Create Partition Keys**: Combine Occupation + City_Tier + Income_Bracket for extreme non-IID

### 4.3 Extreme Non-IID Partitioning

#### Location: [dataset.py](dataset.py) - `load_data()`

**Strategy 1: Multi-dimensional Partitioning**
- Primary split by Occupation + City_Tier + Income_Bracket (36+ unique groups)
- Each client primarily gets data from 1-2 specific demographic groups
- 70% from primary key, 30% from secondary key

**Strategy 2: Income-based Label Skew**
- Odd partition_ids: High income bias (above 25th percentile)
- Even partition_ids: Low income bias (below 75th percentile)

**Strategy 3: Quantity Skew**
- Uneven data distribution: quantity_factor ranges from 0.5 to 1.3
- Some clients get more data, some get less

---

## 5. Model Architecture

### 5.1 MLP for Regression

#### Location: [module.py](module.py) - `Net` class

```
Input: 26 features (after preprocessing)
     │
     ▼
Linear(26→128) + BatchNorm + ReLU + Dropout(0.3)
     │
     ▼
Linear(128→64) + BatchNorm + ReLU + Dropout(0.2)
     │
     ▼
Linear(64→32) + BatchNorm + ReLU
     │
     ▼
Linear(32→1) → Disposable Income (regression output)
```

**Loss Function:** MSE (Mean Squared Error)  
**Optimizer:** Adam with learning rate 0.001  
**Metrics:** R² Score (coefficient of determination), MSE Loss

---

## 6. Configuration Reference

### 6.1 Configuration Constants

#### Location: [module.py](module.py)

```python
# Simulation Configuration
NUM_ROUNDS = 20
NUM_CLIENTS = 10
FRACTION_FIT = 0.5
LOCAL_EPOCHS = 3
LEARNING_RATE = 0.001
BATCH_SIZE = 64
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Strategy configurations
STRATEGIES = {
    "FedAvg": {
        "proximal_mu": 0.0,
        "adaptive_mu_enabled": False,
        "selection_strategy": "random",
        "description": "Baseline FedAvg (μ=0)"
    },
    "FedProx": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": False,
        "selection_strategy": "random",
        "description": "FedProx (μ=0.1, random selection)"
    },
    "SmartFedProx": {
        "proximal_mu": 0.1,
        "adaptive_mu_enabled": True,
        "selection_strategy": "hybrid",
        "description": "SmartFedProx (adaptive μ, hybrid selection)"
    }
}
```

---

## 7. Data Flow Diagram

```
Round N Start
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: FederatedSimulator.run()                                 │
│   1. Select clients (based on strategy)                          │
│   2. Send global model state_dict + config to selected clients   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ CLIENTS: SimulatedClient.train()                                 │
│   1. Load global model weights                                  │
│   2. Load local data partition (extreme non-IID)                │
│   3. Compute adaptive μ (if enabled)                            │
│   4. Train with FedProx proximal term                           │
│   5. Compute post-training divergence                           │
│   6. Update historical divergence (EMA)                         │
│   7. Return: updated weights + metrics                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: FederatedSimulator.aggregate()                           │
│   1. Weighted average aggregation (FedAvg)                      │
│   2. Update global model                                         │
│   3. Track client statistics                                     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: FederatedSimulator.evaluate_global()                    │
│   1. Load centralized test set                                   │
│   2. Evaluate global model                                       │
│   3. Compute R² score and MSE loss                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         Round N+1
```

---

## 8. Running the System

### Basic Run

```bash
cd FLRegression
python main.py
```

### Run Comparison Script

```bash
python run_comparison.py
```

This runs multiple trials for statistical significance and generates comprehensive comparison plots.

### Output Files

- `comparison_results.png`: Comprehensive 6-panel comparison plot
- `r2_comparison.png`: R² score progression comparison
- `mse_comparison.png`: MSE loss progression comparison

---

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Cold start with random** | Need baseline metrics before divergence-based selection |
| **15% exploration rate** | Prevents strategy from getting stuck in local optima |
| **Hybrid selection (30/50/20)** | Best balance of stability and adaptation |
| **EMA for historical divergence** | Smooths noise, adapts to changing client behavior |
| **Proximal term in loss** | Prevents catastrophic client drift in non-IID settings |
| **Extreme non-IID partitioning** | Realistic heterogeneous data distribution |
| **Adaptive μ per client** | Handles varying client behavior and data distributions |

---

## 10. File Summary

| File | Purpose |
|------|---------|
| [dataset.py](dataset.py) | Dataset loading, preprocessing, and extreme non-IID partitioning |
| [module.py](module.py) | Model definition (Net), training/test functions, configuration |
| [client.py](client.py) | SimulatedClient class for local training and evaluation |
| [server.py](server.py) | FederatedSimulator class for orchestration and aggregation |
| [main.py](main.py) | Main entry point for running simulations |
| [run_comparison.py](run_comparison.py) | Comparison script for all three strategies with multiple trials |

---

## 11. Metrics Tracked

| Metric | Description | Location |
|--------|-------------|----------|
| **R² Score** | Coefficient of determination for regression | `module.test()` |
| **MSE Loss** | Mean Squared Error | `module.test()` |
| **Training Loss** | Average training loss per round | `module.train()` |
| **Model Divergence** | L2 norm of parameter differences | `module.compute_model_divergence()` |
| **Effective μ** | Actual proximal coefficient used (may be adaptive) | `module.train()` |
| **Historical Divergence** | EMA-smoothed divergence history per client | `client.SimulatedClient` |

---

## 12. Future Improvements

1. **Gradient-based selection**: Use gradient similarity instead of weight divergence
2. **Fairness constraints**: Ensure all clients participate over time
3. **Dynamic temperature**: Anneal selection randomness over rounds
4. **Client clustering**: Group similar clients for more efficient selection
5. **Bandwidth-aware selection**: Factor in communication costs
6. **Asynchronous training**: Handle stragglers without blocking rounds
7. **Federated evaluation**: Evaluate on client test sets instead of centralized test set
