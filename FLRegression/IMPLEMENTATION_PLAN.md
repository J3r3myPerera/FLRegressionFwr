# FedProx with Smart Client Selection - Implementation Plan

## Overview

This project implements a **Federated Learning** system using **FedProx** algorithm with **intelligent client selection** based on divergence metrics. The implementation extends the Flower framework with custom strategies for handling non-IID data distributions common in real-world federated settings.

---

## Architecture Summary

```
┌─────────────────────────────────────────────────────────────────┐
│                         SERVER (ServerApp)                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │                    SmartFedProx Strategy                     ││
│  │  ┌──────────────────┐  ┌────────────────┐  ┌──────────────┐ ││
│  │  │ Client Selection │  │ Model Aggreg.  │  │ Client Stats │ ││
│  │  │ • Random         │  │ • FedAvg base  │  │ • Divergence │ ││
│  │  │ • Diversity      │  │ • Proximal μ   │  │ • Loss hist. │ ││
│  │  │ • Hybrid         │  │                │  │ • Particip.  │ ││
│  │  └──────────────────┘  └────────────────┘  └──────────────┘ ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ Global Model + Config
                              ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                    CLIENT NODES (ClientApp)                  │
    │  ┌─────────────┐  ┌─────────────┐       ┌─────────────┐     │
    │  │  Client 1   │  │  Client 2   │  ...  │  Client N   │     │
    │  │ • Partition │  │ • Partition │       │ • Partition │     │
    │  │ • Train     │  │ • Train     │       │ • Train     │     │
    │  │ • Adaptive μ│  │ • Adaptive μ│       │ • Adaptive μ│     │
    │  └─────────────┘  └─────────────┘       └─────────────┘     │
    └─────────────────────────────────────────────────────────────┘
```

---

## 1. Client Selection Strategies

### 1.1 Selection Strategy Types

The system supports three client selection strategies defined in `SelectionStrategy`:

| Strategy | Description | Best For |
|----------|-------------|----------|
| **RANDOM** | Uniform random sampling | Baseline, stable convergence |
| **DIVERSITY** | Prioritizes high-divergence clients | Non-IID data, faster adaptation |
| **HYBRID** | Balanced mix of high/low divergence | Best of both worlds |

### 1.2 Implementation Details

#### Location: [pytorchexample/strategy.py](pytorchexample/strategy.py)

#### `SmartFedProx` Class

```python
class SmartFedProx(FedProx):
    """FedProx strategy with intelligent client selection."""
```

**Key Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `selection_strategy` | `"random"` | Which selection algorithm to use |
| `selection_temperature` | `1.0` | Softmax temperature for probabilistic selection |
| `hybrid_high_ratio` | `0.5` | Ratio of high-divergence clients in hybrid mode |
| `cold_start_rounds` | `2` | Random selection rounds before using divergence data |
| `exploration_rate` | `0.1` | Probability of random selection for exploration |

### 1.3 Selection Algorithm Flow

```
┌─────────────────────────────────────┐
│         configure_train()           │
│   Called at start of each round     │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│    Cold Start? (round ≤ 2)          │
│    YES → Random Selection           │
│    NO  → Continue                   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│    Exploration? (10% chance)        │
│    YES → Random Selection           │
│    NO  → Continue                   │
└────────────────┬────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────┐
│    Enough History?                  │
│    NO  → Mix historical + random    │
│    YES → Apply Strategy             │
└────────────────┬────────────────────┘
                 │
        ┌────────┼────────┐
        ▼        ▼        ▼
   ┌────────┐ ┌────────┐ ┌────────┐
   │DIVERSITY│ │ HYBRID │ │ RANDOM │
   │High div │ │50/50   │ │Uniform │
   │priority │ │mix     │ │sample  │
   └────────┘ └────────┘ └────────┘
```

### 1.4 Diversity Selection (`_select_diversity`)

Prioritizes clients with **highest divergence** from the global model:

1. Sort clients by `latest_divergence` (descending)
2. Apply **softmax sampling** with temperature scaling
3. Higher divergence = higher selection probability

**Use Case:** When data is highly non-IID and you want the model to quickly learn from diverse local distributions.

### 1.5 Hybrid Selection (`_select_hybrid`)

Balances convergence stability with diversity:

1. Split clients into **high-divergence** and **low-divergence** groups (median split)
2. Select `hybrid_high_ratio` × k clients from high-divergence group
3. Select remaining from low-divergence group
4. Fill any gaps with random sampling

**Use Case:** Production systems where you need both stability and adaptation.

### 1.6 Softmax Sampling (`_softmax_sample`)

Probabilistic selection based on position scores:

```python
# Temperature controls randomness:
# - Low (<1):  More deterministic (closer to top-k)
# - High (>1): More uniform (closer to random)
scores = [n - i for i in range(n)]  # Position-based
exp_scores = [exp((s - max_score) / temperature) for s in scores]
probs = [s / total for s in exp_scores]
```

---

## 2. FedProx Algorithm Implementation

### 2.1 Core Concept

FedProx adds a **proximal term** to the local objective function:

$$
\min_w F_k(w) + \frac{\mu}{2} \|w - w^t\|^2
$$

Where:
- $F_k(w)$ = Local loss function
- $w^t$ = Global model weights (from server)
- $\mu$ = Proximal coefficient (regularization strength)

### 2.2 Training with Proximal Term

#### Location: [pytorchexample/task.py](pytorchexample/task.py) - `train()` function

```python
# Store global model before training
global_params = [p.clone().detach() for p in net.parameters()]

# In training loop:
loss = criterion(net(images), labels)

# Add proximal term: (μ/2) * ||w - w^t||²
if effective_mu > 0.0:
    proximal_term = 0.0
    for local_param, global_param in zip(net.parameters(), global_params):
        proximal_term += ((local_param - global_param) ** 2).sum()
    loss += (effective_mu / 2) * proximal_term
```

### 2.3 Divergence Computation

Measures how much local model has drifted from global:

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

#### Location: [pytorchexample/task.py](pytorchexample/task.py) - `compute_adaptive_mu()`

```python
# Factor 1: Divergence ratio
divergence_ratio = current_divergence / (historical_divergence + ε)

# Factor 2: Epoch scaling
epoch_factor = 1.0 + 0.1 * (local_epochs - 1)

# Combine
adaptive_mu = base_mu * divergence_ratio * epoch_factor

# Clamp to [mu_min, mu_max]
return max(mu_min, min(mu_max, adaptive_mu))
```

### 3.3 Historical Divergence Tracking

Exponential Moving Average (EMA) maintains client history:

```python
# In client_app.py
alpha = 0.3  # EMA smoothing factor
new_historical = alpha * current_divergence + (1 - alpha) * old_divergence
context.state["adaptive_state"] = ConfigRecord({"historical_divergence": new_historical})
```

---

## 4. Client History Tracking

### 4.1 `ClientHistory` Class

#### Location: [pytorchexample/strategy.py](pytorchexample/strategy.py)

Tracks per-client metrics across rounds:

```python
class ClientHistory:
    divergence: list[float]      # Per-round divergence values
    train_loss: list[float]      # Per-round training losses
    effective_mu: list[float]    # Per-round μ values used
    num_examples: int            # Client's dataset size
    participation_count: int     # Times selected for training
```

### 4.2 Metrics Collected

| Metric | Source | Purpose |
|--------|--------|---------|
| `divergence` | `task.train()` | Client selection decisions |
| `train_loss` | `task.train()` | Convergence monitoring |
| `effective_mu` | `task.train()` | Adaptive μ tuning |
| `num-examples` | DataLoader | Weighted aggregation |

### 4.3 Aggregation Hook

```python
def aggregate_train(self, server_round, replies):
    # Update history from each reply
    for reply in replies:
        node_id = reply.metadata.src_node_id
        metrics = reply.content.get("metrics", {})
        
        history = self.client_history[node_id]
        history.divergence.append(metrics["divergence"])
        history.train_loss.append(metrics["train_loss"])
        # ... etc
        
    # Call parent FedProx aggregation
    return super().aggregate_train(server_round, replies)
```

---

## 5. Configuration Reference

### 5.1 pyproject.toml Settings

```toml
[tool.flwr.app.config]
# Training settings
num-server-rounds = 3
fraction-fit = 0.5          # 50% of clients per round
fraction-evaluate = 0.5
local-epochs = 1
learning-rate = 0.1
batch-size = 32

# FedProx core
proximal-mu = 0.1           # Base μ (0 = FedAvg)

# Adaptive μ
adaptive-mu-enabled = true
mu-min = 0.001
mu-max = 1.0

# Smart client selection
selection-strategy = "hybrid"    # random | diversity | hybrid
selection-temperature = 1.0
hybrid-high-ratio = 0.5
cold-start-rounds = 2
exploration-rate = 0.1
```

### 5.2 GPU Configuration

```toml
[tool.flwr.federations.local-simulation-gpu]
options.num-supernodes = 10
options.backend.client-resources.num-cpus = 2
options.backend.client-resources.num-gpus = 1.0  # Full GPU per client
```

---

## 6. Data Flow Diagram

```
Round N Start
     │
     ▼
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: SmartFedProx.configure_train()                          │
│   1. Get available nodes from Grid                              │
│   2. Apply selection strategy → Select K clients                │
│   3. Create Messages with:                                      │
│      • Global model arrays                                      │
│      • Config (lr, proximal_mu, adaptive settings)              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ CLIENTS: train() in client_app.py                               │
│   1. Load global model weights                                  │
│   2. Load local data partition                                  │
│   3. Compute adaptive μ (if enabled)                            │
│   4. Train with FedProx proximal term                           │
│   5. Compute post-training divergence                           │
│   6. Return: updated weights + metrics                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: SmartFedProx.aggregate_train()                          │
│   1. Extract metrics from replies                               │
│   2. Update ClientHistory for each client                       │
│   3. Aggregate model weights (FedAvg)                           │
│   4. Return aggregated model + metrics                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ SERVER: Evaluation Phase                                         │
│   1. Select subset for evaluation                               │
│   2. Run global_evaluate() on test set                          │
│   3. Log accuracy & loss                                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                         Round N+1
```

---

## 7. Model Architecture

### CNN for CIFAR-10

#### Location: [pytorchexample/task.py](pytorchexample/task.py) - `Net` class

```
Input: 3×32×32 (RGB CIFAR-10 image)
     │
     ▼
Conv2d(3→6, 5×5) + ReLU + MaxPool(2×2)  →  6×14×14
     │
     ▼
Conv2d(6→16, 5×5) + ReLU + MaxPool(2×2) →  16×5×5
     │
     ▼
Flatten → 400
     │
     ▼
Linear(400→120) + ReLU
     │
     ▼
Linear(120→84) + ReLU
     │
     ▼
Linear(84→10) → 10 class logits
```

---

## 8. Running the System

### Basic Run (CPU)
```bash
flwr run .
```

### GPU Run
```bash
flwr run . local-simulation-gpu
```

### Custom Configuration
```bash
flwr run . --run-config "selection-strategy=diversity proximal-mu=0.5 num-server-rounds=10"
```

---

## 9. Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| **Cold start with random** | Need baseline metrics before divergence-based selection |
| **10% exploration rate** | Prevents strategy from getting stuck in local optima |
| **Softmax sampling** | Probabilistic selection maintains diversity |
| **EMA for historical divergence** | Smooths noise, adapts to changing client behavior |
| **Hybrid as default** | Best balance of stability and adaptation |
| **Proximal term in loss** | Prevents catastrophic client drift in non-IID settings |

---

## 10. Future Improvements

1. **Gradient-based selection**: Use gradient similarity instead of weight divergence
2. **Fairness constraints**: Ensure all clients participate over time
3. **Dynamic temperature**: Anneal softmax temperature over rounds
4. **Client clustering**: Group similar clients for more efficient selection
5. **Bandwidth-aware selection**: Factor in communication costs
6. **Asynchronous training**: Handle stragglers without blocking rounds

---

## 11. File Summary

| File | Purpose |
|------|---------|
| [strategy.py](pytorchexample/strategy.py) | SmartFedProx with client selection logic |
| [server_app.py](pytorchexample/server_app.py) | Server entry point, config loading |
| [client_app.py](pytorchexample/client_app.py) | Client training/evaluation logic |
| [task.py](pytorchexample/task.py) | Model, data loading, train/test functions |
| [pyproject.toml](pyproject.toml) | Dependencies and configuration |
