"""
FastAPI backend for Federated Learning Regression simulation.
Replaces the Streamlit-based app.py.

Run with:
    uvicorn api:app --reload --port 8000
"""

import os
import sys
import time
import asyncio
from pathlib import Path
from typing import Optional
from contextlib import asynccontextmanager

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Data path setup (same logic as the old Streamlit app)
# ---------------------------------------------------------------------------
if not os.getenv("DATA_PATH"):
    app_dir = Path(__file__).parent
    for candidate in [
        app_dir / "data" / "indianPersonalFinanceAndSpendingHabits.csv",
        app_dir.parent / "data" / "indianPersonalFinanceAndSpendingHabits.csv",
    ]:
        if candidate.exists():
            os.environ["DATA_PATH"] = str(candidate)
            break

# Add FLRegression package to sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "FLRegression"))

import module as _module
from module import (
    NUM_ROUNDS, NUM_CLIENTS, FRACTION_FIT, LOCAL_EPOCHS,
    LEARNING_RATE, BATCH_SIZE, DEVICE, STRATEGIES,
    get_input_dim, _load_and_preprocess_data, reset_data_cache,
)
from server import FederatedSimulator

# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------
simulation_store: dict = {}          # latest results keyed by run id
simulation_status: dict = {}         # "idle" | "running" | "done" | "error"
current_run_id: Optional[str] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the dataset on startup."""
    reset_data_cache()
    _load_and_preprocess_data()
    print(f"[startup] Data loaded – input dim = {get_input_dim()}, device = {DEVICE}")
    yield


app = FastAPI(
    title="Federated Learning Regression API",
    description="Compare FedAvg, FedProx and SmartFedProx on the Indian Personal Finance dataset.",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve the static frontend
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class SimulationRequest(BaseModel):
    strategies: list[str] = Field(
        default=["FedAvg", "FedProx", "SmartFedProx"],
        description="Which strategies to run",
    )
    num_rounds: int = Field(default=25, ge=5, le=50)
    num_trials: int = Field(default=1, ge=1, le=5)
    seed: Optional[int] = Field(default=42, description="Random seed (null = time-based)")
    # Server-config overrides (defaults come from module.py)
    num_clients: int = Field(default=NUM_CLIENTS, ge=2, le=50)
    fraction_fit: float = Field(default=FRACTION_FIT, ge=0.1, le=1.0)
    local_epochs: int = Field(default=LOCAL_EPOCHS, ge=1, le=20)
    learning_rate: float = Field(default=LEARNING_RATE, ge=0.00001, le=0.1)
    batch_size: int = Field(default=BATCH_SIZE, ge=8, le=512)


class StrategyInfo(BaseModel):
    name: str
    proximal_mu: float
    adaptive_mu_enabled: bool
    selection_strategy: str
    description: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _run_simulation(req: SimulationRequest) -> dict:
    """Blocking simulation runner (called inside a thread)."""
    # ---- Temporarily patch module-level constants so that server.py /
    #      client.py pick up the user-supplied values. We restore them
    #      afterwards to keep the module in a clean state.
    import server as _server_mod
    _saved = {
        "NUM_CLIENTS": _module.NUM_CLIENTS,
        "FRACTION_FIT": _module.FRACTION_FIT,
        "LOCAL_EPOCHS": _module.LOCAL_EPOCHS,
        "LEARNING_RATE": _module.LEARNING_RATE,
        "BATCH_SIZE": _module.BATCH_SIZE,
    }
    _module.NUM_CLIENTS   = req.num_clients
    _module.FRACTION_FIT  = req.fraction_fit
    _module.LOCAL_EPOCHS  = req.local_epochs
    _module.LEARNING_RATE = req.learning_rate
    _module.BATCH_SIZE    = req.batch_size
    # server.py imports these at module-level; patch its namespace too
    _server_mod.NUM_CLIENTS   = req.num_clients
    _server_mod.FRACTION_FIT  = req.fraction_fit
    _server_mod.LOCAL_EPOCHS  = req.local_epochs
    _server_mod.LEARNING_RATE = req.learning_rate
    _server_mod.BATCH_SIZE    = req.batch_size

    try:
        return _run_simulation_inner(req)
    finally:
        # Restore original constants
        for k, v in _saved.items():
            setattr(_module, k, v)
            setattr(_server_mod, k, v)


def _run_simulation_inner(req: SimulationRequest) -> dict:
    """The actual simulation logic (runs with patched module constants)."""
    reset_data_cache()
    _load_and_preprocess_data()

    base_seed = req.seed if req.seed is not None else int(time.time()) % 10000
    all_trial_results: dict[str, list] = {s: [] for s in req.strategies}

    for trial in range(req.num_trials):
        trial_seed = base_seed + trial * 100
        for strategy_name in req.strategies:
            np.random.seed(trial_seed)
            torch.manual_seed(trial_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(trial_seed)

            config = STRATEGIES[strategy_name]
            simulator = FederatedSimulator(strategy_name, config)
            metrics = simulator.run(req.num_rounds)
            all_trial_results[strategy_name].append(metrics)

    # Aggregate across trials
    aggregated: dict = {}
    for strategy_name in req.strategies:
        trials = all_trial_results[strategy_name]
        if not trials:
            continue

        if req.num_trials > 1:
            aggregated[strategy_name] = {
                "rounds": trials[0]["rounds"],
                "r2_scores": [
                    float(np.mean([t["r2_scores"][i] for t in trials]))
                    for i in range(req.num_rounds)
                ],
                "mse_losses": [
                    float(np.mean([t["mse_losses"][i] for t in trials]))
                    for i in range(req.num_rounds)
                ],
                "avg_train_loss": [
                    float(np.mean([t["avg_train_loss"][i] for t in trials]))
                    for i in range(req.num_rounds)
                ],
                "avg_divergence": [
                    float(np.mean([t["avg_divergence"][i] for t in trials]))
                    for i in range(req.num_rounds)
                ],
                "avg_effective_mu": [
                    float(np.mean([t["avg_effective_mu"][i] for t in trials]))
                    for i in range(req.num_rounds)
                ],
            }
        else:
            # Single trial — just convert numpy types to plain floats
            m = trials[0]
            aggregated[strategy_name] = {
                k: [float(v) for v in vals] if isinstance(vals, list) else vals
                for k, vals in m.items()
            }

    # Build a summary table
    summary = []
    for name, m in aggregated.items():
        summary.append({
            "strategy": name,
            "final_r2": round(m["r2_scores"][-1], 4),
            "best_r2": round(max(m["r2_scores"]), 4),
            "final_mse": round(m["mse_losses"][-1], 4),
            "lowest_mse": round(min(m["mse_losses"]), 4),
            "final_mu": round(m["avg_effective_mu"][-1], 4),
        })

    winner = max(aggregated, key=lambda s: aggregated[s]["r2_scores"][-1])

    return {
        "config": {
            "num_rounds": req.num_rounds,
            "num_clients": req.num_clients,
            "fraction_fit": req.fraction_fit,
            "local_epochs": req.local_epochs,
            "learning_rate": req.learning_rate,
            "batch_size": req.batch_size,
            "num_trials": req.num_trials,
            "seed": base_seed,
            "strategies": req.strategies,
        },
        "metrics": aggregated,
        "summary": summary,
        "winner": winner,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/")
async def root():
    """Serve the frontend."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Federated Learning API is running. See /docs for the interactive API docs."}


@app.get("/api/config")
async def get_config():
    """Return the current simulation hyper-parameters and available strategies."""
    return {
        "num_rounds": NUM_ROUNDS,
        "num_clients": NUM_CLIENTS,
        "fraction_fit": FRACTION_FIT,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "device": str(DEVICE),
        "input_dim": get_input_dim(),
        "strategies": {
            name: StrategyInfo(
                name=name,
                proximal_mu=cfg["proximal_mu"],
                adaptive_mu_enabled=cfg["adaptive_mu_enabled"],
                selection_strategy=cfg["selection_strategy"],
                description=cfg["description"],
            ).model_dump()
            for name, cfg in STRATEGIES.items()
        },
    }


@app.get("/api/status")
async def get_status():
    """Poll the status of the current / last simulation run."""
    return {
        "run_id": current_run_id,
        "status": simulation_status.get(current_run_id, "idle"),
    }


@app.post("/api/simulate")
async def run_simulation(req: SimulationRequest):
    """
    Launch a federated learning simulation.

    This runs **synchronously** in a background thread so the event-loop
    stays responsive. Poll ``GET /api/status`` to watch progress, then
    ``GET /api/results`` when status is ``done``.
    """
    global current_run_id

    # Validate strategy names
    for s in req.strategies:
        if s not in STRATEGIES:
            raise HTTPException(400, f"Unknown strategy '{s}'. Choose from {list(STRATEGIES.keys())}")

    run_id = str(int(time.time()))
    current_run_id = run_id
    simulation_status[run_id] = "running"

    try:
        # Run the heavy computation in a thread to avoid blocking the event loop
        result = await asyncio.to_thread(_run_simulation, req)
        simulation_store[run_id] = result
        simulation_status[run_id] = "done"
        return {"run_id": run_id, "status": "done", "result": result}
    except Exception as exc:
        simulation_status[run_id] = "error"
        raise HTTPException(500, detail=str(exc))


@app.get("/api/results")
async def get_results():
    """Return the most recent simulation results."""
    if not current_run_id or current_run_id not in simulation_store:
        raise HTTPException(404, "No results available yet. Run a simulation first.")
    return simulation_store[current_run_id]


@app.get("/api/results/{run_id}")
async def get_results_by_id(run_id: str):
    """Return results for a specific run."""
    if run_id not in simulation_store:
        raise HTTPException(404, f"No results for run_id '{run_id}'")
    return simulation_store[run_id]
