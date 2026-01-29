# CI/CD Pipeline Documentation

This directory contains GitHub Actions workflows for continuous integration and deployment.

## Workflows

### 1. `ci.yml` - Main CI Pipeline

Runs on:
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop`
- Manual trigger via `workflow_dispatch`

**Jobs:**

1. **lint-and-test** (Python 3.10, 3.11)
   - Code linting with flake8
   - Import validation
   - Data loading tests
   - Model creation tests
   - Client initialization tests

2. **quick-simulation** (Python 3.10)
   - Runs a quick 2-round simulation to validate the full pipeline
   - Tests FedAvg strategy with reduced client count

3. **full-simulation** (Python 3.10, only on push to main/develop)
   - Runs full simulation with 5 rounds (reduced from 20 for CI)
   - Generates comparison plots
   - Uploads artifacts

4. **code-quality** (Python 3.10)
   - Checks code formatting with Black
   - Checks import sorting with isort

### 2. `test.yml` - Unit Tests

Runs comprehensive unit tests with pytest and coverage reporting.

**Features:**
- Tests on Python 3.10 and 3.11
- Code coverage reporting
- Uploads coverage to Codecov (optional)

## Environment Variables

The pipeline uses the `DATA_PATH` environment variable to locate the dataset:
```yaml
DATA_PATH: ${{ github.workspace }}/data/indianPersonalFinanceAndSpendingHabits.csv
```

## Local Testing

To test the CI pipeline locally:

1. Install dependencies:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov flake8 black isort
```

2. Run linting:
```bash
flake8 FLRegression/
```

3. Run tests:
```bash
pytest tests/ -v
```

4. Run quick simulation:
```bash
cd FLRegression
export DATA_PATH=$(pwd)/../data/indianPersonalFinanceAndSpendingHabits.csv
python -c "from dataset import _get_data_path; print(_get_data_path())"
```

## Why CI Can Fail While Unit Tests Pass

The **Unit Tests** workflow (`test.yml`) only runs `pytest tests/`. The **CI/CD Pipeline** (`ci.yml`) runs more:

| CI/CD step            | Unit Tests equivalent | Can fail if … |
|-----------------------|------------------------|----------------|
| Lint (flake8)         | Not run                | Syntax/undefined names (E9, F63, F7, F82) |
| Check imports         | Implied by pytest      | Missing or wrong imports when run from `FLRegression/` |
| Test data loading     | `test_data_path_exists`, `test_get_input_dim`, etc. | `DATA_PATH` not set or file missing in checkout |
| Test model creation   | `test_model_creation`   | Same data path; runs from `working-directory: FLRegression` |
| Test client init      | `test_client_initialization` | Same as above |
| **quick-simulation**  | Not run                | FederatedSimulator run fails (assertion, exception, timeout) |
| **full-simulation**   | Not run                | Only on push to main/develop; `main()` or plotting fails |
| code-quality (Black/isort) | Not run         | Currently `continue-on-error: true` so does not fail the job |

So Unit Tests can pass (pytest from repo root with `DATA_PATH` set) while CI fails in a step that runs from `FLRegression/`, or in **quick-simulation** / **full-simulation**.

**How to find the exact failure:** In GitHub, open the failed run → click the failed job (e.g. `lint-and-test` or `quick-simulation`) → expand the step that shows a red X. The log for that step is the cause (e.g. `Data file not found`, assertion error, or traceback).

## Artifacts

The `full-simulation` job uploads the following artifacts:
- Generated PNG plots (comparison_results.png, r2_comparison.png, mse_comparison.png)
- Any CSV output files

Artifacts are retained for 7 days.
