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

## Artifacts

The `full-simulation` job uploads the following artifacts:
- Generated PNG plots (comparison_results.png, r2_comparison.png, mse_comparison.png)
- Any CSV output files

Artifacts are retained for 7 days.
