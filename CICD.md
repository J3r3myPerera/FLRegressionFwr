# CI/CD Pipeline Documentation

This document describes the complete CI/CD pipeline setup for the Federated Learning Regression project.

## Overview

The CI/CD pipeline automatically:
- ✅ Validates code quality and imports
- ✅ Runs unit tests
- ✅ Tests data loading and model creation
- ✅ Runs quick simulations to validate the full pipeline
- ✅ Checks code formatting
- ✅ Generates and stores artifacts

## Pipeline Structure

### Workflows

1. **Main CI Pipeline** (`.github/workflows/ci.yml`)
   - Runs on every push and pull request
   - Multiple jobs for parallel execution
   - Comprehensive validation

2. **Unit Tests** (`.github/workflows/test.yml`)
   - Dedicated test suite with coverage
   - Multiple Python versions
   - Coverage reporting

## Jobs Breakdown

### 1. Lint and Test (`lint-and-test`)

**Purpose**: Basic validation and smoke tests

**Runs on**: Python 3.10, 3.11

**Steps**:
- Install dependencies
- Lint code with flake8
- Validate all imports
- Test data loading
- Test model creation
- Test client initialization

**Duration**: ~2-3 minutes

### 2. Quick Simulation (`quick-simulation`)

**Purpose**: Validate the full federated learning pipeline

**Runs on**: Python 3.10

**Steps**:
- Run a 2-round simulation with 3 clients
- Test FedAvg strategy
- Validate metrics collection

**Duration**: ~3-5 minutes

### 3. Full Simulation (`full-simulation`)

**Purpose**: Complete simulation test (only on main/develop branches)

**Runs on**: Python 3.10

**Steps**:
- Run 5-round simulation (reduced from 20 for CI)
- Test all three strategies
- Generate plots
- Upload artifacts

**Duration**: ~10-15 minutes

**Artifacts**:
- Generated PNG plots
- CSV output files (if any)
- Retained for 7 days

### 4. Code Quality (`code-quality`)

**Purpose**: Formatting and style checks

**Runs on**: Python 3.10

**Steps**:
- Check code formatting with Black
- Check import sorting with isort

**Duration**: ~1 minute

## Environment Setup

### Required Files

1. **`requirements.txt`**: Python dependencies
   ```
   torch>=2.0.0
   pandas>=2.0.0
   scikit-learn>=1.3.0
   numpy>=1.24.0
   matplotlib>=3.7.0
   ```

2. **`.flake8`**: Linting configuration
   - Max line length: 127
   - Max complexity: 10
   - Excludes: `__pycache__`, `.venv`, etc.

3. **`pytest.ini`**: Test configuration
   - Test paths: `tests/`
   - Verbose output
   - Markers for slow/integration tests

### Environment Variables

The pipeline uses:
- `DATA_PATH`: Path to the dataset CSV file
  - Set automatically in CI: `${{ github.workspace }}/data/indianPersonalFinanceAndSpendingHabits.csv`
  - Can be overridden for local testing

## Local Testing

### Prerequisites

```bash
pip install -r requirements.txt
pip install pytest pytest-cov flake8 black isort
```

### Run All Tests

```bash
# Set data path
export DATA_PATH="$(pwd)/data/indianPersonalFinanceAndSpendingHabits.csv"

# Run pytest
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=FLRegression --cov-report=html
```

### Run Quick CI Simulation

```bash
./scripts/run_tests.sh
```

This script mimics the CI environment and runs:
- Linting
- Import checks
- Data loading tests
- Model creation tests
- Full pytest suite

### Manual Pipeline Steps

1. **Linting**:
   ```bash
   flake8 FLRegression/ --count --select=E9,F63,F7,F82 --show-source --statistics
   flake8 FLRegression/ --count --exit-zero --max-complexity=10 --max-line-length=127
   ```

2. **Import Validation**:
   ```bash
   cd FLRegression
   python -c "import dataset; import module; import client; import server"
   ```

3. **Data Loading Test**:
   ```bash
   cd FLRegression
   export DATA_PATH="$(pwd)/../data/indianPersonalFinanceAndSpendingHabits.csv"
   python -c "from dataset import _get_data_path, get_input_dim, reset_data_cache; reset_data_cache(); print(get_input_dim())"
   ```

4. **Quick Simulation**:
   ```bash
   cd FLRegression
   export DATA_PATH="$(pwd)/../data/indianPersonalFinanceAndSpendingHabits.csv"
   python -c "
   from server import FederatedSimulator
   from module import STRATEGIES
   config = STRATEGIES['FedAvg']
   simulator = FederatedSimulator('FedAvg', config)
   metrics = simulator.run(2)  # 2 rounds for quick test
   print(f'R²: {metrics[\"r2_scores\"][-1]:.4f}')
   "
   ```

## Test Coverage

The test suite (`tests/test_basic.py`) covers:

- ✅ Data loading and preprocessing
- ✅ Model creation and forward pass
- ✅ Divergence computation
- ✅ Adaptive μ computation
- ✅ Client initialization and training
- ✅ Client evaluation
- ✅ Server/simulator initialization
- ✅ Client selection
- ✅ Model aggregation

## Continuous Integration Triggers

The pipeline runs automatically on:

1. **Push Events**:
   - Push to `main` branch → All jobs
   - Push to `develop` branch → All jobs
   - Push to other branches → Lint and test only

2. **Pull Request Events**:
   - PR to `main` → All jobs except full simulation
   - PR to `develop` → All jobs except full simulation

3. **Manual Trigger**:
   - Use `workflow_dispatch` to manually trigger any workflow

## Artifacts and Reports

### Generated Artifacts

- **Simulation Results**: PNG plots and CSV files (7-day retention)
- **Coverage Reports**: HTML and XML coverage reports
- **Test Results**: Pytest output and logs

### Accessing Artifacts

1. Go to the GitHub Actions tab
2. Select the workflow run
3. Click on the job
4. Download artifacts from the "Artifacts" section

## Troubleshooting

### Common Issues

1. **Data file not found**:
   - Ensure `data/indianPersonalFinanceAndSpendingHabits.csv` exists
   - Check `DATA_PATH` environment variable

2. **Import errors**:
   - Verify all dependencies are in `requirements.txt`
   - Check Python path includes `FLRegression/`

3. **Test failures**:
   - Run tests locally first: `pytest tests/ -v`
   - Check test logs for specific error messages

4. **Linting failures**:
   - Run `black FLRegression/` to auto-format
   - Run `isort FLRegression/` to sort imports
   - Fix remaining issues manually

## Best Practices

1. **Before Pushing**:
   - Run `./scripts/run_tests.sh` locally
   - Fix any linting issues
   - Ensure all tests pass

2. **Pull Requests**:
   - Keep PRs focused and small
   - Add tests for new features
   - Update documentation if needed

3. **Main Branch**:
   - Only merge after all CI checks pass
   - Review artifacts before merging
   - Tag releases appropriately

## Future Enhancements

Potential improvements:
- [ ] Add performance benchmarks
- [ ] Add regression tests for model accuracy
- [ ] Add security scanning
- [ ] Add dependency vulnerability scanning
- [ ] Add automated documentation generation
- [ ] Add deployment automation (if needed)

## Support

For issues or questions about the CI/CD pipeline:
1. Check the workflow logs in GitHub Actions
2. Review this documentation
3. Run tests locally to reproduce issues
4. Open an issue with detailed error messages
