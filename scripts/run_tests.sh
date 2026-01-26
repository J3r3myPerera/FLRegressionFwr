#!/bin/bash
# Script to run tests locally (mimics CI environment)

set -e

echo "=== Running Local CI Tests ==="

# Set data path
export DATA_PATH="$(pwd)/data/indianPersonalFinanceAndSpendingHabits.csv"

# Check if data file exists
if [ ! -f "$DATA_PATH" ]; then
    echo "ERROR: Data file not found at $DATA_PATH"
    exit 1
fi

echo "Data path: $DATA_PATH"

# Install dependencies
echo "=== Installing dependencies ==="
pip install -r requirements.txt
pip install pytest pytest-cov flake8 black isort

# Run linting
echo "=== Running flake8 ==="
flake8 FLRegression/ --count --select=E9,F63,F7,F82 --show-source --statistics || true
flake8 FLRegression/ --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics || true

# Check imports
echo "=== Checking imports ==="
cd FLRegression
python -c "import dataset; import module; import client; import server; print('All imports successful')"

# Test data loading
echo "=== Testing data loading ==="
python -c "
from dataset import _get_data_path, get_input_dim, reset_data_cache
path = _get_data_path()
print(f'Data path: {path}')
assert path.exists(), f'Data file not found at {path}'
reset_data_cache()
dim = get_input_dim()
print(f'Input dimension: {dim}')
assert dim > 0, 'Input dimension should be positive'
print('Data loading test passed!')
"

# Test model creation
echo "=== Testing model creation ==="
python -c "
from module import Net, get_input_dim
from dataset import reset_data_cache
reset_data_cache()
input_dim = get_input_dim()
model = Net(input_dim=input_dim)
print(f'Model created successfully with input_dim={input_dim}')
print(f'Model parameters: {sum(p.numel() for p in model.parameters())}')
"

# Run pytest
echo "=== Running pytest ==="
cd ..
pytest tests/ -v

echo "=== All tests passed! ==="
