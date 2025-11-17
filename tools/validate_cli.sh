#!/bin/bash
# CLI Tools Validation Script
# Run this after installing requirements: pip install -r requirements.txt

set -e  # Exit on error

echo "=========================================="
echo "Tiny Transformer CLI Tools Validation"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Test counter
TESTS_PASSED=0
TESTS_FAILED=0

# Helper function to run test
run_test() {
    local test_name=$1
    local command=$2

    echo "Testing: $test_name"
    if eval "$command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}: $test_name"
        ((TESTS_PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}: $test_name"
        ((TESTS_FAILED++))
    fi
    echo ""
}

echo "1. Checking Python imports..."
echo "-------------------------------------------"

# Test train.py imports
run_test "train.py imports" "python3 -c 'import sys; sys.path.insert(0, \".\"); from tools import train'"

# Test generate.py imports
run_test "generate.py imports" "python3 -c 'import sys; sys.path.insert(0, \".\"); from tools import generate'"

# Test interactive.py imports
run_test "interactive.py imports" "python3 -c 'import sys; sys.path.insert(0, \".\"); from tools import interactive'"

echo ""
echo "2. Checking CLI help messages..."
echo "-------------------------------------------"

# Test help messages
run_test "train.py --help" "python3 tools/train.py --help"
run_test "generate.py --help" "python3 tools/generate.py --help"
run_test "interactive.py --help" "python3 tools/interactive.py --help"

echo ""
echo "3. Checking configuration files..."
echo "-------------------------------------------"

# Test config file validity
run_test "base.yaml syntax" "python3 -c 'import yaml; yaml.safe_load(open(\"configs/base.yaml\"))'"
run_test "tiny.yaml syntax" "python3 -c 'import yaml; yaml.safe_load(open(\"configs/tiny.yaml\"))'"
run_test "shakespeare.yaml syntax" "python3 -c 'import yaml; yaml.safe_load(open(\"configs/shakespeare.yaml\"))'"

echo ""
echo "=========================================="
echo "Validation Summary"
echo "=========================================="
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All CLI tools validated successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "  1. Download sample data: bash data/download_shakespeare.sh"
    echo "  2. Train a small model: python3 tools/train.py --config configs/tiny.yaml --data-train data/tiny_shakespeare.txt --max-steps 100"
    echo "  3. Generate text: python3 tools/generate.py --checkpoint checkpoints/checkpoint_100.pt --prompt 'ROMEO:' --max-tokens 50"
    echo "  4. Try interactive mode: python3 tools/interactive.py --checkpoint checkpoints/best.pt"
    exit 0
else
    echo -e "${RED}✗ Some tests failed. Please check the errors above.${NC}"
    exit 1
fi
