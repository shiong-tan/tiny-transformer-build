#!/bin/bash
# Automated Review Helper Script
# Assists with systematic repository review

set -e

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Review tracking
REVIEW_DIR="review_logs"
REVIEW_DATE=$(date +%Y%m%d)
REVIEW_LOG="$REVIEW_DIR/review_${REVIEW_DATE}.md"

# Create review directory
mkdir -p "$REVIEW_DIR"

echo -e "${BLUE}=========================================="
echo "Tiny Transformer - Review Helper"
echo -e "==========================================${NC}"
echo ""

# Function to log finding
log_finding() {
    local severity=$1
    local module=$2
    local description=$3

    echo "- [$severity] Module $module: $description" >> "$REVIEW_LOG"
}

# Function to run module review
review_module() {
    local module_num=$1
    local module_name=$2

    echo -e "${BLUE}Reviewing Module $module_num: $module_name${NC}"

    # Create module section in log
    echo "" >> "$REVIEW_LOG"
    echo "## Module $module_num: $module_name" >> "$REVIEW_LOG"
    echo "" >> "$REVIEW_LOG"
    echo "### Files Reviewed" >> "$REVIEW_LOG"

    # Track status
    local has_errors=0

    # Find module directory
    module_dir="docs/modules/${module_num}_${module_name}"

    if [ ! -d "$module_dir" ]; then
        echo -e "${RED}✗ Module directory not found${NC}"
        log_finding "CRITICAL" "$module_num" "Module directory missing"
        return 1
    fi

    # Check for required files
    echo "  Checking documentation..."

    if [ -f "$module_dir/README.md" ]; then
        echo -e "    ${GREEN}✓${NC} README.md"
        echo "- [x] README.md" >> "$REVIEW_LOG"
    else
        echo -e "    ${RED}✗${NC} README.md missing"
        echo "- [ ] README.md - MISSING" >> "$REVIEW_LOG"
        log_finding "HIGH" "$module_num" "README.md missing"
        has_errors=1
    fi

    if [ -f "$module_dir/theory.md" ] || [ -f "$module_dir/walkthrough.md" ]; then
        echo -e "    ${GREEN}✓${NC} Theory documentation"
        echo "- [x] Theory/Walkthrough" >> "$REVIEW_LOG"
    else
        echo -e "    ${YELLOW}⚠${NC} Theory documentation not found"
        echo "- [ ] Theory/Walkthrough - Not found" >> "$REVIEW_LOG"
    fi

    # Check for notebook
    if [ -f "$module_dir/notebook.ipynb" ]; then
        echo -e "    ${GREEN}✓${NC} Notebook present"
        echo "- [x] notebook.ipynb" >> "$REVIEW_LOG"

        # Try to validate notebook
        echo "  Validating notebook..."
        if command -v jupyter &> /dev/null; then
            if jupyter nbconvert --to notebook --execute --stdout "$module_dir/notebook.ipynb" &> /dev/null; then
                echo -e "    ${GREEN}✓${NC} Notebook executes successfully"
            else
                echo -e "    ${RED}✗${NC} Notebook execution failed"
                log_finding "HIGH" "$module_num" "Notebook execution failed"
                has_errors=1
            fi
        else
            echo -e "    ${YELLOW}⚠${NC} Jupyter not installed, skipping notebook execution"
        fi
    fi

    echo ""
    echo "### Issues Found" >> "$REVIEW_LOG"

    if [ $has_errors -eq 0 ]; then
        echo -e "${GREEN}✓ Module $module_num passed basic checks${NC}"
        echo "- No critical issues found" >> "$REVIEW_LOG"
    else
        echo -e "${RED}✗ Module $module_num has issues (see log)${NC}"
    fi

    echo ""
}

# Function to test imports
test_imports() {
    echo -e "${BLUE}Testing Python imports...${NC}"

    python3 << 'EOF'
import sys
sys.path.insert(0, '.')

errors = []

try:
    from tiny_transformer.attention import ScaledDotProductAttention, MultiHeadAttention
    print("✓ tiny_transformer.attention")
except ImportError as e:
    errors.append(f"✗ tiny_transformer.attention: {e}")

try:
    from tiny_transformer.blocks import TransformerBlock, FeedForward
    print("✓ tiny_transformer.blocks")
except ImportError as e:
    errors.append(f"✗ tiny_transformer.blocks: {e}")

try:
    from tiny_transformer.embeddings import TokenEmbedding, PositionalEncoding
    print("✓ tiny_transformer.embeddings")
except ImportError as e:
    errors.append(f"✗ tiny_transformer.embeddings: {e}")

try:
    from tiny_transformer.model import TinyTransformerLM
    print("✓ tiny_transformer.model")
except ImportError as e:
    errors.append(f"✗ tiny_transformer.model: {e}")

try:
    from tiny_transformer.training import Trainer, CharTokenizer, TextDataset
    print("✓ tiny_transformer.training")
except ImportError as e:
    errors.append(f"✗ tiny_transformer.training: {e}")

try:
    from tiny_transformer.sampling import TextGenerator, GeneratorConfig
    print("✓ tiny_transformer.sampling")
except ImportError as e:
    errors.append(f"✗ tiny_transformer.sampling: {e}")

try:
    from tiny_transformer.utils import TrainingLogger, CheckpointManager, ExperimentTracker
    print("✓ tiny_transformer.utils")
except ImportError as e:
    errors.append(f"✗ tiny_transformer.utils: {e}")

if errors:
    print("\nImport Errors:")
    for err in errors:
        print(err)
    sys.exit(1)
else:
    print("\n✓ All imports successful")
EOF

    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ All Python imports working${NC}"
        echo "- [x] All Python imports working" >> "$REVIEW_LOG"
    else
        echo -e "${RED}✗ Import errors found${NC}"
        log_finding "CRITICAL" "Core" "Python import errors"
        return 1
    fi

    echo ""
}

# Function to run tests
run_tests() {
    echo -e "${BLUE}Running test suite...${NC}"

    if ! command -v pytest &> /dev/null; then
        echo -e "${YELLOW}⚠ pytest not installed, skipping tests${NC}"
        return
    fi

    echo "Running pytest..."

    if pytest tests/ -v --tb=short 2>&1 | tee "$REVIEW_DIR/pytest_output_${REVIEW_DATE}.txt"; then
        echo -e "${GREEN}✓ All tests passed${NC}"
        echo "- [x] All tests passed" >> "$REVIEW_LOG"
    else
        echo -e "${RED}✗ Some tests failed${NC}"
        log_finding "CRITICAL" "Tests" "Test failures detected"
        echo "See $REVIEW_DIR/pytest_output_${REVIEW_DATE}.txt for details"
    fi

    echo ""
}

# Function to check config files
check_configs() {
    echo -e "${BLUE}Validating configuration files...${NC}"

    for config in configs/*.yaml; do
        echo "  Checking $(basename $config)..."

        if python3 -c "import yaml; yaml.safe_load(open('$config'))" 2>/dev/null; then
            echo -e "    ${GREEN}✓${NC} Valid YAML"
        else
            echo -e "    ${RED}✗${NC} Invalid YAML"
            log_finding "HIGH" "Config" "Invalid YAML: $(basename $config)"
        fi
    done

    echo ""
}

# Function to generate review summary
generate_summary() {
    echo -e "${BLUE}Generating review summary...${NC}"

    cat > "$REVIEW_LOG.summary" << EOF
# Review Summary - $REVIEW_DATE

## Quick Stats
- Review Date: $(date)
- Commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")
- Branch: $(git branch --show-current 2>/dev/null || echo "N/A")

## Status
EOF

    # Count findings by severity
    critical=$(grep -c "^\- \[CRITICAL\]" "$REVIEW_LOG" 2>/dev/null || echo "0")
    high=$(grep -c "^\- \[HIGH\]" "$REVIEW_LOG" 2>/dev/null || echo "0")
    medium=$(grep -c "^\- \[MEDIUM\]" "$REVIEW_LOG" 2>/dev/null || echo "0")
    low=$(grep -c "^\- \[LOW\]" "$REVIEW_LOG" 2>/dev/null || echo "0")

    cat >> "$REVIEW_LOG.summary" << EOF
- Critical Issues: $critical
- High Priority: $high
- Medium Priority: $medium
- Low Priority: $low

## Recommendation
EOF

    if [ $critical -eq 0 ] && [ $high -eq 0 ]; then
        echo "✅ READY FOR PRODUCTION" >> "$REVIEW_LOG.summary"
        echo -e "${GREEN}✅ Repository is ready for production${NC}"
    elif [ $critical -eq 0 ]; then
        echo "⚠️ READY WITH MINOR FIXES" >> "$REVIEW_LOG.summary"
        echo -e "${YELLOW}⚠️ Repository is ready with minor fixes${NC}"
    else
        echo "❌ REQUIRES CRITICAL FIXES" >> "$REVIEW_LOG.summary"
        echo -e "${RED}❌ Repository requires critical fixes${NC}"
    fi

    cat >> "$REVIEW_LOG.summary" << EOF

## Full Report
See: $REVIEW_LOG

## Next Steps
1. Review detailed findings in $REVIEW_LOG
2. Address critical and high priority issues
3. Re-run validation after fixes
EOF

    echo ""
    echo -e "${BLUE}Review complete!${NC}"
    echo "Summary: $REVIEW_LOG.summary"
    echo "Details: $REVIEW_LOG"
    echo ""
}

# Main review process
main() {
    # Initialize review log
    cat > "$REVIEW_LOG" << EOF
# Repository Review - $REVIEW_DATE

## Overview
- Date: $(date)
- Reviewer: Automated Review Script
- Commit: $(git rev-parse HEAD 2>/dev/null || echo "N/A")

---

## Review Checklist
EOF

    echo ""
    echo "Starting automated review..."
    echo ""

    # Test imports first
    test_imports

    # Run tests
    run_tests

    # Check configs
    check_configs

    # Review each module
    echo -e "${BLUE}Reviewing modules...${NC}"
    echo ""

    # Note: Adjust module names based on actual directory structure
    review_module "00" "setup"
    review_module "01" "attention"
    review_module "02" "multi_head"
    review_module "03" "transformer_block"
    review_module "04" "embeddings"
    review_module "05" "full_model"
    review_module "06" "training"
    review_module "07" "sampling"
    review_module "08" "engineering"
    review_module "09" "capstone"

    # Generate summary
    generate_summary
}

# Run main
main
