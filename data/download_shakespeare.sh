#!/bin/bash
# Download Shakespeare dataset and prepare data directory
# Part of Tiny Transformer Course - Module 09 Capstone

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "=========================================="
echo "Tiny Transformer - Data Preparation"
echo "=========================================="
echo ""

# Data directory
DATA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DATA_DIR"

echo -e "${BLUE}Downloading datasets...${NC}"
echo ""

# 1. Tiny Shakespeare
SHAKESPEARE_URL="https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
SHAKESPEARE_FILE="tiny_shakespeare.txt"

if [ -f "$SHAKESPEARE_FILE" ]; then
    echo "✓ $SHAKESPEARE_FILE already exists"
else
    echo "Downloading Tiny Shakespeare dataset..."
    curl -L -o "$SHAKESPEARE_FILE" "$SHAKESPEARE_URL"

    # Verify download
    if [ -f "$SHAKESPEARE_FILE" ] && [ -s "$SHAKESPEARE_FILE" ]; then
        echo -e "${GREEN}✓ Downloaded $SHAKESPEARE_FILE${NC}"
    else
        echo "Error: Failed to download Shakespeare dataset"
        exit 1
    fi
fi

# 2. Simple sequences (for testing)
SIMPLE_FILE="simple_sequences.txt"

if [ ! -f "$SIMPLE_FILE" ]; then
    echo "Creating simple test sequences..."
    cat > "$SIMPLE_FILE" << 'EOF'
abcdefghijklmnopqrstuvwxyz
ABCDEFGHIJKLMNOPQRSTUVWXYZ
0123456789
Hello world!
The quick brown fox jumps over the lazy dog.
To be or not to be, that is the question.
EOF
    echo -e "${GREEN}✓ Created $SIMPLE_FILE${NC}"
else
    echo "✓ $SIMPLE_FILE already exists"
fi

# Display statistics
echo ""
echo "=========================================="
echo "Dataset Statistics"
echo "=========================================="
echo ""

if [ -f "$SHAKESPEARE_FILE" ]; then
    chars=$(wc -c < "$SHAKESPEARE_FILE" | tr -d ' ')
    lines=$(wc -l < "$SHAKESPEARE_FILE" | tr -d ' ')
    vocab=$(cat "$SHAKESPEARE_FILE" | grep -o . | sort -u | wc -l | tr -d ' ')

    echo "Tiny Shakespeare:"
    echo "  File: $SHAKESPEARE_FILE"
    echo "  Size: $chars characters"
    echo "  Lines: $lines"
    echo "  Unique characters: ~$vocab"
    echo ""
fi

if [ -f "$SIMPLE_FILE" ]; then
    chars=$(wc -c < "$SIMPLE_FILE" | tr -d ' ')

    echo "Simple Sequences:"
    echo "  File: $SIMPLE_FILE"
    echo "  Size: $chars characters"
    echo ""
fi

echo "=========================================="
echo -e "${GREEN}✓ Data preparation complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Train a model: python3 tools/train.py --config configs/shakespeare.yaml"
echo "  2. Or use specialized training: python3 tools/shakespeare_train.py"
echo ""
