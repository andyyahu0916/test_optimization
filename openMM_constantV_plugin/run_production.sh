#!/bin/bash
#
# Production FV-MD Runner
#
# Usage:
#   ./run_production.sh [--precompute-cinv] [--config config.ini]
#

set -e  # Exit on error

# Default config
CONFIG="config_refactored.ini"
CINV_FILE="C_inv_matrix.npy"
PRECOMPUTE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --precompute-cinv)
            PRECOMPUTE=true
            shift
            ;;
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--precompute-cinv] [--config config.ini]"
            exit 1
            ;;
    esac
done

# Setup environment
export LD_LIBRARY_PATH="$(pwd)/ConstantVPlugin/build:$LD_LIBRARY_PATH"
export PYTHONPATH="$(pwd)/ConstantVPlugin/build/python/build/lib.linux-x86_64-cpython-313:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

echo "======================================================================"
echo "Production FV-MD with ConstantVPlugin"
echo "======================================================================"
echo "Config: $CONFIG"
echo "C_inv file: $CINV_FILE"
echo ""

# Check if config exists
if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

# Pre-compute C_inv if requested or if file doesn't exist
if [ "$PRECOMPUTE" = true ] || [ ! -f "$CINV_FILE" ]; then
    echo "[Step 1/2] Pre-computing C_inv matrix..."
    echo "(This may take 5-10 minutes for large systems)"
    echo ""

    python precompute_cinv.py -c "$CONFIG" -o "$CINV_FILE"

    if [ $? -ne 0 ]; then
        echo "Error: C_inv computation failed"
        exit 1
    fi

    echo ""
    echo "✓ C_inv matrix computed and saved to $CINV_FILE"
    echo ""
else
    echo "[Step 1/2] Using existing C_inv: $CINV_FILE"
    echo ""
fi

# Run simulation
echo "[Step 2/2] Running FV-MD simulation..."
echo ""

python run_fv_md_production.py -c "$CONFIG" --load-cinv "$CINV_FILE"

if [ $? -ne 0 ]; then
    echo "Error: Simulation failed"
    exit 1
fi

echo ""
echo "======================================================================"
echo "✓ Production run complete!"
echo "======================================================================"
echo ""
echo "Check output directory for results (see config file for path)"
