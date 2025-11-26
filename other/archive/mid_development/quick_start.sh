#!/bin/bash
#
# Quick Start Script - FV-MD with ConstantVPlugin
#
# This script helps you get started with the plugin using Reference Platform
#

echo "======================================================================"
echo "FV-MD ConstantVPlugin - Quick Start"
echo "======================================================================"
echo ""

# Check exclusions fix
echo "[1/4] Checking exclusions fix..."
if [ -f "check_exclusions_fix.sh" ]; then
    bash check_exclusions_fix.sh
    if [ $? -ne 0 ]; then
        echo "❌ Exclusions check failed!"
        exit 1
    fi
else
    echo "⚠️  Warning: check_exclusions_fix.sh not found"
fi

echo ""
echo "[2/4] Checking config file..."
if [ ! -f "config_refactored.ini" ]; then
    echo "❌ config_refactored.ini not found!"
    echo "Please create it first."
    exit 1
fi

# Check platform setting
PLATFORM=$(grep "^platform" config_refactored.ini | awk '{print $3}')
echo "Current platform: $PLATFORM"

if [ "$PLATFORM" == "CUDA" ]; then
    echo ""
    echo "⚠️  WARNING: Platform is set to CUDA, but CUDA is not available!"
    echo "Changing to Reference platform..."
    sed -i.bak 's/^platform = CUDA/platform = Reference/' config_refactored.ini
    echo "✓ Changed to Reference platform"
fi

echo ""
echo "[3/4] Checking C_inv matrix..."
if [ ! -f "C_inv.npy" ]; then
    echo "⚠️  C_inv.npy not found. You need to compute it first:"
    echo ""
    echo "python precompute_cinv.py -c config_refactored.ini -o C_inv.npy"
    echo ""
    read -p "Do you want to compute it now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python precompute_cinv.py -c config_refactored.ini -o C_inv.npy
        if [ $? -ne 0 ]; then
            echo "❌ C_inv computation failed!"
            exit 1
        fi
    else
        echo "Please compute C_inv matrix first."
        exit 1
    fi
else
    echo "✓ C_inv.npy found"
fi

echo ""
echo "[4/4] Ready to run simulation!"
echo ""
echo "======================================================================"
echo "To run the simulation:"
echo "======================================================================"
echo ""
echo "python run_fv_md_production.py -c config_refactored.ini --load-cinv C_inv.npy"
echo ""
echo "======================================================================"
echo ""
read -p "Do you want to start the simulation now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo "Starting simulation..."
    echo ""
    python run_fv_md_production.py -c config_refactored.ini --load-cinv C_inv.npy
else
    echo ""
    echo "OK, you can run it manually when ready."
    echo ""
fi

echo ""
echo "======================================================================"
echo "Quick Start Complete!"
echo "======================================================================"
