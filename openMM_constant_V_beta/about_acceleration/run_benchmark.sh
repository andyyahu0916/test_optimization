#!/bin/bash
# Quick test script for Poisson benchmarks

echo "=========================================="
echo "Poisson Solver Benchmark Quick Test"
echo "=========================================="
echo ""

# Check if we're in the right directory
if [ ! -f "benchmark_poisson.py" ]; then
    echo "❌ Error: Please run this script from openMM_constant_V_beta directory"
    exit 1
fi

echo "This script will run two types of benchmarks:"
echo ""
echo "1. Minimal benchmark (fast, ~30 seconds)"
echo "   - Tests core algorithm only"
echo "   - Uses simulated data"
echo ""
echo "2. Full system benchmark (slower, ~2-5 minutes)"
echo "   - Tests with real OpenMM system"
echo "   - More realistic results"
echo ""

read -p "Run minimal benchmark? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "=========================================="
    echo "Running Minimal Benchmark"
    echo "=========================================="
    python benchmark_poisson_minimal.py --cathode 1000 --anode 1000 -n 1000 --warmup 100
fi

echo ""
read -p "Run full system benchmark? [Y/n] " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    echo ""
    echo "=========================================="
    echo "Running Full System Benchmark"
    echo "=========================================="
    python benchmark_poisson.py -n 10 -r 5 --warmup 2
fi

echo ""
echo "=========================================="
echo "Benchmark Complete!"
echo "=========================================="
echo ""
echo "Check the output above for results."
echo "Full benchmark results are saved to benchmark_results_*.txt"
