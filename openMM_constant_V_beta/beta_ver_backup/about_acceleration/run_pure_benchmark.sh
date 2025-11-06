#!/bin/bash
# 純算法性能測試 - 快速運行腳本

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Poisson Solver 純算法性能測試                             ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "測試三個版本的 Poisson 核心算法:"
echo "  1. Original  - 原始 Python loop"
echo "  2. Optimized - NumPy vectorization"
echo "  3. Cython    - C-compiled"
echo ""
echo "開始測試..."
echo ""

python benchmark_poisson_minimal.py -n 2000 --warmup 200

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  測試完成！                                                ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "查看詳細結果:"
echo "  cat PURE_ALGORITHM_BENCHMARK_RESULTS.md"
echo "  cat FINAL_BENCHMARK_SUMMARY.txt"
