#!/bin/bash
# compile_cython.sh
#
# 🔥 一鍵編譯 Cython 模組
#
# 使用方法:
#   chmod +x compile_cython.sh
#   ./compile_cython.sh
#
# 或指定環境:
#   conda activate cuda && ./compile_cython.sh

set -e  # Exit on error

echo "🔥 Compiling Cython module..."
echo ""

cd "$(dirname "$0")"

# Detect Python
if [ ! -z "$CONDA_DEFAULT_ENV" ] && [ "$CONDA_DEFAULT_ENV" != "base" ]; then
    # Already in a conda environment (not base)
    PYTHON_CMD="python"
    echo "📦 Using active conda environment: $CONDA_DEFAULT_ENV"
elif [ -f ~/miniforge3/envs/cuda/bin/python ]; then
    # Try to use cuda environment directly
    PYTHON_CMD=~/miniforge3/envs/cuda/bin/python
    echo "📦 Using: ~/miniforge3/envs/cuda/bin/python"
    echo "   (Tip: You can also run: conda activate cuda && ./compile_cython.sh)"
else
    # Fall back to system python
    PYTHON_CMD="python"
    echo "📦 Using system python"
    echo "   ⚠️  Warning: Make sure Cython is installed!"
fi
echo ""

# Clean previous builds
echo "1️⃣  Cleaning old builds..."
rm -rf build/
rm -f electrode_charges_cython*.so electrode_charges_cython*.c

# Compile
echo ""
echo "2️⃣  Compiling with Cython..."
$PYTHON_CMD setup_cython.py build_ext --inplace

# Check result
echo ""
if ls electrode_charges_cython*.so 1> /dev/null 2>&1; then
    echo "✅ Compilation successful!"
    echo ""
    ls -lh electrode_charges_cython*.so
    echo ""
    echo "🚀 You can now use MM_classes_CYTHON.py"
else
    echo "❌ Compilation failed!"
    echo "   No .so file found"
    exit 1
fi
