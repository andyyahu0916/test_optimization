#!/bin/bash
# Run FV-MD plugin test with proper library paths

# Set library paths
export LD_LIBRARY_PATH="$(pwd)/ConstantVPlugin/build:$LD_LIBRARY_PATH"
export PYTHONPATH="$(pwd)/ConstantVPlugin/build/python/build/lib.linux-x86_64-cpython-313:$PYTHONPATH"

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

# Run test
python test_fv_md_with_real_data.py
