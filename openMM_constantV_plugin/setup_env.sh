#!/bin/bash
# ConstantV Plugin 環境設置腳本
# 使用方法: source setup_env.sh

# 設置 OPENMM_PLUGIN_DIR 指向正確的插件目錄
export OPENMM_PLUGIN_DIR=$HOME/miniforge3/envs/cuda/lib/plugins

echo "✅ ConstantV Plugin 環境已設置"
echo "   OPENMM_PLUGIN_DIR = $OPENMM_PLUGIN_DIR"
echo ""
echo "現在可以使用 ConstantV Plugin 了!"
echo ""
echo "測試插件:"
echo "  python test_plugin_simple.py"
