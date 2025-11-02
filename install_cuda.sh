#!/bin/bash

# 先檢查當前環境是否為 base，如果是則終止（怕忘記切環境）
if [ "$CONDA_DEFAULT_ENV" = "base" ]; then
    echo "錯誤：請先切換到非 base 環境（如 conda activate openmm-cuda-dev），否則可能污染 base 環境。"
    exit 1
fi

# 先移除先前安裝的 CUDA 相關包（避免衝突導致灌失敗）
mamba remove -y cuda cuda-toolkit cuda-nvcc cuda-nsight cuda-cupti cuda-sanitizer-api cuda-memcheck cuda-nvml-dev cuda-cudart-dev cuda-samples cuda-documentation || true
# 注意：|| true 是為了如果沒有包可移除，也繼續執行

# 1. 完整 CUDA 開發工具鏈
conda install -c nvidia -c conda-forge \
    cuda-toolkit \
    cuda-nvcc \
    cuda-nsight \           # Nsight Compute（必備）
    cuda-cupti \            # 性能採集
    cuda-sanitizer-api \    # compute-sanitizer（記憶體除錯）
    cuda-memcheck \         # 舊版 memcheck
    cuda-nvml-dev \         # 監控 GPU 狀態
    cuda-cudart-dev \       # 頭文件
    cuda-samples \          # 範例（參考 kernel 寫法）
    cuda-documentation \    # 離線 API 文件
    -y

# 2. 環境變數（永久寫入環境）
conda env config vars set \
    PATH="$CONDA_PREFIX/bin:$PATH" \
    LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" \
    CUDA_HOME="$CONDA_PREFIX" \
    OPENMM_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc"

# 重啟環境變數（或重新 activate 環境）
conda deactivate
conda activate $CONDA_DEFAULT_ENV

echo "OpenMM CUDA 插件開發環境已就緒！"
echo "CUDA_HOME = $CONDA_PREFIX"
nvcc --version
compute-sanitizer --version
python -c "import openmm; print('OpenMM CUDA:', openmm.Platform.getPlatformByName('CUDA'))"