# CUDA API 修正摘要

## 日期
2024年 (最終修正完成)

## 修正的兩個關鍵 API 錯誤

### 1. Stream 句柄獲取錯誤

**問題描述:**
`cu.getStream()` 傳回的是一個 OpenMM `CudaStream` 物件,但 CUDA kernel 啟動 `<<<...>>>` 語法需要的是一個 `cudaStream_t` C 語言句柄。

**錯誤代碼:**
```cpp
// ❌ 錯誤: getCudaStream() 方法不存在
calculateEfKernel<<<numBlocks_N, blockSize, 0, cu.getCudaStream()>>>(...)

// ❌ 錯誤: getStream() 返回 CudaStream 物件,不能直接傳給 kernel
calculateEfKernel<<<numBlocks_N, blockSize, 0, cu.getStream()>>>(...)
```

**正確代碼:**
```cpp
// ✅ 正確: 呼叫兩次 getStream() - 第一次獲取 CudaStream 物件,第二次從物件獲取 cudaStream_t 句柄
calculateEfKernel<<<numBlocks_N, blockSize, 0, cu.getStream().getStream()>>>(...)
scatterWriteChargesKernel<<<numBlocks_N, blockSize, 0, cu.getStream().getStream()>>>(...)
```

**修正位置:**
- `CudaConstantVKernels.cu` line 180: `calculateEfKernel` 啟動
- `CudaConstantVKernels.cu` line 247: `scatterWriteChargesKernel` 啟動

**技術原理:**
```cpp
// OpenMM CudaContext API 結構:
CudaContext::getStream()           // 返回 CudaStream& (C++ 物件)
CudaStream::getStream()            // 返回 cudaStream_t (CUDA runtime 句柄)

// 因此需要:
cudaStream_t handle = cu.getStream().getStream();
kernel<<<grid, block, 0, handle>>>(...)
```

---

### 2. 電容矩陣取得函數簽名錯誤

**問題描述:**
`getInverseCapacitanceMatrix()` 函數被定義為傳回一個 `vector<double>`,而不是填充一個傳入的 vector 參數 (pass-by-reference)。

**錯誤代碼:**
```cpp
// ❌ 錯誤: 試圖使用 pass-by-reference 風格,但 API 實際上是 return-by-value
vector<double> invCapMatrix(N*N);
force.getInverseCapacitanceMatrix(invCapMatrix);  // 編譯錯誤: 沒有此簽名的函數
```

**正確代碼:**
```cpp
// ✅ 正確: 使用 return-by-value 風格
vector<double> invCapMatrix = force.getInverseCapacitanceMatrix();
if (invCapMatrix.size() != (size_t)(N*N)) {
    throw OpenMMException("CudaCalcConstantVKernel::initialize: C_inv matrix size mismatch.");
}
```

**修正位置:**
- `CudaConstantVKernels.cu` line 143: `initialize()` 函數中

**API 定義:**
```cpp
// openmmapi/include/openmm/ConstantVForce.h
class ConstantVForce {
public:
    // 返回值類型是 vector<double>, 不是 void
    std::vector<double> getInverseCapacitanceMatrix() const;
};
```

---

## 修正歷史

1. **初始錯誤 (6 個編譯錯誤):**
   - 使用不存在的 `cu.getCudaStream()` 方法
   - 電容矩陣取得函數簽名錯誤
   - 其他指標類型轉換問題

2. **第一輪修正:**
   - 修正了大部分指標類型轉換
   - 但 stream 句柄獲取仍然錯誤

3. **最終修正 (本次):**
   - ✅ 修正 stream 句柄獲取: `cu.getStream().getStream()`
   - ✅ 修正電容矩陣取得: 改用 return-by-value 風格
   - ✅ 所有 API 調用現在符合 OpenMM 8.x 規範

---

## 編譯狀態

**當前狀態:** CUDA 代碼已修正完成,等待 CUDA toolkit 環境進行編譯測試

**已驗證平台:**
- ✅ Reference Platform: 編譯成功,運行正常
- ⏳ CUDA Platform: 代碼已修正,等待編譯環境

**下一步:**
1. 安裝 CUDA toolkit (如果需要 GPU 加速)
2. 重新編譯 CUDA Platform
3. 運行性能測試,比較 CPU vs GPU 性能

---

## 參考資源

### OpenMM CUDA API 文件
- `CudaContext::getStream()`: 返回 `CudaStream&`
- `CudaStream::getStream()`: 返回 `cudaStream_t` (CUDA runtime 句柄)
- `CudaContext::getPosq()`: 返回 `CudaArray&` (電荷陣列)
- `CudaContext::invalidateMolecules()`: 通知參數已修改

### 相關文件
- [OpenMM CUDA Platform 開發指南](http://docs.openmm.org/latest/developerguide/)
- [CUDA Runtime API 文件](https://docs.nvidia.com/cuda/cuda-runtime-api/)
- `ConstantVForce.h`: 電容恆壓力 API 定義

---

## 技術要點總結

1. **雙層 getStream() 調用:** OpenMM 使用 RAII 包裝,需要兩次調用才能獲取底層 CUDA 句柄
2. **Return-by-value:** 現代 C++ 優化後,返回 vector 的性能通常與 pass-by-reference 相當
3. **零傳輸優化:** 直接在 GPU 上修改 NonbondedForce 的電荷陣列,避免 CPU-GPU 傳輸
4. **參數同步:** 使用 `invalidateMolecules()` 確保 OpenMM 知道參數已在 GPU 上修改

---

**修正完成者:** AI Assistant (GitHub Copilot)  
**用戶指導:** Andy  
**最終檢查:** 2024年
