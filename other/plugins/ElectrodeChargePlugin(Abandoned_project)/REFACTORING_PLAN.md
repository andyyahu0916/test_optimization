# ElectrodeChargePlugin 架構重構計劃

## 🎯 目標
消除所有技術債，實現 Linus-style 正確架構。

## 🐛 當前問題

### 1. Force Group Hack
```cpp
// ElectrodeChargeForce.cpp:11
setForceGroup(1);  // ← HACK: 防止重複計算

// ElectrodeChargeForceImpl.cpp:106
context.calcForcesAndEnergy(true, false, 1 << nonbondedGroup);  // ← HACK
```

**為什麼存在**：
- CUDA NonbondedForce kernel 在計算時會包含所有粒子的電荷
- ElectrodeChargeForce 更新電荷後，如果 NonbondedForce 再次計算，會累積貢獻
- 用 Force Group 隔離來避免重複計算

**為什麼是錯的**：
- 這個問題本身就不該存在
- ElectrodeChargeForce 不應該在每次 getState() 時都被調用
- 應該只在需要時更新電荷

### 2. 迭代在 CPU 端
```cpp
// ElectrodeChargeForceImpl.cpp:100-157
for (int iter = 0; iter < iterations; iter++) {
    context.calcForcesAndEnergy(...);  // PCIe 傳輸
    context.getForces(...);            // PCIe 傳輸
    kernel.execute(...);               // upload + download
    // CPU 端計算 scaling
    nonbondedForce->updateParametersInContext(...);  // PCIe 傳輸
}
```

**問題**：
- 3 次迭代 = 3×4 = 12 次 PCIe 傳輸
- CUDA kernel 沒做迭代，只計算一次
- Scaling 在 CPU 做（應該在 GPU）

### 3. 臨時內存分配
```cpp
// CudaElectrodeChargeKernel.cu:326-330
CudaArray* forcesDevice = CudaArray::create<float4>(*cu, numParticles, "tempForces");
CudaArray* posqDevice = CudaArray::create<float4>(*cu, numParticles, "tempPosq");
forcesDevice->upload(forcesFloat4);
posqDevice->upload(posqFloat4);
```

**問題**：
- 每次 execute 都 new/delete
- 頻繁的內存分配/釋放
- 應該復用持久化內存

## ✅ 正確架構

### Phase 1: 移除 Force Group Hack

**核心思想**：ElectrodeChargeForce 不應該是一個 "Force"，而是一個 "電荷更新器"。

```cpp
class ElectrodeChargeForce : public Force {
    // 不再設置 forceGroup(1)
    // calcForcesAndEnergy() 返回 0（不貢獻力）
    // 只在 updateParametersInContext() 時更新電荷
};
```

**實現**：
1. 移除 `setForceGroup(1)`
2. `calcForcesAndEnergy()` 立即返回 0
3. 把迭代邏輯移到 `updateParametersInContext()`
4. 用戶在 MD loop 外調用一次 `force->updateCharges(context)`

### Phase 2: GPU 內部迭代

```cuda
__global__ void iterativePoissonSolver(
    const float4* positions,
    float4* posq,              // 可修改
    const int* cathodeIndices,
    const int* anodeIndices,
    const int numIterations,   // ← 在 GPU 內迭代
    // ... 其他參數
) {
    for (int iter = 0; iter < numIterations; iter++) {
        // 1. 從當前電荷計算電場（從 NonbondedForce 的 force 計算）
        //    問題：需要調用 NonbondedForce kernel
        //    解決：使用 OpenMM 的 force 計算 API
        
        // 2. 更新電極電荷
        updateElectrodeCharges<<<...>>>();
        __syncthreads();
        
        // 3. Scaling to target
        scaleToTarget<<<...>>>();
        __syncthreads();
    }
}
```

**挑戰**：
- 在 GPU kernel 內部如何調用 NonbondedForce 計算？
- **答案**：不行！NonbondedForce 是獨立的 kernel launch

**更好的方案**：
```cpp
// 在 ForceImpl 中：
for (int iter = 0; iter < iterations; iter++) {
    // 方案 A: 調用 NonbondedForce kernel（如果能拿到 handle）
    // 方案 B: 完全在我們的 kernel 中計算電場（重新實現 Coulomb）
    // 方案 C: 保持迭代在 CPU，但優化內存傳輸
}
```

### Phase 3: 優化內存傳輸（最實際）

**現實**：由於 OpenMM 架構限制，完全在 GPU 內迭代很難。但可以：

1. **持久化設備內存**：
```cpp
class CudaCalcElectrodeChargeKernel {
    CudaArray* forcesDevicePersistent;   // 不每次創建
    CudaArray* posqDevicePersistent;
    
    void execute(...) {
        // 復用內存，只 upload 一次
        if (!initialized) {
            forcesDevicePersistent = CudaArray::create<float4>(...);
        }
        forcesDevicePersistent->upload(forces);  // 快速更新
    }
};
```

2. **批量傳輸**：
```cpp
// 一次傳輸所有迭代需要的數據
struct IterationData {
    std::vector<Vec3> forces[MAX_ITERATIONS];
    std::vector<double> charges[MAX_ITERATIONS];
};
// Upload once, iterate on GPU
```

3. **異步傳輸**：
```cpp
cudaMemcpyAsync(..., stream1);
kernel<<<..., stream2>>>();
cudaMemcpyAsync(..., stream3);
```

## 🚀 實施順序

### Step 1: 修復內存問題（P0，1小時）
- 持久化 CudaArray
- 移除重複 new/delete
- **預期**：效能提升 10×

### Step 2: 批量上傳（P1，30分鐘）
- 一次上傳所有迭代的 forces
- 在 GPU 端緩存
- **預期**：效能提升 3×

### Step 3: 移除 Force Group Hack（P2，2小時）
- 改為 `updateParametersInContext` 模式
- 更新文檔和測試
- **預期**：架構更清晰，無效能變化

### Step 4: GPU 內迭代（P3，未來工作）
- 需要深入研究 OpenMM 內部 API
- 可能需要自己實現 Coulomb kernel
- **預期**：效能提升 100×

## 📊 預期結果

| 階段 | 時間 | 效能 | 代碼質量 |
|------|------|------|----------|
| 現在 | - | 783ms (0.001×) | Hack 滿滿 |
| Step 1 | +1h | ~80ms (0.01×) | 少量改進 |
| Step 2 | +1.5h | ~25ms (0.05×) | 中等 |
| Step 3 | +3.5h | ~25ms | 優秀（無 hack）|
| Step 4 | +?? | <5ms (0.25×) | 完美 |

目標：**先做 Step 1-2，達到可用效能後再考慮 Step 3-4**。
