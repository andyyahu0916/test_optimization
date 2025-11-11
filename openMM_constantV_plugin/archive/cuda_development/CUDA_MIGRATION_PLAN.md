# 🚀 CUDA 移植計畫 - 零傳輸極致性能

**目標**: 將 Reference 平台的完美物理實現移植到 CUDA
**要求**: 保持 ab initio 物理正確性 + CPU-GPU 零傳輸

---

## 🔍 現狀分析

### 現有 CUDA 代碼問題

**文件**: `CudaConstantVKernels.cu` (8373 bytes)

**致命問題**:
```cpp
// Line 28-29: 警告標記
[!!--- 警告：此核心物理上是錯誤的，參見 SOP 階段二 ---!!]
```

**錯誤的物理模型**:
```cpp
// 使用電容矩陣方法（物理錯誤）
q_e = C_inv * (V - E_f)
```

**缺失的關鍵邏輯**:
- ❌ SCF 迭代循環
- ❌ Green's Reciprocity Theorem
- ❌ Maxwell 邊界條件 (σ/(2ε₀) = V/L + E_ext)
- ❌ 電荷歸一化 (scaleChargesAnalytic)
- ❌ 初始電荷計算 (initialize_Charge)

---

## ✅ Reference 實現（物理正確，已驗證）

**文件**: `ReferenceConstantVKernels.cpp`

**驗證結果**:
- ✅ Green's Reciprocity: 誤差 < 1.5e-14
- ✅ 電荷守恆: Q_total = 0.000000e
- ✅ Maxwell 邊界條件正確
- ✅ 所有物理常數正確

**核心算法**:
```cpp
// SCF 迭代
for (int iter = 0; iter < nIterations; iter++) {
    // 1. 獲取力和位置
    State state = context.getState(Forces | Positions);

    // 2. 更新 Cathode 電荷 (Maxwell 邊界)
    for (cathode atoms) {
        Ez_external = force_z / q_old;  // E = F/q
        q_new = 2/(4π) × area × (V/Lgap + Ez) × conversion;
    }

    // 3. 更新 Anode 電荷
    for (anode atoms) {
        Ez_external = force_z / q_old;
        q_new = -2/(4π) × area × (V/Lgap + Ez) × conversion;
    }

    // 4. Green's Reciprocity 校正
    Q_analytic_cathode = compute_analytic_charge(positions);
    scale_factor = Q_analytic / Q_numeric;
    q_new *= scale_factor;

    // 5. 更新 Context
    nonbondedForce->updateParametersInContext(context);
}
```

---

## 🎯 CUDA 移植目標

### 零傳輸架構

**核心原則**: **所有數據保持在 GPU**

```
CPU (Host)                          GPU (Device)
─────────────────────────────────────────────────────────

Context::step()
  │
  ├─> Force Kernel ─────────────────> CudaConstantVKernel::execute()
  │                                     │
  │                                     ├─> [GPU] 獲取 forces (已在GPU)
  │                                     ├─> [GPU] 獲取 positions (已在GPU)
  │                                     │
  │                                     ├─> [GPU Kernel] SCF Iteration {
  │                                     │     ├─> computeEzExternal
  │                                     │     ├─> updateCathodeCharges
  │                                     │     ├─> updateAnodeCharges
  │                                     │     ├─> computeAnalyticCharge
  │                                     │     ├─> scaleCharges
  │                                     │     └─> scatter to posq.w
  │                                     │   }
  │                                     │
  │                                     └─> invalidateMolecules()
  │
  └─> NonbondedForce ───────────────────> 使用更新後的 posq.w
                                          (零傳輸！)
```

**關鍵**:
- ✅ forces 已在 GPU (CudaArray)
- ✅ positions 已在 GPU (cu.getPosq())
- ✅ 電荷直接寫入 posq.w (float4 的第4分量)
- ✅ 無需 CPU-GPU 傳輸！

---

## 🔧 CUDA Kernel 設計

### Kernel 1: `computeEzExternalKernel`
**目的**: 計算電極原子的外部電場 `E_z = F_z / q_old`

```cuda
__global__ void computeEzExternalKernel(
    int numElectrodes,
    const int* electrodeIndices,      // [N] 電極原子索引
    const float4* forces,              // [NumParticles] 力 (已在GPU)
    const float4* posq,                // [NumParticles] 位置+電荷 (已在GPU)
    double* Ez_external,               // [N] 輸出
    const double SMALL_THRESHOLD
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElectrodes) return;

    int atomIdx = electrodeIndices[i];
    double q_old = (double)posq[atomIdx].w;
    double F_z = (double)forces[atomIdx].z;

    // 0.9×threshold 保護（與 Reference 一致）
    if (fabs(q_old) > (0.9 * SMALL_THRESHOLD)) {
        Ez_external[i] = F_z / q_old;
    } else {
        Ez_external[i] = 0.0;
    }
}
```

---

### Kernel 2: `updateElectrodeChargesKernel`
**目的**: 根據 Maxwell 邊界條件更新電極電荷

```cuda
__global__ void updateElectrodeChargesKernel(
    int numElectrodes,
    const int* electrodeIndices,      // [N]
    const double* areaPerAtom,        // [N] 每個原子的面積
    const double* Ez_external,        // [N] 外部電場
    float4* posq,                     // [NumParticles] 位置+電荷
    double voltage,
    double Lgap,
    double sign,                      // +1 (cathode) or -1 (anode)
    const double CONVERSION_KJMOLNM_AU,
    const double SMALL_THRESHOLD
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElectrodes) return;

    int atomIdx = electrodeIndices[i];
    double area = areaPerAtom[i];
    double Ez = Ez_external[i];

    // Maxwell 邊界條件
    double q_new = sign * 2.0 / (4.0 * M_PI) * area *
                   (voltage / Lgap + Ez) * CONVERSION_KJMOLNM_AU;

    // 電荷閾值保護
    if (fabs(q_new) < SMALL_THRESHOLD) {
        q_new = sign * SMALL_THRESHOLD;
    }

    // 直接寫入 posq.w（零傳輸！）
    posq[atomIdx].w = (float)q_new;
}
```

---

### Kernel 3: `computeAnalyticChargeKernel`
**目的**: 計算解析總電荷 (Green's Reciprocity)

**Part 3a: 幾何貢獻**
```cuda
__global__ void computeGeometricChargeKernel(
    double* Q_analytic,               // [1] 輸出（使用 atomic add）
    double voltage,
    double Lgap,
    double Lcell,
    double totalArea,
    double sign,
    const double CONVERSION_KJMOLNM_AU
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double Q_geo = sign / (4.0 * M_PI) * totalArea *
                       (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOLNM_AU;
        atomicAdd(Q_analytic, Q_geo);
    }
}
```

**Part 3b: 鏡像電荷貢獻（並行 reduction）**
```cuda
__global__ void computeImageChargeKernel(
    int numElectrolytes,
    const int* electrolyteIndices,    // [M]
    const float4* posq,                // [NumParticles]
    double* Q_analytic,                // [1] 輸出（atomic add）
    double z_opposite,
    double Lcell
) {
    // Parallel reduction to sum image charges
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    if (i < numElectrolytes) {
        int atomIdx = electrolyteIndices[i];
        double q_i = (double)posq[atomIdx].w;
        double z_atom = (double)posq[atomIdx].z;
        double z_distance = fabs(z_atom - z_opposite);

        // 鏡像電荷貢獻
        local_sum = (z_distance / Lcell) * (-q_i);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block leader adds to global
    if (tid == 0) {
        atomicAdd(Q_analytic, sdata[0]);
    }
}
```

---

### Kernel 4: `scaleChargesKernel`
**目的**: 歸一化電荷 (Green's Reciprocity 校正)

```cuda
__global__ void scaleChargesKernel(
    int numElectrodes,
    const int* electrodeIndices,      // [N]
    float4* posq,                     // [NumParticles]
    double scale_factor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElectrodes) return;

    int atomIdx = electrodeIndices[i];
    double q_old = (double)posq[atomIdx].w;
    double q_new = q_old * scale_factor;
    posq[atomIdx].w = (float)q_new;
}
```

---

### Kernel 5: `sumElectrodeChargesKernel`
**目的**: 計算電極總電荷 (用於 scale_factor 計算)

```cuda
__global__ void sumElectrodeChargesKernel(
    int numElectrodes,
    const int* electrodeIndices,      // [N]
    const float4* posq,                // [NumParticles]
    double* Q_numeric                  // [1] 輸出
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    if (i < numElectrodes) {
        int atomIdx = electrodeIndices[i];
        local_sum = (double)posq[atomIdx].w;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    // Reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(Q_numeric, sdata[0]);
    }
}
```

---

## 📊 完整執行流程

### Host 端 (C++)
```cpp
double CudaCalcConstantVKernel::execute(
    ContextImpl& context,
    bool includeForces,
    bool includeEnergy
) {
    // 獲取 GPU 資源（零傳輸！）
    CudaArray& posq = cu.getPosq();      // 位置+電荷 (已在GPU)
    CudaArray& forces = cu.getForce();   // 力 (已在GPU)

    // SCF 迭代循環
    for (int iter = 0; iter < numSCFIterations; iter++) {

        // === Step 1: 計算外部電場 ===
        computeEzExternalKernel<<<...>>>(
            numCathodes, d_cathodeIndices, forces, posq, d_Ez_cathode, ...
        );
        computeEzExternalKernel<<<...>>>(
            numAnodes, d_anodeIndices, forces, posq, d_Ez_anode, ...
        );

        // === Step 2: 更新電極電荷 (Maxwell 邊界) ===
        updateElectrodeChargesKernel<<<...>>>(
            numCathodes, d_cathodeIndices, d_cathodeAreas, d_Ez_cathode,
            posq, voltage, Lgap, +1.0, ...  // sign = +1
        );
        updateElectrodeChargesKernel<<<...>>>(
            numAnodes, d_anodeIndices, d_anodeAreas, d_Ez_anode,
            posq, voltage, Lgap, -1.0, ...  // sign = -1
        );

        // === Step 3: Green's Reciprocity 校正 ===

        // 3a. 清零 analytic/numeric buffers
        cudaMemset(d_Q_analytic_cathode, 0, sizeof(double));
        cudaMemset(d_Q_analytic_anode, 0, sizeof(double));
        cudaMemset(d_Q_numeric_cathode, 0, sizeof(double));
        cudaMemset(d_Q_numeric_anode, 0, sizeof(double));

        // 3b. 計算解析電荷 (幾何 + 鏡像)
        computeGeometricChargeKernel<<<...>>>(
            d_Q_analytic_cathode, voltage, Lgap, Lcell, totalArea, +1.0, ...
        );
        computeImageChargeKernel<<<...>>>(
            numElectrolytes, d_electrolyteIndices, posq,
            d_Q_analytic_cathode, z_anode, Lcell
        );

        // 同樣處理 Anode...

        // 3c. 計算數值總電荷
        sumElectrodeChargesKernel<<<...>>>(
            numCathodes, d_cathodeIndices, posq, d_Q_numeric_cathode
        );
        sumElectrodeChargesKernel<<<...>>>(
            numAnodes, d_anodeIndices, posq, d_Q_numeric_anode
        );

        // 3d. 計算 scale_factor (需要 D2H transfer，但只是兩個 double)
        double Q_analytic_c, Q_numeric_c, Q_analytic_a, Q_numeric_a;
        cudaMemcpy(&Q_analytic_c, d_Q_analytic_cathode, sizeof(double), D2H);
        cudaMemcpy(&Q_numeric_c, d_Q_numeric_cathode, sizeof(double), D2H);
        cudaMemcpy(&Q_analytic_a, d_Q_analytic_anode, sizeof(double), D2H);
        cudaMemcpy(&Q_numeric_a, d_Q_numeric_anode, sizeof(double), D2H);

        double scale_cathode = Q_analytic_c / Q_numeric_c;
        double scale_anode = Q_analytic_a / Q_numeric_a;

        // 3e. 歸一化電荷
        scaleChargesKernel<<<...>>>(
            numCathodes, d_cathodeIndices, posq, scale_cathode
        );
        scaleChargesKernel<<<...>>>(
            numAnodes, d_anodeIndices, posq, scale_anode
        );

    } // End SCF iteration

    // 通知 OpenMM 電荷已更新
    cu.invalidateMolecules();

    return 0.0;
}
```

---

## 🎯 零傳輸架構總結

### CPU-GPU 數據傳輸分析

| 數據 | 位置 | 傳輸 |
|------|------|------|
| **positions** | GPU (cu.getPosq()) | ✅ 零傳輸 |
| **forces** | GPU (cu.getForce()) | ✅ 零傳輸 |
| **電極電荷** | GPU (posq.w) | ✅ 零傳輸（直接寫入） |
| **電解質電荷** | GPU (posq.w) | ✅ 零傳輸（只讀） |
| **Q_analytic** | GPU → CPU | ⚠️ 4 doubles/iteration (32 bytes) |
| **Q_numeric** | GPU → CPU | ⚠️ 4 doubles/iteration (32 bytes) |

**總傳輸量**（每次 SCF 迭代）:
```
32 bytes × 4 iterations = 128 bytes
```

**相比傳統方法** (假設 800 電極原子):
```
傳統:
  - Forces: 800 × 12 bytes = 9.6 KB
  - Positions: 800 × 12 bytes = 9.6 KB
  - Charges: 800 × 8 bytes = 6.4 KB
  Total: ~26 KB/iteration × 4 = 104 KB

我們的零傳輸:
  - 只傳 Q values: 128 bytes

減少: 104,000 / 128 ≈ 812x ！
```

---

## 📋 實現步驟

### Phase 1: 基礎架構 ✅
- [x] 分析現有 CUDA 代碼
- [x] 確認物理錯誤
- [x] 設計零傳輸架構

### Phase 2: Kernel 實現 (進行中)
- [ ] 創建 CudaConstantVKernels.h (新版)
- [ ] 實現 computeEzExternalKernel
- [ ] 實現 updateElectrodeChargesKernel
- [ ] 實現 computeAnalyticChargeKernel (geometric)
- [ ] 實現 computeImageChargeKernel (reduction)
- [ ] 實現 sumElectrodeChargesKernel (reduction)
- [ ] 實現 scaleChargesKernel
- [ ] 實現 initialize_Charge kernel

### Phase 3: Host 端整合
- [ ] 重寫 CudaCalcConstantVKernel::execute()
- [ ] 實現 SCF 迭代循環
- [ ] 整合所有 kernels
- [ ] 錯誤處理

### Phase 4: 測試驗證
- [ ] 單元測試（每個 kernel）
- [ ] 與 Reference 對比
- [ ] Green's Reciprocity 驗證
- [ ] 電荷守恆驗證
- [ ] 性能測試

### Phase 5: 優化
- [ ] Shared memory 優化
- [ ] Warp shuffle reduction
- [ ] Stream 並行（cathode/anode）
- [ ] Kernel fusion

---

## 🎓 物理正確性保證

### 必須保持的第一性原則

1. **Maxwell 邊界條件** ✅
   ```cuda
   q = sign × 2/(4π) × area × (V/Lgap + E_ext) × conversion
   ```

2. **Green's Reciprocity** ✅
   ```cuda
   Q_analytic = Q_geometric + Q_image
   scale = Q_analytic / Q_numeric
   ```

3. **電場定義 E=F/q** ✅
   ```cuda
   E_z = F_z / q_old  (with 0.9×threshold protection)
   ```

4. **原子單位轉換** ✅
   ```cuda
   CONVERSION_KJMOLNM_AU = 18.8973 / 2625.5
   ```

5. **數值穩定性** ✅
   ```cuda
   if (|q| < threshold) q = sign × threshold
   if (|q_old| > 0.9×threshold) E = F/q
   ```

---

## 🚀 預期性能

### 性能估算

**Python Reference**: 36.4 秒/步 (29427 原子)

**C++ Reference**: 預期 1.8-7.3 秒/步 (5-20x 加速)

**CUDA (零傳輸)**:
- GPU 並行: 10-100x (相比單線程 C++)
- 零傳輸: 額外 10-50% 提升
- **預期總加速**: 50-200x 相比 C++ Reference
- **預期時間**: **0.036-0.36 秒/步**

**對比 Python**: **100-1000x 加速** 🚀

---

## 📝 注意事項

### CUDA 特有問題

1. **Double Precision**
   - 使用 `double` 保持精度
   - 注意 `float4` 轉換

2. **Atomic Operations**
   - `atomicAdd` 在 reduction 中使用
   - 可能的性能瓶頸

3. **Memory Coalescing**
   - 電極原子索引可能不連續
   - 使用 texture memory 優化？

4. **Kernel Launch Overhead**
   - 多個小 kernel vs 單個大 kernel
   - 需要 profiling

---

**編制**: Claude (Anthropic)
**日期**: 2025-11-11
**狀態**: Phase 1 完成，Phase 2 開始
**下一步**: 實現 CUDA kernels
