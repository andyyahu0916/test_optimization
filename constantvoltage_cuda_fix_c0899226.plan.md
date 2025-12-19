---
name: ConstantVoltage CUDA Fix
overview: 修復 C++/CUDA 移植版本中發現的 7 個問題，確保與 Python 原始算法的物理正確性和數學邏輯一致。主要修復集中在 Analytic Charge 計算、Electrode Scale 邏輯和 Contact Normal 計算。
todos:
  - id: fix-1-no-conductor-scale
    content: 添加無 Conductor 時的電極 Scale 邏輯 (P0)
    status: completed
  - id: fix-2-anode-scale
    content: 添加 Anode 獨立 Scale (P0)
    status: completed
  - id: fix-3-contact-normal
    content: 修正 Contact Normal 計算為 (0,0,±1) (P1)
    status: completed
  - id: fix-4-electrolyte-image
    content: 實現 Electrolyte Image Charge 貢獻到 Q_analytic (P1)
    status: completed
  - id: fix-5-recompute-analytic
    content: Conductor 更新後重新計算 Analytic Charge (P2)
    status: completed
  - id: fix-6-conductor-image
    content: 添加 Conductor Atoms 對 Q_analytic 的貢獻 (P2)
    status: completed
  - id: fix-7-comment-typo
    content: 修正 C++ 注釋數值錯誤 (P3)
    status: completed
---

# ConstantVoltage CUDA 移植修復計畫

## 問題優先順序總覽

| 優先級 | 問題 | 檔案 | 複雜度 |
|-------|------|------|--------|
| P0 | 無 Conductor 時缺少 electrode scale | CudaConstantVoltageKernels.cpp | 中 |
| P0 | Anode 電荷未獨立 scale | CudaConstantVoltageKernels.cpp | 中 |
| P1 | Contact normal 計算錯誤 | CudaConstantVoltageKernels.cpp | 低 |
| P1 | Electrolyte image charge 未計入 Q_analytic | CudaConstantVoltageKernels.cpp + constantVoltage.cu | 高 |
| P2 | Conductor 更新後未重新計算 analytic | CudaConstantVoltageKernels.cpp | 中 |
| P2 | Conductor atoms 對 Q_analytic 的貢獻 | CudaConstantVoltageKernels.cpp | 中 |
| P3 | C++ 注釋數值錯誤 | CudaConstantVoltageKernels.cpp | 低 |

---

## 修復 1: 添加無 Conductor 時的電極 Scale (P0)

**問題:** 當系統沒有 Buckyball/Nanotube 時，`updateElectrodeCharges` 不會對電極電荷進行 analytic normalization。

**檔案:** [CudaConstantVoltageKernels.cpp](openmm-8.4.0/plugins/constantvoltage/platforms/cuda/src/CudaConstantVoltageKernels.cpp)

**修復位置:** `updateElectrodeCharges()` 函數末尾 (約 line 529)

**修復方案:**
```cpp
// 在 if (numTotalConductors > 0) { ... } 之後添加 else 分支

} else {
    // No conductors: scale cathode and anode independently
    // Reference: MM_classes.py:547-550
    
    // Compute Q_analytic for cathode
    float qAnalyticCathode = computeAnalyticChargeWithElectrolyte(context, zAnode, true);
    float qNumericCathode = getTotalCathodeCharge(context);
    if (fabsf(qNumericCathode) > smallThreshold) {
        float scaleCathode = qAnalyticCathode / qNumericCathode;
        if (scaleCathode > 0.0f) {
            // Call scaleElectrodeCharges kernel for cathode
        }
    }
    
    // Compute Q_analytic for anode
    float qAnalyticAnode = computeAnalyticChargeWithElectrolyte(context, zCathode, false);
    float qNumericAnode = getTotalAnodeCharge(context);
    if (fabsf(qNumericAnode) > smallThreshold) {
        float scaleAnode = qAnalyticAnode / qNumericAnode;
        if (scaleAnode > 0.0f) {
            // Call scaleElectrodeCharges kernel for anode
        }
    }
}
```

---

## 修復 2: 添加 Anode 獨立 Scale (P0)

**問題:** 有 Conductor 時，Anode 電荷沒有被 scale。

**檔案:** [CudaConstantVoltageKernels.cpp](openmm-8.4.0/plugins/constantvoltage/platforms/cuda/src/CudaConstantVoltageKernels.cpp)

**修復位置:** `updateElectrodeCharges()` 中 conductor 處理區塊 (約 line 505-527)

**修復方案:**
```cpp
if (numTotalConductors > 0) {
    // ... existing conductor image charges and transfer code ...
    
    // Step 2d: Scale anode FIRST (independently)
    // Reference: MM_classes.py:514-515
    float qAnalyticAnode = computeAnalyticChargeWithElectrolyte(context, zCathode, false);
    float qNumericAnode = getTotalAnodeCharge(context);
    if (fabsf(qNumericAnode) > smallThreshold) {
        float scaleAnode = qAnalyticAnode / qNumericAnode;
        if (scaleAnode > 0.0f) {
            // Scale anode charges
            cu.executeKernel(scaleElectrodeChargesKernel, anodeScaleArgs, numAnodes);
        }
    }
    
    // Step 2e: THEN scale cathode + conductors using -qAnalyticAnode
    float qAnalytic = -qAnalyticAnode;
    // ... existing scaleElectrodeChargesWithConductorsKernel call ...
}
```

---

## 修復 3: 修正 Contact Normal 計算 (P1)

**問題:** Contact normal 應為電極表面法向量 (0, 0, ±1)，但實際計算為幾何方向。

**檔案:** [CudaConstantVoltageKernels.cpp](openmm-8.4.0/plugins/constantvoltage/platforms/cuda/src/CudaConstantVoltageKernels.cpp)

**修復位置:** `initializeConductorGeometry()` (約 line 362-372)

**修復方案:**
```cpp
// 修改 contact normal 計算邏輯
if (conductorIsCloseToElectrode[c]) {
    // Contact with flat electrode: use electrode surface normal (0, 0, ±1)
    // Reference: Fixed_Voltage_routines.py:265-266
    // Cathode: normal points +z, Anode: normal points -z
    if (/* conductor attached to cathode */) {
        conductorContactNormals[c] = make_float3(0.0f, 0.0f, 1.0f);
    } else {
        conductorContactNormals[c] = make_float3(0.0f, 0.0f, -1.0f);
    }
} else {
    // Contact with another conductor: use geometric direction
    // ... existing geometric calculation ...
}
```

---

## 修復 4: 實現 Electrolyte Image Charge 貢獻 (P1)

**問題:** Q_analytic 只包含幾何項，缺少 electrolyte 的 image charge 貢獻。

**檔案:** 
- [CudaConstantVoltageKernels.cpp](openmm-8.4.0/plugins/constantvoltage/platforms/cuda/src/CudaConstantVoltageKernels.cpp)
- [constantVoltage.cu](openmm-8.4.0/plugins/constantvoltage/platforms/cuda/src/kernels/constantVoltage.cu)

**修復方案:**

### 4.1 新增 Host 端輔助函數
```cpp
// 添加到 CudaConstantVoltageKernels.cpp
float CudaCalcConstantVoltageForceKernel::computeAnalyticChargeWithElectrolyte(
    ContextImpl& context, float z_opposite, bool isCathode)
{
    float sign = isCathode ? 1.0f : -1.0f;
    
    // Geometric contribution
    // Q = sign / (4π) * area * (V/Lgap + V/Lcell) * K
    float qGeometric = sign * totalArea * (voltage_kjmol / Lgap + voltage_kjmol / Lcell) 
                       * CONVERSION_KJMOL_NM_AU / FOUR_PI;
    
    // Electrolyte image charge contribution
    // Reset accumulator
    float zero = 0.0f;
    electrolyteContribution.upload(&zero);
    
    // Call computeAnalyticCharge kernel
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr electrolyteIdxPtr = electrolyteIndices.getDevicePointer();
    CUdeviceptr contribPtr = electrolyteContribution.getDevicePointer();
    float lcell = (float)Lcell;
    
    void* args[] = { &posqPtr, &electrolyteIdxPtr, &numElectrolytes, 
                     &z_opposite, &lcell, &contribPtr };
    cu.executeKernel(computeAnalyticChargeKernel, args, numElectrolytes);
    
    float imageContrib;
    electrolyteContribution.download(&imageContrib);
    
    return qGeometric + imageContrib;
}
```

### 4.2 添加成員變數
```cpp
// 在 CudaConstantVoltageKernels.h 中添加
CudaArray electrolyteContribution;

// 在 initialize() 中初始化
electrolyteContribution.initialize<float>(cu, 1, "electrolyteContribution");
```

---

## 修復 5: Conductor 更新後重新計算 Analytic Charge (P2)

**問題:** Conductor 電荷更新後，analytic charge 應該重新計算（因為 conductor 原子也算入 image charge）。

**檔案:** [CudaConstantVoltageKernels.cpp](openmm-8.4.0/plugins/constantvoltage/platforms/cuda/src/CudaConstantVoltageKernels.cpp)

**修復位置:** `updateElectrodeCharges()` 中 Step 2c 之後

**修復方案:**
```cpp
// Step 2c: Charge transfer for each conductor
for (int c = 0; c < numTotalConductors; c++) {
    // ... existing charge transfer code ...
}

// Step 2c.5: Recompute analytic charges after conductor update
// Reference: MM_classes.py:358-360
// "because conductors within cell are 'part of electrolyte'..."
float qAnalyticCathode = computeAnalyticChargeWithElectrolytePlusConductors(context, zAnode, true);
float qAnalyticAnode = computeAnalyticChargeWithElectrolytePlusConductors(context, zCathode, false);
```

需要新增函數將 conductor atoms 也加入 image charge 計算。

---

## 修復 6: Conductor Atoms 對 Q_analytic 的貢獻 (P2)

**問題:** Conductor 原子應該像 electrolyte 一樣貢獻 image charge。

**修復方案:** 擴展修復 4 中的函數，添加 conductor atoms 的遍歷。

```cpp
float CudaCalcConstantVoltageForceKernel::computeAnalyticChargeWithElectrolytePlusConductors(
    ContextImpl& context, float z_opposite, bool isCathode)
{
    float result = computeAnalyticChargeWithElectrolyte(context, z_opposite, isCathode);
    
    // Add conductor atoms contribution
    // Reference: Fixed_Voltage_routines.py:336-344
    if (totalConductorAtoms > 0) {
        float zero = 0.0f;
        conductorContribution.upload(&zero);
        
        CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
        CUdeviceptr condIdxPtr = allConductorIndices.getDevicePointer();
        CUdeviceptr contribPtr = conductorContribution.getDevicePointer();
        float lcell = (float)Lcell;
        
        void* args[] = { &posqPtr, &condIdxPtr, &totalConductorAtoms,
                         &z_opposite, &lcell, &contribPtr };
        cu.executeKernel(computeAnalyticChargeKernel, args, totalConductorAtoms);
        
        float condContrib;
        conductorContribution.download(&condContrib);
        result += condContrib;
    }
    
    return result;
}
```

---

## 修復 7: C++ 注釋數值錯誤 (P3)

**問題:** 注釋中的數值 "0.00719475" 應為 "0.00719760046"。

**檔案:** [CudaConstantVoltageKernels.cpp](openmm-8.4.0/plugins/constantvoltage/platforms/cuda/src/CudaConstantVoltageKernels.cpp)

**修復位置:** Line 55

**修復方案:**
```cpp
// 修改前:
static const double CONVERSION_KJMOL_NM_AU = 18.8973 / 2625.5;  // 0.00719475

// 修改後:
static const double CONVERSION_KJMOL_NM_AU = 18.8973 / 2625.5;  // 0.00719760046
```

---

## 修改檔案清單

| 檔案 | 修改類型 | 涉及修復 |
|-----|---------|---------|
| `CudaConstantVoltageKernels.cpp` | 邏輯修改 | 1, 2, 3, 4, 5, 6, 7 |
| `CudaConstantVoltageKernels.h` | 新增成員 | 4, 6 |
| `constantVoltage.cu` | 無需修改 | - |
| `conductorCharge.cu` | 無需修改 | - |

---

## 驗證測試建議

1. **單元測試:** 純平板電極系統（無 Conductor）- 驗證修復 1
2. **單元測試:** 平板電極 + Buckyball - 驗證修復 2, 3, 5, 6
3. **單元測試:** 平板電極 + Nanotube - 驗證修復 3
4. **積分測試:** 完整 NVT 模擬比對 Python 結果
5. **電荷守恆測試:** 驗證 Σ(cathode + anode + conductors) ≈ 0