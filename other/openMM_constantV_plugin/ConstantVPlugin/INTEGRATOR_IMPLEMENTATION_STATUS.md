# ConstantVIntegrator 實現狀態

**更新時間**: 2025-11-10
**目標**: 100%翻譯教授的Python算法到C++

---

## ✅ 已完成（無Conductor版本）

### 1. 核心架構 ✅
- ✅ `ConstantVIntegrator` 類（類似 `DrudeSCFIntegrator`）
- ✅ `IntegrateConstantVStepKernel` 接口
- ✅ `ReferenceIntegrateConstantVStepKernel` Reference平台實現
- ✅ KernelFactory 註冊
- ✅ Python SWIG 綁定

### 2. SCF算法翻譯 ✅
完整翻譯 `MM_classes.py::Poisson_solver_fixed_voltage` (Line 287-374)

#### ✅ 階段0: 解析電荷計算（Green's Reciprocity）
- ✅ `computeElectrodeChargeAnalytic()` - Line 318-345翻譯
  - ✅ 幾何貢獻: `sign/(4π) * sheet_area * (V/Lgap + V/Lcell)`
  - ✅ 鏡像電荷貢獻: 電解質原子（`electrolyte_atom_indices`）
  - ⚠️ **Conductor鏡像電荷貢獻（L336-344）未實現**

#### ✅ 階段1: SCF迭代（4次）
- ✅ 獲取力 `getState(getForces=True)` - Line 313
- ✅ 更新陰極電荷（Line 323-335）
  - ✅ 電場計算: `Ez = forces[i][2] / q_old` (防除零: `0.9*threshold`)
  - ✅ 邊界條件: `q_new = +2.0/(4π) * area * (V/Lgap + Ez)`
  - ✅ 防歸零: `if |q| < threshold: q = +SMALL_THRESHOLD`
- ✅ 更新陽極電荷（Line 338-350）
  - ✅ 邊界條件: `q_new = -2.0/(4π) * area * (V/Lgap + Ez)`
  - ✅ 防歸零: `if |q| < threshold: q = -1.0*SMALL_THRESHOLD`
- ⚠️ **Conductor電荷更新（L352-361）未實現**

#### ✅ 階段2: Green's校正
- ✅ `scaleChargesAnalytic()` - Line 354-372翻譯（簡化版）
  - ✅ 計算數值總電荷 `Q_numeric`
  - ✅ 計算縮放因子 `scale = Q_analytic / Q_numeric`
  - ✅ 縮放所有電極電荷
- ⚠️ **未實現 `Scale_charges_analytic_general()` (L509-551)**

### 3. MD積分 ✅
- ✅ **正確順序**: 先SCF，後MD積分（Line 242-244）
- ✅ **力計算時機**: SCF結束後用新電荷重新計算力
- ✅ Verlet積分（參考 `DrudeSCFIntegrator`）
- ✅ 約束應用
- ✅ 時間/步數更新

### 4. 物理常數 ✅
- ✅ `CONVERSION_NMBOHR = 18.8973`
- ✅ `CONVERSION_KJMOLNM_AU = 18.8973 / 2625.5`
- ✅ `SMALL_THRESHOLD = 1e-6`
- ✅ **關鍵**: 防除零用 `0.9 * SMALL_THRESHOLD`（不是1.0）

---

## ⚠️ 未完成（Conductor支持）

### 當前狀態
**我的實現 = Python的 `else` 分支（無Conductor）**

```python
# MM_classes.py Line 362-365
if self.Conductor_list:  # ← 未實現
    # Numerical_charge_Conductor
    # 重新計算Q_analytic
    # Scale_charges_analytic_general
else:  # ← 只實現了這個
    self.Cathode.Scale_charges_analytic()
    self.Anode.Scale_charges_analytic()
```

### TODO列表（Conductor支持）

#### 1. 數據模型 ❌
- ❌ Python端: `ConstantVForce.addConductorAtom()`
- ❌ C++端: `ConstantVForce.h/.cpp` 實現
- ❌ Kernel: 添加 `conductorAtomGroups`, `conductorAtomAreas`, `conductorAtomNormals` 等成員變量
- ❌ `initialize()`: 從Force緩存Conductor參數

#### 2. Green's校正 - Part 1 ❌
- ❌ `computeElectrodeChargeAnalytic`: 添加Conductor鏡像電荷貢獻（Line 336-344）

#### 3. 核心SCF循環 ❌
**在 `for (iter = 0; iter < nIterations; iter++)` 內部**：

- ❌ **3.1** 實現 `Numerical_charge_Conductor()` 新函數
  - 翻譯 `MM_classes.py` Line 379-478
  - Step 1: 鏡像電荷
  - Step 2: 轉移電荷（使用 `nx,ny,nz` 法向量和 `dr_center_contact`）
  - **陷阱**: 必須使用緩存的 `sig, eps`，不是 `1.0, 0.0`

- ❌ **3.2** 在更新陽極電荷後添加 `if (Conductor)` 區塊：
  ```cpp
  if (!conductorAtomGroups.empty()) {
      Numerical_charge_Conductor(forces);
      updateParametersInContext();  // Line 358
      // 重新計算解析電荷 (Line 360-361)
      computeElectrodeChargeAnalytic(cathode, ...);
      computeElectrodeChargeAnalytic(anode, ...);
      Scale_charges_analytic_general(false);  // Line 363
  } else {
      scaleChargesAnalytic(cathode, Q_analytic_cathode);
      scaleChargesAnalytic(anode, Q_analytic_anode);
  }
  updateParametersInContext();  // Line 365
  ```

- ❌ **3.3** 實現 `Scale_charges_analytic_general()` 新函數
  - 翻譯 `MM_classes.py` Line 509-551
  - 包含 `if/else` 兩個分支

#### 4. 測試 ❌
- ❌ 創建包含Conductor的測試系統
- ❌ 驗證與Python結果一致

---

## 📋 當前測試計劃

### 階段1: 簡單系統（無Conductor）✅ 可測試
**系統**: 2個陰極原子 + 2個陽極原子 + 1個Na+離子

**測試文件**:
- `tests/test_minimal.py` - Python參考實現
- `tests/test_integrator.py` - ConstantVIntegrator測試
- `tests/test_integrator_simple.py` - 簡化測試

**預期**: C++結果與Python完全一致（<1e-6誤差）

### 階段2: Conductor系統 ❌ 待實現
**需要先完成上述TODO**

---

## 📝 關鍵修復記錄

### 修復1: SCF與積分順序
**問題**: 原來是先積分後SCF
**修復**: 改為先SCF後積分（`ReferenceConstantVKernels.cpp:527-530`）
**對應**: `run_openMM_refactored.py` Line 242-244

### 修復2: 力的計算時機
**問題**: SCF更新電荷後，積分使用舊力
**修復**:
1. 移除 `ConstantVIntegrator::step()` 中的 `calcForcesAndEnergy`
2. 在 `execute()` 中，SCF之後、積分之前調用 `calcForcesAndEnergy`

**流程**:
```cpp
execute() {
    scf_iteration();           // 更新電荷（內部每次迭代計算力）
    calcForcesAndEnergy();     // ← 關鍵！用新電荷重新計算力
    extractForces();           // 獲取新力
    // Verlet積分用新力
}
```

---

## 🎯 下一步

1. ✅ 編譯並安裝
2. 🔄 運行測試（簡單系統，無Conductor）
3. ⏸️ 如需要Conductor支持，實現上述TODO列表

---

## 📚 參考代碼位置

### 教授的Python實現
- `MM_classes.py::Poisson_solver_fixed_voltage` (Line 287-374)
- `Fixed_Voltage_routines.py::compute_Electrode_charge_analytic` (Line 318-345)
- `Fixed_Voltage_routines.py::Scale_charges_analytic` (Line 354-372)
- `MM_classes.py::Numerical_charge_Conductor` (Line 379-478)
- `MM_classes.py::Scale_charges_analytic_general` (Line 509-551)

### C++實現
- `openmmapi/include/ConstantVIntegrator.h`
- `openmmapi/src/ConstantVIntegrator.cpp`
- `platforms/reference/include/ReferenceConstantVKernels.h`
- `platforms/reference/src/ReferenceConstantVKernels.cpp`
  - `initialize()` - Line ~490-511
  - `execute()` - Line ~517-560
  - `scf_iteration()` - Line ~562-636
  - `computeElectrodeChargeAnalytic()` - Line ~643-669
  - `scaleChargesAnalytic()` - Line ~676-699
  - `computeKineticEnergy()` - Line ~706-719

---

## ⚠️ 重要提醒

**當前實現適用於**: 無Conductor的簡單電極系統（教授代碼的 `else` 分支）

**不適用於**: 包含Buckyball、Nanotube等額外導體的系統

**如需Conductor支持**: 必須完成上述TODO列表中的所有項目
