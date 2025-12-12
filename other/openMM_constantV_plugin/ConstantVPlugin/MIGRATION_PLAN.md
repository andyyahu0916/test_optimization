# 從舊Plugin遷移到教授算法 - 完整計劃

## 📊 算法對比

### ❌ 舊Plugin算法（錯誤）
```cpp
// File: ReferenceConstantVKernels.cpp
double ReferenceCalcConstantVKernel::execute() {
    // 1. 計算電解質對電極的電位
    for (int i = 0; i < N; i++) {
        E_f[i] = 0.0;
        for (int j = 0; j < M; j++) {
            // 手動計算Coulomb電位
            E_f[i] += k * q_j / r_ij;  // ❌ 忽略PME、VDW等
        }
    }

    // 2. 矩陣運算求解電荷
    q_e = C_inv * (V - E_f);  // ❌ 宏觀近似，不是自洽場

    // 3. 更新一次
    updateParametersInContext();  // ❌ 沒有迭代
}
```

**問題**：
1. ❌ 手動計算電位 → 忽略了OpenMM的PME、極化、VDW等
2. ❌ 逆電容矩陣 → 宏觀近似，失去原子級解析度
3. ❌ 無SCF迭代 → 不是自洽解
4. ❌ 無Green's校正 → 電荷不守恆

---

### ✅ 教授算法（正確）
```python
# File: MM_classes.py::Poisson_solver_fixed_voltage()
def Poisson_solver_fixed_voltage(self, Niterations=4):
    # 0. 計算解析總電荷（Green's reciprocity）
    self.Cathode.compute_Electrode_charge_analytic(...)
    self.Anode.compute_Electrode_charge_analytic(...)

    # 1. SCF迭代
    for i_iter in range(Niterations):
        # 1a. 獲取OpenMM計算的完整力
        state = context.getState(getForces=True)
        forces = state.getForces()  # ✅ 包含所有交互

        # 1b. 更新陰極電荷
        for atom in Cathode.electrode_atoms:
            Ez = forces[atom.index][2] / q_old  # ✅ 從力算電場
            q_new = 2/(4π) * area * (V/Lgap + Ez) * conversion
            atom.charge = q_new
            nbondedForce.setParticleParameters(...)

        # 1c. 更新陽極電荷（類似）

        # 1d. Green's reciprocity校正
        Scale_charges_analytic_general()  # ✅ 守恆

        # 1e. 更新OpenMM
        updateParametersInContext(context)  # ✅ 下次迭代用新電荷
```

**優勢**：
1. ✅ 利用OpenMM完整力場 → PME、極化、VDW全包含
2. ✅ SCF迭代 → 自洽解
3. ✅ Green's校正 → 精確守恆
4. ✅ 動態響應 → 每200fs重新求解

---

## 🔧 需要修改的文件

### 1. `openmmapi/include/ConstantVForce.h`

#### 刪除
```cpp
// ❌ 刪除逆電容矩陣相關
std::vector<double> invCapMatrix;
void setInverseCapacitanceMatrix(const std::vector<double>& flattenedMatrix);
const std::vector<double>& getInverseCapacitanceMatrix() const;
```

#### 新增
```cpp
// ✅ 新增教授算法需要的參數
private:
    // 電極分類
    std::vector<int> cathodeAtomIndices;   // 陰極原子索引
    std::vector<int> anodeAtomIndices;     // 陽極原子索引

    // 系統幾何參數
    double voltage;        // 總電壓降 (V)
    double Lgap;          // 真空間隙 (nm)
    double Lcell;         // 電極間距 (nm)
    double totalArea;     // 電極總面積 (nm^2)

    // 每個原子的面積
    std::vector<double> areaPerAtom;  // nm^2

    // SCF參數
    int nIterations;      // SCF迭代次數（默認4）

    // 轉換因子
    static constexpr double CONVERSION_FACTOR = 0.0072;  // 從教授代碼

public:
    // ✅ 新增setter/getter
    void setVoltage(double v) { voltage = v; }
    double getVoltage() const { return voltage; }

    void setElectrodeGeometry(double gap, double cell, double area);
    void getElectrodeGeometry(double& gap, double& cell, double& area) const;

    void setCathodeAtoms(const std::vector<int>& atoms);
    void setAnodeAtoms(const std::vector<int>& atoms);

    void setAreaPerAtom(const std::vector<double>& areas);

    void setNumIterations(int n) { nIterations = n; }
    int getNumIterations() const { return nIterations; }
```

---

### 2. `platforms/reference/include/ReferenceConstantVKernels.h`

#### 新增私有成員
```cpp
class ReferenceCalcConstantVKernel : public CalcConstantVKernel {
private:
    // ... 現有成員 ...

    // ✅ 新增：電極分類
    std::vector<int> cathodeAtomIndices;
    std::vector<int> anodeAtomIndices;

    // ✅ 新增：幾何參數
    double voltage;
    double Lgap;
    double Lcell;
    double totalArea;
    std::vector<double> areaPerAtom;

    // ✅ 新增：SCF參數
    int nIterations;

    // ✅ 新增：當前電荷（用於迭代）
    std::vector<double> currentCharges;

    // ✅ 新增：輔助函數
    void computeAnalyticCharge(
        const std::vector<int>& electrodeAtoms,
        const std::vector<OpenMM::Vec3>& positions,
        double sign,  // +1 for cathode, -1 for anode
        double& Q_analytic
    );

    void scaleChargesAnalytic(
        const std::vector<int>& electrodeAtoms,
        double Q_analytic
    );
};
```

---

### 3. `platforms/reference/src/ReferenceConstantVKernels.cpp`

#### 完全重寫 `execute()` 方法

```cpp
double ReferenceCalcConstantVKernel::execute(
    ContextImpl& context,
    bool includeForces,
    bool includeEnergy
) {
    const int N_cathode = cathodeAtomIndices.size();
    const int N_anode = anodeAtomIndices.size();
    const int N_total = N_cathode + N_anode;

    if (N_total == 0)
        return 0.0;

    // ══════════════════════════════════════════════════════════
    // 階段0: 計算解析總電荷 (Green's Reciprocity)
    // ══════════════════════════════════════════════════════════
    vector<Vec3> positions = extractPositions(context);

    double Q_analytic_cathode, Q_analytic_anode;
    computeAnalyticCharge(cathodeAtomIndices, positions, +1.0, Q_analytic_cathode);
    computeAnalyticCharge(anodeAtomIndices, positions, -1.0, Q_analytic_anode);

    // ══════════════════════════════════════════════════════════
    // 階段1: SCF迭代 (核心算法)
    // ══════════════════════════════════════════════════════════
    for (int iter = 0; iter < nIterations; iter++) {

        // ─────────────────────────────────────────────────────
        // 步驟1: 獲取當前系統的力
        // ─────────────────────────────────────────────────────
        State state = context.getOwner().getState(State::Forces);
        const vector<Vec3>& forces = state.getForces();

        // ─────────────────────────────────────────────────────
        // 步驟2: 更新陰極電荷
        // ─────────────────────────────────────────────────────
        for (size_t i = 0; i < cathodeAtomIndices.size(); i++) {
            int atomIdx = cathodeAtomIndices[i];
            double q_old = currentCharges[atomIdx];

            // 2a. 從力計算電場 (Ez = Fz / q)
            double Ez_external = 0.0;
            if (fabs(q_old) > 1e-6) {
                Ez_external = forces[atomIdx][2] / q_old;
            }

            // 2b. 邊界條件求解新電荷
            // q = (2 / 4π) * A * (V/Lgap + Ez) * conversion
            double q_new = (2.0 / (4.0 * M_PI)) *
                          areaPerAtom[i] *
                          (voltage / Lgap + Ez_external) *
                          CONVERSION_FACTOR;

            // 2c. 防止數值零（教授的threshold）
            if (fabs(q_new) < 1e-6) {
                q_new = 1e-6;  // 陰極為正
            }

            // 2d. 更新
            currentCharges[atomIdx] = q_new;
            nonbondedForce->setParticleParameters(
                atomIdx,
                q_new,
                particleSigmas[atomIdx],
                particleEpsilons[atomIdx]
            );
        }

        // ─────────────────────────────────────────────────────
        // 步驟3: 更新陽極電荷 (類似陰極，符號相反)
        // ─────────────────────────────────────────────────────
        for (size_t i = 0; i < anodeAtomIndices.size(); i++) {
            int atomIdx = anodeAtomIndices[i];
            double q_old = currentCharges[atomIdx];

            double Ez_external = 0.0;
            if (fabs(q_old) > 1e-6) {
                Ez_external = forces[atomIdx][2] / q_old;
            }

            // 注意：陽極是 -2.0 而不是 +2.0
            double q_new = -(2.0 / (4.0 * M_PI)) *
                          areaPerAtom[N_cathode + i] *
                          (voltage / Lgap + Ez_external) *
                          CONVERSION_FACTOR;

            if (fabs(q_new) < 1e-6) {
                q_new = -1e-6;  // 陽極為負
            }

            currentCharges[atomIdx] = q_new;
            nonbondedForce->setParticleParameters(
                atomIdx,
                q_new,
                particleSigmas[atomIdx],
                particleEpsilons[atomIdx]
            );
        }

        // ─────────────────────────────────────────────────────
        // 步驟4: Green's Reciprocity 解析校正
        // ─────────────────────────────────────────────────────
        scaleChargesAnalytic(cathodeAtomIndices, Q_analytic_cathode);
        scaleChargesAnalytic(anodeAtomIndices, Q_analytic_anode);

        // ─────────────────────────────────────────────────────
        // 步驟5: 更新OpenMM Context
        // ─────────────────────────────────────────────────────
        // 關鍵！下次迭代會用新電荷重新計算力
        nonbondedForce->updateParametersInContext(context.getOwner());
    }

    return 0.0;  // 不貢獻能量
}
```

#### 實現 `computeAnalyticCharge()`

```cpp
void ReferenceCalcConstantVKernel::computeAnalyticCharge(
    const std::vector<int>& electrodeAtoms,
    const std::vector<Vec3>& positions,
    double sign,  // +1.0 for cathode, -1.0 for anode
    double& Q_analytic
) {
    // ═══════════════════════════════════════════════════════════
    // 教授算法：Fixed_Voltage_routines.py::318-345
    // ═══════════════════════════════════════════════════════════

    // 1. 幾何貢獻（平行板電容器公式）
    Q_analytic = sign / (4.0 * M_PI) * totalArea *
                 (voltage / Lgap + voltage / Lcell) *
                 CONVERSION_FACTOR;

    // 2. 鏡像電荷貢獻（電解質原子）
    // 計算對面電極的z位置
    double z_opposite = 0.0;
    if (sign > 0) {  // cathode
        // 對面是anode
        if (!anodeAtomIndices.empty()) {
            z_opposite = positions[anodeAtomIndices[0]][2];
        }
    } else {  // anode
        // 對面是cathode
        if (!cathodeAtomIndices.empty()) {
            z_opposite = positions[cathodeAtomIndices[0]][2];
        }
    }

    // 遍歷所有電解質原子
    for (int idx : electrolyteAtomIndices) {
        double q_i = fixedCharges[idx];  // 電解質的固定電荷
        double z_atom = positions[idx][2];
        double z_distance = fabs(z_atom - z_opposite);

        // 鏡像電荷貢獻
        Q_analytic += (z_distance / Lcell) * (-q_i);
    }

    // 3. 導體貢獻（如果有BuckyBall/Nanotube）
    // TODO: 實現Conductor支持
}
```

#### 實現 `scaleChargesAnalytic()`

```cpp
void ReferenceCalcConstantVKernel::scaleChargesAnalytic(
    const std::vector<int>& electrodeAtoms,
    double Q_analytic
) {
    // ═══════════════════════════════════════════════════════════
    // 教授算法：Fixed_Voltage_routines.py::354-372
    // ═══════════════════════════════════════════════════════════

    // 1. 計算數值總電荷
    double Q_numeric = 0.0;
    for (int idx : electrodeAtoms) {
        Q_numeric += currentCharges[idx];
    }

    // 2. 縮放因子（防止除零）
    double scale_factor = 1.0;
    if (fabs(Q_numeric) > 1e-6) {
        scale_factor = Q_analytic / Q_numeric;
    }

    // 3. 縮放所有電極電荷
    if (scale_factor > 0.0) {
        for (int idx : electrodeAtoms) {
            currentCharges[idx] *= scale_factor;
            nonbondedForce->setParticleParameters(
                idx,
                currentCharges[idx],
                particleSigmas[idx],
                particleEpsilons[idx]
            );
        }
    }
}
```

---

## 📐 轉換常數

從教授代碼提取：
```python
# Fixed_Voltage_routines.py::36-38
conversion_nmBohr = 18.8973
conversion_KjmolNm_Au = conversion_nmBohr / 2625.5  # 0.00719924...
conversion_eV_Kjmol = 96.487
```

C++實現：
```cpp
// ReferenceConstantVKernels.cpp
static constexpr double CONVERSION_NMBOHR = 18.8973;
static constexpr double CONVERSION_KJMOLNM_AU = CONVERSION_NMBOHR / 2625.5;  // ~0.0072
static constexpr double CONVERSION_EV_KJMOL = 96.487;
```

---

## 🧪 驗證計劃

### 1. 單元測試
創建小型測試系統：
- 2個電極（各10個原子）
- 10個電解質原子
- 已知幾何參數

### 2. 對比驗證
運行教授的Python版本和新Plugin：
```python
# 教授版本
MMsys.Poisson_solver_fixed_voltage(Niterations=4)
Q_python = [atom.charge for atom in MMsys.Cathode.electrode_atoms]

# Plugin版本
constantVForce.execute()
Q_cpp = [get charge from OpenMM]

# 驗證
assert np.allclose(Q_python, Q_cpp, rtol=1e-5, atol=1e-6)
```

### 3. 物理驗證
- ✅ 電荷守恆：`Q_cathode + Q_anode ≈ 0`
- ✅ 總電荷匹配解析值：`Q_numeric ≈ Q_analytic`
- ✅ 能量收斂：每次迭代後能量變化 < threshold

---

## 📊 預期結果

| 指標 | 教授Python | 新Plugin (CPU) | 預期改進 |
|------|-----------|---------------|---------|
| **精度** | ✅ 正確 | ✅ 相同 | - |
| **物理** | ✅ SCF | ✅ SCF | - |
| **速度** | 70 ms/步 | 20-30 ms/步 | **2-3x** |
| **內存** | ~100 MB | ~50 MB | 2x |

---

## 🚀 實施順序

### Week 1: API重構
- [ ] 修改 `ConstantVForce.h`（刪除invCapMatrix，新增參數）
- [ ] 修改 `ConstantVForce.cpp`（實現新的setter/getter）
- [ ] 更新 `ReferenceConstantVKernels.h`（新增私有成員）

### Week 2: 算法實現
- [ ] 實現 `computeAnalyticCharge()`
- [ ] 實現 `scaleChargesAnalytic()`
- [ ] 重寫 `execute()` 主循環（4次SCF迭代）

### Week 3: 測試驗證
- [ ] 創建單元測試
- [ ] 與教授Python版本對比
- [ ] 修復任何精度問題

### Week 4: CUDA移植
- [ ] 同步修改 `CudaConstantVKernels.cu`
- [ ] 性能測試
- [ ] 文檔更新

---

## 💡 關鍵注意事項

### 1. 保留教授的數值細節
- ✅ 小threshold (1e-6) 防止除零
- ✅ SCF迭代次數（4次）
- ✅ 轉換常數精度
- ✅ Green's校正順序

### 2. OpenMM State獲取
```cpp
// 教授的Python: getState(getForces=True)
State state = context.getOwner().getState(State::Forces);
```

### 3. 參數更新順序
```cpp
// 教授的順序（重要！）
for (atom : cathode) { update_charge(atom); }
for (atom : anode) { update_charge(atom); }
scale_charges_analytic();
updateParametersInContext();  // 只在最後調用一次
```

---

## ❓ 需要澄清的問題

1. **電解質原子列表**：
   - 教授版本遍歷 `MMsys.electrolyte_atom_indices`
   - Plugin如何獲取？從哪裡傳入？

2. **面積計算**：
   - 教授：`area_atom = sheet_area / Natoms`
   - Plugin：需要預計算還是動態計算？

3. **Conductor支持**（BuckyBall/Nanotube）：
   - 第一版先支持平面電極？
   - 之後再加Conductor？

---

## 📚 參考文件對照表

| 教授文件 | 對應Plugin文件 | 行號 |
|---------|--------------|------|
| `MM_classes.py::Poisson_solver_fixed_voltage` | `ReferenceConstantVKernels.cpp::execute` | 287-374 |
| `Fixed_Voltage_routines.py::compute_Electrode_charge_analytic` | `computeAnalyticCharge()` | 318-345 |
| `Fixed_Voltage_routines.py::Scale_charges_analytic` | `scaleChargesAnalytic()` | 354-372 |
| `Fixed_Voltage_routines.py::conversion_*` | `CONVERSION_*` 常數 | 36-38 |

---

**總結**：架構保留，核心算法完全替換為教授的SCF方法！🎯
