# Beta Version: Full Feature Port Plan

**目標**: 完全照抄 Python Original，確保物理、數學、位元級別一致性

**原則**:
- ✅ 先照抄為原則，不考慮優化
- ✅ 只有程式語言不同，其他邏輯完全相同
- ✅ 逐步移植，每個功能都要通過測試

---

## 📋 需要移植的功能清單

### 1. 複雜電極支持（Conductors）

#### 1.1 Buckyball_Virtual（巴克球導體）
**位置**: `Fixed_Voltage_routines.py:391-473`

**核心特性**:
- 球形導體（C60 巴克球）
- 與平面電極不同：虛擬層之間**有**靜電交互作用（因為需要鏡像電荷貢獻）
- 需要計算：
  - 球心位置 (r_center)
  - 球半徑 (radius)
  - 表面法向量 (normal vectors) 在每個原子位置
  - 每原子面積 (area_atom = 4πr² / N_atoms)

**關鍵方法**:
```python
def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys,
             chain_flag, exclude_element):
    # 繼承 Conductor_Virtual
    # 計算球心、半徑、表面法向量
    # 找到最近的接觸導體
```

**與平面電極的差異**:
| 特性 | 平面電極 | Buckyball |
|------|---------|-----------|
| 虛擬層電荷交互 | 完全排除 | 需要保留（鏡像電荷）|
| 幾何參數 | 面積/原子 | 球面積/原子 |
| 法向量 | (0,0,1) 固定 | 徑向（每個原子不同）|
| 邊界條件 | Maxwell（平面）| Maxwell（球面）|

---

#### 1.2 Nanotube_Virtual（奈米管導體）
**位置**: `Fixed_Voltage_routines.py:482-589`

**核心特性**:
- 圓柱形導體（碳奈米管）
- 需要輸入奈米管軸向 (axis)
- 需要計算：
  - 管中心位置 (r_center)
  - 管半徑 (radius)
  - 徑向法向量（投影到垂直於軸向的平面）
  - 每原子面積 (area_atom = 2πr * length / N_atoms)

**關鍵方法**:
```python
def __init__(self, electrode_identifier, electrode_type, Voltage, MMsys,
             chain_flag, exclude_element, axis):
    # 計算管中心、半徑
    # 投影法向量到垂直於軸向的平面
    # 找到最近的接觸導體

def project_orthogonal_to_axis(self, vec_in):
    # 投影到垂直於軸向的平面
    vec_out = vec_in - axis * dot(vec_in, axis)
```

**與平面電極的差異**:
| 特性 | 平面電極 | Nanotube |
|------|---------|----------|
| 幾何形狀 | 平面 | 圓柱 |
| 法向量 | (0,0,1) | 徑向（垂直於軸向）|
| 面積計算 | box_x * box_y | 2πr * length |

---

#### 1.3 Conductor 電荷計算（數值方法）
**位置**: `MM_classes.py:388-508`

**核心邏輯**:
```python
def Numerical_charge_Conductor(self, Conductor, forces):
    """
    為 Buckyball/Nanotube 計算電荷（數值方法）

    步驟：
    1. 計算每個原子的法向電場分量: E_n = dot(F/q, normal)
    2. 計算接觸電勢差: V_contact
    3. 計算新電荷: q_new = area_atom / (4π) * (V_contact + E_n) * conversion
    4. 更新 virtual 層電荷
    5. 計算總電荷轉移，分配到 real 層（均勻分佈）
    """
```

**關鍵差異**:
- 平面電極：只考慮 z 方向電場
- Conductor：需要投影到表面法向量

---

### 2. QM/MM 集成支持

**位置**: `MM_classes.py` 多處 (Line 50, 60, 80, 144, 151, 158, 168, 290, 371)

**核心功能**:
- 在 SCF 迭代前關閉 Vext grid：`platform.setPropertyValue(context, 'ReferenceVextGrid', "false")`
- 在 SCF 迭代後開啟 Vext grid：`platform.setPropertyValue(context, 'ReferenceVextGrid', "true")`
- 提供方法獲取原子電荷和位置：`get_element_charge_for_atom_lists()`
- 提供方法獲取原子位置：`get_positions_for_atom_lists()`

**用途**:
- 與外部 QM solver 交互
- 在 SCF 過程中暫時關閉 QM 外部勢能

**關鍵方法**:
```python
def get_element_charge_for_atom_lists(self, atom_lists):
    """
    返回原子的元素符號和電荷（用於 QM 計算）

    Returns:
    --------
    lists: [[element, charge], ...]
    """

def get_positions_for_atom_lists(self, atom_lists):
    """
    返回原子位置（用於 QM 計算）

    Returns:
    --------
    positions: [[x, y, z], ...]
    """
```

---

### 3. Monte Carlo Barostat（電極移動）

**位置**: `MM_classes.py:637-755`

**核心功能**:
- Monte Carlo 方法調整系統壓力
- 移動電極位置（Anode 或 Cathode）
- Metropolis 接受/拒絕判據

**關鍵方法**:
```python
def MC_Barostat_step(self):
    """
    MC Barostat 步驟

    流程：
    1. 計算當前能量和壓力
    2. 嘗試移動電極（shift electrode）
    3. 更新所有原子位置（保持相對關係）
    4. 重新計算能量
    5. Metropolis 判據決定接受/拒絕
    """
```

**參數類別**:
```python
class MC_parameters:
    def __init__(self, temperature, celldim, electrode_move="Anode",
                 pressure=1.0*bar, barofreq=25, shiftscale=0.2):
        self.temperature = temperature
        self.pressure = pressure
        self.barofreq = barofreq  # MC 頻率
        self.shiftscale = shiftscale  # 移動尺度
        self.electrode_move = electrode_move  # "Anode" or "Cathode"
```

**用途**:
- 系統密度平衡
- 壓力控制（與 Langevin dynamics 不同）

---

### 4. Umbrella Sampling（傘狀勢能採樣）

**位置**: `MM_classes.py:756-822`

**核心功能**:
- 添加 umbrella 約束勢能：`V_umbrella = 0.5 * k * (r - r0)²`
- 用於增強採樣（例如，離子穿透能壘）

**關鍵方法**:
```python
def setumbrella(self, mol1, k, **kwargs):
    """
    設置 umbrella sampling

    Parameters:
    -----------
    mol1: 分子 1 的原子列表
    k: 彈簧常數（force constant）
    r0: 參考距離（如果不提供，使用當前距離）
    mol2: 分子 2 的原子列表（可選）

    約束類型：
    1. 如果只有 mol1：約束到固定位置
    2. 如果有 mol1 和 mol2：約束兩個分子之間的距離
    """
```

**實現**:
- 使用 OpenMM 的 `CustomCentroidBondForce`
- 計算質心（center of mass）
- 添加彈簧勢能

---

## 🎯 移植策略

### 階段 1：數據結構準備（優先級：P0）

#### 1.1 擴展 ConstantVForce
```cpp
class ConstantVForce {
    // 新增：Conductor 支持
    enum ConductorType { BUCKYBALL, NANOTUBE };

    struct ConductorInfo {
        ConductorType type;
        int virtualChainIdx;
        int realChainIdx;
        std::vector<int> atomIndices;
        std::vector<double> normalVectors;  // (nx, ny, nz) for each atom
        double radius;
        // For nanotube only:
        double axis[3];
        double length;
    };

    std::vector<ConductorInfo> conductors;

    // 新增：QM/MM 支持
    bool enableQMMM = false;

    // 新增：Umbrella sampling
    struct UmbrellaInfo {
        std::vector<int> group1Atoms;
        std::vector<int> group2Atoms;
        double k;  // force constant
        double r0; // reference distance
    };
    std::vector<UmbrellaInfo> umbrellas;
};
```

---

### 階段 2：Conductor 移植（優先級：P1）

#### Step 2.1：移植 Buckyball_Virtual
**目標**：完全照抄 `Fixed_Voltage_routines.py:391-473`

**實現檔案**：
- `openmmapi/include/ConstantVForce.h` - 添加 Buckyball API
- `platforms/reference/src/ReferenceConstantVKernels.cpp` - Reference 實現
- `platforms/cuda/src/CudaConstantVKernels.cu` - CUDA 實現

**測試**：
- 與 Python original 對比巴克球電荷分佈
- 確保數值誤差 < 1e-10

---

#### Step 2.2：移植 Nanotube_Virtual
**目標**：完全照抄 `Fixed_Voltage_routines.py:482-589`

**額外挑戰**：
- 需要實現向量投影：`project_orthogonal_to_axis()`
- 需要正確處理圓柱幾何

---

#### Step 2.3：移植 Numerical_charge_Conductor
**目標**：完全照抄 `MM_classes.py:388-508`

**關鍵**：
- 計算法向電場分量
- 處理接觸電勢差
- 電荷轉移到 real 層

---

### 階段 3：QM/MM 移植（優先級：P2）

**挑戰**：
- 需要與 OpenMM 的 Platform properties 交互
- CUDA 平台可能不支持 Vext grid

**實現**：
- 添加 `setQMMMEnabled()` API
- 在 SCF 前後切換 Vext grid
- 提供原子資訊接口

---

### 階段 4：MC Barostat 移植（優先級：P3）

**挑戰**：
- 需要修改 box vectors
- 需要實現 Metropolis 判據
- 需要正確處理 PBC

**實現**：
- 創建新的 Integrator：`ConstantVMCBarostatIntegrator`
- 實現電極移動邏輯
- 實現能量計算和接受/拒絕

---

### 階段 5：Umbrella Sampling 移植（優先級：P4）

**挑戰**：
- 需要使用 OpenMM 的 `CustomCentroidBondForce`
- 需要正確計算質心

**實現**：
- 添加 `addUmbrellaPotential()` API
- 自動創建 CustomCentroidBondForce
- 支持兩種模式（固定位置 vs 兩分子之間）

---

## 📊 工作量估計

| 功能 | 代碼行數（Python）| 預估工作量（天）| 優先級 | 複雜度 |
|------|----------------|---------------|--------|--------|
| Buckyball_Virtual | ~83 行 | 3-5 天 | P1 | 高 |
| Nanotube_Virtual | ~108 行 | 4-6 天 | P1 | 高 |
| Numerical_charge_Conductor | ~120 行 | 5-7 天 | P1 | 很高 |
| QM/MM 支持 | ~50 行（分散）| 2-3 天 | P2 | 中 |
| MC Barostat | ~118 行 | 6-8 天 | P3 | 很高 |
| Umbrella Sampling | ~67 行 | 2-4 天 | P4 | 中 |
| **總計** | ~546 行 | **22-33 天** | | |

---

## ✅ 移植檢查清單

每個功能移植完成後，必須通過以下測試：

### 物理正確性測試
- [ ] 與 Python original 逐行對比
- [ ] 數值誤差 < 1e-10（float 精度內）
- [ ] 能量守恆（如適用）
- [ ] 電荷守恆

### 數學一致性測試
- [ ] 所有公式與 Python 完全一致
- [ ] 所有常數與 Python 完全一致
- [ ] 所有閾值與 Python 完全一致

### 位元級測試（如可能）
- [ ] 使用相同輸入，輸出完全相同
- [ ] 浮點數精度在可接受範圍內

---

## 🚀 第一步：從哪裡開始？

**建議順序**：
1. **Buckyball_Virtual**（最簡單的 Conductor）
2. **Numerical_charge_Conductor**（核心算法）
3. **Nanotube_Virtual**（最複雜的 Conductor）
4. **QM/MM**（相對獨立）
5. **Umbrella Sampling**（相對獨立）
6. **MC Barostat**（最複雜，最後做）

---

## 📝 當前狀態

- ✅ 已創建 beta 分支：`claude/beta-full-features-01GdZDSQkwZnbgohdRiSpfBV`
- ⏸️ 準備開始移植 Buckyball_Virtual
- ⏸️ 等待確認移植優先級

---

Generated: 2025-11-19
Branch: claude/beta-full-features-01GdZDSQkwZnbgohdRiSpfBV
Goal: 100% 功能完整性，與 Python Original 完全一致
