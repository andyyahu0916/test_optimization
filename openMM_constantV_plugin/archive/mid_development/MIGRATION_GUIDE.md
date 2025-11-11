# 遷移指南: 從自定義插件到 OpenMM ConstantPotentialForce

## 目標
將使用自定義 ConstantV 插件的代碼遷移到使用 OpenMM 8.4.0 內建的 `ConstantPotentialForce`

---

## 快速對照表

### 原來的自定義插件
```python
from openmmconstantv import ConstantVForce

force = ConstantVForce()

# 設置電極 A
electrode_a = [0, 1, 2]
for idx in electrode_a:
    force.addElectrodeAtom(idx, potential_a)

# 設置電極 B  
electrode_b = [3, 4, 5]
for idx in electrode_b:
    force.addElectrodeAtom(idx, potential_b)

# 設置電解液
for idx in electrolyte_indices:
    force.addElectrolyteAtom(idx, charge)

# ❌ 問題: 沒有 PME, 使用真空求和
```

### 新的 OpenMM 內建
```python
import openmm as mm

force = mm.ConstantPotentialForce()

# 添加所有粒子
for i in range(total_particles):
    force.addParticle(initial_charge[i])

# 定義電極 A
electrode_a = set([0, 1, 2])
force.addElectrode(
    electrode_a,
    potential_a,      # kJ/mol/e
    gaussian_width,   # nm
    thomas_fermi_scale
)

# 定義電極 B
electrode_b = set([3, 4, 5])
force.addElectrode(
    electrode_b,
    potential_b,
    gaussian_width,
    thomas_fermi_scale
)

# 設置 PME
force.setCutoffDistance(cutoff)
force.setEwaldErrorTolerance(1e-4)

# ✅ 正確: 使用 PME 處理周期性
```

---

## 詳細遷移步驟

### 步驟 1: 移除自定義插件導入

**Before:**
```python
from openmmconstantv import ConstantVForce
```

**After:**
```python
import openmm as mm
# ConstantPotentialForce 已內建於 mm
```

### 步驟 2: 創建 Force

**Before:**
```python
cv_force = ConstantVForce()
```

**After:**
```python
cv_force = mm.ConstantPotentialForce()
```

### 步驟 3: 添加所有粒子

**Before:** (分別添加電極和電解液)
```python
# 電極
for idx in electrode_indices:
    force.addElectrodeAtom(idx, potential)

# 電解液
for idx, charge in zip(electrolyte_indices, charges):
    force.addElectrolyteAtom(idx, charge)
```

**After:** (統一添加)
```python
# 按順序添加所有粒子
for i in range(num_particles):
    initial_charge = charges[i] if i in electrolyte_indices else 0.0
    force.addParticle(initial_charge)
```

### 步驟 4: 定義電極

**Before:**
```python
for idx in electrode_a_indices:
    force.addElectrodeAtom(idx, potential_a)
```

**After:**
```python
electrode_a = set(electrode_a_indices)
potential_a_kj = potential_a_volts * 96.485  # V -> kJ/mol/e

force.addElectrode(
    electrode_a,
    potential_a_kj,
    gaussian_width=0.05,  # nm, 典型值
    thomas_fermi_scale=0.0  # 不使用 TF 模型
)
```

### 步驟 5: 設置 PME 參數

**Before:** (沒有)
```python
# ❌ 舊插件沒有 PME
```

**After:** (必須設置!)
```python
# 設置截斷距離
force.setCutoffDistance(1.0 * mm.unit.nanometer)

# 設置 Ewald 誤差容忍度
force.setEwaldErrorTolerance(1e-4)

# 可選: 手動設置 PME 網格
# force.setPMEParameters(alpha, nx, ny, nz)
```

### 步驟 6: 選擇求解方法

**Before:** (固定矩陣方法)
```python
# 舊插件使用預計算的電容矩陣
```

**After:** (可選 CG 或 Matrix)
```python
# CG 方法 - 適用於動態系統
force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
force.setCGErrorTolerance(0.1)  # kJ/mol/e

# 或 Matrix 方法 - 僅適用於固定電極
# force.setConstantPotentialMethod(mm.ConstantPotentialForce.Matrix)
```

### 步驟 7: 獲取求解的電荷

**Before:** (需要手動讀取)
```python
# 舊插件可能需要從 context 中手動提取
```

**After:** (簡單調用)
```python
charges = force.getCharges(context)

for i, q in enumerate(charges):
    charge_value = q.value_in_unit(mm.unit.elementary_charge)
    print(f"粒子 {i}: {charge_value} e")
```

---

## 完整遷移示例

### 原來的代碼 (假設)

```python
# 使用自定義插件
from openmmconstantv import ConstantVForce
import openmm as mm

# 創建系統
system = mm.System()
for i in range(num_particles):
    system.addParticle(mass[i])

# 創建 ConstantV Force
cv_force = ConstantVForce()

# 電極 A (石墨烯層 1)
for idx in graphene_layer1:
    cv_force.addElectrodeAtom(idx, 1.0)  # 1.0 V

# 電極 B (石墨烯層 2)
for idx in graphene_layer2:
    cv_force.addElectrodeAtom(idx, 0.0)  # 接地

# 電解液
for idx, q in zip(electrolyte_indices, electrolyte_charges):
    cv_force.addElectrolyteAtom(idx, q)

# 設置預計算的電容矩陣 (如果有)
# cv_force.setInverseCapacitanceMatrix(C_inv)

system.addForce(cv_force)
```

### 遷移後的代碼

```python
# 使用 OpenMM 內建
import openmm as mm

# 創建系統
system = mm.System()
for i in range(num_particles):
    system.addParticle(mass[i])

# 設置週期性盒子
system.setDefaultPeriodicBoxVectors(
    mm.Vec3(box_x, 0, 0),
    mm.Vec3(0, box_y, 0),
    mm.Vec3(0, 0, box_z)
)

# 創建 ConstantPotentialForce
cv_force = mm.ConstantPotentialForce()

# 添加所有粒子
for i in range(num_particles):
    if i in electrolyte_indices:
        idx_in_list = electrolyte_indices.index(i)
        charge = electrolyte_charges[idx_in_list]
    else:
        charge = 0.0  # 電極粒子初始電荷
    cv_force.addParticle(charge)

# 定義電極 A
electrode_a = set(graphene_layer1)
potential_a = 1.0 * 96.485  # 1.0 V -> kJ/mol/e
cv_force.addElectrode(
    electrode_a,
    potential_a,
    gaussian_width=0.05,  # nm
    thomas_fermi_scale=0.0
)

# 定義電極 B
electrode_b = set(graphene_layer2)
potential_b = 0.0  # 接地
cv_force.addElectrode(
    electrode_b,
    potential_b,
    gaussian_width=0.05,
    thomas_fermi_scale=0.0
)

# 設置 PME (關鍵!)
cv_force.setCutoffDistance(1.0 * mm.unit.nanometer)
cv_force.setEwaldErrorTolerance(1e-4)

# 選擇求解方法
if electrodes_are_fixed:
    # Matrix 方法更快
    cv_force.setConstantPotentialMethod(mm.ConstantPotentialForce.Matrix)
else:
    # CG 方法支持動態
    cv_force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
    cv_force.setCGErrorTolerance(0.1)

system.addForce(cv_force)

# 創建 Context
integrator = mm.LangevinIntegrator(...)
context = mm.Context(system, integrator, platform)
context.setPositions(positions)

# 運行模擬
integrator.step(num_steps)

# 獲取電荷分布
charges = cv_force.getCharges(context)
print("電極電荷:", charges[:len(electrode_a)])
```

---

## 常見問題

### Q1: 電容矩陣怎麼辦?

**A:** 不需要預計算!

- **CG 方法**: 自動求解,無需電容矩陣
- **Matrix 方法**: OpenMM 會在第一步自動計算並緩存

### Q2: 如何設置電壓?

**A:** 使用 `addElectrode()` 時設置

```python
# 電壓單位轉換: 1 V = 96.485 kJ/mol/e
voltage_volts = 1.0
potential_kj = voltage_volts * 96.485

force.addElectrode(
    particles,
    potential_kj,  # kJ/mol/e
    gaussian_width,
    thomas_fermi_scale
)
```

### Q3: 如何處理排除項 (exclusions)?

**A:** 使用 `addException()`

```python
# 排除粒子 i 和 j 之間的交互作用
force.addException(i, j, chargeProd=0.0)

# 或批量創建排除項
force.createExceptionsFromBonds(bonds, coulomb14Scale=0.83333)
```

### Q4: PME 參數如何選擇?

**A:** 通常使用自動選擇

```python
# 自動選擇 (推薦)
force.setEwaldErrorTolerance(1e-4)

# 或手動指定
force.setPMEParameters(
    alpha=0.34,  # 1/nm
    nx=64,
    ny=64,
    nz=64
)
```

### Q5: 如何遷移現有的模擬?

**A:** 逐步遷移

1. 保留原始代碼作為備份
2. 創建新的測試腳本使用 `ConstantPotentialForce`
3. 運行短時間模擬比較結果
4. 確認無誤後替換主模擬腳本

---

## 測試檢查清單

遷移完成後,確認:

- [ ] 系統能量合理
- [ ] 電極電荷總和接近預期
- [ ] 電解液電荷保持固定
- [ ] 模擬穩定運行
- [ ] PME 參數已設置
- [ ] 週期性邊界條件正確
- [ ] 電壓轉換正確 (V -> kJ/mol/e)
- [ ] 電荷分布可以正常讀取

---

## 性能優化建議

1. **固定電極使用 Matrix 方法**
   ```python
   force.setConstantPotentialMethod(mm.ConstantPotentialForce.Matrix)
   ```

2. **調整 CG 容忍度**
   ```python
   force.setCGErrorTolerance(0.1)  # 較寬鬆 = 更快
   ```

3. **優化 PME 網格**
   ```python
   # 讓 OpenMM 自動選擇,通常已經很好
   force.setEwaldErrorTolerance(1e-4)
   ```

---

## 參考資源

- **OpenMM 文檔**: http://docs.openmm.org/
- **源碼**: `openmm-8.4.0/openmmapi/include/openmm/ConstantPotentialForce.h`
- **演示**: `demo_builtin_constantpotential.py`
- **論文**: Dufils et al., PRL 123, 195501 (2019)

---

**最後更新**: 2024-11-04  
**狀態**: ✅ 就緒 - 可以開始遷移
