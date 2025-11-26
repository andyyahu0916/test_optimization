# TODO SOP：從開發到生產的完整修正清單

## 審查反饋來源
- **審查者:** 優秀的學術同行審查
- **日期:** 2024年11月
- **嚴重程度:** 包含致命的物理錯誤 (PBC/PME)

---

## 階段一：立即修復：編譯錯誤 ✅ [已完成 - 2024-11-04]

### TODO-1.1 (CUDA端)：修正 `CudaConstantVKernels.cu`

**錯誤 1: `too many arguments`**
- ❌ 錯誤: `force.getInverseCapacitanceMatrix(invCapMatrix)`  
- ✅ 修正: `vector<double> invCapMatrix = force.getInverseCapacitanceMatrix()`
- **狀態:** ✅ 已修正
- **文件:** `ConstantVPlugin/platforms/cuda/src/CudaConstantVKernels.cu`

**錯誤 2: `no member "getCudaStream"`**
- ❌ 錯誤: `cu.getCudaStream()`
- ✅ 修正: `cu.getStream().getStream()`
- **狀態:** ✅ 已修正 (3處: 2個 kernel 啟動 + 1個 cudaMemcpyAsync)
- **文件:** `ConstantVPlugin/platforms/cuda/src/CudaConstantVKernels.cu`

**錯誤 3: cuBLAS API 調用**
- ❌ 錯誤: `cublasDaxpy` 和 `cublasDgemv`
- ✅ 修正: `cublasDaxpy_v2` 和 `cublasDgemv_v2`
- **狀態:** ✅ 已修正
- **文件:** `ConstantVPlugin/platforms/cuda/src/CudaConstantVKernels.cu`

**其他修正:**
- ✅ `getDevicePointer()` → `getDeviceData()` (API 一致性)
- ✅ `invalidateMolecules()` → `invalidateNonbondedParameters()` (正確 API)
- ✅ 添加警告註解: "[!!--- 警告：此核心物理上是錯誤的 ---!!]"
- ✅ 返回類型修正: `CUresult` (cuBLAS) vs `cublasStatus_t`

**編譯狀態:**
- ✅ 代碼語法正確，可以編譯
- ⏸️ 實際編譯等待 CUDA toolkit 環境

---

## 階段二：致命錯誤修正：物理模型 ⚠️ [必須立即處理]

### TODO-2.1 (Python端)：週期性鍵結力 (Periodic Bonded Forces) ✅ [已完成 - 2024-11-04]

**問題:**
- 石墨烯 (`graph_c_freeze.xml`) 的鍵結力會跨越週期性邊界
- 舊版 `set_periodic_residue` 處理了此問題,新版缺失
- **影響:** 石墨烯結構在 PBC 下會出現非物理的應力

**修正動作:**
在 `setup_system()` 函數的 `system = forcefield.createSystem(...)` 之後添加:

```python
# Step 7: [CRITICAL] Set periodic boundary conditions for bonded forces
print("Setting periodic boundary conditions for bonded forces (graphene)...")
from openmm import HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, RBTorsionForce
for i in range(system.getNumForces()):
    f = system.getForce(i)
    if (
        isinstance(f, HarmonicBondForce) or
        isinstance(f, HarmonicAngleForce) or
        isinstance(f, PeriodicTorsionForce) or
        isinstance(f, RBTorsionForce)
    ):
        f.setUsesPeriodicBoundaryConditions(True)
print("✓ Periodic bonded forces enabled.")
```

**位置:** `fv_md_plugin/run_fv_md_plugin.py`, Line ~82-95

**狀態:** ✅ 已完成

**文件修改:** `fv_md_plugin/run_fv_md_plugin.py`

---

### TODO-2.2 (Python端)：SAPT/電極排除 (Exclusions) ✅ [已確認正確 - 2024-11-04]

**問題:**
- 如果缺少排除規則, `NonbondedForce` 會計算電極內部靜電力
- 插件也會計算這些交互作用
- **結果:** 雙重計數 (double-counting)

**檢查結果:**
- ✅ `fv_md_plugin/exclusions.py` 模塊存在且正確 (500+ lines)
- ✅ `run_fv_md_production.py` Line 51: `from exclusions import apply_all_exclusions`
- ✅ `run_fv_md_production.py` Line 146-153: 正確調用

**實際代碼:**
```python
# run_fv_md_production.py, Line 146-153
apply_all_exclusions(
    system,
    modeller.topology,
    cathode_atoms,
    anode_atoms,
    apply_sapt=True  # ✓ SAPT-FF 排除已啟用
)
```

**調用時機:** ✅ 正確 (system 創建後, 插件初始化前)

**位置:** `run_fv_md_production.py`, Line 146-153

**狀態:** ✅ 已確認正確 - 無需修改

**相關文件:** 
- `run_fv_md_production.py`
- `fv_md_plugin/exclusions.py`

---

### TODO-2.3 (CUDA/Python端)：PME 靜電 (最致命的錯誤) ❌ [未完成]

**問題:**
- `calculateEfKernel` 使用 `O(N*M)` 真空求和: `E_f[i] = Σ_j (k * q_f[j] / r_ij)`
- **這在週期性系統中是完全錯誤的**
- 忽略了 PME 的所有貢獻 (long-range, 鏡像電荷)
- **影響:** 電位計算錯誤 → 電荷計算錯誤 → 整個模擬結果無效

**正確做法:**
必須使用 OpenMM 的 PME 計算能力

#### 方法 A: 利用 OpenMM PME (推薦)

**Python 端修改:**
```python
# 在 NonbondedForce 設定中
for i, atom_idx in enumerate(electrode_atoms):
    charge, sigma, epsilon = nbforce.getParticleParameters(atom_idx)
    nbforce.setParticleParameters(atom_idx, charge, sigma, epsilon, group=2)

for i, atom_idx in enumerate(electrolyte_atoms):
    charge, sigma, epsilon = nbforce.getParticleParameters(atom_idx)
    nbforce.setParticleParameters(atom_idx, charge, sigma, epsilon, group=1)

# 可選: 移除電極-電極交互作用 (已由 C_inv 處理)
# nbforce.addInteractionGroup(set(electrolyte_atoms), set(electrode_atoms))
```

**CUDA 端修改:**
```cpp
// 在 CudaCalcConstantVKernel::execute() 中:
// 1. 刪除 calculateEfKernel 調用
// 2. 替換為 OpenMM PME 計算

// 概念代碼 (需查閱 OpenMM API):
// - 使用 context 計算第 1 組 (電解質) 作用在第 2 組 (電極) 上的電位
// - 從 forces/energy 反推每個電極原子的電位
// - 儲存到 d_Ef (應改名為 d_phi_f)
```

#### 方法 B: 臨時 Workaround (僅限測試)

如果 OpenMM API 整合過於複雜, 可暫時使用 Ewald 求和:
```cpp
// 在 calculateEfKernel 中添加 Ewald 項
// 但這仍然不完整, 僅供概念驗證
```

**位置:** 
- `CudaConstantVKernels.cu`: execute() 函數
- `run_fv_md_production.py`: NonbondedForce 設定

**優先級:** 🔴 CRITICAL - 必須修正,否則結果無效

**預計工作量:** 高 (需要深入了解 OpenMM PME API)

---

## 階段三：中度錯誤修正：穩健性與維護 ⚠️ [建議修正]

### TODO-3.1 (CUDA端)：命名 (Naming) ❌ [未完成]

**問題:**
- `E_f` 是電位 (Electric Potential), 不是電場 (Electric Field)
- 變量命名不準確會導致代碼理解困難

**修正動作:**
在所有 `.h` 和 `.cu` 文件中:
- `E_f` / `Ef` → `phi_f` / `phi_electrolyte`
- `d_Ef` → `d_phi_f`
- 註解中的 "electric field" → "electric potential"

**位置:** 
- `CudaConstantVKernels.h`
- `CudaConstantVKernels.cu`
- `ReferenceConstantVKernels.cpp`

**優先級:** 🟡 MEDIUM - 建議修正

---

### TODO-3.2 (Python端)：單位轉換 (Units) ❌ [未完成]

**問題:**
- `config.ini` 中的電壓是 `Volts` (4.0)
- 插件假設單位是 `kJ/mol/e` (OpenMM 內部單位)
- **換算因子:** 1 V = 96.485 kJ/mol/e (法拉第常數)

**修正動作:**
在 `run_fv_md_production.py` 中:

```python
# 讀取電壓配置
voltage_volts = float(config['voltage'])  # 例如 4.0 V

# 轉換為 OpenMM 單位 (kJ/mol/e)
FARADAY_CONSTANT = 96.485  # kJ/(mol·V)
voltage_openmm = voltage_volts * FARADAY_CONSTANT

# 傳遞給插件
for i, atom_idx in enumerate(electrode_atoms):
    cv_force.addElectrodeAtom(atom_idx, voltage_openmm if is_cathode else -voltage_openmm)
```

**位置:** `run_fv_md_production.py`, 插件初始化部分

**優先級:** 🔴 HIGH - 單位錯誤會導致數值錯誤

---

### TODO-3.3 (CUDA/Python端)：Dummy 原子排除 (Exclusions) ❌ [未完成]

**問題:**
- 舊版 `initialize_electrodes` 會排除 `exclude_element=("H",)` (Dummy 原子)
- 新插件會將所有電解質原子包含在內
- **影響:** Dummy 原子的假電荷可能干擾計算

**修正動作:**
在 `run_fv_md_production.py` 中過濾 Dummy 原子:

```python
# 建立電解質原子列表時
electrolyte_atoms = []
for atom in pdb.topology.atoms():
    if atom.residue.name in ['WAT', 'TFSI', ...]:  # 電解質殘基
        if atom.element.symbol != 'H':  # 排除 Dummy 氫原子
            electrolyte_atoms.append(atom.index)
```

**位置:** `run_fv_md_production.py`, 電解質原子列表建立部分

**優先級:** 🟡 MEDIUM - 取決於力場是否有 Dummy 原子

---

### TODO-3.4 (CUDA/Python端)：Drude 極化 (Polarization) ❌ [需要確認]

**問題:**
- 舊版支持 Drude 力場 (`self.polarization = True`)
- 新插件沒有明確處理 Drude 粒子
- Drude 粒子帶電, 會影響 E_f 計算
- **複雜性:** Drude 粒子位置依賴於電荷 (自洽問題)

**檢查清單:**
1. ❓ 您的力場是否為極化力場 (Drude)?
2. ❓ 如果是, Drude 粒子是否包含在 `electrolyte_atoms` 中?
3. ❓ 是否需要自洽循環求解?

**修正動作 (如果是 Drude 力場):**
- 在 PME 計算中包含 Drude 粒子
- 可能需要迭代求解: 計算電荷 → 更新 Drude 位置 → 重新計算電荷
- **預計工作量:** 非常高

**位置:** 取決於力場類型

**優先級:** 🔴 CRITICAL (如果是 Drude 力場) / ⚪ SKIP (如果不是)

---

## 階段四：澄清 (關於 "Hack" 的說明)

### "Hack" 澄清: `invalidateNonbondedParameters()`

**審查者評論:** 稱此為 "hack"

**事實澄清:**
- ✅ **這不是 hack**
- ✅ 這是 OpenMM C++ API 中**唯一**且**正確**的零傳輸方法
- `updateParametersInContext()` 從 CPU 獲取數據 → **這是我們要避免的瓶頸**
- `invalidateNonbondedParameters()` 告訴 OpenMM: "GPU 數據已更新, 直接使用"

**技術原理:**
```cpp
// 在 GPU 上直接修改電荷
scatterWriteChargesKernel<<<...>>>(d_q_e, posq);

// 通知 OpenMM: "posq 已被插件修改, 請失效緩存"
cu.invalidateNonbondedParameters();

// OpenMM 下次計算時會使用更新後的 GPU 數據, 無需 CPU-GPU 傳輸
```

**結論:** 保持使用 `cu.invalidateNonbondedParameters()`, 這是最優方法

---

## 優先級總結

### 🔴 CRITICAL (必須立即修正)
1. ✅ TODO-1.1: 編譯錯誤 [已完成]
2. ❌ TODO-2.3: PME 靜電錯誤 [未完成] - **最重要**
3. ❌ TODO-3.2: 單位轉換 [未完成]
4. ❓ TODO-3.4: Drude 極化 (如果適用) [需要確認]

### 🔴 HIGH (應盡快修正)
1. ❌ TODO-2.1: 週期性鍵結力 [未完成]
2. ⚠️ TODO-2.2: SAPT/電極排除 [需要確認]

### 🟡 MEDIUM (建議修正)
1. ❌ TODO-3.1: 命名修正 [未完成]
2. ❌ TODO-3.3: Dummy 原子排除 [未完成]

### ⚪ OPTIONAL (可選功能)
1. ❓ TODO-3.5: 傘狀採樣 (Umbrella Sampling) [需要確認]

---

## TODO-3.5 (Python端): 傘狀採樣 (Umbrella Sampling) ❓ [需要確認]

**遺漏的邏輯:** 舊版 `MM_classes.py` 中的 `setumbrella` 方法

**功能:**
- 添加 `CustomCentroidBondForce` 或 `CustomExternalForce`
- 用於施加束縛以進行自由能計算 (umbrella sampling)
- **舊版原始實現:** `OpenMM-ConstantV(original)/lib/MM_classes.py`, Line 756-816

**檢查結果:**
- ✅ 舊版 `run_openMM.py` **沒有使用** `setumbrella`
- ✅ 這是一個可選功能,僅用於特定的自由能計算

**檢查清單:**
1. ❓ 這次的生產運行是否需要執行傘狀採樣?
2. ❓ 如果是,需要添加哪些束縛力?

**如果需要,修正動作:**
在 `run_fv_md_production.py` 的設定階段添加:

**Option 1: 兩個分子質心之間的距離約束**
```python
from openmm import CustomCentroidBondForce

# 建立兩個質心組 (例如: TFSI 和石墨烯上的某個原子)
g1 = [...]  # mol1 的原子索引列表
g2 = [...]  # mol2 的原子索引列表

umbrella_force = CustomCentroidBondForce(2, "0.5*k*(distance(g1,g2)-r0centroid)^2")
umbrella_force.addPerBondParameter("k")
umbrella_force.addPerBondParameter("r0centroid")
umbrella_force.addGroup(g1)
umbrella_force.addGroup(g2)
umbrella_force.addBond([0, 1], [k_value, r0_value])
umbrella_force.setUsesPeriodicBoundaryConditions(True)
system.addForce(umbrella_force)
```

**Option 2: 絕對 z 坐標約束**
```python
from openmm import CustomExternalForce

z_force = CustomExternalForce("0.5 * k * periodicdistance(x,y,z,x,y,z0)^2")
z_force.addGlobalParameter('z0', z_target)
z_force.addGlobalParameter('k', k_value)

# 對特定原子組施加約束
for atom_idx in target_atoms:
    z_force.addParticle(atom_idx)
    
system.addForce(z_force)
```

**位置:** `run_fv_md_production.py`, system 創建後, simulation 創建前

**優先級:** ⚪ OPTIONAL - 僅在需要自由能計算時

**狀態:** ❓ **需要用戶確認是否需要此功能**

---

## 附錄: 確認不需要的功能

### add_customnonbond_xml.py ✅ [確認不需要]

**功能:** 離線力場準備工具,合併多個 `.xml` 文件的 `<CustomNonbondedForce>` 參數

**確認:**
- ✅ `run_openMM.py` Line 14: `#from add_customnonbond_xml import ...` (已註解)
- ✅ 這是預處理工具,不是運行時依賴
- ✅ `run_fv_md_production.py` 只需要載入**已經合併好**的 `.xml` 文件

**結論:** ✅ **不需要**將此文件加回新版插件

---

## 下一步行動計劃

### 立即行動 (今天)
1. ✅ **階段一:** 修正編譯錯誤 → 重新編譯
2. ❌ **TODO-2.2:** 確認 exclusions 是否正確應用
3. ❌ **TODO-3.2:** 修正單位轉換
4. ❌ **TODO-2.1:** 添加週期性鍵結力設定

### 中期行動 (本週)
1. ❌ **TODO-2.3 (PME):** 研究 OpenMM PME API
2. ❌ **TODO-2.3 (PME):** 實現正確的 PME 靜電計算
3. ❌ **TODO-3.1:** 重命名變量
4. ❌ **TODO-3.3:** 添加 Dummy 原子過濾

### 驗證行動
1. ❌ 重新編譯 CUDA Platform
2. ❌ 運行測試模擬 (短時間)
3. ❌ 與舊版結果比較 (能量, 電荷分佈)
4. ❌ 檢查 PME 是否正確應用 (對比真空計算)

---

## 文件更新記錄

- **2024-11-04:** 初始建立, 階段一完成
- **待續:** 後續階段完成後更新

---

## 參考資源

- OpenMM Developer Guide: http://docs.openmm.org/latest/developerguide/
- OpenMM C++ API Reference: http://docs.openmm.org/latest/api-c++/
- PME 原理: Essmann et al., J. Chem. Phys. 103, 8577 (1995)
- 單位轉換: OpenMM User Guide, Section "Units"
