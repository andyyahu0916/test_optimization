# 學術同行審查修正 - 進度報告

## 日期: 2024年11月4日

## 審查反饋嚴重性
- **來源:** 優秀的學術同行審查
- **關鍵發現:** 包含**致命的物理錯誤 (PBC/PME)**
- **結論:** 當前插件版本**不能用於生產**,必須立即修正

---

## 修正進度總結

### ✅ 階段一: 編譯錯誤修正 [已完成]

**問題:** 3個 C++ API 使用錯誤導致編譯失敗

**修正內容:**
1. ✅ **`getInverseCapacitanceMatrix()` 簽名錯誤**
   - 修正前: `force.getInverseCapacitanceMatrix(invCapMatrix)` (pass-by-reference)
   - 修正後: `vector<double> invCapMatrix = force.getInverseCapacitanceMatrix()` (return-by-value)

2. ✅ **Stream 句柄錯誤**
   - 修正前: `cu.getCudaStream()` (方法不存在)
   - 修正後: `cu.getStream().getStream()` (正確獲取 cudaStream_t)

3. ✅ **cuBLAS API 版本**
   - 修正前: `cublasDaxpy`, `cublasDgemv`
   - 修正後: `cublasDaxpy_v2`, `cublasDgemv_v2`

4. ✅ **其他修正:**
   - `getDevicePointer()` → `getDeviceData()`
   - `invalidateMolecules()` → `invalidateNonbondedParameters()`
   - 添加警告註解: "[!!--- 警告：此核心物理上是錯誤的 ---!!]"

**狀態:** ✅ **已完成** - 代碼現在可以編譯

---

### ⚠️ 階段二: 致命物理錯誤修正 [部分完成]

#### TODO-2.1: 週期性鍵結力 ✅ [已完成]

**問題:** 
- 石墨烯電極的鍵結力跨越週期性邊界
- 未設定 `setUsesPeriodicBoundaryConditions(True)` 會導致非物理應力

**修正內容:**
```python
# 在 setup_system() 函數中, system.createSystem() 之後添加:
for i in range(system.getNumForces()):
    f = system.getForce(i)
    if (isinstance(f, HarmonicBondForce) or 
        isinstance(f, HarmonicAngleForce) or
        isinstance(f, PeriodicTorsionForce) or 
        isinstance(f, RBTorsionForce)):
        f.setUsesPeriodicBoundaryConditions(True)
```

**位置:** `fv_md_plugin/run_fv_md_plugin.py`, Line ~82-95

**狀態:** ✅ **已完成**

---

#### TODO-2.2: SAPT/電極排除 ✅ [已確認正確]

**問題:** 
- 缺少排除規則會導致雙重計數 (double-counting)
- NonbondedForce 和 ConstantVPlugin 會同時計算電極內部交互作用

**檢查結果:**
```python
# run_fv_md_production.py, Line 51:
from exclusions import apply_all_exclusions

# Line 146-153:
apply_all_exclusions(
    system,
    modeller.topology,
    cathode_atoms,
    anode_atoms,
    apply_sapt=True  # SAPT-FF 排除已啟用
)
```

**狀態:** ✅ **已確認正確** - 排除規則已正確應用

---

#### TODO-2.3: PME 靜電計算 ❌ [未完成 - 最高優先級]

**問題:** 
- `calculateEfKernel` 使用 `O(N*M)` **真空求和**
- 完全忽略 PME 的長程貢獻和鏡像電荷
- **這是最致命的錯誤** - 週期性系統中電位計算完全錯誤

**當前錯誤代碼:**
```cpp
// calculateEfKernel in CudaConstantVKernels.cu
// [!!--- 警告：此核心物理上是錯誤的 ---!!]
for (int j = 0; j < M; j++) {
    float dx = electrodePos.x - electrolytePos.x;  // ❌ 未考慮 PBC
    float r = sqrt(dx*dx + dy*dy + dz*dz);         // ❌ 未考慮鏡像
    sum_Ef += COULOMB_CONSTANT * q_j / r;          // ❌ 只有實空間項
}
// 缺少: PME 倒空間項, Ewald 修正, 自能項
```

**需要的正確實現:**

**方法 A: 利用 OpenMM PME (推薦,但複雜)**
```python
# Python 端: 設定 interaction groups
for atom_idx in electrode_atoms:
    nbforce.setParticleParameters(atom_idx, charge, sigma, epsilon, group=2)
for atom_idx in electrolyte_atoms:
    nbforce.setParticleParameters(atom_idx, charge, sigma, epsilon, group=1)
```

```cpp
// CUDA 端: 利用 OpenMM 的 PME 計算
// 1. 刪除 calculateEfKernel
// 2. 調用 OpenMM API 計算第 1 組對第 2 組的電位
// 3. 提取每個電極原子的電位到 d_phi_f
// (需要研究 OpenMM C++ API)
```

**方法 B: 實現完整 Ewald 求和 (非常複雜)**
- 實空間項 + 倒空間項 + 自能修正
- 需要 FFT, 複雜的數學
- 不推薦 (重複造輪子)

**狀態:** ❌ **未完成** - 需要深入研究 OpenMM PME API

**預計工作量:** 高 (1-2 週)

**臨時 Workaround:** 
- 可以用當前代碼進行**真空系統**測試 (無 PBC)
- 週期性系統的結果**完全無效**

---

### ✅ 階段三: 中度錯誤修正 [部分完成]

#### TODO-3.1: 變量命名 ❌ [未完成]

**問題:** `E_f` 是電位 (Potential), 不是電場 (Field)

**需要修正:**
- `E_f` → `phi_f` (electric potential from fixed charges)
- `d_Ef` → `d_phi_f`
- 所有註解中的 "electric field" → "electric potential"

**狀態:** ❌ **未完成** (低優先級)

---

#### TODO-3.2: 單位轉換 ✅ [已確認正確]

**問題:** 電壓單位從 Volts 轉換為 kJ/mol/e

**檢查結果:**
```python
# fv_md_plugin/run_fv_md_plugin.py, Line 38:
CONVERSION_V_TO_KJMOL = 96.485  # 1 V = 96.485 kJ/mol ✓ 正確

# run_fv_md_production.py, Line 166:
voltage_kjmol = voltage * CONVERSION_V_TO_KJMOL  # ✓ 正確應用
```

**狀態:** ✅ **已確認正確**

---

#### TODO-3.3: Dummy 原子排除 ❓ [需要確認力場]

**問題:** 是否有 Dummy 原子 (如 H) 需要排除?

**檢查動作:**
```bash
# 檢查 PDB 中是否有 Dummy 原子
grep "ATOM.*H" your_pdb_file.pdb | head -20
```

**如果有 Dummy 原子,需要修正:**
```python
# 在建立 electrolyte_atoms 列表時過濾
for atom in topology.atoms():
    if atom.residue.name in ['WAT', 'TFSI']:
        if atom.element.symbol != 'H':  # 排除 Dummy
            electrolyte_atoms.append(atom.index)
```

**狀態:** ❓ **需要用戶確認力場是否有 Dummy 原子**

---

#### TODO-3.4: Drude 極化 ❓ [需要確認力場]

**問題:** 是否為 Drude 極化力場?

**檢查動作:**
```bash
# 檢查力場文件中是否定義了 Drude 粒子
grep -i "drude" ffdir/*.xml
```

**如果是 Drude 力場:**
- ⚠️ **當前插件不能用**
- Drude 粒子位置依賴於電荷 (需要自洽循環)
- 需要重大修改

**狀態:** ❓ **需要用戶確認力場類型**

---

## 優先級排序

### 🔴 CRITICAL (阻塞生產)
1. ❌ **TODO-2.3: PME 靜電** - 最致命,必須修正
2. ❓ **TODO-3.4: Drude 確認** - 如果是 Drude 力場,需要重大修改

### 🟢 COMPLETED (已完成)
1. ✅ TODO-1.1: 編譯錯誤
2. ✅ TODO-2.1: 週期性鍵結力
3. ✅ TODO-2.2: SAPT/電極排除 (已確認)
4. ✅ TODO-3.2: 單位轉換 (已確認)

### 🟡 MEDIUM (建議修正)
1. ❌ TODO-3.1: 變量命名
2. ❓ TODO-3.3: Dummy 原子 (取決於力場)

---

## 下一步行動

### 立即行動 (今天)
1. ✅ ~~修正編譯錯誤~~
2. ✅ ~~添加週期性鍵結力設定~~
3. ✅ ~~確認排除規則~~
4. ✅ ~~確認單位轉換~~
5. ❓ **用戶確認:** 力場是否為 Drude? 是否有 Dummy 原子?
6. ❌ **開始研究:** OpenMM PME API (TODO-2.3)

### 中期行動 (本週)
1. ❌ **實現:** 正確的 PME 靜電計算 (TODO-2.3)
2. ❌ **測試:** 真空系統驗證 (無 PBC, 可用當前代碼)
3. ❌ **測試:** 週期性系統驗證 (PME 修正後)

### 長期行動 (下週)
1. ❌ 變量重命名 (TODO-3.1)
2. ❌ 與原始版本結果對比
3. ❌ 性能基準測試

---

## 文件更新

### 修改的文件
1. ✅ `ConstantVPlugin/platforms/cuda/src/CudaConstantVKernels.cu` - 修正編譯錯誤
2. ✅ `fv_md_plugin/run_fv_md_plugin.py` - 添加週期性鍵結力

### 新建的文件
1. ✅ `TODO_SOP_COMPLETE.md` - 完整修正清單
2. ✅ `ACADEMIC_REVIEW_STATUS.md` (本文件) - 進度報告

---

## 技術要點

### "Hack" 澄清
- ✅ `invalidateNonbondedParameters()` **不是 hack**
- ✅ 這是 OpenMM 零傳輸的**正確且唯一**方法
- ✅ `updateParametersInContext()` 才是我們要避免的 CPU-GPU 瓶頸

### PME 重要性
- PME (Particle Mesh Ewald) 是週期性系統的標準靜電處理方法
- **缺少 PME:** 電位計算錯誤 → 電荷分佈錯誤 → 整個模擬無效
- **無法繞過:** 這是週期性系統的基礎物理要求

---

## 總結

### 可用性狀態
- ❌ **週期性系統 (PBC):** 不可用 - PME 錯誤導致結果無效
- ✅ **真空系統 (無 PBC):** 可用 - 可用於概念驗證測試
- ⏳ **完整生產:** 等待 TODO-2.3 (PME) 修正

### 代碼質量
- **編譯:** ✅ 通過
- **物理正確性:** ❌ 關鍵錯誤 (PME)
- **代碼風格:** ✅ 良好
- **文檔:** ✅ 完整

### 審查者反饋採納
- ✅ 所有高優先級反饋已審視
- ✅ 3/4 關鍵問題已解決
- ❌ 1/4 關鍵問題待解決 (PME)
- ✅ 中度問題部分解決

---

**最後更新:** 2024年11月4日  
**下次更新:** PME 修正完成後
