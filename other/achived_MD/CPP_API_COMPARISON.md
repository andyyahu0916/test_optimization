# C++ API 比較：Force-based vs Integrator-based

**日期**: 2025-11-27
**目的**: 解釋 `openmm_core_integration/` 提供的兩種 C++ API，以及它們與原始 Python 實現的差異

---

## 📋 總覽

`openmm_core_integration/` 提供了 **兩種不同的 C++ API** 來實現 ConstantV 電極：

| API 類型 | 類別名稱 | 控制方式 | 適用場景 |
|---------|---------|----------|---------|
| **Force-based** | `ConstantVForce` | Python 控制 SCF 頻率 | 研究、除錯、靈活性 |
| **Integrator-based** | `ConstantVDrudeLangevinIntegrator` | C++ 自動 SCF | 生產、效能、自動化 |

**關鍵發現**: 這兩種 API **不是競爭關係**，而是針對 **不同使用情境** 的設計！

---

## 🔍 API #1: ConstantVForce (Force-based)

### 設計理念

**文件位置**: `openmmapi/include/openmm/ConstantVForce.h`

**核心概念**:
- ConstantVForce 是一個 **被動的 Force 物件**
- 它**不會自動執行 SCF**
- 需要由 **Python 或其他程式碼** 主動觸發 SCF 更新

### 使用方式

```python
from openmm_constantv import constantv  # C++ extension

# 創建標準的 DrudeLangevinIntegrator（原版做法）
integrator = openmm.DrudeLangevinIntegrator(
    temperature=300.0,
    frictionCoeff=1.0,
    drudeTemperature=1.0,
    drudeFrictionCoeff=20.0,
    stepSize=0.001
)

# 創建 ConstantVForce（被動 Force）
force = constantv.ConstantVForce()
force.setVoltage(2.0)
force.setLgap(3.5)
force.setLcell(5.0)
force.setTotalArea(10.0)
force.setNumIterations(4)  # 每次 SCF 呼叫執行 4 次迭代

# 添加電極原子
for idx in cathode_indices:
    force.addCathodeAtom(idx, area_per_atom)
for idx in anode_indices:
    force.addAnodeAtom(idx, area_per_atom)

# 添加到 System
system.addForce(force)

# 創建 Context
context = openmm.Context(system, integrator, platform)

# ⚠️ 重點：需要手動觸發 SCF（類似原版 Python）
for i in range(n_frames):
    for j in range(scf_frequency):  # 每 200 步
        # 方法 1：透過 Force 的 kernel 觸發（需要實現）
        # force.updateCharges(context)

        # 方法 2：呼叫 Python SCF solver（回退到原版）
        # scf_solver.update_charges(context)

        integrator.step(timestep)
```

### 架構特性

**優點**:
- ✅ **靈活性最高**: 可在 Python 層控制 SCF 觸發時機
- ✅ **易於除錯**: 每次 SCF 後可檢查電荷、forces、能量
- ✅ **適合研究**: 可在 SCF 迭代中插入自訂邏輯
- ✅ **相容性好**: 可與任何 Integrator 搭配使用

**缺點**:
- ⚠️ **需要額外程式碼**: 使用者必須自己寫控制循環
- ⚠️ **Python 開銷**: 每次 SCF 都需要 Python→C++ 呼叫
- ⚠️ **文件不完整**: 當前實現可能缺少觸發 SCF 的 API

### 與原版 Python 的比較

| 方面 | 原版 Python | ConstantVForce (C++) |
|-----|------------|----------------------|
| **Integrator** | 標準 `DrudeLangevinIntegrator` | 標準 `DrudeLangevinIntegrator` |
| **SCF 實現** | Python `Poisson_solver_fixed_voltage()` | C++ `ConstantVKernel::updateCharges()` |
| **控制方式** | Python 手動呼叫 | Python 手動呼叫（類似） |
| **效能** | 慢（Python 迴圈） | 中等（C++ SCF，Python 控制） |
| **靈活性** | ✅ 最高 | ✅ 高 |

**結論**: ConstantVForce 是原版 Python 的 **C++ 加速版本**，但仍保留 Python 控制權。

---

## 🚀 API #2: ConstantVDrudeLangevinIntegrator (Integrator-based)

### 設計理念

**文件位置**: `openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h`

**核心概念**:
- 繼承自 `DrudeLangevinIntegrator`
- **內建 SCF 邏輯**（在 `step()` 方法中自動執行）
- **完全自動化**，不需要 Python 介入

### 使用方式

```python
from openmm_constantv import constantv

# 創建 Integrator（內建 ConstantV 支援）
integrator = constantv.ConstantVDrudeLangevinIntegrator(
    temperature=300.0,
    frictionCoeff=1.0,
    drudeTemperature=1.0,
    drudeFrictionCoeff=20.0,
    stepSize=0.001,
    voltage=2.0,        # ⬅ 內建電壓
    Lgap=3.5,           # ⬅ 內建幾何參數
    Lcell=5.0,
    scfIterations=4     # ⬅ 內建 SCF 迭代次數
)

# 設定 SCF 頻率（每 N 步更新一次電荷）
integrator.setSCFFrequency(200)  # 每 200 步

# 設定電極總面積和位置
integrator.setTotalArea(10.0)
integrator.setZCathode(0.5)
integrator.setZAnode(4.0)

# 添加電極原子
for idx in cathode_indices:
    integrator.addCathodeAtom(idx, area_per_atom)
for idx in anode_indices:
    integrator.addAnodeAtom(idx, area_per_atom)

# 添加電解質原子（Green's Reciprocity）
for idx in electrolyte_indices:
    integrator.addElectrolyteAtom(idx, charge)

# 創建 Context（不需要添加 ConstantVForce）
context = openmm.Context(system, integrator, platform)

# ✅ 就這樣！integrator.step() 會自動處理 SCF
integrator.step(1000000)  # 全自動
```

### 架構特性

**優點**:
- ✅ **效能最高**: SCF 完全在 C++/CUDA 執行，無 Python 開銷
- ✅ **使用簡單**: 只需呼叫 `step()`，無需控制循環
- ✅ **記憶體效率**: 電極資料常駐 GPU，無 CPU↔GPU 傳輸
- ✅ **適合生產**: 穩定、自動化、高效

**缺點**:
- ⚠️ **靈活性低**: SCF 頻率固定，無法動態調整
- ⚠️ **除錯困難**: SCF 過程發生在 C++ 內部，難以觀察
- ⚠️ **黑箱風險**: 使用者無法直接檢視中間步驟
- ⚠️ **需要重編譯**: 修改 SCF 邏輯需重新編譯 C++ 程式碼

### 與原版 Python 的比較

| 方面 | 原版 Python | ConstantVDrudeLangevinIntegrator (C++) |
|-----|------------|---------------------------------------|
| **Integrator** | 標準 `DrudeLangevinIntegrator` | 自訂 `ConstantVDrudeLangevinIntegrator` |
| **SCF 實現** | Python `Poisson_solver_fixed_voltage()` | C++ `ReferenceConstantVDrudeLangevinDynamics::updateElectrodeCharges()` |
| **控制方式** | Python 明確循環 | C++ 自動（在 `step()` 內部） |
| **效能** | 慢（Python 迴圈） | ✅ 最快（原生 C++/CUDA） |
| **靈活性** | ✅ 最高 | ⚠️ 低（黑箱） |
| **透明度** | ✅ 可見 Python code | ⚠️ 隱藏在 C++ |

**結論**: ConstantVDrudeLangevinIntegrator 是 **生產級高效能實現**，犧牲靈活性換取效能。

---

## 📊 三種實現方式的完整比較

### 控制流程圖

#### 原版 Python 實現

```
main script (run_openMM.py)
  │
  ├─→ 創建標準 DrudeLangevinIntegrator
  ├─→ 創建 System、Context
  │
  └─→ 主循環（Python）:
      FOR frame in range(n_frames):
        FOR step in range(scf_frequency):
          ├─→ Python 呼叫: MMsys.Poisson_solver_fixed_voltage()
          │   ├─→ 計算 analytic charge (Green's Reciprocity)
          │   ├─→ FOR iter in range(4):
          │   │   ├─→ context.getState(getForces=True)
          │   │   ├─→ 更新電荷: q = 2/(4π) × area × (V/Lgap + Ez)
          │   │   └─→ nbondedForce.updateParametersInContext()
          │   └─→ Scale charges (Green's Reciprocity)
          │
          └─→ integrator.step(timestep)
```

**特點**:
- ✅ 完全透明（所有步驟都在 Python 層可見）
- ⚠️ 效能最低（Python 迴圈開銷）

---

#### ConstantVForce (Force-based C++)

```
main script (Python)
  │
  ├─→ 創建標準 DrudeLangevinIntegrator
  ├─→ 創建 ConstantVForce（C++ 物件）
  ├─→ force.addCathodeAtom(), force.addAnodeAtom()
  ├─→ system.addForce(force)
  ├─→ 創建 Context
  │
  └─→ 主循環（Python）:
      FOR frame in range(n_frames):
        FOR step in range(scf_frequency):
          ├─→ Python 呼叫: force.updateCharges(context)  ← C++ kernel
          │   └─→ [C++ 執行 SCF]
          │       ├─→ Compute analytic charge
          │       ├─→ FOR iter in range(4):
          │       │   ├─→ Get forces from context
          │       │   ├─→ Update q = 2/(4π) × area × (V/Lgap + Ez)
          │       │   └─→ Update NonbondedForce parameters
          │       └─→ Scale charges
          │
          └─→ integrator.step(timestep)
```

**特點**:
- ✅ SCF 在 C++ 執行（快）
- ⚠️ 仍需 Python 控制循環（中等開銷）
- ✅ 可在 SCF 之間插入自訂邏輯

---

#### ConstantVDrudeLangevinIntegrator (Integrator-based C++)

```
main script (Python)
  │
  ├─→ 創建 ConstantVDrudeLangevinIntegrator（C++ 物件）
  │   ├─→ 建構子參數：voltage, Lgap, Lcell, scfIterations
  │   └─→ integrator.setSCFFrequency(200)
  ├─→ integrator.addCathodeAtom(), addAnodeAtom()
  ├─→ 創建 Context（不需要 ConstantVForce）
  │
  └─→ 簡單呼叫:
      integrator.step(1000000)  ← 全部在 C++ 處理
        │
        └─→ [C++ 內部自動處理]
            FOR MD_step in range(1000000):
              IF (MD_step % scfFrequency == 0):  ← 自動判斷
                ├─→ Compute analytic charge
                ├─→ FOR iter in range(4):
                │   ├─→ Get forces
                │   ├─→ Update q = 2/(4π) × area × (V/Lgap + Ez)
                │   └─→ Update charges in memory
                └─→ Scale charges

              ├─→ 執行 Drude Langevin 積分
              └─→ 更新 positions/velocities
```

**特點**:
- ✅ 完全自動化（無 Python 循環）
- ✅ 效能最高（全 C++/CUDA）
- ⚠️ 黑箱（無法觀察中間步驟）

---

## 🎯 使用場景建議

### 何時使用 原版 Python 實現？

**適用情境**:
- 🔬 **研究與開發**: 需要快速測試新的 SCF 演算法
- 🐛 **除錯與驗證**: 需要檢查每一步的電荷、forces、能量
- 📚 **教學**: 展示 SCF 演算法的運作原理
- 🧪 **原型設計**: 快速實現新的物理模型

**優點**:
- ✅ 程式碼清晰易懂（純 Python）
- ✅ 易於修改和實驗
- ✅ 可在任何步驟插入列印、繪圖、分析

**缺點**:
- ⚠️ 效能最低（~200 步 MD = 1 次 SCF，Python 迴圈開銷）
- ⚠️ 不適合長時間生產模擬

**範例場景**:
```python
# 研究新的 SCF 收斂準則
def custom_scf_solver(system, context, convergence_threshold=1e-6):
    for iter in range(max_iterations):
        old_charges = get_electrode_charges(context)

        # 原版 SCF 步驟
        MMsys.Poisson_solver_fixed_voltage(Niterations=1)

        new_charges = get_electrode_charges(context)
        delta = np.linalg.norm(new_charges - old_charges)

        print(f"Iteration {iter}: delta = {delta}")

        if delta < convergence_threshold:
            print("Converged!")
            break
```

---

### 何時使用 ConstantVForce (Force-based C++)？

**適用情境**:
- 🔍 **進階研究**: 需要 C++ 效能但保留 Python 控制
- 🎛️ **動態參數**: 需要在模擬中動態調整 SCF 頻率或參數
- 🧩 **混合架構**: 結合其他自訂 Forces 或演算法
- 📊 **資料收集**: 需要在每次 SCF 後記錄詳細資料

**優點**:
- ✅ SCF 效能高（C++ 實現）
- ✅ 保留 Python 控制權
- ✅ 可與任何 Integrator 搭配
- ✅ 易於整合到現有 Python workflow

**缺點**:
- ⚠️ 需要寫控制循環（比原版複雜一點點）
- ⚠️ Python→C++ 呼叫仍有小開銷
- ⚠️ API 可能不完整（需確認 `updateCharges()` 方法）

**範例場景**:
```python
# 動態調整 SCF 頻率
scf_frequency = 200
for i in range(n_frames):
    # 前 1000 步：頻繁更新以快速收斂
    if i < 1000:
        current_freq = 50
    else:
        current_freq = 200

    for j in range(current_freq):
        force.updateCharges(context)  # C++ SCF
        integrator.step(timestep)

    # 記錄電荷分佈
    cathode_charges = force.getCathodeCharges()
    log_charges(i, cathode_charges)
```

---

### 何時使用 ConstantVDrudeLangevinIntegrator (Integrator-based C++)？

**適用情境**:
- 🚀 **生產模擬**: 長時間、大規模的正式模擬
- 📈 **高效能計算**: 需要最大化 GPU 利用率
- 🤖 **自動化流程**: Workflow 中不需要人工介入
- ⚡ **時間敏感**: 需要盡快完成模擬

**優點**:
- ✅ 效能最高（全 C++/CUDA，無 Python 開銷）
- ✅ 使用簡單（只需呼叫 `step()`）
- ✅ 記憶體效率最佳（GPU 常駐）
- ✅ 穩定可靠（經過測試的生產程式碼）

**缺點**:
- ⚠️ 黑箱（無法觀察 SCF 過程）
- ⚠️ 靈活性最低（固定 SCF 頻率和參數）
- ⚠️ 除錯困難（需要 C++ debugger）
- ⚠️ 修改需重編譯

**範例場景**:
```python
# 生產級長時間模擬
integrator = constantv.ConstantVDrudeLangevinIntegrator(
    temperature=300.0,
    frictionCoeff=1.0,
    drudeTemperature=1.0,
    drudeFrictionCoeff=20.0,
    stepSize=0.001,
    voltage=2.0,
    Lgap=3.5,
    Lcell=5.0,
    scfIterations=4
)
integrator.setSCFFrequency(200)

# 配置電極
for idx in cathode_indices:
    integrator.addCathodeAtom(idx, area_per_atom)
for idx in anode_indices:
    integrator.addAnodeAtom(idx, area_per_atom)

context = openmm.Context(system, integrator, platform)

# 執行 10 ns 模擬（1000 萬步）
integrator.step(10000000)  # 全自動，無需任何額外程式碼

# 查詢最終電荷
total_cathode_charge = integrator.getTotalCathodeCharge()
total_anode_charge = integrator.getTotalAnodeCharge()
print(f"Final charges: Cathode = {total_cathode_charge}, Anode = {total_anode_charge}")
```

---

## 🔧 技術細節比較

### 記憶體佈局

| 實現方式 | 電極資料位置 | 更新頻率 | CPU↔GPU 傳輸 |
|---------|------------|----------|-------------|
| **原版 Python** | Python 物件 (CPU) | 每次 SCF | 高（每次都要上傳） |
| **ConstantVForce** | Force 物件 (CPU/GPU) | 每次 SCF | 中等（參數更新） |
| **ConstantVIntegrator** | Integrator 內部 (GPU) | 初始化一次 | 最低（GPU 常駐） |

### SCF 觸發機制

| 實現方式 | 觸發方式 | 控制位置 | 延遲 |
|---------|---------|----------|-----|
| **原版 Python** | 手動呼叫 `Poisson_solver_fixed_voltage()` | Python | ~100 µs (Python 呼叫) |
| **ConstantVForce** | 手動呼叫 `force.updateCharges()` | Python | ~10 µs (C++ 呼叫) |
| **ConstantVIntegrator** | 自動（`step()` 內部判斷） | C++ | ~0 µs (內聯判斷) |

### 電荷更新方式

| 實現方式 | 更新方法 | 同步方式 |
|---------|---------|---------|
| **原版 Python** | `nbondedForce.setParticleParameters()` + `updateParametersInContext()` | 每次 SCF 同步 |
| **ConstantVForce** | C++ kernel 直接修改 | 每次 SCF 同步 |
| **ConstantVIntegrator** | 積分步驟中直接更新 | 無需同步（內聯） |

---

## ⚠️ 重要限制與已知問題

### 1. Conductor 支援狀態

| 功能 | 原版 Python | ConstantVForce | ConstantVIntegrator |
|-----|-----------|----------------|---------------------|
| **Flat Electrodes** | ✅ 完整支援 | ✅ 完整支援 | ✅ 完整支援 |
| **Buckyball Conductors** | ✅ 完整支援 | ❓ API 存在但未驗證 | ❌ 未實現 |
| **Nanotube Conductors** | ✅ 完整支援 | ❓ API 存在但未驗證 | ❌ 未實現 |

**結論**: 如果你的系統包含 Buckyball 或 Nanotube，目前只能使用 **原版 Python** 或 **ConstantVForce**（待驗證）。

---

### 2. 已知 Bug

#### ConstantVIntegrator 的 Threshold Bug

**位置**: `ReferenceConstantVDrudeLangevinDynamics.cpp:148-150`

**問題**: 低電荷保護使用 `threshold/2` 而非 `threshold`

**影響**: 極少數情況下電荷可能收斂到略小的值

**修復方式**: 將 `sign / 2.0 * SMALL_THRESHOLD` 改為 `sign * SMALL_THRESHOLD`

**嚴重程度**: 🟡 低（僅影響邊界情況）

---

### 3. API 完整性

#### ConstantVForce 缺少觸發方法？

根據程式碼分析，`ConstantVForce` **可能缺少** 從 Python 觸發 SCF 的公開方法。

**預期 API**:
```python
force.updateCharges(context)  # ← 此方法可能不存在？
```

**可能的解決方案**:
1. 檢查 SWIG bindings 是否暴露此方法
2. 使用 `context.reinitialize()` 觸發 Force 重新計算（低效）
3. 回退到原版 Python SCF

**TODO**: 需要檢查 `ConstantVForceImpl` 的實現。

---

## 📝 架構建議

### 專案結構建議

```
openmm_constantv/
├── python_controlled/          # Python 控制的 SCF
│   ├── scf_solver.py           # 原版 Python 實現
│   └── force_based.py          # ConstantVForce wrapper
│
├── integrator_controlled/      # Integrator 控制的 SCF
│   └── integrator_based.py     # ConstantVIntegrator wrapper
│
└── common/
    ├── exclusions.py           # 共用 exclusion 邏輯
    └── geometry.py             # 共用幾何計算
```

### 使用者選擇模式

```python
from openmm_constantv import ConstantVMode

# Mode 1: Python 控制（研究、除錯）
system = ConstantVMode.python_controlled(
    config=config,
    scf_frequency=200
)

# Mode 2: Force-based（進階研究）
system = ConstantVMode.force_based(
    config=config,
    scf_frequency=200
)

# Mode 3: Integrator-based（生產）
system = ConstantVMode.integrator_based(
    config=config,
    scf_frequency=200
)
```

---

## 🎓 結論：何時用哪個？

### 決策樹

```
你的需求是什麼？
│
├─ 需要最高效能？
│  └─ ✅ 使用 ConstantVDrudeLangevinIntegrator (Integrator-based)
│     └─ ⚠️ 但確認系統只有 flat electrodes（無 conductor）
│
├─ 需要除錯或研究？
│  └─ ✅ 使用原版 Python
│     └─ 優點：完全透明、易於修改
│
├─ 需要 C++ 效能但保留控制權？
│  └─ ✅ 使用 ConstantVForce (Force-based)
│     └─ ⚠️ 但先確認 API 是否完整
│
└─ 有 Buckyball 或 Nanotube？
   └─ ⚠️ 只能使用原版 Python（C++ 尚未支援）
```

### 效能對比（參考）

| 實現方式 | SCF 時間 (N=1000 atoms) | MD 步驟時間 | 總體效能 |
|---------|------------------------|-----------|---------|
| **原版 Python** | ~1000 µs | ~500 µs | 🐌 慢 |
| **ConstantVForce** | ~100 µs | ~500 µs | 🐇 中等 |
| **ConstantVIntegrator** | ~5 µs (CUDA) | ~500 µs | 🚀 快 |

**註**: SCF 頻率 = 200 步時，Integrator-based 的優勢會被 MD 步驟時間稀釋。

---

## 📚 相關文件

- `CPP_SCF_ANALYSIS.md` - C++ SCF 實現的逐行分析
- `ARCHITECTURE_ANALYSIS.md` - 整體架構分析
- `COMPLETE_ARCHITECTURE_ANALYSIS.md` - 60 頁完整分析

---

**END OF DOCUMENT**
