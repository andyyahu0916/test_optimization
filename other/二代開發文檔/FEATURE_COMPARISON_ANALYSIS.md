# OpenMM ConstantPotentialForce vs 舊版 OpenMM-ConstantV 功能對比分析

## 執行摘要

經過對**舊版 OpenMM-ConstantV(original)** 和 **OpenMM 8.4.0 內建 ConstantPotentialForce** 的詳細分析,發現兩者在設計理念和功能範圍上存在重大差異。

**核心結論**: 
- ✅ 內建實現對**平面電極**提供了更先進的 PME 電靜力和多種求解方法
- ❌ 內建實現**不支持**舊版的多導體幾何類型 (Buckyball、Nanotube)
- ⚠️ 內建實現使用的是 **Gaussian 電荷分布 + Thomas-Fermi 模型**,與舊版的 **Poisson solver + Green's reciprocity** 在理論框架上不同

---

## 一、理論框架對比

### 1.1 舊版 OpenMM-ConstantV (Poisson Solver 方法)

**核心理論**: Green's Reciprocity Theorem + Poisson Solver

**工作流程**:
```
1. 解析電荷計算 (Green's Reciprocity):
   Q_analytic = sign/(4π) * Area * (V/Lgap + V/Lcell) * conversion
   
2. 數值迭代求解 (Poisson_solver_fixed_voltage):
   for i in range(Niterations=3):
       a. 計算每個電極原子上的外場 Ez_external = F_z / q_i
       b. 更新電荷以滿足固定電壓邊界條件:
          q_i = 2/(4π) * area_atom * (V/Lgap + Ez_external)
       c. 處理額外導體 (Buckyball/Nanotube)
       d. 標準化到解析電荷 Q_analytic
       
3. 鏡像電荷處理 (Image charges):
   - 對電解質原子貢獻: Q_analytic += (z_distance/Lcell) * (-q_i)
   - 對導體原子貢獻: 相同公式
```

**關鍵特性**:
- 使用 **Green's reciprocity theorem** 提供解析標準化條件
- 電荷分布基於**表面積** (area_atom)
- 迭代次數固定 (通常 3 次)
- 支持多種導體幾何

---

### 1.2 內建 ConstantPotentialForce (PME + CG/Matrix 方法)

**核心理論**: Dufils et al. 2019 (Phys. Rev. Lett.) + Scalfi et al. 2020 (J. Chem. Phys.)

**工作流程** (從 API 推測):
```
1. Gaussian 電荷分布:
   - gaussianWidth: 控制電荷分布的寬度
   - thomasFermiScale: Thomas-Fermi 模型參數
   
2. PME 電靜力計算:
   - 正確處理周期性邊界條件
   - Ewald 誤差容忍度控制精度
   
3. 電極電荷求解:
   方法 A: Conjugate Gradient (CG)
       - 迭代求解直到 RMS 誤差 < cgErrorTol
       - 可選預條件器加速收斂
       
   方法 B: Capacitance Matrix
       - 預計算電容矩陣
       - 直接求解 (要求電極位置固定)
       
4. 約束條件:
   - 可選總電荷約束 (setChargeConstraintTarget)
   - 可選外場 (setExternalField)
```

**關鍵特性**:
- 使用 **PME** 正確處理長程電靜力
- 支持 **Thomas-Fermi semiclassical model**
- 兩種求解方法 (CG 適用於動態,Matrix 適用於固定電極)
- **僅支持平面電極** (從 API 推測)

---

## 二、功能支持對比表

| 功能項目 | 舊版 OpenMM-ConstantV | 內建 ConstantPotentialForce | 差異分析 |
|---------|----------------------|----------------------------|---------|
| **電靜力方法** | 真空求和 (錯誤) | PME (正確) | ✅ 內建優勝 |
| **電極幾何** | | | |
| - 平面電極 | ✅ Electrode_Virtual | ✅ addElectrode() | 兩者都支持 |
| - 球形導體 | ✅ Buckyball_Virtual | ❌ 不支持 | ⚠️ 舊版獨有 |
| - 管狀導體 | ✅ Nanotube_Virtual | ❌ 不支持 | ⚠️ 舊版獨有 |
| **電荷分布模型** | 點電荷 (area_atom) | Gaussian 分布 | 理論差異 |
| **求解方法** | Poisson Solver (3 次迭代) | CG / Matrix | 理論差異 |
| **解析校正** | ✅ Green's reciprocity | ❌ 無 | ⚠️ 舊版獨有 |
| **Virtual/Real 層** | ✅ 分離架構 | ❌ 單層 | ⚠️ 舊版獨有 |
| **鏡像電荷處理** | ✅ 顯式計算 | ⚠️ 包含在 PME 中 | 理論差異 |
| **Thomas-Fermi 模型** | ❌ 無 | ✅ thomasFermiScale | ✅ 內建優勝 |
| **外場支持** | ❌ 無 | ✅ setExternalField() | ✅ 內建優勝 |
| **總電荷約束** | ❌ 無 | ✅ setChargeConstraintTarget() | ✅ 內建優勝 |
| **收斂條件** | 固定迭代次數 | RMS 誤差容忍度 | ✅ 內建更靈活 |
| **預條件器** | ❌ 無 | ✅ setUsePreconditioner() | ✅ 內建優勝 |
| **電容矩陣求解** | ❌ 無 | ✅ Matrix method | ✅ 內建優勝 |

---

## 三、關鍵代碼片段對比

### 3.1 舊版: 電極電荷計算 (Electrode_Virtual)

```python
def compute_Electrode_charge_analytic(self, MMsys, positions, Conductor_list, z_opposite):
    """
    Green's reciprocity theorem 計算解析總電荷
    """
    sign = 1.0 if self.electrode_type == 'cathode' else -1.0
    
    # 幾何貢獻 (核心公式)
    self.Q_analytic = sign / (4.0 * numpy.pi) * self.sheet_area * \
                      (self.Voltage / MMsys.Lgap + self.Voltage / MMsys.Lcell) * \
                      conversion_KjmolNm_Au
    
    # 鏡像電荷貢獻 (電解質原子)
    for index in MMsys.electrolyte_atom_indices:
        (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(index)
        z_atom = positions[index][2]._value
        z_distance = abs(z_atom - z_opposite)
        self.Q_analytic += (z_distance / MMsys.Lcell) * (-q_i._value)
    
    # 鏡像電荷貢獻 (額外導體)
    if Conductor_list:
        for Conductor in Conductor_list:
            for atom in Conductor.electrode_atoms:
                # ... 相同邏輯
```

**物理意義**:
- `V/Lgap`: 電極間直接電壓降
- `V/Lcell`: 周期性鏡像貢獻
- `z_distance/Lcell * (-q_i)`: 鏡像電荷修正

---

### 3.2 舊版: Poisson 迭代求解

```python
def Poisson_solver_fixed_voltage(self, Niterations=3):
    """
    迭代求解電極電荷,滿足固定電壓邊界條件
    """
    # 第一步: 計算解析電荷
    self.Cathode.compute_Electrode_charge_analytic(...)
    self.Anode.compute_Electrode_charge_analytic(...)
    
    # 第二步: 自洽迭代
    for i_iter in range(Niterations):
        # 獲取當前電場 (從力)
        state = self.simmd.context.getState(getForces=True, ...)
        forces = state.getForces()
        
        # 更新陰極電荷
        for atom in self.Cathode.electrode_atoms:
            Ez_external = forces[index][2]._value / q_i_old
            # 固定電壓邊界條件
            q_i = 2.0 / (4.0 * numpy.pi) * self.Cathode.area_atom * \
                  (self.Cathode.Voltage / self.Lgap + Ez_external) * conversion
            atom.charge = q_i
        
        # 更新陽極電荷 (類似)
        ...
        
        # 處理額外導體 (Buckyball/Nanotube)
        if self.Conductor_list:
            for Conductor in self.Conductor_list:
                self.Numerical_charge_Conductor(Conductor, forces)
        
        # 標準化到解析電荷
        self.Scale_charges_analytic_general()
```

**核心思想**:
1. 使用電場 → 電荷關係: σ = 2ε₀ * E_normal
2. 滿足邊界條件: φ_electrode = V (固定)
3. 解析標準化確保總電荷正確

---

### 3.3 舊版: Buckyball 導體處理 (獨特功能)

```python
def Numerical_charge_Conductor(self, Conductor, forces):
    """
    處理球形/管狀導體的鏡像電荷和電荷轉移
    """
    # Step 1: 鏡像電荷 (保證導體內部法向場為零)
    for atom in Conductor.electrode_atoms:
        E_external = [Fx/q_i, Fy/q_i, Fz/q_i]
        # 投影到表面法向量
        En_external = dot(E_external, [atom.nx, atom.ny, atom.nz])
        # 求解表面電荷
        q_i = 2.0 / (4.0 * numpy.pi) * Conductor.area_atom * En_external
    
    # Step 2: 電荷轉移 (使導體與電極等電位)
    conductor_atom = Conductor.Electrode_contact_atom
    En_external = dot(E_external, [atom.nx, atom.ny, atom.nz])
    
    if Conductor.close_conductor_Electrode:
        # 與電極接觸: σ/ε = dV/L
        dE_conductor = -(En_external + self.Cathode.Voltage / self.Lgap / 2.0)
    else:
        # 與其他導體接觸
        dE_conductor = -En_external
    
    # 幾何依賴的電荷計算
    if type(Conductor).__name__ == "Buckyball_Virtual":
        dQ_conductor = sign * dE_conductor * Conductor.dr_center_contact**2
    elif type(Conductor).__name__ == "Nanotube_Virtual":
        dQ_conductor = sign * dE_conductor * Conductor.dr_center_contact * Conductor.length / 2.0
```

**物理意義**:
- 球形: Gauss 定律 → Q = E * r²
- 管狀: Gauss 定律 → Q = E * r * L/2

---

### 3.4 內建: API 使用示例

```python
# 創建內建 Force
force = mm.ConstantPotentialForce()

# 添加粒子 (固定電荷)
for atom in topology.atoms():
    force.addParticle(charge)

# 添加電極 (浮動電荷)
cathode_particles = set([100, 101, 102, ...])
force.addElectrode(
    electrodeParticles=cathode_particles,
    potential=-2.0,           # kJ/mol/e
    gaussianWidth=0.05,       # nm
    thomasFermiScale=0.0      # 1/nm (0 表示不使用 TF 模型)
)

# 設置求解方法
force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
force.setUsePreconditioner(True)
force.setCGErrorTolerance(0.01)  # kJ/mol/e

# 可選: 總電荷約束
force.setUseChargeConstraint(True)
force.setChargeConstraintTarget(0.0)

# 可選: 外場
force.setExternalField(mm.Vec3(0, 0, 10))  # kJ/mol/nm/e

# 添加到系統
system.addForce(force)
```

**優勢**:
- API 清晰,參數物理意義明確
- 支持外場和電荷約束
- 兩種求解方法可選

**限制**:
- 無法指定多種導體幾何
- 無 Virtual/Real 層分離

---

## 四、關鍵問題分析

### 問題 1: 內建能否處理 Buckyball/Nanotube?

**答案: ❌ 不能**

**證據**:
1. API 僅提供 `addElectrode(electrodeParticles, ...)`,沒有幾何類型參數
2. 源碼中沒有 `Buckyball`、`Nanotube`、`conductor` 等關鍵字
3. 論文 (Dufils 2019, Scalfi 2020) 僅討論**平面電極**

**舊版優勢**:
```python
# 舊版支持 3 種導體類型
class Electrode_Virtual(Conductor_Virtual):     # 平面
class Buckyball_Virtual(Conductor_Virtual):     # 球形
class Nanotube_Virtual(Conductor_Virtual):      # 管狀

# 每種類型都有專門的:
- 幾何計算 (area_atom, radius, length)
- 法向量計算 (nx, ny, nz)
- 鏡像電荷處理
- 電荷轉移邊界條件
```

---

### 問題 2: Green's Reciprocity 解析校正是否必要?

**答案: ⚠️ 取決於精度要求**

**舊版的解析校正 (Q_analytic) 提供**:
1. **幾何精確性**: 
   - 公式 `Q = A * (V/Lgap + V/Lcell)` 來自靜電學基本原理
   - 對平面電極是精確解
   
2. **鏡像電荷修正**:
   - 考慮電解質原子和導體原子的鏡像貢獻
   - 確保總電荷守恆

3. **數值穩定性**:
   - 每次迭代後標準化到 Q_analytic
   - 防止電荷累積誤差

**內建方法的替代**:
- PME 本身隱含處理鏡像電荷 (通過 Ewald 求和)
- CG/Matrix 方法迭代到收斂 (RMS 誤差 < tolerance)
- Thomas-Fermi 模型提供額外的物理約束

**評估**: 
- 對**平面電極**,內建 PME 可能已足夠精確
- 對**複雜幾何** (Buckyball/Nanotube),解析校正可能更重要

---

### 問題 3: Virtual/Real 層分離的必要性?

**答案: ⚠️ 針對 Buckyball/Nanotube 必要**

**舊版設計**:
```python
# Buckyball_Virtual 類
self.electrode_atoms = []         # Virtual 層 (鏡像電荷)
self.electrode_atoms_real = []    # Real 層 (實際原子)

# 排除項設置:
# - Virtual/Virtual: 不排除 (需要交互以產生正確的法向場)
# - Real/Real: 排除
# - Real/Virtual: 排除
```

**物理意義**:
- **Virtual 層**: 表示導體表面的鏡像電荷分布
- **Real 層**: 表示導體內部的實際原子
- Virtual 層之間的交互產生正確的表面法向電場

**內建方法**:
- 僅單層電極原子
- Gaussian 電荷分布可能部分替代 Virtual 層效果

**評估**:
- 對平面電極,單層足夠
- 對球形/管狀,Virtual/Real 分離可能必要

---

## 五、遷移策略建議

### 方案 A: 完全使用內建 (適用於平面電極)

**適用場景**:
- ✅ 僅使用平面電極 (Electrode_Virtual)
- ✅ 不需要 Buckyball/Nanotube
- ✅ 希望使用正確的 PME 電靜力
- ✅ 需要 Thomas-Fermi 模型或外場

**遷移步驟**:
1. 識別舊代碼中的電極定義 (Cathode/Anode)
2. 轉換為 `force.addElectrode()`
3. 調整電壓單位 (kJ/mol/e)
4. 選擇 `gaussianWidth` (建議從 0.05 nm 開始)
5. 測試並比較結果

**優勢**:
- 正確的 PME 長程電靜力
- 更快的收斂 (CG + 預條件器)
- 支持外場和電荷約束
- 官方支持和持續維護

**劣勢**:
- 失去 Green's reciprocity 解析校正
- 無 Virtual/Real 層分離

---

### 方案 B: 混合方案 (保留舊版 Poisson Solver)

**適用場景**:
- ⚠️ 使用 Buckyball/Nanotube
- ⚠️ 需要 Green's reciprocity 精確校正
- ⚠️ 需要 Virtual/Real 層分離
- ✅ 但希望使用 PME 電靜力

**策略**:
1. **使用內建 ConstantPotentialForce 計算 PME 電靜力**
   - 不添加電極 (或添加但設為 0 電壓)
   - 僅用於正確的長程相互作用

2. **保留舊版 Poisson Solver 計算電極電荷**
   - 繼續使用 `Poisson_solver_fixed_voltage()`
   - 使用 PME 提供的正確電場
   - 應用 Green's reciprocity 校正

3. **整合兩者**
   - 舊版代碼讀取 PME 計算的力/電場
   - 舊版代碼更新電極電荷
   - 更新 Context 中的 NonbondedForce 參數

**實現難度**: ⚠️ 高
- 需要協調兩個 Force 對象
- 可能有性能損失
- 需要仔細測試

---

### 方案 C: 完全保留舊版 + 修復 PME (最保守)

**適用場景**:
- ❌ 使用 Buckyball/Nanotube (必須)
- ❌ 不想改變任何物理模型
- ✅ 只需修復 PME 錯誤

**策略**:
1. 在舊版代碼中添加 `NonbondedForce` (PME 模式)
2. 與現有 `CustomNonbondedForce` 協調
3. 保持所有其他邏輯不變

**優勢**:
- 最小改動
- 保留所有舊版特性

**劣勢**:
- 失去內建實現的所有優勢
- 需要手動維護

---

## 六、具體問題回答

### Q1: "所以我們現在不用寫 PME 是真?"

**A1**: 
- ✅ **對平面電極**: 是的,內建 ConstantPotentialForce 已經實現了正確的 PME
- ❌ **對 Buckyball/Nanotube**: 內建不支持,需要其他方案

---

### Q2: "之前的 Poisson 部分還是有用吧?"

**A2**: ⚠️ **部分有用,取決於需求**

**有用的部分**:
1. **Buckyball/Nanotube 支持** (內建沒有)
2. **Green's reciprocity 解析校正** (可能提供更高精度)
3. **Virtual/Real 層分離架構** (對複雜幾何可能必要)
4. **明確的物理模型** (容易理解和調試)

**可替代的部分**:
1. **平面電極的電荷求解** (內建 CG/Matrix 更先進)
2. **PME 電靜力** (內建實現更正確)
3. **Thomas-Fermi 模型** (內建有,舊版沒有)

---

### Q3: "考慮怎麼遷移?"

**A3**: **分情況討論**

**情況 1: 只用平面電極**
→ **推薦方案 A** (完全使用內建)
- 遷移成本低
- 長期受益最大

**情況 2: 需要 Buckyball/Nanotube**
→ **推薦方案 C** (保留舊版 + 修復 PME)
- 保留核心功能
- 僅修復 PME 錯誤
- 或考慮用內建實現類似效果 (但需要研究)

**情況 3: 平面為主,偶爾需要複雜幾何**
→ **考慮方案 B** (混合方案)
- 但需要評估實現複雜度

---

## 七、建議的下一步行動

### 立即行動 (不需要編碼)

1. **明確需求**:
   - [ ] 系統中是否使用 Buckyball/Nanotube?
   - [ ] 是否需要 Green's reciprocity 精確校正?
   - [ ] 對性能和精度的要求如何?

2. **閱讀文獻**:
   - [ ] Dufils et al., *Phys. Rev. Lett.* **123**, 195501 (2019)
   - [ ] Scalfi et al., *J. Chem. Phys.* **153**, 174704 (2020)
   - 理解內建實現的理論基礎

3. **小規模測試**:
   - [ ] 用簡單平面電極系統測試內建 Force
   - [ ] 比較與舊版的結果差異
   - [ ] 評估 `gaussianWidth` 參數的影響

### 長期計劃 (需要編碼)

4. **根據測試結果選擇方案**:
   - 方案 A: 遷移腳本 (Python)
   - 方案 B: 混合框架 (Python + C++)
   - 方案 C: PME 修復 (在舊版代碼中)

5. **驗證和基準測試**:
   - 能量守恆
   - 電極電荷收斂性
   - 與實驗/理論對比

---

## 八、總結

| 方面 | 舊版優勢 | 內建優勢 |
|------|---------|---------|
| **電靜力** | ❌ 真空求和 (錯誤) | ✅ PME (正確) |
| **電極類型** | ✅ 多種幾何 (Plane/Sphere/Tube) | ❌ 僅平面 |
| **理論模型** | ✅ Green's reciprocity + Poisson | ✅ Gaussian + Thomas-Fermi |
| **求解方法** | ⚠️ 固定迭代 (3 次) | ✅ CG/Matrix + 容忍度 |
| **高級功能** | ❌ 無外場/約束 | ✅ 外場 + 總電荷約束 |
| **維護支持** | ❌ 無官方支持 | ✅ OpenMM 官方維護 |

**最終建議**: 
- 先明確系統需求 (是否需要 Buckyball/Nanotube)
- 進行小規模測試比較
- **不要急著寫代碼,先驗證理論適用性** ✅
