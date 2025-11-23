# 決策檢查清單: 選擇 OpenMM ConstantPotentialForce 實現方案

## 🎯 快速決策流程

```
開始
  |
  v
你的系統中是否使用 Buckyball 或 Nanotube 導體?
  |
  |-- 是 --> [保留舊版實現] 
  |           - 內建不支持這些幾何
  |           - 跳到方案 C (修復 PME)
  |
  |-- 否 --> 僅使用平面電極?
              |
              |-- 是 --> 是否需要 Green's reciprocity 解析校正?
              |           |
              |           |-- 否 --> [使用內建] ✅ 推薦
              |           |           - 跳到方案 A
              |           |
              |           |-- 是 --> 是否能接受 Gaussian 電荷分布?
              |                       |
              |                       |-- 是 --> [使用內建] ✅ 推薦
              |                       |           - 測試後決定
              |                       |
              |                       |-- 否 --> [混合方案] ⚠️
              |                                   - 跳到方案 B
              |
              |-- 否 --> 重新檢查系統設置
                         (你說只用平面電極但又說不是?)
```

---

## ✅ 方案 A: 完全使用內建 ConstantPotentialForce

### 前置條件檢查

- [ ] **系統僅使用平面電極** (無 Buckyball/Nanotube)
- [ ] **可接受 Gaussian 電荷分布** (vs 點電荷)
- [ ] **OpenMM 版本 ≥ 8.4.0**
- [ ] **希望使用正確的 PME 電靜力**

### 優勢評估

- [ ] ✅ 正確的 PME 長程電靜力
- [ ] ✅ 兩種求解方法 (CG/Matrix)
- [ ] ✅ 支持 Thomas-Fermi 模型
- [ ] ✅ 支持外場和總電荷約束
- [ ] ✅ 官方維護,持續更新
- [ ] ✅ CUDA/OpenCL 優化

### 劣勢評估

- [ ] ❌ 無 Green's reciprocity 解析校正
- [ ] ❌ 無 Virtual/Real 層分離
- [ ] ❌ 僅支持平面電極

### 遷移步驟

1. **閱讀文獻** (建議)
   - [ ] Dufils et al., *Phys. Rev. Lett.* **123**, 195501 (2019)
   - [ ] Scalfi et al., *J. Chem. Phys.* **153**, 174704 (2020)

2. **識別舊代碼中的電極定義**
   ```python
   # 舊版代碼示例
   Cathode = Electrode_Virtual(...)
   Anode = Electrode_Virtual(...)
   ```
   - [ ] 記錄陰極電壓: _______ V
   - [ ] 記錄陽極電壓: _______ V
   - [ ] 記錄電極原子索引

3. **轉換為內建 API**
   ```python
   force = mm.ConstantPotentialForce()
   
   # 添加非電極粒子
   for atom in topology.atoms():
       force.addParticle(charge)
   
   # 添加陰極
   cathode_particles = set([...])  # 從舊代碼提取
   force.addElectrode(
       electrodeParticles=cathode_particles,
       potential=-2.0,           # kJ/mol/e (注意單位!)
       gaussianWidth=0.05,       # nm (從 0.05 開始測試)
       thomasFermiScale=0.0      # 1/nm (暫時不用 TF 模型)
   )
   
   # 添加陽極
   anode_particles = set([...])
   force.addElectrode(
       electrodeParticles=anode_particles,
       potential=2.0,
       gaussianWidth=0.05,
       thomasFermiScale=0.0
   )
   
   # 設置求解方法
   force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
   force.setUsePreconditioner(True)
   force.setCGErrorTolerance(0.01)  # kJ/mol/e
   
   system.addForce(force)
   ```
   - [ ] 完成代碼轉換
   - [ ] 檢查單位轉換 (V → kJ/mol/e)

4. **參數調優**
   - [ ] 測試不同 `gaussianWidth` 值: 0.03, 0.05, 0.08 nm
   - [ ] 測試不同 `cgErrorTolerance` 值: 0.01, 0.001, 0.0001
   - [ ] 評估是否需要 Thomas-Fermi 模型 (thomasFermiScale > 0)

5. **驗證測試**
   - [ ] 能量守恆檢查
   - [ ] 電極電荷收斂性
   - [ ] 與舊版結果對比 (電荷分布、能量)
   - [ ] 性能測試 (速度比較)

6. **可選: 添加高級功能**
   ```python
   # 總電荷約束
   force.setUseChargeConstraint(True)
   force.setChargeConstraintTarget(0.0)
   
   # 外場
   force.setExternalField(mm.Vec3(0, 0, Ez))  # kJ/mol/nm/e
   ```
   - [ ] 評估是否需要外場
   - [ ] 評估是否需要總電荷約束

---

## ⚠️ 方案 B: 混合方案 (內建 PME + 舊版 Poisson Solver)

### 前置條件檢查

- [ ] **需要 Buckyball/Nanotube 支持**
- [ ] **需要 Green's reciprocity 精確校正**
- [ ] **需要 Virtual/Real 層分離**
- [ ] **同時需要正確的 PME 電靜力**
- [ ] **有足夠的開發資源實現混合架構**

### 挑戰評估

- [ ] ⚠️ 需要協調兩個 Force 對象
- [ ] ⚠️ 可能有性能損失
- [ ] ⚠️ 實現複雜度高
- [ ] ⚠️ 測試和驗證成本高

### 實現策略

1. **架構設計**
   ```
   [ConstantPotentialForce]         [舊版 Poisson Solver]
   (僅用於 PME 電靜力)              (用於電極電荷求解)
           |                                |
           v                                v
      計算正確的電場              讀取 PME 電場
           |                                |
           +--------------------------------+
                            |
                            v
                   更新 NonbondedForce
                   (應用新電極電荷)
   ```

2. **代碼修改點**
   - [ ] 添加 ConstantPotentialForce (0 電壓或不添加電極)
   - [ ] 修改舊版 `Poisson_solver_fixed_voltage()` 讀取 PME 電場
   - [ ] 協調 CustomNonbondedForce 和 ConstantPotentialForce
   - [ ] 測試兩個 Force 的交互

3. **風險評估**
   - [ ] 是否會有重複計算?
   - [ ] 是否會有數值不穩定?
   - [ ] 性能損失是否可接受?

**建議**: 除非絕對必要,否則不推薦此方案。先嘗試方案 A 或 C。

---

## ✅ 方案 C: 完全保留舊版 + 修復 PME (最保守)

### 前置條件檢查

- [ ] **必須使用 Buckyball/Nanotube**
- [ ] **不想改變任何物理模型**
- [ ] **只需修復 PME 錯誤**
- [ ] **可接受手動維護代碼**

### 實現步驟

1. **分析當前排除項設置**
   - [ ] 檢查 `generate_exclusions()` 函數
   - [ ] 檢查 `SAPT_FF_exclusions()` 函數
   - [ ] 識別所有排除項邏輯

2. **添加 NonbondedForce (PME 模式)**
   ```python
   # 在舊版代碼中添加
   nonbonded = mm.NonbondedForce()
   nonbonded.setNonbondedMethod(mm.NonbondedForce.PME)
   nonbonded.setCutoffDistance(1.0 * unit.nanometers)
   nonbonded.setEwaldErrorTolerance(1e-5)
   
   # 添加所有粒子
   for atom in topology.atoms():
       nonbonded.addParticle(charge, sigma, epsilon)
   
   # 複製排除項 (從 CustomNonbondedForce)
   for i in range(customNonbonded.getNumExclusions()):
       particle1, particle2 = customNonbonded.getExclusionParticles(i)
       nonbonded.addException(particle1, particle2, 0, 0, 0)
   
   system.addForce(nonbonded)
   ```
   - [ ] 完成代碼修改
   - [ ] 確保排除項一致

3. **協調兩個 Nonbonded Force**
   - [ ] 決定是否保留 CustomNonbondedForce
   - [ ] 或完全用 NonbondedForce 替代
   - [ ] 測試 Lennard-Jones 相互作用

4. **驗證 PME 計算**
   - [ ] 能量測試
   - [ ] 力的測試
   - [ ] 與方案 A (內建) 比較電場

5. **保留所有舊版特性**
   - [ ] Buckyball_Virtual 類正常工作
   - [ ] Nanotube_Virtual 類正常工作
   - [ ] Green's reciprocity 校正正常
   - [ ] Virtual/Real 層分離正常

---

## 📊 決策矩陣

| 需求 | 方案 A (內建) | 方案 B (混合) | 方案 C (舊版+PME) |
|------|--------------|--------------|------------------|
| 平面電極 | ✅ 最佳 | ✅ 可以 | ✅ 可以 |
| Buckyball/Nanotube | ❌ 不支持 | ✅ 支持 | ✅ 支持 |
| 正確 PME | ✅ 內建 | ✅ 內建 | ✅ 手動添加 |
| Green's reciprocity | ❌ 無 | ✅ 保留 | ✅ 保留 |
| Thomas-Fermi 模型 | ✅ 內建 | ⚠️ 複雜 | ❌ 無 |
| 外場支持 | ✅ 內建 | ⚠️ 複雜 | ❌ 需手動添加 |
| 實現難度 | ✅ 低 | ❌ 高 | ⚠️ 中 |
| 維護成本 | ✅ 低 | ❌ 高 | ⚠️ 中 |
| 性能 | ✅ 優化 | ⚠️ 可能較差 | ⚠️ 取決於實現 |

---

## 🚦 最終決策

**我的系統選擇**: 
- [ ] 方案 A (完全使用內建)
- [ ] 方案 B (混合方案)
- [ ] 方案 C (舊版 + 修復 PME)

**理由**:
```
(請填寫你的具體需求和理由)




```

**下一步行動**:
1. [ ] _______________________________________________
2. [ ] _______________________________________________
3. [ ] _______________________________________________

---

## 📚 參考資源

### 必讀文獻
- [ ] Dufils et al., *Phys. Rev. Lett.* **123**, 195501 (2019)
  - 標題: "Finite-size effects in periodic constant potential simulations"
  - 核心內容: PME 方法在常電壓模擬中的應用
  
- [ ] Scalfi et al., *J. Chem. Phys.* **153**, 174704 (2020)
  - 標題: "Molecular simulation of electrode-solution interfaces"
  - 核心內容: Thomas-Fermi 模型的實現

### OpenMM 文檔
- [ ] ConstantPotentialForce API: http://docs.openmm.org/latest/api-python/generated/openmm.openmm.ConstantPotentialForce.html
- [ ] PME 教程: http://docs.openmm.org/latest/userguide/theory/04_nonbonded_interactions.html

### 測試文件 (源碼中)
- [ ] `openmm-8.4.0/platforms/cuda/tests/TestCudaConstantPotentialForce.cpp`
- [ ] `openmm-8.4.0/platforms/reference/tests/TestReferenceConstantPotentialForce.cpp`

---

## ⏰ 時間線估計

### 方案 A (內建)
- 學習和測試: 1-2 天
- 代碼遷移: 2-3 天
- 驗證和調優: 3-5 天
- **總計**: 1-2 周

### 方案 B (混合)
- 架構設計: 2-3 天
- 實現: 1-2 周
- 測試和調試: 1-2 周
- **總計**: 3-5 周

### 方案 C (舊版+PME)
- PME 添加: 1-2 天
- 測試和驗證: 3-5 天
- **總計**: 1 周

---

## 💡 建議

1. **如果只用平面電極**: 
   → 強烈推薦**方案 A** (內建)
   - 最省時
   - 最優性能
   - 最佳長期維護

2. **如果必須用 Buckyball/Nanotube**:
   → 推薦**方案 C** (舊版+PME)
   - 保留所有功能
   - 僅修復 PME 錯誤
   - 避免方案 B 的複雜性

3. **如果不確定**:
   → 先做**小規模測試**
   - 用簡單系統測試方案 A
   - 比較與舊版的結果
   - 根據測試結果決定

**記住**: "先別急著 code" - 先明確需求,再選擇方案! 🎯
