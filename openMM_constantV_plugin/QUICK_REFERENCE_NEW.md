# OpenMM ConstantPotentialForce 快速參考

## 🚀 一分鐘上手

```python
import openmm as mm

# 創建 + 添加粒子
force = mm.ConstantPotentialForce()
for i in range(N):
    force.addParticle(charge[i])

# 定義電極
electrode = set([0, 1, 2])
V_kj = voltage_volts * 96.485  # V -> kJ/mol/e
force.addElectrode(electrode, V_kj, 0.05, 0.0)

# PME 設置
force.setCutoffDistance(1.0 * mm.unit.nanometer)
force.setEwaldErrorTolerance(1e-4)

# 求解方法
force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)

# 添加並運行
system.addForce(force)
```

---

## 📋 API 速查

### 創建與配置

```python
force = mm.ConstantPotentialForce()
force.addParticle(charge)                    # 添加粒子
force.addElectrode(particles, V, σ, λ_TF)    # 添加電極
force.addException(i, j, chargeProd)         # 添加排除項
```

### PME 設置

```python
force.setCutoffDistance(r_cut)               # 截斷距離
force.setEwaldErrorTolerance(tol)            # 誤差容忍度
force.setPMEParameters(α, nx, ny, nz)        # 手動 PME (可選)
```

### 求解方法

```python
# CG 方法 (動態系統)
force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
force.setCGErrorTolerance(0.1)

# Matrix 方法 (固定電極)
force.setConstantPotentialMethod(mm.ConstantPotentialForce.Matrix)
```

### 運行時

```python
charges = force.getCharges(context)          # 獲取電荷
force.updateParametersInContext(context)     # 更新參數
```

---

## 🔢 單位轉換

### 電壓 ↔ OpenMM

```python
# Volts -> kJ/mol/e
V_kj = voltage_volts * 96.485

# kJ/mol/e -> Volts
voltage_volts = V_kj / 96.485
```

### 電荷單位

```python
# 使用基本電荷單位 (e)
charge_e = 1.0  # +1e (Na+)

# 提取數值
q = charges[i].value_in_unit(mm.unit.elementary_charge)
```

---

## ⚙️ 典型參數

### PME 參數

```python
cutoff = 1.0 * mm.unit.nanometer    # 截斷距離
tolerance = 1e-4                     # Ewald 誤差
# 讓 OpenMM 自動選擇 α, nx, ny, nz
```

### 電極參數

```python
gaussian_width = 0.05      # nm (典型: 0.03-0.07)
thomas_fermi = 0.0         # 不使用 TF 模型
```

### CG 求解器

```python
cg_tolerance = 0.1         # kJ/mol/e (典型: 0.01-1.0)
```

---

## 📊 常見系統配置

### 雙電極系統

```python
# 電極 A: +1.0 V
electrode_a = set(range(0, 100))
force.addElectrode(electrode_a, 96.485, 0.05, 0.0)

# 電極 B: 接地 (0 V)
electrode_b = set(range(100, 200))
force.addElectrode(electrode_b, 0.0, 0.05, 0.0)

# 電解液: 固定電荷
for i in range(200, 300):
    force.addParticle(charge[i])
```

### 單電極 + 外場

```python
# 電極
electrode = set(range(0, 100))
force.addElectrode(electrode, 96.485, 0.05, 0.0)

# 其他粒子
for i in range(100, N):
    force.addParticle(charge[i])
```

---

## 🐛 故障排查

### 問題: 能量不穩定

```python
# 解決方案 1: 降低 CG 容忍度
force.setCGErrorTolerance(0.01)  # 更嚴格

# 解決方案 2: 檢查 PME 參數
force.setEwaldErrorTolerance(1e-5)  # 更精確
```

### 問題: 電荷不合理

```python
# 檢查電壓單位
V_kj = voltage * 96.485  # 確保轉換正確

# 檢查電極定義
print(f"電極粒子數: {len(electrode)}")
print(f"電壓: {potential / 96.485} V")
```

---

## 📚 資源鏈接

### 文檔

- **OpenMM 用戶指南**: http://docs.openmm.org/
- **源碼**: `openmm-8.4.0/openmmapi/include/openmm/ConstantPotentialForce.h`

### 論文

- **方法**: Dufils et al., Phys. Rev. Lett. 123, 195501 (2019)
- **TF 模型**: Scalfi et al., J. Chem. Phys. 153, 174704 (2020)

### 本地文件

- **完整演示**: `demo_builtin_constantpotential.py`
- **遷移指南**: `MIGRATION_GUIDE.md`
- **發現報告**: `CRITICAL_DISCOVERY.md`

---

**最後更新**: 2024-11-04  
**版本**: OpenMM 8.4.0  
**狀態**: ✅ 生產就緒
