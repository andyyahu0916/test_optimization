# 🚨 發現嚴重 Bug！

## 問題描述

當 `config.ini` 設定 `anti_explosion_strategy = original` 時，
**實際上並沒有使用教授的原始算法！**

## 證據

### 教授的原始版本 (backups/OpenMM-ConstantV(original)/lib/MM_classes.py, lines 320-350)

```python
# 只有一個檢查
Ez_external = ( forces[index][2]._value / q_i_old ) if abs(q_i_old) > (0.9*self.small_threshold) else 0.
q_i = prefactor * (voltage_term + Ez_external)
if abs(q_i) < self.small_threshold:
    q_i = sign * self.small_threshold
```

**特點：**
- ✅ 只檢查 `abs(q_i_old) > 0.9*small_threshold`
- ✅ 沒有 damping
- ✅ 沒有 q_max 限制
- ✅ 沒有 Ez_max 限制

### 目前的 Cython 版本 (lib/electrode_charges_cython.pyx, lines 77-120)

```python
cdef double damping = 0.5  # ❌ 硬編碼！
cdef double q_max = 0.5   # ❌ 硬編碼！
cdef double Ez_max = 1e4   # ❌ 硬編碼！

# ... 各種 clamp 和 damping ...
```

**問題：**
- ❌ 不管 config.ini 設什麼，都在用這些硬編碼值
- ❌ 等於一直在跑 aggressive 策略！
- ❌ config.ini 的 `anti_explosion_strategy` 完全沒作用

## 影響

1. **測試結果不可信**：我們以為在測試 original，實際上在測 aggressive
2. **無法驗證原始算法**：想知道教授原始版本行為？現在根本測不到
3. **Debug 方向錯誤**：一直在調整不存在的 "original" 策略

## 需要做的修正

### 方案 A: 傳遞參數（推薦）✅

```python
# In MM_classes_CYTHON.py
anti_explosion_strategy = self.config['Simulation']['anti_explosion_strategy']

if anti_explosion_strategy == 'original':
    damping = 0.0  # No damping
    q_max = 999.0  # No limit
    Ez_max = 999.0  # No limit
elif anti_explosion_strategy == 'conservative':
    damping = 0.0
    q_max = 0.5
    Ez_max = 1e4
elif anti_explosion_strategy == 'aggressive':
    damping = 0.5
    q_max = 0.5
    Ez_max = 1e4

# Pass to Cython
cathode_q_new = ec_cython.compute_electrode_charges_cython(
    forces_z, cathode_q_old, self._cathode_indices,
    cathode_prefactor, voltage_term_cathode,
    threshold_check, self.small_threshold,
    1.0,
    damping, q_max, Ez_max  # ← 新增參數
)
```

### 方案 B: 創建三個不同的 Cython 函數

```python
compute_electrode_charges_original()   # 純原始
compute_electrode_charges_conservative()  # 有限制無阻尼
compute_electrode_charges_aggressive()   # 有限制有阻尼
```

## 緊急程度

🔴 **高優先級**

目前所有的測試結果都需要重新解讀，因為實際上沒有跑到 original 策略。

## 建議行動

1. ✅ 先檢查並記錄這個 bug
2. ⚠️ 決定修復方案（A 或 B）
3. ⚠️ 重新編譯 Cython
4. ⚠️ 重新測試所有策略
5. ⚠️ 更新 debug report

---

**發現時間**: 2025-10-25  
**發現者**: GitHub Copilot (during code verification)
