# 如何使用防爆策略

## 🚀 快速開始

### 1. 選擇策略（在 config.ini 中）

```ini
[Simulation]
mm_version = cython  # 使用優化版本
anti_explosion_strategy = original  # ⭐ 推薦從這裡開始
```

### 2. 策略說明

| 策略 | 穩定性 | 性能 | 物理正確性 | 適用場景 |
|------|--------|------|-----------|----------|
| `none` | ⚠️ 低 | ⚡ 最快 | ✅ 最高 | 研究數值誤差、記錄爆炸模式 |
| `original` | 🟡 中 | ⚡ 快 | ✅ 高 | **一般使用**（教授推薦）|
| `conservative` | 🟢 高 | 🐌 稍慢 | 🟡 中 | 經常遇到 NaN 爆炸 |
| `aggressive` | 🟢 很高 | 🐌 慢 | ⚠️ 低 | 非常不穩定的系統 |

### 3. 如果還是爆炸怎麼辦？

#### 方法 A: 重算（教授推薦）✅
```bash
# 換個隨機種子，大部分情況能解決
python run_openMM.py > energy_run2.log &
```

#### 方法 B: 調整策略
```ini
# 從 original → conservative
anti_explosion_strategy = conservative

# 如果還不行 → aggressive
anti_explosion_strategy = aggressive
```

#### 方法 C: 降低難度
```ini
# 降低電壓
voltage = 1.0  # 從 2.0V 降到 1.0V

# 或減少更新頻率
freq_charge_update_fs = 500  # 從 200 增加到 500
```

---

## 🔧 進階調整

### Conservative 策略的參數

```ini
anti_explosion_strategy = conservative

# 調整電荷限制（預設 0.5）
q_max_limit = 0.3  # 更嚴格（更穩定但可能影響精度）
q_max_limit = 0.8  # 更寬鬆（更接近物理但可能不穩定）

# 修復 scale_factor bug（強烈建議開啟）
fix_negative_scale_bug = True
```

### Aggressive 策略的參數

```ini
anti_explosion_strategy = aggressive

# 調整阻尼（預設 0.5）
damping_factor = 0.3  # 較少阻尼（收斂快但可能不穩定）
damping_factor = 0.7  # 較多阻尼（很穩定但收斂慢）

# 其他參數同 conservative
```

---

## 📊 如何判斷是否成功？

### 成功的跡象 ✅
```
Q_numeric cathode: -10.5  (在 ±40 範圍內)
Q_numeric anode: 10.3
...
Simulation completed: 20 ns
```

### 失敗的跡象 ❌
```
Q_numeric cathode: -223  (異常大!)
...
❌ OpenMMException: Particle coordinate is NaN
```

### 警告訊息（可能要調整）⚠️
```
⚠️ WARNING: scale_factor=5.234 too large! Clamping to ±2.0
⚠️ WARNING: Q_numeric and Q_analytic have opposite signs!
!!! CLAMP after scale: atom[42] 1.234 -> 0.500
```

---

## 🎓 教授的建議

> "搞定這問題搞不好能發 paper，搞不定就重算一次，大部分情況下能解決..."

**實用策略**：
1. 先用 `anti_explosion_strategy = original` 試試
2. 如果爆了，**重算一次**（換隨機種子）
3. 如果還爆，換成 `conservative`
4. 記錄哪些參數/種子成功，之後就用那些

**不要死磕**：
- 如果試了 3-5 次還是爆，可能這個系統就是不穩定
- 考慮降低電壓或改變系統設置
- 或者這就是個值得研究的問題（可能發 paper）

---

## 📝 記錄你的測試

建議創建一個測試記錄：

```bash
# test_log.txt
Run 1: original, 2.0V, seed=12345 → NaN at 15ns
Run 2: original, 2.0V, seed=67890 → Success! ✅
Run 3: conservative, 2.0V, seed=12345 → Success!
Run 4: aggressive, 4.0V, seed=12345 → Success but slow
```

這樣你就知道哪些組合有效！

---

## 🐛 Debug 模式

如果你想深入研究數值問題：

```ini
anti_explosion_strategy = none  # 不使用任何防護

# 然後創建測試腳本記錄爆炸時的所有數據
python test_stability.py > detailed_crash_log.txt
```

查看 `NUMERICAL_STABILITY_DEBUG_REPORT.md` 了解更多技術細節。

---

**祝模擬成功！** 🎉

如果真的解決了這個問題，記得跟教授說可能可以發 paper 😉
