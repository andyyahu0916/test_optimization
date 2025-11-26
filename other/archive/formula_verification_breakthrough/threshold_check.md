## Threshold保護邏輯對比

### 1. 計算Ez時的threshold (MM_classes.py:327)

**Python:**
```python
Ez_external = ( forces[index][2]._value / q_i_old ) if abs(q_i_old) > (0.9*self.small_threshold) else 0.
```

**C++:**
```cpp
if (fabs(q_i_old) > (0.9 * SMALL_THRESHOLD)) {
    Ez_external = forces[atomIdx][2] / q_i_old;
}
```

✓ 完全一致！注意是 **0.9*** threshold，不是1.0

### 2. 防止電荷歸零 (MM_classes.py:332-333)

**Python - Cathode:**
```python
if abs(q_i) < self.small_threshold:
    q_i = self.small_threshold  # Cathode, make positive
```

**C++ - Cathode:**
```cpp
if (fabs(q_i) < SMALL_THRESHOLD) {
    q_i = SMALL_THRESHOLD;  // Cathode為正
}
```

✓ 完全一致！Cathode設為正的threshold

**Python - Anode:**
```python
if abs(q_i) < self.small_threshold:
    q_i = -1.0 * self.small_threshold  # Anode, make negative
```

**C++ - Anode:**
```cpp
if (fabs(q_i) < SMALL_THRESHOLD) {
    q_i = -1.0 * SMALL_THRESHOLD;  // Anode為負
}
```

✓ 完全一致！Anode設為負的threshold

### 物理意義：
- 0.9*threshold: 留一些餘量，避免數值不穩定
- 防止歸零：確保下一次迭代能計算Ez = F/q
- 符號保持：Cathode正，Anode負
