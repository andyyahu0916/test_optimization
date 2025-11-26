## 初始電荷計算對比

### Python (Fixed_Voltage_routines.py:293)

**Cathode (sign=1.0):**
```python
q_i = sign / ( 4.0 * numpy.pi ) * self.area_atom * (self.Voltage / Lgap + self.Voltage / Lcell) * conversion_KjmolNm_Au
```

即：
```python
q_i = 1.0 / ( 4.0 * numpy.pi ) * area_atom * (Voltage / Lgap + Voltage / Lcell) * conversion_KjmolNm_Au
```

### C++ (ReferenceConstantVKernels.cpp:179-180) - Force kernel

**Cathode:**
```cpp
double q_i = 1.0 / (4.0 * M_PI) * areaPerAtom[i] *
             (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOLNM_AU;
```

✓ 完全一致！

**Anode (sign=-1.0):**
```python
q_i = -1.0 / ( 4.0 * numpy.pi ) * area_atom * (Voltage / Lgap + Voltage / Lcell) * conversion_KjmolNm_Au
```

```cpp
double q_i = -1.0 / (4.0 * M_PI) * areaPerAtom[cathodeAtomIndices.size() + i] *
             (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOLNM_AU;
```

✓ 完全一致！

### 物理意義：
- 初始電荷基於平板電容器公式
- V/L_gap: 真空間隙的電場貢獻
- V/L_cell: 週期性邊界條件的鏡像貢獻
- 兩項相加反映了週期性系統的總效應

### 低電壓保護 (Line 286-296)：

**Python:**
```python
if abs(self.Voltage) < 0.01:
    print( "adding small value..." )
    flag_small=True
# ...
if flag_small:
    q_i = q_i + sign * MMsys.small_threshold
```

**C++:**
```cpp
if (fabs(voltage) < 0.01) {
    std::cout << "adding small value..." << std::endl;
    flag_small = true;
}
// ...
if (flag_small) {
    q_i = q_i + SMALL_THRESHOLD;  // Cathode
}
```

✓ 完全一致！
