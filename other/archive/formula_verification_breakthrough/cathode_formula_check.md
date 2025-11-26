## 陰極電荷更新公式對比

### Python (MM_classes.py:330)
```python
q_i = 2.0 / ( 4.0 * numpy.pi ) * self.Cathode.area_atom * (self.Cathode.Voltage / self.Lgap + Ez_external) * conversion_KjmolNm_Au
```

### C++ (ReferenceConstantVKernels.cpp:386-388)
```cpp
double q_i = 2.0 / (4.0 * M_PI) * areaPerAtom[i] *
            (voltage / Lgap + Ez_external) *
            CONVERSION_KJMOLNM_AU;
```

### 對比項目：
1. 係數：2.0 / (4.0 * π) ✓
2. 面積：area_atom vs areaPerAtom[i] ✓
3. 電壓項：Cathode.Voltage / Lgap vs voltage / Lgap ✓
4. 電場項：Ez_external ✓
5. 轉換因子：conversion_KjmolNm_Au ✓

### 問題檢查：
- Python: self.Cathode.Voltage (cathode的電壓)
- C++: voltage (共用一個voltage變量)

需要確認：Python中Cathode.Voltage == Anode.Voltage？
