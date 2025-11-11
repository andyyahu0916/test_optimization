## 陽極電荷更新公式對比

### Python (MM_classes.py:345)
```python
q_i = -2.0 / ( 4.0 * numpy.pi ) * self.Anode.area_atom * (self.Anode.Voltage / self.Lgap + Ez_external) * conversion_KjmolNm_Au
```

### C++ (ReferenceConstantVKernels.cpp:426-428)
```cpp
double q_i = -2.0 / (4.0 * M_PI) * areaPerAtom[cathodeAtomIndices.size() + i] *
            (voltage / Lgap + Ez_external) *
            CONVERSION_KJMOLNM_AU;
```

### 關鍵檢查：
1. 負號：-2.0 ✓ (陽極帶負電)
2. 係數：/ (4.0 * π) ✓
3. 面積：areaPerAtom[cathodeAtomIndices.size() + i] 
   - 需要確認索引正確！

### Python中的area_atom：
- Cathode.area_atom: 單個原子的面積 (uniform)
- Anode.area_atom: 單個原子的面積 (uniform)

### C++中的areaPerAtom：
- areaPerAtom[0..N_cathode-1]: cathode面積
- areaPerAtom[N_cathode..N_cathode+N_anode-1]: anode面積

所以：areaPerAtom[cathodeAtomIndices.size() + i] 是正確的！✓
