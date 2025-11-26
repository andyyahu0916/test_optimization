# Integrator Kernel 的 updateParametersInContext 檢查

## Python代碼 (MM_classes.py:365)

在SCF迭代循環**內部**：
```python
for i_iter in range(Niterations):
    # 更新電荷...
    
    # Line 362-363: Green's校正
    self.Scale_charges_analytic_general()
    
    # Line 365: 每次迭代都更新Context
    self.nbondedForce.updateParametersInContext(self.simmd.context)
```

## C++ Integrator Kernel (ReferenceConstantVKernels.cpp:735-738)

```cpp
// Line 676: SCF迭代循環
for (int iter = 0; iter < nIterations; iter++) {
    // 更新Cathode電荷
    // 更新Anode電荷
    
    // Line 735-738: Green's校正 + 更新Context
    scaleChargesAnalytic(cathodeAtomIndices, Q_analytic_cathode);
    scaleChargesAnalytic(anodeAtomIndices, Q_analytic_anode);
    nonbondedForce->updateParametersInContext(context.getOwner());
}
```

✅ **完全一致！每次迭代都更新Context**

## 為什麼每次迭代都要更新？

因為：
1. SCF迭代需要獲取**最新電荷**的力
2. 每次迭代開始時調用 `calcForcesAndEnergy()`
3. 這個函數使用Context中的電荷來計算力
4. 所以必須在每次迭代結束時更新Context

## Force Kernel 對比

Force kernel只在整個execute()結束時調用一次updateParametersInContext：
```cpp
// Line 461: 只在最後更新一次
nonbondedForce->updateParametersInContext(context.getOwner());
```

這是因為Force kernel不自己調用calcForcesAndEnergy()，由外部調用。
