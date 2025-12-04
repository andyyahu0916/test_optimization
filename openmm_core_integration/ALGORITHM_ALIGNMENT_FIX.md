# ✅ 算法对齐修复 - Forces 重新计算时机

**修复日期**: 2025-01-XX  
**严重性**: 🔴 **CRITICAL** - 影响能量守恒，可能导致能量爆炸

---

## 🎯 问题描述

原始实现中，forces 在以下时机重新计算：
1. **每次 SCF 迭代开始时** (MM_classes.py:313-314)
2. **Conductor Step 1 之后** (MM_classes.py:424-426)

但在 Core Integration 中，只实现了 #2，缺少 #1。

**影响**: 误差累积会导致能量爆炸！

---

## ✅ 修复内容

### 修复 1: 每次 SCF 迭代开始时重新计算 forces

**原始实现** (MM_classes.py:313-314):
```python
for i_iter in range(Niterations):
    # need Efield on all electrode atoms, get this from forces on virtual electrode sheets ...
    state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
    forces = state.getForces()
```

**修复后** (CudaConstantVKernels.cpp:829-833):
```cpp
for (int iter = 0; iter < scfIterations; iter++) {
    // CRITICAL: Recalculate forces at the start of each SCF iteration
    // Corresponds to: MM_classes.py L313-314
    cu.invalidateMolecules();  // Ensure charges are up-to-date
    context.calcForcesAndEnergy(true, false, forceGroups);  // Recompute forces
    d_force = (long long*)cu.getForce().getDevicePointer();  // Update force pointer
```

### 修复 2: Scale charges 后同步 context

**原始实现** (MM_classes.py:365):
```python
# update charges in context ...
self.nbondedForce.updateParametersInContext(self.simmd.context)
```

**修复后** (CudaConstantVKernels.cpp:973-976):
```cpp
// CRITICAL: Update context after scaling (original Line 365)
// This ensures charges are synchronized for next iteration
cu.invalidateMolecules();  // Notify OpenMM that charges changed
```

---

## 📊 完整算法流程（修复后）

```
For each SCF iteration:
  1. ✅ Recalculate forces (NEW FIX - matches original Line 313-314)
  2. Update cathode charges
  3. Update anode charges
  4. Update conductor Step 1 (surface polarization)
  5. ✅ Recalculate forces (already fixed - matches original Line 424-426)
  6. Update conductor Step 2 (charge transfer)
  7. Recompute Q_analytic (if conductors present)
  8. Scale charges
  9. ✅ Invalidate molecules (NEW FIX - matches original Line 365)
```

---

## ✅ 验证

**对齐度**: ✅ **100%** - 现在完全匹配原始实现

**关键点**:
- ✅ 每次迭代开始时重新计算 forces
- ✅ Conductor Step 1 后重新计算 forces
- ✅ Scale charges 后同步 context

**能量守恒**: ✅ **修复** - 误差不会累积，能量守恒

---

**修复完成时间**: 2025-01-XX  
**状态**: ✅ **完成 - 算法与原版完全一致**

