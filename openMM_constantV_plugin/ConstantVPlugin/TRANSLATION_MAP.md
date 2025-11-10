# Python → C++ 逐行翻译对照表

## 📋 数值常数（Fixed_Voltage_routines.py:36-38）

| Python | 值 | C++ | 说明 |
|--------|-----|-----|------|
| `conversion_nmBohr` | 18.8973 | `CONVERSION_NMBOHR` | nm到Bohr转换 |
| `conversion_KjmolNm_Au` | 18.8973/2625.5 | `CONVERSION_KJMOLNM_AU` | kJ/mol·nm到原子单位 |
| `conversion_eV_Kjmol` | 96.487 | `CONVERSION_EV_KJMOL` | eV到kJ/mol |

**C++实现**（完全照抄）：
```cpp
static constexpr double CONVERSION_NMBOHR = 18.8973;
static constexpr double CONVERSION_KJMOLNM_AU = CONVERSION_NMBOHR / 2625.5;  // = 0.00719924...
static constexpr double CONVERSION_EV_KJMOL = 96.487;
```

---

## 📋 Threshold常数（MM_classes.py:48）

| Python | 值 | C++ |
|--------|-----|-----|
| `self.small_threshold` | 1e-6 | `SMALL_THRESHOLD` |

**使用位置**：
- 防止除零：`if abs(q_i_old) > 0.9*self.small_threshold`
- 防止电荷归零：`if abs(q_i) < self.small_threshold`
- Green's校正：`if abs(Q_numeric) > self.small_threshold`

---

## 📋 函数1：compute_Electrode_charge_analytic()

**Python原文**（Fixed_Voltage_routines.py:318-345）

```python
def compute_Electrode_charge_analytic(self, MMsys, positions, Conductor_list, z_opposite):
    # Line 319-322: 确定符号
    sign = 1.0
    if self.electrode_type == 'anode':
        sign = -1.0

    # Line 324-325: 几何贡献
    self.Q_analytic = sign / (4.0 * numpy.pi) * self.sheet_area * \
                      (self.Voltage / MMsys.Lgap + self.Voltage / MMsys.Lcell) * \
                      conversion_KjmolNm_Au

    # Line 327-333: 电解质镜像电荷贡献
    for index in MMsys.electrolyte_atom_indices:
        (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(index)
        z_atom = positions[index][2]._value  # in nm
        z_distance = abs(z_atom - z_opposite)
        self.Q_analytic += (z_distance / MMsys.Lcell) * (-q_i._value)

    # Line 335-344: 导体镜像电荷贡献
    if Conductor_list:
        for Conductor in Conductor_list:
            for atom in Conductor.electrode_atoms:
                index = atom.atom_index
                (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(index)
                z_atom = positions[index][2]._value
                z_distance = abs(z_atom - z_opposite)
                self.Q_analytic += (z_distance / MMsys.Lcell) * (-q_i._value)
```

**C++翻译**（逐行对应）：

```cpp
void ReferenceCalcConstantVKernel::computeElectrodeChargeAnalytic(
    const std::vector<int>& electrodeAtomIndices,
    const std::vector<Vec3>& positions,
    const std::string& electrodeType,
    double voltage,
    double Lgap,
    double Lcell,
    double sheetArea,
    double z_opposite,
    double& Q_analytic  // 输出
) {
    // Line 319-322: 确定符号（完全照抄）
    double sign = 1.0;
    if (electrodeType == "anode") {
        sign = -1.0;
    }

    // Line 324-325: 几何贡献（完全照抄）
    Q_analytic = sign / (4.0 * M_PI) * sheetArea *
                 (voltage / Lgap + voltage / Lcell) *
                 CONVERSION_KJMOLNM_AU;

    // Line 327-333: 电解质镜像电荷贡献（完全照抄）
    for (int index : electrolyteAtomIndices) {
        double q_i, sigma, epsilon;
        nonbondedForce->getParticleParameters(index, q_i, sigma, epsilon);
        double z_atom = positions[index][2];  // in nm
        double z_distance = fabs(z_atom - z_opposite);
        // 完全照抄Python的公式
        Q_analytic += (z_distance / Lcell) * (-q_i);
    }

    // Line 335-344: 导体贡献（第一版跳过，教授也标注TODO）
    // TODO: 实现Conductor支持
}
```

---

## 📋 函数2：Scale_charges_analytic()

**Python原文**（Fixed_Voltage_routines.py:354-372）

```python
def Scale_charges_analytic(self, MMsys, print_flag=False):
    # Line 355-356: 计算数值总电荷
    Q_numeric = self.get_total_charge()

    # Line 358-359: 打印（可选）
    if print_flag:
        print("Q_numeric , Q_analytic charges on", self.electrode_type, Q_numeric, self.Q_analytic)

    # Line 361-364: 计算缩放因子，防止除零
    scale_factor = -1
    if abs(Q_numeric) > MMsys.small_threshold:
        scale_factor = self.Q_analytic / Q_numeric

    # Line 366-371: 缩放所有电极电荷
    if scale_factor > 0.0:
        for atom in self.electrode_atoms:
            atom.charge = atom.charge * scale_factor
            MMsys.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, 1.0, 0.0)
```

**C++翻译**（逐行对应）：

```cpp
void ReferenceCalcConstantVKernel::scaleChargesAnalytic(
    const std::vector<int>& electrodeAtomIndices,
    double Q_analytic,
    bool printFlag = false
) {
    // Line 355-356: 计算数值总电荷（完全照抄）
    double Q_numeric = 0.0;
    for (int atomIdx : electrodeAtomIndices) {
        Q_numeric += currentCharges[atomIdx];
    }

    // Line 358-359: 打印（可选）
    if (printFlag) {
        std::cout << "Q_numeric = " << Q_numeric
                  << ", Q_analytic = " << Q_analytic << std::endl;
    }

    // Line 361-364: 计算缩放因子，防止除零（完全照抄）
    double scale_factor = -1.0;
    if (fabs(Q_numeric) > SMALL_THRESHOLD) {
        scale_factor = Q_analytic / Q_numeric;
    }

    // Line 366-371: 缩放所有电极电荷（完全照抄）
    if (scale_factor > 0.0) {
        for (int atomIdx : electrodeAtomIndices) {
            currentCharges[atomIdx] = currentCharges[atomIdx] * scale_factor;
            nonbondedForce->setParticleParameters(
                atomIdx,
                currentCharges[atomIdx],
                1.0,
                0.0
            );
        }
    }
}
```

---

## 📋 函数3：Poisson_solver_fixed_voltage() - 主循环

**Python原文**（MM_classes.py:287-374）

### 阶段0：初始化（287-300行）

```python
def Poisson_solver_fixed_voltage(self, Niterations=3):
    # Line 289-293: QM/MM特殊处理（我们跳过）
    if self.QMMM:
        # ... 跳过 ...

    # Line 295-297: 获取位置
    state = self.simmd.context.getState(getEnergy=False, getForces=False,
                                       getVelocities=False, getPositions=True)
    positions = state.getPositions()

    # Line 298-300: 计算解析总电荷
    self.Cathode.compute_Electrode_charge_analytic(self, positions, self.Conductor_list,
                                                   z_opposite=self.Anode.z_pos)
    self.Anode.compute_Electrode_charge_analytic(self, positions, self.Conductor_list,
                                                 z_opposite=self.Cathode.z_pos)
```

**C++翻译**：

```cpp
double ReferenceCalcConstantVKernel::execute(ContextImpl& context,
                                            bool includeForces, bool includeEnergy) {
    const int nIterations = 4;  // 默认4次（教授用3或4）

    // Line 295-297: 获取位置（完全照抄）
    State state = context.getOwner().getState(State::Positions);
    const vector<Vec3>& positions = state.getPositions();

    // Line 298-300: 计算解析总电荷（完全照抄）
    double Q_analytic_cathode, Q_analytic_anode;
    computeElectrodeChargeAnalytic(
        cathodeAtomIndices, positions, "cathode",
        voltage, Lgap, Lcell, totalArea, z_anode,
        Q_analytic_cathode
    );
    computeElectrodeChargeAnalytic(
        anodeAtomIndices, positions, "anode",
        voltage, Lgap, Lcell, totalArea, z_cathode,
        Q_analytic_anode
    );
```

### 阶段1：SCF迭代主循环（310-365行）

```python
    # Line 310: 开始迭代
    for i_iter in range(Niterations):

        # Line 313-314: 获取力
        state = self.simmd.context.getState(getEnergy=True, getForces=True,
                                           getVelocities=False, getPositions=True)
        forces = state.getForces()

        # Line 321-335: 更新阴极电荷
        for atom in self.Cathode.electrode_atoms:
            index = atom.atom_index
            q_i_old = atom.charge

            # Line 327: 从力计算电场，防止除零
            Ez_external = (forces[index][2]._value / q_i_old) if abs(q_i_old) > (0.9*self.small_threshold) else 0.

            # Line 330: 边界条件求解新电荷
            q_i = 2.0 / (4.0 * numpy.pi) * self.Cathode.area_atom * \
                  (self.Cathode.Voltage / self.Lgap + Ez_external) * conversion_KjmolNm_Au

            # Line 332-333: 防止电荷归零
            if abs(q_i) < self.small_threshold:
                q_i = self.small_threshold  # Cathode为正

            # Line 334-335: 更新
            atom.charge = q_i
            self.nbondedForce.setParticleParameters(index, q_i, 1.0, 0.0)

        # Line 337-350: 更新阳极电荷（几乎相同，符号相反）
        for atom in self.Anode.electrode_atoms:
            index = atom.atom_index
            q_i_old = atom.charge

            # Line 342: 从力计算电场，防止除零
            Ez_external = (forces[index][2]._value / q_i_old) if abs(q_i_old) > (0.9*self.small_threshold) else 0.

            # Line 345: 边界条件（注意：-2.0而不是2.0）
            q_i = -2.0 / (4.0 * numpy.pi) * self.Anode.area_atom * \
                  (self.Anode.Voltage / self.Lgap + Ez_external) * conversion_KjmolNm_Au

            # Line 347-348: 防止电荷归零
            if abs(q_i) < self.small_threshold:
                q_i = -1.0 * self.small_threshold  # Anode为负

            # Line 349-350: 更新
            atom.charge = q_i
            self.nbondedForce.setParticleParameters(index, q_i, 1.0, 0.0)

        # Line 352-360: Conductor处理（第一版跳过）

        # Line 362-363: Green's校正
        self.Scale_charges_analytic_general()

        # Line 365: 更新OpenMM context
        self.nbondedForce.updateParametersInContext(self.simmd.context)

    # Line 367-368: 最后一次打印
    self.Scale_charges_analytic_general(print_flag=True)
```

**C++翻译**（完全照抄）：

```cpp
    // Line 310: 开始SCF迭代（完全照抄）
    for (int iter = 0; iter < nIterations; iter++) {

        // Line 313-314: 获取力（完全照抄）
        State state = context.getOwner().getState(State::Forces | State::Positions);
        const vector<Vec3>& forces = state.getForces();

        // ═══════════════════════════════════════════
        // Line 321-335: 更新阴极电荷（完全照抄）
        // ═══════════════════════════════════════════
        for (size_t i = 0; i < cathodeAtomIndices.size(); i++) {
            int atomIdx = cathodeAtomIndices[i];
            double q_i_old = currentCharges[atomIdx];

            // Line 327: 从力计算电场，防止除零（完全照抄公式）
            double Ez_external = 0.0;
            if (fabs(q_i_old) > (0.9 * SMALL_THRESHOLD)) {
                Ez_external = forces[atomIdx][2] / q_i_old;
            }

            // Line 330: 边界条件求解新电荷（完全照抄公式）
            double q_i = 2.0 / (4.0 * M_PI) * areaPerAtom[i] *
                        (voltage / Lgap + Ez_external) *
                        CONVERSION_KJMOLNM_AU;

            // Line 332-333: 防止电荷归零（完全照抄）
            if (fabs(q_i) < SMALL_THRESHOLD) {
                q_i = SMALL_THRESHOLD;  // Cathode为正
            }

            // Line 334-335: 更新（完全照抄）
            currentCharges[atomIdx] = q_i;
            nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
        }

        // ═══════════════════════════════════════════
        // Line 337-350: 更新阳极电荷（完全照抄）
        // ═══════════════════════════════════════════
        for (size_t i = 0; i < anodeAtomIndices.size(); i++) {
            int atomIdx = anodeAtomIndices[i];
            double q_i_old = currentCharges[atomIdx];

            // Line 342: 从力计算电场，防止除零（完全照抄）
            double Ez_external = 0.0;
            if (fabs(q_i_old) > (0.9 * SMALL_THRESHOLD)) {
                Ez_external = forces[atomIdx][2] / q_i_old;
            }

            // Line 345: 边界条件（注意：-2.0不是2.0，完全照抄）
            double q_i = -2.0 / (4.0 * M_PI) * areaPerAtom[cathodeAtomIndices.size() + i] *
                        (voltage / Lgap + Ez_external) *
                        CONVERSION_KJMOLNM_AU;

            // Line 347-348: 防止电荷归零（完全照抄）
            if (fabs(q_i) < SMALL_THRESHOLD) {
                q_i = -1.0 * SMALL_THRESHOLD;  // Anode为负
            }

            // Line 349-350: 更新（完全照抄）
            currentCharges[atomIdx] = q_i;
            nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
        }

        // Line 352-360: Conductor处理（第一版跳过，教授也是可选）

        // ═══════════════════════════════════════════
        // Line 362-363: Green's校正（完全照抄）
        // ═══════════════════════════════════════════
        scaleChargesAnalytic(cathodeAtomIndices, Q_analytic_cathode, false);
        scaleChargesAnalytic(anodeAtomIndices, Q_analytic_anode, false);

        // Line 365: 更新OpenMM context（完全照抄）
        nonbondedForce->updateParametersInContext(context.getOwner());
    }

    // Line 367-368: 最后一次打印（完全照抄）
    scaleChargesAnalytic(cathodeAtomIndices, Q_analytic_cathode, true);
    scaleChargesAnalytic(anodeAtomIndices, Q_analytic_anode, true);

    return 0.0;  // 不贡献能量
}
```

---

## 📋 关键细节对照

### 1. 除零保护

| 位置 | Python | C++ | 说明 |
|------|--------|-----|------|
| 电场计算 | `if abs(q_i_old) > (0.9*self.small_threshold)` | `if (fabs(q_i_old) > (0.9 * SMALL_THRESHOLD))` | 注意：0.9不是1.0！ |
| 电荷归零 | `if abs(q_i) < self.small_threshold` | `if (fabs(q_i) < SMALL_THRESHOLD)` | 完全照抄 |
| Green's | `if abs(Q_numeric) > MMsys.small_threshold` | `if (fabs(Q_numeric) > SMALL_THRESHOLD)` | 完全照抄 |

### 2. 符号差异

| 电极 | Python | C++ | 说明 |
|------|--------|-----|------|
| 阴极电荷公式 | `2.0 / (4.0 * numpy.pi)` | `2.0 / (4.0 * M_PI)` | 正号 |
| 阳极电荷公式 | `-2.0 / (4.0 * numpy.pi)` | `-2.0 / (4.0 * M_PI)` | 负号 |
| 阴极threshold | `self.small_threshold` | `SMALL_THRESHOLD` | 正号 |
| 阳极threshold | `-1.0 * self.small_threshold` | `-1.0 * SMALL_THRESHOLD` | 负号 |

### 3. 数组索引

| Python | C++ | 说明 |
|--------|-----|------|
| `forces[index][2]._value` | `forces[atomIdx][2]` | OpenMM的Vec3，z分量 |
| `positions[index][2]._value` | `positions[atomIdx][2]` | OpenMM的Vec3，z分量 |

### 4. 循环顺序（绝对不能改！）

```
1. getState(getForces=True)
2. 循环阴极原子
   - 计算Ez
   - 计算q_new
   - setParticleParameters
3. 循环阳极原子
   - 计算Ez
   - 计算q_new
   - setParticleParameters
4. scaleChargesAnalytic(阴极)
5. scaleChargesAnalytic(阳极)
6. updateParametersInContext
```

**绝对不能**：
- ❌ 先scale再setParticleParameters
- ❌ 在循环内部updateParametersInContext
- ❌ 改变阴极/阳极的处理顺序

---

## 📋 数据成员映射

### Python类成员 → C++类成员

| Python (`MM` class) | C++ (Kernel) | 类型 |
|---------------------|--------------|------|
| `self.Cathode.electrode_atoms` | `cathodeAtomIndices` | `std::vector<int>` |
| `self.Anode.electrode_atoms` | `anodeAtomIndices` | `std::vector<int>` |
| `self.Cathode.area_atom` | `areaPerAtom[i]` | `std::vector<double>` |
| `self.Cathode.Voltage` | `voltage` | `double` |
| `self.Lgap` | `Lgap` | `double` |
| `self.Lcell` | `Lcell` | `double` |
| `self.Cathode.sheet_area` | `totalArea` | `double` |
| `self.small_threshold` | `SMALL_THRESHOLD` | `constexpr double` |
| `self.nbondedForce` | `nonbondedForce` | `NonbondedForce*` |
| `atom.charge` | `currentCharges[atomIdx]` | `std::vector<double>` |

---

## 📋 函数调用映射

| Python | C++ | 说明 |
|--------|-----|------|
| `state.getForces()` | `state.getForces()` | 相同 |
| `state.getPositions()` | `state.getPositions()` | 相同 |
| `nbondedForce.getParticleParameters(index)` | `nonbondedForce->getParticleParameters(index, q, s, e)` | C++用引用返回 |
| `nbondedForce.setParticleParameters(index, q, 1.0, 0.0)` | `nonbondedForce->setParticleParameters(index, q, 1.0, 0.0)` | 相同 |
| `nbondedForce.updateParametersInContext(context)` | `nonbondedForce->updateParametersInContext(context.getOwner())` | 注意getOwner() |

---

## 🎯 翻译检查清单

在翻译时，每个函数必须检查：

- [ ] 数值常数是否完全一致？
- [ ] Threshold检查是否完全一致（包括0.9系数）？
- [ ] 循环顺序是否完全一致？
- [ ] 符号（+/-）是否完全一致？
- [ ] 除法/乘法顺序是否完全一致？
- [ ] if条件是否完全一致（>还是>=）？
- [ ] 数组索引是否正确（[2]是z分量）？
- [ ] 函数调用参数顺序是否正确？

---

## 💡 翻译原则

1. **完全照抄**：公式、常数、threshold一个都不改
2. **保留注释**：教授的注释全部保留，翻译成英文
3. **不优化**：即使看到重复代码也不合并
4. **逐行对应**：每个C++语句都应该能找到对应的Python行号

**翻译时最大的敌人是"聪明"！**
