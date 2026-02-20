# 修复 SWIG 绑定问题

## 问题

`ConstantVoltageForce` 无法添加到 `System`，因为 SWIG 生成的 Python 类没有正确继承 OpenMM 的 `Force` 基类。

## 解决方案

需要重新编译 Python 绑定以包含 `asForce()` 方法。

### 方法 1: 使用 CMake 重新编译（推荐）

```bash
cd /home/andy/test_optimization/openmm-8.4.0/build
# 删除旧的 Python 绑定
rm -f plugins/constantvoltage/python/_constantvoltage.so
rm -f plugins/constantvoltage/python/constantvoltage.py

# 重新配置（如果需要）
cmake ..

# 重新编译 Python 绑定
# 注意：Python 绑定可能需要通过 setup.py 编译
cd ../plugins/constantvoltage/python
python setup.py build_ext --inplace
```

### 方法 2: 手动编译 SWIG 绑定

```bash
cd /home/andy/test_optimization/openmm-8.4.0/plugins/constantvoltage/python

# 确保 SWIG 已安装
which swig

# 生成 SWIG 包装代码
swig -python -c++ -I../../openmmapi/include -I../../../openmmapi/include constantvoltage.i

# 编译
g++ -shared -fPIC constantvoltage_wrap.cxx -o _constantvoltage.so \
    -I/usr/include/python3.13 \
    -I../../openmmapi/include \
    -I../../../openmmapi/include \
    -L/home/andy/miniforge3/envs/cuda/lib \
    -lOpenMM -lOpenMMConstantVoltage \
    $(python3-config --ldflags)
```

### 临时解决方案

在重新编译之前，可以使用以下 workaround（已在 `run_openMM_plugin.py` 中实现）：

```python
# 使用 asForce() 方法（如果可用）
if hasattr(cv_force, 'asForce'):
    force_ptr = cv_force.asForce()
    system.addForce(force_ptr)
```

## 验证

重新编译后，验证 `asForce()` 方法是否存在：

```python
import sys
sys.path.insert(0, 'plugins/constantvoltage/python')
import constantvoltage
cv = constantvoltage.ConstantVoltageForce()
print('Has asForce:', hasattr(cv, 'asForce'))  # 应该输出 True
```

