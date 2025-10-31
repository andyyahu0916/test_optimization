# Linus式简化计划

## 要删除的过度设计

1. **should_use_warmstart()函数** - 复杂的warmstart逻辑
   → warmstart只在cython版本支持，其他版本不需要

2. **run_python_validation_step()函数** - A/B验证逻辑
   → 如果要验证就跑两次手动对比log，不需要自动化

3. **logging_mode选择** - efficient vs legacy_print
   → 一律用print输出，简单直接

4. **[Validation] section** - 整个删除
   → enable, interval, tol_charge, tol_energy_rel全删

5. **[Physics] section** - 整个删除
   → iterations=4, enforce_analytic_scaling_each_iter=True硬编码

6. **warmstart相关所有配置** - 删除
   → enable_warmstart, verify_interval, warmstart_after_ns, warmstart_after_frames

7. **write_charges, write_components** - 删除
   → 不需要写文件，print就够了

## 保留的核心功能

1. mm_version选择（original/optimized/cython/plugin）
2. 基本simulation参数（time, freq, voltage）
3. platform选择
4. Files和Electrodes配置
5. 核心模拟循环

## 简化原则

- Print直接输出，不写log文件
- Poisson参数硬编码（iterations=4）
- 删除所有fallback逻辑
- 如果99%情况用默认值，就硬编码
