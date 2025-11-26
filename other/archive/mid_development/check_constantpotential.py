#!/usr/bin/env python3
"""檢查 OpenMM 是否有 ConstantPotentialForce"""

import openmm as mm

print(f"OpenMM 版本: {mm.version.full_version}")
print(f"有 ConstantPotentialForce: {hasattr(mm, 'ConstantPotentialForce')}")

if hasattr(mm, 'ConstantPotentialForce'):
    print("\n✅ OpenMM 已經內建 ConstantPotentialForce!")
    print("   這是一個完整的 PME 電極電壓控制實現")
    print("   API:")
    print("   - addElectrode(particles, potential, gaussianWidth, thomasFermiScale)")
    print("   - setConstantPotentialMethod(CG or Matrix)")
    print("   - setPMEParameters(alpha, nx, ny, nz)")
    print("\n建議:")
    print("   1. 使用 OpenMM 內建的 ConstantPotentialForce")
    print("   2. 我們的插件可能不需要了!")
else:
    print("\n❌ OpenMM 沒有 ConstantPotentialForce")
    print("   需要自己實現")
