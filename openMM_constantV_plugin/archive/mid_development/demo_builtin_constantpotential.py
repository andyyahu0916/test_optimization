#!/usr/bin/env python3
"""
演示如何使用 OpenMM 內建的 ConstantPotentialForce
這是正確的 PME 實現,取代我們自己的插件
"""

import openmm as mm
import openmm.app as app
import openmm.unit as unit
import numpy as np

def demo_constantpotential_force():
    """演示 ConstantPotentialForce 的使用"""
    
    print("=" * 70)
    print("OpenMM ConstantPotentialForce 演示")
    print("=" * 70)
    
    # 1. 創建簡單系統
    print("\n1. 創建系統:")
    print("   - 6 個石墨烯碳原子 (模擬電極)")
    print("   - 2 個離子 (電解液)")
    
    system = mm.System()
    
    # 添加 6 個碳原子 (電極) + 2 個離子 (電解液)
    for i in range(8):
        system.addParticle(12.0 * unit.amu if i < 6 else 23.0 * unit.amu)
    
    # 設置周期性邊界條件
    box_size = 2.0 * unit.nanometer
    system.setDefaultPeriodicBoxVectors(
        mm.Vec3(box_size, 0, 0),
        mm.Vec3(0, box_size, 0),
        mm.Vec3(0, 0, box_size)
    )
    
    # 2. 創建 ConstantPotentialForce
    print("\n2. 創建 ConstantPotentialForce:")
    force = mm.ConstantPotentialForce()
    
    # 添加所有粒子的電荷 (初始值)
    # 電極粒子 (0-5): 初始電荷 0 (會被求解)
    # 電解液 (6-7): 固定電荷
    for i in range(6):
        force.addParticle(0.0)  # 電極原子,初始電荷
    force.addParticle(+1.0)  # Na+ 離子
    force.addParticle(-1.0)  # Cl- 離子
    
    print("   ✅ 添加了 8 個粒子")
    
    # 3. 定義電極
    print("\n3. 定義電極:")
    
    # 電極 A: 前 3 個碳原子 (左側)
    electrode_a = set([0, 1, 2])
    potential_a = 1.0 * 96.485  # 1.0 V = 96.485 kJ/mol/e
    gaussian_width = 0.05  # 0.05 nm
    thomas_fermi = 0.0  # 不使用 TF 模型
    
    idx_a = force.addElectrode(electrode_a, potential_a, gaussian_width, thomas_fermi)
    print(f"   電極 A (索引 {idx_a}):")
    print(f"      粒子: {electrode_a}")
    print(f"      電壓: {potential_a / 96.485:.2f} V")
    print(f"      Gaussian 寬度: {gaussian_width} nm")
    
    # 電極 B: 後 3 個碳原子 (右側) - 接地
    electrode_b = set([3, 4, 5])
    potential_b = 0.0  # 接地
    
    idx_b = force.addElectrode(electrode_b, potential_b, gaussian_width, thomas_fermi)
    print(f"   電極 B (索引 {idx_b}):")
    print(f"      粒子: {electrode_b}")
    print(f"      電壓: {potential_b / 96.485:.2f} V (接地)")
    
    # 4. 設置 PME 參數
    print("\n4. 設置 PME 參數:")
    force.setCutoffDistance(1.0 * unit.nanometer)
    force.setEwaldErrorTolerance(1e-4)
    
    # 可以手動設置 PME 參數 (或讓 OpenMM 自動選擇)
    # force.setPMEParameters(alpha, nx, ny, nz)
    
    print("   截斷距離: 1.0 nm")
    print("   Ewald 誤差容忍度: 1e-4")
    print("   ✅ PME 將自動處理周期性邊界條件")
    
    # 5. 選擇求解方法
    print("\n5. 選擇求解方法:")
    
    # CG 方法: 適用於動態系統 (電極可以移動)
    force.setConstantPotentialMethod(mm.ConstantPotentialForce.CG)
    force.setCGErrorTolerance(0.1)  # kJ/mol/e
    print("   方法: Conjugate Gradient (CG)")
    print("   CG 誤差容忍度: 0.1 kJ/mol/e")
    
    # 或使用 Matrix 方法: 僅適用於固定電極
    # force.setConstantPotentialMethod(mm.ConstantPotentialForce.Matrix)
    # print("   方法: Capacitance Matrix (更快,但電極必須固定)")
    
    # 6. 添加 Force 到 System
    system.addForce(force)
    print("\n6. Force 已添加到 System")
    
    # 7. 創建 Context 並測試
    print("\n7. 創建 Context 並運行測試:")
    
    integrator = mm.LangevinIntegrator(
        300 * unit.kelvin,
        1.0 / unit.picosecond,
        0.001 * unit.picosecond
    )
    
    try:
        platform = mm.Platform.getPlatformByName('CUDA')
        properties = {'CudaPrecision': 'mixed'}
        print("   使用 CUDA Platform")
    except:
        platform = mm.Platform.getPlatformByName('Reference')
        properties = {}
        print("   使用 Reference Platform")
    
    context = mm.Context(system, integrator, platform, properties)
    
    # 設置初始位置
    positions = []
    # 電極 A (左側)
    for i in range(3):
        positions.append(mm.Vec3(0.3, 0.5 + i*0.2, 1.0) * unit.nanometer)
    # 電極 B (右側)
    for i in range(3):
        positions.append(mm.Vec3(1.7, 0.5 + i*0.2, 1.0) * unit.nanometer)
    # 離子 (中間)
    positions.append(mm.Vec3(0.8, 1.0, 1.0) * unit.nanometer)  # Na+
    positions.append(mm.Vec3(1.2, 1.0, 1.0) * unit.nanometer)  # Cl-
    
    context.setPositions(positions)
    
    # 計算能量
    state = context.getState(getEnergy=True, getForces=True)
    energy = state.getPotentialEnergy()
    forces = state.getForces()
    
    print(f"   ✅ 初始能量: {energy}")
    print(f"   ✅ 力的數量: {len(forces)}")
    
    # 獲取實際的電荷分布 (包括求解的電極電荷)
    charges = force.getCharges(context)
    print(f"\n8. 求解的電荷分布:")
    for i, q in enumerate(charges):
        particle_type = "電極A" if i < 3 else ("電極B" if i < 6 else ("Na+" if i == 6 else "Cl-"))
        print(f"   粒子 {i} ({particle_type}): {q.value_in_unit(unit.elementary_charge):+.6f} e")
    
    # 運行幾步模擬
    print(f"\n9. 運行 10 步模擬...")
    integrator.step(10)
    
    state = context.getState(getEnergy=True)
    energy_after = state.getPotentialEnergy()
    print(f"   10 步後能量: {energy_after}")
    
    # 再次獲取電荷
    charges_after = force.getCharges(context)
    print(f"\n10. 10 步後的電荷分布:")
    for i, q in enumerate(charges_after):
        particle_type = "電極A" if i < 3 else ("電極B" if i < 6 else ("Na+" if i == 6 else "Cl-"))
        q_val = q.value_in_unit(unit.elementary_charge)
        q_init = charges[i].value_in_unit(unit.elementary_charge)
        charge_change = q_val - q_init
        print(f"   粒子 {i} ({particle_type}): {q_val:+.6f} e (變化: {charge_change:+.6f})")
    
    print("\n" + "=" * 70)
    print("✅ ConstantPotentialForce 演示完成!")
    print("=" * 70)
    print("\n關鍵優勢:")
    print("  1. ✅ 使用 PME - 正確處理周期性電靜力")
    print("  2. ✅ 自動求解電極電荷")
    print("  3. ✅ 支持動態模擬 (CG 方法)")
    print("  4. ✅ 內建於 OpenMM - 無需自定義插件")
    print("  5. ✅ 經過充分測試和優化")
    
    return True

if __name__ == '__main__':
    try:
        demo_constantpotential_force()
    except Exception as e:
        print(f"\n❌ 錯誤: {e}")
        import traceback
        traceback.print_exc()
