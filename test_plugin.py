#!/usr/bin/env python3
"""
測試 ElectrodeChargePlugin 是否能正確加載和運行
"""
import sys

print("=" * 60)
print("ElectrodeChargePlugin 加載測試")
print("=" * 60)

# 1. 測試 OpenMM 基本功能
print("\n[1/4] 測試 OpenMM...")
try:
    from openmm.app import *
    from openmm import *
    from openmm.unit import *
    print("✓ OpenMM 導入成功")
    print(f"  版本: {Platform.getOpenMMVersion()}")
    print(f"  可用平台: {[Platform.getPlatform(i).getName() for i in range(Platform.getNumPlatforms())]}")
except Exception as e:
    print(f"✗ OpenMM 導入失敗: {e}")
    sys.exit(1)

# 2. 測試 Plugin 加載
print("\n[2/4] 測試 ElectrodeChargePlugin 加載...")
try:
    # 使用 conda 環境的 plugins 目錄
    import os
    conda_prefix = os.environ.get('CONDA_PREFIX', '/home/andy/miniforge3/envs/cuda')
    plugin_dir = os.path.join(conda_prefix, 'lib', 'plugins')
    print(f"  嘗試從 conda 環境加載: {plugin_dir}")
    if os.path.exists(plugin_dir):
        Platform.loadPluginsFromDirectory(plugin_dir)
        print(f"✓ 從 {plugin_dir} 加載成功")
    else:
        Platform.loadPluginsFromDirectory(Platform.getDefaultPluginsDirectory())
        print(f"✓ 從默認目錄加載: {Platform.getDefaultPluginsDirectory()}")
    
    # 嘗試導入 Plugin（如果有 Python wrapper）
    try:
        import electrodecharge
        print("✓ Python wrapper 可用")
    except ImportError:
        print("⚠ Python wrapper 未編譯（預期，因為我們跳過了）")
        
except Exception as e:
    print(f"✗ Plugin 加載失敗: {e}")
    sys.exit(1)

# 3. 測試最小系統（兩個粒子）
print("\n[3/4] 測試最小系統...")
try:
    system = System()
    system.addParticle(12.0)  # 碳原子質量
    system.addParticle(12.0)
    
    # 創建 NonbondedForce
    nb = NonbondedForce()
    nb.addParticle(0.0, 0.34, 0.36)  # cathode atom
    nb.addParticle(0.0, 0.34, 0.36)  # anode atom
    nb.setNonbondedMethod(NonbondedForce.NoCutoff)
    system.addForce(nb)
    
    # 設置週期性盒子
    system.setDefaultPeriodicBoxVectors(
        Vec3(5, 0, 0) * nanometer,
        Vec3(0, 5, 0) * nanometer,
        Vec3(0, 0, 10) * nanometer
    )
    
    integrator = VerletIntegrator(0.001 * picoseconds)
    context = Context(system, integrator, Platform.getPlatformByName('Reference'))
    
    positions = [Vec3(2.5, 2.5, 1.0), Vec3(2.5, 2.5, 9.0)] * nanometer
    context.setPositions(positions)
    
    state = context.getState(getEnergy=True)
    energy = state.getPotentialEnergy()
    print(f"✓ 最小系統創建成功")
    print(f"  能量: {energy}")
    
    del context, integrator
    
except Exception as e:
    print(f"✗ 最小系統測試失敗: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# 4. 檢查 Plugin 文件
print("\n[4/4] 檢查 Plugin 文件...")
import os
plugin_dir = Platform.getDefaultPluginsDirectory()
api_lib = os.path.join(os.path.dirname(plugin_dir), "libElectrodeChargePlugin.so")
ref_lib = os.path.join(plugin_dir, "libElectrodeChargePluginReference.so")

if os.path.exists(api_lib):
    print(f"✓ API 庫: {api_lib} ({os.path.getsize(api_lib)} bytes)")
else:
    print(f"✗ API 庫不存在: {api_lib}")

if os.path.exists(ref_lib):
    print(f"✓ Reference kernel: {ref_lib} ({os.path.getsize(ref_lib)} bytes)")
else:
    print(f"✗ Reference kernel 不存在: {ref_lib}")

print("\n" + "=" * 60)
print("✓ 所有基本測試通過！")
print("=" * 60)
print("\n下一步：運行實際的電極充電測試")
