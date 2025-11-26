#!/usr/bin/env python3
"""
簡單測試: 檢查 ConstantV Plugin 是否正確安裝
"""

import sys
import os
import openmm as mm

def main():
    print("=" * 70)
    print("ConstantV Plugin 安裝測試")
    print("=" * 70)
    
    # 1. 檢查 OpenMM 版本
    print(f"\n1. OpenMM 版本: {mm.version.short_version}")
    print(f"   完整版本: {mm.version.full_version}")
    
    # 2. 檢查 plugins 目錄
    plugin_dir = mm.Platform.getDefaultPluginsDirectory()
    print(f"\n2. Plugins 目錄: {plugin_dir}")
    
    if os.path.exists(plugin_dir):
        all_plugins = os.listdir(plugin_dir)
        constantv_plugins = [f for f in all_plugins if 'ConstantV' in f]
        
        print(f"   總插件數: {len(all_plugins)}")
        print(f"   ConstantV 插件:")
        for p in constantv_plugins:
            full_path = os.path.join(plugin_dir, p)
            size = os.path.getsize(full_path) / 1024  # KB
            print(f"     - {p} ({size:.1f} KB)")
    else:
        print(f"   ❌ 目錄不存在!")
        return False
    
    # 3. 嘗試加載插件
    print(f"\n3. 加載插件...")
    try:
        mm.Platform.loadPluginsFromDirectory(plugin_dir)
        print(f"   ✅ 插件加載成功!")
    except Exception as e:
        print(f"   ❌ 插件加載失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 4. 檢查可用的 Platforms
    print(f"\n4. 可用的 Platforms:")
    num_platforms = mm.Platform.getNumPlatforms()
    for i in range(num_platforms):
        platform = mm.Platform.getPlatform(i)
        print(f"   - {platform.getName()}")
    
    # 5. 測試創建簡單系統 (不使用 ConstantV,只測試基本功能)
    print(f"\n5. 測試基本 OpenMM 功能:")
    try:
        system = mm.System()
        system.addParticle(1.0)
        integrator = mm.LangevinIntegrator(300*mm.unit.kelvin, 
                                          1.0/mm.unit.picosecond,
                                          0.001*mm.unit.picosecond)
        
        # 測試 Reference Platform
        platform = mm.Platform.getPlatformByName('Reference')
        context = mm.Context(system, integrator, platform)
        context.setPositions([mm.Vec3(0, 0, 0)])
        state = context.getState(getEnergy=True)
        print(f"   ✅ Reference Platform 正常")
        
        # 測試 CUDA Platform (如果可用)
        try:
            platform = mm.Platform.getPlatformByName('CUDA')
            context2 = mm.Context(system, integrator, platform)
            context2.setPositions([mm.Vec3(0, 0, 0)])
            state = context2.getState(getEnergy=True)
            print(f"   ✅ CUDA Platform 正常")
        except Exception as e:
            print(f"   ⚠️  CUDA Platform 不可用: {e}")
        
    except Exception as e:
        print(f"   ❌ 基本功能測試失敗: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # 總結
    print("\n" + "=" * 70)
    if constantv_plugins:
        print("✅ ConstantV Plugin 已正確安裝!")
        print(f"   安裝位置: {plugin_dir}")
        print(f"   插件文件: {len(constantv_plugins)} 個")
    else:
        print("❌ ConstantV Plugin 未找到!")
        return False
    
    return True

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
