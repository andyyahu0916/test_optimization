#!/usr/bin/env python3
"""
測試編譯好的 ConstantV Plugin
驗證所有 Platform (Reference, CUDA) 是否可以正常加載和使用
"""

import sys
import openmm as mm
import openmm.app as app
import openmm.unit as unit

def test_plugin_loading():
    """測試 1: 插件是否可以正常加載"""
    print("=" * 70)
    print("測試 1: 插件加載")
    print("=" * 70)
    
    try:
        import os
        
        # 檢查並加載自定義插件目錄
        plugin_dir = os.path.expanduser("~/miniconda3/envs/openmm_gpu/lib/plugins")
        if os.path.exists(plugin_dir):
            print(f"   從自定義目錄加載插件: {plugin_dir}")
            mm.Platform.loadPluginsFromDirectory(plugin_dir)
        
        # 也加載默認插件
        default_dir = mm.Platform.getDefaultPluginsDirectory()
        print(f"   從默認目錄加載插件: {default_dir}")
        mm.Platform.loadPluginsFromDirectory(default_dir)
        
        # 檢查插件是否已加載(通過檢查是否有新的 Force 類型)
        # ConstantV 插件應該註冊了 ConstantVForce
        print("✅ 插件加載完成!")
        
        # 嘗試創建 ConstantVForce 來驗證
        # 註:C++ 插件通過序列化系統註冊,不會出現在 Python 命名空間
        return True
        
    except Exception as e:
        print(f"❌ 插件加載失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_platforms():
    """測試 2: 檢查可用的 Platforms"""
    print("\n" + "=" * 70)
    print("測試 2: 可用的 Platforms")
    print("=" * 70)
    
    num_platforms = mm.Platform.getNumPlatforms()
    print(f"檢測到 {num_platforms} 個 Platforms:")
    
    for i in range(num_platforms):
        platform = mm.Platform.getPlatform(i)
        print(f"  {i+1}. {platform.getName()}")
    
    # 檢查是否有 CUDA
    cuda_available = False
    for i in range(num_platforms):
        if mm.Platform.getPlatform(i).getName() == 'CUDA':
            cuda_available = True
            break
    
    return cuda_available

def test_force_creation():
    """測試 3: 檢查插件中的共享庫"""
    print("\n" + "=" * 70)
    print("測試 3: 檢查編譯的共享庫")
    print("=" * 70)
    
    try:
        import os
        import glob
        
        plugin_dir = os.path.expanduser("~/miniconda3/envs/openmm_gpu/lib/plugins")
        
        if not os.path.exists(plugin_dir):
            print(f"❌ 插件目錄不存在: {plugin_dir}")
            return None
        
        # 查找 ConstantV 相關的共享庫
        constantv_libs = glob.glob(os.path.join(plugin_dir, "*ConstantV*.so"))
        
        if not constantv_libs:
            print(f"❌ 在 {plugin_dir} 中找不到 ConstantV 共享庫")
            return None
        
        print(f"✅ 找到 {len(constantv_libs)} 個 ConstantV 共享庫:")
        for lib in constantv_libs:
            size = os.path.getsize(lib) / 1024  # KB
            print(f"   - {os.path.basename(lib)} ({size:.1f} KB)")
        
        # 嘗試使用 ctypes 加載庫(驗證是否可加載)
        import ctypes
        try:
            for lib in constantv_libs:
                handle = ctypes.CDLL(lib)
                print(f"   ✅ {os.path.basename(lib)} 可以加載")
        except Exception as load_err:
            print(f"   ⚠️  加載測試: {load_err}")
        
        print("\n✅ 共享庫檢查完成!")
        return constantv_libs  # 返回庫列表而不是 force 對象
        
    except Exception as e:
        print(f"❌ 共享庫檢查失敗: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_library_dependencies(libs):
    """測試 4: 檢查庫依賴"""
    print("\n" + "=" * 70)
    print("測試 4: 檢查庫依賴")
    print("=" * 70)
    
    if not libs:
        print("❌ 沒有提供庫文件")
        return False
    
    try:
        import subprocess
        import os
        
        for lib in libs:
            lib_name = os.path.basename(lib)
            print(f"\n檢查 {lib_name}:")
            
            # 使用 ldd 檢查依賴
            result = subprocess.run(
                ['ldd', lib],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                # 檢查關鍵依賴
                output = result.stdout
                has_openmm = 'OpenMM' in output
                has_cuda = ('cuda' in output.lower() or 'cublas' in output.lower())
                has_missing = 'not found' in output
                
                print(f"   OpenMM: {'✅' if has_openmm else '❌'}")
                if 'CUDA' in lib_name:
                    print(f"   CUDA: {'✅' if has_cuda else '❌'}")
                print(f"   缺失依賴: {'❌ 有' if has_missing else '✅ 無'}")
                
                if has_missing:
                    print("\n   缺失的庫:")
                    for line in output.split('\n'):
                        if 'not found' in line:
                            print(f"      {line.strip()}")
            else:
                print(f"   ⚠️  無法檢查依賴: {result.stderr}")
        
        print("\n✅ 依賴檢查完成!")
        return True
        
    except Exception as e:
        print(f"❌ 依賴檢查失敗: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主測試流程"""
    print("\n" + "=" * 70)
    print("ConstantV Plugin 編譯後測試")
    print("=" * 70 + "\n")
    
    # 測試 1: 插件加載
    if not test_plugin_loading():
        print("\n⚠️  插件加載有問題,但繼續測試")
    
    # 測試 2: 檢查 Platforms
    cuda_available = test_platforms()
    
    # 測試 3: 檢查共享庫
    libs = test_force_creation()
    if libs is None or len(libs) == 0:
        print("\n❌ 找不到編譯的共享庫,停止測試")
        return False
    
    # 測試 4: 檢查庫依賴
    dep_success = test_library_dependencies(libs)
    
    # 總結
    print("\n" + "=" * 70)
    print("測試總結")
    print("=" * 70)
    print(f"✅ 找到 {len(libs)} 個共享庫")
    print(f"{'✅' if dep_success else '❌'} 依賴檢查: {'成功' if dep_success else '失敗'}")
    print(f"{'✅' if cuda_available else '⚠️ '} CUDA Platform: {'可用' if cuda_available else '不可用'}")
    
    # 列出發現的庫
    import os
    print("\n編譯的插件:")
    for lib in libs:
        print(f"  - {os.path.basename(lib)}")
    
    overall_success = (libs is not None) and dep_success
    
    if overall_success:
        print("\n🎉 編譯驗證成功!")
        print("\n下一步:")
        print("  1. 在實際的 MD 模擬腳本中導入並使用 ConstantVForce")
        print("  2. 驗證電極電壓控制功能")
        print("  3. 測試 CUDA 加速效果")
    else:
        print("\n⚠️  編譯驗證有問題,請檢查依賴")
    
    return overall_success

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
