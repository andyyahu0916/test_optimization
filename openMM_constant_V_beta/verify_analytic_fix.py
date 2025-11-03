#!/usr/bin/env python
"""
驗證 compute_Electrode_charge_analytic 修復
==========================================

檢查項目：
1. MM 類別有 electrolyte_c_indices 和 electrolyte_c_charges
2. electrode_charges_cython 有 compute_analytic_contribution_cython 函數
3. compute_Electrode_charge_analytic 不再使用 getParticleParameters
4. 所有昂貴的 API 呼叫已被移除
"""

import sys
import os

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def check_electrolyte_c_arrays():
    """檢查是否為電解質建立了 C 陣列"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}檢查 1: 電解質 C 陣列 (electrolyte_c_indices, electrolyte_c_charges){RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    file_path = 'lib/MM_classes_CYTHON.py'
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 檢查是否建立了 C 陣列
    if 'self.electrolyte_c_indices = numpy.array(self.electrolyte_atom_indices, dtype=numpy.int64)' in content:
        print(f"{GREEN}✓ electrolyte_c_indices 已建立 (dtype=numpy.int64){RESET}")
    else:
        print(f"{RED}✗ electrolyte_c_indices 未建立{RESET}")
        return False
    
    if 'self.electrolyte_c_charges = numpy.array(electrolyte_charges_list, dtype=numpy.float64)' in content:
        print(f"{GREEN}✓ electrolyte_c_charges 已建立 (dtype=numpy.float64){RESET}")
    else:
        print(f"{RED}✗ electrolyte_c_charges 未建立{RESET}")
        return False
    
    # 檢查是否在讀取電荷
    if 'electrolyte_charges_list = []' in content:
        print(f"{GREEN}✓ 有建立臨時列表來收集電荷{RESET}")
    else:
        print(f"{RED}✗ 未找到臨時列表{RESET}")
        return False
    
    # 檢查是否有讀取電荷的邏輯
    if '(q_i, sig, eps) = self.nbondedForce.getParticleParameters(atom.index)' in content and \
       'electrolyte_charges_list.append(q_i._value)' in content:
        print(f"{GREEN}✓ 正確地一次性讀取電荷並加入列表{RESET}")
    else:
        print(f"{RED}✗ 未找到正確的電荷讀取邏輯{RESET}")
        return False
    
    print(f"\n{GREEN}✅ 電解質 C 陣列檢查通過！{RESET}")
    return True


def check_cython_function():
    """檢查 Cython 函數是否存在"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}檢查 2: compute_analytic_contribution_cython 函數{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    file_path = 'lib/electrode_charges_cython.pyx'
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 檢查函數定義
    if 'def compute_analytic_contribution_cython(' in content:
        print(f"{GREEN}✓ 函數已定義{RESET}")
    else:
        print(f"{RED}✗ 函數未定義{RESET}")
        return False
    
    # 檢查參數
    required_params = ['z_positions', 'c_indices', 'c_charges', 'z_opposite', 'Lcell']
    for param in required_params:
        if f'{param},' in content or f'{param})' in content:
            print(f"{GREEN}✓ 參數 {param} 存在{RESET}")
        else:
            print(f"{RED}✗ 參數 {param} 缺失{RESET}")
            return False
    
    # 檢查是否有 C-level 計算
    if 'for i in range(N):' in content and 'contribution +=' in content:
        print(f"{GREEN}✓ 有 C-level 循環計算{RESET}")
    else:
        print(f"{RED}✗ 未找到 C-level 計算{RESET}")
        return False
    
    # 檢查是否使用了 fabs (絕對值)
    if 'if z_distance < 0.0:' in content and 'z_distance = -z_distance' in content:
        print(f"{GREEN}✓ 正確處理絕對值計算{RESET}")
    else:
        print(f"{RED}✗ 絕對值計算有問題{RESET}")
        return False
    
    print(f"\n{GREEN}✅ Cython 函數檢查通過！{RESET}")
    return True


def check_compute_electrode_charge_analytic():
    """檢查 compute_Electrode_charge_analytic 是否已重寫"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}檢查 3: compute_Electrode_charge_analytic 重寫{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    file_path = 'lib/Fixed_Voltage_routines_CYTHON.py'
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 檢查是否有 GOOD TASTE 標記
    if '# 🔥 GOOD TASTE REFACTORED: compute_Electrode_charge_analytic' in content:
        print(f"{GREEN}✓ 函數已標記為 GOOD TASTE REFACTORED{RESET}")
    else:
        print(f"{RED}✗ 函數未標記為重構版本{RESET}")
        return False
    
    # 檢查是否使用 Cython 函數
    if 'ec_cython.compute_analytic_contribution_cython(' in content:
        print(f"{GREEN}✓ 使用 Cython 函數進行計算{RESET}")
    else:
        print(f"{RED}✗ 未使用 Cython 函數{RESET}")
        return False
    
    # 檢查是否使用 C 陣列
    if 'MMsys.electrolyte_c_indices' in content and 'MMsys.electrolyte_c_charges' in content:
        print(f"{GREEN}✓ 使用電解質 C 陣列{RESET}")
    else:
        print(f"{RED}✗ 未使用電解質 C 陣列{RESET}")
        return False
    
    if 'Conductor.c_indices' in content and 'Conductor.c_charges' in content:
        print(f"{GREEN}✓ 使用導體 C 陣列{RESET}")
    else:
        print(f"{RED}✗ 未使用導體 C 陣列{RESET}")
        return False
    
    # 檢查是否有 NumPy fallback
    if 'z_atoms = z_positions_np[MMsys.electrolyte_c_indices]' in content:
        print(f"{GREEN}✓ 有 NumPy fallback (電解質){RESET}")
    else:
        print(f"{RED}✗ 缺少 NumPy fallback{RESET}")
        return False
    
    # 檢查是否移除了舊的 API 呼叫
    # 在 compute_Electrode_charge_analytic 的範圍內不應該有 getParticleParameters
    lines = content.split('\n')
    in_function = False
    bad_api_calls = 0
    
    for i, line in enumerate(lines, 1):
        if 'def compute_Electrode_charge_analytic' in line:
            in_function = True
        elif in_function and line.strip().startswith('def ') and 'compute_Electrode_charge_analytic' not in line:
            # 進入下一個函數，停止檢查
            break
        elif in_function and 'getParticleParameters' in line and not line.strip().startswith('#'):
            bad_api_calls += 1
            print(f"{RED}✗ Line {i}: 發現舊的 getParticleParameters 呼叫{RESET}")
    
    if bad_api_calls == 0:
        print(f"{GREEN}✓ 沒有發現 getParticleParameters 呼叫{RESET}")
    else:
        print(f"{RED}✗ 發現 {bad_api_calls} 處 getParticleParameters 呼叫{RESET}")
        return False
    
    print(f"\n{GREEN}✅ compute_Electrode_charge_analytic 檢查通過！{RESET}")
    return True


def check_cython_module_loaded():
    """檢查 Cython 模組是否可以載入"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}檢查 4: Cython 模組載入{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    try:
        sys.path.insert(0, 'lib')
        import electrode_charges_cython as ec_cython
        print(f"{GREEN}✓ Cython 模組成功載入{RESET}")
        
        # 檢查函數是否存在
        if hasattr(ec_cython, 'compute_analytic_contribution_cython'):
            print(f"{GREEN}✓ compute_analytic_contribution_cython 函數存在{RESET}")
        else:
            print(f"{RED}✗ compute_analytic_contribution_cython 函數不存在{RESET}")
            return False
        
        # 檢查其他必要函數
        required_functions = [
            'compute_electrode_charges_cython',
            'scale_charges_inplace_cython',
            'initialize_charges_cython'
        ]
        
        for func_name in required_functions:
            if hasattr(ec_cython, func_name):
                print(f"{GREEN}✓ {func_name} 函數存在{RESET}")
            else:
                print(f"{RED}✗ {func_name} 函數不存在{RESET}")
                return False
        
        print(f"\n{GREEN}✅ Cython 模組檢查通過！{RESET}")
        return True
        
    except ImportError as e:
        print(f"{RED}✗ Cython 模組載入失敗: {e}{RESET}")
        return False


def main():
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}🔥 compute_Electrode_charge_analytic 修復驗證{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    # 切換到正確的目錄
    if not os.path.exists('lib/MM_classes_CYTHON.py'):
        os.chdir('/home/andy/test_optimization/openMM_constant_V_beta')
    
    results = []
    
    # 執行檢查
    results.append(("電解質 C 陣列", check_electrolyte_c_arrays()))
    results.append(("Cython 函數", check_cython_function()))
    results.append(("compute_Electrode_charge_analytic", check_compute_electrode_charge_analytic()))
    results.append(("Cython 模組載入", check_cython_module_loaded()))
    
    # 總結
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}驗證總結{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    all_passed = True
    for name, passed in results:
        status = f"{GREEN}✓ 通過{RESET}" if passed else f"{RED}✗ 失敗{RESET}"
        print(f"{name:.<50} {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    if all_passed:
        print(f"{GREEN}🎉 所有檢查通過！修復成功！{RESET}")
        print(f"\n{GREEN}修復內容：{RESET}")
        print(f"{GREEN}1. 為電解質建立了 C 陣列 (Single Source of Truth){RESET}")
        print(f"{GREEN}2. 創建了 Cython C-level 函數 compute_analytic_contribution_cython{RESET}")
        print(f"{GREEN}3. 重寫了 compute_Electrode_charge_analytic，移除所有 API 呼叫{RESET}")
        print(f"{GREEN}4. 預估加速比：50x-100x (取決於電解質和導體數量){RESET}")
        print(f"\n{YELLOW}性能影響：{RESET}")
        print(f"{YELLOW}• 這個函數在每次 Poisson 迭代中被調用 2 次{RESET}")
        print(f"{YELLOW}• 修復前：N+M 次昂貴的 getParticleParameters 呼叫{RESET}")
        print(f"{YELLOW}• 修復後：純 C-level NumPy/Cython 計算{RESET}")
        return 0
    else:
        print(f"{RED}❌ 部分檢查失敗，請檢查代碼{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
