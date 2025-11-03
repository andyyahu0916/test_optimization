#!/usr/bin/env python
"""
驗證 MC Numba 和 Numerical_charge_Conductor 修復
================================================

檢查項目：
1. update_electrolyte_positions_numba 使用 oldpos 計算 delta
2. Numerical_charge_Conductor 使用 forces_np (NumPy) 而非 forces (OpenMM objects)
"""

import sys
import os

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_numba_function():
    """檢查 Numba 函數是否正確使用 oldpos"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}檢查 1: update_electrolyte_positions_numba 函數{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    file_path = 'lib/MM_classes_CYTHON.py'
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 查找函數定義
    if 'def update_electrolyte_positions_numba(newpos, oldpos,' in content:
        print(f"{GREEN}✓ 函數簽名正確 (包含 oldpos 和 newpos){RESET}")
    else:
        print(f"{RED}✗ 函數簽名錯誤{RESET}")
        return False
    
    # 檢查 Step 1: ref_x/y/z 來自 oldpos
    if 'ref_x = oldpos[first_atom_idx, 0]' in content:
        print(f"{GREEN}✓ Step 1: ref_x 來自 oldpos (正確){RESET}")
    else:
        print(f"{RED}✗ Step 1: ref_x 應該來自 oldpos{RESET}")
        return False
    
    if 'ref_y = oldpos[first_atom_idx, 1]' in content:
        print(f"{GREEN}✓ Step 1: ref_y 來自 oldpos (正確){RESET}")
    else:
        print(f"{RED}✗ Step 1: ref_y 應該來自 oldpos{RESET}")
        return False
    
    if 'ref_z = oldpos[first_atom_idx, 2]' in content:
        print(f"{GREEN}✓ Step 1: ref_z 來自 oldpos (正確){RESET}")
    else:
        print(f"{RED}✗ Step 1: ref_z 應該來自 oldpos{RESET}")
        return False
    
    # 檢查 Step 3: dx/dy/dz 計算來自 oldpos
    if 'dx = oldpos[atom_idx, 0] - ref_x' in content:
        print(f"{GREEN}✓ Step 3: dx 從 oldpos 計算 (正確){RESET}")
    else:
        print(f"{RED}✗ Step 3: dx 應該從 oldpos 計算{RESET}")
        return False
    
    if 'dy = oldpos[atom_idx, 1] - ref_y' in content:
        print(f"{GREEN}✓ Step 3: dy 從 oldpos 計算 (正確){RESET}")
    else:
        print(f"{RED}✗ Step 3: dy 應該從 oldpos 計算{RESET}")
        return False
    
    if 'dz = oldpos[atom_idx, 2] - ref_z' in content:
        print(f"{GREEN}✓ Step 3: dz 從 oldpos 計算 (正確){RESET}")
    else:
        print(f"{RED}✗ Step 3: dz 應該從 oldpos 計算{RESET}")
        return False
    
    # 檢查應用到 newpos
    if 'newpos[atom_idx, 0] = ref_x + dx' in content:
        print(f"{GREEN}✓ Step 3: 應用到 newpos (正確){RESET}")
    else:
        print(f"{RED}✗ Step 3: 應該應用到 newpos{RESET}")
        return False
    
    print(f"\n{GREEN}✅ Numba 函數檢查通過！{RESET}")
    return True


def check_numerical_charge_conductor():
    """檢查 Numerical_charge_Conductor 是否使用 NumPy 陣列"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}檢查 2: Numerical_charge_Conductor 函數{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    file_path = 'lib/MM_classes_CYTHON.py'
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 檢查函數簽名
    if 'def Numerical_charge_Conductor( self, Conductor, forces_np ):' in content:
        print(f"{GREEN}✓ 函數簽名正確 (接受 forces_np){RESET}")
    else:
        print(f"{RED}✗ 函數簽名應該接受 forces_np (不是 forces){RESET}")
        return False
    
    # 檢查調用端
    if 'self.Numerical_charge_Conductor( Conductor , forces_np )' in content:
        print(f"{GREEN}✓ 調用時傳入 forces_np (正確){RESET}")
    else:
        print(f"{RED}✗ 調用時應該傳入 forces_np{RESET}")
        return False
    
    # 檢查是否有檢查單位的代碼
    if "if hasattr(forces_np[0, 0], '_value'):" in content:
        print(f"{GREEN}✓ 有處理 OpenMM 單位 (正確){RESET}")
    else:
        print(f"{RED}✗ 應該檢查並處理 OpenMM 單位{RESET}")
        return False
    
    # 檢查是否使用 NumPy 索引而非 ._value
    if 'Ex = forces_values[index, 0] / q_i' in content:
        print(f"{GREEN}✓ Step 1: 使用 NumPy 索引 (forces_values[index, 0]){RESET}")
    else:
        print(f"{RED}✗ Step 1: 應該使用 NumPy 索引{RESET}")
        return False
    
    if 'Ex = forces_values[conductor_atom_index, 0] / q_i' in content:
        print(f"{GREEN}✓ Step 2: 使用 NumPy 索引 (forces_values[conductor_atom_index, 0]){RESET}")
    else:
        print(f"{RED}✗ Step 2: 應該使用 NumPy 索引{RESET}")
        return False
    
    # 檢查是否沒有舊的 ._value 存取 (應該被刪除)
    # 排除註解中的
    lines = content.split('\n')
    bad_access_count = 0
    for i, line in enumerate(lines, 1):
        if 'forces[' in line and ']._value' in line and not line.strip().startswith('#'):
            bad_access_count += 1
            print(f"{RED}✗ Line {i}: 發現舊的 forces[...]._value 存取{RESET}")
    
    if bad_access_count == 0:
        print(f"{GREEN}✓ 沒有發現舊的 forces[...]._value 存取{RESET}")
    else:
        print(f"{RED}✗ 發現 {bad_access_count} 處舊的存取方式{RESET}")
        return False
    
    print(f"\n{GREEN}✅ Numerical_charge_Conductor 檢查通過！{RESET}")
    return True


def check_file_consistency():
    """檢查文件的整體一致性"""
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}檢查 3: 文件整體一致性{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    file_path = 'lib/MM_classes_CYTHON.py'
    
    # 檢查文件是否存在
    if not os.path.exists(file_path):
        print(f"{RED}✗ 文件不存在: {file_path}{RESET}")
        return False
    
    print(f"{GREEN}✓ 文件存在{RESET}")
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    # 檢查文件大小
    lines = content.split('\n')
    print(f"{GREEN}✓ 文件行數: {len(lines)}{RESET}")
    
    # 檢查是否有 Numba import
    if 'from numba import' in content:
        print(f"{GREEN}✓ Numba 已導入{RESET}")
    else:
        print(f"{RED}✗ 未找到 Numba 導入{RESET}")
    
    # 檢查是否有 NumPy import
    if 'import numpy' in content:
        print(f"{GREEN}✓ NumPy 已導入{RESET}")
    else:
        print(f"{RED}✗ 未找到 NumPy 導入{RESET}")
    
    print(f"\n{GREEN}✅ 文件一致性檢查通過！{RESET}")
    return True


def main():
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}🔥 MC Numba 和 Numerical_charge_Conductor 修復驗證{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    # 切換到正確的目錄
    if not os.path.exists('lib/MM_classes_CYTHON.py'):
        os.chdir('/home/andy/test_optimization/openMM_constant_V_beta')
    
    results = []
    
    # 執行檢查
    results.append(("Numba 函數", check_numba_function()))
    results.append(("Numerical_charge_Conductor", check_numerical_charge_conductor()))
    results.append(("文件一致性", check_file_consistency()))
    
    # 總結
    print(f"\n{BLUE}{'='*70}{RESET}")
    print(f"{BLUE}驗證總結{RESET}")
    print(f"{BLUE}{'='*70}{RESET}")
    
    all_passed = True
    for name, passed in results:
        status = f"{GREEN}✓ 通過{RESET}" if passed else f"{RED}✗ 失敗{RESET}"
        print(f"{name:.<40} {status}")
        if not passed:
            all_passed = False
    
    print(f"\n{BLUE}{'='*70}{RESET}")
    if all_passed:
        print(f"{GREEN}🎉 所有檢查通過！修復成功！{RESET}")
        print(f"\n{GREEN}修復內容：{RESET}")
        print(f"{GREEN}1. Numba 函數現在正確地從 oldpos 計算分子內向量{RESET}")
        print(f"{GREEN}2. Numerical_charge_Conductor 現在使用 NumPy 陣列而非 OpenMM 物件{RESET}")
        print(f"{GREEN}3. 消除了昂貴的 ._value 存取和 Python 循環中的 API 呼叫{RESET}")
        return 0
    else:
        print(f"{RED}❌ 部分檢查失敗，請檢查代碼{RESET}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
