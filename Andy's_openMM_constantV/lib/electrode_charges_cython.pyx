# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
electrode_charges_cython.pyx

🔥 GOOD TASTE VERSION - Pure C-level computation
計算和同步徹底分離

將 Python/NumPy 編譯成 C 級別代碼，只操作 memoryviews
絕不呼叫 OpenMM API，絕不接觸 Python 物件列表
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, sqrt

# 定義 C 類型
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

#=========================================================================
# 🔥 CORE FUNCTION: compute_electrode_charges_cython
# 這個函數已經是「好品味」——保持原樣
# 純 C-level 數學，只操作 memoryviews，無 API 呼叫
#=========================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_electrode_charges_cython(
    double[:] forces_z,           # C array view (fast!)
    double[:] q_old,
    long[:] indices,
    double prefactor,
    double voltage_term,
    double threshold_check,
    double small_threshold,
    double sign
):
    """
    ✅ GOOD TASTE - 純 C-level electrode charge 計算
    
    這個函數是完美的：
    - 只接受 memoryviews (C arrays)
    - 無 Python 物件
    - 無 API 呼叫
    - 純數學計算
    
    Parameters:
    -----------
    forces_z : memoryview of float64
        All z-forces (full array, length = total_atoms)
    q_old : memoryview of float64
        Old electrode charges (length = N_electrode)
    indices : memoryview of int64
        Electrode atom indices in full array
    prefactor : float64
        Charge calculation prefactor
    voltage_term : float64
        Voltage contribution
    threshold_check : float64
        Threshold for safe division (0.9 * small_threshold)
    small_threshold : float64
        Minimum charge magnitude
    sign : float64
        Sign (+1 for cathode, -1 for anode)
    
    Returns:
    --------
    q_new : ndarray of float64
        New electrode charges
    """
    cdef Py_ssize_t i, atom_idx
    cdef Py_ssize_t N = len(indices)
    cdef double q_i_old, Ez_external, q_i
    
    # Pre-allocate output array
    cdef np.ndarray[DTYPE_t, ndim=1] q_new = np.empty(N, dtype=np.float64)
    cdef double[:] q_new_view = q_new  # Memoryview for fast access
    
    # C-level for loop (無 Python overhead!)
    for i in range(N):
        atom_idx = indices[i]
        q_i_old = q_old[i]
        
        # Safe division (matches NumPy where logic)
        if fabs(q_i_old) > threshold_check:
            Ez_external = forces_z[atom_idx] / q_i_old
        else:
            Ez_external = 0.0
        
        # Compute new charge
        q_i = prefactor * (voltage_term + Ez_external)
        
        # Apply threshold
        if fabs(q_i) < small_threshold:
            q_i = sign * small_threshold
        
        q_new_view[i] = q_i
    
    return q_new


#=========================================================================
# 🔥 NEW FUNCTION: scale_charges_inplace_cython
# 快速、就地縮放 charges (用於 Scale_charges_analytic)
# 純 C-level 數學，無 API 呼叫
#=========================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def scale_charges_inplace_cython(
    double[:] c_charges,   # 傳入 c_charges NumPy 陣列
    double scale_factor
):
    """
    ✅ GOOD TASTE - 快速、就地縮放 C 陣列中的 charges
    
    純 C-level 數學，無 API 呼叫
    
    Parameters:
    -----------
    c_charges : memoryview of float64
        Charge array to scale (modified in-place)
    scale_factor : float64
        Factor to scale by
    
    Returns:
    --------
    None (modifies c_charges in-place)
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = c_charges.shape[0]
    
    for i in range(N):
        c_charges[i] = c_charges[i] * scale_factor


#=========================================================================
# 🔥 NEW FUNCTION: initialize_charges_cython
# 快速初始化 charges (用於 initialize_Charge)
# 純 C-level 數學，無 API 呼叫
#=========================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def initialize_charges_cython(
    double[:] c_charges,      # 要填充的陣列
    double charge_per_atom,
    double small_threshold,
    double sign
):
    """
    ✅ GOOD TASTE - 快速初始化 C 陣列中的 charges
    
    純 C-level 數學，無 API 呼叫
    
    Parameters:
    -----------
    c_charges : memoryview of float64
        Charge array to initialize (modified in-place)
    charge_per_atom : float64
        Base charge value
    small_threshold : float64
        Minimum charge magnitude
    sign : float64
        Sign (+1 for cathode, -1 for anode)
    
    Returns:
    --------
    None (modifies c_charges in-place)
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = c_charges.shape[0]
    cdef double q_i
    
    for i in range(N):
        q_i = charge_per_atom
        
        # Apply threshold
        if fabs(q_i) < small_threshold:
            q_i = q_i + sign * small_threshold
        
        c_charges[i] = q_i


#=========================================================================
# � NEW FUNCTION: compute_analytic_contribution_cython
# 快速計算 Q_analytic 貢獻 (用於 compute_Electrode_charge_analytic)
# 純 C-level 數學，無 API 呼叫
#=========================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_analytic_contribution_cython(
    double[:] z_positions,    # 全部原子的 z 座標 (N_total_atoms)
    long[:] c_indices,        # 要加總的原子索引 (N_contrib)
    double[:] c_charges,      # 要加總的原子電荷 (N_contrib)
    double z_opposite,        # 對面電極的 z 座標
    double Lcell              # Cell 長度
):
    """
    ✅ GOOD TASTE - 快速計算 Q_analytic 貢獻
    
    純 C-level 數學，只操作 memoryviews
    用於 compute_Electrode_charge_analytic 中的電解質和導體貢獻計算
    
    計算：sum( |z_atom - z_opposite| / Lcell * (-q_atom) )
    
    Parameters:
    -----------
    z_positions : memoryview of float64
        All atom z-coordinates (length = N_total_atoms)
    c_indices : memoryview of int64
        Indices of atoms to sum over (length = N_contrib)
    c_charges : memoryview of float64
        Charges of atoms to sum over (length = N_contrib)
    z_opposite : float64
        Z-coordinate of opposite electrode
    Lcell : float64
        Cell length
    
    Returns:
    --------
    contribution : float64
        Computed contribution to Q_analytic
    """
    cdef Py_ssize_t i, atom_idx
    cdef Py_ssize_t N = c_indices.shape[0]
    cdef double z_atom, z_distance
    cdef double contribution = 0.0
    
    # C-level for loop (快速！)
    for i in range(N):
        atom_idx = c_indices[i]
        z_atom = z_positions[atom_idx]
        
        # abs(z_atom - z_opposite)
        z_distance = z_atom - z_opposite
        if z_distance < 0.0:
            z_distance = -z_distance
        
        # 累加: sum(|z - z_opp| / Lcell * (-q))
        contribution += (z_distance / Lcell) * (-c_charges[i])
    
    return contribution


#=========================================================================
# �🗑️ END OF FILE - All zombie code deleted
#
# Previously deleted (11 functions, ~300 lines):
# - extract_forces_z_cython          → Use NumPy slicing: forces_np[:, 2]
# - update_openmm_charges_batch      → Sync logic moved to Python layer
# - scale_electrode_charges_cython   → Replaced by scale_charges_inplace_cython
# - get_total_charge_cython          → Use numpy.sum(c_charges) directly
# - compute_z_position_cython        → Not critical, keep in Python
# - collect_electrode_charges_cython → Use self.c_charges directly
# - initialize_electrode_charge_cython → Replaced by initialize_charges_cython
# - compute_buckyball_center_cython  → Not critical, keep in Python
# - set_normal_vectors_cython        → Not critical, keep in Python
# - compute_buckyball_radius_cython  → Not critical, keep in Python
# - compute_normal_vectors_buckyball_cython → Not critical, keep in Python
#
# These functions mixed computation with API calls or Python object access.
# They are the remnants of bad design. Now deleted.
#=========================================================================
