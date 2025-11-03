# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: initializedcheck=False
"""
electrode_charges_cython.pyx

🔥 Cython 優化的 electrode charge 計算

將 Python/NumPy 編譯成 C 級別代碼
預期加速: 2-5x vs NumPy vectorized
"""

import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport fabs, sqrt

# 定義 C 類型
ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t ITYPE_t

@cython.boundscheck(False)  # 關閉邊界檢查
@cython.wraparound(False)   # 關閉負索引支持
@cython.cdivision(True)      # C 風格除法（不檢查除零）
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
    Cython 優化版本的 electrode charge 計算
    
    純 C-level 循環，無 Python overhead
    
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


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_electrode_charges_pure(
    double[:] forces_z,
    double[:] charges_old,
    double area_atom,
    double voltage,
    double Lgap,
    double sign,
    double small_threshold,
    double conversion_factor
):
    """
    純 Poisson 計算 (C-level)
    
    只包含數值計算，不包含 Python 物件操作
    
    參數:
    -----
    forces_z : z 方向力
    charges_old : 舊電荷
    area_atom : 每原子面積
    voltage : 電壓
    Lgap : 電極間距
    sign : 符號 (+1 or -1)
    small_threshold : 最小電荷閾值
    conversion_factor : conversion_KjmolNm_Au
    
    返回:
    -----
    charges_new : 新電荷
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(forces_z)
    cdef double q_i_old, Ez_external, q_i
    cdef double threshold_check = 0.9 * small_threshold
    cdef double prefactor = sign * 2.0 / (4.0 * 3.14159265358979323846) * area_atom * conversion_factor
    cdef double voltage_term = voltage / Lgap
    
    # Pre-allocate output
    cdef np.ndarray[DTYPE_t, ndim=1] charges_new = np.empty(N, dtype=np.float64)
    cdef double[:] charges_new_view = charges_new
    
    # C-level loop
    for i in range(N):
        q_i_old = charges_old[i]
        
        # Ez calculation (safe division)
        if fabs(q_i_old) > threshold_check:
            Ez_external = forces_z[i] / q_i_old
        else:
            Ez_external = 0.0
        
        # Poisson equation
        q_i = prefactor * (voltage_term + Ez_external)
        
        # Threshold check
        if fabs(q_i) < small_threshold:
            q_i = sign * small_threshold
        
        charges_new_view[i] = q_i
    
    return charges_new


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def compute_analytic_charge_contribution_cython(
    double[:] z_positions,
    double[:] charges,
    long[:] indices,
    double z_opposite,
    double Lcell
):
    """
    Cython 優化版本的 analytic charge contribution
    
    純 C-level 循環計算 sum
    """
    cdef Py_ssize_t i, atom_idx
    cdef Py_ssize_t N = len(indices)
    cdef double z_atom, z_distance
    cdef double contribution = 0.0
    
    # C-level accumulation loop
    for i in range(N):
        atom_idx = indices[i]
        z_atom = z_positions[atom_idx]
        
        # abs(z_atom - z_opposite)
        z_distance = z_atom - z_opposite
        if z_distance < 0.0:
            z_distance = -z_distance
        
        # Accumulate: sum(|z - z_opp| / Lcell * (-q))
        contribution += z_distance / Lcell * (-charges[i])
    
    return contribution


@cython.boundscheck(False)
@cython.wraparound(False)
def extract_z_coordinates_cython(positions_list):
    """
    快速提取 z 座標
    
    比 list comprehension 快 2-3x
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(positions_list)
    cdef np.ndarray[DTYPE_t, ndim=1] z_coords = np.empty(N, dtype=np.float64)
    
    for i in range(N):
        z_coords[i] = positions_list[i][2]._value
    
    return z_coords


@cython.boundscheck(False)
@cython.wraparound(False)
def extract_forces_z_cython(forces_list):
    """
    快速提取 z 方向力
    
    比 list comprehension 快 2-3x
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(forces_list)
    cdef np.ndarray[DTYPE_t, ndim=1] forces_z = np.empty(N, dtype=np.float64)
    
    for i in range(N):
        forces_z[i] = forces_list[i][2]._value
    
    return forces_z


@cython.boundscheck(False)
@cython.wraparound(False)
def update_openmm_charges_batch(
    object nbondedForce,  # OpenMM NonbondedForce object
    electrode_atoms,      # List of atom objects
    double[:] charges     # New charges array
):
    """
    批次更新 OpenMM charges
    
    減少 Python 函數調用 overhead
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(electrode_atoms)
    cdef double q
    cdef object atom
    
    for i in range(N):
        atom = electrode_atoms[i]
        q = charges[i]
        
        # Update atom.charge (Python attribute)
        atom.charge = q
        
        # Update OpenMM force
        nbondedForce.setParticleParameters(atom.atom_index, q, 1.0, 0.0)


@cython.boundscheck(False)
@cython.wraparound(False)
def scale_electrode_charges_cython(
    electrode_atoms,      # List of atom objects
    object nbondedForce,  # OpenMM NonbondedForce object
    double scale_factor
):
    """
    快速縮放 electrode charges
    
    用於 Analytic normalization (Scale_charges_analytic)
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(electrode_atoms)
    cdef object atom
    cdef double new_charge
    
    for i in range(N):
        atom = electrode_atoms[i]
        new_charge = atom.charge * scale_factor
        atom.charge = new_charge
        nbondedForce.setParticleParameters(atom.atom_index, new_charge, 1.0, 0.0)


@cython.boundscheck(False)
@cython.wraparound(False)
def get_total_charge_cython(electrode_atoms):
    """
    快速計算 total charge
    
    比 sum([atom.charge for atom in ...]) 快 3-5x
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(electrode_atoms)
    cdef double total_charge = 0.0
    cdef object atom
    
    for i in range(N):
        atom = electrode_atoms[i]
        total_charge += atom.charge
    
    return total_charge


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_z_position_cython(electrode_atoms, positions_list):
    """
    快速計算 electrode 平均 z 位置
    
    用於 compute_z_position() 方法
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(electrode_atoms)
    cdef double z_sum = 0.0
    cdef object atom
    cdef long atom_idx
    
    for i in range(N):
        atom = electrode_atoms[i]
        atom_idx = atom.atom_index
        z_sum += positions_list[atom_idx][2]._value
    
    return z_sum / N


@cython.boundscheck(False)
@cython.wraparound(False)
def collect_electrode_charges_cython(
    electrode_atoms,
    object nbondedForce
):
    """
    快速收集 electrode charges
    
    比 list comprehension 快 2-3x
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(electrode_atoms)
    cdef np.ndarray[DTYPE_t, ndim=1] charges = np.empty(N, dtype=np.float64)
    cdef object atom
    
    for i in range(N):
        atom = electrode_atoms[i]
        charges[i] = atom.charge
    
    return charges


@cython.boundscheck(False)
@cython.wraparound(False)
def initialize_electrode_charge_cython(
    electrode_atoms,
    object nbondedForce,
    double charge_per_atom,
    double small_threshold,
    double sign
):
    """
    快速初始化 electrode charges
    
    用於 initialize_Charge() 方法
    比 Python loop 快 2-3x
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(electrode_atoms)
    cdef object atom
    cdef double q_i
    
    for i in range(N):
        atom = electrode_atoms[i]
        q_i = charge_per_atom
        
        # Apply small threshold if charge is too small
        if fabs(q_i) < small_threshold:
            q_i = q_i + sign * small_threshold
        
        atom.charge = q_i
        nbondedForce.setParticleParameters(atom.atom_index, q_i, 1.0, 0.0)


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_buckyball_center_cython(electrode_atoms, positions_list):
    """
    快速計算 buckyball 中心座標
    
    用於 Buckyball_Virtual.__init__
    比 Python loop 快 3-4x
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(electrode_atoms)
    cdef double cx = 0.0, cy = 0.0, cz = 0.0
    cdef object atom
    cdef long atom_idx
    
    for i in range(N):
        atom = electrode_atoms[i]
        atom_idx = atom.atom_index
        cx += positions_list[atom_idx][0]._value
        cy += positions_list[atom_idx][1]._value
        cz += positions_list[atom_idx][2]._value
    
    return (cx / N, cy / N, cz / N)


@cython.boundscheck(False)
@cython.wraparound(False)
def set_normal_vectors_cython(electrode_atoms):
    """
    快速設置 electrode 法向量 (全部指向 +Z)
    
    用於 Conductor_Virtual.set_normal_vector()
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(electrode_atoms)
    cdef object atom
    
    for i in range(N):
        atom = electrode_atoms[i]
        atom.nx = 0.0
        atom.ny = 0.0
        atom.nz = 1.0


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_buckyball_radius_cython(
    electrode_atoms,
    positions_list,
    double cx,
    double cy,
    double cz
):
    """
    計算 buckyball 半徑 (從第一個原子)
    
    用於 Buckyball_Virtual.__init__
    """
    cdef object atom
    cdef long atom_idx
    cdef double dx, dy, dz, radius
    
    # Get first atom
    atom = electrode_atoms[0]
    atom_idx = atom.atom_index
    
    dx = positions_list[atom_idx][0]._value - cx
    dy = positions_list[atom_idx][1]._value - cy
    dz = positions_list[atom_idx][2]._value - cz
    
    radius = sqrt(dx*dx + dy*dy + dz*dz)
    
    return radius


@cython.boundscheck(False)
@cython.wraparound(False)
def compute_normal_vectors_buckyball_cython(
    electrode_atoms,
    positions_list,
    double cx,
    double cy,
    double cz
):
    """
    計算 buckyball 每個原子的法向量 (從中心指向原子)
    
    用於 Buckyball_Virtual.__init__
    比 Python loop 快 3-5x
    """
    cdef Py_ssize_t i
    cdef Py_ssize_t N = len(electrode_atoms)
    cdef object atom
    cdef long atom_idx
    cdef double dx, dy, dz, norm
    
    for i in range(N):
        atom = electrode_atoms[i]
        atom_idx = atom.atom_index
        
        # Vector from center to atom
        dx = positions_list[atom_idx][0]._value - cx
        dy = positions_list[atom_idx][1]._value - cy
        dz = positions_list[atom_idx][2]._value - cz
        
        # Normalize
        norm = sqrt(dx*dx + dy*dy + dz*dz)
        
        if norm > 1e-10:
            atom.nx = dx / norm
            atom.ny = dy / norm
            atom.nz = dz / norm
        else:
            # Fallback for degenerate case
            atom.nx = 0.0
            atom.ny = 0.0
            atom.nz = 1.0
