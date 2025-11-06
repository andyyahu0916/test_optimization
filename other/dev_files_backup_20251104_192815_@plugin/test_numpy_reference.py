#!/usr/bin/env python3
"""
Numpy reference implementation of the electrode charge algorithm.

Algorithm (single-pass, no iterations):
    q_e = C_inv * (V - E_f)

Where:
    q_e: electrode charges (output, size N)
    C_inv: inverse capacitance matrix (input, size N*N)
    V: target potentials (input, size N)
    E_f: electric field from electrolyte (computed, size N)
"""

import numpy as np

# Coulomb constant in OpenMM units (kJ/mol * nm / e^2)
COULOMB_CONSTANT = 138.935456

def compute_electrode_charges(
    electrode_positions,     # (N, 3) array
    electrolyte_positions,   # (M, 3) array
    target_potentials,       # (N,) array - in kJ/mol
    fixed_charges,           # (M,) array - in elementary charge
    inv_cap_matrix           # (N, N) array
):
    """
    Compute electrode charges using the matrix method.

    Returns:
        q_e: (N,) array of electrode charges
    """
    N = len(electrode_positions)
    M = len(electrolyte_positions)

    # Step 1: Compute E_f[i] = Σ_j (k * q_f[j] / r_ij)
    E_f = np.zeros(N)
    for i in range(N):
        for j in range(M):
            r_vec = electrode_positions[i] - electrolyte_positions[j]
            r_squared = np.dot(r_vec, r_vec)
            if r_squared > 1e-10:  # Avoid division by zero
                r_inv = 1.0 / np.sqrt(r_squared)
                E_f[i] += COULOMB_CONSTANT * fixed_charges[j] * r_inv

    # Step 2: Compute b = V - E_f
    b = target_potentials - E_f

    # Step 3: Matrix multiply q_e = C_inv * b
    q_e = inv_cap_matrix @ b

    return q_e


if __name__ == "__main__":
    # ========== Test: Simple 2-electrode, 1-electrolyte system ==========
    print("="*60)
    print("Numpy Reference Implementation Test")
    print("="*60)

    # Define a simple system
    N = 2  # 2 electrode atoms
    M = 1  # 1 electrolyte atom

    # Positions (nm)
    electrode_positions = np.array([
        [0.0, 0.0, 0.0],   # Electrode 1 at origin
        [0.0, 0.0, 2.0],   # Electrode 2 at z=2nm
    ])

    electrolyte_positions = np.array([
        [0.0, 0.0, 1.0],   # Electrolyte at z=1nm (middle)
    ])

    # Target potentials (kJ/mol)
    # For testing, use simple values
    target_potentials = np.array([10.0, -10.0])  # +10 and -10 kJ/mol

    # Fixed charges (elementary charge)
    fixed_charges = np.array([1.0])  # +1e charge

    # Inverse capacitance matrix (manually constructed for testing)
    # For this test, use a simple diagonal matrix
    inv_cap_matrix = np.array([
        [0.1, 0.0],
        [0.0, 0.1],
    ])

    # Compute charges
    q_e = compute_electrode_charges(
        electrode_positions,
        electrolyte_positions,
        target_potentials,
        fixed_charges,
        inv_cap_matrix
    )

    print(f"\nElectrode positions:\n{electrode_positions}")
    print(f"\nElectrolyte positions:\n{electrolyte_positions}")
    print(f"\nTarget potentials (kJ/mol): {target_potentials}")
    print(f"\nFixed charges (e): {fixed_charges}")
    print(f"\nInverse capacitance matrix:\n{inv_cap_matrix}")
    print(f"\nComputed electrode charges (e): {q_e}")
    print("\n" + "="*60)
    print("Test complete. This is the golden reference.")
    print("="*60)
