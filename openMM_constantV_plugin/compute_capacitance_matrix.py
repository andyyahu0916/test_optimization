#!/usr/bin/env python3
"""
Compute inverse capacitance matrix for ConstantV plugin.

Converts the iterative SCF algorithm to a single-pass matrix method:
   Original (SCF):  q^(n+1) = α * (V/L + E_z^(n))   [iterates 3-4 times]
   Matrix form:     q = C_inv * v                    [single matrix multiply]

where:
   C_inv = (I - M)^(-1)
   M_ij = α_i * k * cos(θ_ij) / r_ij²
   α_i = (2/4π) * area_i  (area per atom)
   v_i = α_i * V_i / L
   k = 138.935456  (Coulomb constant in kJ/mol·nm/e²)
"""

import numpy as np
from openmm import *
from openmm.app import *
from openmm.unit import *

# Coulomb constant (OpenMM units: kJ/mol · nm / e²)
COULOMB_CONSTANT = 138.935456

def compute_inverse_capacitance_matrix(
    electrode_positions,  # (N, 3) array in nm
    electrode_areas,      # (N,) array of area per atom in nm²
    gap_length=None       # Optional: gap length in nm for scaling
):
    """
    Compute the inverse capacitance matrix C_inv.

    Parameters:
    -----------
    electrode_positions : np.ndarray, shape (N, 3)
        Positions of electrode atoms in nanometers
    electrode_areas : np.ndarray, shape (N,)
        Area assigned to each electrode atom in nm²
        For flat electrodes: sheet_area / N_atoms
        For spheres: 4π*R² / N_atoms
        For cylinders: 2π*R*L / N_atoms
    gap_length : float, optional
        Electrode separation in nm (only used for v vector, not C_inv)

    Returns:
    --------
    C_inv : np.ndarray, shape (N, N)
        Inverse capacitance matrix
    """
    N = len(electrode_positions)

    # Compute α_i = (2/4π) * area_i
    alpha = (2.0 / (4.0 * np.pi)) * electrode_areas

    # Compute M matrix using vectorized NumPy operations (MUCH faster!)
    # M_ij = α_i * k / r_ij²  (for i ≠ j, M_ii = 0)
    print(f"  Building M matrix ({N}×{N}) using vectorized operations...")

    import time
    t_start = time.time()

    # Compute all pairwise distances at once using broadcasting
    # pos[i] - pos[j] for all i,j pairs
    # Shape: (N, 1, 3) - (1, N, 3) = (N, N, 3)
    diff = electrode_positions[:, np.newaxis, :] - electrode_positions[np.newaxis, :, :]

    # Compute r_squared for all pairs
    # Shape: (N, N)
    r_squared = np.sum(diff ** 2, axis=2)

    # Avoid division by zero (diagonal elements and any coincident atoms)
    r_squared_safe = np.where(r_squared > 0, r_squared, 1.0)

    # Compute M_ij = α_i * k / r²
    # Broadcasting alpha (N,) with r_squared_safe (N, N)
    M = alpha[:, np.newaxis] * COULOMB_CONSTANT / r_squared_safe

    # Set diagonal to zero (M_ii = 0)
    np.fill_diagonal(M, 0.0)

    # Also zero out any elements where r_squared was originally 0 (coincident atoms)
    M = np.where(r_squared > 0, M, 0.0)

    t_elapsed = time.time() - t_start
    print(f"  ✓ M matrix built in {t_elapsed:.2f} seconds (vectorized)")

    # Compute C_inv = (I - M)^(-1)
    print(f"  Computing (I - M) matrix...")
    I = np.eye(N)
    I_minus_M = I - M

    print(f"  Computing matrix inverse (this may take a while for large N={N})...")
    import time
    t_start = time.time()
    C_inv = np.linalg.inv(I_minus_M)
    t_elapsed = time.time() - t_start
    print(f"  ✓ Matrix inversion completed in {t_elapsed:.2f} seconds")

    return C_inv


def compute_flat_electrode_areas(N_atoms, sheet_length_x, sheet_length_y):
    """
    Compute area per atom for a flat rectangular electrode.

    Parameters:
    -----------
    N_atoms : int
        Number of atoms in electrode
    sheet_length_x, sheet_length_y : float
        Dimensions of electrode sheet in nm

    Returns:
    --------
    areas : np.ndarray, shape (N_atoms,)
        Area per atom (uniform for flat electrode)
    """
    sheet_area = sheet_length_x * sheet_length_y
    area_per_atom = sheet_area / N_atoms
    return np.full(N_atoms, area_per_atom)


def example_graphene_electrodes():
    """
    Example: Compute C_inv for two parallel graphene sheets.
    """
    print("=" * 70)
    print("Example: Two parallel graphene electrodes")
    print("=" * 70)

    # Simple test: 2 atoms per electrode, separated by 2 nm
    N = 2
    positions = np.array([
        [0.0, 0.0, 0.0],  # Cathode atom 1
        [0.0, 0.0, 2.0],  # Anode atom 1
    ])

    # Assume 1 nm² per atom (arbitrary for this test)
    areas = np.array([1.0, 1.0])

    # Compute C_inv
    C_inv = compute_inverse_capacitance_matrix(positions, areas)

    print(f"\nElectrode positions (nm):")
    print(positions)
    print(f"\nAreas per atom (nm²):")
    print(areas)
    print(f"\nInverse capacitance matrix C_inv:")
    print(C_inv)
    print(f"\nMatrix shape: {C_inv.shape}")
    print(f"Determinant: {np.linalg.det(C_inv):.6f}")

    # Test: apply voltage and compute charges
    gap_length = 2.0  # nm
    voltage_cathode = 1.0  # V (in kJ/mol after conversion)
    voltage_anode = -1.0

    # Convert V to kJ/mol (1 eV = 96.485 kJ/mol)
    conversion_V_to_kJmol = 96.485
    V = np.array([voltage_cathode, voltage_anode]) * conversion_V_to_kJmol

    # Compute v = α * V / L
    alpha = (2.0 / (4.0 * np.pi)) * areas
    v = alpha * V / gap_length

    # Compute charges
    q = C_inv @ v

    print(f"\nApplied voltages: {V} kJ/mol")
    print(f"Computed charges: {q} e")
    print(f"Total charge: {np.sum(q):.6f} e (should be ~0 for charge neutral)")

    return C_inv


if __name__ == "__main__":
    # Run example
    C_inv = example_graphene_electrodes()

    print("\n" + "=" * 70)
    print("Matrix saved and ready for use with ConstantVPlugin")
    print("=" * 70)
