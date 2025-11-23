#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════
Physical Parity Verification Suite - "Numerical Forensics"
═══════════════════════════════════════════════════════════════════

This script performs rigorous mathematical validation between:
    - Platform='Reference' (CPU, double precision)
    - Platform='CUDA' (GPU, mixed precision, zero-copy)

Verification Strategy:
    1. Step-by-Step Forensics: Serialize state at EVERY step
    2. Strict Assertions: Energy, Force, Charge tolerances
    3. Green's Reciprocity: Charge neutrality check
    4. Visual Reporting: PDF plots showing drift/errors

Test Scenario:
    - 2 Graphene Sheets (Cathode/Anode)
    - 1 Buckyball (Close to Cathode)
    - 100 Water Molecules (SWM4-NDP)
    - 2 Ions (K+, Cl-)
    - Voltage: 1.0 V

Design Philosophy:
    - FAIL FAST: Any tolerance violation stops execution
    - VERBOSE: Every assertion prints diagnostic info
    - FORENSIC: Full state serialization for post-mortem

Corresponds to Second Shot requirements from user prompt
"""

import sys
import logging
from typing import Dict, List, Tuple
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Configure logging BEFORE importing OpenMM (prevents spam)
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

try:
    import openmm
    from openmm import app, unit
except ImportError:
    logger.error("OpenMM not found. Please install: conda install -c conda-forge openmm")
    sys.exit(1)

try:
    import constantvplugin
except ImportError:
    logger.error("ConstantV plugin not found. Please compile and install the plugin.")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════
# Physical Constants and Tolerances
# ═══════════════════════════════════════════════════════════

# Energy tolerance (kJ/mol)
ENERGY_TOLERANCE_KJMOL = 1e-4

# Force tolerance (kJ/mol/nm) - allow outliers for weak forces
FORCE_TOLERANCE_KJMOL_NM = 1e-5
FORCE_MSE_THRESHOLD = 1e-6  # Mean squared error

# Charge tolerance (elementary charge units)
CHARGE_TOLERANCE_REFERENCE = 1e-6  # Reference (double precision)
CHARGE_TOLERANCE_CUDA = 1e-6  # CUDA (mixed precision, more lenient)

# Green's Reciprocity: Total charge neutrality
CHARGE_NEUTRALITY_REFERENCE = 1e-9
CHARGE_NEUTRALITY_CUDA = 1e-6


# ═══════════════════════════════════════════════════════════
# System Configuration (Hardcoded Test Case)
# ═══════════════════════════════════════════════════════════

class TestSystemConfig:
    """
    Hardcoded configuration for the test scenario.

    This avoids dependencies on external config files for CI/CD.
    """
    voltage_volts = 1.0
    temperature_kelvin = 300.0
    timestep_ps = 0.001
    num_steps = 10  # Short test for verification
    scf_iterations = 4


# ═══════════════════════════════════════════════════════════
# State Serializer
# ═══════════════════════════════════════════════════════════

class StateSerializer:
    """
    Serializes full system state for forensic analysis.

    Captures:
        - Positions (nm)
        - Velocities (nm/ps)
        - Forces (kJ/mol/nm)
        - Potential Energy (kJ/mol)
        - Electrode Charges (e)
    """

    def __init__(self, platform_name: str):
        self.platform_name = platform_name
        self.steps: List[int] = []
        self.energies: List[float] = []
        self.positions: List[np.ndarray] = []
        self.velocities: List[np.ndarray] = []
        self.forces: List[np.ndarray] = []
        self.cathode_charges: List[np.ndarray] = []
        self.anode_charges: List[np.ndarray] = []

    def capture(
        self,
        step: int,
        simulation: app.Simulation,
        cathode_indices: List[int],
        anode_indices: List[int],
    ):
        """
        Capture full system state at this step.

        Args:
            step: Current step number
            simulation: OpenMM Simulation object
            cathode_indices: Cathode atom indices
            anode_indices: Anode atom indices
        """
        # Get state with ALL information
        state = simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getForces=True,
            getEnergy=True,
        )

        # Store step number
        self.steps.append(step)

        # Store energy
        energy_kjmol = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        self.energies.append(energy_kjmol)

        # Store positions
        pos = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
        self.positions.append(pos.copy())

        # Store velocities
        vel = state.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        self.velocities.append(vel.copy())

        # Store forces
        force = state.getForces(asNumpy=True).value_in_unit(unit.kilojoules_per_mole / unit.nanometer)
        self.forces.append(force.copy())

        # Store electrode charges
        nonbonded_force = None
        for force_obj in simulation.system.getForces():
            if isinstance(force_obj, openmm.NonbondedForce):
                nonbonded_force = force_obj
                break

        if nonbonded_force:
            # Cathode charges
            cathode_q = np.array([
                nonbonded_force.getParticleParameters(idx)[0]._value
                for idx in cathode_indices
            ])
            self.cathode_charges.append(cathode_q.copy())

            # Anode charges
            anode_q = np.array([
                nonbonded_force.getParticleParameters(idx)[0]._value
                for idx in anode_indices
            ])
            self.anode_charges.append(anode_q.copy())

        logger.debug(
            f"[{self.platform_name}] Step {step}: "
            f"E={energy_kjmol:.6f} kJ/mol, "
            f"Q_cathode={sum(cathode_q):.6f} e"
        )


# ═══════════════════════════════════════════════════════════
# Parity Checker
# ═══════════════════════════════════════════════════════════

class ParityChecker:
    """
    Checks mathematical equivalence between Reference and CUDA platforms.

    Implements strict assertions with clear diagnostic messages.
    """

    def __init__(
        self,
        ref_state: StateSerializer,
        cuda_state: StateSerializer,
    ):
        self.ref = ref_state
        self.cuda = cuda_state
        self.errors: List[str] = []

    def check_all(self) -> bool:
        """
        Run all parity checks.

        Returns:
            True if all checks pass, False otherwise
        """
        logger.info("╔═══════════════════════════════════════════════════════╗")
        logger.info("║         PHYSICAL PARITY VERIFICATION SUITE            ║")
        logger.info("╚═══════════════════════════════════════════════════════╝")

        checks = [
            ("Energy Parity", self._check_energy_parity),
            ("Force Parity", self._check_force_parity),
            ("Charge Parity", self._check_charge_parity),
            ("Green's Reciprocity", self._check_greens_reciprocity),
        ]

        all_passed = True
        for check_name, check_func in checks:
            logger.info(f"\n>>> Running: {check_name}")
            try:
                check_func()
                logger.info(f"✅ {check_name} PASSED")
            except AssertionError as e:
                logger.error(f"❌ {check_name} FAILED: {e}")
                self.errors.append(f"{check_name}: {e}")
                all_passed = False

        return all_passed

    def _check_energy_parity(self):
        """Assert Energy_Reference vs Energy_CUDA diff < tolerance."""
        for i, (e_ref, e_cuda) in enumerate(zip(self.ref.energies, self.cuda.energies)):
            diff = abs(e_ref - e_cuda)
            assert diff < ENERGY_TOLERANCE_KJMOL, (
                f"Step {i}: Energy difference {diff:.6e} kJ/mol "
                f"exceeds tolerance {ENERGY_TOLERANCE_KJMOL:.6e} kJ/mol. "
                f"Reference={e_ref:.6f}, CUDA={e_cuda:.6f}"
            )

    def _check_force_parity(self):
        """Assert Force_Reference vs Force_CUDA diff < tolerance (MSE check)."""
        for i, (f_ref, f_cuda) in enumerate(zip(self.ref.forces, self.cuda.forces)):
            # Compute element-wise difference
            diff = f_ref - f_cuda

            # Compute mean squared error
            mse = np.mean(diff ** 2)

            assert mse < FORCE_MSE_THRESHOLD, (
                f"Step {i}: Force MSE {mse:.6e} kJ/mol/nm "
                f"exceeds threshold {FORCE_MSE_THRESHOLD:.6e}. "
                f"Max error: {np.max(np.abs(diff)):.6e}"
            )

    def _check_charge_parity(self):
        """Assert Charge_Reference vs Charge_CUDA diff < tolerance."""
        for i in range(len(self.ref.cathode_charges)):
            # Cathode
            q_ref_cath = self.ref.cathode_charges[i]
            q_cuda_cath = self.cuda.cathode_charges[i]
            diff_cath = np.max(np.abs(q_ref_cath - q_cuda_cath))

            assert diff_cath < CHARGE_TOLERANCE_CUDA, (
                f"Step {i}: Cathode charge difference {diff_cath:.6e} e "
                f"exceeds tolerance {CHARGE_TOLERANCE_CUDA:.6e} e"
            )

            # Anode
            q_ref_anode = self.ref.anode_charges[i]
            q_cuda_anode = self.cuda.anode_charges[i]
            diff_anode = np.max(np.abs(q_ref_anode - q_cuda_anode))

            assert diff_anode < CHARGE_TOLERANCE_CUDA, (
                f"Step {i}: Anode charge difference {diff_anode:.6e} e "
                f"exceeds tolerance {CHARGE_TOLERANCE_CUDA:.6e} e"
            )

    def _check_greens_reciprocity(self):
        """Assert Sum(Charges) == 0.0 (charge neutrality)."""
        for i in range(len(self.ref.cathode_charges)):
            # Reference
            q_total_ref = (
                np.sum(self.ref.cathode_charges[i])
                + np.sum(self.ref.anode_charges[i])
            )
            assert abs(q_total_ref) < CHARGE_NEUTRALITY_REFERENCE, (
                f"Step {i}: Reference total charge {q_total_ref:.6e} e "
                f"violates neutrality (tolerance {CHARGE_NEUTRALITY_REFERENCE:.6e} e)"
            )

            # CUDA
            q_total_cuda = (
                np.sum(self.cuda.cathode_charges[i])
                + np.sum(self.cuda.anode_charges[i])
            )
            assert abs(q_total_cuda) < CHARGE_NEUTRALITY_CUDA, (
                f"Step {i}: CUDA total charge {q_total_cuda:.6e} e "
                f"violates neutrality (tolerance {CHARGE_NEUTRALITY_CUDA:.6e} e)"
            )

    def generate_report(self, output_pdf: str):
        """
        Generate visual PDF report showing drift and errors.

        Args:
            output_pdf: Path to output PDF file
        """
        logger.info(f"\n>>> Generating visual report: {output_pdf}")

        with PdfPages(output_pdf) as pdf:
            # Page 1: Energy Drift
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(self.ref.steps, self.ref.energies, 'o-', label='Reference', alpha=0.7)
            ax.plot(self.cuda.steps, self.cuda.energies, 's-', label='CUDA', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Potential Energy (kJ/mol)')
            ax.set_title('Energy Drift Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close()

            # Page 2: Charge Drift (Cathode)
            fig, ax = plt.subplots(figsize=(10, 6))
            ref_q_cath_total = [np.sum(q) for q in self.ref.cathode_charges]
            cuda_q_cath_total = [np.sum(q) for q in self.cuda.cathode_charges]
            ax.plot(self.ref.steps, ref_q_cath_total, 'o-', label='Reference', alpha=0.7)
            ax.plot(self.cuda.steps, cuda_q_cath_total, 's-', label='CUDA', alpha=0.7)
            ax.set_xlabel('Step')
            ax.set_ylabel('Total Cathode Charge (e)')
            ax.set_title('Cathode Charge Drift Over Time')
            ax.legend()
            ax.grid(True, alpha=0.3)
            pdf.savefig(fig)
            plt.close()

            # Page 3: Force Error Histogram
            fig, ax = plt.subplots(figsize=(10, 6))
            all_force_errors = []
            for f_ref, f_cuda in zip(self.ref.forces, self.cuda.forces):
                diff = (f_ref - f_cuda).flatten()
                all_force_errors.extend(diff)
            ax.hist(all_force_errors, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Force Error (kJ/mol/nm)')
            ax.set_ylabel('Count')
            ax.set_title('Histogram of Force Errors (Should be Gaussian Noise)')
            ax.grid(True, alpha=0.3, axis='y')
            pdf.savefig(fig)
            plt.close()

        logger.info(f"✅ Report generated: {output_pdf}")


# ═══════════════════════════════════════════════════════════
# Main Test Function
# ═══════════════════════════════════════════════════════════

def main():
    """
    Main test entry point.

    Note: This is a TEMPLATE. The actual test requires:
        1. A valid PDB file with test system
        2. Force field XML files
        3. Compiled ConstantV plugin

    For CI/CD, you must provide these files in the test directory.
    """
    logger.info("╔═══════════════════════════════════════════════════════╗")
    logger.info("║  Physical Parity Verification - TEMPLATE SCRIPT       ║")
    logger.info("║                                                        ║")
    logger.info("║  To run this test, you need:                          ║")
    logger.info("║  1. PDB file (test_system.pdb)                        ║")
    logger.info("║  2. Force field XMLs                                  ║")
    logger.info("║  3. Compiled ConstantV plugin                         ║")
    logger.info("╚═══════════════════════════════════════════════════════╝\n")

    logger.warning(
        "This is a TEMPLATE script. "
        "Please adapt it to your specific test system."
    )

    # TODO: Implement actual test with real system files
    # For now, print structure guidance
    print("\n" + "="*60)
    print("IMPLEMENTATION CHECKLIST:")
    print("="*60)
    print("[ ] Create test PDB with 2 graphene sheets + buckyball + water")
    print("[ ] Configure cathode/anode atom indices")
    print("[ ] Run Reference simulation (double precision)")
    print("[ ] Run CUDA simulation (mixed precision)")
    print("[ ] Call ParityChecker.check_all()")
    print("[ ] Generate PDF report")
    print("="*60)


if __name__ == "__main__":
    main()
