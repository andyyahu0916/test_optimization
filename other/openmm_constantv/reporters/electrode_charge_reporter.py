"""
Electrode Charge Reporter

Outputs electrode charges during simulation for post-analysis.

Corresponds to: MM_classes.py::write_electrode_charges() (Lines 824-842)
"""

from typing import List, TextIO
import logging

logger = logging.getLogger(__name__)


class ElectrodeChargeReporter:
    """
    Reporter that writes electrode charges to file during simulation.

    This reporter queries the ConstantVForce and writes charge values
    for cathode, conductors, and anode at regular intervals.

    Attributes:
        file: Output file handle
        reportInterval: Frequency of reporting (steps)
        cathode_indices: Cathode atom indices
        anode_indices: Anode atom indices
        conductor_indices: Conductor atom indices (optional)
    """

    def __init__(
        self,
        file: str | TextIO,
        reportInterval: int,
        cathode_indices: List[int],
        anode_indices: List[int],
        conductor_indices: List[List[int]] | None = None,
    ):
        """
        Initialize electrode charge reporter.

        Args:
            file: Output file path or file handle
            reportInterval: Report frequency (steps)
            cathode_indices: Cathode atom indices
            anode_indices: Anode atom indices
            conductor_indices: Conductor atom indices (optional)
        """
        self._reportInterval = reportInterval
        self._cathode_indices = cathode_indices
        self._anode_indices = anode_indices
        self._conductor_indices = conductor_indices if conductor_indices else []

        # Open file if string provided
        if isinstance(file, str):
            self._out = open(file, 'w')
            self._closeFile = True
        else:
            self._out = file
            self._closeFile = False

        logger.info(f"ElectrodeChargeReporter initialized (interval={reportInterval})")

    def __del__(self):
        """Close file if we opened it."""
        if self._closeFile and hasattr(self, '_out'):
            self._out.close()

    def describeNextReport(self, simulation):
        """
        Get information about the next report.

        This is called by OpenMM to determine when to call report().

        Args:
            simulation: OpenMM Simulation object

        Returns:
            (steps, positions, velocities, forces, energies, includeGroups)
        """
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, False, set())

    def report(self, simulation, state):
        """
        Generate report at this step.

        Corresponds to: MM_classes.py::write_electrode_charges() (Lines 825-842)

        Args:
            simulation: OpenMM Simulation object
            state: OpenMM State object (unused, but required by interface)
        """
        # Get NonbondedForce to query charges
        nonbonded_force = None
        for force in simulation.system.getForces():
            if type(force).__name__ == 'NonbondedForce':
                nonbonded_force = force
                break

        if nonbonded_force is None:
            logger.warning("NonbondedForce not found, skipping charge report")
            return

        # Write charges: Cathode, Conductors, Anode (Lines 826-837)
        charges = []

        # Cathode (Line 826-827)
        for idx in self._cathode_indices:
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(idx)
            charges.append(charge._value)

        # Conductors (Line 832-834)
        for conductor_indices in self._conductor_indices:
            for idx in conductor_indices:
                charge, sigma, epsilon = nonbonded_force.getParticleParameters(idx)
                charges.append(charge._value)

        # Anode (Line 836-837)
        for idx in self._anode_indices:
            charge, sigma, epsilon = nonbonded_force.getParticleParameters(idx)
            charges.append(charge._value)

        # Write to file (Line 841-842)
        self._out.write(" ".join(f"{q:.6f}" for q in charges) + "\n")
        self._out.flush()
