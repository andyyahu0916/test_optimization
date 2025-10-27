import ctypes
import math
import os
import sys
import unittest

from openmm import Context, LangevinIntegrator, NonbondedForce, Platform, System, unit
from openmm import Vec3

_PYTHON_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "build", "python"))
if _PYTHON_BUILD_DIR not in sys.path:
    sys.path.insert(0, _PYTHON_BUILD_DIR)

_REFERENCE_LIB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "build", "platforms", "reference", "libElectrodeChargePluginReference.so")
)
_PLUGIN_LIB = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, "build", "libElectrodeChargePlugin.so"))
_CUDA_LIB = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, "build", "platforms", "cuda", "libElectrodeChargePluginCUDA.so")
)
_RTLD_GLOBAL = getattr(ctypes, "RTLD_GLOBAL", 0)
ctypes.CDLL(_REFERENCE_LIB, mode=_RTLD_GLOBAL)
if os.path.exists(_CUDA_LIB):
    ctypes.CDLL(_CUDA_LIB, mode=_RTLD_GLOBAL)
ctypes.CDLL(_PLUGIN_LIB, mode=_RTLD_GLOBAL)

import electrodecharge


CONVERSION_NM_BOHR = 18.8973
CONVERSION_KJMOL_NM_AU = CONVERSION_NM_BOHR / 2625.5
CONVERSION_EV_KJMOL = 96.487
TWO_OVER_FOUR_PI = 2.0 / (4.0 * math.pi)
ONE_OVER_FOUR_PI = 1.0 / (4.0 * math.pi)


class ElectrodeChargeReferenceTest(unittest.TestCase):
    def setUp(self):
        plugin_root = os.path.abspath(
            os.path.join(os.path.dirname(__file__), os.pardir, "build")
        )
        Platform.loadPluginsFromDirectory(plugin_root)
        self.system_params = {
            "cathode_indices": [0, 1],
            "anode_indices": [2, 3],
            "cathode_voltage": 1.0,
            "anode_voltage": 1.0,
            "num_iterations": 3,
            "small_threshold": 1.0e-6,
            "l_gap": 0.8,
            "l_cell": 2.0,
        }
        self.positions = [
            Vec3(0.0, 0.0, 0.1),
            Vec3(0.8, 0.0, 0.1),
            Vec3(0.0, 0.0, 1.9),
            Vec3(0.8, 0.0, 1.9),
            Vec3(0.4, 0.4, 1.0),
        ]
        self.box_vectors = (
            Vec3(1.6, 0.0, 0.0),
            Vec3(0.0, 1.6, 0.0),
            Vec3(0.0, 0.0, 2.0),
        )

    def _build_system(self, with_plugin):
        system = System()
        num_particles = len(self.positions)
        for _ in range(num_particles):
            system.addParticle(40.0)
        nonbonded = NonbondedForce()
        for i in range(num_particles):
            charge = 0.0
            if i == 4:
                charge = 0.1
            nonbonded.addParticle(charge, 1.0, 0.0)
        system.addForce(nonbonded)

        electrode_force = None
        if with_plugin:
            electrode_force = electrodecharge.ElectrodeChargeForce()
            electrode_force.setCathode(
                self.system_params["cathode_indices"],
                self.system_params["cathode_voltage"],
            )
            electrode_force.setAnode(
                self.system_params["anode_indices"],
                self.system_params["anode_voltage"],
            )
            electrode_force.setNumIterations(self.system_params["num_iterations"])
            electrode_force.setSmallThreshold(self.system_params["small_threshold"])
            electrode_force.setCellGap(self.system_params["l_gap"])
            electrode_force.setCellLength(self.system_params["l_cell"])
            system.addForce(electrode_force)
        return system, nonbonded, electrode_force

    def _set_initial_state(self, context):
        context.setPositions(self.positions)
        context.setPeriodicBoxVectors(*self.box_vectors)
        context.setVelocitiesToTemperature(300 * unit.kelvin)

    def _collect_charges(self, nonbonded):
        charges = []
        for index in range(nonbonded.getNumParticles()):
            charge, sigma, epsilon = nonbonded.getParticleParameters(index)
            charges.append(float(charge))
        return charges

    def _compute_sheet_area(self, box_vectors):
        (ax, ay, az), (bx, by, bz) = box_vectors[0], box_vectors[1]
        cross_x = ay * bz - az * by
        cross_y = az * bx - ax * bz
        cross_z = ax * by - ay * bx
        return math.sqrt(cross_x * cross_x + cross_y * cross_y + cross_z * cross_z)

    def _run_reference_solver(self, context, nonbonded):
        params = self.system_params
        cathode_indices = params["cathode_indices"]
        anode_indices = params["anode_indices"]
        num_iterations = params["num_iterations"]
        small_threshold = params["small_threshold"]

        state = context.getState(getPositions=True)
        positions_unit = state.getPositions()
        box_vectors_unit = state.getPeriodicBoxVectors()
        positions = [
            (
                float(pos.x / unit.nanometer),
                float(pos.y / unit.nanometer),
                float(pos.z / unit.nanometer),
            )
            for pos in positions_unit
        ]
        box_vectors = [
            (
                float(vec.x / unit.nanometer),
                float(vec.y / unit.nanometer),
                float(vec.z / unit.nanometer),
            )
            for vec in box_vectors_unit
        ]

        sheet_area = self._compute_sheet_area(box_vectors)
        if sheet_area == 0.0:
            raise AssertionError("Sheet area must be non-zero for the test system")

        cathode_z = positions[cathode_indices[0]][2]
        anode_z = positions[anode_indices[0]][2]

        charges = self._collect_charges(nonbonded)
        sigma_cache = {}
        epsilon_cache = {}
        for idx in cathode_indices:
            charge, sigma, epsilon = nonbonded.getParticleParameters(idx)
            sigma_cache[idx] = float(sigma)
            epsilon_cache[idx] = float(epsilon)
        for idx in anode_indices:
            charge, sigma, epsilon = nonbonded.getParticleParameters(idx)
            sigma_cache[idx] = float(sigma)
            epsilon_cache[idx] = float(epsilon)

        cathode_voltage_kj = params["cathode_voltage"] * CONVERSION_EV_KJMOL
        anode_voltage_kj = params["anode_voltage"] * CONVERSION_EV_KJMOL

        for _ in range(num_iterations):
            state = context.getState(getForces=True)
            forces_unit = state.getForces()
            forces = [
                (
                    float(force.x / (unit.kilojoule_per_mole / unit.nanometer)),
                    float(force.y / (unit.kilojoule_per_mole / unit.nanometer)),
                    float(force.z / (unit.kilojoule_per_mole / unit.nanometer)),
                )
                for force in forces_unit
            ]

            cathode_charges = []
            anode_charges = []

            cathode_area = sheet_area / float(len(cathode_indices))
            anode_area = sheet_area / float(len(anode_indices))

            cathode_target = (
                ONE_OVER_FOUR_PI
                * sheet_area
                * ((cathode_voltage_kj / params["l_gap"]) + (cathode_voltage_kj / params["l_cell"]))
                * CONVERSION_KJMOL_NM_AU
            )
            anode_target = (
                -ONE_OVER_FOUR_PI
                * sheet_area
                * ((anode_voltage_kj / params["l_gap"]) + (anode_voltage_kj / params["l_cell"]))
                * CONVERSION_KJMOL_NM_AU
            )

            cathode_set = set(cathode_indices)
            anode_set = set(anode_indices)

            for index, charge in enumerate(charges):
                if index in cathode_set or index in anode_set:
                    continue
                z_pos = positions[index][2]
                cathode_distance = abs(z_pos - anode_z)
                anode_distance = abs(z_pos - cathode_z)
                cathode_target += (cathode_distance / params["l_cell"]) * (-charge)
                anode_target += (anode_distance / params["l_cell"]) * (-charge)

            for idx in cathode_indices:
                charge = charges[idx]
                ez_external = 0.0
                if abs(charge) > 0.9 * small_threshold:
                    ez_external = forces[idx][2] / charge
                new_charge = (
                    TWO_OVER_FOUR_PI
                    * cathode_area
                    * ((cathode_voltage_kj / params["l_gap"]) + ez_external)
                    * CONVERSION_KJMOL_NM_AU
                )
                if abs(new_charge) < small_threshold:
                    new_charge = small_threshold
                cathode_charges.append(new_charge)

            for idx in anode_indices:
                charge = charges[idx]
                ez_external = 0.0
                if abs(charge) > 0.9 * small_threshold:
                    ez_external = forces[idx][2] / charge
                new_charge = (
                    -TWO_OVER_FOUR_PI
                    * anode_area
                    * ((anode_voltage_kj / params["l_gap"]) + ez_external)
                    * CONVERSION_KJMOL_NM_AU
                )
                if abs(new_charge) < small_threshold:
                    new_charge = -small_threshold
                anode_charges.append(new_charge)

            cathode_total = sum(cathode_charges)
            if abs(cathode_total) > small_threshold:
                scale = cathode_target / cathode_total
                if scale > 0.0:
                    cathode_charges = [value * scale for value in cathode_charges]

            anode_total = sum(anode_charges)
            if abs(anode_total) > small_threshold:
                scale = anode_target / anode_total
                if scale > 0.0:
                    anode_charges = [value * scale for value in anode_charges]

            for offset, idx in enumerate(cathode_indices):
                nonbonded.setParticleParameters(
                    idx,
                    cathode_charges[offset],
                    sigma_cache[idx],
                    epsilon_cache[idx],
                )
                charges[idx] = cathode_charges[offset]
            for offset, idx in enumerate(anode_indices):
                nonbonded.setParticleParameters(
                    idx,
                    anode_charges[offset],
                    sigma_cache[idx],
                    epsilon_cache[idx],
                )
                charges[idx] = anode_charges[offset]
            nonbonded.updateParametersInContext(context)
        return charges

    def test_reference_kernel_matches_python_solver(self):
        system_plugin, nonbonded_plugin, _ = self._build_system(with_plugin=True)
        integrator_plugin = LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds)
        context_plugin = Context(system_plugin, integrator_plugin)
        self._set_initial_state(context_plugin)

        # Trigger the plugin's iterative update
        context_plugin.getState(getForces=True)
        charges_plugin = self._collect_charges(nonbonded_plugin)

        system_ref, nonbonded_ref, _ = self._build_system(with_plugin=False)
        integrator_ref = LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picosecond, 0.001 * unit.picoseconds)
        context_ref = Context(system_ref, integrator_ref)
        self._set_initial_state(context_ref)

        charges_reference = self._run_reference_solver(context_ref, nonbonded_ref)

        self.assertEqual(len(charges_plugin), len(charges_reference))
        for idx, (plugin_charge, ref_charge) in enumerate(zip(charges_plugin, charges_reference)):
            self.assertEqual(
                plugin_charge,
                ref_charge,
                msg=f"Charge mismatch at index {idx}: plugin={plugin_charge} ref={ref_charge}",
            )

        # Invoke the plugin a second time to ensure recursion guard stability
        context_plugin.getState(getForces=True)
        charges_second = self._collect_charges(nonbonded_plugin)
        self.assertEqual(charges_plugin, charges_second)

        integrator_plugin.dispose()
        integrator_ref.dispose()
        context_plugin.dispose()
        context_ref.dispose()


if __name__ == "__main__":
    unittest.main()
