
"""
This script demonstrates the IDEAL workflow, combining the 'openmm_constantv' SDK
for system building and the 'openMM_constantV_plugin' for high-performance execution.

This is the "brains + brawn" approach.
"""
import sys
from openmm.app import *
from openmm import *
from openmm.unit import *

# --- Part 1: The "Brains" ---
# Use the high-level SDK from 'openmm_constantv' to configure and build the system.
# We need to add its path to import the modules.
sys.path.insert(0, '/home/andy/test_optimization/openmm_constantv')
from openmm_constantv.models.config import ConstantVConfig, ElectrodeConfig, ForceFieldConfig, SimulationConfig, SystemCreationConfig
from openmm_constantv.core.system_builder import ConstantVSystemBuilder

# --- Part 2: The "Brawn" ---
# Use the high-performance compiled components from 'openMM_constantV_plugin'.
# Assuming the plugin is installed, we can import it directly.
from constantv.constantvplugin import ConstantVLangevinIntegrator, ConstantVForce

# --- Main Combined Workflow ---

def run_ideal_simulation():
    """
    Sets up and runs the simulation using the combined SDK and Plugin workflow.
    """
    print("--- Ideal Workflow: Combining the SDK (Brains) and Plugin (Brawn) ---")

    # --- Step 1: Configure the System using the SDK's Pydantic models ---
    # This provides type safety and clear structure for your settings.
    print("Step 1: Defining configuration using the 'openmm_constantv' SDK...")
    config = ConstantVConfig(
        pdb_file='for_openmm.pdb',
        simulation_config=SimulationConfig(
            temperature=300,  # in Kelvin
            pressure=1.0,     # in bar
            timestep=2.0,     # in femtoseconds
            platform='CUDA'
        ),
        forcefield_config=ForceFieldConfig(
            ff_files=[
                'ffdir/sapt.xml',
                'ffdir/sapt_residues.xml',
                'ffdir/graph_c.xml',
                'ffdir/graph_residue_c.xml',
            ]
        ),
        system_creation_config=SystemCreationConfig(
            nonbonded_method='PME',
            nonbonded_cutoff=1.0,  # in nanometers
            constraints='HBonds'
        ),
        electrode_config=ElectrodeConfig(
            voltage=1.0,  # in volts
            cathode_identifier=(0, 2),
            anode_identifier=(1, 3)
        )
    )
    print("Configuration created successfully.")

    # --- Step 2: Build the System using the SDK's SystemBuilder ---
    # The builder takes the complex config and returns a standard OpenMM system.
    print("\nStep 2: Building the system with 'ConstantVSystemBuilder'...")
    builder = ConstantVSystemBuilder(config)
    system, topology, positions = builder.build()
    print("System, Topology, and Positions built successfully.")

    # At this point, the ConstantVForce has ALREADY been added to the system by the builder.
    # We can verify this.
    found_force = any(isinstance(f, ConstantVForce) for f in system.getForces())
    print(f"Verification: 'ConstantVForce' found in system? -> {found_force}")

    # --- Step 3: Create the Integrator from the Plugin ---
    print("\nStep 3: Creating 'ConstantVLangevinIntegrator' from the plugin...")
    integrator = ConstantVLangevinIntegrator(
        config.simulation_config.temperature * kelvin,
        1.0 / picosecond,
        config.simulation_config.timestep * femtoseconds
    )
    print("Integrator created.")

    # --- Step 4: Run the Simulation with standard OpenMM tools ---
    print("\nStep 4: Setting up and running the simulation...")
    platform = Platform.getPlatformByName(config.simulation_config.platform)
    simulation = Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    # Minimize energy
    print("Performing energy minimization...")
    simulation.minimizeEnergy()

    # Add reporters for output
    total_steps = 100000  # Example run
    report_interval = 1000
    simulation.reporters.append(
        StateDataReporter(
            sys.stdout,
            report_interval,
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=total_steps,
            separator='\t'
        )
    )
    simulation.reporters.append(DCDReporter('trajectory_combined.dcd', report_interval))
    print(f"Reporters added. Starting simulation for {total_steps} steps.")

    # Run!
    simulation.step(total_steps)

    print("\n--- Simulation finished successfully using the combined workflow! ---")

if __name__ == "__main__":
    run_ideal_simulation()
