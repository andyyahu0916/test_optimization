#!/usr/bin/env python3
"""
============================================================================
Fixed-Voltage MD Simulation - Corrected Version with Exclusions
============================================================================
This script faithfully replicates run_openMM.py including electrode exclusions.
============================================================================
"""

from __future__ import print_function
import sys
import os
import shutil
import numpy as np
from datetime import datetime

from openmm.app import *
from openmm import *
from openmm.unit import *

# Physical Constants
conversion_KjmolNm_Au = 0.00719475
small_threshold = 1e-6
VOLTAGE_TO_KJMOL = 96.485

# Simulation Parameters
simulation_time_ns = 0.5
freq_charge_update_fs = 200
freq_traj_output_ps = 10

outPath = '1v_0.5ns_native'
if os.path.exists(outPath):
    shutil.rmtree(outPath)
os.mkdir(outPath)

Voltage = 0.0
cathode_chains = (0, 2)
anode_chains = (1, 3)
exclude_elements = ('H',)

ffdir = '/home/andy/test_optimization/OpenMM-ConstantV(original)/ffdir/'

# ============================================================================
# Electrode Class
# ============================================================================
class atom_MM:
    def __init__(self, element, charge, atom_index):
        self.element = element
        self.charge = charge
        self.atom_index = atom_index

class Electrode_Virtual:
    def __init__(self, topology, chain_indices, electrode_type, voltage, nbondedForce, exclude_element=('H',)):
        self.electrode_type = electrode_type
        self.Voltage = voltage * VOLTAGE_TO_KJMOL
        self.electrode_atoms = []
        self.Q_analytic = 0.0
        self.z_pos = 0.0
        
        for chain in topology.chains():
            if chain.index in chain_indices:
                for atom in chain.atoms():
                    if atom.element is not None and atom.element.symbol not in exclude_element:
                        q, sig, eps = nbondedForce.getParticleParameters(atom.index)
                        self.electrode_atoms.append(atom_MM(atom.element.symbol, q._value, atom.index))
        
        self.Natoms = len(self.electrode_atoms)
        print(f"  {electrode_type}: {self.Natoms} atoms, Voltage = {voltage} V")
    
    def get_total_charge(self):
        return sum([a.charge for a in self.electrode_atoms])
    
    def get_atom_indices(self):
        return [a.atom_index for a in self.electrode_atoms]
    
    def initialize_Charge(self, Lgap, Lcell, area_atom, nbondedForce):
        sign = 1.0 if self.electrode_type == 'cathode' else -1.0
        flag_small = abs(self.Voltage) < 0.01
        if flag_small:
            print("adding small value to initial charges in initialize_Charge routine for small Voltage input...")
        
        for atom in self.electrode_atoms:
            q_i = sign / (4.0 * np.pi) * area_atom * (self.Voltage / Lgap + self.Voltage / Lcell) * conversion_KjmolNm_Au
            if flag_small:
                q_i = q_i + sign * small_threshold
            atom.charge = q_i
            nbondedForce.setParticleParameters(atom.atom_index, q_i, 1.0, 0.0)
    
    def compute_Electrode_charge_analytic(self, positions, electrolyte_atoms, Lcell, z_opposite, nbondedForce, total_area, Lgap):
        sign = 1.0 if self.electrode_type == 'cathode' else -1.0
        self.Q_analytic = sign * (self.Voltage / Lgap) * total_area * conversion_KjmolNm_Au / (4.0 * np.pi)
        
        for idx in electrolyte_atoms:
            q, _, _ = nbondedForce.getParticleParameters(idx)
            z_atom = positions[idx][2]._value
            z_distance = abs(z_atom - z_opposite)
            self.Q_analytic += (z_distance / Lcell) * (-q._value)
    
    def Scale_charges_analytic(self, nbondedForce, print_flag=False):
        Q_numeric = self.get_total_charge()
        if print_flag:
            print(f"Q_numeric , Q_analytic charges on  {self.electrode_type} {Q_numeric} {self.Q_analytic}")
        
        scale_factor = -1
        if abs(Q_numeric) > small_threshold:
            scale_factor = self.Q_analytic / Q_numeric
        
        if scale_factor > 0.0:
            for atom in self.electrode_atoms:
                atom.charge *= scale_factor
                nbondedForce.setParticleParameters(atom.atom_index, atom.charge, 1.0, 0.0)

# ============================================================================
# Electrode Exclusions (CRITICAL - matching original)
# ============================================================================
def generate_electrode_exclusions(electrode_list, customNonbondedForce, nbondedForce):
    """
    Add exclusions for intra-electrode interactions.
    This prevents electrode atoms from interacting with each other.
    
    Matching original: electrode_sapt_exclusions.py::exclusion_Electrode_NonbondedForce
    """
    # Track existing exclusions
    existing = set()
    for i in range(customNonbondedForce.getNumExclusions()):
        p1, p2 = customNonbondedForce.getExclusionParticles(i)
        existing.add((min(p1, p2), max(p1, p2)))
    
    added_count = 0
    for i in range(len(electrode_list)):
        for j in range(i + 1, len(electrode_list)):
            idx_i = electrode_list[i]
            idx_j = electrode_list[j]
            pair = (min(idx_i, idx_j), max(idx_i, idx_j))
            
            if pair not in existing:
                customNonbondedForce.addExclusion(idx_i, idx_j)
                nbondedForce.addException(idx_i, idx_j, 0, 1, 0, True)
                existing.add(pair)
                added_count += 1
    
    return added_count

# ============================================================================
# System Setup
# ============================================================================
print("=" * 70)
print("Fixed-Voltage MD Simulation (with Electrode Exclusions)")
print("=" * 70)

residue_xml_list = [ffdir + 'sapt_residues.xml', ffdir + 'graph_residue_c.xml', ffdir + 'graph_residue_n.xml']
ff_xml_list = [ffdir + 'sapt_noDB_2sheets.xml', ffdir + 'graph_c_freeze.xml', ffdir + 'graph_n_freeze.xml']

for rf in residue_xml_list:
    Topology().loadBondDefinitions(rf)

pdb = PDBFile('/home/andy/test_optimization/OpenMM-ConstantV(original)/nvt_0V_15ns.pdb')
modeller = Modeller(pdb.topology, pdb.positions)
forcefield = ForceField(*ff_xml_list)
modeller.addExtraParticles(forcefield)

system = forcefield.createSystem(modeller.topology, nonbondedCutoff=1.2*nanometer, constraints=HBonds, rigidWater=True)

for i in range(system.getNumForces()):
    f = system.getForce(i)
    f.setForceGroup(i)
    if isinstance(f, (HarmonicBondForce, HarmonicAngleForce, PeriodicTorsionForce, RBTorsionForce)):
        f.setUsesPeriodicBoundaryConditions(True)

nbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if isinstance(f, NonbondedForce)][0]
nbondedForce.setNonbondedMethod(NonbondedForce.PME)

customNonbondedForce = [f for f in [system.getForce(i) for i in range(system.getNumForces())] if isinstance(f, CustomNonbondedForce)][0]
customNonbondedForce.setNonbondedMethod(min(nbondedForce.getNonbondedMethod(), NonbondedForce.CutoffPeriodic))

# ============================================================================
# Initialize Electrodes
# ============================================================================
print("\nInitializing electrodes...")
Cathode = Electrode_Virtual(modeller.topology, cathode_chains, "cathode", Voltage, nbondedForce, exclude_elements)
Anode = Electrode_Virtual(modeller.topology, anode_chains, "anode", Voltage, nbondedForce, exclude_elements)

# Geometry
positions = modeller.positions
boxVecs = modeller.topology.getPeriodicBoxVectors()
crossBox = np.cross([boxVecs[0][0]._value, boxVecs[0][1]._value, boxVecs[0][2]._value],
                    [boxVecs[1][0]._value, boxVecs[1][1]._value, boxVecs[1][2]._value])
total_area = np.linalg.norm(crossBox)
area_atom = total_area / Cathode.Natoms

cathode_z = np.mean([positions[a.atom_index][2]._value for a in Cathode.electrode_atoms])
anode_z = np.mean([positions[a.atom_index][2]._value for a in Anode.electrode_atoms])
Cathode.z_pos = cathode_z
Anode.z_pos = anode_z

Lcell = abs(anode_z - cathode_z)
box_z = boxVecs[2][2]._value
Lgap = box_z - Lcell

print(f"  Lcell: {Lcell:.3f} nm, Lgap: {Lgap:.3f} nm, Area: {total_area:.3f} nm²")

# Initialize charges
Cathode.initialize_Charge(Lgap, Lcell, area_atom, nbondedForce)
Anode.initialize_Charge(Lgap, Lcell, area_atom, nbondedForce)

# ============================================================================
# CRITICAL: Generate Electrode Exclusions
# ============================================================================
print("\nGenerating electrode exclusions...")
cathode_list = Cathode.get_atom_indices()
anode_list = Anode.get_atom_indices()

n_cathode = generate_electrode_exclusions(cathode_list, customNonbondedForce, nbondedForce)
n_anode = generate_electrode_exclusions(anode_list, customNonbondedForce, nbondedForce)
print(f"  Added {n_cathode} cathode exclusions, {n_anode} anode exclusions")

# Identify electrolyte atoms
all_electrode_atoms = set(cathode_list + anode_list)
electrolyte_atoms = []
for residue in modeller.topology.residues():
    if len(list(residue.atoms())) < 100:
        for atom in residue.atoms():
            if atom.index not in all_electrode_atoms:
                electrolyte_atoms.append(atom.index)
print(f"  Electrolyte atoms: {len(electrolyte_atoms)}")

# ============================================================================
# Create Integrator and Simulation
# ============================================================================
has_drude = any(isinstance(system.getForce(i), DrudeForce) for i in range(system.getNumForces()))
if has_drude:
    integrator = DrudeLangevinIntegrator(300*kelvin, 1/picosecond, 1*kelvin, 40/picosecond, 0.001*picosecond)
    integrator.setMaxDrudeDistance(0.02*nanometer)
    print("Using DrudeLangevinIntegrator")
else:
    integrator = LangevinMiddleIntegrator(300*kelvin, 1/picosecond, 0.001*picosecond)

platform = Platform.getPlatformByName('CUDA')
simulation = Simulation(modeller.topology, system, integrator, platform, {'Precision': 'mixed'})
simulation.context.setPositions(modeller.positions)

# CRITICAL: Reinitialize context after adding exclusions
print("Reinitializing context with exclusions...")
state = simulation.context.getState(getPositions=True)
positions_snapshot = state.getPositions()
simulation.context.reinitialize()
simulation.context.setPositions(positions_snapshot)
nbondedForce.updateParametersInContext(simulation.context)

state = simulation.context.getState(getEnergy=True, getForces=True, getPositions=True)
print(f"\n{state.getKineticEnergy()}")
print(f"{state.getPotentialEnergy()}")
for j in range(system.getNumForces()):
    f = system.getForce(j)
    print(f"{type(f).__name__}: {simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()}")

PDBFile.writeFile(simulation.topology, state.getPositions(), open(os.path.join(outPath, 'start_drudes.pdb'), 'w'))
simulation.reporters.append(DCDReporter(os.path.join(outPath, 'FV_NVT.dcd'), int(freq_traj_output_ps * 1000)))

# ============================================================================
# Poisson Solver
# ============================================================================
def Poisson_solver_fixed_voltage(Niterations=4):
    state = simulation.context.getState(getPositions=True)
    positions = state.getPositions()
    
    Cathode.compute_Electrode_charge_analytic(positions, electrolyte_atoms, Lcell, Anode.z_pos, nbondedForce, total_area, Lgap)
    Anode.compute_Electrode_charge_analytic(positions, electrolyte_atoms, Lcell, Cathode.z_pos, nbondedForce, total_area, Lgap)
    
    for i_iter in range(Niterations):
        state = simulation.context.getState(getForces=True)
        forces = state.getForces()
        
        for atom in Cathode.electrode_atoms:
            idx = atom.atom_index
            q_old = atom.charge
            Ez = (forces[idx][2]._value / q_old) if abs(q_old) > 0.9 * small_threshold else 0.0
            q_new = 2.0 / (4.0 * np.pi) * area_atom * (Cathode.Voltage / Lgap + Ez) * conversion_KjmolNm_Au
            if abs(q_new) < small_threshold:
                q_new = small_threshold
            atom.charge = q_new
            nbondedForce.setParticleParameters(idx, q_new, 1.0, 0.0)
        
        for atom in Anode.electrode_atoms:
            idx = atom.atom_index
            q_old = atom.charge
            Ez = (forces[idx][2]._value / q_old) if abs(q_old) > 0.9 * small_threshold else 0.0
            q_new = -2.0 / (4.0 * np.pi) * area_atom * (Anode.Voltage / Lgap + Ez) * conversion_KjmolNm_Au
            if abs(q_new) < small_threshold:
                q_new = -small_threshold
            atom.charge = q_new
            nbondedForce.setParticleParameters(idx, q_new, 1.0, 0.0)
        
        print_flag = (i_iter == Niterations - 1)
        Cathode.Scale_charges_analytic(nbondedForce, print_flag)
        Anode.Scale_charges_analytic(nbondedForce, print_flag)
        nbondedForce.updateParametersInContext(simulation.context)

# ============================================================================
# Run Simulation
# ============================================================================
print("\nStarting simulation...")

num_iterations = int(simulation_time_ns * 1000 / freq_traj_output_ps)
steps_per_scf = int(freq_charge_update_fs)
scf_calls_per_output = int(freq_traj_output_ps * 1000 / freq_charge_update_fs)

for i in range(num_iterations):
    state = simulation.context.getState(getEnergy=True)
    print(f"{i} iteration")
    print(f"{state.getKineticEnergy()}")
    print(f"{state.getPotentialEnergy()}")
    for j in range(system.getNumForces()):
        f = system.getForce(j)
        print(f"{type(f).__name__}: {simulation.context.getState(getEnergy=True, groups=2**j).getPotentialEnergy()}")
    
    for j in range(scf_calls_per_output):
        Poisson_solver_fixed_voltage(Niterations=4)
        simulation.step(steps_per_scf)

print("\ndone!")
