# ConstantV Plugin - Complete Usage Guide

**Version:** Baseline 1.0 (Exact replica of Original Python code for flat electrodes)
**Author:** Based on Professor's published code at `/home/andy/test_optimization/OpenMM-ConstantV(original)`

---

## ⚠️ CRITICAL WARNING - READ THIS FIRST

### YOU MUST ADD ELECTRODE EXCLUSIONS

**Without electrode exclusions, your simulation WILL produce incorrect results or crash!**

Electrode atoms have the same sign charge (all positive for cathode, all negative for anode). Without exclusions:
- ❌ They will electrostatically repel each other
- ❌ Produce huge non-physical forces
- ❌ Cause energy to explode
- ❌ Lead to simulation instability

**The Original Python code automatically handles this. The plugin requires you to explicitly call:**

```python
from constantvplugin_helpers import add_electrode_exclusions

# BEFORE creating context:
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)

# AFTER creating context:
context.reinitialize(preserveState=True)  # ← CRITICAL!
```

**See [Quick Start](#quick-start) below for complete workflow.**

---

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Detailed Step-by-Step Guide](#detailed-step-by-step-guide)
4. [Helper Functions](#helper-functions)
5. [Comparison with Original Python](#comparison-with-original-python)
6. [Validation and Troubleshooting](#validation-and-troubleshooting)
7. [Limitations](#limitations)
8. [References](#references)

---

## Installation

### Build and Install the Plugin

```bash
cd ConstantVPlugin
mkdir build
cd build

# Configure
cmake ..

# Build
make

# Install C++ libraries
sudo make install  # Or make install with appropriate permissions

# Install Python bindings
make PythonInstall
```

### Verify Installation

```python
python3 -c "from constantvplugin import ConstantVIntegrator; print('✓ Plugin installed successfully')"
```

---

## Quick Start

**See `python/example_usage.py` for a complete working example.**

### Minimal Working Example (5 Critical Steps)

```python
import openmm as mm
import openmm.app as app
import openmm.unit as unit
from constantvplugin import ConstantVIntegrator
from constantvplugin_helpers import add_electrode_exclusions

# 1. Load system (standard OpenMM)
pdb = app.PDBFile('system.pdb')
forcefield = app.ForceField('forcefield.xml')
system = forcefield.createSystem(pdb.topology, ...)

# 2. Create ConstantVIntegrator
integrator = ConstantVIntegrator(1.0*unit.femtoseconds)
integrator.setVoltage(0.0)  # Voltage in Volts
integrator.setNumSCFIterations(4)

# 3. Add electrode atoms (example: by chain index)
for chain in pdb.topology.chains():
    if chain.index == 0:  # Cathode
        for atom in chain.atoms():
            if atom.element.symbol != 'H':
                integrator.addCathodeAtom(atom.index, area_per_atom)

# 4. ⚠️ CRITICAL: Add exclusions BEFORE creating context
nonbonded_force = [f for f in system.getForces()
                   if isinstance(f, mm.NonbondedForce)][0]
add_electrode_exclusions(integrator, nonbonded_force, None)

# 5. Create context and reinitialize
context = mm.Context(system, integrator)
context.setPositions(pdb.positions)
context.reinitialize(preserveState=True)  # ⚠️ CRITICAL!

# Run simulation
integrator.step(1000)
```

**For a complete example with all parameters, see [`python/example_usage.py`](ConstantVPlugin/python/example_usage.py).**

---

## Detailed Step-by-Step Guide

### Step 1: Load System (Standard OpenMM)

```python
import openmm as mm
import openmm.app as app
import openmm.unit as unit

pdb = app.PDBFile('system.pdb')
forcefield = app.ForceField('forcefield.xml')

system = forcefield.createSystem(
    pdb.topology,
    nonbondedCutoff=1.4*unit.nanometers,
    constraints=app.HBonds,
    rigidWater=True
)
```

### Step 2: Create ConstantVIntegrator

```python
from constantvplugin import ConstantVIntegrator

# Create integrator with timestep
timestep = 1.0 * unit.femtoseconds
integrator = ConstantVIntegrator(timestep)

# Set voltage (in Volts, converted internally to kJ/mol)
voltage_volts = 0.0
integrator.setVoltage(voltage_volts)

# Set SCF parameters (Original defaults)
integrator.setNumSCFIterations(4)     # Number of self-consistent iterations
integrator.setSCFFrequency(1)         # Update charges every N MD steps
```

**Original reference:** `run_openMM.py` Line 163 uses `Niterations=4` by default.

### Step 3: Add Electrode Atoms

You must manually identify and add electrode atoms. Two common approaches:

#### Method A: By Chain Index (like Original)

```python
# Identify electrodes by chain index (like Original run_openMM.py:78-79)
CATHODE_CHAIN = 0
ANODE_CHAIN = 1
EXCLUDE_ELEMENT = 'H'  # Exclude hydrogens

cathode_atoms = []
anode_atoms = []

for chain in pdb.topology.chains():
    if chain.index == CATHODE_CHAIN:
        for atom in chain.atoms():
            if atom.element.symbol != EXCLUDE_ELEMENT:
                cathode_atoms.append(atom.index)
    elif chain.index == ANODE_CHAIN:
        for atom in chain.atoms():
            if atom.element.symbol != EXCLUDE_ELEMENT:
                anode_atoms.append(atom.index)
```

#### Method B: By Residue Name

```python
cathode_atoms = []
for res in pdb.topology.residues():
    if res.name == 'CATH':  # Your cathode residue name
        for atom in res.atoms():
            if atom.element.symbol != 'H':
                cathode_atoms.append(atom.index)
```

#### Compute Area Per Atom

For flat electrodes, area per atom is the sheet area divided by number of atoms:

```python
from constantvplugin_helpers import compute_electrode_area_per_atom

cathode_area_per_atom, total_area = compute_electrode_area_per_atom(
    pdb.topology, cathode_atoms
)
```

**Or manually:**

```python
# Get box vectors
box_vectors = pdb.topology.getPeriodicBoxVectors()

# Cross product of a × b (sheet area)
a = box_vectors[0]
b = box_vectors[1]
cross = [
    a[1]*b[2] - a[2]*b[1],
    a[2]*b[0] - a[0]*b[2],
    a[0]*b[1] - a[1]*b[0]
]
total_area_nm2 = ((cross[0]**2 + cross[1]**2 + cross[2]**2)**0.5).value_in_unit(unit.nanometer**2)

area_per_atom = total_area_nm2 / len(cathode_atoms)
```

**Original reference:** `Electrode_Virtual.__init__()` Lines 256-259

#### Add to Integrator

```python
# Add cathode atoms
for atom_idx in cathode_atoms:
    integrator.addCathodeAtom(atom_idx, cathode_area_per_atom)

# Add anode atoms
for atom_idx in anode_atoms:
    integrator.addAnodeAtom(atom_idx, anode_area_per_atom)
```

### Step 4: Add Electrolyte Atoms

#### Manual Method

```python
nonbonded_force = [f for f in system.getForces()
                   if isinstance(f, mm.NonbondedForce)][0]

for atom_idx in electrolyte_atom_indices:
    charge, sigma, epsilon = nonbonded_force.getParticleParameters(atom_idx)
    integrator.addElectrolyteAtom(atom_idx, charge)
```

#### Automatic Method (Recommended)

Uses the Original's logic: residues with < 100 atoms are electrolyte.

```python
from constantvplugin_helpers import add_electrolyte_atoms_auto

electrolyte_atoms = add_electrolyte_atoms_auto(
    pdb.topology,
    integrator,
    nonbonded_force,
    natom_cutoff=100,  # Original default
    exclude_chains=[CATHODE_CHAIN, ANODE_CHAIN]  # Exclude electrode chains
)
```

**Original reference:** `MM_classes.py::initialize_electrolyte()` Lines 256-279

### Step 5: Set Geometry Parameters

#### Manual Method

```python
# You must calculate these from your system
integrator.setLgap(Lgap_nm)         # Vacuum gap (nm)
integrator.setLcell(Lcell_nm)       # Electrode separation (nm)
integrator.setTotalArea(area_nm2)   # Sheet area (nm²)
integrator.setZCathode(z_cath_nm)   # Cathode z position (nm)
integrator.setZAnode(z_anod_nm)     # Anode z position (nm)
```

#### Automatic Method (Recommended)

```python
from constantvplugin_helpers import configure_geometry_from_context

# Create temporary context to get positions
temp_integrator = mm.VerletIntegrator(timestep)
temp_context = mm.Context(system, temp_integrator)
temp_context.setPositions(pdb.positions)

# Auto-configure
params = configure_geometry_from_context(
    temp_context,
    integrator,
    cathode_atoms[0],  # First cathode atom
    anode_atoms[0]     # First anode atom
)

del temp_context, temp_integrator
```

**Original reference:** `MM_classes.py::set_electrochemical_cell_parameters()` Lines 229-245

### Step 6: ⚠️ CRITICAL - Add Electrode Exclusions

**This is the MOST IMPORTANT step. Do NOT skip!**

```python
from constantvplugin_helpers import add_electrode_exclusions

# Get forces
nonbonded_force = [f for f in system.getForces()
                   if isinstance(f, mm.NonbondedForce)][0]

# Get CustomNonbondedForce (if using SAPT-FF or similar)
custom_nb_forces = [f for f in system.getForces()
                    if isinstance(f, mm.CustomNonbondedForce)]
custom_nonbonded_force = custom_nb_forces[0] if custom_nb_forces else None

# Add exclusions
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)
```

**What this does:**
- Adds exclusions to CustomNonbondedForce (if present)
- Adds zero-interaction exceptions to NonbondedForce
- Prevents electrode atoms from interacting with each other

**Original reference:** `MM_classes.py::generate_exclusions()` Lines 570-590

### Step 7: Create Context and Reinitialize

```python
# Create context
platform = mm.Platform.getPlatformByName('CUDA')  # Or 'CPU', 'Reference'
properties = {'Precision': 'mixed'}
context = mm.Context(system, integrator, platform, properties)
context.setPositions(pdb.positions)

# ⚠️ CRITICAL: Reinitialize to apply exclusions
context.reinitialize(preserveState=True)
```

**Why reinitialize?**
Exclusions added after system creation require reinitialization to take effect.

**Original reference:** `MM_classes.py` Line 621

### Step 8: Run Simulation

```python
# Set initial velocities
context.setVelocitiesToTemperature(300*unit.kelvin)

# Run simulation
# The integrator automatically:
# 1. Calls SCF solver every SCF_FREQUENCY steps
# 2. Updates electrode charges
# 3. Performs MD integration with updated charges
num_steps = 1000000  # 1 ns at 1 fs timestep
integrator.step(num_steps)
```

**Original reference:** `run_openMM.py` Lines 163-164
- Line 163: `MMsys.Poisson_solver_fixed_voltage(Niterations=4)`
- Line 164: `MMsys.simmd.step(freq_charge_update_fs)`

The plugin integrator does both automatically in the correct order.

---

## Helper Functions

All helper functions are in `constantvplugin_helpers`:

### `add_electrode_exclusions(constantv_obj, nonbonded_force, custom_nonbonded_force=None)`

**Purpose:** Add exclusions between electrode atoms (CRITICAL!)

**Original:** `electrode_sapt_exclusions.py::exclusion_Electrode_NonbondedForce()` Lines 28-66

**Usage:**
```python
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)
```

### `configure_geometry_from_context(context, integrator, cathode_atom_idx, anode_atom_idx)`

**Purpose:** Auto-compute geometry parameters (Lgap, Lcell, totalArea, z positions)

**Original:** `MM_classes.py::set_electrochemical_cell_parameters()` Lines 229-245

**Usage:**
```python
params = configure_geometry_from_context(context, integrator, cathode_atoms[0], anode_atoms[0])
```

**Returns:** Dictionary with keys `'Lgap'`, `'Lcell'`, `'totalArea'`, `'z_cathode'`, `'z_anode'`

### `add_electrolyte_atoms_auto(topology, integrator, nonbonded_force, natom_cutoff=100, exclude_chains=None)`

**Purpose:** Auto-identify electrolyte atoms (residues with < natom_cutoff atoms)

**Original:** `MM_classes.py::initialize_electrolyte()` Lines 256-279

**Usage:**
```python
electrolyte_atoms = add_electrolyte_atoms_auto(
    pdb.topology, integrator, nonbonded_force,
    natom_cutoff=100,
    exclude_chains=[0, 1]  # Electrode chains
)
```

**Returns:** List of electrolyte atom indices

### `compute_electrode_area_per_atom(topology, electrode_atom_indices)`

**Purpose:** Compute area per atom for flat electrode

**Original:** `Electrode_Virtual.__init__()` Lines 256-259

**Usage:**
```python
area_per_atom, total_area = compute_electrode_area_per_atom(pdb.topology, cathode_atoms)
```

**Returns:** Tuple of (area_per_atom, total_area) in nm²

### `validate_setup(context, integrator)`

**Purpose:** Validate that setup is correct before running

**Usage:**
```python
valid, messages = validate_setup(context, integrator)
if not valid:
    print("Setup errors found!")
    for msg in messages:
        print(msg)
```

**Returns:** Tuple of (valid: bool, messages: list of str)

---

## Comparison with Original Python

| Feature | Original Python | Plugin | Notes |
|---------|----------------|--------|-------|
| **Core Physics (Flat Electrodes)** | | | |
| Initial charge formula | ✅ | ✅ | Identical |
| SCF charge update | ✅ | ✅ | Identical (2.0 coefficient) |
| Analytic charge (Green's) | ✅ | ✅ | Identical |
| Charge normalization | ✅ | ✅ | Identical |
| All constants | ✅ | ✅ | Identical |
| SCF→Force→MD order | ✅ | ✅ | Identical |
| **Exclusions** | | | |
| Electrode exclusions | ✅ Auto | ✅ Manual | Must call helper function |
| CustomNonbonded | ✅ | ✅ | Supported |
| Context reinitialize | ✅ Auto | ✅ Manual | Must call explicitly |
| **Usability** | | | |
| Auto-geometry | ✅ | ✅ | Via helper function |
| Auto-electrolyte | ✅ | ✅ | Via helper function |
| Charge output | ✅ | ❌ | Future feature |
| **Out of Scope** | | | |
| Conductors (Buckyball/Nanotube) | ✅ | ❌ | Excluded |
| MC equilibration | ✅ | ❌ | Excluded |
| QM/MM interface | ✅ | ❌ | Excluded |

---

## Validation and Troubleshooting

### How to Validate Your Simulation

#### 1. Check Charge Conservation

Total charge should be approximately zero:

```python
state = context.getState(getPositions=True)

# Get cathode total charge
Q_cathode = 0.0
for i in range(integrator.getNumCathodeAtoms()):
    particle, area = integrator.getCathodeAtomParameters(i)
    # Note: charge is stored in NonbondedForce, not returned here
    # This is just for demonstration

# Better: Use validatesetup helper
valid, messages = validate_setup(context, integrator)
```

#### 2. Monitor Energy

Potential energy should NOT explode. If it does:
- ❌ Exclusions are missing or not applied
- ❌ Geometry parameters are incorrect

```python
state = context.getState(getEnergy=True)
print(f"Potential energy: {state.getPotentialEnergy()}")
```

If energy is NaN, Inf, or grows rapidly (e.g., 1e10 kJ/mol), stop and check exclusions.

#### 3. Compare with Original (If Possible)

Run the same system with Original Python code and compare:
- Energies at each timestep
- Final trajectory
- Electrode charges over time

### Common Issues and Solutions

#### Issue: Simulation explodes / huge forces

**Symptoms:**
- Potential energy becomes NaN or Inf
- Atoms move very far very fast
- Simulation crashes

**Cause:** Exclusions not added or not applied

**Fix:**
```python
# Make sure you did BOTH:
add_electrode_exclusions(...)  # Before creating context
context.reinitialize(preserveState=True)  # After creating context
```

#### Issue: "AttributeError: 'ConstantVIntegrator' object has no attribute 'getNumCathodeAtoms'"

**Cause:** Plugin not installed or wrong version

**Fix:**
```bash
cd ConstantVPlugin/build
make PythonInstall
# Restart Python interpreter
```

#### Issue: Results don't match Original

**Possible causes:**
1. Different exclusions
2. Different geometry parameters
3. Missing electrolyte atoms
4. Different force field

**Debug:**
- Print all geometry parameters and compare
- Check number of electrode/electrolyte atoms
- Verify exclusions were added

#### Issue: "Charge not conserved" warnings

**Cause:** Incorrect geometry parameters or missing electrolyte atoms

**Fix:**
- Use `configure_geometry_from_context()` helper
- Use `add_electrolyte_atoms_auto()` helper
- Manually verify Lgap, Lcell values are reasonable

---

## Limitations

### Current Scope (Baseline v1.0)

✅ **Supported:**
- Flat electrodes (graphene, metal sheets, etc.)
- Polarizable and non-polarizable force fields
- CUDA, CPU, and Reference platforms
- SAPT-FF and other CustomNonbondedForce systems

❌ **NOT Supported (Out of Scope):**
- Conductors (Buckyball, Nanotube) on electrodes
- Monte Carlo equilibration
- QM/MM interface
- Dynamic box size (NPT with electrode movement)

### Known Differences from Original

1. **Manual exclusion addition:** Original does this automatically, plugin requires explicit call
2. **Manual geometry setup:** Original auto-computes, plugin provides helper function
3. **No diagnostic output:** Original can write charge time series, plugin doesn't yet

**These are intentional design choices for the baseline version.**

---

## References

### Original Python Code

Located at: `/home/andy/test_optimization/OpenMM-ConstantV(original)`

Key files:
- `run_openMM.py` - Main simulation script
- `lib/MM_classes.py` - Core algorithm (`Poisson_solver_fixed_voltage`)
- `lib/Fixed_Voltage_routines.py` - Electrode classes
- `lib/electrode_sapt_exclusions.py` - Exclusion system

### Plugin Documentation

- [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) - Detailed algorithm comparison
- [python/example_usage.py](ConstantVPlugin/python/example_usage.py) - Complete working example
- [python/helpers.py](ConstantVPlugin/python/helpers.py) - Helper function source code

### Papers

If you use this plugin, please cite:
- [Professor's original publication - TBD]

### Support

For issues:
1. Check [Validation and Troubleshooting](#validation-and-troubleshooting)
2. See [IMPLEMENTATION_AUDIT.md](IMPLEMENTATION_AUDIT.md) for algorithm details
3. Compare with Original Python code for reference

---

## Quick Reference Card

### Critical Steps Checklist

- [ ] Create ConstantVIntegrator
- [ ] Add cathode atoms with correct area
- [ ] Add anode atoms with correct area
- [ ] Add electrolyte atoms
- [ ] Set geometry parameters (Lgap, Lcell, etc.)
- [ ] **⚠️ Add electrode exclusions (BEFORE context)**
- [ ] Create context with positions
- [ ] **⚠️ Call context.reinitialize() (AFTER context)**
- [ ] Set velocities
- [ ] Run simulation

### Must-Have Code Snippets

```python
# 1. Import helpers
from constantvplugin_helpers import add_electrode_exclusions

# 2. Before creating context
add_electrode_exclusions(integrator, nonbonded_force, custom_nonbonded_force)

# 3. After creating context
context = mm.Context(system, integrator)
context.setPositions(positions)
context.reinitialize(preserveState=True)  # ← DON'T FORGET!
```

---

*Last updated: 2025-01-11*
*Based on Original Python code from Professor's lab*
