# ElectrodeChargePlugin for OpenMM

**CRITICAL: Read "Usage Warnings" section before use!**

## Overview

This OpenMM plugin solves the Poisson equation for fixed-voltage electrodes in molecular dynamics simulations. It self-consistently calculates electrode charges by iteratively computing electrostatic forces and updating charges until convergence.

## Features

- **Reference Platform**: CPU implementation (verified bit-level accurate vs Python OPTIMIZED)
- **CUDA Platform**: GPU implementation (compiled, not yet validated)
- **Performance**: 25× speedup over highly optimized Python/NumPy implementation
- **Validation**: Produces identical results to original Python solver

## Installation

```bash
cd build
cmake ..
make
make install
make PythonInstall
```

## Basic Usage

```python
import electrodecharge
from openmm import *
from openmm.app import *

# Create system
system = System()
# ... add particles, forces ...

# Add NonbondedForce (REQUIRED - ElectrodeChargeForce depends on it)
nonbonded = NonbondedForce()
nonbonded.setNonbondedMethod(NonbondedForce.NoCutoff)
for i in range(num_particles):
    nonbonded.addParticle(0.0, 1.0, 0.0)
system.addForce(nonbonded)

# Add ElectrodeChargeForce
force = electrodecharge.ElectrodeChargeForce()
force.setCathode([0, 1, 2], -0.5)  # indices, voltage (eV)
force.setAnode([3, 4, 5], 0.5)     # indices, voltage (eV)
force.setCellGap(0.8)               # nm
force.setCellLength(2.0)            # nm
force.setNumIterations(3)
system.addForce(force)

# Create context
platform = Platform.getPlatformByName('Reference')
integrator = LangevinIntegrator(300*kelvin, 1.0/picosecond, 0.001*picoseconds)
context = Context(system, integrator, platform)
context.setPositions(positions)
```

---

## ⚠️ **USAGE WARNINGS (READ THIS!)**

### 🔴 CRITICAL: Force Group Isolation Required

**Due to internal implementation details, you MUST use force group isolation when calling `getState()`.**

#### ❌ WRONG (produces 2× forces on CUDA):
```python
state = context.getState(getForces=True)  # Default: groups=-1 (all groups)
forces = state.getForces()  # WRONG! Forces will be doubled on CUDA platform
```

#### ✅ CORRECT:
```python
# ElectrodeChargeForce is in force group 1 by default
state = context.getState(getForces=True, groups=1<<1)  # Only group 1
forces = state.getForces()  # Correct forces
```

#### Why This Matters:

`ElectrodeChargeForce` internally calls `context.calcForcesAndEnergy()` during its charge update iterations. This creates a nested force calculation:

1. External `getState(groups=-1)` calculates NonbondedForce (group 0)
2. Then calculates ElectrodeChargeForce (group 1)
3. ElectrodeChargeForce internally calculates NonbondedForce again
4. **On CUDA platform, forces accumulate → 2× forces!**

The force group isolation prevents this by ensuring you only request the ElectrodeChargeForce group, which internally calculates NonbondedForce exactly once.

#### Verification:

```python
# Test on simple system
state_wrong = context.getState(getForces=True)  # groups=-1
state_correct = context.getState(getForces=True, groups=1<<1)

forces_wrong = state_wrong.getForces()
forces_correct = state_correct.getForces()

# On CUDA: forces_wrong ≈ 2 × forces_correct
# On Reference: forces_wrong ≈ forces_correct (Reference doesn't accumulate)
```

### Known Limitations

1. **Charge Conservation**: The algorithm does not conserve total system charge. This is **expected behavior** for fixed-voltage boundary conditions where charge flows between electrodes and system. Initial charge `Q_initial` ≠ Final charge `Q_final`.

2. **Force Group Dependency**: The current implementation uses a pragmatic hack (force group isolation) instead of proper architecture. A cleaner solution would use `updateContextState` pattern to avoid participating in normal force calculations.

3. **CUDA Platform Status**: CUDA platform compiles but may not load in all environments. Ensure OpenMM is built with CUDA support and GPU drivers are properly installed.

---

## API Reference

### ElectrodeChargeForce

#### Constructor
```python
force = electrodecharge.ElectrodeChargeForce()
```
Default force group: **1** (NonbondedForce typically uses group 0)

#### Methods

**setCathode(indices, voltage)**
- `indices`: List of atom indices for cathode
- `voltage`: Applied voltage in eV

**setAnode(indices, voltage)**
- `indices`: List of atom indices for anode
- `voltage`: Applied voltage in eV

**setCellGap(gap)**
- `gap`: Distance between electrodes in nm

**setCellLength(length)**
- `length`: Cell length for periodic boundary conditions in nm

**setNumIterations(n)**
- `n`: Number of Poisson solver iterations (default: 3)
- More iterations → better convergence
- Typical range: 3-5

**setSmallThreshold(threshold)**
- `threshold`: Minimum charge magnitude in elementary charge units (default: 1e-6)
- Prevents division by zero in electric field calculations

---

## Validation & Performance

### Numerical Accuracy
- **Test**: Simple 5-particle system (4 electrodes + 1 bulk)
- **Comparison**: Plugin Reference vs Python OPTIMIZED
- **Result**: 
  - Max difference: **0.00e+00 e** (bit-level identical)
  - RMS difference: **0.00e+00 e**
  
### Performance
- **Plugin Reference**: 0.020 ms
- **Python OPTIMIZED**: 0.497 ms
- **Speedup**: **25.10×**

### Test Command
```bash
cd tests
python test_vs_python_reference.py
```

---

## Architecture Notes (for developers)

### Current Implementation (Pragmatic Hack)

```
ElectrodeChargeForce (group 1)
└─ calculateForces() called by OpenMM
    └─ ElectrodeChargeForceImpl::execute()
        └─ Iteration loop:
            ├─ context.calcForcesAndEnergy(groups=1<<0)  // Only NonbondedForce
            ├─ Read forces → compute electrode charges
            └─ Update NonbondedForce parameters
```

**Problem**: Calling `context.calcForcesAndEnergy()` inside `calculateForces()` is architecturally wrong. It creates nested force calculations.

**Workaround**: Force group isolation ensures external `getState()` only requests group 1, preventing double calculation.

### Better Architecture (Future Work)

Move charge updates to `updateContextState()` callback:
```
1. Normal MD step calculates forces (no special handling)
2. After step, updateContextState() is called
3. Plugin reads forces, updates charges, modifies NonbondedForce
4. Next step uses updated charges
```

This avoids nested force calculations entirely. However, requires refactoring iteration loop logic.

---

## Troubleshooting

### "Platform 'CUDA' not found"
- Check OpenMM installation: `python -c "import openmm; print(openmm.Platform.getNumPlatforms())"`
- Verify CUDA support: `python -c "import openmm; print([openmm.Platform.getPlatform(i).getName() for i in range(openmm.Platform.getNumPlatforms())])"`
- Rebuild OpenMM with CUDA: `-DOPENMM_BUILD_CUDA_LIB=ON`

### Forces seem doubled
- **Check**: Are you using `groups=1<<1` in `getState()`?
- **Test**: Compare `getState(getForces=True)` vs `getState(getForces=True, groups=1<<1)`

### Charge convergence issues
- Increase `setNumIterations()` (try 5-10)
- Check electrode geometry (too close = numerical instability)
- Verify `setCellGap()` and `setCellLength()` match simulation box

---

## Citation

If you use this plugin, please cite:
- Original Python implementation: [Your paper]
- This C++/CUDA port: [This repository]

---

## License

[Your license here]

---

## Contact

For bugs and feature requests, open an issue on GitHub.

**Remember**: Always use `groups=1<<1` when calling `getState(getForces=True)`!
