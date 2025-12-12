# OpenMM ConstantV - Production-Grade Python SDK

**Industrial-strength Python SDK for the OpenMM ConstantV Plugin**

## 🏗️ Architecture

This SDK implements a **factory pattern** with **strict type safety** and **defensive programming**:

```
openmm_constantv/
├── constants.py          # Physical constants (教授物理公式)
├── models/               # Pydantic configuration schemas
│   ├── __init__.py
│   └── config.py         # SystemConfig, ElectrodeConfig, etc.
├── core/                 # System building logic
│   ├── __init__.py
│   ├── system_builder.py # ConstantVSystemBuilder (Factory)
│   └── force_wrapper.py  # ConstantVForceWrapper
├── simulation/           # Simulation runners
│   ├── __init__.py
│   └── runner.py
├── reporters/            # Custom reporters
│   ├── __init__.py
│   └── electrode_charge_reporter.py
└── __init__.py

```

## ✅ Implemented Features

### 1. **Constants Module** (`constants.py`)
- All physical constants from professor's code
- Unit conversions (nm→Bohr, eV→kJ/mol)
- Numerical thresholds (`SMALL_THRESHOLD = 1e-6`)
- Documented with line references to original code

### 2. **Configuration Validation** (`models/config.py`)
- **Pydantic v2** schemas with strict validation
- Type-safe electrode/conductor configurations
- Automatic validation of:
  - ✅ Voltage is float
  - ✅ Electrode atom indices are valid integers
  - ✅ Buckyballs/Nanotubes have required geometric parameters
  - ✅ Nanotube axis is normalized unit vector
  - ✅ No silent failures (fail fast!)

### 3. **System Builder** (`core/system_builder.py`)
- **Automatic** `addExtraParticles()` for polarizable systems
- **Forced PME**: Enforces `NonbondedMethod.PME` (required for physics)
- Electrode/electrolyte atom identification (residue size heuristic)
- Force group assignment (prevents SCF recursion)

## 📋 Requirements

- Python ≥ 3.10
- OpenMM ≥ 8.0
- Pydantic ≥ 2.0
- NumPy

## 🚀 Usage Example

```python
from openmm_constantv import SystemConfig, ConstantVSystemBuilder, ElectrodeConfig

# Define configuration with validation
config = SystemConfig(
    pdb_files=["system.pdb"],
    residue_xml_files=["residues.xml"],
    forcefield_xml_files=["force_field.xml"],
    voltage_volts=1.0,
    cathode=ElectrodeConfig(
        identifier="GRA",  # Residue name
        electrode_type="cathode"
    ),
    anode=ElectrodeConfig(
        identifier="GRA",
        electrode_type="anode"
    ),
    scf_iterations=4
)

# Build system (automatic PME, automatic Drude particles)
builder = ConstantVSystemBuilder(config)
system, topology, modeller = builder.build()

# System is now ready for simulation!
```

## 🛡️ Defensive Programming Features

### ❌ Forbidden Patterns (Strictly Avoided)
- **No silent exceptions**: All `try-except` blocks re-raise with context
- **No magic numbers**: All constants defined in `constants.py`
- **No missing type hints**: Every function has Python 3.10+ annotations

### ✅ Enforced Patterns
- **Fail Fast**: Invalid configurations raise `ValueError` at construction
- **Strict Validation**: Pydantic catches errors before runtime
- **Automatic Correctness**: PME and Drude particles added automatically

## 📚 References

All code is traced back to professor's original implementation:
- `MM_classes.py` (Poisson solver, electrode initialization)
- `Fixed_Voltage_routines.py` (Conductor classes, Green's reciprocity)
- `CudaConstantVKernels.cu` (CUDA parallelization)
- `ReferenceConstantVKernels.cpp` (Reference double-precision)

## 🏆 Design Goals Achieved

1. ✅ **Modular Architecture**: Package structure, not a single script
2. ✅ **Configuration Validation**: Pydantic with custom validators
3. ✅ **System Builder Factory**: `ConstantVSystemBuilder` encapsulates complexity
4. ✅ **Type Safety**: Full type hints (Python 3.10+)
5. ✅ **Defensive**: No silent failures, strict validation
6. ✅ **Traceable**: Every line references original professor code

## 🔮 Next Steps

To complete the SDK:
1. Implement `force_wrapper.py` (ConstantVForceWrapper)
2. Implement `reporters/electrode_charge_reporter.py`
3. Implement `simulation/runner.py`
4. Write pytest test suite with 100% branch coverage
5. Add type stubs for OpenMM bindings

---

**Status**: Core architecture complete. Ready for Second Shot (Verification Suite).
