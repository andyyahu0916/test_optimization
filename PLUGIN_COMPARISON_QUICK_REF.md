# Plugin vs Original: Quick Reference Card

## Side-by-Side Code Comparison

### 1. Basic Setup

| Step | Original Python | Plugin C++ (via Python) |
|------|----------------|------------------------|
| **Import** | `from MM_classes import MM` <br> `from Fixed_Voltage_routines import *` | `from constantvplugin import ConstantVIntegrator` <br> `from constantvplugin_helpers import *` |
| **Create integrator** | `MMsys = MM(pdb_list, residue_xml, ff_xml)` | `system = forcefield.createSystem(...)` <br> `integrator = ConstantVIntegrator(timestep)` |
| **Set voltage** | (passed to `initialize_electrodes()`) | `integrator.setVoltage(0.0)` |

---

### 2. Electrode Setup

| Step | Original Python (Lines) | Plugin C++ (Lines) |
|------|------------------------|-------------------|
| **Initialize** | `MMsys.initialize_electrodes(` <br> `    Voltage,` <br> `    cathode_identifier=(0,2),` <br> `    anode_identifier=(1,3),` <br> `    chain=True,` <br> `    exclude_element=("H",)` <br> `)` <br> **(1 call)** | `for chain in topology.chains():` <br> `    if chain.index in [0, 2]:  # Cathode` <br> `        for atom in chain.atoms():` <br> `            if atom.element.symbol != 'H':` <br> `                integrator.addCathodeAtom(atom.index, area)` <br> `    elif chain.index in [1, 3]:  # Anode` <br> `        for atom in chain.atoms():` <br> `            if atom.element.symbol != 'H':` <br> `                integrator.addAnodeAtom(atom.index, area)` <br> **(~10 lines + loops)** |
| **Electrolyte** | `MMsys.initialize_electrolyte(Natom_cutoff=100)` <br> **(1 call)** | `add_electrolyte_atoms_auto(` <br> `    topology, system, integrator,` <br> `    nonbonded_force, natom_cutoff=100,` <br> `    exclude_chains=[0, 1, 2, 3]` <br> `)` <br> **(1 call to helper)** |
| **Exclusions** | `MMsys.generate_exclusions(flag_SAPT_FF_exclusions=True)` <br> **(1 call)** | `add_electrode_exclusions(integrator, nonbonded_force, custom_nb)` <br> `context = mm.Context(system, integrator)` <br> `context.setPositions(positions)` <br> `context.reinitialize(preserveState=True)  # ⚠️ CRITICAL` <br> **(3 calls)** |
| **Geometry** | (Automatic in `initialize_electrodes()`) | `temp_context = mm.Context(system, mm.VerletIntegrator(timestep))` <br> `temp_context.setPositions(positions)` <br> `configure_geometry_from_context(` <br> `    temp_context, integrator,` <br> `    cathode_atoms[0], anode_atoms[0]` <br> `)` <br> `del temp_context` <br> **(4+ lines)** |

---

### 3. Main Simulation Loop

| Step | Original Python (Lines 160-167) | Plugin C++ (Lines) |
|------|--------------------------------|-------------------|
| **SCF + MD loop** | `for i in range(traj_output_iterations):` <br> `    for j in range(charge_update_iterations):` <br> `        MMsys.Poisson_solver_fixed_voltage(Niterations=4)` <br> `        MMsys.simmd.step(freq_charge_update_fs)` <br> `    if write_charges:` <br> `        MMsys.write_electrode_charges(chargeFile)` <br> **(6 lines, nested loops)** | `integrator.setNumSCFIterations(4)` <br> `scf_freq = int(freq_charge_update_fs / timestep_fs)` <br> `integrator.setSCFFrequency(scf_freq)` <br> `simulation.step(num_steps)` <br> **(4 lines, single call)** |

---

### 4. Buckyball Support

| Step | Original Python (Line 81, 108) | Plugin C++ |
|------|-------------------------------|-----------|
| **Add Buckyball** | `BuckyBalls = [1]  # Chain index` <br> `MMsys.initialize_electrodes(..., BuckyBalls=BuckyBalls)` <br> **(2 lines)** | `# ⚠️ C++ API exists but no Python helper yet!` <br> `# Manual setup required:` <br> `virtual_atoms = [...]  # Collect indices` <br> `real_atoms = [...]     # Collect indices` <br> `force = ConstantVForce()` <br> `force.addBuckyballConductor(` <br> `    virtual_atoms, real_atoms, "cathode", voltage` <br> `)` <br> **(Manual topology parsing)** |

---

### 5. Charge Output

| Step | Original Python (Line 167) | Plugin C++ |
|------|---------------------------|-----------|
| **Write charges** | `MMsys.write_electrode_charges(chargeFile)` <br> **(1 call)** | `# ⚠️ Not implemented - manual workaround:` <br> `class ChargeReporter:` <br> `    def report(self, simulation, state):` <br> `        for idx in cathode_atoms:` <br> `            q, sig, eps = nonbonded_force.getParticleParameters(idx)` <br> `            file.write(f"{q._value} ")` <br> `        # ... repeat for anode ...` <br> `simulation.reporters.append(ChargeReporter(...))` <br> **(~15 lines custom class)** |

---

## Parameter Conversion Table

| Original Parameter | Value | Plugin Parameter | Value | Conversion Formula |
|-------------------|-------|------------------|-------|-------------------|
| `Voltage` | `0.0` Volts | `setVoltage(v)` | `0.0` | 1:1 |
| `Niterations` | `4` | `setNumSCFIterations(n)` | `4` | 1:1 |
| `freq_charge_update_fs` | `200` fs | `setSCFFrequency(f)` | `200` steps | `f = freq_fs / timestep_fs` |
| `simulation_time_ns` | `0.5` ns | `step(n)` | `500000` steps | `n = time_ns * 1e6 / timestep_fs` |
| `freq_traj_output_ps` | `10` ps | Reporter arg | `10000` steps | `n = freq_ps * 1000 / timestep_fs` |
| `Natom_cutoff` | `100` | Helper arg | `100` | 1:1 |
| `exclude_element` | `("H",)` | User filter | `if atom.element.symbol != 'H'` | Manual check |

---

## Feature Support Quick Lookup

| Feature | Symbol | Meaning |
|---------|--------|---------|
| ✅ | Full support | Works out-of-box |
| ⚠️ | Partial support | Works with workaround/extra code |
| 🔄 | Different API | Feature exists but API changed |
| ❌ | Not supported | Cannot be done |

| Feature | Original | Plugin | Migration Notes |
|---------|----------|--------|-----------------|
| Flat electrodes | ✅ | ✅ | Same physics, different API |
| Buckyballs | ✅ | ⚠️ | C++ API exists, no Python helper |
| Nanotubes | ✅ | ❌ | **BLOCKER** - not implemented |
| Constant V MD | ✅ | ✅ | Simpler API (no manual SCF calls) |
| MC equilibration | ✅ | ❌ | **BLOCKER** - use hybrid workflow |
| CUDA platform | ✅ | ✅ | Same performance |
| Reference platform | ✅ | ✅ | Validation/debugging |
| CPU platform | ✅ | ⚠️ | May fall back to Reference |
| Residue name ID | ✅ | 🔄 | Must convert to indices manually |
| Chain index ID | ✅ | ✅ | Preferred method |
| Auto-exclusions | ✅ | ⚠️ | Helper available, must call reinitialize() |
| Auto-geometry | ✅ | ⚠️ | Helper available |
| Charge output | ✅ | ❌ | Custom reporter needed |
| QM/MM | ✅ | ❌ | Not in scope |

---

## Common Errors and Fixes

| Error | Symptom | Fix |
|-------|---------|-----|
| **Forgot `reinitialize()`** | Energy explosion, atoms fly apart | Add `context.reinitialize(preserveState=True)` after creating Context |
| **Wrong SCF frequency units** | Charges update incorrectly | Convert: `scf_freq_steps = freq_charge_update_fs / timestep_fs` |
| **Missing cathode/anode atoms** | Charges all zero | Check loops add all atoms, use `validate_setup()` |
| **Wrong area calculation** | Quantitative charge errors | Use `compute_electrode_area_per_atom()` helper |
| **Missing electrolyte atoms** | Charges don't respond to electrolyte | Use `add_electrolyte_atoms_auto()` with system arg |
| **Excluded wrong elements** | Too many/few electrode atoms | Check filter: `if atom.element.symbol != 'H'` |

---

## File Organization

| Purpose | Original Python | Plugin |
|---------|----------------|--------|
| **Main script** | `run_openMM.py` | `example_usage.py` |
| **Core class** | `MM_classes.py` | `ConstantVIntegrator` (C++) |
| **Conductor classes** | `Fixed_Voltage_routines.py` | `ConstantVForce` (C++) |
| **Helpers** | Built into classes | `constantvplugin_helpers.py` |
| **Force field** | `ffdir/*.xml` | Same |
| **PDB input** | `nvt_0V_15ns.pdb` | Same |

---

## Performance Notes

- **CUDA**: Both ~500 ns/day (comparable)
- **Reference**: Both ~50 ns/day (slow, debugging only)
- **CPU**: Original ~200 ns/day, Plugin unclear (may fall back to Reference)

**Plugin advantage**: Maintainability, extensibility, cleaner API
**Original advantage**: Feature completeness (MC, Nanotubes, charge output)

---

## Migration Time Estimates

| User Type | Setup | Effort | Total Time |
|-----------|-------|--------|------------|
| **Basic (flat electrodes)** | CUDA GPU | Write loops + helpers | 2-4 hours |
| **Advanced (Buckyballs)** | CUDA GPU | Manual Buckyball setup | 1-2 days |
| **Nanotube researcher** | Any | **BLOCKED** | N/A |
| **MC + MD workflow** | Any | Hybrid workflow setup | 4-8 hours |

---

## Decision Matrix

| Your Workflow | Can Migrate? | Recommendation |
|---------------|--------------|----------------|
| Flat electrodes + Constant V MD + CUDA | ✅ YES | Migrate (simpler API) |
| Flat electrodes + Constant V MD + CPU | ⚠️ MAYBE | Test Reference platform first |
| Buckyballs + Constant V MD + CUDA | ⚠️ MAYBE | Wait for Python helper OR code manually |
| Nanotubes + any | ❌ NO | Stay on Original |
| Any + MC equilibration | ⚠️ MAYBE | Use hybrid workflow |
| Any + charge trajectory needed | ⚠️ MAYBE | Implement custom reporter |
| Any + QM/MM | ❌ NO | Stay on Original |

---

**Print this page for quick reference during migration!**
