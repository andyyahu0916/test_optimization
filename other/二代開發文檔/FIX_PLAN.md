# Execution Plan: Plugin Fixes

Based on the "Critical Audit" (`PLUGIN_AUDIT_REPORT.md`), the following fixes will be executed to align the Plugin with the Original Python "Golden Standard".

## 1. Critical Physics Fix: Nanotube Charge Transfer

**Problem:**
The C++ Plugin (`Reference` and `CUDA`) incorrectly uses the Buckyball formula for Nanotube charge transfer.
*   **Original (Python)**: `dQ = sign * dE * dr * length / 2.0` (Cylindrical area scaling).
*   **Plugin (C++)**: `dQ = sign * dE * dr * dr` (Spherical area scaling, copy-paste error).

**Fix:**
Modify `numericalChargeNanotube` in both kernels to use `conductor.length`.

**Target Files:**
*   `openMM_constantV_plugin/ConstantVPlugin/platforms/reference/src/ReferenceConstantVKernels.cpp`
*   `openMM_constantV_plugin/ConstantVPlugin/platforms/cuda/src/CudaConstantVKernels.cu`

## 2. Architectural Fix: Migration of Exclusions

**Problem:**
Electrode exclusions are currently handled in Python (`helpers.py`), making the C++ `ConstantVForce` unsafe if used standalone.

**Plan:**
Move `add_electrode_exclusions` logic into `ConstantVForce::createSystem` or similar initialization path. *Note: For this immediate task, I will focus on the critical physics fix first.*

## 3. MC Barostat Strategy

**Problem:**
The C++ Integrator lacks the Monte Carlo Barostat logic found in the Original's `MM_classes.py`.

**Plan:**
Since porting the entire MC logic to C++ is a major feature addition, the immediate fix is to **standardize the Python implementation**. The `helpers.py` implementation of `MC_Barostat` should be treated as the canonical solution and integrated into the example scripts.

---

**Immediate Action Items:**
1.  Apply the Nanotube formula correction to Reference C++.
2.  Apply the Nanotube formula correction to CUDA C++.
