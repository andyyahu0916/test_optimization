# Phase 1: Physics Core & CUDA Implementation

## 🚨 Critical Issues
1.  **Nanotube Kernel Atom Limit (Logic/Segfault Risk)**
    *   **Location:** `openmm_core_integration/platforms/cuda/src/kernels/constantVDrudeLangevin.cu` (Line 989 vs Kernel definition).
    *   **Problem:** The kernel `updateNanotubeChargesKernel` is launched with a fixed grid configuration `<<<1, 256>>>` in host code, but the kernel source uses `int i = blockIdx.x * blockDim.x + threadIdx.x` without a stride loop.
    *   **Impact:** If a nanotube has more than 256 atoms (very likely), atoms with index $\ge$ 256 will **never be updated**. Their charges will remain static, completely breaking the physical simulation for large nanotubes.
    *   **Fix:** Change the kernel to use a grid-stride loop: `for (int i = threadIdx.x; i < tube.numAtoms; i += blockDim.x)`.

## ⚠️ Performance Warnings
1.  **JIT Compiler Constant Memory Limit**
    *   **Location:** `openmm_core_integration/kernel_compiler.py`
    *   **Problem:** The script bakes `cathode_indices` directly into `__constant__` memory. CUDA `__constant__` memory is limited to 64KB.
    *   **Impact:** If `num_cathodes` exceeds approx. 16,000 (assuming 4-byte ints), the kernel compilation will fail or behave unpredictably.
    *   **Fix:** Add a check in `kernel_compiler.py` to switch to global memory (texture memory or read-only cache `__restrict__`) if the size exceeds 64KB.

# Phase 2: C++ Glue & Memory Management

## 🚨 Critical Issues
1.  **Broken Integrator Logic (Non-Functional)**
    *   **Location:** `ConstantVDrudeLangevinIntegrator.cpp`, `step()` method.
    *   **Issue:** The method contains a `TODO` and calls `DrudeLangevinIntegrator::step(1)`, but **never calls the custom kernel** to update charges. The SCF loop is completely missing in the C++ logic, meaning the simulation will run as standard Drude/Langevin *without* constant voltage.
    *   **Fix:** The `IntegrateConstantVDrudeLangevinStepKernel` must be retrieved from the `Platform` and executed.

2.  **Missing Kernel Stream Management**
    *   **Location:** `CudaConstantVKernels.cpp`
    *   **Issue:** Kernels are launched on the default stream (implicitly), but OpenMM uses specific streams for execution. This causes serialization and potential race conditions with other OpenMM kernels.
    *   **Fix:** Use `cu.getCurrentStream()` in the kernel launch configuration.

3.  **"Lazy Upload" Trap**
    *   **Location:** `CudaConstantVKernels.cpp`
    *   **Issue:** While `uploadElectrodeDataToGPU` exists, the logic to trigger it relies on `numBuckyballs > 0`. If the system is re-initialized or parameters change without adding new buckyballs, the main `ElectrodeData` struct (containing pointers to cathode/anode arrays) might point to stale addresses if those arrays were reallocated.

4.  **Destructor Double Free / Use-After-Free Risk**
    *   **Location:** `openmm_core_integration/platforms/cuda/src/CudaConstantVKernels.cpp`, destructor `~CudaCalcConstantVKernel`.
    *   **Problem:** The destructor iterates over `conductorArrays` (vector of `CudaArray*`) and deletes them. However, `conductorArrays` contains pointers that might *also* be referenced or managed elsewhere if the `addBuckyballConductor` logic isn't perfectly encapsulated.
    *   **Specific Check:** In `addBuckyballConductor`, `virtualIndicesGPU`, `realIndicesGPU`, `normalsGPU` are added to `conductorArrays`. If `addBuckyballConductor` is called multiple times or if `initialize` is called twice (resetting vectors), we might have dangling pointers or leaks. *Correction*: `initialize` sets `hasInitialized` but doesn't clear `conductorArrays`. This looks safe for a single run, but `initialize` shouldn't be called twice.

5.  **Invalid Device Pointer Usage in Host Code (Zero-Copy Trap)**
    *   **Location:** `openmm_core_integration/platforms/cuda/src/CudaConstantVKernels.cpp`, `initialize` method.
    *   **Problem:** `hostElectrodeData.cathodeIndices` is set to `(int*)cathodeIndicesGPU->getDevicePointer()`. This is a **Device Pointer**.
    *   **Context:** This struct `hostElectrodeData` is then uploaded to the GPU via `electrodeDataGPU->upload(&hostElectrodeData, 1)`.
    *   **Verdict:** This is **actually correct** for this specific architecture. The Host creates a struct containing Device Pointers, then copies that struct to the Device. The Device Kernel then reads the struct (from Device Memory) and follows the pointers (to Device Memory).
    *   **Risk:** This pattern is fragile. If `cathodeIndicesGPU` is reallocated (e.g., if context is reinitialized), the pointer inside `electrodeDataGPU` becomes stale unless `uploadElectrodeDataToGPU()` is called again. The code seems to handle this via `uploadElectrodeDataToGPU`, but it relies on explicit calls.

# Phase 3: Python SDK & System Build

## 🚨 Critical Issues
1.  **Race Condition in Force Group Assignment**
    *   **Location:** `system_builder.py`, `_assign_force_groups` method.
    *   **Issue:** The method blindly assigns force groups using `i % 32`. `ConstantVForce` uses a hardcoded force group `31` (defined in `constants.py` as `CONSTANTV_FORCE_GROUP`).
    *   **Impact:** If the system has 32+ forces (rare but possible with custom forces) or if the modulo logic assigns group 31 to another force, `ConstantVForce` will collide. More importantly, `ConstantVForce` expects to run in a specific way relative to other forces during SCF.
    *   **Fix:** Reserve group 31 explicitly. Iterate `i` from 0 to 30 only. Ensure `ConstantVForce` (if present) is *always* in group 31 (or whatever `CONSTANTV_FORCE_GROUP` is set to).

2.  **SWIG Interface Memory Leak Risk (Vectors)**
    *   **Location:** `ConstantVPlugin.i`
    *   **Issue:** The interface uses `std::vector<double>&` for output parameters in `getElectrodeCharges`.
    *   **Problem:** Python lists cannot be passed by reference to fill C++ vectors directly without typemaps. SWIG defaults might try to copy *in*, but not copy *out* if the typemap isn't `INOUT` or `ARGNOUT`.
    *   **Fix:** Use `%apply` or specific helper methods that return a Python list/tuple.

3.  **Factory Pattern Side Effect**
    *   **Location:** `system_builder.py`
    *   **Issue:** `create_constantv_force` checks `if self.constantv_force is not None: return self.constantv_force`.
    *   **Problem:** If `build()` is called multiple times, it returns the stale force object.
    *   **Fix:** Reset `self.constantv_force = None` at the start of `build()`.