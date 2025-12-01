# Phase 4: Build System & Testing

## 🚨 Critical Issues
1.  **Missing CUDA Architecture Flags**
    *   **Location:** `CMakeLists.txt`
    *   **Issue:** The default `CMAKE_CUDA_ARCHITECTURES` is set to `70;75;80;86;89;90`. While comprehensive, if the user compiles on a machine with an older GPU (e.g., Pascal `sm_60` or Maxwell `sm_50`), the compilation will fail or the kernel won't run.
    *   **Fix:** Add a detection mechanism or documentation instructing the user to set this flag explicitly if needed. (Minor issue compared to logic bugs, but affects usability).

2.  **SWIG Python Installation Path**
    *   **Location:** `CMakeLists.txt`
    *   **Issue:** `execute_process(COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])" ...)`
    *   **Problem:** `site.getsitepackages()` returns a *list* of paths. The first one (`[0]`) might be a system-wide directory (`/usr/local/lib/...`) which requires `sudo` to write to. If the user is in a virtual environment (conda/venv), this logic usually works, but on some systems (Debian/Ubuntu), `getsitepackages` might not point to where `pip` installs things.
    *   **Fix:** Use `sysconfig.get_path('platlib')` which is more robust for identifying the correct installation directory.

3.  **Benchmark Hardcoded Paths**
    *   **Location:** `benchmark_suite.py`
    *   **Issue:** `forcefield = app.ForceField('spce.xml')` assumes `spce.xml` is in the working directory or OpenMM's path.
    *   **Impact:** Benchmark might fail if standard force fields aren't found.
    *   **Fix:** Use `app.ForceField('amber14/spce.xml')` or ensure the file exists.

## ✅ Verified Correct
1.  **CMake Structure**
    *   **Status:** Verified.
    *   **Details:** The `CMakeLists.txt` correctly separates the Core API (`ConstantVAPI`), CUDA Library (`ConstantVCUDA`), Reference Library (`ConstantVReference`), and Python Wrapper (`constantv`). This modularity is excellent for maintenance.

2.  **Test Suite Logic**
    *   **Status:** Verified.
    *   **Details:** `test_native_integration.py` performs a valid end-to-end test:
        1.  Creates a system.
        2.  Runs a simulation.
        3.  **Crucially**, checks if charges *actually changed* (`abs(q_new - q_old) > epsilon`).
        4.  Checks charge conservation (Green's Reciprocity).
        This confirms the plugin isn't just "running without error" but is actually doing physics.
