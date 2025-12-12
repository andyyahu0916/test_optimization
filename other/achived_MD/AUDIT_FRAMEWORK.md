# The "30 Perspectives" Audit Framework

This document defines the multi-dimensional analytical strategy used to verify the `OpenMM-ConstantV(original)` (Golden Standard) against the C++ Plugin implementation.

## 1. The Physics & Mathematics Perspectives
1.  **Physical Laws Alignment:** Does the code strictly adhere to the underlying physical equations (e.g., Poisson-Boltzmann, Gauss's Law)?
2.  **Units & Constants:** Are units (kJ/mol, nm, V) and constants (Avogadro, dielectric) identical and correctly converted?
3.  **Green's Reciprocity:** Specifically checks if the reciprocity theorem ($Q_{induced}$ scaling) is implemented exactly as in the Python prototype.
4.  **Convergence Criteria:** Are the mathematical conditions for stopping the SCF loop (tolerance, error metric) identical?
5.  **Boundary Conditions:** How are periodic boundaries (PBC) or infinite slabs treated?
6.  **Precision & Numerics:** Float (single) vs. Double precision. Where does truncation occur? Does it affect physics?
7.  **Arithmetic Operations:** Integer division vs. float division, order of operations (associativity in float math).

## 2. The Data & Memory Perspectives
8.  **Data Flow:** Tracing a variable from input -> transformation -> output.
9.  **Memory Life-cycle:** Allocation, initialization, updates, and deallocation.
10. **Caching vs. Freshness:** (Crucial) Is a value cached when it should be recomputed? Is it stale?
11. **Indexing & Ordering:** 0-based vs 1-based, sorting orders, atom index mapping.
12. **Global vs. Local State:** Are variables modifying global state (side effects) or purely local?
13. **Buffer Boundaries:** Checks for off-by-one errors or buffer overflows (C++ specific).

## 3. The Logic & Control Flow Perspectives
14. **Branching Logic:** Are `if/else` conditions mathematically equivalent?
15. **Loop Mechanics:** Start indices, end indices, stride.
16. **Recursion vs. Iteration:** Handling of iterative solvers.
17. **Initialization State:** What is the exact state at $t=0$? (Charges, Potentials).
18. **Termination Logic:** How does the simulation or loop end?

## 4. The System & Architecture Perspectives
19. **Configuration Parsing:** How `config.ini` is read. Default values.
20. **Interface/API Parity:** Do the C++ kernels expose the same "knobs" as the Python classes?
21. **Concurrency & Parallelism:** Race conditions, barriers, atomic operations (CUDA specific).
22. **Hardware Specifics:** CPU (Python/NumPy) vs GPU (CUDA) intrinsic differences (e.g., `rsqrt`, fast math).
23. **Dependencies:** Does Python rely on implicit NumPy behavior that C++ must implement explicitly?

## 5. The "Meta" & Maintenance Perspectives
24. **Dead Code / "The Vase":** Functions present in Python but unused. Are they truly useless or safety nets?
25. **Logging & Observability:** Are debug prints and outputs comparable?
26. **Error Handling:** How are invalid states (e.g., $V=0$, infinite field) handled?
27. **Input Validation:** checks for impossible physics configurations.
28. **Magic Numbers:** Identification of hardcoded values (e.g., `0.0001`, `18.8973`).
29. **Naming & Semantics:** Do variable names reflect their physical meaning? (confusion risk).
30. **Code Structure:** Is the modularity (Classes/Functions) comparable, or is logic inlined?
