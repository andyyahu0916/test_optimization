#cython: language_level=3, boundscheck=False, wraparound=False, cdivision=True, initializedcheck=False

import numpy as np
cimport numpy as np

# A small, pure-C helper function could be defined here for speed if needed.

def calculate_new_charges_cython(
    np.ndarray[np.float64_t, ndim=1] forces_z,
    np.ndarray[np.int64_t, ndim=1] indices,
    np.ndarray[np.float64_t, ndim=1] q_old,
    double prefactor,
    double voltage_term,
    double small_threshold,
    double sign):
    """
    Cython implementation of the core charge calculation logic.
    This function is pure and operates only on NumPy arrays.
    """
    cdef np.ndarray[np.float64_t, ndim=1] ez_external = np.zeros_like(q_old)
    cdef np.ndarray[np.float64_t, ndim=1] q_new
    cdef Py_ssize_t i, idx
    cdef double q_val

    for i in range(q_old.shape[0]):
        q_val = q_old[i]
        if abs(q_val) > 0.9 * small_threshold:
            idx = indices[i]
            ez_external[i] = forces_z[idx] / q_val

    q_new = prefactor * (voltage_term + ez_external)

    for i in range(q_new.shape[0]):
        if abs(q_new[i]) < small_threshold:
            q_new[i] = small_threshold * sign

    return q_new
