import numpy
from MM_classes import MM
try:
    import electrode_charges_cython as ec_cython
    CYTHON_AVAILABLE = True
    print("✅ Cython module loaded successfully for MM_classes_CYTHON!")
except ImportError:
    CYTHON_AVAILABLE = False
    print("⚠️ Cython module not found. Falling back to NumPy implementation.")

class MM_CYTHON(MM):
    """
    Cython-accelerated MM class.
    Inherits all logic from the unified MM class and only overrides the
    core charge calculation method to call the compiled Cython function.
    """
    def _calculate_new_charges(self, forces_z, indices, q_old, prefactor, voltage_term, sign):
        """
        Overrides the base method to use the faster Cython implementation.
        If the Cython module is not available, it gracefully falls back to the
        parent's NumPy implementation.
        """
        if CYTHON_AVAILABLE:
            return ec_cython.calculate_new_charges_cython(
                forces_z,
                indices,
                q_old,
                prefactor,
                voltage_term,
                self.small_threshold,
                sign
            )
        else:
            return super()._calculate_new_charges(forces_z, indices, q_old, prefactor, voltage_term, sign)
