/* ═══════════════════════════════════════════════════════════════════════════
 * SWIG Interface for ConstantV Native Core Integration
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This file exposes the ConstantV C++ API to Python:
 * 1. ConstantVForce - Force-based API (add to System like any Force)
 * 2. ConstantVIntegrator - Verlet integration with SCF updates
 * 3. ConstantVDrudeLangevinIntegrator - Drude Langevin with SCF updates
 *
 * Author: Claude (Anthropic)
 * License: MIT (compatible with OpenMM)
 */

%module constantv

%include "std_vector.i"
%include "std_string.i"

namespace std {
    %template(IntVector) vector<int>;
    %template(DoubleVector) vector<double>;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Vec3 Typemap Support
 * ═══════════════════════════════════════════════════════════════════════════
 * 
 * FIX P3-SWIG: Add Vec3 typemap support for addNanotubeConductor.
 * This allows Python to pass openmm.Vec3 objects to C++ methods expecting Vec3.
 */

// Fragment for converting Python sequence to Vec3
%fragment("Py_SequenceToVec3", "header", fragment="Py_StripOpenMMUnits") {
OpenMM::Vec3 Py_SequenceToVec3(PyObject* obj, int& status) {
    PyObject* s, *o, *o1;
    double x[3];
    int i, length;
    s = Py_StripOpenMMUnits(obj);
    if (s == NULL) {
        status = SWIG_ERROR;
        return OpenMM::Vec3(0, 0, 0);
    }
    if (PySequence_Check(s)) {
        length = PySequence_Size(s);
        if (length != 3) {
            Py_DECREF(s);
            status = SWIG_ERROR;
            return OpenMM::Vec3(0, 0, 0);
        }
        for (i = 0; i < 3; i++) {
            o = PySequence_GetItem(s, i);
            o1 = Py_StripOpenMMUnits(o);
            if (o1 == NULL) {
                Py_DECREF(s);
                Py_DECREF(o);
                status = SWIG_ERROR;
                return OpenMM::Vec3(0, 0, 0);
            }
            x[i] = PyFloat_AsDouble(o1);
            Py_DECREF(o);
            Py_DECREF(o1);
            if (PyErr_Occurred() != NULL) {
                Py_DECREF(s);
                status = SWIG_ERROR;
                return OpenMM::Vec3(0, 0, 0);
            }
        }
        Py_DECREF(s);
        status = SWIG_OK;
        return OpenMM::Vec3(x[0], x[1], x[2]);
    }
    Py_DECREF(s);
    status = SWIG_ERROR;
    return OpenMM::Vec3(0, 0, 0);
}
}

// Fragment for stripping OpenMM units
// FIX P3-SWIG: Use OpenMM's full implementation for unit stripping
%fragment("Py_StripOpenMMUnits", "header") {
PyObject* Py_StripOpenMMUnits(PyObject *input) {
    static PyObject *__s_Quantity = NULL;
    static PyObject *__s_md_unit_system_tuple = NULL;
    static PyObject *__s_bar_tuple = NULL;

    if (__s_Quantity == NULL) {
        PyObject* module = NULL;
        module = PyImport_ImportModule("openmm.unit");
        if (!module) {
            PyErr_SetString(PyExc_ImportError, "openmm.unit");
            Py_CLEAR(module);
            return NULL;
        }

        __s_Quantity = PyObject_GetAttrString(module, "Quantity");
        if (!__s_Quantity) {
            PyErr_SetString(PyExc_AttributeError, "'module' object has no attribute 'Quantity'");
            Py_CLEAR(module);
            Py_CLEAR(__s_Quantity);
            return NULL;
        }

        PyObject* bar = NULL;
        bar = PyObject_GetAttrString(module, "bar");
        if (!bar) {
            PyErr_SetString(PyExc_AttributeError, "'module' object has no attribute 'bar'");
            Py_CLEAR(module);
            Py_CLEAR(__s_Quantity);
            Py_CLEAR(bar);
            return NULL;
        }

        PyObject* md_unit_system = NULL;
        md_unit_system = PyObject_GetAttrString(module, "md_unit_system");
        if (!md_unit_system) {
            PyErr_SetString(PyExc_AttributeError, "'module' object has no attribute 'md_unit_system'");
            Py_CLEAR(module);
            Py_CLEAR(__s_Quantity);
            Py_CLEAR(bar);
            Py_CLEAR(md_unit_system);
            return NULL;
        }
        __s_md_unit_system_tuple = PyTuple_Pack(1, md_unit_system);
        __s_bar_tuple = PyTuple_Pack(1, bar);
        Py_DECREF(md_unit_system);
        Py_DECREF(bar);
        Py_DECREF(module);
    }
    PyObject *val;

    if (PyObject_IsInstance(input, __s_Quantity)) {
        PyObject* input_unit = NULL, *is_compatible = NULL, *compatible_with_bar = NULL;
        input_unit = PyObject_GetAttrString(input, "unit");
        is_compatible = PyObject_GetAttrString(input_unit, "is_compatible");
        compatible_with_bar = PyObject_Call(is_compatible, __s_bar_tuple, NULL);
        if (PyObject_IsTrue(compatible_with_bar)) {
            PyObject* value_in_unit = PyObject_GetAttrString(input, "value_in_unit");
            val = PyObject_Call(value_in_unit, __s_bar_tuple, NULL);
            Py_DECREF(value_in_unit);
        } else {
            PyObject* value_in_unit_system = PyObject_GetAttrString(input, "value_in_unit_system");
            val = PyObject_Call(value_in_unit_system, __s_md_unit_system_tuple, NULL);
            Py_DECREF(value_in_unit_system);
        }
        Py_CLEAR(input_unit);
        Py_CLEAR(is_compatible);
        Py_CLEAR(compatible_with_bar);
        if (PyErr_Occurred() != NULL) {
            return NULL;
        }
    } else {
        val = input;
        Py_INCREF(val);
    }
    return val;
}
}

// Typemap for const Vec3& (used in addNanotubeConductor)
%typemap(in, fragment="Py_SequenceToVec3") const Vec3& (OpenMM::Vec3 myVec, int res=0) {
    myVec = Py_SequenceToVec3($input, res);
    if (!SWIG_IsOK(res)) {
        PyErr_SetString(PyExc_ValueError, "in method $symname, argument $argnum could not be converted to type Vec3");
        SWIG_fail;
    }
    $1 = &myVec;
}

%typemap(typecheck, precedence=SWIG_TYPECHECK_DOUBLE_ARRAY, fragment="Py_SequenceToVec3") const Vec3& {
    int res = 0;
    Py_SequenceToVec3($input, res);
    $1 = SWIG_IsOK(res);
}

%{
#include "openmm/ConstantVForce.h"
#include "openmm/ConstantVIntegrator.h"
#include "openmm/ConstantVDrudeLangevinIntegrator.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include "openmm/Force.h"
#include "openmm/Integrator.h"
#include "OpenMM.h"
#include <vector>
#include <string>
%}

/* ═══════════════════════════════════════════════════════════════════════════
 * Import OpenMM SWIG bindings for proper inheritance
 * ═══════════════════════════════════════════════════════════════════════════
 */

// Forward declare OpenMM base classes with minimal interface
// This allows our classes to inherit from them without full reimplementation
namespace OpenMM {
    %nodefaultctor Force;
    %nodefaultdtor Force;
    class Force {
    public:
        int getForceGroup() const;
        void setForceGroup(int group);
    };

    %nodefaultctor Integrator;
    %nodefaultdtor Integrator;
    class Integrator {
    public:
        virtual void step(int steps) = 0;
        double getStepSize() const;
        void setStepSize(double size);
    };

    // DrudeLangevinIntegrator is required for our integrator inheritance
    %nodefaultctor DrudeLangevinIntegrator;
    %nodefaultdtor DrudeLangevinIntegrator;
    class DrudeLangevinIntegrator : public Integrator {
    public:
        double getTemperature() const;
        void setTemperature(double temp);
        double getFriction() const;
        void setFriction(double coeff);
        double getDrudeTemperature() const;
        void setDrudeTemperature(double temp);
        double getDrudeFriction() const;
        void setDrudeFriction(double coeff);
        double getMaxDrudeDistance() const;
        void setMaxDrudeDistance(double distance);
        void setRandomNumberSeed(int seed);
        int getRandomNumberSeed() const;
    };
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Exception Handling
 * ═══════════════════════════════════════════════════════════════════════════
 */

%exception {
    try {
        $action
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        SWIG_fail;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ConstantVForce - Force-based API
 * ═══════════════════════════════════════════════════════════════════════════
 */

%feature("docstring") OpenMM::ConstantVForce "
Force-based API for constant voltage simulations.

This Force can be added to any System and used with any Integrator.
Electrode charges are updated self-consistently via the SCF method.

Example:
--------
>>> force = constantv.ConstantVForce()
>>> force.setVoltage(1.0)  # 1.0 V
>>> force.setLgap(3.5)     # 3.5 nm
>>> force.setLcell(5.0)    # 5.0 nm
>>> force.setTotalArea(10.0)  # 10.0 nm²
>>> force.setNumIterations(4)
>>>
>>> # Add electrode atoms
>>> for i in cathode_atoms:
...     force.addCathodeAtom(i, area_per_atom)
>>> for i in anode_atoms:
...     force.addAnodeAtom(i, area_per_atom)
>>> for i in electrolyte_atoms:
...     force.addElectrolyteAtom(i, charge)
>>>
>>> system.addForce(force)
>>> context = Context(system, integrator)
";

%rename(ConstantVForce) OpenMM::ConstantVForce;

namespace OpenMM {

class ConstantVForce : public Force {
public:
    ConstantVForce();
    ~ConstantVForce();

    // Electrode atom management
    int addCathodeAtom(int particle, double area);
    int getNumCathodeAtoms() const;
    void getCathodeAtomParameters(int index, int& particle, double& area) const;
    void setCathodeAtomParameters(int index, int particle, double area);

    int addAnodeAtom(int particle, double area);
    int getNumAnodeAtoms() const;
    void getAnodeAtomParameters(int index, int& particle, double& area) const;
    void setAnodeAtomParameters(int index, int particle, double area);

    int addElectrolyteAtom(int particle, double charge);
    int getNumElectrolyteAtoms() const;
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;
    void setElectrolyteAtomParameters(int index, int particle, double charge);

    // Conductor management
    int addBuckyballConductor(const std::vector<int>& virtualAtoms,
                              const std::vector<int>& realAtoms,
                              const std::string& electrodeType,
                              double voltage);
    int getNumBuckyballConductors() const;

    int addNanotubeConductor(const std::vector<int>& virtualAtoms,
                             const std::vector<int>& realAtoms,
                             const std::string& electrodeType,
                             double voltage,
                             const std::vector<double>& axis);
    int getNumNanotubeConductors() const;

    // System parameters
    void setVoltage(double voltage);
    double getVoltage() const;

    void setLgap(double gap);
    double getLgap() const;

    void setLcell(double cell);
    double getLcell() const;

    void setTotalArea(double area);
    double getTotalArea() const;

    void setZCathode(double z);
    double getZCathode() const;

    void setZAnode(double z);
    double getZAnode() const;

    void setNumIterations(int n);
    int getNumIterations() const;
};

} // namespace OpenMM

/* ═══════════════════════════════════════════════════════════════════════════
 * ConstantVIntegrator - Verlet with SCF
 * ═══════════════════════════════════════════════════════════════════════════
 * Uses standard Verlet integration with CUDA-accelerated SCF charge updates.
 * This is the simplest integrator for testing CUDA SCF performance.
 */

%feature("docstring") OpenMM::ConstantVIntegrator "
Velocity Verlet integrator with constant voltage boundary conditions.

This integrator performs standard velocity Verlet integration while
periodically updating electrode charges via the SCF method.

Suitable for:
- NVE ensemble simulations
- Rigid water models
- Testing and validation

Example:
--------
>>> integrator = constantv.ConstantVIntegrator(0.001)  # 1 fs timestep
>>> integrator.setVoltage(1.0)
>>> integrator.setLgap(3.5)
>>> integrator.setLcell(5.0)
>>> integrator.setTotalArea(10.0)
>>> integrator.setNumSCFIterations(4)
>>> integrator.setSCFFrequency(1)  # Update every step
>>>
>>> for i in cathode_atoms:
...     integrator.addCathodeAtom(i, area_per_atom)
>>>
>>> context = Context(system, integrator)
>>> integrator.step(1000000)
";

%rename(ConstantVIntegrator) OpenMM::ConstantVIntegrator;

namespace OpenMM {

// Forward declare OpenMM::Integrator for inheritance
class Integrator {
public:
    double getStepSize() const;
    void setStepSize(double size);
    virtual void step(int steps) = 0;
};

class ConstantVIntegrator : public Integrator {
public:
    explicit ConstantVIntegrator(double stepSize);
    ~ConstantVIntegrator();

    // Physical parameters
    void setVoltage(double voltage);
    double getVoltage() const;

    void setLgap(double gap);
    double getLgap() const;

    void setLcell(double cell);
    double getLcell() const;

    void setTotalArea(double area);
    double getTotalArea() const;

    void setZCathode(double z);
    double getZCathode() const;

    void setZAnode(double z);
    double getZAnode() const;

    // SCF parameters
    void setNumSCFIterations(int n);
    int getNumSCFIterations() const;

    void setSCFFrequency(int freq);
    int getSCFFrequency() const;

    // Electrode atom management
    int addCathodeAtom(int particle, double area);
    int getNumCathodeAtoms() const;
    void getCathodeAtomParameters(int index, int& particle, double& area) const;
    void setCathodeAtomParameters(int index, int particle, double area);

    int addAnodeAtom(int particle, double area);
    int getNumAnodeAtoms() const;
    void getAnodeAtomParameters(int index, int& particle, double& area) const;
    void setAnodeAtomParameters(int index, int particle, double area);

    int addElectrolyteAtom(int particle, double charge);
    int getNumElectrolyteAtoms() const;
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;
    void setElectrolyteAtomParameters(int index, int particle, double charge);

    // Integrator interface
    void step(int steps);
};

} // namespace OpenMM

/* ═══════════════════════════════════════════════════════════════════════════
 * ConstantVDrudeLangevinIntegrator - Drude Langevin with SCF
 * ═══════════════════════════════════════════════════════════════════════════
 */

%feature("docstring") OpenMM::ConstantVDrudeLangevinIntegrator "
Drude Langevin integrator with constant voltage boundary conditions.

This integrator combines dual-temperature Langevin dynamics for polarizable
Drude oscillators with self-consistent field (SCF) electrode charge updates.

Suitable for:
- Polarizable force fields (Drude model)
- Constant temperature simulations
- Production runs

Example:
--------
>>> integrator = constantv.ConstantVDrudeLangevinIntegrator(
...     temperature=300.0,
...     frictionCoeff=1.0,
...     drudeTemperature=1.0,
...     drudeFrictionCoeff=20.0,
...     stepSize=0.001,
...     voltage=2.0,
...     Lgap=3.5,
...     Lcell=5.0,
...     scfIterations=4
... )
>>>
>>> integrator.setTotalArea(10.0)
>>> integrator.setSCFFrequency(1)
>>>
>>> for i in cathode_atoms:
...     integrator.addCathodeAtom(i, area_per_atom)
>>>
>>> context = Context(system, integrator)
>>> integrator.step(1000000)
";

%rename(ConstantVDrudeLangevinIntegrator) OpenMM::ConstantVDrudeLangevinIntegrator;

namespace OpenMM {

class ConstantVDrudeLangevinIntegrator : public DrudeLangevinIntegrator {
public:
    ConstantVDrudeLangevinIntegrator(
        double temperature,
        double frictionCoeff,
        double drudeTemperature,
        double drudeFrictionCoeff,
        double stepSize,
        double voltage,
        double Lgap,
        double Lcell,
        int scfIterations = 4
    );
    ~ConstantVDrudeLangevinIntegrator();

    // Electrode configuration
    void addCathodeAtom(int particle, double area);
    int getNumCathodeAtoms() const;

    void addAnodeAtom(int particle, double area);
    int getNumAnodeAtoms() const;

    void addElectrolyteAtom(int particle, double charge);
    int getNumElectrolyteAtoms() const;

    // Conductor support
    void addBuckyballConductor(
        const std::vector<int>& virtualIndices,
        const std::vector<int>& realIndices,
        const std::string& electrodeType,
        double voltage
    );
    int getNumBuckyballConductors() const;

    // FIX P3-SWIG: Expose addNanotubeConductor with Vec3 typemap support
    void addNanotubeConductor(
        const std::vector<int>& virtualIndices,
        const std::vector<int>& realIndices,
        const std::string& electrodeType,
        double voltage,
        const Vec3& axis
    );
    int getNumNanotubeConductors() const;

    // System geometry
    void setTotalArea(double area);
    double getTotalArea() const;

    void setZCathode(double z);
    double getZCathode() const;

    void setZAnode(double z);
    double getZAnode() const;

    // Voltage and SCF parameters
    void setVoltage(double v);
    double getVoltage() const;

    void setLgap(double gap);
    double getLgap() const;

    void setLcell(double cell);
    double getLcell() const;

    void setNumSCFIterations(int n);
    int getNumSCFIterations() const;

    void setSCFFrequency(int freq);
    int getSCFFrequency() const;

    // Query methods
    void getElectrodeCharges(
        std::vector<double>& cathodeCharges,
        std::vector<double>& anodeCharges
    ) const;

    double getTotalCathodeCharge() const;
    double getTotalAnodeCharge() const;

    // Integrator interface
    void step(int steps);
};

} // namespace OpenMM
