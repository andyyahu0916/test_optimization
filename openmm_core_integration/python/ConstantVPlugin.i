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

%{
#include "openmm/ConstantVForce.h"
#include "openmm/ConstantVIntegrator.h"
#include "openmm/ConstantVDrudeLangevinIntegrator.h"
#include "openmm/Context.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include <vector>
#include <string>
%}

/* ═══════════════════════════════════════════════════════════════════════════
 * Import OpenMM SWIG bindings
 * ═══════════════════════════════════════════════════════════════════════════
 */

%import(module="openmm") "swig/OpenMMSwigHeaders.i"

/* ═══════════════════════════════════════════════════════════════════════════
 * STL Container Support
 * ═══════════════════════════════════════════════════════════════════════════
 */

%include "std_vector.i"
%include "std_string.i"

namespace std {
    %template(IntVector) vector<int>;
    %template(DoubleVector) vector<double>;
}

/* ═══════════════════════════════════════════════════════════════════════════
 * Exception Handling
 * ═══════════════════════════════════════════════════════════════════════════
 */

%exception {
    try {
        $action
    } catch (const OpenMM::OpenMMException& e) {
        PyErr_SetString(PyExc_RuntimeError, e.what());
        SWIG_fail;
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

    void addNanotubeConductor(
        const std::vector<int>& virtualIndices,
        const std::vector<int>& realIndices,
        const std::string& electrodeType,
        double voltage,
        const Vec3& axis
    );

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
