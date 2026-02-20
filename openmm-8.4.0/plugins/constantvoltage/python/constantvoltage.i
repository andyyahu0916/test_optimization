/* -------------------------------------------------------------------------- *
 * ConstantVoltage Plugin SWIG Interface File
 * -------------------------------------------------------------------------- *
 * Exposes ConstantVoltageForce and ConstantVDrudeLangevinIntegrator to Python.
 * 
 * Usage in Python:
 *   from openmm.constantvoltage import ConstantVoltageForce, ConstantVDrudeLangevinIntegrator
 * -------------------------------------------------------------------------- */

%module(directors="1") constantvoltage

%include "std_string.i"
%include "std_vector.i"

namespace std {
    %template(vectord) vector<double>;
    %template(vectori) vector<int>;
}

%include "typemaps.i"

%{
#define SWIG_FILE_WITH_INIT

#include <exception>
#include "OpenMM.h"
#include "openmm/ConstantVoltageForce.h"
#include "openmm/ConstantVDrudeLangevinIntegrator.h"

using namespace OpenMM;
%}

// Add type casting functions to help with inheritance
%extend OpenMM::ConstantVoltageForce {
    // Cast to Force* for System.addForce()
    OpenMM::Force* asForce() {
        return (OpenMM::Force*) $self;
    }
}

%extend OpenMM::ConstantVDrudeLangevinIntegrator {
    // Cast to Integrator* if needed
    OpenMM::Integrator* asIntegrator() {
        return (OpenMM::Integrator*) $self;
    }
}

%feature("autodoc", "0");
%nodefaultctor;

/* Include exception handling */
%exception {
    try {
        $action
    }
    catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 * ConstantVoltageForce
 * Matches: plugins/constantvoltage/openmmapi/include/openmm/ConstantVoltageForce.h
 * ═══════════════════════════════════════════════════════════════════════════ */

namespace OpenMM {

class ConstantVoltageForce : public Force {
public:
    ConstantVoltageForce();
    
    // --- Cathode atoms ---
    int addCathodeAtom(int particle, double area);
    int getNumCathodeAtoms() const;
    
    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& area};
    void getCathodeAtomParameters(int index, int& particle, double& area) const;
    %clear int& particle;
    %clear double& area;
    
    void setCathodeAtomParameters(int index, int particle, double area);
    
    // --- Anode atoms ---
    int addAnodeAtom(int particle, double area);
    int getNumAnodeAtoms() const;
    
    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& area};
    void getAnodeAtomParameters(int index, int& particle, double& area) const;
    %clear int& particle;
    %clear double& area;
    
    void setAnodeAtomParameters(int index, int particle, double area);
    
    // --- Electrolyte atoms ---
    int addElectrolyteAtom(int particle);
    int getNumElectrolyteAtoms() const;
    int getElectrolyteAtomParticle(int index) const;
    
    // --- System parameters ---
    void setVoltage(double voltage);
    double getVoltage() const;
    
    void setLgap(double lgap);
    double getLgap() const;
    
    void setLcell(double lcell);
    double getLcell() const;
    
    void setTotalArea(double area);
    double getTotalArea() const;
    
    void setElectrodeZPositions(double zCathode, double zAnode);
    double getZCathode() const;
    double getZAnode() const;
    
    // --- SCF parameters ---
    void setNumSCFIterations(int n);
    int getNumSCFIterations() const;
    
    void setSCFFrequency(int freq);
    int getSCFFrequency() const;
    
    void setSmallThreshold(double threshold);
    double getSmallThreshold() const;
    
    bool usesPeriodicBoundaryConditions() const;
};

/* ═══════════════════════════════════════════════════════════════════════════
 * ConstantVDrudeLangevinIntegrator
 * Matches: plugins/constantvoltage/openmmapi/include/openmm/ConstantVDrudeLangevinIntegrator.h
 * ═══════════════════════════════════════════════════════════════════════════ */

class ConstantVDrudeLangevinIntegrator : public Integrator {
public:
    ConstantVDrudeLangevinIntegrator(double temperature, double frictionCoeff, 
                                      double drudeTemperature, double drudeFriction,
                                      double stepSize);
    
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
    
    int getRandomNumberSeed() const;
    void setRandomNumberSeed(int seed);
    
    void step(int steps);
};

} // namespace OpenMM

%pythoncode %{
    __all__ = ['ConstantVoltageForce', 'ConstantVDrudeLangevinIntegrator']
%}
