%module electrodecharge

%import(module="openmm") "swig/OpenMMSwigHeaders.i"
%include "swig/typemaps.i"
%include "std_vector.i"

namespace std {
  %template(IntVector) vector<int>;
}

%{
#include "ElectrodeChargeForce.h"
#include "OpenMM.h"
#include "OpenMMException.h"
%}

%exception {
    try {
        $action
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_Exception, e.what());
        return NULL;
    }
}

namespace ElectrodeChargePlugin {

class ElectrodeChargeForce : public OpenMM::Force {
public:
    struct ElectrodeRegion {
        std::vector<int> atomIndices;
        double voltage;
    };

    ElectrodeChargeForce();
    void setCathode(const std::vector<int>& indices, double voltage);
    void setAnode(const std::vector<int>& indices, double voltage);
    const ElectrodeRegion& getCathode() const;
    const ElectrodeRegion& getAnode() const;
    void setNumIterations(int iterations);
    int getNumIterations() const;
    void setSmallThreshold(double value);
    double getSmallThreshold() const;
    void setCellGap(double gap);
    double getCellGap() const;
    void setCellLength(double length);
    double getCellLength() const;
    bool usesPeriodicBoundaryConditions() const;
};

}

