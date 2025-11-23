%module electrodecharge

// Linus: Import only what exists. Don't fail on missing plugin classes.
%{
#include "OpenMM.h"
#include "ElectrodeChargeForce.h"
%}

// Declare minimal OpenMM interface
namespace OpenMM {
    class Context;
    class Force {
    public:
        virtual ~Force();
    protected:
        Force();
    };
}

%include "swig/typemaps.i"
%include "std_vector.i"
namespace std {
  %template(IntVector) vector<int>;
  %template(DoubleVector) vector<double>;
}

%{
#include "ElectrodeChargeForce.h"
#include "OpenMM.h"
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
    ElectrodeChargeForce();
    void setCathode(const std::vector<int>& indices, double voltage);
    void setAnode(const std::vector<int>& indices, double voltage);
    void setNumIterations(int iterations);
    void setSmallThreshold(double value);
    void setCellGap(double gap);
    void setCellLength(double length);
    void setSheetArea(double area);
    void setCathodeZ(double z);
    void setAnodeZ(double z);
    bool usesPeriodicBoundaryConditions() const;

    // Inherit Force base class methods
    void setForceGroup(int group);
    int getForceGroup() const;

    // Conductor data setters
    // Good taste: geometry factor already encodes conductor type (Buckyball: dr², Nanotube: dr×L/2)
    void setConductorData(const std::vector<int>& indices,
                          const std::vector<double>& normals,
                          const std::vector<double>& areas,
                          const std::vector<int>& contactIndices,
                          const std::vector<double>& contactNormals,
                          const std::vector<double>& geometries,
                          const std::vector<int>& atomCondIds,
                          const std::vector<int>& atomCountsPerConductor);
};

}

