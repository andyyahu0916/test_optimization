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

    // Conductor data setters
    void setConductorData(const std::vector<int>& indices,
                          const std::vector<double>& normals,
                          const std::vector<double>& areas,
                          const std::vector<int>& contactIndices,
                          const std::vector<double>& contactNormals,
                          const std::vector<double>& geometries,
                          const std::vector<int>& types,
                          const std::vector<int>& atomCondIds,
                          const std::vector<int>& atomCountsPerConductor);
};

}

