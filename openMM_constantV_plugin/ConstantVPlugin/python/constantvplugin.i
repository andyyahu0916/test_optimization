%module constantvplugin

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "ConstantVForce.h"
#include "openmm/Force.h"
#include "openmm/Context.h"
%}

// Forward declare OpenMM::Force for inheritance (abstract class)
namespace OpenMM {
    %nodefaultctor Force;
    %nodefaultdtor Force;
    class Force {
    };

    class Context;
}

%pythoncode %{
import openmm as mm
import openmm.unit as unit
%}

%exception {
    try {
        $action
    } catch (std::exception &e) {
        PyErr_SetString(PyExc_Exception, const_cast<char*>(e.what()));
        return NULL;
    }
}

namespace ConstantVPlugin {

class ConstantVForce : public OpenMM::Force {
public:
    ConstantVForce();

    int getNumElectrodeAtoms() const;
    int getNumElectrolyteAtoms() const;

    int addElectrodeAtom(int particle, double potential);
    int addElectrolyteAtom(int particle, double charge);

    void setElectrodeAtomParameters(int index, int particle, double potential);
    void setInverseCapacitanceMatrix(const std::vector<double>& flattenedMatrix);

    const std::vector<double>& getInverseCapacitanceMatrix() const;

    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& potential};
    void getElectrodeAtomParameters(int index, int& particle, double& potential) const;
    %clear int& particle;
    %clear double& potential;

    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& charge};
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;
    %clear int& particle;
    %clear double& charge;

    %extend {
        static ConstantVPlugin::ConstantVForce& cast(OpenMM::Force& force) {
            return dynamic_cast<ConstantVPlugin::ConstantVForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<ConstantVPlugin::ConstantVForce*>(&force) != NULL);
        }
    }
};

}
