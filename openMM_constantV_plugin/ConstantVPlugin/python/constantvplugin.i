%module constantvplugin

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "ConstantVForce.h"
#include "ConstantVIntegrator.h"
#include "openmm/Force.h"
#include "openmm/Integrator.h"
#include "openmm/Context.h"
%}

// Forward declare OpenMM classes for inheritance
namespace OpenMM {
    %nodefaultctor Force;
    %nodefaultdtor Force;
    class Force {
    };

    %nodefaultctor Integrator;
    %nodefaultdtor Integrator;
    class Integrator {
    public:
        virtual void step(int steps) = 0;
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

    // 阴极原子
    int getNumCathodeAtoms() const;
    int addCathodeAtom(int particle, double area);
    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& area};
    void getCathodeAtomParameters(int index, int& particle, double& area) const;
    %clear int& particle;
    %clear double& area;

    // 阳极原子
    int getNumAnodeAtoms() const;
    int addAnodeAtom(int particle, double area);
    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& area};
    void getAnodeAtomParameters(int index, int& particle, double& area) const;
    %clear int& particle;
    %clear double& area;

    // 电解质原子
    int getNumElectrolyteAtoms() const;
    int addElectrolyteAtom(int particle, double charge);
    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& charge};
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;
    %clear int& particle;
    %clear double& charge;

    // 系统几何参数
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

    // SCF参数
    void setNumIterations(int n);
    int getNumIterations() const;

    %extend {
        static ConstantVPlugin::ConstantVForce& cast(OpenMM::Force& force) {
            return dynamic_cast<ConstantVPlugin::ConstantVForce&>(force);
        }

        static bool isinstance(OpenMM::Force& force) {
            return (dynamic_cast<ConstantVPlugin::ConstantVForce*>(&force) != NULL);
        }
    }
};

// ═══════════════════════════════════════════════════════════
// ConstantVIntegrator（恒电压Verlet积分器）
// ═══════════════════════════════════════════════════════════

class ConstantVIntegrator : public OpenMM::Integrator {
public:
    ConstantVIntegrator(double stepSize);

    // 物理参数
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

    // SCF参数
    void setNumSCFIterations(int n);
    int getNumSCFIterations() const;

    void setSCFFrequency(int freq);
    int getSCFFrequency() const;

    // 电极原子设置
    int addCathodeAtom(int particle, double area);
    int getNumCathodeAtoms() const;
    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& area};
    void getCathodeAtomParameters(int index, int& particle, double& area) const;
    %clear int& particle;
    %clear double& area;

    int addAnodeAtom(int particle, double area);
    int getNumAnodeAtoms() const;
    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& area};
    void getAnodeAtomParameters(int index, int& particle, double& area) const;
    %clear int& particle;
    %clear double& area;

    int addElectrolyteAtom(int particle, double charge);
    int getNumElectrolyteAtoms() const;
    %apply int& OUTPUT {int& particle};
    %apply double& OUTPUT {double& charge};
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;
    %clear int& particle;
    %clear double& charge;

    // Integrator接口
    void step(int steps);

    %extend {
        static ConstantVPlugin::ConstantVIntegrator& cast(OpenMM::Integrator& integrator) {
            return dynamic_cast<ConstantVPlugin::ConstantVIntegrator&>(integrator);
        }

        static bool isinstance(OpenMM::Integrator& integrator) {
            return (dynamic_cast<ConstantVPlugin::ConstantVIntegrator*>(&integrator) != NULL);
        }
    }
};

}
