%module constantvplugin

%include "std_vector.i"
namespace std {
  %template(vectord) vector<double>;
  %template(vectori) vector<int>;
};

%{
#include "ConstantVForce.h"
#include "ConstantVIntegrator.h"
#include "ConstantVDrudeLangevinIntegrator.h"
#include "openmm/Force.h"
#include "openmm/Integrator.h"
#include "openmm/DrudeIntegrator.h"
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

    %nodefaultctor DrudeIntegrator;
    %nodefaultdtor DrudeIntegrator;
    class DrudeIntegrator : public Integrator {
    public:
        double getDrudeTemperature() const;
        void setDrudeTemperature(double temp);
        double getMaxDrudeDistance() const;
        void setMaxDrudeDistance(double distance);
        void setRandomNumberSeed(int seed);
        int getRandomNumberSeed() const;
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

    // Buckyball Conductor API
    int addBuckyballConductor(const std::vector<int>& virtualAtoms,
                               const std::vector<int>& realAtoms,
                               const std::string& electrodeType,
                               double voltage);
    int getNumBuckyballConductors() const;
    %apply std::vector<int>& OUTPUT {std::vector<int>& virtualAtoms};
    %apply std::vector<int>& OUTPUT {std::vector<int>& realAtoms};
    %apply std::string& OUTPUT {std::string& electrodeType};
    %apply double& OUTPUT {double& voltage};
    void getBuckyballConductorParameters(int index,
                                          std::vector<int>& virtualAtoms,
                                          std::vector<int>& realAtoms,
                                          std::string& electrodeType,
                                          double& voltage) const;
    %clear std::vector<int>& virtualAtoms;
    %clear std::vector<int>& realAtoms;
    %clear std::string& electrodeType;
    %clear double& voltage;

    // Nanotube Conductor API
    int addNanotubeConductor(const std::vector<int>& virtualAtoms,
                             const std::vector<int>& realAtoms,
                             const std::string& electrodeType,
                             double voltage,
                             const std::vector<double>& axis);
    int getNumNanotubeConductors() const;
    %apply std::vector<int>& OUTPUT {std::vector<int>& virtualAtoms};
    %apply std::vector<int>& OUTPUT {std::vector<int>& realAtoms};
    %apply std::string& OUTPUT {std::string& electrodeType};
    %apply double& OUTPUT {double& voltage};
    %apply std::vector<double>& OUTPUT {std::vector<double>& axis};
    void getNanotubeConductorParameters(int index,
                                        std::vector<int>& virtualAtoms,
                                        std::vector<int>& realAtoms,
                                        std::string& electrodeType,
                                        double& voltage,
                                        std::vector<double>& axis) const;
    %clear std::vector<int>& virtualAtoms;
    %clear std::vector<int>& realAtoms;
    %clear std::string& electrodeType;
    %clear double& voltage;
    %clear std::vector<double>& axis;

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

// ═══════════════════════════════════════════════════════════
// ConstantVDrudeLangevinIntegrator (Dual-temperature Langevin + Constant Voltage)
// ═══════════════════════════════════════════════════════════

class ConstantVDrudeLangevinIntegrator : public OpenMM::DrudeIntegrator {
public:
    ConstantVDrudeLangevinIntegrator(
        double temperature,
        double frictionCoeff,
        double drudeTemperature,
        double drudeFrictionCoeff,
        double stepSize
    );

    // Langevin parameters
    double getTemperature() const;
    void setTemperature(double temp);
    double getFriction() const;
    void setFriction(double coeff);
    double getDrudeFriction() const;
    void setDrudeFriction(double coeff);

    // Constant voltage parameters
    double getVoltage() const;
    void setVoltage(double v);
    int getNumSCFIterations() const;
    void setNumSCFIterations(int n);
    int getSCFFrequency() const;
    void setSCFFrequency(int freq);

    // Electrode atoms
    void addCathodeAtom(int particle, double area);
    void addAnodeAtom(int particle, double area);
    void addElectrolyteAtom(int particle, double charge);
    int getNumCathodeAtoms() const;
    int getNumAnodeAtoms() const;
    int getNumElectrolyteAtoms() const;

    // Geometry parameters
    void setLgap(double gap);
    void setLcell(double cell);
    void setTotalArea(double area);
    void setZCathode(double z);
    void setZAnode(double z);
    double getLgap() const;
    double getLcell() const;
    double getTotalArea() const;
    double getZCathode() const;
    double getZAnode() const;

    // Integrator interface
    void step(int steps);

    %extend {
        static ConstantVPlugin::ConstantVDrudeLangevinIntegrator& cast(OpenMM::Integrator& integrator) {
            return dynamic_cast<ConstantVPlugin::ConstantVDrudeLangevinIntegrator&>(integrator);
        }

        static bool isinstance(OpenMM::Integrator& integrator) {
            return (dynamic_cast<ConstantVPlugin::ConstantVDrudeLangevinIntegrator*>(&integrator) != NULL);
        }
    }
};

}
