#ifndef OPENMM_CONSTANTVFORCE_H_
#define OPENMM_CONSTANTVFORCE_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVForce: Native implementation of constant voltage boundary         *
 * conditions for electrochemical simulations.                                *
 * Copyright (c) 2025 - Present                                               *
 * -------------------------------------------------------------------------- */

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include <vector>
#include <string>
#include <cmath>

namespace OpenMM {

class OPENMM_EXPORT ConstantVForce : public Force {
public:
    // Nested Data Classes (must be defined before use)
    class CathodeAtomInfo {
    public:
        int particle;
        double area;
        CathodeAtomInfo() : particle(-1), area(0.0) {}
        CathodeAtomInfo(int p, double a) : particle(p), area(a) {}
    };

    class AnodeAtomInfo {
    public:
        int particle;
        double area;
        AnodeAtomInfo() : particle(-1), area(0.0) {}
        AnodeAtomInfo(int p, double a) : particle(p), area(a) {}
    };

    class ElectrolyteAtomInfo {
    public:
        int particle;
        double charge;
        ElectrolyteAtomInfo() : particle(-1), charge(0.0) {}
        ElectrolyteAtomInfo(int p, double c) : particle(p), charge(c) {}
    };

    class BuckyballConductorInfo {
    public:
        std::vector<int> virtualAtomIndices;
        std::vector<int> realAtomIndices;
        std::string electrodeType;
        double voltageVolts;
        double voltageKjMol;
        Vec3 center;
        double radius;
        double area_atom;
        std::vector<Vec3> normalVectors;
        int contactAtomIndex;
        double dr_center_contact;
        bool closeToElectrode;
        double closeThreshold;

        BuckyballConductorInfo() :
            electrodeType(""), voltageVolts(0.0), voltageKjMol(0.0),
            radius(0.0), area_atom(0.0), contactAtomIndex(-1),
            dr_center_contact(0.0), closeToElectrode(true), closeThreshold(1.5) {}

        BuckyballConductorInfo(const std::vector<int>& vAtoms,
                               const std::vector<int>& rAtoms,
                               const std::string& type, double voltage) :
            virtualAtomIndices(vAtoms), realAtomIndices(rAtoms),
            electrodeType(type), voltageVolts(voltage), voltageKjMol(voltage * 96.487),
            radius(0.0), area_atom(0.0), contactAtomIndex(-1),
            dr_center_contact(0.0), closeToElectrode(true), closeThreshold(1.5) {}
    };

    class NanotubeConductorInfo {
    public:
        std::vector<int> virtualAtomIndices;
        std::vector<int> realAtomIndices;
        std::string electrodeType;
        double voltageVolts;
        double voltageKjMol;
        Vec3 axis;
        Vec3 center;
        double radius;
        double length;
        double area_atom;
        std::vector<Vec3> normalVectors;
        int contactAtomIndex;
        double dr_center_contact;
        bool closeToElectrode;
        double closeThreshold;

        NanotubeConductorInfo() :
            electrodeType(""), voltageVolts(0.0), voltageKjMol(0.0),
            radius(0.0), length(0.0), area_atom(0.0), contactAtomIndex(-1),
            dr_center_contact(0.0), closeToElectrode(true), closeThreshold(1.5) {}

        NanotubeConductorInfo(const std::vector<int>& vAtoms,
                              const std::vector<int>& rAtoms,
                              const std::string& type, double voltage,
                              const std::vector<double>& axisVec) :
            virtualAtomIndices(vAtoms), realAtomIndices(rAtoms),
            electrodeType(type), voltageVolts(voltage), voltageKjMol(voltage * 96.487),
            radius(0.0), length(0.0), area_atom(0.0), contactAtomIndex(-1),
            dr_center_contact(0.0), closeToElectrode(true), closeThreshold(1.5) {
            if (axisVec.size() == 3) {
                axis = Vec3(axisVec[0], axisVec[1], axisVec[2]);
                double norm = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
                if (norm > 1e-10) axis *= (1.0 / norm);
                else axis = Vec3(1, 0, 0);
            } else {
                axis = Vec3(1, 0, 0);
            }
        }
    };

    // Constructor / Destructor
    ConstantVForce();
    ~ConstantVForce();

    // Flat Electrode Management
    int addCathodeAtom(int particle, double area);
    int addAnodeAtom(int particle, double area);
    int getNumCathodeAtoms() const { return cathodeAtoms.size(); }
    int getNumAnodeAtoms() const { return anodeAtoms.size(); }
    void getCathodeAtomParameters(int index, int& particle, double& area) const;
    void setCathodeAtomParameters(int index, int particle, double area);
    void getAnodeAtomParameters(int index, int& particle, double& area) const;
    void setAnodeAtomParameters(int index, int particle, double area);

    // Electrolyte Atom Management
    int addElectrolyteAtom(int particle, double charge);
    int getNumElectrolyteAtoms() const { return electrolyteAtoms.size(); }
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;
    void setElectrolyteAtomParameters(int index, int particle, double charge);

    // Buckyball Conductor Management
    int addBuckyballConductor(const std::vector<int>& virtualAtoms,
                              const std::vector<int>& realAtoms,
                              const std::string& electrodeType, double voltage);
    int getNumBuckyballConductors() const { return buckyballConductors.size(); }
    void getBuckyballConductorParameters(int index, std::vector<int>& virtualAtoms,
                                         std::vector<int>& realAtoms,
                                         std::string& electrodeType, double& voltage) const;
    void setBuckyballConductorParameters(int index, const std::vector<int>& virtualAtoms,
                                         const std::vector<int>& realAtoms,
                                         const std::string& electrodeType, double voltage);

    // Nanotube Conductor Management
    int addNanotubeConductor(const std::vector<int>& virtualAtoms,
                             const std::vector<int>& realAtoms,
                             const std::string& electrodeType,
                             double voltage, const std::vector<double>& axis);
    int getNumNanotubeConductors() const { return nanotubeConductors.size(); }
    void getNanotubeConductorParameters(int index, std::vector<int>& virtualAtoms,
                                        std::vector<int>& realAtoms, std::string& electrodeType,
                                        double& voltage, std::vector<double>& axis) const;
    void setNanotubeConductorParameters(int index, const std::vector<int>& virtualAtoms,
                                        const std::vector<int>& realAtoms,
                                        const std::string& electrodeType,
                                        double voltage, const std::vector<double>& axis);

    // System Geometry Parameters
    void setVoltage(double voltage);
    double getVoltage() const { return voltageVolts; }
    void setLgap(double gap);
    double getLgap() const { return Lgap; }
    void setLcell(double cell);
    double getLcell() const { return Lcell; }
    void setTotalArea(double area);
    double getTotalArea() const { return totalArea; }
    void setZCathode(double z);
    double getZCathode() const { return z_cathode; }
    void setZAnode(double z);
    double getZAnode() const { return z_anode; }

    // SCF Parameters
    void setNumIterations(int n);
    int getNumIterations() const { return nIterations; }

    bool usesPeriodicBoundaryConditions() const override { return true; }

    // Internal Accessors (for ConstantVForceImpl)
    const std::vector<CathodeAtomInfo>& getCathodeAtoms() const { return cathodeAtoms; }
    const std::vector<AnodeAtomInfo>& getAnodeAtoms() const { return anodeAtoms; }
    const std::vector<ElectrolyteAtomInfo>& getElectrolyteAtoms() const { return electrolyteAtoms; }

protected:
    ForceImpl* createImpl() const override;

private:
    friend class ConstantVForceImpl;

    std::vector<CathodeAtomInfo> cathodeAtoms;
    std::vector<AnodeAtomInfo> anodeAtoms;
    std::vector<ElectrolyteAtomInfo> electrolyteAtoms;
    std::vector<BuckyballConductorInfo> buckyballConductors;
    std::vector<NanotubeConductorInfo> nanotubeConductors;

    double voltageVolts;
    double voltageKjMol;
    double Lgap;
    double Lcell;
    double totalArea;
    double z_cathode;
    double z_anode;
    int nIterations;
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVFORCE_H_
