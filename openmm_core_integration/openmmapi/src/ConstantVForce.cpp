/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVForce implementation                                              *
 * Copyright (c) 2025 - Present                                               *
 * -------------------------------------------------------------------------- */

#include "openmm/ConstantVForce.h"
#include "openmm/internal/ConstantVForceImpl.h"
#include "openmm/OpenMMException.h"
#include <cmath>

using namespace OpenMM;

ConstantVForce::ConstantVForce() :
    voltageVolts(0.0),
    voltageKjMol(0.0),
    Lgap(0.0),
    Lcell(0.0),
    totalArea(0.0),
    z_cathode(0.0),
    z_anode(0.0),
    nIterations(4)  // Professor's default
{
}

ConstantVForce::~ConstantVForce() {
}

// ═══════════════════════════════════════════════════════════════════════════
// Cathode Atom Management
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVForce::addCathodeAtom(int particle, double area) {
    cathodeAtoms.push_back(CathodeAtomInfo(particle, area));
    return cathodeAtoms.size() - 1;
}

void ConstantVForce::getCathodeAtomParameters(int index, int& particle, double& area) const {
    if (index < 0 || index >= (int)cathodeAtoms.size())
        throw OpenMMException("ConstantVForce: cathode atom index out of range");
    particle = cathodeAtoms[index].particle;
    area = cathodeAtoms[index].area;
}

void ConstantVForce::setCathodeAtomParameters(int index, int particle, double area) {
    if (index < 0 || index >= (int)cathodeAtoms.size())
        throw OpenMMException("ConstantVForce: cathode atom index out of range");
    cathodeAtoms[index].particle = particle;
    cathodeAtoms[index].area = area;
}

// ═══════════════════════════════════════════════════════════════════════════
// Anode Atom Management
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVForce::addAnodeAtom(int particle, double area) {
    anodeAtoms.push_back(AnodeAtomInfo(particle, area));
    return anodeAtoms.size() - 1;
}

void ConstantVForce::getAnodeAtomParameters(int index, int& particle, double& area) const {
    if (index < 0 || index >= (int)anodeAtoms.size())
        throw OpenMMException("ConstantVForce: anode atom index out of range");
    particle = anodeAtoms[index].particle;
    area = anodeAtoms[index].area;
}

void ConstantVForce::setAnodeAtomParameters(int index, int particle, double area) {
    if (index < 0 || index >= (int)anodeAtoms.size())
        throw OpenMMException("ConstantVForce: anode atom index out of range");
    anodeAtoms[index].particle = particle;
    anodeAtoms[index].area = area;
}

// ═══════════════════════════════════════════════════════════════════════════
// Electrolyte Atom Management
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVForce::addElectrolyteAtom(int particle, double charge) {
    electrolyteAtoms.push_back(ElectrolyteAtomInfo(particle, charge));
    return electrolyteAtoms.size() - 1;
}

void ConstantVForce::getElectrolyteAtomParameters(int index, int& particle, double& charge) const {
    if (index < 0 || index >= (int)electrolyteAtoms.size())
        throw OpenMMException("ConstantVForce: electrolyte atom index out of range");
    particle = electrolyteAtoms[index].particle;
    charge = electrolyteAtoms[index].charge;
}

void ConstantVForce::setElectrolyteAtomParameters(int index, int particle, double charge) {
    if (index < 0 || index >= (int)electrolyteAtoms.size())
        throw OpenMMException("ConstantVForce: electrolyte atom index out of range");
    electrolyteAtoms[index].particle = particle;
    electrolyteAtoms[index].charge = charge;
}

// ═══════════════════════════════════════════════════════════════════════════
// Buckyball Conductor Management
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVForce::addBuckyballConductor(const std::vector<int>& virtualAtoms,
                                          const std::vector<int>& realAtoms,
                                          const std::string& electrodeType,
                                          double voltage) {
    buckyballConductors.push_back(BuckyballConductorInfo(virtualAtoms, realAtoms, electrodeType, voltage));
    return buckyballConductors.size() - 1;
}

void ConstantVForce::getBuckyballConductorParameters(int index,
                                                     std::vector<int>& virtualAtoms,
                                                     std::vector<int>& realAtoms,
                                                     std::string& electrodeType,
                                                     double& voltage) const {
    if (index < 0 || index >= (int)buckyballConductors.size())
        throw OpenMMException("ConstantVForce: Buckyball conductor index out of range");

    const BuckyballConductorInfo& bucky = buckyballConductors[index];
    virtualAtoms = bucky.virtualAtomIndices;
    realAtoms = bucky.realAtomIndices;
    electrodeType = bucky.electrodeType;
    voltage = bucky.voltageVolts;
}

void ConstantVForce::setBuckyballConductorParameters(int index,
                                                     const std::vector<int>& virtualAtoms,
                                                     const std::vector<int>& realAtoms,
                                                     const std::string& electrodeType,
                                                     double voltage) {
    if (index < 0 || index >= (int)buckyballConductors.size())
        throw OpenMMException("ConstantVForce: Buckyball conductor index out of range");

    BuckyballConductorInfo& bucky = buckyballConductors[index];
    bucky.virtualAtomIndices = virtualAtoms;
    bucky.realAtomIndices = realAtoms;
    bucky.electrodeType = electrodeType;
    bucky.voltageVolts = voltage;
    bucky.voltageKjMol = voltage * 96.487;  // conversion_eV_to_kJmol
}

// ═══════════════════════════════════════════════════════════════════════════
// Nanotube Conductor Management
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVForce::addNanotubeConductor(const std::vector<int>& virtualAtoms,
                                         const std::vector<int>& realAtoms,
                                         const std::string& electrodeType,
                                         double voltage,
                                         const std::vector<double>& axis) {
    nanotubeConductors.push_back(NanotubeConductorInfo(virtualAtoms, realAtoms, electrodeType, voltage, axis));
    return nanotubeConductors.size() - 1;
}

void ConstantVForce::getNanotubeConductorParameters(int index,
                                                    std::vector<int>& virtualAtoms,
                                                    std::vector<int>& realAtoms,
                                                    std::string& electrodeType,
                                                    double& voltage,
                                                    std::vector<double>& axis) const {
    if (index < 0 || index >= (int)nanotubeConductors.size())
        throw OpenMMException("ConstantVForce: Nanotube conductor index out of range");

    const NanotubeConductorInfo& nano = nanotubeConductors[index];
    virtualAtoms = nano.virtualAtomIndices;
    realAtoms = nano.realAtomIndices;
    electrodeType = nano.electrodeType;
    voltage = nano.voltageVolts;
    axis.resize(3);
    axis[0] = nano.axis[0];
    axis[1] = nano.axis[1];
    axis[2] = nano.axis[2];
}

void ConstantVForce::setNanotubeConductorParameters(int index,
                                                    const std::vector<int>& virtualAtoms,
                                                    const std::vector<int>& realAtoms,
                                                    const std::string& electrodeType,
                                                    double voltage,
                                                    const std::vector<double>& axis) {
    if (index < 0 || index >= (int)nanotubeConductors.size())
        throw OpenMMException("ConstantVForce: Nanotube conductor index out of range");

    NanotubeConductorInfo& nano = nanotubeConductors[index];
    nano.virtualAtomIndices = virtualAtoms;
    nano.realAtomIndices = realAtoms;
    nano.electrodeType = electrodeType;
    nano.voltageVolts = voltage;
    nano.voltageKjMol = voltage * 96.487;

    if (axis.size() == 3) {
        nano.axis = Vec3(axis[0], axis[1], axis[2]);
        // Normalize
        double norm = std::sqrt(nano.axis[0]*nano.axis[0] + nano.axis[1]*nano.axis[1] + nano.axis[2]*nano.axis[2]);
        if (norm > 1e-10) {
            nano.axis *= (1.0 / norm);
        } else {
            nano.axis = Vec3(1, 0, 0);
        }
    } else {
        nano.axis = Vec3(1, 0, 0);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// System Geometry Parameters
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVForce::setVoltage(double voltage) {
    voltageVolts = voltage;
    voltageKjMol = voltage * 96.487;  // conversion_eV_to_kJmol
}

void ConstantVForce::setLgap(double gap) {
    if (gap <= 0)
        throw OpenMMException("ConstantVForce: Lgap must be positive");
    Lgap = gap;
}

void ConstantVForce::setLcell(double cell) {
    if (cell <= 0)
        throw OpenMMException("ConstantVForce: Lcell must be positive");
    Lcell = cell;
}

void ConstantVForce::setTotalArea(double area) {
    if (area <= 0)
        throw OpenMMException("ConstantVForce: totalArea must be positive");
    totalArea = area;
}

void ConstantVForce::setZCathode(double z) {
    z_cathode = z;
}

void ConstantVForce::setZAnode(double z) {
    z_anode = z;
}

// ═══════════════════════════════════════════════════════════════════════════
// SCF Parameters
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVForce::setNumIterations(int n) {
    if (n < 1)
        throw OpenMMException("ConstantVForce: number of iterations must be at least 1");
    nIterations = n;
}

// ═══════════════════════════════════════════════════════════════════════════
// ForceImpl Creation
// ═══════════════════════════════════════════════════════════════════════════

ForceImpl* ConstantVForce::createImpl() const {
    return new ConstantVForceImpl(*this);
}
