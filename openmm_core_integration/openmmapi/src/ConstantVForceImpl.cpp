/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVForceImpl implementation                                          *
 * Copyright (c) 2025 - Present                                               *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/ConstantVForceImpl.h"
#include "openmm/internal/ConstantVGeometry.h"
#include "openmm/ConstantVKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Vec3.h"
#include <vector>
#include <string>
#include <cmath>

using namespace OpenMM;
using std::vector;
using std::string;

ConstantVForceImpl::ConstantVForceImpl(const ConstantVForce& owner) : owner(owner) {
}

ConstantVForceImpl::~ConstantVForceImpl() {
}

void ConstantVForceImpl::initialize(ContextImpl& context) {
    // Create the platform-specific kernel
    kernel = context.getPlatform().createKernel(CalcConstantVKernel::Name(), context);

    // Get current atomic positions for geometry initialization
    vector<Vec3> positions;
    context.getPositions(positions);

    // Get flat electrode data
    const vector<ConstantVForce::CathodeAtomInfo>& cathodeAtoms = owner.getCathodeAtoms();
    const vector<ConstantVForce::AnodeAtomInfo>& anodeAtoms = owner.getAnodeAtoms();
    const vector<ConstantVForce::ElectrolyteAtomInfo>& electrolyteAtoms = owner.getElectrolyteAtoms();

    // Extract flat electrode data into arrays
    vector<int> cathodeIndices, anodeIndices, electrolyteIndices;
    vector<double> cathodeAreas, anodeAreas, electrolyteCharges;

    cathodeIndices.reserve(cathodeAtoms.size());
    cathodeAreas.reserve(cathodeAtoms.size());
    for (const auto& atom : cathodeAtoms) {
        cathodeIndices.push_back(atom.particle);
        cathodeAreas.push_back(atom.area);
    }

    anodeIndices.reserve(anodeAtoms.size());
    anodeAreas.reserve(anodeAtoms.size());
    for (const auto& atom : anodeAtoms) {
        anodeIndices.push_back(atom.particle);
        anodeAreas.push_back(atom.area);
    }

    electrolyteIndices.reserve(electrolyteAtoms.size());
    electrolyteCharges.reserve(electrolyteAtoms.size());
    for (const auto& atom : electrolyteAtoms) {
        electrolyteIndices.push_back(atom.particle);
        electrolyteCharges.push_back(atom.charge);
    }

    // Initialize kernel with flat electrode data
    CalcConstantVKernel& calcKernel = dynamic_cast<CalcConstantVKernel&>(kernel.getImpl());
    calcKernel.initialize(
        context.getSystem(),
        cathodeIndices,
        cathodeAreas,
        anodeIndices,
        anodeAreas,
        electrolyteIndices,
        electrolyteCharges,
        owner.getVoltage(),
        owner.getLgap(),
        owner.getLcell(),
        owner.getTotalArea(),
        owner.getZCathode(),
        owner.getZAnode(),
        owner.getNumIterations()
    );

    // Initialize Buckyball conductors
    for (int i = 0; i < owner.getNumBuckyballConductors(); i++) {
        initializeBuckyballGeometry(context, i, positions);
    }

    // Initialize Nanotube conductors
    for (int i = 0; i < owner.getNumNanotubeConductors(); i++) {
        initializeNanotubeGeometry(context, i, positions);
    }
}

void ConstantVForceImpl::initializeBuckyballGeometry(ContextImpl& context,
                                                      int conductorIndex,
                                                      const vector<Vec3>& positions) {
    // Get Buckyball parameters from owner
    vector<int> virtualAtomIndices, realAtomIndices;
    string electrodeType;
    double voltage;
    owner.getBuckyballConductorParameters(conductorIndex, virtualAtomIndices,
                                          realAtomIndices, electrodeType, voltage);

    // Gather virtual atom positions
    vector<Vec3> virtualPositions;
    virtualPositions.reserve(virtualAtomIndices.size());
    for (int idx : virtualAtomIndices) {
        if (idx < 0 || idx >= (int)positions.size()) {
            throw OpenMMException("ConstantVForceImpl: Buckyball virtual atom index out of range");
        }
        virtualPositions.push_back(positions[idx]);
    }

    // Compute sphere geometry using ConstantVGeometry.h
    Vec3 center = computeSphereCenter(virtualPositions);
    double radius = computeSphereRadius(virtualPositions, center);
    vector<Vec3> normalVectors = computeSphereNormals(virtualPositions, center);
    double areaPerAtom = computeSphereAreaPerAtom(radius, virtualPositions.size());

    // Find contact electrode atom (nearest to center)
    vector<Vec3> electrodePositions;
    if (electrodeType == "cathode") {
        const vector<ConstantVForce::CathodeAtomInfo>& cathodeAtoms = owner.getCathodeAtoms();
        electrodePositions.reserve(cathodeAtoms.size());
        for (const auto& atom : cathodeAtoms) {
            if (atom.particle < 0 || atom.particle >= (int)positions.size()) {
                throw OpenMMException("ConstantVForceImpl: Cathode atom index out of range");
            }
            electrodePositions.push_back(positions[atom.particle]);
        }
    } else if (electrodeType == "anode") {
        const vector<ConstantVForce::AnodeAtomInfo>& anodeAtoms = owner.getAnodeAtoms();
        electrodePositions.reserve(anodeAtoms.size());
        for (const auto& atom : anodeAtoms) {
            if (atom.particle < 0 || atom.particle >= (int)positions.size()) {
                throw OpenMMException("ConstantVForceImpl: Anode atom index out of range");
            }
            electrodePositions.push_back(positions[atom.particle]);
        }
    } else {
        throw OpenMMException("ConstantVForceImpl: Invalid electrode type for Buckyball: " + electrodeType);
    }

    int contactAtomIndex;
    double contactDistance;
    findContactNeighbor(center, electrodePositions, contactAtomIndex, contactDistance);

    // Pass geometry to kernel
    CalcConstantVKernel& calcKernel = dynamic_cast<CalcConstantVKernel&>(kernel.getImpl());
    calcKernel.addBuckyballConductor(
        virtualAtomIndices,
        realAtomIndices,
        electrodeType,
        voltage,
        center,
        radius,
        normalVectors,
        areaPerAtom,
        contactAtomIndex,
        contactDistance
    );
}

void ConstantVForceImpl::initializeNanotubeGeometry(ContextImpl& context,
                                                     int conductorIndex,
                                                     const vector<Vec3>& positions) {
    // Get Nanotube parameters from owner
    vector<int> virtualAtomIndices, realAtomIndices;
    string electrodeType;
    double voltage;
    vector<double> axisVec;
    owner.getNanotubeConductorParameters(conductorIndex, virtualAtomIndices,
                                         realAtomIndices, electrodeType, voltage, axisVec);

    Vec3 axis(axisVec[0], axisVec[1], axisVec[2]);
    // Normalize axis
    double axisNorm = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (axisNorm < 1e-10) {
        axis = Vec3(1, 0, 0);  // Default to X axis
    } else {
        axis *= (1.0 / axisNorm);
    }

    // Gather virtual atom positions
    vector<Vec3> virtualPositions;
    virtualPositions.reserve(virtualAtomIndices.size());
    for (int idx : virtualAtomIndices) {
        if (idx < 0 || idx >= (int)positions.size()) {
            throw OpenMMException("ConstantVForceImpl: Nanotube virtual atom index out of range");
        }
        virtualPositions.push_back(positions[idx]);
    }

    // Compute nanotube geometry using ConstantVGeometry.h
    Vec3 center = computeNanotubeCenter(virtualPositions);
    double radius = computeNanotubeRadius(virtualPositions, center, axis);
    vector<Vec3> normalVectors = computeNanotubeNormals(virtualPositions, center, axis);

    // Get nanotube length from box vectors
    Vec3 boxA, boxB, boxC;
    context.getPeriodicBoxVectors(boxA, boxB, boxC);
    double length = computeNanotubeLength(boxA, boxB, boxC, axis);

    double areaPerAtom = computeCylinderAreaPerAtom(radius, length, virtualPositions.size());

    // Find contact electrode atom (nearest to center)
    vector<Vec3> electrodePositions;
    if (electrodeType == "cathode") {
        const vector<ConstantVForce::CathodeAtomInfo>& cathodeAtoms = owner.getCathodeAtoms();
        electrodePositions.reserve(cathodeAtoms.size());
        for (const auto& atom : cathodeAtoms) {
            if (atom.particle < 0 || atom.particle >= (int)positions.size()) {
                throw OpenMMException("ConstantVForceImpl: Cathode atom index out of range");
            }
            electrodePositions.push_back(positions[atom.particle]);
        }
    } else if (electrodeType == "anode") {
        const vector<ConstantVForce::AnodeAtomInfo>& anodeAtoms = owner.getAnodeAtoms();
        electrodePositions.reserve(anodeAtoms.size());
        for (const auto& atom : anodeAtoms) {
            if (atom.particle < 0 || atom.particle >= (int)positions.size()) {
                throw OpenMMException("ConstantVForceImpl: Anode atom index out of range");
            }
            electrodePositions.push_back(positions[atom.particle]);
        }
    } else {
        throw OpenMMException("ConstantVForceImpl: Invalid electrode type for Nanotube: " + electrodeType);
    }

    int contactAtomIndex;
    double contactDistance;
    findContactNeighbor(center, electrodePositions, contactAtomIndex, contactDistance);

    // Pass geometry to kernel
    CalcConstantVKernel& calcKernel = dynamic_cast<CalcConstantVKernel&>(kernel.getImpl());
    calcKernel.addNanotubeConductor(
        virtualAtomIndices,
        realAtomIndices,
        electrodeType,
        voltage,
        center,
        axis,
        radius,
        length,
        normalVectors,
        areaPerAtom,
        contactAtomIndex,
        contactDistance
    );
}

void ConstantVForceImpl::calcForce(ContextImpl& context, const vector<Vec3>& positions,
                                    vector<Vec3>& forces) {
    // This is called by the context integration loop
    // We delegate to the platform-specific kernel
    CalcConstantVKernel& calcKernel = dynamic_cast<CalcConstantVKernel&>(kernel.getImpl());
    calcKernel.execute(context, true, false, -1);
}

double ConstantVForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces,
                                                bool includeEnergy, int groups) {
    // Calculate forces and/or energy
    CalcConstantVKernel& calcKernel = dynamic_cast<CalcConstantVKernel&>(kernel.getImpl());
    return calcKernel.execute(context, includeForces, includeEnergy, groups);
}

std::map<std::string, double> ConstantVForceImpl::getDefaultParameters() {
    std::map<std::string, double> parameters;
    parameters["voltage"] = owner.getVoltage();
    parameters["Lgap"] = owner.getLgap();
    parameters["Lcell"] = owner.getLcell();
    parameters["totalArea"] = owner.getTotalArea();
    parameters["z_cathode"] = owner.getZCathode();
    parameters["z_anode"] = owner.getZAnode();
    parameters["nIterations"] = (double)owner.getNumIterations();
    return parameters;
}

std::vector<std::string> ConstantVForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcConstantVKernel::Name());
    return names;
}

void ConstantVForceImpl::updateParametersInContext(ContextImpl& context) {
    // Update kernel with new parameters
    CalcConstantVKernel& calcKernel = dynamic_cast<CalcConstantVKernel&>(kernel.getImpl());
    calcKernel.updateParameters(context, owner);

    // Re-initialize conductor geometry with current positions
    vector<Vec3> positions;
    context.getPositions(positions);

    for (int i = 0; i < owner.getNumBuckyballConductors(); i++) {
        initializeBuckyballGeometry(context, i, positions);
    }

    for (int i = 0; i < owner.getNumNanotubeConductors(); i++) {
        initializeNanotubeGeometry(context, i, positions);
    }
}

void ConstantVForceImpl::updateConductorGeometry(ContextImpl& context) {
    // Get current positions
    vector<Vec3> positions;
    context.getPositions(positions);

    // Re-initialize all conductors
    for (int i = 0; i < owner.getNumBuckyballConductors(); i++) {
        initializeBuckyballGeometry(context, i, positions);
    }

    for (int i = 0; i < owner.getNumNanotubeConductors(); i++) {
        initializeNanotubeGeometry(context, i, positions);
    }
}
