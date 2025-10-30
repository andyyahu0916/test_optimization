#include "ReferenceElectrodeChargeKernel.h"
#include "ElectrodeChargeForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <numeric>
#include <iostream>

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

// Define constants matching the Python implementation
const double CONVERSION_NM_BOHR = 18.8973;
const double CONVERSION_KJMOL_NM_AU = CONVERSION_NM_BOHR / 2625.5;
const double CONVERSION_EV_KJMOL = 96.487;
const double PI = 3.14159265358979323846;

ReferenceCalcElectrodeChargeKernel::ReferenceCalcElectrodeChargeKernel(const std::string& name, const Platform& platform) : CalcElectrodeChargeKernel(name, platform) {
}

void ReferenceCalcElectrodeChargeKernel::initialize(const System& system, const ElectrodeChargeForce& force) {
    numParticles = system.getNumParticles();
    // Manually copy parameters to avoid type mismatch
    const ElectrodeChargeForce::Parameters& forceParams = force.getParameters();
    parameters.cathodeIndices = forceParams.cathodeIndices;
    parameters.anodeIndices = forceParams.anodeIndices;
    parameters.cathodeVoltage = forceParams.cathodeVoltage;
    parameters.anodeVoltage = forceParams.anodeVoltage;
    parameters.numIterations = forceParams.numIterations;
    parameters.smallThreshold = forceParams.smallThreshold;
    parameters.lGap = forceParams.lGap;
    parameters.lCell = forceParams.lCell;
    parameters.conductorIndices = forceParams.conductorIndices;
    parameters.conductorNormals = forceParams.conductorNormals;
    parameters.conductorAreas = forceParams.conductorAreas;
    parameters.conductorContactIndices = forceParams.conductorContactIndices;
    parameters.conductorContactNormals = forceParams.conductorContactNormals;
    parameters.conductorGeometries = forceParams.conductorGeometries;
    parameters.conductorAtomCondIds = forceParams.conductorAtomCondIds;
    parameters.conductorAtomCounts = forceParams.conductorAtomCounts;

    // Find the NonbondedForce and cache sigmas and epsilons.
    nonbondedForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        if (dynamic_cast<const NonbondedForce*>(&system.getForce(i)) != nullptr) {
            nonbondedForce = const_cast<NonbondedForce*>((const NonbondedForce*)&system.getForce(i));
            break;
        }
    }
    if (nonbondedForce == nullptr)
        throw OpenMMException("ElectrodeChargeForce requires a NonbondedForce to be present in the System.");

    sigmas.resize(numParticles);
    epsilons.resize(numParticles);
    for (int i = 0; i < numParticles; i++) {
        double charge;
        nonbondedForce->getParticleParameters(i, charge, sigmas[i], epsilons[i]);
    }
    
    initialized = true;
}

double ReferenceCalcElectrodeChargeKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    if (!initialized)
        throw OpenMMException("ReferenceCalcElectrodeChargeKernel has not been initialized.");

    // Get geometric parameters from the context/force object
    Vec3 boxA, boxB, boxC;
    context.getPeriodicBoxVectors(boxA, boxB, boxC);
    const Vec3 crossAB = boxA.cross(boxB);
    const double sheetArea = std::sqrt(crossAB.dot(crossAB));
    
    std::vector<Vec3> positions(numParticles);
    context.getPositions(positions);
    const double cathodeZ = positions[parameters.cathodeIndices[0]][2];
    const double anodeZ = positions[parameters.anodeIndices[0]][2];
    const double lCell = std::abs(cathodeZ - anodeZ);
    const double lGap = boxC[2] - lCell;

    const double cathodeVoltageKj = std::fabs(parameters.cathodeVoltage) * CONVERSION_EV_KJMOL;
    const double anodeVoltageKj = std::fabs(parameters.anodeVoltage) * CONVERSION_EV_KJMOL;
    const double cathodeAreaAtom = sheetArea / static_cast<double>(parameters.cathodeIndices.size());
    const double anodeAreaAtom = sheetArea / static_cast<double>(parameters.anodeIndices.size());

    // This vector will hold the charges as they are updated through the iterative process.
    std::vector<double> currentCharges(numParticles);
    for (int i = 0; i < numParticles; ++i) {
        double q, sig, eps;
        nonbondedForce->getParticleParameters(i, q, sig, eps);
        currentCharges[i] = q;
    }

    // --- Main SCF Iteration Loop ---
    for (int iter = 0; iter < parameters.numIterations; ++iter) {
        // Get forces based on the current charges
        context.calcForcesAndEnergy(true, false);
        std::vector<Vec3> forces(numParticles);
        context.getForces(forces);

        // Temporary vectors for newly computed charges for this iteration
        std::vector<double> nextCathodeCharges(parameters.cathodeIndices.size());
        std::vector<double> nextAnodeCharges(parameters.anodeIndices.size());
        std::vector<double> nextConductorCharges = currentCharges; // Start with current charges

        // 1. Update flat electrode charges
        for (size_t i = 0; i < parameters.cathodeIndices.size(); ++i) {
            int index = parameters.cathodeIndices[i];
            double oldCharge = currentCharges[index];
            double ezExternal = (std::fabs(oldCharge) > 0.9 * parameters.smallThreshold) ? forces[index][2] / oldCharge : 0.0;
            double newCharge = (2.0 / (4.0 * PI)) * cathodeAreaAtom * ((cathodeVoltageKj / lGap) + ezExternal) * CONVERSION_KJMOL_NM_AU;
            nextCathodeCharges[i] = (std::fabs(newCharge) < parameters.smallThreshold) ? parameters.smallThreshold : newCharge;
        }
        for (size_t i = 0; i < parameters.anodeIndices.size(); ++i) {
            int index = parameters.anodeIndices[i];
            double oldCharge = currentCharges[index];
            double ezExternal = (std::fabs(oldCharge) > 0.9 * parameters.smallThreshold) ? forces[index][2] / oldCharge : 0.0;
            double newCharge = -(2.0 / (4.0 * PI)) * anodeAreaAtom * ((anodeVoltageKj / lGap) + ezExternal) * CONVERSION_KJMOL_NM_AU;
            nextAnodeCharges[i] = (std::fabs(newCharge) < parameters.smallThreshold) ? -parameters.smallThreshold : newCharge;
        }

        // 2. Update conductor charges (if any)
        if (!parameters.conductorIndices.empty()) {
            // Step 2a: Image Charges
            for (size_t i = 0; i < parameters.conductorIndices.size(); ++i) {
                int atomIdx = parameters.conductorIndices[i];
                double oldCharge = currentCharges[atomIdx];
                if (std::fabs(oldCharge) > 0.9 * parameters.smallThreshold) {
                    const Vec3& force = forces[atomIdx];
                    const Vec3 normal(parameters.conductorNormals[3*i], parameters.conductorNormals[3*i+1], parameters.conductorNormals[3*i+2]);
                    const double enExternal = (force[0]*normal[0] + force[1]*normal[1] + force[2]*normal[2]) / oldCharge;
                    const double area = parameters.conductorAreas[i];
                    double imageCharge = (2.0 / (4.0 * PI)) * area * enExternal * CONVERSION_KJMOL_NM_AU;
                    nextConductorCharges[atomIdx] = (std::fabs(imageCharge) < parameters.smallThreshold) ? parameters.smallThreshold : imageCharge;
                } else {
                    nextConductorCharges[atomIdx] = parameters.smallThreshold;
                }
            }

            // Update context with image charges before recalculating forces
            for (size_t i=0; i<parameters.conductorIndices.size(); ++i) {
                int atomIdx = parameters.conductorIndices[i];
                nonbondedForce->setParticleParameters(atomIdx, nextConductorCharges[atomIdx], sigmas[atomIdx], epsilons[atomIdx]);
            }
            nonbondedForce->updateParametersInContext(context.getOwner());

            // *** Force Recalculation (to match original Python algorithm) ***
            context.calcForcesAndEnergy(true, false);
            context.getForces(forces);

            // Step 2b: Charge Transfer (using the NEW forces)
            std::vector<double> dQ_conductors(parameters.conductorAtomCounts.size(), 0.0);
            for (size_t i = 0; i < parameters.conductorContactIndices.size(); ++i) {
                int contactIdx = parameters.conductorContactIndices[i];
                double q_i = nextConductorCharges[contactIdx]; // Use the new image charge
                double enExternal = 0.0;
                if (std::fabs(q_i) > 0.9 * parameters.smallThreshold) {
                    const Vec3& force = forces[contactIdx];
                    const Vec3 normal(parameters.conductorContactNormals[3*i], parameters.conductorContactNormals[3*i+1], parameters.conductorContactNormals[3*i+2]);
                    enExternal = (force[0]*normal[0] + force[1]*normal[1] + force[2]*normal[2]) / q_i;
                }
                double dE_conductor = -(enExternal + (cathodeVoltageKj / lGap / 2.0)) * CONVERSION_KJMOL_NM_AU;
                dQ_conductors[i] = -1.0 * dE_conductor * parameters.conductorGeometries[i];
            }
            
            for (size_t i = 0; i < parameters.conductorIndices.size(); ++i) {
                int atomIdx = parameters.conductorIndices[i];
                int condId = parameters.conductorAtomCondIds[i];
                int count = parameters.conductorAtomCounts[condId];
                if (count > 0) {
                    nextConductorCharges[atomIdx] += dQ_conductors[condId] / static_cast<double>(count);
                }
            }
        }

        // 3. Calculate Analytic Targets
        double geomTerm = (1.0 / (4.0 * PI)) * sheetArea * ((cathodeVoltageKj / lGap) + (cathodeVoltageKj / lCell)) * CONVERSION_KJMOL_NM_AU;
        double cathodeTarget = geomTerm;
        double anodeTarget = -geomTerm;

        std::vector<bool> isElectrode(numParticles, false);
        for (int idx : parameters.cathodeIndices) isElectrode[idx] = true;
        for (int idx : parameters.anodeIndices) isElectrode[idx] = true;
        std::vector<bool> isConductor(numParticles, false);
        for (int idx : parameters.conductorIndices) isConductor[idx] = true;

        for (int i = 0; i < numParticles; ++i) {
            if (!isElectrode[i] && !isConductor[i]) { // Electrolyte
                double zDist = std::fabs(positions[i][2] - anodeZ);
                cathodeTarget += (zDist / lCell) * (-currentCharges[i]);
            }
        }
        // Conductor contribution to flat electrode targets
        for (size_t i = 0; i < parameters.conductorIndices.size(); ++i) {
            int atomIdx = parameters.conductorIndices[i];
            double zDist = std::fabs(positions[atomIdx][2] - anodeZ);
            cathodeTarget += (zDist / lCell) * (-nextConductorCharges[atomIdx]);
        }
        anodeTarget = -cathodeTarget; // For a neutral system, the targets must be equal and opposite.

        // 4. Grouped Scaling
        double anodeNumericSum = std::accumulate(nextAnodeCharges.begin(), nextAnodeCharges.end(), 0.0);
        if (std::fabs(anodeNumericSum) > parameters.smallThreshold) {
            double anodeScale = anodeTarget / anodeNumericSum;
            if (anodeScale > 0.0) {
                for (double& charge : nextAnodeCharges) {
                    charge *= anodeScale;
                }
            }
        }

        double cathodeNumericSum = std::accumulate(nextCathodeCharges.begin(), nextCathodeCharges.end(), 0.0);
        double conductorNumericSum = 0.0;
        for (int idx : parameters.conductorIndices) {
            conductorNumericSum += nextConductorCharges[idx];
        }
        double cathodeAndConductorsNumericSum = cathodeNumericSum + conductorNumericSum;
        
        double finalAnodeSum = std::accumulate(nextAnodeCharges.begin(), nextAnodeCharges.end(), 0.0);
        double cathodeSideTarget = -finalAnodeSum;

        if (std::fabs(cathodeAndConductorsNumericSum) > parameters.smallThreshold) {
            double combinedScale = cathodeSideTarget / cathodeAndConductorsNumericSum;
            if (combinedScale > 0.0) {
                for (double& charge : nextCathodeCharges) {
                    charge *= combinedScale;
                }
                for (int idx : parameters.conductorIndices) {
                    nextConductorCharges[idx] *= combinedScale;
                }
            }
        }

        // 5. Update charges in the main vector for the next iteration
        for (size_t i = 0; i < parameters.cathodeIndices.size(); ++i) currentCharges[parameters.cathodeIndices[i]] = nextCathodeCharges[i];
        for (size_t i = 0; i < parameters.anodeIndices.size(); ++i) currentCharges[parameters.anodeIndices[i]] = nextAnodeCharges[i];
        for (int idx : parameters.conductorIndices) currentCharges[idx] = nextConductorCharges[idx];

        // 6. Update the NonbondedForce in the context
        for (int i = 0; i < numParticles; ++i) {
            nonbondedForce->setParticleParameters(i, currentCharges[i], sigmas[i], epsilons[i]);
        }
        nonbondedForce->updateParametersInContext(context.getOwner());
    }

    return 0.0; // This force does not contribute to potential energy.
}

void ReferenceCalcElectrodeChargeKernel::copyParametersToContext(ContextImpl& context, const ElectrodeChargeForce& force) {
    // This is called when parameters are changed on the Force object.
    // Re-initialize to update all cached parameters.
    initialize(context.getSystem(), force);
}