#include "internal/ElectrodeChargeForceImpl.h"
#include "ElectrodeChargeKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>
#include <numeric>

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

ElectrodeChargeForceImpl::ElectrodeChargeForceImpl(const ElectrodeChargeForce& owner) : owner(owner) {
}

ElectrodeChargeForceImpl::~ElectrodeChargeForceImpl() = default;

void ElectrodeChargeForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcElectrodeChargeKernel::Name(), context);
    kernel.getAs<CalcElectrodeChargeKernel>().initialize(context.getSystem(), owner);

    const System& system = context.getSystem();
    nonbondedForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        NonbondedForce* candidate = dynamic_cast<NonbondedForce*>(&const_cast<Force&>(system.getForce(i)));
        if (candidate != nullptr) {
            nonbondedForce = candidate;
            break;
        }
    }
    if (nonbondedForce == nullptr)
        throw OpenMMException("ElectrodeChargeForce requires a NonbondedForce in the System.");

    cathodeSigmas.resize(owner.getCathode().atomIndices.size());
    cathodeEpsilons.resize(owner.getCathode().atomIndices.size());
    for (size_t i = 0; i < owner.getCathode().atomIndices.size(); i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(owner.getCathode().atomIndices[i], charge, sigma, epsilon);
        cathodeSigmas[i] = sigma;
        cathodeEpsilons[i] = epsilon;
    }

    anodeSigmas.resize(owner.getAnode().atomIndices.size());
    anodeEpsilons.resize(owner.getAnode().atomIndices.size());
    for (size_t i = 0; i < owner.getAnode().atomIndices.size(); i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(owner.getAnode().atomIndices[i], charge, sigma, epsilon);
        anodeSigmas[i] = sigma;
        anodeEpsilons[i] = epsilon;
    }
}

double ElectrodeChargeForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    if ((groups & (1 << owner.getForceGroup())) == 0)
        return 0.0;
    if (inInternalEvaluation)
        return 0.0;
    if (nonbondedForce == nullptr)
        throw OpenMMException("ElectrodeChargeForceImpl not initialized with NonbondedForce.");

    const System& system = context.getSystem();
    int numParticles = system.getNumParticles();
    if (numParticles == 0)
        return 0.0;

    if (owner.getCathode().atomIndices.empty() || owner.getAnode().atomIndices.empty())
        return 0.0;

    std::vector<Vec3> positions(numParticles);
    context.getPositions(positions);

    Vec3 boxA, boxB, boxC;
    context.getPeriodicBoxVectors(boxA, boxB, boxC);
    Vec3 crossAB = boxA.cross(boxB);
    double sheetArea = std::sqrt(crossAB.dot(crossAB));
    if (sheetArea == 0.0)
        throw OpenMMException("ElectrodeChargeForce detected zero sheet area; check periodic box vectors.");

    double cathodeZ = positions[owner.getCathode().atomIndices[0]][2];
    double anodeZ = positions[owner.getAnode().atomIndices[0]][2];

    std::vector<double> allCharges(numParticles);
    double charge, sigma, epsilon;
    for (int i = 0; i < numParticles; i++) {
        nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);
        allCharges[i] = charge;
    }

    std::vector<Vec3> forces;
    std::vector<double> cathodeCharges;
    std::vector<double> anodeCharges;
    double cathodeTarget = 0.0;
    double anodeTarget = 0.0;

    int iterations = owner.getNumIterations();
    if (iterations < 1)
        iterations = 1;

    for (int iter = 0; iter < iterations; iter++) {
        forces.clear();
        try {
            inInternalEvaluation = true;
            context.calcForcesAndEnergy(true, false);
            context.getForces(forces);
            inInternalEvaluation = false;
        } catch (...) {
            inInternalEvaluation = false;
            throw;
        }

        kernel.getAs<CalcElectrodeChargeKernel>().execute(context,
                                                          positions,
                                                          forces,
                                                          allCharges,
                                                          sheetArea,
                                                          cathodeZ,
                                                          anodeZ,
                                                          cathodeCharges,
                                                          anodeCharges,
                                                          cathodeTarget,
                                                          anodeTarget);

        double cathodeTotal = std::accumulate(cathodeCharges.begin(), cathodeCharges.end(), 0.0);
        if (std::fabs(cathodeTotal) > owner.getSmallThreshold()) {
            double scale = cathodeTarget / cathodeTotal;
            if (scale > 0.0) {
                for (double& value : cathodeCharges)
                    value *= scale;
            }
        }

        double anodeTotal = std::accumulate(anodeCharges.begin(), anodeCharges.end(), 0.0);
        if (std::fabs(anodeTotal) > owner.getSmallThreshold()) {
            double scale = anodeTarget / anodeTotal;
            if (scale > 0.0) {
                for (double& value : anodeCharges)
                    value *= scale;
            }
        }

        for (size_t i = 0; i < owner.getCathode().atomIndices.size(); i++) {
            int index = owner.getCathode().atomIndices[i];
            nonbondedForce->setParticleParameters(index, cathodeCharges[i], cathodeSigmas[i], cathodeEpsilons[i]);
            allCharges[index] = cathodeCharges[i];
        }
        for (size_t i = 0; i < owner.getAnode().atomIndices.size(); i++) {
            int index = owner.getAnode().atomIndices[i];
            nonbondedForce->setParticleParameters(index, anodeCharges[i], anodeSigmas[i], anodeEpsilons[i]);
            allCharges[index] = anodeCharges[i];
        }
        nonbondedForce->updateParametersInContext(context.getOwner());
    }

    (void) includeForces;
    (void) includeEnergy;
    return 0.0;
}

std::map<std::string, double> ElectrodeChargeForceImpl::getDefaultParameters() {
    return std::map<std::string, double>();
}

std::vector<std::string> ElectrodeChargeForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcElectrodeChargeKernel::Name());
    return names;
}

void ElectrodeChargeForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcElectrodeChargeKernel>().copyParametersToContext(context, owner);

    if (nonbondedForce == nullptr)
        return;

    cathodeSigmas.resize(owner.getCathode().atomIndices.size());
    cathodeEpsilons.resize(owner.getCathode().atomIndices.size());
    for (size_t i = 0; i < owner.getCathode().atomIndices.size(); i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(owner.getCathode().atomIndices[i], charge, sigma, epsilon);
        cathodeSigmas[i] = sigma;
        cathodeEpsilons[i] = epsilon;
    }

    anodeSigmas.resize(owner.getAnode().atomIndices.size());
    anodeEpsilons.resize(owner.getAnode().atomIndices.size());
    for (size_t i = 0; i < owner.getAnode().atomIndices.size(); i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(owner.getAnode().atomIndices[i], charge, sigma, epsilon);
        anodeSigmas[i] = sigma;
        anodeEpsilons[i] = epsilon;
    }
}
