#include "internal/ElectrodeChargeForceImpl.h"
#include "ElectrodeChargeKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/NonbondedForce.h"

using namespace ElectrodeChargePlugin;
using namespace OpenMM;
using namespace std;

ElectrodeChargeForceImpl::ElectrodeChargeForceImpl(const ElectrodeChargeForce& owner) :
    owner(owner), nonbondedForce(nullptr) {
}

ElectrodeChargeForceImpl::~ElectrodeChargeForceImpl() {
}

void ElectrodeChargeForceImpl::initialize(ContextImpl& context) {
    // Find NonbondedForce in the system
    const System& system = context.getSystem();
    nonbondedForce = nullptr;

    for (int i = 0; i < system.getNumForces(); i++) {
        const NonbondedForce* nbForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
        if (nbForce != nullptr) {
            nonbondedForce = const_cast<NonbondedForce*>(nbForce);
            break;
        }
    }

    if (nonbondedForce == nullptr)
        throw OpenMMException("ElectrodeChargeForce requires a NonbondedForce in the System");

    // Cache sigma and epsilon for all particles
    int numParticles = system.getNumParticles();
    particleSigmas.resize(numParticles);
    particleEpsilons.resize(numParticles);

    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);
        particleSigmas[i] = sigma;
        particleEpsilons[i] = epsilon;
    }

    // Create platform-specific kernel
    kernel = context.getPlatform().createKernel(CalcElectrodeChargeKernel::Name(), context);
    kernel.getAs<CalcElectrodeChargeKernel>().initialize(system, owner);
}

double ElectrodeChargeForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    // Execute the kernel to update charges
    // This force does not contribute to forces or energy
    return kernel.getAs<CalcElectrodeChargeKernel>().execute(context, false, false);
}

vector<string> ElectrodeChargeForceImpl::getKernelNames() {
    vector<string> names;
    names.push_back(CalcElectrodeChargeKernel::Name());
    return names;
}

void ElectrodeChargeForceImpl::updateParametersInContext(ContextImpl& context) {
    kernel.getAs<CalcElectrodeChargeKernel>().copyParametersToContext(context, owner);
}
