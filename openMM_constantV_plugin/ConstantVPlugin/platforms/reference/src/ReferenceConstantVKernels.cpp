#include "ReferenceConstantVKernels.h"
#include "ConstantVForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include <cmath>

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

// Coulomb constant in OpenMM units (kJ/mol * nm / e^2)
static const double COULOMB_CONSTANT = 138.935456;

static vector<RealVec>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *((vector<RealVec>*) data->positions);
}

void ReferenceCalcConstantVKernel::initialize(const System& system, const ConstantVForce& force) {
    // Cache electrode parameters
    int numElectrodes = force.getNumElectrodeAtoms();
    electrodeAtomIndices.resize(numElectrodes);
    targetPotentials.resize(numElectrodes);
    for (int i = 0; i < numElectrodes; i++) {
        int particle;
        double potential;
        force.getElectrodeAtomParameters(i, particle, potential);
        electrodeAtomIndices[i] = particle;
        targetPotentials[i] = potential;
    }

    // Cache electrolyte parameters
    int numElectrolytes = force.getNumElectrolyteAtoms();
    electrolyteAtomIndices.resize(numElectrolytes);
    fixedCharges.resize(numElectrolytes);
    for (int i = 0; i < numElectrolytes; i++) {
        int particle;
        double charge;
        force.getElectrolyteAtomParameters(i, particle, charge);
        electrolyteAtomIndices[i] = particle;
        fixedCharges[i] = charge;
    }

    // Find NonbondedForce
    nonbondedForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        const NonbondedForce* nbForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
        if (nbForce != nullptr) {
            nonbondedForce = const_cast<NonbondedForce*>(nbForce);
            break;
        }
    }

    if (nonbondedForce == nullptr)
        throw OpenMMException("ConstantVForce: NonbondedForce not found");

    // Cache sigma and epsilon
    int numParticles = system.getNumParticles();
    particleSigmas.resize(numParticles);
    particleEpsilons.resize(numParticles);

    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);
        particleSigmas[i] = sigma;
        particleEpsilons[i] = epsilon;
    }

    // Cache inverse capacitance matrix
    invCapMatrix = force.getInverseCapacitanceMatrix();
}

double ReferenceCalcConstantVKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    const int N = electrodeAtomIndices.size();
    const int M = electrolyteAtomIndices.size();

    if (N == 0)
        return 0.0;

    // Step 1: Get positions
    vector<RealVec>& pos = extractPositions(context);

    // Step 2: Compute E_f[i] = Σ_j (k * q_f[j] / r_ij)
    vector<double> E_f(N, 0.0);
    for (int i = 0; i < N; i++) {
        int elecIdx = electrodeAtomIndices[i];
        const RealVec& pos_i = pos[elecIdx];

        for (int j = 0; j < M; j++) {
            int lytIdx = electrolyteAtomIndices[j];
            const RealVec& pos_j = pos[lytIdx];

            RealVec delta = pos_i - pos_j;
            RealOpenMM r_squared = delta.dot(delta);

            if (r_squared > 1e-10) {
                RealOpenMM r_inv = 1.0 / sqrt(r_squared);
                E_f[i] += COULOMB_CONSTANT * fixedCharges[j] * r_inv;
            }
        }
    }

    // Step 3: Compute b = V - E_f
    vector<double> b(N);
    for (int i = 0; i < N; i++) {
        b[i] = targetPotentials[i] - E_f[i];
    }

    // Step 4: Matrix multiply q_e = C_inv * b
    vector<double> q_e(N, 0.0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            q_e[i] += invCapMatrix[i * N + j] * b[j];
        }
    }

    // Step 5: Update NonbondedForce charges
    for (int i = 0; i < N; i++) {
        int atomIdx = electrodeAtomIndices[i];
        nonbondedForce->setParticleParameters(
            atomIdx,
            q_e[i],
            particleSigmas[atomIdx],
            particleEpsilons[atomIdx]
        );
    }

    // Update context once
    nonbondedForce->updateParametersInContext(context.getOwner());

    return 0.0;  // No energy contribution
}

void ReferenceCalcConstantVKernel::copyParametersToContext(ContextImpl& context, const ConstantVForce& force) {
    // Update cached parameters
    int numElectrodes = force.getNumElectrodeAtoms();
    for (int i = 0; i < numElectrodes; i++) {
        int particle;
        double potential;
        force.getElectrodeAtomParameters(i, particle, potential);
        electrodeAtomIndices[i] = particle;
        targetPotentials[i] = potential;
    }

    invCapMatrix = force.getInverseCapacitanceMatrix();
}
