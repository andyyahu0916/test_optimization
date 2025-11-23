#include "ReferenceElectrodeChargeKernel.h"
#include "ElectrodeChargeForce.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Vec3.h"
#include <cmath>

using namespace ElectrodeChargePlugin;
using namespace OpenMM;
using namespace std;

// Coulomb constant in OpenMM units (kJ/mol * nm / e^2)
// This is 1/(4*pi*epsilon_0) converted to OpenMM units
static const double COULOMB_CONSTANT = 138.935456;

void ReferenceCalcElectrodeChargeKernel::initialize(const System& system, const ElectrodeChargeForce& force) {
    // Cache force parameters
    electrodeAtomIndices = force.getElectrodeAtomIndices();
    targetPotentials = force.getTargetPotentials();
    electrolyteAtomIndices = force.getElectrolyteAtomIndices();
    fixedCharges = force.getFixedCharges();
    invCapMatrix = force.getInverseCapacitanceMatrix();

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
        throw OpenMMException("ElectrodeChargeForce: NonbondedForce not found");

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
}

double ReferenceCalcElectrodeChargeKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    const int N = electrodeAtomIndices.size();
    const int M = electrolyteAtomIndices.size();

    if (N == 0)
        return 0.0;  // No electrode atoms

    // ========== Step 1: Get positions (one read) ==========
    vector<Vec3> positions;
    context.getPositions(positions);

    // ========== Step 2: Compute E_f[i] = Σ_j (k * q_f[j] / r_ij) ==========
    vector<double> E_f(N, 0.0);

    for (int i = 0; i < N; i++) {
        int elecIdx = electrodeAtomIndices[i];
        const Vec3& pos_i = positions[elecIdx];

        for (int j = 0; j < M; j++) {
            int lytIdx = electrolyteAtomIndices[j];
            const Vec3& pos_j = positions[lytIdx];

            // Compute distance
            Vec3 delta = pos_i - pos_j;
            double r_squared = delta.dot(delta);

            if (r_squared > 1e-10) {  // Avoid division by zero
                double r_inv = 1.0 / sqrt(r_squared);
                E_f[i] += COULOMB_CONSTANT * fixedCharges[j] * r_inv;
            }
        }
    }

    // ========== Step 3: Compute b = V - E_f ==========
    vector<double> b(N);
    for (int i = 0; i < N; i++) {
        b[i] = targetPotentials[i] - E_f[i];
    }

    // ========== Step 4: Matrix multiply: q_e = C_inv * b ==========
    vector<double> q_e(N, 0.0);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            q_e[i] += invCapMatrix[i * N + j] * b[j];
        }
    }

    // ========== Step 5: Update charges (one write) ==========
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

    return 0.0;  // This force does not contribute to energy
}

void ReferenceCalcElectrodeChargeKernel::copyParametersToContext(ContextImpl& context, const ElectrodeChargeForce& force) {
    // Update cached parameters
    electrodeAtomIndices = force.getElectrodeAtomIndices();
    targetPotentials = force.getTargetPotentials();
    electrolyteAtomIndices = force.getElectrolyteAtomIndices();
    fixedCharges = force.getFixedCharges();
    invCapMatrix = force.getInverseCapacitanceMatrix();
}
