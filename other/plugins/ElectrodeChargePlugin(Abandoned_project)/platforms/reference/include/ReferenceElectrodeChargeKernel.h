#ifndef REFERENCE_ELECTRODE_CHARGE_KERNEL_H_
#define REFERENCE_ELECTRODE_CHARGE_KERNEL_H_

#include "ElectrodeChargeKernels.h"
#include "openmm/Platform.h"
#include "openmm/NonbondedForce.h"
#include <vector>

namespace ElectrodeChargePlugin {

/**
 * Reference platform implementation of CalcElectrodeChargeKernel.
 * This is the "golden standard" - simple, correct, slow.
 */
class ReferenceCalcElectrodeChargeKernel : public CalcElectrodeChargeKernel {
public:
    ReferenceCalcElectrodeChargeKernel(std::string name, const OpenMM::Platform& platform) :
        CalcElectrodeChargeKernel(name, platform), nonbondedForce(nullptr) {
    }

    void initialize(const OpenMM::System& system, const ElectrodeChargeForce& force) override;

    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;

    void copyParametersToContext(OpenMM::ContextImpl& context, const ElectrodeChargeForce& force) override;

private:
    // Cached data from force
    std::vector<int> electrodeAtomIndices;      // size N
    std::vector<double> targetPotentials;       // size N
    std::vector<int> electrolyteAtomIndices;    // size M
    std::vector<double> fixedCharges;           // size M
    std::vector<double> invCapMatrix;           // size N*N

    // Cached from NonbondedForce
    OpenMM::NonbondedForce* nonbondedForce;
    std::vector<double> particleSigmas;
    std::vector<double> particleEpsilons;
};

} // namespace ElectrodeChargePlugin

#endif // REFERENCE_ELECTRODE_CHARGE_KERNEL_H_
