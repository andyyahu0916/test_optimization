#ifndef REFERENCE_CONSTANTV_KERNELS_H_
#define REFERENCE_CONSTANTV_KERNELS_H_

#include "ConstantVKernels.h"
#include "openmm/Platform.h"
#include "openmm/NonbondedForce.h"
#include <vector>

namespace ConstantVPlugin {

/**
 * Reference platform implementation (CPU, golden standard).
 */
class ReferenceCalcConstantVKernel : public CalcConstantVKernel {
public:
    ReferenceCalcConstantVKernel(std::string name, const OpenMM::Platform& platform) :
        CalcConstantVKernel(name, platform), nonbondedForce(nullptr) {
    }

    void initialize(const OpenMM::System& system, const ConstantVForce& force) override;

    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;

    void copyParametersToContext(OpenMM::ContextImpl& context, const ConstantVForce& force) override;

private:
    // Cached from force
    std::vector<int> electrodeAtomIndices;
    std::vector<double> targetPotentials;
    std::vector<int> electrolyteAtomIndices;
    std::vector<double> fixedCharges;
    std::vector<double> invCapMatrix;

    // Cached from NonbondedForce
    OpenMM::NonbondedForce* nonbondedForce;
    std::vector<double> particleSigmas;
    std::vector<double> particleEpsilons;
};

} // namespace ConstantVPlugin

#endif // REFERENCE_CONSTANTV_KERNELS_H_
