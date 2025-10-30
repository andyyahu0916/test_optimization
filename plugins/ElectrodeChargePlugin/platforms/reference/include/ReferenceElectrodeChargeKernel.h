#ifndef REFERENCE_ELECTRODE_CHARGE_KERNEL_H_
#define REFERENCE_ELECTRODE_CHARGE_KERNEL_H_

#include "ElectrodeChargeKernels.h"
#include "openmm/System.h"
#include "openmm/NonbondedForce.h"
#include <vector>

namespace ElectrodeChargePlugin {

class ReferenceCalcElectrodeChargeKernel : public CalcElectrodeChargeKernel {
public:
    ReferenceCalcElectrodeChargeKernel(const std::string& name, const OpenMM::Platform& platform);
    void initialize(const OpenMM::System& system, const ElectrodeChargeForce& force) override;
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void copyParametersToContext(OpenMM::ContextImpl& context, const ElectrodeChargeForce& force) override;

private:
    bool initialized = false;
    ElectrodeChargeParameters parameters;
    int numParticles = 0;
    OpenMM::NonbondedForce* nonbondedForce = nullptr;
    std::vector<double> sigmas;
    std::vector<double> epsilons;
};

} // namespace ElectrodeChargePlugin

#endif // REFERENCE_ELECTRODE_CHARGE_KERNEL_H_
