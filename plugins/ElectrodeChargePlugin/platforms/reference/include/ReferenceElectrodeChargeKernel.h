#ifndef REFERENCE_ELECTRODE_CHARGE_KERNEL_H_
#define REFERENCE_ELECTRODE_CHARGE_KERNEL_H_

#include "ElectrodeChargeKernels.h"
#include "openmm/System.h"
#include <vector>

namespace ElectrodeChargePlugin {

class ReferenceCalcElectrodeChargeKernel : public CalcElectrodeChargeKernel {
public:
    ReferenceCalcElectrodeChargeKernel(const std::string& name, const OpenMM::Platform& platform);
    void initialize(const OpenMM::System& system, const ElectrodeChargeForce& force) override;
    double execute(OpenMM::ContextImpl& context,
                   const std::vector<OpenMM::Vec3>& positions,
                   const std::vector<OpenMM::Vec3>& forces,
                   const std::vector<double>& allParticleCharges,
                   double sheetArea,
                   double cathodeZ,
                   double anodeZ,
                   std::vector<double>& cathodeCharges,
                   std::vector<double>& anodeCharges,
                   std::vector<std::vector<double>>& conductorCharges,
                   double& cathodeTarget,
                   double& anodeTarget) override;
    void copyParametersToContext(OpenMM::ContextImpl& context, const ElectrodeChargeForce& force) override;

private:
    ElectrodeChargeParameters parameters;
    int numParticles = 0;
    std::vector<bool> electrodeMask;
    std::vector<bool> conductorMask;  // 新增
};

} // namespace ElectrodeChargePlugin

#endif // REFERENCE_ELECTRODE_CHARGE_KERNEL_H_
