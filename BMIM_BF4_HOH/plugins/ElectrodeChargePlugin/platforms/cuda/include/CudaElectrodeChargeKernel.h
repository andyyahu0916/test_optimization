#ifndef CUDA_ELECTRODE_CHARGE_KERNEL_H_
#define CUDA_ELECTRODE_CHARGE_KERNEL_H_

#include "ElectrodeChargeKernels.h"

namespace OpenMM {
class CudaContext;
}

namespace ElectrodeChargePlugin {

class CudaCalcElectrodeChargeKernel : public CalcElectrodeChargeKernel {
public:
    CudaCalcElectrodeChargeKernel(const std::string& name, const OpenMM::Platform& platform);
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
                   double& cathodeTarget,
                   double& anodeTarget) override;
    void copyParametersToContext(OpenMM::ContextImpl& context, const ElectrodeChargeForce& force) override;

private:
    ElectrodeChargeParameters parameters;
    OpenMM::CudaContext* cudaContext = nullptr;
};

} // namespace ElectrodeChargePlugin

#endif // CUDA_ELECTRODE_CHARGE_KERNEL_H_
