/**
 * CUDA Kernel Header (對應 CudaElectrodeChargeKernel_LINUS.cu)
 */

#ifndef CUDA_ELECTRODE_CHARGE_KERNEL_LINUS_H_
#define CUDA_ELECTRODE_CHARGE_KERNEL_LINUS_H_

#include "ElectrodeChargeKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

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
    void initializeDeviceMemory();
    
    ElectrodeChargeParameters parameters;
    OpenMM::CudaContext* cu;
    int numParticles;
    
    // Device arrays
    OpenMM::CudaArray* cathodeChargesDevice;
    OpenMM::CudaArray* anodeChargesDevice;
    OpenMM::CudaArray* cathodeIndicesDevice;
    OpenMM::CudaArray* anodeIndicesDevice;
    OpenMM::CudaArray* electrodeMaskDevice;
    OpenMM::CudaArray* cathodeTargetDevice;
    OpenMM::CudaArray* anodeTargetDevice;
    OpenMM::CudaArray* chargeSum;
};

} // namespace ElectrodeChargePlugin

#endif // CUDA_ELECTRODE_CHARGE_KERNEL_LINUS_H_
