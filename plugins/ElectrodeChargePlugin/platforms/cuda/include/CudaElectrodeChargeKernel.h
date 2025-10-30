/**
 * Optimized CUDA Kernel Header for ElectrodeChargeForce (Linus Version)
 */

#ifndef CUDA_ELECTRODE_CHARGE_KERNEL_H_
#define CUDA_ELECTRODE_CHARGE_KERNEL_H_

#include "ElectrodeChargeKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"

namespace ElectrodeChargePlugin {

class CudaCalcElectrodeChargeKernel : public CalcElectrodeChargeKernel {
public:
    CudaCalcElectrodeChargeKernel(const std::string& name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu);
    ~CudaCalcElectrodeChargeKernel();

    void initialize(const OpenMM::System& system, const ElectrodeChargeForce& force) override;

    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;

    void copyParametersToContext(OpenMM::ContextImpl& context, const ElectrodeChargeForce& force) override;

private:
    void initializeDeviceMemory();
    void cleanup();

    bool hasInitialized = false;
    OpenMM::CudaContext* cu;
    int numParticles;
    ElectrodeChargeForce::Parameters parameters;

    // Device arrays for electrode and conductor indices/masks
    OpenMM::CudaArray* cathodeIndicesDevice = nullptr;
    OpenMM::CudaArray* anodeIndicesDevice = nullptr;
    OpenMM::CudaArray* electrodeMaskDevice = nullptr;
    OpenMM::CudaArray* conductorMaskDevice = nullptr;

    // Device arrays for conductor-specific data
    OpenMM::CudaArray* conductorIndicesDevice = nullptr;
    OpenMM::CudaArray* conductorNormalsDevice = nullptr;
    OpenMM::CudaArray* conductorAreasDevice = nullptr;
    OpenMM::CudaArray* conductorContactIndicesDevice = nullptr;
    OpenMM::CudaArray* conductorContactNormalsDevice = nullptr;
    OpenMM::CudaArray* conductorGeometriesDevice = nullptr;
    OpenMM::CudaArray* conductorAtomCondIdsDevice = nullptr;
    OpenMM::CudaArray* conductorAtomCountsDevice = nullptr;
    OpenMM::CudaArray* conductorDQDevice = nullptr;

    // Persistent buffer for reduction results
    OpenMM::CudaArray* chargeSumBuffers = nullptr;

    bool inInternalEvaluation = false;
};

} // namespace ElectrodeChargePlugin

#endif // CUDA_ELECTRODE_CHARGE_KERNEL_H_