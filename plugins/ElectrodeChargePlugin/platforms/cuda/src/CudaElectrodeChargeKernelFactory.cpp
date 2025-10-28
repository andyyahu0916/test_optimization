#include "CudaElectrodeChargeKernelFactory.h"
#include "CudaElectrodeChargeKernel.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/cuda/CudaContext.h"

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

KernelImpl* CudaElectrodeChargeKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    // Get CUDA context immediately - this is when Context is being initialized
    CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
    return new CudaCalcElectrodeChargeKernel(name, platform, cu);
}
