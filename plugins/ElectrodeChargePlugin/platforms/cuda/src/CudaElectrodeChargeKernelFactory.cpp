#include "CudaElectrodeChargeKernelFactory.h"
#include "CudaElectrodeChargeKernel.h"

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

KernelImpl* CudaElectrodeChargeKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    return new CudaCalcElectrodeChargeKernel(name, platform);
}
