#include "ReferenceElectrodeChargeKernelFactory.h"
#include "ReferenceElectrodeChargeKernel.h"

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

KernelImpl* ReferenceElectrodeChargeKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    return new ReferenceCalcElectrodeChargeKernel(name, platform);
}
