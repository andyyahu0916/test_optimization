#include "ReferenceConstantVKernelFactory.h"
#include "ReferenceConstantVKernels.h"
#include "openmm/reference/ReferencePlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/OpenMMException.h"

using namespace ConstantVPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<ReferencePlatform*>(&platform) != NULL) {
            ReferenceConstantVKernelFactory* factory = new ReferenceConstantVKernelFactory();
            platform.registerKernelFactory(CalcConstantVKernel::Name(), factory);
            platform.registerKernelFactory(IntegrateConstantVStepKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerConstantVReferenceKernelFactories() {
    registerKernelFactories();
}

KernelImpl* ReferenceConstantVKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    ReferencePlatform::PlatformData& data = *static_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    if (name == CalcConstantVKernel::Name())
        return new ReferenceCalcConstantVKernel(name, platform);
    if (name == IntegrateConstantVStepKernel::Name())
        return new ReferenceIntegrateConstantVStepKernel(name, platform);
    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
