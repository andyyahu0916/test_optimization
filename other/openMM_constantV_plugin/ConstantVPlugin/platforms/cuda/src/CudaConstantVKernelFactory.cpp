#include "CudaConstantVKernelFactory.h"
#include "CudaConstantVKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/windowsExport.h"
#include "openmm/OpenMMException.h"

using namespace ConstantVPlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerPlatforms() {
}

extern "C" OPENMM_EXPORT void registerKernelFactories() {
    for (int i = 0; i < Platform::getNumPlatforms(); i++) {
        Platform& platform = Platform::getPlatform(i);
        if (dynamic_cast<CudaPlatform*>(&platform) != NULL) {
            CudaConstantVKernelFactory* factory = new CudaConstantVKernelFactory();
            platform.registerKernelFactory(CalcConstantVKernel::Name(), factory);
            platform.registerKernelFactory(IntegrateConstantVStepKernel::Name(), factory);
        }
    }
}

extern "C" OPENMM_EXPORT void registerConstantVCudaKernelFactories() {
    registerKernelFactories();
}

KernelImpl* CudaConstantVKernelFactory::createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const {
    CudaPlatform::PlatformData& data = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData());
    CudaContext& cu = *data.contexts[0];

    if (name == CalcConstantVKernel::Name())
        return new CudaCalcConstantVKernel(name, platform, cu);

    if (name == IntegrateConstantVStepKernel::Name())
        return new CudaIntegrateConstantVStepKernel(name, platform, cu);

    throw OpenMMException((std::string("Tried to create kernel with illegal kernel name '")+name+"'").c_str());
}
