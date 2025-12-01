/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVKernelFactory implementation                                      *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/ConstantVKernelFactory.h"
#include "openmm/ConstantVKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

// Include platform-specific headers
#ifdef OPENMM_BUILD_CUDA_LIB
#include "CudaConstantVKernels.h"
#include "openmm/cuda/CudaContext.h"
#endif

#ifdef OPENMM_BUILD_REFERENCE_LIB
#include "ReferenceConstantVKernels.h"
#endif

using namespace OpenMM;

ConstantVKernelFactory::ConstantVKernelFactory() {
}

KernelImpl* ConstantVKernelFactory::createKernelImpl(std::string name,
                                                      const Platform& platform,
                                                      ContextImpl& context) const
{
    // CalcConstantV kernel (for ConstantVForce)
    if (name == CalcConstantVKernel::Name()) {
#ifdef OPENMM_BUILD_CUDA_LIB
        if (platform.getName() == "CUDA") {
            return new CudaCalcConstantVKernel(name, platform,
                *static_cast<CudaContext*>(context.getPlatformData()));
        }
#endif

#ifdef OPENMM_BUILD_REFERENCE_LIB
        if (platform.getName() == "Reference") {
            return new ReferenceCalcConstantVKernel(name, platform);
        }
#endif
    }

    // IntegrateConstantVDrudeLangevinStep kernel
    if (name == "IntegrateConstantVDrudeLangevinStep") {
#ifdef OPENMM_BUILD_CUDA_LIB
        if (platform.getName() == "CUDA") {
            return new CudaIntegrateConstantVDrudeLangevinStepKernel(name, platform,
                *static_cast<CudaContext*>(context.getPlatformData()));
        }
#endif

#ifdef OPENMM_BUILD_REFERENCE_LIB
        if (platform.getName() == "Reference") {
            return new ReferenceIntegrateConstantVDrudeLangevinStepKernel(name, platform);
        }
#endif
    }

    // IntegrateVerletStep kernel (for ConstantVIntegrator)
    // This is handled by the standard OpenMM Verlet kernel

    return nullptr;
}
