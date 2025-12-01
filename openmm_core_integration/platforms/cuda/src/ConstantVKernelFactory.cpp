/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVKernelFactory implementation                                      *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/ConstantVKernelFactory.h"
#include "openmm/ConstantVKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

// CUDA platform headers (this file is only compiled in CUDA library)
#include "CudaConstantVKernels.h"
#include "openmm/cuda/CudaContext.h"

using namespace OpenMM;

ConstantVKernelFactory::ConstantVKernelFactory() {
}

KernelImpl* ConstantVKernelFactory::createKernelImpl(std::string name,
                                                      const Platform& platform,
                                                      ContextImpl& context) const
{
    // CalcConstantV kernel (for ConstantVForce)
    if (name == CalcConstantVKernel::Name()) {
        if (platform.getName() == "CUDA") {
            return new CudaCalcConstantVKernel(name, platform,
                *static_cast<CudaContext*>(context.getPlatformData()));
        }
    }

    // IntegrateConstantVDrudeLangevinStep kernel
    if (name == "IntegrateConstantVDrudeLangevinStep") {
        if (platform.getName() == "CUDA") {
            return new CudaIntegrateConstantVDrudeLangevinStepKernel(name, platform,
                *static_cast<CudaContext*>(context.getPlatformData()));
        }
    }

    // IntegrateVerletStep kernel (for ConstantVIntegrator)
    // This is handled by the standard OpenMM Verlet kernel

    return nullptr;
}
