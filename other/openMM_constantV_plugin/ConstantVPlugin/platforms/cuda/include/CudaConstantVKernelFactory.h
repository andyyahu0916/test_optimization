#ifndef OPENMM_CUDACONSTANTVKERNELFACTORY_H_
#define OPENMM_CUDACONSTANTVKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace ConstantVPlugin {

/**
 * Factory for creating CUDA implementations of ConstantV kernels.
 */
class CudaConstantVKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const;
};

} // namespace ConstantVPlugin

#endif /*OPENMM_CUDACONSTANTVKERNELFACTORY_H_*/
