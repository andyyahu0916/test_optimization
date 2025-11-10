#ifndef OPENMM_REFERENCECONSTANTVKERNELFACTORY_H_
#define OPENMM_REFERENCECONSTANTVKERNELFACTORY_H_

#include "openmm/KernelFactory.h"

namespace OpenMM {

class ReferenceConstantVKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const;
};

} // namespace OpenMM

#endif // OPENMM_REFERENCECONSTANTVKERNELFACTORY_H_
