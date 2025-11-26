#ifndef OPENMM_CONSTANTVKERNELFACTORY_H_
#define OPENMM_CONSTANTVKERNELFACTORY_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVKernelFactory - Kernel registration and factory                  *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelFactory.h"

namespace OpenMM {

/**
 * This KernelFactory creates kernels for the ConstantV integration.
 * It registers implementations for both CUDA and Reference platforms.
 */
class ConstantVKernelFactory : public KernelFactory {
public:
    /**
     * Create a new ConstantVKernelFactory.
     */
    ConstantVKernelFactory();

    /**
     * Create a kernel with the specified name for the specified platform.
     *
     * @param name         the name of the kernel to create
     * @param platform     the platform to create the kernel for
     * @param context      the context the kernel will be used with
     * @return             the created kernel, or nullptr if no matching kernel found
     */
    KernelImpl* createKernelImpl(std::string name, const Platform& platform,
                                 ContextImpl& context) const override;
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVKERNELFACTORY_H_
