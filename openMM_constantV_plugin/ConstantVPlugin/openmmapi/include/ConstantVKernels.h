#ifndef CONSTANTV_KERNELS_H_
#define CONSTANTV_KERNELS_H_

#include "ConstantVForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace ConstantVPlugin {

/**
 * This kernel updates electrode charges. It does NOT compute forces or energy.
 */
class CalcConstantVKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcConstantV";
    }

    CalcConstantVKernel(std::string name, const OpenMM::Platform& platform) :
        OpenMM::KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     */
    virtual void initialize(const OpenMM::System& system, const ConstantVForce& force) = 0;

    /**
     * Execute the kernel: update electrode charges.
     *
     * @return 0.0 (this force does not contribute to energy)
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    /**
     * Copy changed parameters to the context.
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const ConstantVForce& force) = 0;
};

} // namespace ConstantVPlugin

#endif // CONSTANTV_KERNELS_H_
