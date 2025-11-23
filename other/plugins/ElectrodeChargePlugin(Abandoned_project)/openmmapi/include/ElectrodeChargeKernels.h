#ifndef OPENMM_ELECTRODECHARGEKERNELS_H_
#define OPENMM_ELECTRODECHARGEKERNELS_H_

#include "openmm/KernelImpl.h"
#include "openmm/System.h"
#include "openmm/Platform.h"
#include <string>

namespace ElectrodeChargePlugin {

class ElectrodeChargeForce;

/**
 * CalcElectrodeChargeKernel computes electrode charges using the formula:
 *   q_e = C_inv * (V - E_f)
 *
 * This kernel does NOT compute forces or energy. It only updates charges.
 */
class CalcElectrodeChargeKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcElectrodeCharge";
    }

    CalcElectrodeChargeKernel(std::string name, const OpenMM::Platform& platform) :
        OpenMM::KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     *
     * @param system the System this kernel will be applied to
     * @param force  the ElectrodeChargeForce this kernel will be used for
     */
    virtual void initialize(const OpenMM::System& system, const ElectrodeChargeForce& force) = 0;

    /**
     * Execute the kernel: compute and update electrode charges.
     *
     * @param context        the context in which to execute this kernel
     * @param includeForces  not used (always false for this force)
     * @param includeEnergy  not used (always false for this force)
     * @return 0.0 (this force does not contribute to energy)
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    /**
     * Copy changed parameters to the context.
     *
     * @param context    the context to copy parameters to
     * @param force      the ElectrodeChargeForce to copy the parameters from
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const ElectrodeChargeForce& force) = 0;
};

} // namespace ElectrodeChargePlugin

#endif // OPENMM_ELECTRODECHARGEKERNELS_H_
