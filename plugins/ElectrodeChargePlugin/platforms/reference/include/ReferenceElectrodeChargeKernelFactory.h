#ifndef REFERENCE_ELECTRODE_CHARGE_KERNEL_FACTORY_H_
#define REFERENCE_ELECTRODE_CHARGE_KERNEL_FACTORY_H_

#include "openmm/KernelFactory.h"

namespace ElectrodeChargePlugin {

class ReferenceElectrodeChargeKernelFactory : public OpenMM::KernelFactory {
public:
    OpenMM::KernelImpl* createKernelImpl(std::string name, const OpenMM::Platform& platform, OpenMM::ContextImpl& context) const override;
};

} // namespace ElectrodeChargePlugin

#endif // REFERENCE_ELECTRODE_CHARGE_KERNEL_FACTORY_H_
