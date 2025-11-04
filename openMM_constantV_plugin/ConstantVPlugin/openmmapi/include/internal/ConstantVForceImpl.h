#ifndef OPENMM_CONSTANTVFORCEIMPL_H_
#define OPENMM_CONSTANTVFORCEIMPL_H_

#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include "ConstantVForce.h"
#include "ConstantVKernels.h"

namespace ConstantVPlugin {

class ConstantVForceImpl : public OpenMM::ForceImpl {
public:
    ConstantVForceImpl(const ConstantVForce& owner);
    ~ConstantVForceImpl();

    void initialize(OpenMM::ContextImpl& context) override;

    const ConstantVForce& getOwner() const {
        return owner;
    }

    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups) override;

    std::map<std::string, double> getDefaultParameters() override {
        return std::map<std::string, double>();
    }

    std::vector<std::string> getKernelNames() override;

private:
    const ConstantVForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace ConstantVPlugin

#endif // OPENMM_CONSTANTVFORCEIMPL_H_
