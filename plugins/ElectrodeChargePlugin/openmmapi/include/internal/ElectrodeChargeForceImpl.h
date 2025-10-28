#ifndef ELECTRODE_CHARGE_FORCE_IMPL_H_
#define ELECTRODE_CHARGE_FORCE_IMPL_H_

#include "ElectrodeChargeForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include "openmm/NonbondedForce.h"
#include <map>
#include <vector>

namespace ElectrodeChargePlugin {

class ElectrodeChargeForceImpl : public OpenMM::ForceImpl {
public:
    explicit ElectrodeChargeForceImpl(const ElectrodeChargeForce& owner);
    ~ElectrodeChargeForceImpl() override;

    void initialize(OpenMM::ContextImpl& context) override;
    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups) override;
    std::map<std::string, double> getDefaultParameters() override;
    std::vector<std::string> getKernelNames() override;
    void updateParametersInContext(OpenMM::ContextImpl& context);
    const ElectrodeChargeForce& getOwner() const { return owner; }

private:
    const ElectrodeChargeForce& owner;
    OpenMM::Kernel kernel;
    OpenMM::NonbondedForce* nonbondedForce = nullptr;
    std::vector<double> cathodeSigmas;
    std::vector<double> cathodeEpsilons;
    std::vector<double> anodeSigmas;
    std::vector<double> anodeEpsilons;
    bool inInternalEvaluation = false;
};

} // namespace ElectrodeChargePlugin

#endif // ELECTRODE_CHARGE_FORCE_IMPL_H_
