#ifndef OPENMM_ELECTRODECHARGEFORCEIMP_H_
#define OPENMM_ELECTRODECHARGEFORCEIMP_H_

#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include "openmm/NonbondedForce.h"
#include "ElectrodeChargeForce.h"
#include "ElectrodeChargeKernels.h"

namespace ElectrodeChargePlugin {

/**
 * ElectrodeChargeForceImpl is the internal implementation of ElectrodeChargeForce.
 * It serves as a data container and creates the platform-specific kernel.
 */
class ElectrodeChargeForceImpl : public OpenMM::ForceImpl {
public:
    ElectrodeChargeForceImpl(const ElectrodeChargeForce& owner);
    ~ElectrodeChargeForceImpl();

    void initialize(OpenMM::ContextImpl& context) override;

    const ElectrodeChargeForce& getOwner() const {
        return owner;
    }

    double calcForcesAndEnergy(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy, int groups) override;

    std::map<std::string, double> getDefaultParameters() override {
        return std::map<std::string, double>();
    }

    std::vector<std::string> getKernelNames() override;

    void updateParametersInContext(OpenMM::ContextImpl& context);

    // Data accessible to kernels
    OpenMM::NonbondedForce* nonbondedForce;
    std::vector<double> particleSigmas;
    std::vector<double> particleEpsilons;

private:
    const ElectrodeChargeForce& owner;
    OpenMM::Kernel kernel;
};

} // namespace ElectrodeChargePlugin

#endif // OPENMM_ELECTRODECHARGEFORCEIMP_H_
