#ifndef ELECTRODE_CHARGE_KERNELS_H_
#define ELECTRODE_CHARGE_KERNELS_H_

#include "ElectrodeChargeForce.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include <string>
#include <vector>

namespace ElectrodeChargePlugin {

struct ElectrodeChargeParameters {
    std::vector<int> cathodeIndices;
    std::vector<int> anodeIndices;
    double cathodeVoltage = 0.0;
    double anodeVoltage = 0.0;
    int numIterations = 0;
    double smallThreshold = 0.0;
    double lGap = 0.0;
    double lCell = 0.0;
};

/**
 * Abstract base class for platform specific kernels.
 */
class CalcElectrodeChargeKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcElectrodeCharge";
    }

    CalcElectrodeChargeKernel(const std::string& name, const OpenMM::Platform& platform) : OpenMM::KernelImpl(name, platform) {
    }

    virtual void initialize(const OpenMM::System& system, const ElectrodeChargeForce& force) = 0;

    virtual double execute(OpenMM::ContextImpl& context,
                           const std::vector<OpenMM::Vec3>& positions,
                           const std::vector<OpenMM::Vec3>& forces,
                           const std::vector<double>& allParticleCharges,
                           double sheetArea,
                           double cathodeZ,
                           double anodeZ,
                           std::vector<double>& cathodeCharges,
                           std::vector<double>& anodeCharges,
                           double& cathodeTarget,
                           double& anodeTarget) = 0;

    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const ElectrodeChargeForce& force) = 0;
};

} // namespace ElectrodeChargePlugin

#endif // ELECTRODE_CHARGE_KERNELS_H_
