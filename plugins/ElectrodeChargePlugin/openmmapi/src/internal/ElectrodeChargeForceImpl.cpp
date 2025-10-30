#include "internal/ElectrodeChargeForceImpl.h"
#include "ElectrodeChargeKernels.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath> // Linus: Added to fix build error.

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

ElectrodeChargeForceImpl::ElectrodeChargeForceImpl(const ElectrodeChargeForce& owner) : owner(owner) {
}

ElectrodeChargeForceImpl::~ElectrodeChargeForceImpl() = default;

void ElectrodeChargeForceImpl::initialize(ContextImpl& context) {
    // The kernel is the CudaCalcElectrodeChargeKernel, which does all the work.
    kernel = context.getPlatform().createKernel(CalcElectrodeChargeKernel::Name(), context);

    // Pass geometric constants to the force object so the kernel can access them.
    // This is done once at initialization.
    Vec3 boxA, boxB, boxC;
    context.getPeriodicBoxVectors(boxA, boxB, boxC);
    Vec3 crossAB = boxA.cross(boxB);
    double sheetArea = std::sqrt(crossAB.dot(crossAB));
    if (sheetArea == 0.0)
        throw OpenMMException("ElectrodeChargeForce detected zero sheet area; check periodic box vectors.");

    const auto& params = owner.getParameters();
    if (params.cathodeIndices.empty() || params.anodeIndices.empty())
        throw OpenMMException("ElectrodeChargeForce needs at least one cathode and one anode atom.");

    std::vector<Vec3> positions(context.getSystem().getNumParticles());
    context.getPositions(positions);
    double cathodeZ = positions[params.cathodeIndices[0]][2];
    double anodeZ = positions[params.anodeIndices[0]][2];

    // Modify the const owner via a const_cast to store these initial geometry values.
    // This is a common pattern for initialization logic within a ForceImpl.
    const_cast<ElectrodeChargeForce&>(owner).setSheetArea(sheetArea);
    const_cast<ElectrodeChargeForce&>(owner).setCathodeZ(cathodeZ);
    const_cast<ElectrodeChargeForce&>(owner).setAnodeZ(anodeZ);

    // Now, initialize the kernel with all parameters, including the geometry we just set.
    kernel.getAs<CalcElectrodeChargeKernel>().initialize(context.getSystem(), owner);
}

double ElectrodeChargeForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    // This force only modifies charges, so it should only be called when its group is active.
    if ((groups & (1 << owner.getForceGroup())) == 0)
        return 0.0;

    // The new CUDA kernel handles everything on the device in a single call.
    // It gets forces and positions from the context directly.
    // It updates the charges in the context's posq buffer directly.
    return kernel.getAs<CalcElectrodeChargeKernel>().execute(context, includeForces, includeEnergy);
}

std::map<std::string, double> ElectrodeChargeForceImpl::getDefaultParameters() {
    // No default parameters needed for this force.
    return std::map<std::string, double>();
}

std::vector<std::string> ElectrodeChargeForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcElectrodeChargeKernel::Name());
    return names;
}

void ElectrodeChargeForceImpl::updateParametersInContext(ContextImpl& context) {
    // This is called when parameters are changed on the Force object.
    // We just need to forward this to the kernel.
    kernel.getAs<CalcElectrodeChargeKernel>().copyParametersToContext(context, owner);
}
