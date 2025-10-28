#include "ElectrodeChargeForce.h"
#include "internal/ElectrodeChargeForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

ElectrodeChargeForce::ElectrodeChargeForce() {
    // CRITICAL: Put in different group than NonbondedForce (which is in group 0)
    // This prevents double-calculation when getState() calls all forces
    setForceGroup(1);
}

void ElectrodeChargeForce::setCathode(const std::vector<int>& indices, double voltage) {
    cathode.atomIndices = indices;
    cathode.voltage = voltage;
}

void ElectrodeChargeForce::setAnode(const std::vector<int>& indices, double voltage) {
    anode.atomIndices = indices;
    anode.voltage = voltage;
}

ForceImpl* ElectrodeChargeForce::createImpl() const {
    return new ElectrodeChargeForceImpl(*this);
}
