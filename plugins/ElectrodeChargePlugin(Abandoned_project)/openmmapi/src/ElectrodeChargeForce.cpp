#include "ElectrodeChargeForce.h"
#include "internal/ElectrodeChargeForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace ElectrodeChargePlugin;
using namespace OpenMM;
using namespace std;

ElectrodeChargeForce::ElectrodeChargeForce() : numElectrodeAtoms(0) {
}

void ElectrodeChargeForce::addElectrodeAtoms(const vector<int>& atomIndices, const vector<double>& potentials) {
    if (atomIndices.size() != potentials.size())
        throw OpenMMException("ElectrodeChargeForce: atomIndices and potentials must have the same size");

    electrodeAtomIndices = atomIndices;
    targetPotentials = potentials;
    numElectrodeAtoms = atomIndices.size();
}

void ElectrodeChargeForce::addElectrolyteAtoms(const vector<int>& atomIndices, const vector<double>& charges) {
    if (atomIndices.size() != charges.size())
        throw OpenMMException("ElectrodeChargeForce: atomIndices and charges must have the same size");

    electrolyteAtomIndices = atomIndices;
    fixedCharges = charges;
}

void ElectrodeChargeForce::setInverseCapacitanceMatrix(int numElectrodes, const vector<double>& flattenedMatrix) {
    if (numElectrodes <= 0)
        throw OpenMMException("ElectrodeChargeForce: numElectrodes must be positive");
    if (flattenedMatrix.size() != static_cast<size_t>(numElectrodes * numElectrodes))
        throw OpenMMException("ElectrodeChargeForce: flattenedMatrix size must be N*N");

    numElectrodeAtoms = numElectrodes;
    invCapMatrix = flattenedMatrix;
}

ForceImpl* ElectrodeChargeForce::createImpl() const {
    return new ElectrodeChargeForceImpl(*this);
}
