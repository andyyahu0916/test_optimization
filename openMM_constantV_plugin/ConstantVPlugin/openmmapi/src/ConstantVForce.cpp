#include "ConstantVForce.h"
#include "internal/ConstantVForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

ConstantVForce::ConstantVForce() {
}

int ConstantVForce::addElectrodeAtom(int particle, double potential) {
    electrodeAtoms.push_back(ElectrodeAtomInfo(particle, potential));
    electrodeAtomIndices.push_back(particle);
    targetPotentials.push_back(potential);
    return electrodeAtoms.size()-1;
}

void ConstantVForce::getElectrodeAtomParameters(int index, int& particle, double& potential) const {
    particle = electrodeAtoms[index].particle;
    potential = electrodeAtoms[index].potential;
}

void ConstantVForce::setElectrodeAtomParameters(int index, int particle, double potential) {
    electrodeAtoms[index].particle = particle;
    electrodeAtoms[index].potential = potential;
    electrodeAtomIndices[index] = particle;
    targetPotentials[index] = potential;
}

int ConstantVForce::addElectrolyteAtom(int particle, double charge) {
    electrolyteAtoms.push_back(ElectrolyteAtomInfo(particle, charge));
    electrolyteAtomIndices.push_back(particle);
    fixedCharges.push_back(charge);
    return electrolyteAtoms.size()-1;
}

void ConstantVForce::getElectrolyteAtomParameters(int index, int& particle, double& charge) const {
    particle = electrolyteAtoms[index].particle;
    charge = electrolyteAtoms[index].charge;
}

void ConstantVForce::setInverseCapacitanceMatrix(const vector<double>& flattenedMatrix) {
    int N = electrodeAtoms.size();
    if (flattenedMatrix.size() != (size_t)(N*N))
        throw OpenMMException("ConstantVForce: inverse capacitance matrix size must be N*N");
    invCapMatrix = flattenedMatrix;
}

ForceImpl* ConstantVForce::createImpl() const {
    return new ConstantVForceImpl(*this);
}
