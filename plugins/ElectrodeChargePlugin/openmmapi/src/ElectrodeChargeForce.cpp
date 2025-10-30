#include "ElectrodeChargeForce.h"
#include "internal/ElectrodeChargeForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

ElectrodeChargeForce::ElectrodeChargeForce() {
    setForceGroup(1);
}

void ElectrodeChargeForce::setCathode(const std::vector<int>& indices, double voltage) {
    params.cathodeIndices = indices;
    params.cathodeVoltage = voltage;
}

void ElectrodeChargeForce::setAnode(const std::vector<int>& indices, double voltage) {
    params.anodeIndices = indices;
    params.anodeVoltage = voltage;
}

void ElectrodeChargeForce::setNumIterations(int iterations) {
    params.numIterations = iterations;
}

void ElectrodeChargeForce::setSmallThreshold(double value) {
    params.smallThreshold = value;
}

void ElectrodeChargeForce::setCellGap(double gap) {
    params.lGap = gap;
}

void ElectrodeChargeForce::setCellLength(double length) {
    params.lCell = length;
}

void ElectrodeChargeForce::setSheetArea(double area) {
    params.sheetArea = area;
}

void ElectrodeChargeForce::setCathodeZ(double z) {
    params.cathodeZ = z;
}

void ElectrodeChargeForce::setAnodeZ(double z) {
    params.anodeZ = z;
}

void ElectrodeChargeForce::setConductorData(const std::vector<int>& indices,
                                          const std::vector<double>& normals,
                                          const std::vector<double>& areas,
                                          const std::vector<int>& contactIndices,
                                          const std::vector<double>& contactNormals,
                                          const std::vector<double>& geometries,
                                          const std::vector<int>& atomCondIds,
                                          const std::vector<int>& atomCountsPerConductor) {
    params.conductorIndices = indices;
    params.conductorNormals = normals;
    params.conductorAreas = areas;
    params.conductorContactIndices = contactIndices;
    params.conductorContactNormals = contactNormals;
    params.conductorGeometries = geometries;
    params.conductorAtomCondIds = atomCondIds;
    params.conductorAtomCounts = atomCountsPerConductor;
}

ForceImpl* ElectrodeChargeForce::createImpl() const {
    return new ElectrodeChargeForceImpl(*this);
}
