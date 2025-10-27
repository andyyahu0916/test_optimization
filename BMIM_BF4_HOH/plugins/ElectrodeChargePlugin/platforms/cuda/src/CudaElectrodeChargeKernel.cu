#include "CudaElectrodeChargeKernel.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/internal/ContextImpl.h"

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

CudaCalcElectrodeChargeKernel::CudaCalcElectrodeChargeKernel(const std::string& name, const Platform& platform) : CalcElectrodeChargeKernel(name, platform) {
}

void CudaCalcElectrodeChargeKernel::initialize(const System& system, const ElectrodeChargeForce& force) {
    parameters.cathodeIndices = force.getCathode().atomIndices;
    parameters.anodeIndices = force.getAnode().atomIndices;
    parameters.cathodeVoltage = force.getCathode().voltage;
    parameters.anodeVoltage = force.getAnode().voltage;
    parameters.numIterations = force.getNumIterations();
    parameters.smallThreshold = force.getSmallThreshold();
    parameters.lGap = force.getCellGap();
    parameters.lCell = force.getCellLength();
    (void) system;
    cudaContext = nullptr;
}

double CudaCalcElectrodeChargeKernel::execute(ContextImpl& context,
                                              const std::vector<Vec3>& positions,
                                              const std::vector<Vec3>& forces,
                                              const std::vector<double>& allParticleCharges,
                                              double sheetArea,
                                              double cathodeZ,
                                              double anodeZ,
                                              std::vector<double>& cathodeCharges,
                                              std::vector<double>& anodeCharges,
                                              double& cathodeTarget,
                                              double& anodeTarget) {
    (void) context;
    (void) positions;
    (void) forces;
    (void) allParticleCharges;
    (void) sheetArea;
    (void) cathodeZ;
    (void) anodeZ;
    cathodeCharges.assign(parameters.cathodeIndices.size(), 0.0);
    anodeCharges.assign(parameters.anodeIndices.size(), 0.0);
    cathodeTarget = 0.0;
    anodeTarget = 0.0;
    return 0.0;
}

void CudaCalcElectrodeChargeKernel::copyParametersToContext(ContextImpl& context, const ElectrodeChargeForce& force) {
    (void) context;
    initialize(context.getSystem(), force);
}
