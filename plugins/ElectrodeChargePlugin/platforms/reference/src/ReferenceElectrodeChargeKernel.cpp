#include "ReferenceElectrodeChargeKernel.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include <cmath>

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

ReferenceCalcElectrodeChargeKernel::ReferenceCalcElectrodeChargeKernel(const std::string& name, const Platform& platform) : CalcElectrodeChargeKernel(name, platform) {
}

void ReferenceCalcElectrodeChargeKernel::initialize(const System& system, const ElectrodeChargeForce& force) {
    parameters.cathodeIndices = force.getCathode().atomIndices;
    parameters.anodeIndices = force.getAnode().atomIndices;
    parameters.cathodeVoltage = force.getCathode().voltage;
    parameters.anodeVoltage = force.getAnode().voltage;
    parameters.numIterations = force.getNumIterations();
    parameters.smallThreshold = force.getSmallThreshold();
    parameters.lGap = force.getCellGap();
    parameters.lCell = force.getCellLength();
    numParticles = system.getNumParticles();

    electrodeMask.assign(numParticles, false);
    conductorMask.assign(numParticles, false);  // 新增
    for (int index : parameters.cathodeIndices)
        electrodeMask[index] = true;
    for (int index : parameters.anodeIndices)
        electrodeMask[index] = true;
    // TODO: 需要從 Python 端傳入導體索引來設定 conductorMask
}

double ReferenceCalcElectrodeChargeKernel::execute(ContextImpl& context,
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
    if (numParticles == 0)
        return 0.0;
    if (positions.size() != static_cast<size_t>(numParticles) || forces.size() != static_cast<size_t>(numParticles) || allParticleCharges.size() != static_cast<size_t>(numParticles))
        throw OpenMMException("ReferenceCalcElectrodeChargeKernel received buffers with incorrect sizes.");

    if (parameters.cathodeIndices.empty() || parameters.anodeIndices.empty())
        return 0.0;

    cathodeCharges.resize(parameters.cathodeIndices.size());
    anodeCharges.resize(parameters.anodeIndices.size());

    const double conversionNmBohr = 18.8973;
    const double conversionKjmolNmAu = conversionNmBohr / 2625.5;
    const double conversionEvKjmol = 96.487;
    const double pi = 3.14159265358979323846;
    const double twoOverFourPi = 2.0 / (4.0 * pi);
    const double oneOverFourPi = 1.0 / (4.0 * pi);

    double cathodeArea = sheetArea / static_cast<double>(parameters.cathodeIndices.size());
    double anodeArea = sheetArea / static_cast<double>(parameters.anodeIndices.size());

    // CRITICAL FIX: Use absolute values for voltage magnitude
    // Sign is determined by the formula coefficients, not the voltage value
    // This matches Python implementation where Cathode.Voltage == Anode.Voltage (same magnitude)
    double cathodeVoltageKj = std::fabs(parameters.cathodeVoltage) * conversionEvKjmol;
    double anodeVoltageKj = std::fabs(parameters.anodeVoltage) * conversionEvKjmol;

    cathodeTarget = oneOverFourPi * sheetArea * ((cathodeVoltageKj / parameters.lGap) + (cathodeVoltageKj / parameters.lCell)) * conversionKjmolNmAu;
    anodeTarget = -oneOverFourPi * sheetArea * ((anodeVoltageKj / parameters.lGap) + (anodeVoltageKj / parameters.lCell)) * conversionKjmolNmAu;

    for (int i = 0; i < numParticles; i++) {
        if (electrodeMask[i])
            continue;
        double charge = allParticleCharges[i];
        double zPos = positions[i][2];
        double cathodeDistance = std::fabs(zPos - anodeZ);
        double anodeDistance = std::fabs(zPos - cathodeZ);
        cathodeTarget += (cathodeDistance / parameters.lCell) * (-charge);
        anodeTarget += (anodeDistance / parameters.lCell) * (-charge);
    }
    
    // Image charge contribution from conductors (新增)
    for (int i = 0; i < numParticles; i++) {
        if (conductorMask[i]) {
            double charge = allParticleCharges[i];
            double zPos = positions[i][2];
            double cathodeDistance = std::fabs(zPos - anodeZ);
            double anodeDistance = std::fabs(zPos - cathodeZ);
            cathodeTarget += (cathodeDistance / parameters.lCell) * (-charge);
            anodeTarget += (anodeDistance / parameters.lCell) * (-charge);
        }
    }

    // Update electrode charges
    for (size_t i = 0; i < parameters.cathodeIndices.size(); i++) {
        int index = parameters.cathodeIndices[i];
        double charge = allParticleCharges[index];
        double ezExternal = 0.0;
        if (std::fabs(charge) > 0.9 * parameters.smallThreshold)
            ezExternal = forces[index][2] / charge;
        // Cathode: positive coefficient (matches Python)
        double newCharge = twoOverFourPi * cathodeArea * ((cathodeVoltageKj / parameters.lGap) + ezExternal) * conversionKjmolNmAu;
        if (std::fabs(newCharge) < parameters.smallThreshold)
            newCharge = parameters.smallThreshold;  // Cathode: positive threshold
        cathodeCharges[i] = newCharge;
    }

    for (size_t i = 0; i < parameters.anodeIndices.size(); i++) {
        int index = parameters.anodeIndices[i];
        double charge = allParticleCharges[index];
        double ezExternal = 0.0;
        if (std::fabs(charge) > 0.9 * parameters.smallThreshold)
            ezExternal = forces[index][2] / charge;
        // Anode: negative coefficient (matches Python)
        double newCharge = -twoOverFourPi * anodeArea * ((anodeVoltageKj / parameters.lGap) + ezExternal) * conversionKjmolNmAu;
        if (std::fabs(newCharge) < parameters.smallThreshold)
            newCharge = -parameters.smallThreshold;  // Anode: negative threshold
        anodeCharges[i] = newCharge;
    }
    
    // Conductor two-stage method (if conductors exist)
    for (int i = 0; i < numParticles; i++) {
        if (conductorMask[i]) {
            // Step 1: Image charges (normal field projection)
            double charge = allParticleCharges[i];
            if (std::fabs(charge) > 0.9 * parameters.smallThreshold) {
                // Simplified: assume normal is (0,0,1) for now
                // TODO: Get actual normal from conductor geometry
                double En_external = forces[i][2] / charge;
                double newCharge = 2.0 / (4.0 * 3.14159265358979323846) * 
                                 (sheetArea / numParticles) * En_external * 
                                 (18.8973 / 2625.5);  // conversionKjmolNmAu
                
                if (std::fabs(newCharge) < parameters.smallThreshold) {
                    newCharge = parameters.smallThreshold;
                }
                
                // Update charge (simplified - would need to update NonbondedForce in real implementation)
                allParticleCharges[i] = newCharge;
            }
            
            // Step 2: Charge transfer (simplified - would need contact geometry)
            // TODO: Implement proper contact point calculation and uniform distribution
        }
    }
    
    // Apply scaling
    double cathodeTotal = std::accumulate(cathodeCharges.begin(), cathodeCharges.end(), 0.0);
    if (std::fabs(cathodeTotal) > parameters.smallThreshold) {
        double scale = cathodeTarget / cathodeTotal;
        if (scale > 0.0) {
            for (double& value : cathodeCharges)
                value *= scale;
        }
    }
    
    double anodeTotal = std::accumulate(anodeCharges.begin(), anodeCharges.end(), 0.0);
    if (std::fabs(anodeTotal) > parameters.smallThreshold) {
        double scale = anodeTarget / anodeTotal;
        if (scale > 0.0) {
            for (double& value : anodeCharges)
                value *= scale;
        }
    }

    return 0.0;
}

void ReferenceCalcElectrodeChargeKernel::copyParametersToContext(ContextImpl& context, const ElectrodeChargeForce& force) {
    initialize(context.getSystem(), force);
}
