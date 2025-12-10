/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * https://openmm.org                                                         *
 *                                                                            *
 * Copyright (c) 2024 Stanford University and the Authors.                    *
 * Authors: Andy (Constant Voltage Integration)                               *
 * Contributors: Prof. McDaniel (Original Algorithm)                          *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

/**
 * CUDA Implementation of ConstantVoltage Kernels
 *
 * Implements GPU-native SCF charge updates + Drude Langevin dynamics.
 * Uses Common platform's ComputeKernel API for portable CUDA/OpenCL code.
 */

#include "CudaConstantVoltageKernels.h"
#include "ConstantVoltageKernelSources.h"
#include "openmm/ConstantVoltageForce.h"
#include "openmm/ConstantVDrudeLangevinIntegrator.h"
#include "openmm/DrudeForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaIntegrationUtilities.h"
#include "openmm/common/ContextSelector.h"
#include "SimTKOpenMMRealType.h"
#include <set>
#include <cmath>

using namespace OpenMM;
using namespace std;

// Physical constants (matching Professor's Python code)
static const double CONVERSION_KJMOL_NM_AU = 18.8973 / 2625.5;  // 0.00719475
static const double FOUR_PI = 4.0 * 3.14159265358979323846;
static const double SMALL_THRESHOLD = 1e-6;
static const double VOLTAGE_TO_KJMOL = 96.485;  // 1V * 1e = 96.485 kJ/mol
static const double BOLTZMANN_CONSTANTV = 1.380649e-23 * 6.022140857e23 / 1000.0;  // kJ/(mol*K)

// ═══════════════════════════════════════════════════════════════════════════
// CudaCalcConstantVoltageForceKernel
// ═══════════════════════════════════════════════════════════════════════════

CudaCalcConstantVoltageForceKernel::~CudaCalcConstantVoltageForceKernel() {
}

void CudaCalcConstantVoltageForceKernel::initialize(const System& system, const ConstantVoltageForce& f) {
    force = &f;
    hasInitialized = true;
    ContextSelector selector(cu);

    // Copy electrode data
    numCathodes = f.getNumCathodeAtoms();
    numAnodes = f.getNumAnodeAtoms();
    numElectrolytes = f.getNumElectrolyteAtoms();

    // Upload cathode data
    if (numCathodes > 0) {
        vector<int> cathodeIdx(numCathodes);
        vector<float> cathodeAr(numCathodes);
        for (int i = 0; i < numCathodes; i++) {
            int particle;
            double area;
            f.getCathodeAtomParameters(i, particle, area);
            cathodeIdx[i] = particle;
            cathodeAr[i] = (float)area;
        }
        cathodeIndices.initialize(cu, numCathodes, sizeof(int), "cathodeIndices");
        cathodeAreas.initialize(cu, numCathodes, sizeof(float), "cathodeAreas");
        cathodeIndices.upload(cathodeIdx);
        cathodeAreas.upload(cathodeAr);
    }

    // Upload anode data
    if (numAnodes > 0) {
        vector<int> anodeIdx(numAnodes);
        vector<float> anodeAr(numAnodes);
        for (int i = 0; i < numAnodes; i++) {
            int particle;
            double area;
            f.getAnodeAtomParameters(i, particle, area);
            anodeIdx[i] = particle;
            anodeAr[i] = (float)area;
        }
        anodeIndices.initialize(cu, numAnodes, sizeof(int), "anodeIndices");
        anodeAreas.initialize(cu, numAnodes, sizeof(float), "anodeAreas");
        anodeIndices.upload(anodeIdx);
        anodeAreas.upload(anodeAr);
    }

    // Upload electrolyte data (for Green's reciprocity calculation)
    if (numElectrolytes > 0) {
        vector<int> electrolyteIdx(numElectrolytes);
        for (int i = 0; i < numElectrolytes; i++) {
            electrolyteIdx[i] = f.getElectrolyteAtomParticle(i);
        }
        electrolyteIndices.initialize(cu, numElectrolytes, sizeof(int), "electrolyteIndices");
        electrolyteIndices.upload(electrolyteIdx);
    }

    // Store parameters (convert voltage from V to kJ/mol)
    voltage_kjmol = f.getVoltage() * VOLTAGE_TO_KJMOL;
    Lgap = f.getLgap();
    Lcell = f.getLcell();
    totalArea = f.getTotalArea();
    zCathode = f.getZCathode();
    zAnode = f.getZAnode();
    smallThreshold = f.getSmallThreshold();
    numSCFIterations = f.getNumSCFIterations();

    // ═══════════════════════════════════════════════════════════════════════
    // Initialize Conductor Support (Buckyball / Nanotube)
    // ═══════════════════════════════════════════════════════════════════════
    
    numBuckyballs = f.getNumBuckyballConductors();
    numNanotubes = f.getNumNanotubeConductors();
    totalConductorAtoms = 0;
    
    // Initialize Buckyballs
    for (int b = 0; b < numBuckyballs; b++) {
        vector<int> virtualParticles, realParticles;
        string electrodeType;
        f.getBuckyballConductorParameters(b, virtualParticles, realParticles, electrodeType);
        
        // Create GPU array for this conductor's particles
        CudaArray indices;
        indices.initialize<int>(cu, virtualParticles.size(), "buckyballIndices_" + to_string(b));
        indices.upload(virtualParticles);
        conductorIndices.push_back(std::move(indices));
        
        // Normals will be computed in geometry initialization kernel
        CudaArray normals;
        normals.initialize<float4>(cu, virtualParticles.size(), "buckyballNormals_" + to_string(b));
        conductorNormals.push_back(std::move(normals));
        
        conductorTypes.push_back(0);  // CONDUCTOR_BUCKYBALL
        conductorLengths.push_back(0.0f);  // Not used for Buckyball
        totalConductorAtoms += virtualParticles.size();
    }
    
    // Initialize Nanotubes
    for (int n = 0; n < numNanotubes; n++) {
        vector<int> virtualParticles, realParticles;
        string electrodeType;
        Vec3 axis;
        f.getNanotubeConductorParameters(n, virtualParticles, realParticles, electrodeType, axis);
        
        // Create GPU array for this conductor's particles
        CudaArray indices;
        indices.initialize<int>(cu, virtualParticles.size(), "nanotubeIndices_" + to_string(n));
        indices.upload(virtualParticles);
        conductorIndices.push_back(std::move(indices));
        
        // Normals will be computed in geometry initialization kernel
        CudaArray normals;
        normals.initialize<float4>(cu, virtualParticles.size(), "nanotubeNormals_" + to_string(n));
        conductorNormals.push_back(std::move(normals));
        
        conductorTypes.push_back(1);  // CONDUCTOR_NANOTUBE
        // Nanotube length - using Lcell as approximation (should come from box dimension)
        conductorLengths.push_back((float)Lcell);
        totalConductorAtoms += virtualParticles.size();
    }
    
    // Create flattened conductor indices for combined scaling
    if (totalConductorAtoms > 0) {
        vector<int> allIdx;
        for (int c = 0; c < numBuckyballs + numNanotubes; c++) {
            // Download indices from GPU array
            int size = (c < numBuckyballs) ? 
                f.getNumBuckyballConductors() : f.getNumNanotubeConductors();
            // For now, reserve space - actual flattening done per-conductor
        }
        allConductorIndices.initialize<int>(cu, max(totalConductorAtoms, 1), "allConductorIndices");
    }

    // Allocate reduction buffers
    totalCathodeCharge.initialize<float>(cu, 1, "totalCathodeCharge");
    totalAnodeCharge.initialize<float>(cu, 1, "totalAnodeCharge");

    // Compile CUDA kernels from JIT source
    map<string, string> defines;
    defines["NUM_CATHODES"] = cu.intToString(numCathodes);
    defines["NUM_ANODES"] = cu.intToString(numAnodes);
    defines["NUM_ELECTROLYTES"] = cu.intToString(numElectrolytes);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["SMALL_THRESHOLD"] = cu.doubleToString(SMALL_THRESHOLD);
    defines["CONVERSION_KJMOL_NM_AU"] = cu.doubleToString(CONVERSION_KJMOL_NM_AU);
    defines["FOUR_PI"] = cu.doubleToString(FOUR_PI);

    CUmodule module = cu.createModule(ConstantVoltageKernelSources::constantVoltage, defines);
    updateCathodeChargesKernel = cu.getKernel(module, "updateCathodeCharges");
    updateAnodeChargesKernel = cu.getKernel(module, "updateAnodeCharges");
    computeElectrodeChargeKernel = cu.getKernel(module, "computeElectrodeCharge");
    computeAnalyticChargeKernel = cu.getKernel(module, "computeAnalyticCharge");
    scaleElectrodeChargesKernel = cu.getKernel(module, "scaleElectrodeCharges");
    
    // Compile conductor kernels if needed
    if (numBuckyballs + numNanotubes > 0) {
        CUmodule conductorModule = cu.createModule(ConstantVoltageKernelSources::conductorCharge, defines);
        computeConductorImageChargesKernel = cu.getKernel(conductorModule, "computeConductorImageCharges");
        computeConductorChargeTransferKernel = cu.getKernel(conductorModule, "computeConductorChargeTransfer");
        scaleElectrodeChargesWithConductorsKernel = cu.getKernel(conductorModule, "scaleElectrodeChargesWithConductors");
        initConductorGeometryKernel = cu.getKernel(conductorModule, "initConductorGeometry");
    }
}

double CudaCalcConstantVoltageForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // ConstantVoltageForce doesn't compute forces directly
    // SCF charge updates are triggered by the Integrator
    return 0.0;
}

void CudaCalcConstantVoltageForceKernel::updateElectrodeCharges(ContextImpl& context) {
    if (!hasInitialized || (numCathodes == 0 && numAnodes == 0))
        return;
    
    ContextSelector selector(cu);
    
    // Get buffer pointers for kernel arguments
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    
    // Update cathode charges
    if (numCathodes > 0) {
        float v_kjmol = (float)voltage_kjmol;
        float lgap = (float)Lgap;
        CUdeviceptr cathodeIdxPtr = cathodeIndices.getDevicePointer();
        CUdeviceptr cathodeAreaPtr = cathodeAreas.getDevicePointer();
        
        void* cathodeArgs[] = {
            &posqPtr,
            &forcePtr,
            &cathodeIdxPtr,
            &cathodeAreaPtr,
            &numCathodes,
            &paddedNumAtoms,
            &v_kjmol,
            &lgap
        };
        cu.executeKernel(updateCathodeChargesKernel, cathodeArgs, numCathodes);
    }
    
    // Update anode charges
    if (numAnodes > 0) {
        float v_kjmol = (float)voltage_kjmol;
        float lgap = (float)Lgap;
        CUdeviceptr anodeIdxPtr = anodeIndices.getDevicePointer();
        CUdeviceptr anodeAreaPtr = anodeAreas.getDevicePointer();
        
        void* anodeArgs[] = {
            &posqPtr,
            &forcePtr,
            &anodeIdxPtr,
            &anodeAreaPtr,
            &numAnodes,
            &paddedNumAtoms,
            &v_kjmol,
            &lgap
        };
        cu.executeKernel(updateAnodeChargesKernel, anodeArgs, numAnodes);
    }
    
    // TODO: Implement analytic charge scaling (Green's reciprocity)
}

double CudaCalcConstantVoltageForceKernel::getTotalCathodeCharge(ContextImpl& context) {
    if (numCathodes == 0)
        return 0.0;
    
    ContextSelector selector(cu);
    
    // Reset accumulator
    float zero = 0.0f;
    totalCathodeCharge.upload(&zero);
    
    // Compute charge sum
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr cathodeIdxPtr = cathodeIndices.getDevicePointer();
    CUdeviceptr chargePtr = totalCathodeCharge.getDevicePointer();
    
    void* args[] = {
        &posqPtr,
        &cathodeIdxPtr,
        &numCathodes,
        &chargePtr
    };
    cu.executeKernel(computeElectrodeChargeKernel, args, numCathodes);
    
    // Download result
    float result;
    totalCathodeCharge.download(&result);
    return (double)result;
}

double CudaCalcConstantVoltageForceKernel::getTotalAnodeCharge(ContextImpl& context) {
    if (numAnodes == 0)
        return 0.0;
    
    ContextSelector selector(cu);
    
    // Reset accumulator
    float zero = 0.0f;
    totalAnodeCharge.upload(&zero);
    
    // Compute charge sum
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr anodeIdxPtr = anodeIndices.getDevicePointer();
    CUdeviceptr chargePtr = totalAnodeCharge.getDevicePointer();
    
    void* args[] = {
        &posqPtr,
        &anodeIdxPtr,
        &numAnodes,
        &chargePtr
    };
    cu.executeKernel(computeElectrodeChargeKernel, args, numAnodes);
    
    // Download result
    float result;
    totalAnodeCharge.download(&result);
    return (double)result;
}

// ═══════════════════════════════════════════════════════════════════════════
// CudaIntegrateConstantVDrudeLangevinStepKernel
// ═══════════════════════════════════════════════════════════════════════════

CudaIntegrateConstantVDrudeLangevinStepKernel::~CudaIntegrateConstantVDrudeLangevinStepKernel() {
}

void CudaIntegrateConstantVDrudeLangevinStepKernel::initialize(
    const System& system,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    hasInitialized = true;
    stepCount = 0;
    maxDrudeDistance = integrator.getMaxDrudeDistance();
    prevStepSize = -1.0;
    ContextSelector selector(cu);
    
    // Initialize random number generator
    cu.getIntegrationUtilities().initRandomNumberGenerator((unsigned int)integrator.getRandomNumberSeed());

    // Find ConstantVoltageForce to get SCF parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const ConstantVoltageForce* cvForce = dynamic_cast<const ConstantVoltageForce*>(&system.getForce(i));
        if (cvForce != nullptr) {
            scfFrequency = cvForce->getSCFFrequency();
            numSCFIterations = cvForce->getNumSCFIterations();
            break;
        }
    }

    // Find DrudeForce and identify Drude particle pairs
    const DrudeForce* drudeForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        drudeForce = dynamic_cast<const DrudeForce*>(&system.getForce(i));
        if (drudeForce != nullptr)
            break;
    }

    // Identify particle pairs and ordinary particles
    set<int> particles;
    vector<int> normalParticleVec;
    vector<int2> pairParticleVec;
    
    for (int i = 0; i < system.getNumParticles(); i++)
        particles.insert(i);
    
    if (drudeForce != nullptr) {
        for (int i = 0; i < drudeForce->getNumParticles(); i++) {
            int p, p1, p2, p3, p4;
            double charge, polarizability, aniso12, aniso34;
            drudeForce->getParticleParameters(i, p, p1, p2, p3, p4, charge, polarizability, aniso12, aniso34);
            particles.erase(p);   // Drude particle
            particles.erase(p1);  // Parent particle
            int2 pair;
            pair.x = p;
            pair.y = p1;
            pairParticleVec.push_back(pair);
        }
    }
    
    normalParticleVec.insert(normalParticleVec.begin(), particles.begin(), particles.end());
    
    // Upload particle arrays to GPU
    normalParticles.initialize<int>(cu, max((int)normalParticleVec.size(), 1), "constantVNormalParticles");
    pairParticles.initialize<int2>(cu, max((int)pairParticleVec.size(), 1), "constantVPairParticles");
    
    if (normalParticleVec.size() > 0)
        normalParticles.upload(normalParticleVec);
    if (pairParticleVec.size() > 0)
        pairParticles.upload(pairParticleVec);
    
    numNormalParticles = normalParticleVec.size();
    numDrudePairs = pairParticleVec.size();

    // Compile Drude Langevin integration kernels
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_NORMAL_PARTICLES"] = cu.intToString(numNormalParticles);
    defines["NUM_PAIRS"] = cu.intToString(numDrudePairs);

    CUmodule module = cu.createModule(ConstantVoltageKernelSources::drudeLangevin, defines);
    kernel1 = cu.getKernel(module, "integrateDrudeLangevinPart1");
    kernelPairs = cu.getKernel(module, "integrateDrudePairs");
    kernel2 = cu.getKernel(module, "integrateDrudeLangevinPart2");
    hardwallKernel = cu.getKernel(module, "applyHardWallConstraints");
}

void CudaIntegrateConstantVDrudeLangevinStepKernel::execute(
    ContextImpl& context,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    ContextSelector selector(cu);
    CudaIntegrationUtilities& integration = cu.getIntegrationUtilities();
    int numAtoms = cu.getNumAtoms();
    
    // Check if SCF update is needed
    if (stepCount % scfFrequency == 0 && forceKernel != nullptr) {
        for (int i = 0; i < numSCFIterations; i++) {
            // Recalculate forces
            context.calcForcesAndEnergy(true, false, context.getIntegrator().getIntegrationForceGroups());
            // Update electrode charges
            forceKernel->updateElectrodeCharges(context);
        }
    }

    // Compute integrator coefficients
    double stepSize = integrator.getStepSize();
    double temperature = integrator.getTemperature();
    double friction = integrator.getFriction();
    double drudeTemperature = integrator.getDrudeTemperature();
    double drudeFriction = integrator.getDrudeFriction();
    
    double vscale = exp(-stepSize * friction);
    double fscale = (1 - vscale) / friction / (double)0x100000000;
    double noisescale = sqrt(2 * BOLTZMANN_CONSTANTV * temperature * friction) * sqrt(0.5 * (1 - vscale*vscale) / friction);
    
    double vscaleDrude = exp(-stepSize * drudeFriction);
    double fscaleDrude = (1 - vscaleDrude) / drudeFriction / (double)0x100000000;
    double noisescaleDrude = sqrt(2 * BOLTZMANN_CONSTANTV * drudeTemperature * drudeFriction) * sqrt(0.5 * (1 - vscaleDrude*vscaleDrude) / drudeFriction);
    
    double hardwallscaleDrude = sqrt(BOLTZMANN_CONSTANTV * drudeTemperature);
    
    // Update step size buffer if needed
    if (stepSize != prevStepSize) {
        float2 ss;
        ss.x = 0.0f;
        ss.y = (float)stepSize;
        integration.getStepSize().upload(&ss);
        prevStepSize = stepSize;
    }

    // Get buffer pointers
    CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr posDeltaPtr = integration.getPosDelta().getDevicePointer();
    CUdeviceptr normalPtr = normalParticles.getDevicePointer();
    CUdeviceptr pairPtr = pairParticles.getDevicePointer();
    CUdeviceptr stepSizePtr = integration.getStepSize().getDevicePointer();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    
    // Get random numbers
    unsigned int randomIndex = integration.prepareRandomNumbers(numNormalParticles + 2*numDrudePairs);
    CUdeviceptr randomPtr = integration.getRandom().getDevicePointer();
    
    // Convert to float for kernel args
    float vscaleF = (float)vscale;
    float fscaleF = (float)fscale;
    float noisescaleF = (float)noisescale;
    float vscaleDrudeF = (float)vscaleDrude;
    float fscaleDrudeF = (float)fscaleDrude;
    float noisescaleDrudeF = (float)noisescaleDrude;
    float maxDrudeDistF = (float)maxDrudeDistance;
    float hardwallscaleF = (float)hardwallscaleDrude;

    // Execute Part 1: Update normal particles
    if (numNormalParticles > 0) {
        void* args1[] = {
            &velmPtr,
            &forcePtr,
            &posDeltaPtr,
            &normalPtr,
            &pairPtr,
            &stepSizePtr,
            &vscaleF,
            &fscaleF,
            &noisescaleF,
            &vscaleDrudeF,
            &fscaleDrudeF,
            &noisescaleDrudeF,
            &randomPtr,
            &randomIndex,
            &numNormalParticles,
            &numDrudePairs,
            &paddedNumAtoms
        };
        cu.executeKernel(kernel1, args1, numNormalParticles);
    }
    
    // Execute: Update Drude pairs
    if (numDrudePairs > 0) {
        unsigned int pairRandomIndex = randomIndex + numNormalParticles;
        void* argsPairs[] = {
            &velmPtr,
            &forcePtr,
            &posDeltaPtr,
            &pairPtr,
            &stepSizePtr,
            &vscaleF,
            &fscaleF,
            &noisescaleF,
            &vscaleDrudeF,
            &fscaleDrudeF,
            &noisescaleDrudeF,
            &randomPtr,
            &pairRandomIndex,
            &numDrudePairs,
            &paddedNumAtoms
        };
        cu.executeKernel(kernelPairs, argsPairs, numDrudePairs);
    }
    
    // Apply constraints
    integration.applyConstraints(integrator.getConstraintTolerance());
    
    // Execute Part 2: Update positions
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    void* args2[] = {
        &posqPtr,
        &posDeltaPtr,
        &velmPtr,
        &stepSizePtr,
        &numAtoms
    };
    cu.executeKernel(kernel2, args2, numAtoms);
    
    // Apply hard wall constraints
    if (maxDrudeDistance > 0 && numDrudePairs > 0) {
        void* argsHW[] = {
            &posqPtr,
            &velmPtr,
            &pairPtr,
            &stepSizePtr,
            &maxDrudeDistF,
            &hardwallscaleF,
            &numDrudePairs
        };
        cu.executeKernel(hardwallKernel, argsHW, numDrudePairs);
    }
    
    // Compute virtual sites
    integration.computeVirtualSites();

    // Update time and step count
    stepCount++;
    cu.setTime(cu.getTime() + stepSize);
    cu.setStepCount(cu.getStepCount() + 1);
    cu.reorderAtoms();
}

double CudaIntegrateConstantVDrudeLangevinStepKernel::computeKineticEnergy(
    ContextImpl& context,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    return cu.getIntegrationUtilities().computeKineticEnergy(0.5 * integrator.getStepSize());
}
