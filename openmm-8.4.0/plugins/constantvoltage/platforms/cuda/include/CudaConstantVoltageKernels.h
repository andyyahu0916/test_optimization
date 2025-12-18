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

#ifndef OPENMM_CUDA_CONSTANTVOLTAGE_KERNELS_H_
#define OPENMM_CUDA_CONSTANTVOLTAGE_KERNELS_H_

#include "openmm/ConstantVoltageKernels.h"
#include "openmm/ConstantVoltageForce.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <cuda_runtime.h>  // For float3

namespace OpenMM {

/**
 * This kernel is invoked by ConstantVoltageForce to handle electrode data on GPU.
 */
class CudaCalcConstantVoltageForceKernel : public CalcConstantVoltageForceKernel {
public:
    CudaCalcConstantVoltageForceKernel(std::string name, const Platform& platform, CudaContext& cu)
        : CalcConstantVoltageForceKernel(name, platform), cu(cu), hasInitialized(false) {}
    ~CudaCalcConstantVoltageForceKernel();

    void initialize(const System& system, const ConstantVoltageForce& force) override;
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void updateElectrodeCharges(ContextImpl& context) override;
    double getTotalCathodeCharge(ContextImpl& context) override;
    double getTotalAnodeCharge(ContextImpl& context) override;

private:
    CudaContext& cu;
    bool hasInitialized;
    const ConstantVoltageForce* force;

    // GPU buffers
    CudaArray cathodeIndices;
    CudaArray cathodeAreas;
    CudaArray anodeIndices;
    CudaArray anodeAreas;
    CudaArray electrolyteIndices;
    CudaArray totalCathodeCharge;
    CudaArray totalAnodeCharge;

    // Parameters
    int numCathodes;
    int numAnodes;
    int numElectrolytes;
    double voltage_kjmol;
    double Lgap;
    double Lcell;
    double totalArea;
    double zCathode;
    double zAnode;
    double smallThreshold;
    int numSCFIterations;

    // Kernels (JIT compiled)
    CUfunction updateCathodeChargesKernel;
    CUfunction updateAnodeChargesKernel;
    CUfunction computeElectrodeChargeKernel;
    CUfunction computeAnalyticChargeKernel;
    CUfunction scaleElectrodeChargesKernel;

    // ═══════════════════════════════════════════════════════════════════════
    // Conductor Support (Buckyball / Nanotube)
    // ═══════════════════════════════════════════════════════════════════════
    
    // Conductor data
    int numBuckyballs;
    int numNanotubes;
    std::vector<CudaArray> conductorIndices;      // Per-conductor particle indices
    std::vector<CudaArray> conductorNormals;      // Per-conductor surface normals
    std::vector<float> conductorAreas;            // Area per atom for each conductor
    std::vector<int> conductorTypes;              // 0=Buckyball, 1=Nanotube
    std::vector<float> conductorDrContact;        // dr_center_contact for each conductor
    std::vector<float> conductorLengths;          // Length (Nanotube only)
    std::vector<int> conductorContactAtoms;       // Contact atom index
    std::vector<float3> conductorContactNormals;  // Contact normal
    std::vector<bool> conductorIsCloseToElectrode;
    
    // Flattened conductor indices for combined scaling
    CudaArray allConductorIndices;
    int totalConductorAtoms;
    
    // Conductor kernels
    CUfunction computeConductorImageChargesKernel;
    CUfunction computeConductorChargeTransferKernel;
    CUfunction scaleElectrodeChargesWithConductorsKernel;
    CUfunction initConductorGeometryKernel;
    
    // Helper methods
    void initializeConductorGeometry(ContextImpl& context);
    
    /**
     * Compute analytic charge including electrolyte image charge contribution.
     * Reference: Fixed_Voltage_routines.py:318-344
     * 
     * @param context The context to get positions from
     * @param z_opposite Z position of the opposite electrode
     * @param isCathode True for cathode, false for anode
     * @return The analytic charge value
     */
    double computeAnalyticChargeWithElectrolyte(ContextImpl& context, double z_opposite, bool isCathode);
    
    /**
     * Compute analytic charge including both electrolyte and conductor contributions.
     * Reference: Fixed_Voltage_routines.py:336-344
     * 
     * @param context The context to get positions from
     * @param z_opposite Z position of the opposite electrode
     * @param isCathode True for cathode, false for anode
     * @return The analytic charge value
     */
    double computeAnalyticChargeWithElectrolytePlusConductors(ContextImpl& context, double z_opposite, bool isCathode);
    
    // Reduction buffer for analytic charge calculation
    CudaArray analyticChargeBuffer;
};

/**
 * This kernel is invoked by ConstantVDrudeLangevinIntegrator to take one time step.
 */
class CudaIntegrateConstantVDrudeLangevinStepKernel : public IntegrateConstantVDrudeLangevinStepKernel {
public:
    CudaIntegrateConstantVDrudeLangevinStepKernel(std::string name, const Platform& platform, CudaContext& cu)
        : IntegrateConstantVDrudeLangevinStepKernel(name, platform), cu(cu), hasInitialized(false), 
          forceKernel(nullptr), prevStepSize(-1.0) {}
    ~CudaIntegrateConstantVDrudeLangevinStepKernel();

    void initialize(const System& system, const ConstantVDrudeLangevinIntegrator& integrator) override;
    void execute(ContextImpl& context, const ConstantVDrudeLangevinIntegrator& integrator) override;
    double computeKineticEnergy(ContextImpl& context, const ConstantVDrudeLangevinIntegrator& integrator) override;

private:
    CudaContext& cu;
    bool hasInitialized;
    int stepCount;
    int scfFrequency;
    int numSCFIterations;
    double maxDrudeDistance;
    double prevStepSize;
    CudaCalcConstantVoltageForceKernel* forceKernel;

    // Drude particle data (GPU buffers)
    CudaArray pairParticles;   // int2: (drude, parent)
    CudaArray normalParticles;
    int numDrudePairs;
    int numNormalParticles;

    // Integration kernels (JIT compiled from drudeLangevin.cu)
    CUfunction kernel1;         // integrateDrudeLangevinPart1
    CUfunction kernelPairs;     // integrateDrudePairs
    CUfunction kernel2;         // integrateDrudeLangevinPart2
    CUfunction hardwallKernel;  // applyHardWallConstraints
};

} // namespace OpenMM

#endif // OPENMM_CUDA_CONSTANTVOLTAGE_KERNELS_H_
