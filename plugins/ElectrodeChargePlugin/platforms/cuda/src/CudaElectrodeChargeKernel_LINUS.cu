/**
 * 🔥 Linus-style CUDA Kernel for ElectrodeChargeForce
 * 
 * Performance-critical Poisson solver for constant-voltage MD.
 * 
 * Bottleneck analysis (from Python profiling):
 *   - Python OPTIMIZED: ~20ms per call (3 iterations × PCIe transfers)
 *   - This CUDA version: <2ms target (GPU-resident iteration)
 * 
 * Design principles:
 *   1. Keep forces on GPU (no download between iterations)
 *   2. Iterate 3x in CUDA kernel (not in host code)
 *   3. Minimize branching (vectorize where possible)
 *   4. Use shared memory for cathode/anode charge accumulation
 * 
 * Author: Refactored by Linus principles
 * Date: 2025-10-28
 */

#include "CudaElectrodeChargeKernel.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/internal/ContextImpl.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

// ============================================================================
// CUDA Kernels (device code)
// ============================================================================

/**
 * Kernel 1: Compute analytic target charges for cathode and anode
 * 
 * This kernel calculates the total charge each electrode should have based on:
 *   1. Geometric contribution: Q_geom = sign/(4π) * A * V * (1/Lgap + 1/Lcell)
 *   2. Image charge contribution from electrolyte: Q_img = Σ(z_distance/Lcell) * (-q_i)
 * 
 * Parallelization: each thread processes one electrolyte particle
 */
__global__ void computeAnalyticTargets(
    const float4* __restrict__ posq,        // Positions and charges (x, y, z, q)
    const int numParticles,
    const int* __restrict__ cathodeIndices, // Cathode atom indices
    const int numCathode,
    const int* __restrict__ anodeIndices,   // Anode atom indices
    const int numAnode,
    const int* __restrict__ electrodeMask,  // 1 if electrode atom, 0 if electrolyte
    const float cathodeZ,
    const float anodeZ,
    const float sheetArea,
    const float cathodeVoltageKj,
    const float anodeVoltageKj,
    const float lGap,
    const float lCell,
    const float conversionKjmolNmAu,
    float* __restrict__ cathodeTargetOut,   // Output: analytic target charge
    float* __restrict__ anodeTargetOut
) {
    // Shared memory for reduction
    __shared__ float cathodeSum[256];
    __shared__ float anodeSum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory
    cathodeSum[tid] = 0.0f;
    anodeSum[tid] = 0.0f;
    
    // Constants
    const float oneOverFourPi = 1.0f / (4.0f * 3.14159265358979323846f);
    
    // Geometric contribution (only first thread in first block)
    if (idx == 0) {
        float cathodeGeom = oneOverFourPi * sheetArea * 
                           ((cathodeVoltageKj / lGap) + (cathodeVoltageKj / lCell)) * 
                           conversionKjmolNmAu;
        float anodeGeom = -oneOverFourPi * sheetArea * 
                         ((anodeVoltageKj / lGap) + (anodeVoltageKj / lCell)) * 
                         conversionKjmolNmAu;
        cathodeSum[0] = cathodeGeom;
        anodeSum[0] = anodeGeom;
    }
    
    // Image charge contribution from electrolyte
    if (idx < numParticles && electrodeMask[idx] == 0) {
        float4 particle = posq[idx];
        float charge = particle.w;
        float zPos = particle.z;
        
        float cathodeDistance = fabsf(zPos - anodeZ);
        float anodeDistance = fabsf(zPos - cathodeZ);
        
        cathodeSum[tid] += (cathodeDistance / lCell) * (-charge);
        anodeSum[tid] += (anodeDistance / lCell) * (-charge);
    }
    
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cathodeSum[tid] += cathodeSum[tid + s];
            anodeSum[tid] += anodeSum[tid + s];
        }
        __syncthreads();
    }
    
    // Write block result to global memory
    if (tid == 0) {
        atomicAdd(cathodeTargetOut, cathodeSum[0]);
        atomicAdd(anodeTargetOut, anodeSum[0]);
    }
}


/**
 * Kernel 2: Update electrode charges based on electric field
 * 
 * This implements the Poisson solver iteration:
 *   q_i = (2/4π) * A_i * (V/L_gap + E_z) * conversion
 * 
 * Where E_z = F_z / q_old (electric field from forces)
 * 
 * Parallelization: each thread processes one electrode atom
 */
__global__ void updateElectrodeCharges(
    const float4* __restrict__ force,       // Forces (x, y, z, unused)
    const float4* __restrict__ posq,        // Current charges in .w component
    const int* __restrict__ indices,        // Electrode atom indices
    const int numAtoms,
    const float areaPerAtom,
    const float voltageKj,
    const float lGap,
    const float conversionKjmolNmAu,
    const float smallThreshold,
    const float sign,                       // +1 for cathode, -1 for anode
    float* __restrict__ chargesOut          // Output: new charges
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numAtoms) {
        int atomIndex = indices[tid];
        float4 f = force[atomIndex];
        float4 pq = posq[atomIndex];
        
        float oldCharge = pq.w;
        float ezExternal = 0.0f;
        
        // Compute electric field (avoid division by zero)
        if (fabsf(oldCharge) > 0.9f * smallThreshold) {
            ezExternal = f.z / oldCharge;
        }
        
        // Update charge
        const float twoOverFourPi = 2.0f / (4.0f * 3.14159265358979323846f);
        float newCharge = sign * twoOverFourPi * areaPerAtom * 
                         ((voltageKj / lGap) + ezExternal) * conversionKjmolNmAu;
        
        // Apply threshold
        if (fabsf(newCharge) < smallThreshold) {
            newCharge = sign * smallThreshold;
        }
        
        chargesOut[tid] = newCharge;
    }
}


/**
 * Kernel 3: Scale charges to match analytic target
 * 
 * Q_numeric = Σ q_i (computed charges)
 * Q_analytic (from image charges and geometry)
 * scale = Q_analytic / Q_numeric
 * q_i *= scale
 * 
 * Two-pass: first compute sum, then scale
 */
__global__ void computeChargeSum(
    const float* __restrict__ charges,
    const int numAtoms,
    float* __restrict__ sumOut
) {
    __shared__ float sharedSum[256];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sharedSum[tid] = (idx < numAtoms) ? charges[idx] : 0.0f;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sharedSum[tid] += sharedSum[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(sumOut, sharedSum[0]);
    }
}

__global__ void scaleCharges(
    float* __restrict__ charges,
    const int numAtoms,
    const float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numAtoms) {
        charges[idx] *= scale;
    }
}


/**
 * Kernel 4: Copy electrode charges back to main charge array
 */
__global__ void copyChargesToPosq(
    float4* __restrict__ posq,
    const int* __restrict__ indices,
    const float* __restrict__ charges,
    const int numAtoms
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numAtoms) {
        int atomIndex = indices[idx];
        float4 pq = posq[atomIndex];
        pq.w = charges[idx];
        posq[atomIndex] = pq;
    }
}


// ============================================================================
// Host code (C++ member functions)
// ============================================================================

CudaCalcElectrodeChargeKernel::CudaCalcElectrodeChargeKernel(
    const std::string& name, 
    const Platform& platform
) : CalcElectrodeChargeKernel(name, platform),
    cu(nullptr),
    cathodeChargesDevice(nullptr),
    anodeChargesDevice(nullptr),
    cathodeIndicesDevice(nullptr),
    anodeIndicesDevice(nullptr),
    electrodeMaskDevice(nullptr),
    cathodeTargetDevice(nullptr),
    anodeTargetDevice(nullptr),
    chargeSum(nullptr)
{
}

void CudaCalcElectrodeChargeKernel::initialize(
    const System& system, 
    const ElectrodeChargeForce& force
) {
    // Store parameters
    parameters.cathodeIndices = force.getCathode().atomIndices;
    parameters.anodeIndices = force.getAnode().atomIndices;
    parameters.cathodeVoltage = force.getCathode().voltage;
    parameters.anodeVoltage = force.getAnode().voltage;
    parameters.numIterations = force.getNumIterations();
    parameters.smallThreshold = force.getSmallThreshold();
    parameters.lGap = force.getCellGap();
    parameters.lCell = force.getCellLength();
    
    numParticles = system.getNumParticles();
    
    // Get CUDA context (will be set when kernel is first executed)
    // Cannot get it here because Context isn't fully initialized yet
}

double CudaCalcElectrodeChargeKernel::execute(
    ContextImpl& context,
    const std::vector<Vec3>& positions,
    const std::vector<Vec3>& forces,
    const std::vector<double>& allParticleCharges,
    double sheetArea,
    double cathodeZ,
    double anodeZ,
    std::vector<double>& cathodeCharges,
    std::vector<double>& anodeCharges,
    double& cathodeTarget,
    double& anodeTarget
) {
    // Get CUDA context
    if (cu == nullptr) {
        cu = &dynamic_cast<CudaPlatform&>(context.getPlatform()).getContextByIndex(context.getContextIndex());
        initializeDeviceMemory();
    }
    
    // TODO: Implement full CUDA execution
    // For now, return stub values
    cathodeCharges.resize(parameters.cathodeIndices.size(), 0.0);
    anodeCharges.resize(parameters.anodeIndices.size(), 0.0);
    cathodeTarget = 0.0;
    anodeTarget = 0.0;
    
    return 0.0;
}

void CudaCalcElectrodeChargeKernel::initializeDeviceMemory() {
    // Allocate device memory for electrode charges
    cathodeChargesDevice = CudaArray::create<float>(*cu, parameters.cathodeIndices.size(), "cathodeCharges");
    anodeChargesDevice = CudaArray::create<float>(*cu, parameters.anodeIndices.size(), "anodeCharges");
    
    // Upload electrode indices
    cathodeIndicesDevice = CudaArray::create<int>(*cu, parameters.cathodeIndices.size(), "cathodeIndices");
    cathodeIndicesDevice->upload(parameters.cathodeIndices);
    
    anodeIndicesDevice = CudaArray::create<int>(*cu, parameters.anodeIndices.size(), "anodeIndices");
    anodeIndicesDevice->upload(parameters.anodeIndices);
    
    // Create electrode mask
    std::vector<int> electrodeMask(numParticles, 0);
    for (int idx : parameters.cathodeIndices)
        electrodeMask[idx] = 1;
    for (int idx : parameters.anodeIndices)
        electrodeMask[idx] = 1;
    electrodeMaskDevice = CudaArray::create<int>(*cu, numParticles, "electrodeMask");
    electrodeMaskDevice->upload(electrodeMask);
    
    // Allocate temporary buffers
    cathodeTargetDevice = CudaArray::create<float>(*cu, 1, "cathodeTarget");
    anodeTargetDevice = CudaArray::create<float>(*cu, 1, "anodeTarget");
    chargeSum = CudaArray::create<float>(*cu, 1, "chargeSum");
}

void CudaCalcElectrodeChargeKernel::copyParametersToContext(
    ContextImpl& context, 
    const ElectrodeChargeForce& force
) {
    // Reinitialize if parameters changed
    initialize(context.getSystem(), force);
}
