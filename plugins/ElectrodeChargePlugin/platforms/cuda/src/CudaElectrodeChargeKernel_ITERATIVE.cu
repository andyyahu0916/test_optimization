/**
 * 🔥 CUDA Kernel with GPU-resident Iteration
 * 
 * Goal: Keep ALL data on GPU during Poisson solver iterations.
 * No CPU-GPU transfers inside iteration loop!
 * 
 * Architecture:
 *   1. Upload initial data (positions, charges) - ONCE
 *   2. Iterate 3 times on GPU:
 *      - Compute Coulomb forces (our own kernel)
 *      - Update electrode charges
 *      - Scale to target
 *   3. Download final charges - ONCE
 * 
 * Expected speedup: 50-100× over CPU iteration
 */

#include "CudaElectrodeChargeKernel.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

// ============================================================================
// CUDA Kernels for GPU-resident iteration
// ============================================================================

// Removed: computeCoulombForcesSimple. We must use NonbondedForce forces for physical equivalence.


/**
 * Kernel: Update electrode charges from electric field
 * Combined for both cathode and anode to minimize kernel launches
 */
__global__ void updateElectrodeChargesIterative(
    const float3* __restrict__ forces,       // Input forces
    float4* __restrict__ posq,               // In/Out: positions + charges
    const int* __restrict__ cathodeIndices,
    const int numCathode,
    const int* __restrict__ anodeIndices,
    const int numAnode,
    const float cathodeArea,
    const float anodeArea,
    const float cathodeVoltageKj,
    const float anodeVoltageKj,
    const float lGap,
    const float conversionKjmolNmAu,
    const float smallThreshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    const float twoOverFourPi = 2.0f / (4.0f * 3.14159265358979323846f);
    
    // Process cathode
    if (tid < numCathode) {
        int atomIdx = cathodeIndices[tid];
        float4 pq = posq[atomIdx];
        float3 force = forces[atomIdx];
        
        float oldCharge = pq.w;
        float ezExternal = 0.0f;
        
        if (fabsf(oldCharge) > 0.9f * smallThreshold) {
            ezExternal = force.z / oldCharge;
        }
        
        float newCharge = twoOverFourPi * cathodeArea * 
                         ((cathodeVoltageKj / lGap) + ezExternal) * conversionKjmolNmAu;
        
        // branchless clamp to preserve threshold rule
        float absNC = fabsf(newCharge);
        float clamped = fmaxf(absNC, smallThreshold);
        newCharge = copysignf(clamped, 1.0f);
        
        pq.w = newCharge;
        posq[atomIdx] = pq;
    }
    
    // Process anode
    int anodeTid = tid - numCathode;
    if (anodeTid >= 0 && anodeTid < numAnode) {
        int atomIdx = anodeIndices[anodeTid];
        float4 pq = posq[atomIdx];
        float3 force = forces[atomIdx];
        
        float oldCharge = pq.w;
        float ezExternal = 0.0f;
        
        if (fabsf(oldCharge) > 0.9f * smallThreshold) {
            ezExternal = force.z / oldCharge;
        }
        
        float newCharge = -twoOverFourPi * anodeArea * 
                         ((anodeVoltageKj / lGap) + ezExternal) * conversionKjmolNmAu;
        
        // branchless clamp with negative sign for anode
        float absNA = fabsf(newCharge);
        float clampedA = fmaxf(absNA, smallThreshold);
        newCharge = -clampedA;
        
        pq.w = newCharge;
        posq[atomIdx] = pq;
    }
}


/**
 * Kernel: Compute analytic target and scale charges (包含導體影像項)
 */
__global__ void computeTargetAndScale(
    const float4* __restrict__ posq,
    float4* __restrict__ posqOut,           // Output after scaling
    const int* __restrict__ cathodeIndices,
    const int numCathode,
    const int* __restrict__ anodeIndices,
    const int numAnode,
    const int* __restrict__ electrodeMask,  // 1=electrode, 0=electrolyte
    const int* __restrict__ conductorMask,  // 1=conductor, 0=other (新增)
    const int numParticles,
    const float cathodeZ,
    const float anodeZ,
    const float sheetArea,
    const float cathodeVoltageKj,
    const float anodeVoltageKj,
    const float lGap,
    const float lCell,
    const float conversionKjmolNmAu,
    const float smallThreshold,
    float* __restrict__ cathodeScaleOut,    // Output: scaling factor (kept for compatibility)
    float* __restrict__ anodeScaleOut
) {
    __shared__ float cathodeSumShared[256];
    __shared__ float anodeSumShared[256];
    __shared__ float cathodeTargetShared[256];
    __shared__ float anodeTargetShared[256];
    __shared__ float scaleCathode;
    __shared__ float scaleAnode;
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    cathodeSumShared[tid] = 0.0f;
    anodeSumShared[tid] = 0.0f;
    cathodeTargetShared[tid] = 0.0f;
    anodeTargetShared[tid] = 0.0f;
    
    const float oneOverFourPi = 1.0f / (4.0f * 3.14159265358979323846f);
    
    // Geometric contribution (thread 0 only)
    if (idx == 0) {
        float cathodeGeom = oneOverFourPi * sheetArea * 
                           ((cathodeVoltageKj / lGap) + (cathodeVoltageKj / lCell)) * 
                           conversionKjmolNmAu;
        float anodeGeom = -oneOverFourPi * sheetArea * 
                         ((anodeVoltageKj / lGap) + (anodeVoltageKj / lCell)) * 
                         conversionKjmolNmAu;
        cathodeTargetShared[0] = cathodeGeom;
        anodeTargetShared[0] = anodeGeom;
    }
    
    // Sum electrode charges
    if (idx < numCathode) {
        int atomIdx = cathodeIndices[idx];
        cathodeSumShared[tid] = posq[atomIdx].w;
    }
    
    if (idx < numAnode) {
        int atomIdx = anodeIndices[idx];
        anodeSumShared[tid] = posq[atomIdx].w;
    }
    
    // Image charge contribution from electrolyte AND conductors
    if (idx < numParticles && electrodeMask[idx] == 0) {
        float4 particle = posq[idx];
        float charge = particle.w;
        float zPos = particle.z;
        
        float cathodeDistance = fabsf(zPos - anodeZ);
        float anodeDistance = fabsf(zPos - cathodeZ);
        
        cathodeTargetShared[tid] += (cathodeDistance / lCell) * (-charge);
        anodeTargetShared[tid] += (anodeDistance / lCell) * (-charge);
    }
    
    // Image charge contribution from conductors (新增)
    if (idx < numParticles && conductorMask[idx] == 1) {
        float4 particle = posq[idx];
        float charge = particle.w;
        float zPos = particle.z;
        
        float cathodeDistance = fabsf(zPos - anodeZ);
        float anodeDistance = fabsf(zPos - cathodeZ);
        
        cathodeTargetShared[tid] += (cathodeDistance / lCell) * (-charge);
        anodeTargetShared[tid] += (anodeDistance / lCell) * (-charge);
    }
    
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            cathodeSumShared[tid] += cathodeSumShared[tid + s];
            anodeSumShared[tid] += anodeSumShared[tid + s];
            cathodeTargetShared[tid] += cathodeTargetShared[tid + s];
            anodeTargetShared[tid] += anodeTargetShared[tid + s];
        }
        __syncthreads();
    }
    
    // Compute scaling factors
    if (tid == 0) {
        float cathodeSum = cathodeSumShared[0];
        float anodeSum = anodeSumShared[0];
        float cathodeTarget = cathodeTargetShared[0];
        float anodeTarget = anodeTargetShared[0];
        
        float cathodeScale = 1.0f;
        if (fabsf(cathodeSum) > smallThreshold && cathodeTarget > 0.0f) {
            cathodeScale = cathodeTarget / cathodeSum;
        }
        
        float anodeScale = 1.0f;
        if (fabsf(anodeSum) > smallThreshold && anodeTarget < 0.0f) {
            anodeScale = anodeTarget / anodeSum;
        }
        // write back for compatibility and store to shared for in-kernel scaling
        atomicExch(cathodeScaleOut, cathodeScale);
        atomicExch(anodeScaleOut, anodeScale);
        scaleCathode = cathodeScale;
        scaleAnode = anodeScale;
    }
    __syncthreads();

    // Apply scaling in the same kernel (eliminate extra launch)
    if (idx < numCathode) {
        int atomIdx = cathodeIndices[idx];
        float4 pq = posqOut[atomIdx];
        pq.w *= scaleCathode;
        posqOut[atomIdx] = pq;
    }
    if (idx < numAnode) {
        int atomIdx = anodeIndices[idx];
        float4 pq = posqOut[atomIdx];
        pq.w *= scaleAnode;
        posqOut[atomIdx] = pq;
    }
}


/**
 * Kernel: Conductor two-stage method - Step 1: Image charges (normal field projection)
 */
__global__ void conductorImageCharges(
    const float3* __restrict__ forces,
    float4* __restrict__ posq,
    const int* __restrict__ conductorIndices,
    const int numConductors,
    const float3* __restrict__ conductorNormals,  // nx, ny, nz for each conductor atom
    const float* __restrict__ conductorAreas,     // area_atom for each conductor
    const float conversionKjmolNmAu,
    const float smallThreshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < numConductors) {
        int atomIdx = conductorIndices[tid];
        float4 pq = posq[atomIdx];
        float3 force = forces[atomIdx];
        float3 normal = conductorNormals[tid];
        
        float oldCharge = pq.w;
        float ezExternal = 0.0f;
        
        if (fabsf(oldCharge) > 0.9f * smallThreshold) {
            // Project field to surface normal
            float En_external = (force.x * normal.x + force.y * normal.y + force.z * normal.z) / oldCharge;
            // Solve for surface charge requiring normal field be zero inside conductor
            float newCharge = 2.0f / (4.0f * 3.14159265358979323846f) * conductorAreas[tid] * En_external * conversionKjmolNmAu;
            // branchless clamp positive
            float absNC = fabsf(newCharge);
            float clamped = fmaxf(absNC, smallThreshold);
            newCharge = copysignf(clamped, 1.0f);
            pq.w = newCharge;
            posq[atomIdx] = pq;
        }
    }
}

/**
 * Kernel: Conductor two-stage method - Step 2: Charge transfer (uniform distribution)
 */
__global__ void conductorChargeTransfer(
    const float3* __restrict__ forces,
    const int* __restrict__ conductorContactIndices,  // per conductor
    const float3* __restrict__ conductorContactNormals,
    const float* __restrict__ conductorGeometries,
    const int* __restrict__ conductorTypes,
    float* __restrict__ dQPerConductor,
    const int numConductors,
    const float cathodeVoltageKj,
    const float lGap,
    const float conversionKjmolNmAu,
    const float smallThreshold
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numConductors) {
        int contactIdx = conductorContactIndices[tid];
        float3 force = forces[contactIdx];
        float3 normal = conductorContactNormals[tid];
        float q_i = 1.0f; // threshold guard only
        float En_external = 0.0f;
        if (fabsf(q_i) > 0.9f * smallThreshold) {
            En_external = (force.x * normal.x + force.y * normal.y + force.z * normal.z) / q_i;
        }
        float dE_conductor = -(En_external + cathodeVoltageKj / lGap / 2.0f) * conversionKjmolNmAu;
        float dQ_conductor = (conductorTypes[tid] == 0)
            ? (-1.0f * dE_conductor * conductorGeometries[tid] * conductorGeometries[tid])
            : (-1.0f * dE_conductor * conductorGeometries[tid]);
        dQPerConductor[tid] = dQ_conductor;
    }
}

__global__ void applyConductorDQ(
    float4* __restrict__ posq,
    const int* __restrict__ conductorIndices,      // per atom
    const int* __restrict__ conductorAtomCondIds,  // per atom -> cond id
    const int numConductorAtoms,
    const float* __restrict__ dQPerConductor,      // per conductor
    const int* __restrict__ atomCountsPerConductor // per conductor
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < numConductorAtoms) {
        int atomIdx = conductorIndices[tid];
        int condId = conductorAtomCondIds[tid];
        int count = atomCountsPerConductor[condId];
        float dq = dQPerConductor[condId] / (float)count;
        float4 pq = posq[atomIdx];
        pq.w += dq;
        posq[atomIdx] = pq;
    }
}


/**
 * Kernel: Apply scaling to electrode charges (reads scale from device)
 */
__global__ void applyScaling(
    float4* __restrict__ posq,
    const int* __restrict__ cathodeIndices,
    const int numCathode,
    const int* __restrict__ anodeIndices,
    const int numAnode,
    const float* __restrict__ cathodeScalePtr,
    const float* __restrict__ anodeScalePtr
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float cathodeScale = *cathodeScalePtr;
    float anodeScale = *anodeScalePtr;

    if (tid < numCathode) {
        int atomIdx = cathodeIndices[tid];
        float4 pq = posq[atomIdx];
        pq.w *= cathodeScale;
        posq[atomIdx] = pq;
    }

    int anodeTid = tid - numCathode;
    if (anodeTid >= 0 && anodeTid < numAnode) {
        int atomIdx = anodeIndices[anodeTid];
        float4 pq = posq[atomIdx];
        pq.w *= anodeScale;
        posq[atomIdx] = pq;
    }
}

// ============================================================================
// Host code: execute with GPU-resident iteration
// ============================================================================

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
    ContextSelector selector(*cu);
    
    // Initialize persistent buffers on first call
    if (forcesDevicePersistent == nullptr) {
        forcesDevicePersistent = CudaArray::create<float3>(*cu, numParticles, "forcesPersistent");
        posqDevicePersistent = CudaArray::create<float4>(*cu, numParticles, "posqPersistent");
    }
    
    // Upload initial data ONCE
    std::vector<float4> posqFloat4(numParticles);
    for (int i = 0; i < numParticles; i++) {
        posqFloat4[i].x = static_cast<float>(positions[i][0]);
        posqFloat4[i].y = static_cast<float>(positions[i][1]);
        posqFloat4[i].z = static_cast<float>(positions[i][2]);
        posqFloat4[i].w = static_cast<float>(allParticleCharges[i]);
    }
    posqDevicePersistent->upload(posqFloat4);
    
    // Conversion constants
    const float conversionNmBohr = 18.8973f;
    const float conversionKjmolNmAu = conversionNmBohr / 2625.5f;
    const float conversionEvKjmol = 96.487f;
    float cathodeVoltageKj = std::fabs(parameters.cathodeVoltage) * conversionEvKjmol;
    float anodeVoltageKj = std::fabs(parameters.anodeVoltage) * conversionEvKjmol;
    float cathodeArea = sheetArea / static_cast<float>(parameters.cathodeIndices.size());
    float anodeArea = sheetArea / static_cast<float>(parameters.anodeIndices.size());
    
    int threadsPerBlock = 256;
    int numBlocksParticles = (numParticles + threadsPerBlock - 1) / threadsPerBlock;
    int numBlocksElectrodes = (parameters.cathodeIndices.size() + parameters.anodeIndices.size() + threadsPerBlock - 1) / threadsPerBlock;
    
    // Ensure scale buffers exist (persistent)
    if (cathodeScaleDevice == nullptr)
        cathodeScaleDevice = CudaArray::create<float>(*cu, 1, "cathodeScale");
    if (anodeScaleDevice == nullptr)
        anodeScaleDevice = CudaArray::create<float>(*cu, 1, "anodeScale");
    
    // ========================================================================
    // SINGLE ITERATION (ForceImpl 外層控制迭代次數，這裡只做一次)
    // ========================================================================
    // Upload current forces from NonbondedForce (由 ForceImpl 提供)
    std::vector<float3> forcesFloat3(numParticles);
    for (int i = 0; i < numParticles; i++) {
        forcesFloat3[i].x = static_cast<float>(forces[i][0]);
        forcesFloat3[i].y = static_cast<float>(forces[i][1]);
        forcesFloat3[i].z = static_cast<float>(forces[i][2]);
    }
    forcesDevicePersistent->upload(forcesFloat3);
    
    // Update electrode charges (ON GPU)
    updateElectrodeChargesIterative<<<numBlocksElectrodes, threadsPerBlock>>>(
        (const float3*)forcesDevicePersistent->getDevicePointer(),
        (float4*)posqDevicePersistent->getDevicePointer(),
        (const int*)cathodeIndicesDevice->getDevicePointer(),
        parameters.cathodeIndices.size(),
        (const int*)anodeIndicesDevice->getDevicePointer(),
        parameters.anodeIndices.size(),
        cathodeArea,
        anodeArea,
        cathodeVoltageKj,
        anodeVoltageKj,
        static_cast<float>(parameters.lGap),
        conversionKjmolNmAu,
        static_cast<float>(parameters.smallThreshold)
    );
    
    // Compute target and scaling (ON GPU)
    computeTargetAndScale<<<numBlocksParticles, threadsPerBlock>>>(
        (const float4*)posqDevicePersistent->getDevicePointer(),
        (float4*)posqDevicePersistent->getDevicePointer(),
        (const int*)cathodeIndicesDevice->getDevicePointer(),
        parameters.cathodeIndices.size(),
        (const int*)anodeIndicesDevice->getDevicePointer(),
        parameters.anodeIndices.size(),
        (const int*)electrodeMaskDevice->getDevicePointer(),
        (const int*)conductorMaskDevice->getDevicePointer(),  // 新增
        numParticles,
        static_cast<float>(cathodeZ),
        static_cast<float>(anodeZ),
        static_cast<float>(sheetArea),
        cathodeVoltageKj,
        anodeVoltageKj,
        static_cast<float>(parameters.lGap),
        static_cast<float>(parameters.lCell),
        conversionKjmolNmAu,
        static_cast<float>(parameters.smallThreshold),
        (float*)cathodeScaleDevice->getDevicePointer(),
        (float*)anodeScaleDevice->getDevicePointer()
    );
    
    // Conductor two-stage method (if conductors exist)
    int numConductorAtoms = conductorIndicesDevice->getSize();
    int numConductors = conductorContactIndicesDevice->getSize();
    if (numConductors > 0) {
        int numBlocksCondAtoms = (numConductorAtoms + threadsPerBlock - 1) / threadsPerBlock;
        int numBlocksConductors = (numConductors + threadsPerBlock - 1) / threadsPerBlock;
        
        // Step 1: Image charges (normal field projection)
        conductorImageCharges<<<numBlocksCondAtoms, threadsPerBlock>>>(
            (const float3*)forcesDevicePersistent->getDevicePointer(),
            (float4*)posqDevicePersistent->getDevicePointer(),
            (const int*)conductorIndicesDevice->getDevicePointer(),
            numConductorAtoms,
            (const float3*)conductorNormalsDevice->getDevicePointer(),
            (const float*)conductorAreasDevice->getDevicePointer(),
            conversionKjmolNmAu,
            static_cast<float>(parameters.smallThreshold)
        );
        
        // Step 2: Charge transfer (uniform distribution)
        conductorChargeTransfer<<<numBlocksConductors, threadsPerBlock>>>(
            (const float3*)forcesDevicePersistent->getDevicePointer(),
            (const int*)conductorContactIndicesDevice->getDevicePointer(),
            (const float3*)conductorContactNormalsDevice->getDevicePointer(),
            (const float*)conductorGeometriesDevice->getDevicePointer(),
            (const int*)conductorTypesDevice->getDevicePointer(),
            (float*)conductorDQDevice->getDevicePointer(),
            numConductors,
            cathodeVoltageKj,
            static_cast<float>(parameters.lGap),
            conversionKjmolNmAu,
            static_cast<float>(parameters.smallThreshold)
        );
        applyConductorDQ<<<numBlocksCondAtoms, threadsPerBlock>>>(
            (float4*)posqDevicePersistent->getDevicePointer(),
            (const int*)conductorIndicesDevice->getDevicePointer(),
            (const int*)conductorAtomCondIdsDevice->getDevicePointer(),
            numConductorAtoms,
            (const float*)conductorDQDevice->getDevicePointer(),
            (const int*)conductorAtomCountsDevice->getDevicePointer()
        );
    }
    
    // Scaling already applied inside computeTargetAndScale; no extra kernel needed
    
    // Download final results ONCE
    posqDevicePersistent->download(posqFloat4);
    
    cathodeCharges.resize(parameters.cathodeIndices.size());
    anodeCharges.resize(parameters.anodeIndices.size());
    
    for (size_t i = 0; i < parameters.cathodeIndices.size(); i++) {
        int idx = parameters.cathodeIndices[i];
        cathodeCharges[i] = static_cast<double>(posqFloat4[idx].w);
    }
    
    for (size_t i = 0; i < parameters.anodeIndices.size(); i++) {
        int idx = parameters.anodeIndices[i];
        anodeCharges[i] = static_cast<double>(posqFloat4[idx].w);
    }
    
    // Set dummy targets (not used in this version)
    cathodeTarget = 0.0;
    anodeTarget = 0.0;
    
    return 0.0;
}

// ============================================================================
// Other member functions (unchanged)
// ============================================================================

CudaCalcElectrodeChargeKernel::CudaCalcElectrodeChargeKernel(
    const std::string& name, 
    const Platform& platform,
    CudaContext& cu
) : CalcElectrodeChargeKernel(name, platform),
    cu(&cu),
    cathodeChargesDevice(nullptr),
    anodeChargesDevice(nullptr),
    cathodeIndicesDevice(nullptr),
    anodeIndicesDevice(nullptr),
    electrodeMaskDevice(nullptr),
    conductorMaskDevice(nullptr),
    cathodeTargetDevice(nullptr),
    anodeTargetDevice(nullptr),
    chargeSum(nullptr),
    forcesDevicePersistent(nullptr),
    posqDevicePersistent(nullptr),
    // Conductor arrays (新增)
    conductorIndicesDevice(nullptr),
    conductorNormalsDevice(nullptr),
    conductorAreasDevice(nullptr),
    conductorContactIndicesDevice(nullptr),
    conductorContactNormalsDevice(nullptr),
    conductorGeometriesDevice(nullptr),
    conductorTypesDevice(nullptr)
{
}

void CudaCalcElectrodeChargeKernel::initialize(
    const System& system, 
    const ElectrodeChargeForce& force
) {
    parameters.cathodeIndices = force.getCathode().atomIndices;
    parameters.anodeIndices = force.getAnode().atomIndices;
    parameters.cathodeVoltage = force.getCathode().voltage;
    parameters.anodeVoltage = force.getAnode().voltage;
    parameters.numIterations = force.getNumIterations();
    parameters.smallThreshold = force.getSmallThreshold();
    parameters.lGap = force.getCellGap();
    parameters.lCell = force.getCellLength();
    
    numParticles = system.getNumParticles();
    
    ContextSelector selector(*cu);
    initializeDeviceMemory();
}

void CudaCalcElectrodeChargeKernel::initializeDeviceMemory() {
    cathodeChargesDevice = CudaArray::create<float>(*cu, parameters.cathodeIndices.size(), "cathodeCharges");
    anodeChargesDevice = CudaArray::create<float>(*cu, parameters.anodeIndices.size(), "anodeCharges");
    
    cathodeIndicesDevice = CudaArray::create<int>(*cu, parameters.cathodeIndices.size(), "cathodeIndices");
    cathodeIndicesDevice->upload(parameters.cathodeIndices);
    
    anodeIndicesDevice = CudaArray::create<int>(*cu, parameters.anodeIndices.size(), "anodeIndices");
    anodeIndicesDevice->upload(parameters.anodeIndices);
    
    std::vector<int> electrodeMask(numParticles, 0);
    for (int idx : parameters.cathodeIndices)
        electrodeMask[idx] = 1;
    for (int idx : parameters.anodeIndices)
        electrodeMask[idx] = 1;
    electrodeMaskDevice = CudaArray::create<int>(*cu, numParticles, "electrodeMask");
    electrodeMaskDevice->upload(electrodeMask);
    
    // 初始化 conductorMask (暫時全為 0，需要從 Python 端傳入導體索引)
    std::vector<int> conductorMask(numParticles, 0);
    conductorMaskDevice = CudaArray::create<int>(*cu, numParticles, "conductorMask");
    conductorMaskDevice->upload(conductorMask);
    
    // 初始化導體相關陣列 (暫時為空，需要從 Python 端傳入)
    conductorIndicesDevice = CudaArray::create<int>(*cu, 0, "conductorIndices");
    conductorNormalsDevice = CudaArray::create<float3>(*cu, 0, "conductorNormals");
    conductorAreasDevice = CudaArray::create<float>(*cu, 0, "conductorAreas");
    conductorContactIndicesDevice = CudaArray::create<int>(*cu, 0, "conductorContactIndices");
    conductorContactNormalsDevice = CudaArray::create<float3>(*cu, 0, "conductorContactNormals");
    conductorGeometriesDevice = CudaArray::create<float>(*cu, 0, "conductorGeometries");
    conductorTypesDevice = CudaArray::create<int>(*cu, 0, "conductorTypes");
    conductorAtomCondIdsDevice = CudaArray::create<int>(*cu, 0, "conductorAtomCondIds");
    conductorAtomCountsDevice = CudaArray::create<int>(*cu, 0, "conductorAtomCounts");
    conductorDQDevice = CudaArray::create<float>(*cu, 0, "conductorDQ");
    
    cathodeTargetDevice = CudaArray::create<float>(*cu, 1, "cathodeTarget");
    anodeTargetDevice = CudaArray::create<float>(*cu, 1, "anodeTarget");
    chargeSum = CudaArray::create<float>(*cu, 1, "chargeSum");
}

void CudaCalcElectrodeChargeKernel::copyParametersToContext(
    ContextImpl& context, 
    const ElectrodeChargeForce& force
) {
    // Upload conductor metadata if provided
    const std::vector<int>& cIdx = force.getConductorIndices();
    const std::vector<double>& cNormals = force.getConductorNormals();
    const std::vector<double>& cAreas = force.getConductorAreas();
    const std::vector<int>& cContactIdx = force.getConductorContactIndices();
    const std::vector<double>& cContactNormals = force.getConductorContactNormals();
    const std::vector<double>& cGeom = force.getConductorGeometries();
    const std::vector<int>& cTypes = force.getConductorTypes();
    const std::vector<int>& cAtomCondIds = force.getConductorAtomCondIds();
    const std::vector<int>& cAtomCounts = force.getConductorAtomCounts();

    if (cIdx.size() > 0) {
        // conductor mask (per-particle)
        std::vector<int> conductorMask(numParticles, 0);
        for (int idx : cIdx) if (idx >= 0 && idx < numParticles) conductorMask[idx] = 1;
        if (conductorMaskDevice == nullptr || (int)conductorMaskDevice->getSize() != numParticles)
            conductorMaskDevice = CudaArray::create<int>(*cu, numParticles, "conductorMask");
        conductorMaskDevice->upload(conductorMask);

        // indices
        if (conductorIndicesDevice == nullptr || (int)conductorIndicesDevice->getSize() != (int)cIdx.size())
            conductorIndicesDevice = CudaArray::create<int>(*cu, (int)cIdx.size(), "conductorIndices");
        conductorIndicesDevice->upload(cIdx);

        // normals per atom (packed float3)
        int nAtoms = (int)cIdx.size();
        std::vector<float3> normals3(nAtoms);
        for (int i = 0; i < nAtoms; i++) {
            normals3[i].x = (float)cNormals[3*i+0];
            normals3[i].y = (float)cNormals[3*i+1];
            normals3[i].z = (float)cNormals[3*i+2];
        }
        if (conductorNormalsDevice == nullptr || (int)conductorNormalsDevice->getSize() != nAtoms)
            conductorNormalsDevice = CudaArray::create<float3>(*cu, nAtoms, "conductorNormals");
        conductorNormalsDevice->upload(normals3);

        // areas per atom
        std::vector<float> areasF(cAreas.begin(), cAreas.end());
        if (conductorAreasDevice == nullptr || (int)conductorAreasDevice->getSize() != nAtoms)
            conductorAreasDevice = CudaArray::create<float>(*cu, nAtoms, "conductorAreas");
        conductorAreasDevice->upload(areasF);

        // per-conductor arrays
        int nCond = (int)cContactIdx.size();
        if (conductorContactIndicesDevice == nullptr || (int)conductorContactIndicesDevice->getSize() != nCond)
            conductorContactIndicesDevice = CudaArray::create<int>(*cu, nCond, "conductorContactIndices");
        conductorContactIndicesDevice->upload(cContactIdx);

        std::vector<float3> cNormals3(nCond);
        for (int i = 0; i < nCond; i++) {
            cNormals3[i].x = (float)cContactNormals[3*i+0];
            cNormals3[i].y = (float)cContactNormals[3*i+1];
            cNormals3[i].z = (float)cContactNormals[3*i+2];
        }
        if (conductorContactNormalsDevice == nullptr || (int)conductorContactNormalsDevice->getSize() != nCond)
            conductorContactNormalsDevice = CudaArray::create<float3>(*cu, nCond, "conductorContactNormals");
        conductorContactNormalsDevice->upload(cNormals3);

        std::vector<float> geomF(cGeom.begin(), cGeom.end());
        if (conductorGeometriesDevice == nullptr || (int)conductorGeometriesDevice->getSize() != nCond)
            conductorGeometriesDevice = CudaArray::create<float>(*cu, nCond, "conductorGeometries");
        conductorGeometriesDevice->upload(geomF);

        if (conductorTypesDevice == nullptr || (int)conductorTypesDevice->getSize() != nCond)
            conductorTypesDevice = CudaArray::create<int>(*cu, nCond, "conductorTypes");
        conductorTypesDevice->upload(cTypes);

        // atom -> conductor id
        if (conductorAtomCondIdsDevice == nullptr || (int)conductorAtomCondIdsDevice->getSize() != (int)cAtomCondIds.size())
            conductorAtomCondIdsDevice = CudaArray::create<int>(*cu, (int)cAtomCondIds.size(), "conductorAtomCondIds");
        conductorAtomCondIdsDevice->upload(cAtomCondIds);

        // per conductor counts
        if (conductorAtomCountsDevice == nullptr || (int)conductorAtomCountsDevice->getSize() != nCond)
            conductorAtomCountsDevice = CudaArray::create<int>(*cu, nCond, "conductorAtomCounts");
        conductorAtomCountsDevice->upload(cAtomCounts);

        // dQ per conductor buffer
        if (conductorDQDevice == nullptr || (int)conductorDQDevice->getSize() != nCond)
            conductorDQDevice = CudaArray::create<float>(*cu, nCond, "conductorDQ");
    }
}
