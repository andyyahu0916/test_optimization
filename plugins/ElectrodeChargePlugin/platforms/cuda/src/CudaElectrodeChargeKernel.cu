/**
 * 🔥 Corrected and Refactored CUDA Kernel for ElectrodeChargeForce (Linus Version)
 *
 * This version is algorithmically pure and uses correct, modern OpenMM/CUDA APIs.
 * - All constants are passed as kernel arguments.
 * - `cudaMemset` is used to correctly clear buffers.
 * - All C++ compilation errors are fixed.
 */

#include "CudaElectrodeChargeKernel.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/common/ContextSelector.h"
#include "openmm/internal/ContextImpl.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

// ============================================================================
// WARP SHUFFLE REDUCTION
// ============================================================================
__device__ inline double warpReduceSum(double val) {
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;
}

// ============================================================================
// CUDA KERNELS (DE-FUSED for Algorithmic Purity)
// ============================================================================

__global__ void updateFlatElectrodeCharges(
    const double4* __restrict__ forces,
    double4* __restrict__ posq,
    const int* __restrict__ cathodeIndices,
    const int numCathode,
    const int* __restrict__ anodeIndices,
    const int numAnode,
    const double cathodeArea,
    const double anodeArea,
    const double cathodeVoltageKj,
    const double anodeVoltageKj,
    const double lGap,
    const double smallThreshold,
    const double PI,
    const double CONV_KJMOL_NM_AU) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const double twoOverFourPi = 2.0 / (4.0 * PI);

    if (i < numCathode) {
        const int atomIdx = cathodeIndices[i];
        const double oldCharge = posq[atomIdx].w;
        double ezExternal = 0.0;
        if (fabs(oldCharge) > 0.9 * smallThreshold) {
            ezExternal = forces[atomIdx].z / oldCharge;
        }
        double newCharge = twoOverFourPi * cathodeArea * ((cathodeVoltageKj / lGap) + ezExternal) * CONV_KJMOL_NM_AU;
        posq[atomIdx].w = copysign(fmax(fabs(newCharge), smallThreshold), 1.0);
    }

    if (i < numAnode) {
        const int atomIdx = anodeIndices[i];
        const double oldCharge = posq[atomIdx].w;
        double ezExternal = 0.0;
        if (fabs(oldCharge) > 0.9 * smallThreshold) {
            ezExternal = forces[atomIdx].z / oldCharge;
        }
        double newCharge = -twoOverFourPi * anodeArea * ((anodeVoltageKj / lGap) + ezExternal) * CONV_KJMOL_NM_AU;
        posq[atomIdx].w = -fmax(fabs(newCharge), smallThreshold);
    }
}

__global__ void conductorImageCharges(
    const double4* __restrict__ forces,
    double4* __restrict__ posq,
    const int* __restrict__ conductorAtomIndices,
    const int numConductorAtoms,
    const double3* __restrict__ conductorNormals,
    const double* __restrict__ conductorAreas,
    const double smallThreshold,
    const double PI,
    const double CONV_KJMOL_NM_AU) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numConductorAtoms) return;

    const int atomIdx = conductorAtomIndices[i];
    const double oldCharge = posq[atomIdx].w;

    if (fabs(oldCharge) > 0.9 * smallThreshold) {
        const double3 force = ((const double3*)forces)[atomIdx];
        const double3 normal = conductorNormals[i];
        const double En_external = (force.x * normal.x + force.y * normal.y + force.z * normal.z) / oldCharge;
        const double newCharge = (2.0 / (4.0 * PI)) * conductorAreas[i] * En_external * CONV_KJMOL_NM_AU;
        // Physical necessity: |q| >= threshold to prevent E=F/q blow-up in next iteration
        // Preserve sign: if physics correct, q>0 naturally; if q<0, reveals a bug
        posq[atomIdx].w = (fabs(newCharge) < smallThreshold) ?
                          copysign(smallThreshold, newCharge) : newCharge;
    } else {
        posq[atomIdx].w = smallThreshold;
    }
}

__global__ void conductorChargeTransfer(
    const double4* __restrict__ forces,
    const double4* __restrict__ posq,
    const int* __restrict__ conductorContactIndices,
    const double3* __restrict__ conductorContactNormals,
    const double* __restrict__ conductorGeometries,
    double* __restrict__ dQPerConductor,
    const int numConductors,
    const double cathodeVoltageKj,
    const double lGap,
    const double smallThreshold,
    const double CONV_KJMOL_NM_AU) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numConductors) return;

    const int contactIdx = conductorContactIndices[i];
    const double q_i = posq[contactIdx].w;
    double En_external = 0.0;

    if (fabs(q_i) > 0.9 * smallThreshold) {
        const double3 force = ((const double3*)forces)[contactIdx];
        const double3 normal = conductorContactNormals[i];
        En_external = (force.x * normal.x + force.y * normal.y + force.z * normal.z) / q_i;
    }

    const double dE_conductor = -(En_external + (cathodeVoltageKj / lGap / 2.0)) * CONV_KJMOL_NM_AU;

    // Good taste: geometry factor already encodes conductor type (Buckyball: dr^2, Nanotube: dr*L/2)
    // No need for type-based branching - the formula is universal: dQ = -dE * geometry
    const double geom = conductorGeometries[i];
    const double dQ = -1.0 * dE_conductor * geom;
    dQPerConductor[i] = dQ;
}

__global__ void applyConductorDQ(
    double4* __restrict__ posq,
    const int* __restrict__ conductorAtomIndices,
    const int* __restrict__ conductorAtomCondIds,
    const int numConductorAtoms,
    const double* __restrict__ dQPerConductor,
    const int* __restrict__ atomCountsPerConductor) {

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numConductorAtoms) return;

    const int atomIdx = conductorAtomIndices[i];
    const int condId = conductorAtomCondIds[i];
    const int count = atomCountsPerConductor[condId];
    if (count > 0) {
        const double dq_atom = dQPerConductor[condId] / (double)count;
        posq[atomIdx].w += dq_atom;
    }
}

__global__ void computeTargetAndNumericSums(
    const double4* __restrict__ posq,
    const int numParticles,
    const int* __restrict__ electrodeMask,
    const int* __restrict__ conductorMask,
    const double anodeZ,
    const double lCell,
    const double sheetArea,
    const double voltageKj,
    const double lGap,
    double* __restrict__ chargeSumBuffers,
    const double PI,
    const double CONV_KJMOL_NM_AU) {

    double localAnodeTarget = 0.0;
    double localCathodeTarget = 0.0;
    double localAnodeNumeric = 0.0;
    double localCathodeNumeric = 0.0;
    const int lane = threadIdx.x & 31;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numParticles; i += gridDim.x * blockDim.x) {
        const double4 p = posq[i];
        if (electrodeMask[i] == 0) {
            const double zDist = fabs(p.z - anodeZ);
            const double imgCharge = (zDist / lCell) * (-p.w);
            localAnodeTarget += imgCharge;
            localCathodeTarget += imgCharge;
        }
        if (electrodeMask[i] == 2) {
            localAnodeNumeric += p.w;
        } else if (electrodeMask[i] == 1 || conductorMask[i] == 1) {
            localCathodeNumeric += p.w;
        }
    }

    localAnodeTarget = warpReduceSum(localAnodeTarget);
    localCathodeTarget = warpReduceSum(localCathodeTarget);
    localAnodeNumeric = warpReduceSum(localAnodeNumeric);
    localCathodeNumeric = warpReduceSum(localCathodeNumeric);

    __shared__ double sharedSums[4*32];
    const int warp = threadIdx.x >> 5;
    if (lane == 0) {
        sharedSums[4*warp + 0] = localAnodeTarget;
        sharedSums[4*warp + 1] = localCathodeTarget;
        sharedSums[4*warp + 2] = localAnodeNumeric;
        sharedSums[4*warp + 3] = localCathodeNumeric;
    }
    __syncthreads();

    if (warp == 0) {
        localAnodeTarget = (lane < (blockDim.x >> 5)) ? sharedSums[4*lane + 0] : 0.0;
        localCathodeTarget = (lane < (blockDim.x >> 5)) ? sharedSums[4*lane + 1] : 0.0;
        localAnodeNumeric = (lane < (blockDim.x >> 5)) ? sharedSums[4*lane + 2] : 0.0;
        localCathodeNumeric = (lane < (blockDim.x >> 5)) ? sharedSums[4*lane + 3] : 0.0;

        localAnodeTarget = warpReduceSum(localAnodeTarget);
        localCathodeTarget = warpReduceSum(localCathodeTarget);
        localAnodeNumeric = warpReduceSum(localAnodeNumeric);
        localCathodeNumeric = warpReduceSum(localCathodeNumeric);

        if (lane == 0) {
            const double geomTerm = (1.0 / (4.0 * PI)) * sheetArea * ((voltageKj / lGap) + (voltageKj / lCell)) * CONV_KJMOL_NM_AU;
            atomicAdd(&chargeSumBuffers[0], localAnodeTarget - geomTerm);
            atomicAdd(&chargeSumBuffers[1], localCathodeTarget + geomTerm);
            atomicAdd(&chargeSumBuffers[2], localAnodeNumeric);
            atomicAdd(&chargeSumBuffers[3], localCathodeNumeric);
        }
    }
}

__global__ void applyScaling(
    double4* __restrict__ posq,
    const int numParticles,
    const int* __restrict__ electrodeMask,
    const int* __restrict__ conductorMask,
    const double* __restrict__ chargeSumBuffers,
    const double smallThreshold) {

    __shared__ double scales[2];
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        const double anodeTarget = chargeSumBuffers[0];
        const double cathodeSideTarget = -anodeTarget;
        const double anodeNumericSum = chargeSumBuffers[2];
        const double cathodeAndConductorNumericSum = chargeSumBuffers[3];

        double anodeScale = (fabs(anodeNumericSum) > smallThreshold) ? (anodeTarget / anodeNumericSum) : 1.0;
        double cathodeAndConductorScale = (fabs(cathodeAndConductorNumericSum) > smallThreshold) ? (cathodeSideTarget / cathodeAndConductorNumericSum) : 1.0;

        scales[0] = (anodeScale > 0.0) ? anodeScale : 1.0;
        scales[1] = (cathodeAndConductorScale > 0.0) ? cathodeAndConductorScale : 1.0;
    }
    __syncthreads();

    const double anodeScale = scales[0];
    const double cathodeAndConductorScale = scales[1];

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numParticles; i += gridDim.x * blockDim.x) {
        if (electrodeMask[i] == 2) { // Anode
            posq[i].w *= anodeScale;
        } else if (electrodeMask[i] == 1 || conductorMask[i] == 1) { // Cathode or Conductor
            posq[i].w *= cathodeAndConductorScale;
        }
    }
}


CudaCalcElectrodeChargeKernel::CudaCalcElectrodeChargeKernel(
    const std::string& name,
    const Platform& platform,
    CudaContext& cu
) : CalcElectrodeChargeKernel(name, platform), cu(&cu) {
}

CudaCalcElectrodeChargeKernel::~CudaCalcElectrodeChargeKernel() {
    cleanup();
}

void CudaCalcElectrodeChargeKernel::cleanup() {
    delete cathodeIndicesDevice;
    delete anodeIndicesDevice;
    delete electrodeMaskDevice;
    delete conductorMaskDevice;
    delete conductorIndicesDevice;
    delete conductorNormalsDevice;
    delete conductorAreasDevice;
    delete conductorContactIndicesDevice;
    delete conductorContactNormalsDevice;
    delete conductorGeometriesDevice;
    delete conductorAtomCondIdsDevice;
    delete conductorAtomCountsDevice;
    delete conductorDQDevice;
    delete chargeSumBuffers;
}

void CudaCalcElectrodeChargeKernel::initialize(
    const System& system,
    const ElectrodeChargeForce& force) {
    parameters = force.getParameters();
    numParticles = system.getNumParticles();

    ContextSelector selector(*cu);
    initializeDeviceMemory();
}

double CudaCalcElectrodeChargeKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    ContextSelector selector(*cu);

    if (inInternalEvaluation)
        return 0.0;

    CudaArray& posqArray = cu->getPosq();
    CudaArray& forceArray = cu->getForce();

    const double CONV_EV_KJOL_val = 96.487;
    const double CONV_KJMOL_NM_AU_val = 18.8973 / 2625.5;
    const double PI_val = 3.14159265358979323846;

    const double cathodeVoltageKj = std::fabs(parameters.cathodeVoltage) * CONV_EV_KJOL_val;
    const double anodeVoltageKj = std::fabs(parameters.anodeVoltage) * CONV_EV_KJOL_val;
    const double cathodeAreaAtom = parameters.cathodeIndices.empty() ? 0.0 : parameters.sheetArea / static_cast<double>(parameters.cathodeIndices.size());
    const double anodeAreaAtom = parameters.anodeIndices.empty() ? 0.0 : parameters.sheetArea / static_cast<double>(parameters.anodeIndices.size());
    const int numElectrodes = parameters.cathodeIndices.size() + parameters.anodeIndices.size();
    const int electrodeBlocks = (numElectrodes + 255) / 256;
    const int numConductorAtoms = conductorIndicesDevice ? conductorIndicesDevice->getSize() : 0;
    const int condAtomBlocks = (numConductorAtoms + 255) / 256;
    const int particleBlocks = (numParticles + 255) / 256;

    for (int iter = 0; iter < parameters.numIterations; ++iter) {
        // Step 1: Get forces from the context for the current iteration.
        try {
            inInternalEvaluation = true;
            context.calcForcesAndEnergy(true, false, 1<<0); // Recalculate NonbondedForce (group 0)
            inInternalEvaluation = false;
        } catch (...) {
            inInternalEvaluation = false;
            throw;
        }

        // Step 2: Update flat electrode charges based on current forces.
        if (electrodeBlocks > 0) {
            updateFlatElectrodeCharges<<<electrodeBlocks, 256, 0, cu->getCurrentStream()>>>(
                (const double4*)forceArray.getDevicePointer(),
                (double4*)posqArray.getDevicePointer(),
                (const int*)cathodeIndicesDevice->getDevicePointer(), parameters.cathodeIndices.size(),
                (const int*)anodeIndicesDevice->getDevicePointer(), parameters.anodeIndices.size(),
                cathodeAreaAtom, anodeAreaAtom, cathodeVoltageKj, anodeVoltageKj,
                parameters.lGap, parameters.smallThreshold, PI_val, CONV_KJMOL_NM_AU_val);
        }

        // Step 3: Update conductor charges (two-stage process).
        if (numConductorAtoms > 0) {
            // Step 3a: Image charges
            conductorImageCharges<<<condAtomBlocks, 256, 0, cu->getCurrentStream()>>>(
                (const double4*)forceArray.getDevicePointer(),
                (double4*)posqArray.getDevicePointer(),
                (const int*)conductorIndicesDevice->getDevicePointer(),
                numConductorAtoms,
                (const double3*)conductorNormalsDevice->getDevicePointer(),
                (const double*)conductorAreasDevice->getDevicePointer(),
                parameters.smallThreshold, PI_val, CONV_KJMOL_NM_AU_val);

            // Physical necessity: image charges changed the field.
            // Charge transfer MUST use the new field to satisfy constant-potential boundary condition.
            try {
                inInternalEvaluation = true;
                context.calcForcesAndEnergy(true, false, 1<<0);
                inInternalEvaluation = false;
            } catch (...) {
                inInternalEvaluation = false;
                throw;
            }
            
            // Step 3c: Charge transfer (using the NEW forces)
            const int numConductors = conductorContactIndicesDevice->getSize();
            const int condBlocks = (numConductors + 255) / 256;
            if (condBlocks > 0) {
                conductorChargeTransfer<<<condBlocks, 256, 0, cu->getCurrentStream()>>>(
                    (const double4*)forceArray.getDevicePointer(),
                    (const double4*)posqArray.getDevicePointer(),
                    (const int*)conductorContactIndicesDevice->getDevicePointer(),
                    (const double3*)conductorContactNormalsDevice->getDevicePointer(),
                    (const double*)conductorGeometriesDevice->getDevicePointer(),
                    (double*)conductorDQDevice->getDevicePointer(),
                    numConductors, cathodeVoltageKj, parameters.lGap, parameters.smallThreshold, CONV_KJMOL_NM_AU_val);

                applyConductorDQ<<<condAtomBlocks, 256, 0, cu->getCurrentStream()>>>(
                    (double4*)posqArray.getDevicePointer(),
                    (const int*)conductorIndicesDevice->getDevicePointer(),
                    (const int*)conductorAtomCondIdsDevice->getDevicePointer(),
                    numConductorAtoms,
                    (const double*)conductorDQDevice->getDevicePointer(),
                    (const int*)conductorAtomCountsDevice->getDevicePointer());
            }
        }

        // Step 4: Final scaling based on all updated charges.
        cudaMemsetAsync((void*)chargeSumBuffers->getDevicePointer(), 0, chargeSumBuffers->getSize()*sizeof(double), cu->getCurrentStream());

        computeTargetAndNumericSums<<<particleBlocks, 256, 0, cu->getCurrentStream()>>>(
            (const double4*)posqArray.getDevicePointer(),
            numParticles,
            (const int*)electrodeMaskDevice->getDevicePointer(),
            (const int*)conductorMaskDevice->getDevicePointer(),
            parameters.anodeZ, parameters.lCell, parameters.sheetArea,
            cathodeVoltageKj,
            parameters.lGap,
            (double*)chargeSumBuffers->getDevicePointer(),
            PI_val, CONV_KJMOL_NM_AU_val
        );

        applyScaling<<<particleBlocks, 256, 0, cu->getCurrentStream()>>>(
            (double4*)posqArray.getDevicePointer(),
            numParticles,
            (const int*)electrodeMaskDevice->getDevicePointer(),
            (const int*)conductorMaskDevice->getDevicePointer(),
            (const double*)chargeSumBuffers->getDevicePointer(),
            parameters.smallThreshold
        );
    }

    return 0.0;
}

void CudaCalcElectrodeChargeKernel::initializeDeviceMemory() {
    auto createOrResizeInt = [this](CudaArray*& array, int size, const std::string& name) {
        if (size == 0) { delete array; array = nullptr; return; }
        if (array == nullptr || array->getSize() != size) {
            delete array;
            array = CudaArray::create<int>(*cu, size, name);
        }
    };
    auto createOrResizeDouble = [this](CudaArray*& array, int size, const std::string& name) {
        if (size == 0) { delete array; array = nullptr; return; }
        if (array == nullptr || array->getSize() != size) {
            delete array;
            array = CudaArray::create<double>(*cu, size, name);
        }
    };
    auto createOrResizeDouble3 = [this](CudaArray*& array, int size, const std::string& name) {
        if (size == 0) { delete array; array = nullptr; return; }
        if (array == nullptr || array->getSize() != size) {
            delete array;
            array = CudaArray::create<double3>(*cu, size, name);
        }
    };

    createOrResizeInt(cathodeIndicesDevice, parameters.cathodeIndices.size(), "cathodeIndices");
    if(cathodeIndicesDevice) cathodeIndicesDevice->upload(parameters.cathodeIndices);
    createOrResizeInt(anodeIndicesDevice, parameters.anodeIndices.size(), "anodeIndices");
    if(anodeIndicesDevice) anodeIndicesDevice->upload(parameters.anodeIndices);

    std::vector<int> electrodeMask(numParticles, 0);
    for (int idx : parameters.cathodeIndices) electrodeMask[idx] = 1;
    for (int idx : parameters.anodeIndices) electrodeMask[idx] = 2;
    createOrResizeInt(electrodeMaskDevice, numParticles, "electrodeMask");
    electrodeMaskDevice->upload(electrodeMask);

    std::vector<int> conductorMask(numParticles, 0);
    for (int idx : parameters.conductorIndices) if (idx >= 0 && idx < numParticles) conductorMask[idx] = 1;
    createOrResizeInt(conductorMaskDevice, numParticles, "conductorMask");
    conductorMaskDevice->upload(conductorMask);

    createOrResizeInt(conductorIndicesDevice, parameters.conductorIndices.size(), "conductorIndices");
    if(conductorIndicesDevice) conductorIndicesDevice->upload(parameters.conductorIndices);

    createOrResizeDouble3(conductorNormalsDevice, parameters.conductorIndices.size(), "conductorNormals");
    if(conductorNormalsDevice) {
        std::vector<double3> normals3(parameters.conductorIndices.size());
        for(size_t i=0; i<parameters.conductorIndices.size(); ++i) {
            normals3[i] = make_double3(parameters.conductorNormals[3*i], parameters.conductorNormals[3*i+1], parameters.conductorNormals[3*i+2]);
        }
        conductorNormalsDevice->upload(normals3);
    }

    createOrResizeDouble(conductorAreasDevice, parameters.conductorAreas.size(), "conductorAreas");
    if(conductorAreasDevice) conductorAreasDevice->upload(parameters.conductorAreas);

    createOrResizeInt(conductorContactIndicesDevice, parameters.conductorContactIndices.size(), "conductorContactIndices");
    if(conductorContactIndicesDevice) conductorContactIndicesDevice->upload(parameters.conductorContactIndices);

    createOrResizeDouble3(conductorContactNormalsDevice, parameters.conductorContactIndices.size(), "conductorContactNormals");
    if(conductorContactNormalsDevice) {
        std::vector<double> cNormals_flat = parameters.conductorContactNormals;
        std::vector<double3> cnormals3(parameters.conductorContactIndices.size());
        for(size_t i=0; i<parameters.conductorContactIndices.size(); ++i) {
            cnormals3[i] = make_double3(cNormals_flat[3*i], cNormals_flat[3*i+1], cNormals_flat[3*i+2]);
        }
        conductorContactNormalsDevice->upload(cnormals3);
    }
    
    createOrResizeDouble(conductorGeometriesDevice, parameters.conductorGeometries.size(), "conductorGeometries");
    if(conductorGeometriesDevice) conductorGeometriesDevice->upload(parameters.conductorGeometries);

    createOrResizeInt(conductorAtomCondIdsDevice, parameters.conductorAtomCondIds.size(), "conductorAtomCondIds");
    if(conductorAtomCondIdsDevice) conductorAtomCondIdsDevice->upload(parameters.conductorAtomCondIds);

    createOrResizeInt(conductorAtomCountsDevice, parameters.conductorAtomCounts.size(), "conductorAtomCounts");
    if(conductorAtomCountsDevice) conductorAtomCountsDevice->upload(parameters.conductorAtomCounts);

    createOrResizeDouble(conductorDQDevice, parameters.conductorAtomCounts.size(), "conductorDQ");

    createOrResizeDouble(chargeSumBuffers, 4, "chargeSumBuffers");
}

void CudaCalcElectrodeChargeKernel::copyParametersToContext(ContextImpl& context, const ElectrodeChargeForce& force) {
    initialize(context.getSystem(), force);
}