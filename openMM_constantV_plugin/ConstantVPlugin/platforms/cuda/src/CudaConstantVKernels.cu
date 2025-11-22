/**
 * CUDA Implementation of ConstantV Plugin
 *
 * 直接翻譯自 Reference 平台實現（ReferenceConstantVKernels.cpp）
 * 物理邏輯完全照抄教授的 Python 代碼
 *
 * 驗證狀態: Reference 實現已通過 ab initio 測試
 * - Green's Reciprocity: 誤差 < 1.5e-14
 * - 電荷守恆: Q_total = 0.000000e
 * - 符合物理第一性原則
 *
 * 此 CUDA 版本僅將算法並行化，不改變任何物理公式
 */

#include "CudaConstantVKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaNonbondedUtilities.h"
#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <algorithm>  // for std::sort
#include <utility>    // for std::pair

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

// ═══════════════════════════════════════════════════════════
// 物理常數（照抄 Reference）
// ═══════════════════════════════════════════════════════════

static const double CONVERSION_NMBOHR = 18.8973;
static const double CONVERSION_KJMOLNM_AU = CONVERSION_NMBOHR / 2625.5;
static const double CONVERSION_EV_KJMOL = 96.487;  // 照抄Python Line 38
static const double SMALL_THRESHOLD = 1e-6;  // 照抄Python Line 48 (不是1e-10！)

// ═══════════════════════════════════════════════════════════
// 🔥 CRITICAL: Force Group Assignment (prevents infinite recursion)
// ═══════════════════════════════════════════════════════════
// ConstantVForce should be assigned to this group (31) to prevent recursion:
static const int CONSTANTV_FORCE_GROUP = 31;

// ═══════════════════════════════════════════════════════════
// CUDA Kernels
// ═══════════════════════════════════════════════════════════

/**
 * Kernel: 初始化電極電荷
 */
__global__ void initializeChargesKernel(
    int numElectrodes,
    const int* __restrict__ electrodeIndices,
    const double* __restrict__ areaPerAtom,
    float4* __restrict__ posq,
    double voltage,
    double Lgap,
    double Lcell,
    double sign,  // +1.0 for cathode, -1.0 for anode
    bool flagSmall
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElectrodes) return;

    int atomIdx = electrodeIndices[i];
    double area = areaPerAtom[i];

    // Line 179-180: 完全照抄公式
    double q_i = sign / (4.0 * M_PI) * area *
                 (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOLNM_AU;

    // Line 182-184: 低電壓保護
    // P3 FIX: Low Voltage Protection (Additive, not replacement)
    if (flagSmall) {
        q_i = q_i + sign * SMALL_THRESHOLD;
    }

    // Line 186-187: 寫入電荷到 posq.w
    posq[atomIdx].w = (float)q_i;
}

/**
 * Kernel: 計算外部電場 Ez_external = F_z / q_old
 */
__global__ void computeEzExternalKernel(
    int numElectrodes,
    const int* __restrict__ electrodeIndices,
    const float4* __restrict__ forces,  // 已在GPU
    const float4* __restrict__ posq,    // 已在GPU
    double* __restrict__ Ez_external
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElectrodes) return;

    int atomIdx = electrodeIndices[i];
    double q_old = (double)posq[atomIdx].w;
    double F_z = (double)forces[atomIdx].z;

    if (fabs(q_old) > (0.9 * SMALL_THRESHOLD)) {
        Ez_external[i] = F_z / q_old;
    } else {
        Ez_external[i] = 0.0;
    }
}

/**
 * Kernel: 更新電極電荷（Maxwell 邊界條件）
 */
__global__ void updateElectrodeChargesKernel(
    int numElectrodes,
    const int* __restrict__ electrodeIndices,
    const double* __restrict__ areaPerAtom,
    const double* __restrict__ Ez_external,
    float4* __restrict__ posq,
    double voltage,
    double Lgap,
    double sign  // +2.0 for cathode, -2.0 for anode
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElectrodes) return;

    int atomIdx = electrodeIndices[i];
    double area = areaPerAtom[i];
    double Ez = Ez_external[i];

    double q_i = sign / (4.0 * M_PI) * area *
                 (voltage / Lgap + Ez) * CONVERSION_KJMOLNM_AU;

    if (fabs(q_i) < SMALL_THRESHOLD) {
        q_i = sign / 2.0 * SMALL_THRESHOLD;
    }

    posq[atomIdx].w = (float)q_i;
}

/**
 * OPTIMIZED Kernel: Fused computeEz + updateCharge
 */
__global__ void computeAndUpdateChargesFusedKernel(
    int numElectrodes,
    const int* __restrict__ electrodeIndices,
    const double* __restrict__ areaPerAtom,
    const float4* __restrict__ forces,
    float4* __restrict__ posq,
    double voltage,
    double Lgap,
    double sign  // +2.0 for cathode, -2.0 for anode
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElectrodes) return;

    int atomIdx = electrodeIndices[i];
    double area = areaPerAtom[i];

    double q_old = (double)posq[atomIdx].w;
    double F_z = (double)forces[atomIdx].z;

    double Ez_external = 0.0;
    if (fabs(q_old) > (0.9 * SMALL_THRESHOLD)) {
        Ez_external = F_z / q_old;
    }

    const double factor = sign / (4.0 * M_PI) * CONVERSION_KJMOLNM_AU;
    const double v_over_lgap = voltage / Lgap;

    double q_new = factor * area * (v_over_lgap + Ez_external);

    if (fabs(q_new) < SMALL_THRESHOLD) {
        q_new = sign / 2.0 * SMALL_THRESHOLD;
    }

    posq[atomIdx].w = (float)q_new;
}

/**
 * Kernel: Buckyball/Nanotube Numerical Charge
 * 翻譯自 Reference: numericalChargeConductor (Step 1)
 * Formula: q_i = 2.0 / (4pi) * area * E_ext * conversion
 */
__global__ void numericalChargeConductorKernel(
    int numAtoms,
    const int* __restrict__ indices,
    const double* __restrict__ normals, // Interleaved nx, ny, nz
    const float4* __restrict__ forces,
    float4* __restrict__ posq,
    double areaPerAtom
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    int atomIdx = indices[i];
    double q_old = (double)posq[atomIdx].w;

    double nx = normals[3*i + 0];
    double ny = normals[3*i + 1];
    double nz = normals[3*i + 2];

    double q_new;

    if (fabs(q_old) > (0.9 * SMALL_THRESHOLD)) {
        // E = F / q
        double Ex = (double)forces[atomIdx].x / q_old;
        double Ey = (double)forces[atomIdx].y / q_old;
        double Ez = (double)forces[atomIdx].z / q_old;

        // Project to normal
        double En_external = Ex * nx + Ey * ny + Ez * nz;

        // Solve
        // q = 2.0 / (4.0 * pi) * area * En * conversion
        q_new = 2.0 / (4.0 * M_PI) * areaPerAtom * En_external * CONVERSION_KJMOLNM_AU;
    } else {
        q_new = SMALL_THRESHOLD;
    }

    posq[atomIdx].w = (float)q_new;
}

/**
 * Kernel: Copy Contact Force (New for Step 2)
 * Extracts force of a single atom to a GPU buffer
 */
__global__ void extractContactForceKernel(
    int contactAtomIndex,
    const float4* __restrict__ forces,
    float4* __restrict__ d_contactForce
) {
    // Single thread kernel
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_contactForce[0] = forces[contactAtomIndex];
    }
}

/**
 * Kernel: Buckyball/Nanotube Charge Transfer (Step 2)
 * 翻譯自 Reference: numericalChargeConductor (Step 2)
 * Calculates dq based on contact force and updates charges.
 */
__global__ void computeAndApplyChargeTransferKernel(
    int numAtoms,
    const int* __restrict__ indices,
    const float4* __restrict__ d_contactForce, // Single value from GPU buffer
    float4* __restrict__ posq,
    double voltage,
    double Lgap,
    double dr_center_contact,
    double length, // For Nanotube (use 0.0 for Buckyball to switch logic? No, easier to pass pre-calc factor)
    double geometricFactor, // Buckyball: dr^2, Nanotube: dr*L/2
    int sign_electrode, // +1 for cathode contact, -1 for anode contact
    int contactAtomIndex // Replaced q_contact
) {
    // Part 1: Calculate dq (Thread 0 only)
    __shared__ double dq_atom;

    if (threadIdx.x == 0) {
        float4 f = d_contactForce[0];
        // Read q_contact from global memory
        double q_i = (double)posq[contactAtomIndex].w;

        double En_external = 0.0;
        if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
            double Ex = (double)f.x / q_i;
            double Ey = (double)f.y / q_i;
            double Ez = (double)f.z / q_i;

            // P2 FIX: Correct Anode Sign Logic
            // Cathode (sign=1) -> Normal +Z (1.0)
            // Anode (sign=-1)  -> Normal -Z (-1.0)
            double normal_z = (sign_electrode > 0) ? 1.0 : -1.0;
            En_external = Ez * normal_z;
        }

        double dE_conductor = -(En_external + voltage / Lgap / 2.0) * CONVERSION_KJMOLNM_AU;
        double dQ = -1.0 * dE_conductor * geometricFactor;
        dq_atom = dQ / numAtoms;
    }
    __syncthreads();

    // Part 2: Apply dq (All threads)
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    int atomIdx = indices[i];
    posq[atomIdx].w += (float)dq_atom;
}

/**
 * Kernel: 計算解析電荷的幾何貢獻
 */
__global__ void computeGeometricChargeKernel(
    double* __restrict__ Q_analytic,
    double voltage,
    double Lgap,
    double Lcell,
    double totalArea,
    double sign
) {
    // 單線程執行
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        *Q_analytic = sign / (4.0 * M_PI) * totalArea *
                      (voltage / Lgap + voltage / Lcell) *
                      CONVERSION_KJMOLNM_AU;
    }
}

/**
 * Kernel: 計算解析電荷的鏡像電荷貢獻（並行 reduction）
 */
__global__ void computeImageChargeKernel(
    int numElectrolytes,
    const int* __restrict__ electrolyteIndices,
    const float4* __restrict__ posq,
    double* __restrict__ Q_analytic_partial,
    double z_opposite,
    double Lcell
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    if (i < numElectrolytes) {
        int index = electrolyteIndices[i];
        double q_i = (double)posq[index].w;
        double z_atom = (double)posq[index].z;
        double z_distance = fabs(z_atom - z_opposite);

        local_sum = (z_distance / Lcell) * (-q_i);
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        Q_analytic_partial[blockIdx.x] = sdata[0];
    }
}

/**
 * Kernel: 合併部分和（最後一步）
 */
__global__ void reducePartialSumsKernel(
    int numBlocks,
    const double* __restrict__ partialSums,
    double* __restrict__ Q_analytic
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = threadIdx.x;

    double local_sum = 0.0;
    if (i < numBlocks) {
        local_sum = partialSums[i];
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s && (tid + s) < numBlocks) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(Q_analytic, sdata[0]);
    }
}

// Level 2 Optimizations
struct SumFunctor {
    __device__ double operator()(int i, const int* __restrict__ indices,
                                const float4* __restrict__ posq, double, double) const {
        int atomIdx = indices[i];
        return (double)posq[atomIdx].w;
    }
};

struct ImageChargeFunctor {
    __device__ double operator()(int i, const int* __restrict__ indices,
                                const float4* __restrict__ posq,
                                double z_opposite, double Lcell) const {
        int index = indices[i];
        double q_i = (double)posq[index].w;
        double z_atom = (double)posq[index].z;
        double z_distance = fabs(z_atom - z_opposite);
        return (z_distance / Lcell) * (-q_i);
    }
};

template <typename Loader>
__global__ void warpAssistedReductionKernel(
    int numItems,
    const int* __restrict__ indices,
    const float4* __restrict__ posq,
    double* __restrict__ partialSums,
    double arg1,
    double arg2
) {
    extern __shared__ double sdata[];
    double sum = 0.0;
    Loader loader;

    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numItems; i += gridDim.x * blockDim.x) {
        sum += loader(i, indices, posq, arg1, arg2);
    }

    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (threadIdx.x % 32 == 0)
        sdata[threadIdx.x / 32] = sum;

    __syncthreads();

    if (threadIdx.x < 32) {
        sum = (threadIdx.x < blockDim.x / 32) ? sdata[threadIdx.x] : 0.0;
        if (threadIdx.x < 16) {
            for (int offset = 16; offset > 0; offset /= 2)
                sum += __shfl_down_sync(0xffffffff, sum, offset);
        }
    }

    if (threadIdx.x == 0)
        partialSums[blockIdx.x] = sum;
}

__global__ void sumElectrodeChargesKernel(
    int numElectrodes,
    const int* __restrict__ electrodeIndices,
    const float4* __restrict__ posq,
    double* __restrict__ Q_numeric_partial
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double local_sum = 0.0;
    if (i < numElectrodes) {
        int atomIdx = electrodeIndices[i];
        local_sum = (double)posq[atomIdx].w;
    }

    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        Q_numeric_partial[blockIdx.x] = sdata[0];
    }
}

__global__ void scaleChargesKernel(
    int numElectrodes,
    const int* __restrict__ electrodeIndices,
    float4* __restrict__ posq,
    double scale_factor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElectrodes) return;

    int atomIdx = electrodeIndices[i];

    double q_old = (double)posq[atomIdx].w;
    double q_new = q_old * scale_factor;
    posq[atomIdx].w = (float)q_new;
}

__global__ void computeScaleAndNormalizeKernel(
    int numElectrodes,
    const int* __restrict__ electrodeIndices,
    float4* __restrict__ posq,
    const double* __restrict__ Q_analytic,  // [1]
    const double* __restrict__ Q_numeric    // [1]
) {
    __shared__ double scale_factor;
    __shared__ bool valid_scale;

    if (threadIdx.x == 0) {
        double Q_a = *Q_analytic;
        double Q_n = *Q_numeric;

        if (fabs(Q_n) > SMALL_THRESHOLD) {
            scale_factor = Q_a / Q_n;
            valid_scale = true;
        } else {
            valid_scale = false;
        }
    }
    __syncthreads();

    if (valid_scale) {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < numElectrodes) {
            int atomIdx = electrodeIndices[i];
            double q_old = (double)posq[atomIdx].w;
            posq[atomIdx].w = (float)(q_old * scale_factor);
        }
    }
}

// ═══════════════════════════════════════════════════════════
// CudaCalcConstantVKernel Implementation
// ═══════════════════════════════════════════════════════════

CudaCalcConstantVKernel::CudaCalcConstantVKernel(string name, const Platform& platform, CudaContext& cu) :
    CalcConstantVKernel(name, platform), cu(cu), gpuInitialized(false) {

    // Initialize pointers to nullptr
    d_cathodeIndices = nullptr;
    d_anodeIndices = nullptr;
    d_cathodeAreas = nullptr;
    d_anodeAreas = nullptr;
    d_electrolyteIndices = nullptr;
    d_Ez_cathode = nullptr;
    d_Ez_anode = nullptr;
    d_Q_analytic_cathode = nullptr;
    d_Q_analytic_anode = nullptr;
    d_Q_numeric_cathode = nullptr;
    d_Q_numeric_anode = nullptr;
    d_cathode_partial = nullptr;
    d_anode_partial = nullptr;
    d_cathode_numeric_partial = nullptr;
    d_anode_numeric_partial = nullptr;

    // Contact force buffer (1 float4)
    d_contactForceBuffer = nullptr;
}

CudaCalcConstantVKernel::~CudaCalcConstantVKernel() {
    delete d_cathodeIndices;
    delete d_anodeIndices;
    delete d_cathodeAreas;
    delete d_anodeAreas;
    delete d_electrolyteIndices;
    delete d_Ez_cathode;
    delete d_Ez_anode;
    delete d_Q_analytic_cathode;
    delete d_Q_analytic_anode;
    delete d_Q_numeric_cathode;
    delete d_Q_numeric_anode;
    delete d_cathode_partial;
    delete d_anode_partial;
    delete d_cathode_numeric_partial;
    delete d_anode_numeric_partial;

    delete d_contactForceBuffer;

    // Delete conductor data
    for (auto* c : buckyballs) delete c;
    for (auto* c : nanotubes) delete c;
}

void CudaCalcConstantVKernel::initialize(const System& system, const ConstantVForce& force) {
    std::cout << "[CUDA] initialize() called" << std::endl;

    // Parameters
    voltage = force.getVoltage() * CONVERSION_EV_KJMOL;
    Lgap = force.getLgap();
    Lcell = force.getLcell();
    totalArea = force.getTotalArea();
    z_cathode = force.getZCathode();
    z_anode = force.getZAnode();
    nIterations = force.getNumIterations();

    // Load Cathode/Anode/Electrolyte (same as before)
    numCathodes = force.getNumCathodeAtoms();
    cathodeIndices.resize(numCathodes);
    cathodeAreas.resize(numCathodes);
    for (int i = 0; i < numCathodes; i++) {
        int particle;
        double area;
        force.getCathodeAtomParameters(i, particle, area);
        cathodeIndices[i] = particle;
        cathodeAreas[i] = area;
    }

    numAnodes = force.getNumAnodeAtoms();
    anodeIndices.resize(numAnodes);
    anodeAreas.resize(numAnodes);
    for (int i = 0; i < numAnodes; i++) {
        int particle;
        double area;
        force.getAnodeAtomParameters(i, particle, area);
        anodeIndices[i] = particle;
        anodeAreas[i] = area;
    }

    // Sorting for coalescing
    vector<pair<int, double>> cathode_pairs;
    for (int i = 0; i < numCathodes; i++) cathode_pairs.push_back({cathodeIndices[i], cathodeAreas[i]});
    std::sort(cathode_pairs.begin(), cathode_pairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) { return a.first < b.first; });
    for (int i = 0; i < numCathodes; i++) { cathodeIndices[i] = cathode_pairs[i].first; cathodeAreas[i] = cathode_pairs[i].second; }

    vector<pair<int, double>> anode_pairs;
    for (int i = 0; i < numAnodes; i++) anode_pairs.push_back({anodeIndices[i], anodeAreas[i]});
    std::sort(anode_pairs.begin(), anode_pairs.end(), [](const pair<int, double>& a, const pair<int, double>& b) { return a.first < b.first; });
    for (int i = 0; i < numAnodes; i++) { anodeIndices[i] = anode_pairs[i].first; anodeAreas[i] = anode_pairs[i].second; }

    numElectrolytes = force.getNumElectrolyteAtoms();
    electrolyteIndices.resize(numElectrolytes);
    for (int i = 0; i < numElectrolytes; i++) {
        int particle;
        double charge;
        force.getElectrolyteAtomParameters(i, particle, charge);
        electrolyteIndices[i] = particle;
    }
    std::sort(electrolyteIndices.begin(), electrolyteIndices.end());

    // Load Buckyballs
    int numBucky = force.getNumBuckyballConductors();
    buckyballInitData.resize(numBucky);
    for(int i=0; i<numBucky; i++) {
        force.getBuckyballConductorParameters(i,
            buckyballInitData[i].virtualAtoms,
            buckyballInitData[i].realAtoms,
            buckyballInitData[i].electrodeType,
            buckyballInitData[i].voltage);
    }

    // Load Nanotubes
    int numNano = force.getNumNanotubeConductors();
    nanotubeInitData.resize(numNano);
    for(int i=0; i<numNano; i++) {
        force.getNanotubeConductorParameters(i,
            nanotubeInitData[i].virtualAtoms,
            nanotubeInitData[i].realAtoms,
            nanotubeInitData[i].electrodeType,
            nanotubeInitData[i].voltage,
            nanotubeInitData[i].axis);
    }

    // Get NonbondedForce
    nonbondedForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        const NonbondedForce* nbForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
        if (nbForce != nullptr) {
            nonbondedForce = const_cast<NonbondedForce*>(nbForce);
            break;
        }
    }
    if (nonbondedForce == nullptr)
        throw OpenMMException("ConstantVForce: NonbondedForce not found");
}

// ═══════════════════════════════════════════════════════════
// Geometry Helpers (CPU Logic)
// ═══════════════════════════════════════════════════════════

void CudaCalcConstantVKernel::projectOrthogonalToAxis(const double vec_in[3], const double axis[3], double vec_out[3]) {
    double dot_product = vec_in[0]*axis[0] + vec_in[1]*axis[1] + vec_in[2]*axis[2];
    vec_out[0] = vec_in[0] - axis[0] * dot_product;
    vec_out[1] = vec_in[1] - axis[1] * dot_product;
    vec_out[2] = vec_in[2] - axis[2] * dot_product;
}

void CudaCalcConstantVKernel::initializeBuckyballGeometry(CudaConductorData* conductor, const std::vector<OpenMM::Vec3>& positions) {
    // Note: "positions" are in nm, as OpenMM::Vec3 uses nm (standard unit in openmm internal)
    // But we double check context state in initializeGPU

    // This logic exactly matches ReferenceConstantVKernels.cpp
    int Natoms = conductor->d_virtualAtomIndices->getSize();
    std::vector<int> indices(Natoms);
    conductor->d_virtualAtomIndices->download(indices);

    conductor->r_center = {0.0, 0.0, 0.0};
    for(int idx : indices) {
        conductor->r_center[0] += positions[idx][0];
        conductor->r_center[1] += positions[idx][1];
        conductor->r_center[2] += positions[idx][2];
    }
    conductor->r_center[0] /= Natoms;
    conductor->r_center[1] /= Natoms;
    conductor->r_center[2] /= Natoms;

    if (Natoms > 0) {
        int first = indices[0];
        double rx = positions[first][0] - conductor->r_center[0];
        double ry = positions[first][1] - conductor->r_center[1];
        double rz = positions[first][2] - conductor->r_center[2];
        conductor->radius = sqrt(rx*rx + ry*ry + rz*rz);
    }

    conductor->area_atom = 4.0 * M_PI * conductor->radius * conductor->radius / Natoms;

    std::vector<double> normals(3 * Natoms);
    for(size_t i=0; i<indices.size(); i++) {
        int idx = indices[i];
        double nx = positions[idx][0] - conductor->r_center[0];
        double ny = positions[idx][1] - conductor->r_center[1];
        double nz = positions[idx][2] - conductor->r_center[2];
        double norm = sqrt(nx*nx + ny*ny + nz*nz);
        normals[3*i+0] = nx/norm;
        normals[3*i+1] = ny/norm;
        normals[3*i+2] = nz/norm;
    }
    conductor->d_normals->upload(normals);
}

void CudaCalcConstantVKernel::initializeNanotubeGeometry(CudaConductorData* conductor, const std::vector<OpenMM::Vec3>& positions, const OpenMM::Vec3 boxVectors[3]) {
    int Natoms = conductor->d_virtualAtomIndices->getSize();
    std::vector<int> indices(Natoms);
    conductor->d_virtualAtomIndices->download(indices);

    conductor->r_center = {0.0, 0.0, 0.0};
    for(int idx : indices) {
        conductor->r_center[0] += positions[idx][0];
        conductor->r_center[1] += positions[idx][1];
        conductor->r_center[2] += positions[idx][2];
    }
    conductor->r_center[0] /= Natoms;
    conductor->r_center[1] /= Natoms;
    conductor->r_center[2] /= Natoms;

    conductor->length = boxVectors[0][0]; // Assuming aligned with A vector (standard convention)

    conductor->radius = 0.0;
    std::vector<double> normals(3 * Natoms);
    double axis_arr[3] = {conductor->axis[0], conductor->axis[1], conductor->axis[2]};

    for(size_t i=0; i<indices.size(); i++) {
        int idx = indices[i];
        double dr[3];
        dr[0] = positions[idx][0] - conductor->r_center[0];
        dr[1] = positions[idx][1] - conductor->r_center[1];
        dr[2] = positions[idx][2] - conductor->r_center[2];

        double radial[3];
        projectOrthogonalToAxis(dr, axis_arr, radial);
        double r = sqrt(radial[0]*radial[0] + radial[1]*radial[1] + radial[2]*radial[2]);

        if(conductor->radius == 0.0) conductor->radius = r; // First approx

        // Store radial normal
        normals[3*i+0] = radial[0]/r;
        normals[3*i+1] = radial[1]/r;
        normals[3*i+2] = radial[2]/r;
    }

    conductor->area_atom = 2.0 * M_PI * conductor->radius * conductor->length / Natoms;
    conductor->d_normals->upload(normals);
}

void CudaCalcConstantVKernel::findContactNeighborConductor(CudaConductorData* conductor, const std::vector<OpenMM::Vec3>& positions) {
    const std::vector<int>* electrodeContact = (conductor->electrodeType == "cathode") ? &cathodeIndices : &anodeIndices;

    double min_dist = 10.0;
    conductor->contactAtomIndex = -1;

    for (int atomIdx : *electrodeContact) {
        double dx = conductor->r_center[0] - positions[atomIdx][0];
        double dy = conductor->r_center[1] - positions[atomIdx][1];
        double dz = conductor->r_center[2] - positions[atomIdx][2];
        double dr_atom = sqrt(dx*dx + dy*dy + dz*dz);

        if (dr_atom < min_dist) {
            conductor->contactAtomIndex = atomIdx;
            min_dist = dr_atom;
        }
    }

    if (min_dist < conductor->closeThreshold) {
        conductor->dr_center_contact = min_dist;
        conductor->closeToElectrode = true;
    } else {
        conductor->closeToElectrode = false;
        std::cerr << "[CUDA] Warning: Conductor not close to electrode. Physics may be inaccurate." << std::endl;
    }
}

void CudaCalcConstantVKernel::initializeGPU() {
    std::cout << "[CUDA] initializeGPU() called" << std::endl;
    cu.setAsCurrent();

    // --- Allocate & Upload Flat Electrode Data (Existing Logic) ---
    d_cathodeIndices = CudaArray::create<int>(cu, numCathodes, "cathodeIndices");
    d_anodeIndices = CudaArray::create<int>(cu, numAnodes, "anodeIndices");
    d_cathodeAreas = CudaArray::create<double>(cu, numCathodes, "cathodeAreas");
    d_anodeAreas = CudaArray::create<double>(cu, numAnodes, "anodeAreas");
    d_electrolyteIndices = CudaArray::create<int>(cu, numElectrolytes, "electrolyteIndices");

    d_Ez_cathode = CudaArray::create<double>(cu, numCathodes, "Ez_cathode");
    d_Ez_anode = CudaArray::create<double>(cu, numAnodes, "Ez_anode");

    d_Q_analytic_cathode = CudaArray::create<double>(cu, 1, "Q_analytic_cathode");
    d_Q_analytic_anode = CudaArray::create<double>(cu, 1, "Q_analytic_anode");
    d_Q_numeric_cathode = CudaArray::create<double>(cu, 1, "Q_numeric_cathode");
    d_Q_numeric_anode = CudaArray::create<double>(cu, 1, "Q_numeric_anode");

    int blockSize = 256;
    int numBlocks_cathode = (numElectrolytes + blockSize - 1) / blockSize;
    int numBlocks_anode = (numElectrolytes + blockSize - 1) / blockSize;
    d_cathode_partial = CudaArray::create<double>(cu, numBlocks_cathode, "cathode_partial");
    d_anode_partial = CudaArray::create<double>(cu, numBlocks_anode, "anode_partial");

    int numBlocks_cath_numeric = (numCathodes + blockSize - 1) / blockSize;
    int numBlocks_anode_numeric = (numAnodes + blockSize - 1) / blockSize;
    d_cathode_numeric_partial = CudaArray::create<double>(cu, numBlocks_cath_numeric, "cathode_numeric_partial");
    d_anode_numeric_partial = CudaArray::create<double>(cu, numBlocks_anode_numeric, "anode_numeric_partial");

    d_cathodeIndices->upload(cathodeIndices);
    d_anodeIndices->upload(anodeIndices);
    d_cathodeAreas->upload(cathodeAreas);
    d_anodeAreas->upload(anodeAreas);
    d_electrolyteIndices->upload(electrolyteIndices);

    // Allocate contact force buffer
    // WARNING: This buffer size is 1.
    // All kernels writing to this buffer MUST be serialized in the same CUDA stream.
    // Do not use concurrent streams for different conductors without expanding this buffer.
    d_contactForceBuffer = CudaArray::create<float4>(cu, 1, "contactForce");

    // --- Initial Charge Assignment (Flat Electrodes) ---
    bool flag_small = (fabs(voltage) < 0.01);
    CudaArray& posq = cu.getPosq();

    int blockSize_init = 256;
    initializeChargesKernel<<<(numCathodes + blockSize_init - 1) / blockSize_init, blockSize_init, 0, cu.getCurrentStream()>>>(
        numCathodes, (const int*)d_cathodeIndices->getDevicePointer(), (const double*)d_cathodeAreas->getDevicePointer(),
        (float4*)posq.getDevicePointer(), voltage, Lgap, Lcell, +1.0, flag_small);

    initializeChargesKernel<<<(numAnodes + blockSize_init - 1) / blockSize_init, blockSize_init, 0, cu.getCurrentStream()>>>(
        numAnodes, (const int*)d_anodeIndices->getDevicePointer(), (const double*)d_anodeAreas->getDevicePointer(),
        (float4*)posq.getDevicePointer(), voltage, Lgap, Lcell, -1.0, flag_small);

    // --- Initialize Buckyballs/Nanotubes (New Logic) ---
    // Need positions on CPU for geometry init
    std::vector<OpenMM::Vec3> positions = cu.getPositions(); // Downloads positions from GPU
    OpenMM::Vec3 boxVectors[3];
    cu.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);

    // Buckyballs
    for(const auto& init : buckyballInitData) {
        CudaConductorData* c = new CudaConductorData();
        c->electrodeType = init.electrodeType;
        c->voltage = init.voltage * CONVERSION_EV_KJMOL;
        c->closeThreshold = 1.5;

        int Natoms = init.virtualAtoms.size();
        c->d_virtualAtomIndices = CudaArray::create<int>(cu, Natoms, "bb_virt");
        c->d_realAtomIndices = CudaArray::create<int>(cu, init.realAtoms.size(), "bb_real");
        c->d_normals = CudaArray::create<double>(cu, 3*Natoms, "bb_norm");

        // Fix 1: Zip Sort (Synchronized Sorting)
        // 1. Create pairs for synchronized sorting
        std::vector<std::pair<int, int>> pairs;
        pairs.reserve(Natoms);
        for(size_t k=0; k < init.virtualAtoms.size(); k++) {
            pairs.push_back({init.virtualAtoms[k], init.realAtoms[k]});
        }

        // 2. Sort based on virtual index
        std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.first < b.first;
            }
        );

        // 3. Unzip back to separate vectors
        std::vector<int> sortedVirtual, sortedReal;
        sortedVirtual.reserve(Natoms);
        sortedReal.reserve(Natoms);
        for(const auto& p : pairs) {
            sortedVirtual.push_back(p.first);
            sortedReal.push_back(p.second);
        }

        // 4. Upload sorted data
        c->d_virtualAtomIndices->upload(sortedVirtual);
        c->d_realAtomIndices->upload(sortedReal);

        initializeBuckyballGeometry(c, positions);
        findContactNeighborConductor(c, positions);
        buckyballs.push_back(c);
    }
    buckyballInitData.clear(); // Free memory

    // Nanotubes
    for(const auto& init : nanotubeInitData) {
        CudaConductorData* c = new CudaConductorData();
        c->electrodeType = init.electrodeType;
        c->voltage = init.voltage * CONVERSION_EV_KJMOL;
        c->closeThreshold = 1.5;
        c->axis = init.axis;

        int Natoms = init.virtualAtoms.size();
        c->d_virtualAtomIndices = CudaArray::create<int>(cu, Natoms, "nt_virt");
        c->d_realAtomIndices = CudaArray::create<int>(cu, init.realAtoms.size(), "nt_real");
        c->d_normals = CudaArray::create<double>(cu, 3*Natoms, "nt_norm");

        // Fix 1: Zip Sort (Synchronized Sorting)
        // 1. Create pairs
        std::vector<std::pair<int, int>> pairs;
        pairs.reserve(Natoms);
        for(size_t k=0; k < init.virtualAtoms.size(); k++) {
            pairs.push_back({init.virtualAtoms[k], init.realAtoms[k]});
        }

        // 2. Sort
        std::sort(pairs.begin(), pairs.end(),
            [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
                return a.first < b.first;
            }
        );

        // 3. Unzip
        std::vector<int> sortedVirtual, sortedReal;
        sortedVirtual.reserve(Natoms);
        sortedReal.reserve(Natoms);
        for(const auto& p : pairs) {
            sortedVirtual.push_back(p.first);
            sortedReal.push_back(p.second);
        }

        // 4. Upload
        c->d_virtualAtomIndices->upload(sortedVirtual);
        c->d_realAtomIndices->upload(sortedReal);

        initializeNanotubeGeometry(c, positions, boxVectors);
        findContactNeighborConductor(c, positions);
        nanotubes.push_back(c);
    }
    nanotubeInitData.clear();

    cu.invalidateMolecules();
    gpuInitialized = true;
    std::cout << "[CUDA] initializeGPU() complete. Loaded " << buckyballs.size() << " Buckyballs, " << nanotubes.size() << " Nanotubes." << std::endl;
}

double CudaCalcConstantVKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // Initialize on first execute
    if (!gpuInitialized) {
        initializeGPU();

        // P1 FIX: Enforce Virtual Site Parameters (Inline Logic)
        // Collect all virtual indices
        std::vector<int> allVirtualIndices;
        allVirtualIndices.insert(allVirtualIndices.end(), cathodeIndices.begin(), cathodeIndices.end());
        allVirtualIndices.insert(allVirtualIndices.end(), anodeIndices.begin(), anodeIndices.end());

        for(auto* c : buckyballs) {
            std::vector<int> idx(c->d_virtualAtomIndices->getSize());
            c->d_virtualAtomIndices->download(idx);
            allVirtualIndices.insert(allVirtualIndices.end(), idx.begin(), idx.end());
        }
        for(auto* c : nanotubes) {
            std::vector<int> idx(c->d_virtualAtomIndices->getSize());
            c->d_virtualAtomIndices->download(idx);
            allVirtualIndices.insert(allVirtualIndices.end(), idx.begin(), idx.end());
        }

        // Update Parameters on Host
        for(int index : allVirtualIndices) {
            double charge, sigma, epsilon;
            nonbondedForce->getParticleParameters(index, charge, sigma, epsilon);
            // Force sigma=1.0, epsilon=0.0 (Matches Python usage)
            nonbondedForce->setParticleParameters(index, charge, 1.0, 0.0);
        }

        nonbondedForce->updateParametersInContext(context.getOwner());
    }

    CudaArray& posq = cu.getPosq();
    int blockSize = 256;

    // 1. Calculate Q_analytic (Zero Transfer)
    cudaMemsetAsync((void*)d_Q_analytic_cathode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());
    cudaMemsetAsync((void*)d_Q_analytic_anode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());

    computeGeometricChargeKernel<<<1, 1, 0, cu.getCurrentStream()>>>((double*)d_Q_analytic_cathode->getDevicePointer(), voltage, Lgap, Lcell, totalArea, +1.0);
    computeGeometricChargeKernel<<<1, 1, 0, cu.getCurrentStream()>>>((double*)d_Q_analytic_anode->getDevicePointer(), voltage, Lgap, Lcell, totalArea, -1.0);

    int nbElectrolyte = (numElectrolytes + blockSize - 1) / blockSize;
    if (nbElectrolyte > 0) {
        warpAssistedReductionKernel<ImageChargeFunctor><<<nbElectrolyte, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(
            numElectrolytes, (const int*)d_electrolyteIndices->getDevicePointer(), (const float4*)posq.getDevicePointer(),
            (double*)d_cathode_partial->getDevicePointer(), z_anode, Lcell);
        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(nbElectrolyte, (const double*)d_cathode_partial->getDevicePointer(), (double*)d_Q_analytic_cathode->getDevicePointer());

        warpAssistedReductionKernel<ImageChargeFunctor><<<nbElectrolyte, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(
            numElectrolytes, (const int*)d_electrolyteIndices->getDevicePointer(), (const float4*)posq.getDevicePointer(),
            (double*)d_anode_partial->getDevicePointer(), z_cathode, Lcell);
        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(nbElectrolyte, (const double*)d_anode_partial->getDevicePointer(), (double*)d_Q_analytic_anode->getDevicePointer());
    }

    // P0 FIX: Add Image Charge Contribution from Conductors
    // Buckyballs
    for(auto* c : buckyballs) {
        int N = c->d_virtualAtomIndices->getSize();
        int nb = (N + blockSize - 1) / blockSize;

        // Add to Cathode Q_analytic (z_opposite = z_anode)
        warpAssistedReductionKernel<ImageChargeFunctor><<<nb, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(
            N, (const int*)c->d_virtualAtomIndices->getDevicePointer(), (const float4*)posq.getDevicePointer(),
            (double*)d_cathode_partial->getDevicePointer(), z_anode, Lcell);
        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(nb, (const double*)d_cathode_partial->getDevicePointer(), (double*)d_Q_analytic_cathode->getDevicePointer());

        // Add to Anode Q_analytic (z_opposite = z_cathode)
        warpAssistedReductionKernel<ImageChargeFunctor><<<nb, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(
            N, (const int*)c->d_virtualAtomIndices->getDevicePointer(), (const float4*)posq.getDevicePointer(),
            (double*)d_anode_partial->getDevicePointer(), z_cathode, Lcell);
        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(nb, (const double*)d_anode_partial->getDevicePointer(), (double*)d_Q_analytic_anode->getDevicePointer());
    }

    // Nanotubes
    for(auto* c : nanotubes) {
        int N = c->d_virtualAtomIndices->getSize();
        int nb = (N + blockSize - 1) / blockSize;

        // To Cathode Analytic
        warpAssistedReductionKernel<ImageChargeFunctor><<<nb, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(
            N, (const int*)c->d_virtualAtomIndices->getDevicePointer(), (const float4*)posq.getDevicePointer(),
            (double*)d_cathode_partial->getDevicePointer(), z_anode, Lcell);
        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(nb, (const double*)d_cathode_partial->getDevicePointer(), (double*)d_Q_analytic_cathode->getDevicePointer());

        // To Anode Analytic
        warpAssistedReductionKernel<ImageChargeFunctor><<<nb, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(
            N, (const int*)c->d_virtualAtomIndices->getDevicePointer(), (const float4*)posq.getDevicePointer(),
            (double*)d_anode_partial->getDevicePointer(), z_cathode, Lcell);
        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(nb, (const double*)d_anode_partial->getDevicePointer(), (double*)d_Q_analytic_anode->getDevicePointer());
    }

    // --- Loop ---
    for (int iter = 0; iter < nIterations; iter++) {
        int forceGroups = context.getIntegrator().getIntegrationForceGroups();
        forceGroups &= ~(1U << CONSTANTV_FORCE_GROUP);
        context.calcForcesAndEnergy(true, false, forceGroups);
        CudaArray& forces = cu.getForce();

        // Update Flat Electrodes
        int nbCathode = (numCathodes + blockSize - 1) / blockSize;
        if(nbCathode > 0) computeAndUpdateChargesFusedKernel<<<nbCathode, blockSize, 0, cu.getCurrentStream()>>>(numCathodes, (const int*)d_cathodeIndices->getDevicePointer(), (const double*)d_cathodeAreas->getDevicePointer(), (const float4*)forces.getDevicePointer(), (float4*)posq.getDevicePointer(), voltage, Lgap, +2.0);

        int nbAnode = (numAnodes + blockSize - 1) / blockSize;
        if(nbAnode > 0) computeAndUpdateChargesFusedKernel<<<nbAnode, blockSize, 0, cu.getCurrentStream()>>>(numAnodes, (const int*)d_anodeIndices->getDevicePointer(), (const double*)d_anodeAreas->getDevicePointer(), (const float4*)forces.getDevicePointer(), (float4*)posq.getDevicePointer(), voltage, Lgap, -2.0);

        // Update Buckyballs
        for(CudaConductorData* c : buckyballs) {
            int N = c->d_virtualAtomIndices->getSize();
            int nb = (N + blockSize - 1) / blockSize;
            numericalChargeConductorKernel<<<nb, blockSize, 0, cu.getCurrentStream()>>>(
                N, (const int*)c->d_virtualAtomIndices->getDevicePointer(),
                (const double*)c->d_normals->getDevicePointer(),
                (const float4*)forces.getDevicePointer(),
                (float4*)posq.getDevicePointer(),
                c->area_atom
            );
        }

        // Update Nanotubes
        for(CudaConductorData* c : nanotubes) {
            int N = c->d_virtualAtomIndices->getSize();
            int nb = (N + blockSize - 1) / blockSize;
            numericalChargeConductorKernel<<<nb, blockSize, 0, cu.getCurrentStream()>>>(
                N, (const int*)c->d_virtualAtomIndices->getDevicePointer(),
                (const double*)c->d_normals->getDevicePointer(),
                (const float4*)forces.getDevicePointer(),
                (float4*)posq.getDevicePointer(),
                c->area_atom
            );
        }

        // Step 2: Charge Transfer (Zero Transfer Logic)
        // Requirement: Need current charge of contact atom.
        // Problem: Contact atom is on electrode, not in conductor struct.
        // Solution: Read from global posq array using contactAtomIndex.
        // BUT: We need to know which electrode (for sign).
        // We have c->electrodeType and c->contactAtomIndex.

        if(!buckyballs.empty() || !nanotubes.empty()) {
            cu.invalidateMolecules(); // Notify charges changed (from Step 1)
            context.calcForcesAndEnergy(true, false, forceGroups); // Recompute forces for Step 2
            // forces reference remains valid

            // --- Buckyballs Step 2 ---
            for(CudaConductorData* c : buckyballs) {
                if(!c->closeToElectrode || c->contactAtomIndex < 0) continue;

                // 1. Extract force of contact atom to buffer
                extractContactForceKernel<<<1, 1, 0, cu.getCurrentStream()>>>(
                    c->contactAtomIndex, (const float4*)forces.getDevicePointer(), (float4*)d_contactForceBuffer->getDevicePointer()
                );

                // 2. Compute and Apply dq
                int N = c->d_virtualAtomIndices->getSize();
                int nb = (N + blockSize - 1) / blockSize;

                int sign_electrode = (c->electrodeType == "cathode") ? 1 : -1;

                computeAndApplyChargeTransferKernel<<<nb, blockSize, 0, cu.getCurrentStream()>>>(
                    N, (const int*)c->d_virtualAtomIndices->getDevicePointer(),
                    (const float4*)d_contactForceBuffer->getDevicePointer(),
                    (float4*)posq.getDevicePointer(),
                    voltage, Lgap, c->dr_center_contact, 0.0, // length unused for Bucky
                    c->dr_center_contact * c->dr_center_contact, // geometricFactor = dr^2
                    sign_electrode,
                    c->contactAtomIndex // New arg
                );
            }

            // --- Nanotubes Step 2 ---
            for(CudaConductorData* c : nanotubes) {
                if(!c->closeToElectrode || c->contactAtomIndex < 0) continue;

                extractContactForceKernel<<<1, 1, 0, cu.getCurrentStream()>>>(
                    c->contactAtomIndex, (const float4*)forces.getDevicePointer(), (float4*)d_contactForceBuffer->getDevicePointer()
                );

                int N = c->d_virtualAtomIndices->getSize();
                int nb = (N + blockSize - 1) / blockSize;
                int sign_electrode = (c->electrodeType == "cathode") ? 1 : -1;

                computeAndApplyChargeTransferKernel<<<nb, blockSize, 0, cu.getCurrentStream()>>>(
                    N, (const int*)c->d_virtualAtomIndices->getDevicePointer(),
                    (const float4*)d_contactForceBuffer->getDevicePointer(),
                    (float4*)posq.getDevicePointer(),
                    voltage, Lgap, c->dr_center_contact, c->length,
                    c->dr_center_contact * c->length / 2.0, // geometricFactor = dr*L/2
                    sign_electrode,
                    c->contactAtomIndex
                );
            }
        }

        // Green's Reciprocity
        cudaMemsetAsync((void*)d_Q_numeric_cathode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());
        cudaMemsetAsync((void*)d_Q_numeric_anode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());

        // Flat numeric
        if(nbCathode > 0) {
            warpAssistedReductionKernel<SumFunctor><<<nbCathode, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(numCathodes, (const int*)d_cathodeIndices->getDevicePointer(), (const float4*)posq.getDevicePointer(), (double*)d_cathode_numeric_partial->getDevicePointer(), 0.0, 0.0);
            reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(nbCathode, (const double*)d_cathode_numeric_partial->getDevicePointer(), (double*)d_Q_numeric_cathode->getDevicePointer());
        }
        if(nbAnode > 0) {
            warpAssistedReductionKernel<SumFunctor><<<nbAnode, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(numAnodes, (const int*)d_anodeIndices->getDevicePointer(), (const float4*)posq.getDevicePointer(), (double*)d_anode_numeric_partial->getDevicePointer(), 0.0, 0.0);
            reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(nbAnode, (const double*)d_anode_numeric_partial->getDevicePointer(), (double*)d_Q_numeric_anode->getDevicePointer());
        }

        // Bucky/Nano numeric charges need to be added to Q_numeric_total!
        for(CudaConductorData* c : buckyballs) {
            int N = c->d_virtualAtomIndices->getSize();
            int nb = (N + blockSize - 1) / blockSize;
            warpAssistedReductionKernel<SumFunctor><<<nb, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(
                N, (const int*)c->d_virtualAtomIndices->getDevicePointer(),
                (const float4*)posq.getDevicePointer(),
                (double*)d_cathode_numeric_partial->getDevicePointer(), 0.0, 0.0);

            reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(
                nb, (const double*)d_cathode_numeric_partial->getDevicePointer(),
                (double*)d_Q_numeric_cathode->getDevicePointer());
        }
        for(CudaConductorData* c : nanotubes) {
             int N = c->d_virtualAtomIndices->getSize();
             int nb = (N + blockSize - 1) / blockSize;
             warpAssistedReductionKernel<SumFunctor><<<nb, blockSize, (blockSize/32)*sizeof(double), cu.getCurrentStream()>>>(
                N, (const int*)c->d_virtualAtomIndices->getDevicePointer(),
                (const float4*)posq.getDevicePointer(),
                (double*)d_cathode_numeric_partial->getDevicePointer(), 0.0, 0.0);

            reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(
                nb, (const double*)d_cathode_numeric_partial->getDevicePointer(),
                (double*)d_Q_numeric_cathode->getDevicePointer());
        }

        // Scale
        if(nbCathode > 0) computeScaleAndNormalizeKernel<<<nbCathode, blockSize, 0, cu.getCurrentStream()>>>(numCathodes, (const int*)d_cathodeIndices->getDevicePointer(), (float4*)posq.getDevicePointer(), (const double*)d_Q_analytic_cathode->getDevicePointer(), (const double*)d_Q_numeric_cathode->getDevicePointer());
        if(nbAnode > 0) computeScaleAndNormalizeKernel<<<nbAnode, blockSize, 0, cu.getCurrentStream()>>>(numAnodes, (const int*)d_anodeIndices->getDevicePointer(), (float4*)posq.getDevicePointer(), (const double*)d_Q_analytic_anode->getDevicePointer(), (const double*)d_Q_numeric_anode->getDevicePointer());

        // Scale Buckyballs/Nanotubes
        for(CudaConductorData* c : buckyballs) {
            int N = c->d_virtualAtomIndices->getSize();
            int nb = (N + blockSize - 1) / blockSize;
            computeScaleAndNormalizeKernel<<<nb, blockSize, 0, cu.getCurrentStream()>>>(
                N, (const int*)c->d_virtualAtomIndices->getDevicePointer(),
                (float4*)posq.getDevicePointer(),
                (const double*)d_Q_analytic_cathode->getDevicePointer(),
                (const double*)d_Q_numeric_cathode->getDevicePointer());
        }
        for(CudaConductorData* c : nanotubes) {
            int N = c->d_virtualAtomIndices->getSize();
            int nb = (N + blockSize - 1) / blockSize;
            computeScaleAndNormalizeKernel<<<nb, blockSize, 0, cu.getCurrentStream()>>>(
                N, (const int*)c->d_virtualAtomIndices->getDevicePointer(),
                (float4*)posq.getDevicePointer(),
                (const double*)d_Q_analytic_cathode->getDevicePointer(),
                (const double*)d_Q_numeric_cathode->getDevicePointer());
        }

        cu.invalidateMolecules();
    }

    cu.invalidateMolecules();
    return 0.0;
}

// (Kernel definitions need to match call signatures)
// Updated computeAndApplyChargeTransferKernel definition:
__global__ void computeAndApplyChargeTransferKernel(
    int numAtoms,
    const int* __restrict__ indices,
    const float4* __restrict__ d_contactForce,
    float4* __restrict__ posq,
    double voltage,
    double Lgap,
    double dr_center_contact,
    double length,
    double geometricFactor,
    int sign_electrode,
    int contactAtomIndex // Replaced q_contact
) {
    __shared__ double dq_atom;

    if (threadIdx.x == 0) {
        float4 f = d_contactForce[0];
        // Read q_contact from global memory
        double q_i = (double)posq[contactAtomIndex].w;

        double En_external = 0.0;
        if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
            double Ex = (double)f.x / q_i;
            double Ey = (double)f.y / q_i;
            double Ez = (double)f.z / q_i;

            // P2 FIX: Correct Anode Sign Logic
            // Cathode (sign=1) -> Normal +Z (1.0)
            // Anode (sign=-1)  -> Normal -Z (-1.0)
            double normal_z = (sign_electrode > 0) ? 1.0 : -1.0;
            En_external = Ez * normal_z;
        }

        double dE_conductor = -(En_external + voltage / Lgap / 2.0) * CONVERSION_KJMOLNM_AU;
        double dQ = -1.0 * dE_conductor * geometricFactor;
        dq_atom = dQ / numAtoms;
    }
    __syncthreads();

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numAtoms) return;

    int atomIdx = indices[i];
    posq[atomIdx].w += (float)dq_atom;
}
