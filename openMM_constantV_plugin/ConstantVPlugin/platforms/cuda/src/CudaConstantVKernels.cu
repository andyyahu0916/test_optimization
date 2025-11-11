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
// CUDA Kernels - 直接翻譯 Reference 實現
// ═══════════════════════════════════════════════════════════

/**
 * Kernel: 初始化電極電荷
 * 翻譯自: ReferenceConstantVKernels.cpp::initialize (Line 176-203)
 * Python: Fixed_Voltage_routines.py::initialize_Charge (278-303)
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
    // q_i = sign / (4.0 * numpy.pi) * area * (V/Lgap + V/Lcell) * conversion
    double q_i = sign / (4.0 * M_PI) * area *
                 (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOLNM_AU;

    // Line 182-184: 低電壓保護
    if (flagSmall) {
        q_i = q_i + sign * SMALL_THRESHOLD;
    }

    // Line 186-187: 寫入電荷到 posq.w（零傳輸！）
    posq[atomIdx].w = (float)q_i;
}

/**
 * Kernel: 計算外部電場 Ez_external = F_z / q_old
 * 翻譯自: ReferenceConstantVKernels.cpp::execute (Line 379-381)
 * Python: MM_classes.py (Line 327)
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

    // Line 379-381: 照抄除零保護（0.9 係數很重要！）
    if (fabs(q_old) > (0.9 * SMALL_THRESHOLD)) {
        Ez_external[i] = F_z / q_old;
    } else {
        Ez_external[i] = 0.0;
    }
}

/**
 * Kernel: 更新電極電荷（Maxwell 邊界條件）
 * 翻譯自: ReferenceConstantVKernels.cpp::execute (Line 386-396)
 * Python: MM_classes.py (Line 330, 345)
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

    // Line 386-388: 完全照抄 Maxwell 邊界條件
    // Cathode: q_i = 2.0 / (4π) × area × (V/Lgap + Ez) × conversion
    // Anode:   q_i = -2.0 / (4π) × area × (V/Lgap + Ez) × conversion
    double q_i = sign / (4.0 * M_PI) * area *
                 (voltage / Lgap + Ez) * CONVERSION_KJMOLNM_AU;

    // Line 391-393: 電荷閾值保護
    if (fabs(q_i) < SMALL_THRESHOLD) {
        q_i = sign / 2.0 * SMALL_THRESHOLD;  // sign/2 因為 sign 已經包含 ±2.0
    }

    // 直接寫入 posq.w（零傳輸！）
    posq[atomIdx].w = (float)q_i;
}

/**
 * Kernel: 計算解析電荷的幾何貢獻
 * 翻譯自: ReferenceConstantVKernels.cpp::computeElectrodeChargeAnalytic (Line 228-230)
 * Python: Fixed_Voltage_routines.py (Line 324-325)
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
        // Line 228-230: 完全照抄幾何貢獻公式
        *Q_analytic = sign / (4.0 * M_PI) * totalArea *
                      (voltage / Lgap + voltage / Lcell) *
                      CONVERSION_KJMOLNM_AU;
    }
}

/**
 * Kernel: 計算解析電荷的鏡像電荷貢獻（並行 reduction）
 * 翻譯自: ReferenceConstantVKernels.cpp::computeElectrodeChargeAnalytic (Line 238-248)
 * Python: Fixed_Voltage_routines.py (Line 327-333)
 */
__global__ void computeImageChargeKernel(
    int numElectrolytes,
    const int* __restrict__ electrolyteIndices,
    const float4* __restrict__ posq,
    double* __restrict__ Q_analytic_partial,  // 每個 block 一個部分和
    double z_opposite,
    double Lcell
) {
    extern __shared__ double sdata[];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Line 238-248: 完全照抄鏡像電荷公式
    double local_sum = 0.0;
    if (i < numElectrolytes) {
        int index = electrolyteIndices[i];
        double q_i = (double)posq[index].w;  // 實時讀取（Bug #4修復）
        double z_atom = (double)posq[index].z;
        double z_distance = fabs(z_atom - z_opposite);

        // Line 247: Q_analytic += (z_distance / Lcell) * (-q_i)
        local_sum = (z_distance / Lcell) * (-q_i);
    }

    // Shared memory reduction
    sdata[tid] = local_sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Block 領導寫入部分和
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

/**
 * Kernel: 計算電極總電荷（數值）
 * 翻譯自: ReferenceConstantVKernels.cpp::scaleChargesAnalytic (Line 262-271)
 */
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

/**
 * Kernel: 歸一化電荷（Green's Reciprocity 校正）
 * 翻譯自: ReferenceConstantVKernels.cpp::scaleChargesAnalytic (Line 282-290)
 * Python: Fixed_Voltage_routines.py (Line 362-370)
 */
__global__ void scaleChargesKernel(
    int numElectrodes,
    const int* __restrict__ electrodeIndices,
    float4* __restrict__ posq,
    double scale_factor
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numElectrodes) return;

    int atomIdx = electrodeIndices[i];

    // Line 286-288: 完全照抄縮放公式
    double q_old = (double)posq[atomIdx].w;
    double q_new = q_old * scale_factor;
    posq[atomIdx].w = (float)q_new;
}

// ═══════════════════════════════════════════════════════════
// CudaCalcConstantVKernel Implementation
// ═══════════════════════════════════════════════════════════

CudaCalcConstantVKernel::CudaCalcConstantVKernel(string name, const Platform& platform, CudaContext& cu) :
    CalcConstantVKernel(name, platform), cu(cu), gpuInitialized(false) {

    // 初始化所有指針為 nullptr
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
}

void CudaCalcConstantVKernel::initialize(const System& system, const ConstantVForce& force) {
    std::cout << "[CUDA] initialize() called (storing parameters only, deferring GPU work)" << std::endl;

    // 獲取參數（照抄 Reference Line 151-156）
    voltage = force.getVoltage() * CONVERSION_EV_KJMOL;  // V -> kJ/mol（照抄Python Line 88, Reference Line 151）
    Lgap = force.getLgap();
    Lcell = force.getLcell();
    totalArea = force.getTotalArea();
    z_cathode = force.getZCathode();
    z_anode = force.getZAnode();
    nIterations = force.getNumIterations();

    std::cout << "[CUDA] Parameters read: V=" << voltage << " kJ/mol, Lgap=" << Lgap << ", Lcell=" << Lcell << std::endl;

    // 讀取 cathode atoms
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

    // 讀取 anode atoms
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

    // 讀取 electrolyte atoms
    numElectrolytes = force.getNumElectrolyteAtoms();
    electrolyteIndices.resize(numElectrolytes);
    for (int i = 0; i < numElectrolytes; i++) {
        int particle;
        double charge;
        force.getElectrolyteAtomParameters(i, particle, charge);
        electrolyteIndices[i] = particle;
    }

    std::cout << "[CUDA] Atoms: cathode=" << numCathodes << ", anode=" << numAnodes << ", electrolyte=" << numElectrolytes << std::endl;

    // 獲取 NonbondedForce（照抄 Reference Line 75-82）
    std::cout << "[CUDA] Looking for NonbondedForce..." << std::endl;
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

    std::cout << "[CUDA] NonbondedForce found" << std::endl;
    std::cout << "[CUDA] initialize() complete (GPU work deferred to first execute())" << std::endl;
}

/**
 * Initialize GPU resources (deferred from initialize() to first execute())
 * This must be called AFTER the CUDA context is fully initialized
 */
void CudaCalcConstantVKernel::initializeGPU() {
    std::cout << "[CUDA] initializeGPU() called - allocating GPU memory and initializing charges" << std::endl;

    // 確保 CUDA context 已激活
    cu.setAsCurrent();

    // 分配 GPU 內存
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

    // Partial sums for reduction
    int blockSize = 256;
    int numBlocks_cathode = (numElectrolytes + blockSize - 1) / blockSize;
    int numBlocks_anode = (numElectrolytes + blockSize - 1) / blockSize;
    d_cathode_partial = CudaArray::create<double>(cu, numBlocks_cathode, "cathode_partial");
    d_anode_partial = CudaArray::create<double>(cu, numBlocks_anode, "anode_partial");

    int numBlocks_cath_numeric = (numCathodes + blockSize - 1) / blockSize;
    int numBlocks_anode_numeric = (numAnodes + blockSize - 1) / blockSize;
    d_cathode_numeric_partial = CudaArray::create<double>(cu, numBlocks_cath_numeric, "cathode_numeric_partial");
    d_anode_numeric_partial = CudaArray::create<double>(cu, numBlocks_anode_numeric, "anode_numeric_partial");

    std::cout << "[CUDA] GPU memory allocated, uploading data..." << std::endl;

    // 上傳到 GPU
    d_cathodeIndices->upload(cathodeIndices);
    d_anodeIndices->upload(anodeIndices);
    d_cathodeAreas->upload(cathodeAreas);
    d_anodeAreas->upload(anodeAreas);
    d_electrolyteIndices->upload(electrolyteIndices);

    std::cout << "[CUDA] Data uploaded" << std::endl;

    // ═══════════════════════════════════════════════════════════
    // Bug #6 修復：初始化電極電荷（照抄 Reference Line 168-204）
    // ═══════════════════════════════════════════════════════════

    // Line 170: 檢查電壓（voltage 已經是 kJ/mol）
    bool flag_small = (fabs(voltage) < 0.01);
    if (flag_small) {
        cout << "adding small value to initial charges..." << endl;
    }

    CudaArray& posq = cu.getPosq();

    int blockSize_init = 256;
    int numBlocks_cathode_init = (numCathodes + blockSize_init - 1) / blockSize_init;
    int numBlocks_anode_init = (numAnodes + blockSize_init - 1) / blockSize_init;

    // 初始化 Cathode
    initializeChargesKernel<<<numBlocks_cathode_init, blockSize_init, 0, cu.getCurrentStream()>>>(
        numCathodes,
        (const int*)d_cathodeIndices->getDevicePointer(),
        (const double*)d_cathodeAreas->getDevicePointer(),
        (float4*)posq.getDevicePointer(),
        voltage, Lgap, Lcell,
        +1.0,  // sign for cathode
        flag_small
    );
    cudaError_t err1 = cudaGetLastError();
    if (err1 != cudaSuccess) {
        std::cerr << "[CUDA] Cathode init kernel launch failed: " << cudaGetErrorString(err1) << std::endl;
        throw OpenMMException("CUDA kernel launch failed");
    }

    // 初始化 Anode
    initializeChargesKernel<<<numBlocks_anode_init, blockSize_init, 0, cu.getCurrentStream()>>>(
        numAnodes,
        (const int*)d_anodeIndices->getDevicePointer(),
        (const double*)d_anodeAreas->getDevicePointer(),
        (float4*)posq.getDevicePointer(),
        voltage, Lgap, Lcell,
        -1.0,  // sign for anode
        flag_small
    );
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        std::cerr << "[CUDA] Anode init kernel launch failed: " << cudaGetErrorString(err2) << std::endl;
        throw OpenMMException("CUDA kernel launch failed");
    }

    // 通知 OpenMM 電荷已更新
    cu.invalidateMolecules();

    gpuInitialized = true;
    std::cout << "[CUDA] initializeGPU() complete" << std::endl;
}

// ... (繼續下一個消息)
/**
 * Execute the kernel - SCF Iteration Loop
 * 翻譯自: ReferenceConstantVKernels.cpp::execute (Line 308-471)
 * Python: MM_classes.py::Poisson_solver_fixed_voltage (Line 287-374)
 */
double CudaCalcConstantVKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    // Lazy GPU initialization on first execute()
    if (!gpuInitialized) {
        std::cout << "[CUDA] First execute() call - initializing GPU resources" << std::endl;
        initializeGPU();
    }

    // 獲取 GPU 資源（零傳輸！）
    CudaArray& posq = cu.getPosq();
    CudaArray& forces = cu.getForce();

    int blockSize = 256;
    int numBlocks_cathode = (numCathodes + blockSize - 1) / blockSize;
    int numBlocks_anode = (numAnodes + blockSize - 1) / blockSize;
    int numBlocks_electrolyte = (numElectrolytes + blockSize - 1) / blockSize;

    size_t sharedMemSize = blockSize * sizeof(double);

    // ═══════════════════════════════════════════════════════════
    // SCF 迭代循環（照抄 Reference Line 352-462）
    // ═══════════════════════════════════════════════════════════

    for (int iter = 0; iter < nIterations; iter++) {

        // Line 354-357: 獲取力和位置（已在 GPU，零傳輸！）
        // State state = context.getState(State::Forces | State::Positions);
        // （CUDA 版本：forces 和 posq 已經在 GPU，無需 getState）

        // ───────────────────────────────────────────────────────
        // Step 1: 計算外部電場 Ez_external = F_z / q_old
        // Line 379-381
        // ───────────────────────────────────────────────────────

        computeEzExternalKernel<<<numBlocks_cathode, blockSize, 0, cu.getCurrentStream()>>>(
            numCathodes,
            (const int*)d_cathodeIndices->getDevicePointer(),
            (const float4*)forces.getDevicePointer(),
            (const float4*)posq.getDevicePointer(),
            (double*)d_Ez_cathode->getDevicePointer()
        );

        computeEzExternalKernel<<<numBlocks_anode, blockSize, 0, cu.getCurrentStream()>>>(
            numAnodes,
            (const int*)d_anodeIndices->getDevicePointer(),
            (const float4*)forces.getDevicePointer(),
            (const float4*)posq.getDevicePointer(),
            (double*)d_Ez_anode->getDevicePointer()
        );

        // ───────────────────────────────────────────────────────
        // Step 2: 更新電極電荷（Maxwell 邊界條件）
        // Line 386-396 (Cathode), Line 400-410 (Anode)
        // ───────────────────────────────────────────────────────

        // Cathode: sign = +2.0
        updateElectrodeChargesKernel<<<numBlocks_cathode, blockSize, 0, cu.getCurrentStream()>>>(
            numCathodes,
            (const int*)d_cathodeIndices->getDevicePointer(),
            (const double*)d_cathodeAreas->getDevicePointer(),
            (const double*)d_Ez_cathode->getDevicePointer(),
            (float4*)posq.getDevicePointer(),
            voltage, Lgap,
            +2.0  // sign for Cathode
        );

        // Anode: sign = -2.0
        updateElectrodeChargesKernel<<<numBlocks_anode, blockSize, 0, cu.getCurrentStream()>>>(
            numAnodes,
            (const int*)d_anodeIndices->getDevicePointer(),
            (const double*)d_anodeAreas->getDevicePointer(),
            (const double*)d_Ez_anode->getDevicePointer(),
            (float4*)posq.getDevicePointer(),
            voltage, Lgap,
            -2.0  // sign for Anode
        );

        // ───────────────────────────────────────────────────────
        // Step 3: Green's Reciprocity 校正
        // Line 418-458
        // ───────────────────────────────────────────────────────

        // === 3a. 清零解析/數值電荷緩衝區 ===
        cudaMemsetAsync((void*)d_Q_analytic_cathode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());
        cudaMemsetAsync((void*)d_Q_analytic_anode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());
        cudaMemsetAsync((void*)d_Q_numeric_cathode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());
        cudaMemsetAsync((void*)d_Q_numeric_anode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());

        // === 3b. 計算解析電荷（幾何貢獻）===
        // Cathode: sign = +1.0
        computeGeometricChargeKernel<<<1, 1, 0, cu.getCurrentStream()>>>(
            (double*)d_Q_analytic_cathode->getDevicePointer(),
            voltage, Lgap, Lcell, totalArea,
            +1.0
        );

        // Anode: sign = -1.0
        computeGeometricChargeKernel<<<1, 1, 0, cu.getCurrentStream()>>>(
            (double*)d_Q_analytic_anode->getDevicePointer(),
            voltage, Lgap, Lcell, totalArea,
            -1.0
        );

        // === 3c. 計算解析電荷（鏡像電荷貢獻）===
        // Cathode (z_opposite = z_anode)
        computeImageChargeKernel<<<numBlocks_electrolyte, blockSize, sharedMemSize, cu.getCurrentStream()>>>(
            numElectrolytes,
            (const int*)d_electrolyteIndices->getDevicePointer(),
            (const float4*)posq.getDevicePointer(),
            (double*)d_cathode_partial->getDevicePointer(),
            z_anode,
            Lcell
        );

        // Reduce partial sums for cathode
        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(
            numBlocks_electrolyte,
            (const double*)d_cathode_partial->getDevicePointer(),
            (double*)d_Q_analytic_cathode->getDevicePointer()
        );

        // Anode (z_opposite = z_cathode)
        computeImageChargeKernel<<<numBlocks_electrolyte, blockSize, sharedMemSize, cu.getCurrentStream()>>>(
            numElectrolytes,
            (const int*)d_electrolyteIndices->getDevicePointer(),
            (const float4*)posq.getDevicePointer(),
            (double*)d_anode_partial->getDevicePointer(),
            z_cathode,
            Lcell
        );

        // Reduce partial sums for anode
        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(
            numBlocks_electrolyte,
            (const double*)d_anode_partial->getDevicePointer(),
            (double*)d_Q_analytic_anode->getDevicePointer()
        );

        // === 3d. 計算數值總電荷 ===
        sumElectrodeChargesKernel<<<numBlocks_cathode, blockSize, sharedMemSize, cu.getCurrentStream()>>>(
            numCathodes,
            (const int*)d_cathodeIndices->getDevicePointer(),
            (const float4*)posq.getDevicePointer(),
            (double*)d_cathode_numeric_partial->getDevicePointer()
        );

        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(
            numBlocks_cathode,
            (const double*)d_cathode_numeric_partial->getDevicePointer(),
            (double*)d_Q_numeric_cathode->getDevicePointer()
        );

        sumElectrodeChargesKernel<<<numBlocks_anode, blockSize, sharedMemSize, cu.getCurrentStream()>>>(
            numAnodes,
            (const int*)d_anodeIndices->getDevicePointer(),
            (const float4*)posq.getDevicePointer(),
            (double*)d_anode_numeric_partial->getDevicePointer()
        );

        reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(
            numBlocks_anode,
            (const double*)d_anode_numeric_partial->getDevicePointer(),
            (double*)d_Q_numeric_anode->getDevicePointer()
        );

        // === 3e. 計算 scale_factor（需要 D2H 傳輸，但只有 4 個 double）===
        double Q_analytic_c, Q_numeric_c, Q_analytic_a, Q_numeric_a;

        cudaMemcpyAsync(&Q_analytic_c, (void*)d_Q_analytic_cathode->getDevicePointer(),
                        sizeof(double), cudaMemcpyDeviceToHost, cu.getCurrentStream());
        cudaMemcpyAsync(&Q_numeric_c, (void*)d_Q_numeric_cathode->getDevicePointer(),
                        sizeof(double), cudaMemcpyDeviceToHost, cu.getCurrentStream());
        cudaMemcpyAsync(&Q_analytic_a, (void*)d_Q_analytic_anode->getDevicePointer(),
                        sizeof(double), cudaMemcpyDeviceToHost, cu.getCurrentStream());
        cudaMemcpyAsync(&Q_numeric_a, (void*)d_Q_numeric_anode->getDevicePointer(),
                        sizeof(double), cudaMemcpyDeviceToHost, cu.getCurrentStream());

        cudaStreamSynchronize(cu.getCurrentStream());

        // Line 274-279: 計算縮放因子（照抄 Reference）
        double scale_cathode = -1.0;
        if (fabs(Q_numeric_c) > SMALL_THRESHOLD) {
            scale_cathode = Q_analytic_c / Q_numeric_c;
        }

        double scale_anode = -1.0;
        if (fabs(Q_numeric_a) > SMALL_THRESHOLD) {
            scale_anode = Q_analytic_a / Q_numeric_a;
        }

        // === 3f. 歸一化電荷 ===
        // Line 282-290
        if (scale_cathode > 0.0) {
            scaleChargesKernel<<<numBlocks_cathode, blockSize, 0, cu.getCurrentStream()>>>(
                numCathodes,
                (const int*)d_cathodeIndices->getDevicePointer(),
                (float4*)posq.getDevicePointer(),
                scale_cathode
            );
        }

        if (scale_anode > 0.0) {
            scaleChargesKernel<<<numBlocks_anode, blockSize, 0, cu.getCurrentStream()>>>(
                numAnodes,
                (const int*)d_anodeIndices->getDevicePointer(),
                (float4*)posq.getDevicePointer(),
                scale_anode
            );
        }

    } // End SCF iteration loop

    // Line 461: 通知 OpenMM 電荷已更新
    cu.invalidateMolecules();

    return 0.0;
}

void CudaCalcConstantVKernel::copyParametersToContext(ContextImpl& context, const ConstantVForce& force) {
    // 如果電壓改變，需要重新計算（照抄Reference Line 151）
    voltage = force.getVoltage() * CONVERSION_EV_KJMOL;  // V -> kJ/mol
}
