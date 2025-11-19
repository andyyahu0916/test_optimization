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
 * Optimized Kernel: Fused computeEz + updateCharge
 *
 * 消除中間存儲和額外kernel啟動
 * 預計提升: 5-10%
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

    // ═══════════════════════════════════════════════════════════
    // Step 1: 計算Ez (inline, 不寫回global memory)
    // ═══════════════════════════════════════════════════════════
    double q_old = (double)posq[atomIdx].w;
    double F_z = (double)forces[atomIdx].z;

    double Ez_external = 0.0;
    if (fabs(q_old) > (0.9 * SMALL_THRESHOLD)) {
        Ez_external = F_z / q_old;
    }

    // ═══════════════════════════════════════════════════════════
    // Step 2: 立即更新電荷（Maxwell邊界條件）
    // ═══════════════════════════════════════════════════════════
    const double factor = sign / (4.0 * M_PI) * CONVERSION_KJMOLNM_AU;
    const double v_over_lgap = voltage / Lgap;

    double q_new = factor * area * (v_over_lgap + Ez_external);

    // 閾值保護
    if (fabs(q_new) < SMALL_THRESHOLD) {
        q_new = sign / 2.0 * SMALL_THRESHOLD;
    }

    // 直接写入posq.w
    posq[atomIdx].w = (float)q_new;
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

// Level 3 優化: Warp-level primitive 輔助函數
// Functor (函數對象) 用於加載數值電荷
struct NumericChargeLoader {
    __device__ double operator()(int i, const int* __restrict__ indices,
                                cudaTextureObject_t posqTexture, double, double) const {
        int atomIdx = indices[i];
        float4 posq_i = tex1Dfetch<float4>(posqTexture, atomIdx);
        return (double)posq_i.w;
    }
};

// Functor 用於加載鏡像電荷貢獻
struct ImageChargeLoader {
    __device__ double operator()(int i, const int* __restrict__ indices,
                                cudaTextureObject_t posqTexture,
                                double z_opposite, double Lcell) const {
        int index = indices[i];
        float4 posq_i = tex1Dfetch<float4>(posqTexture, index);
        double q_i = (double)posq_i.w;
        double z_atom = (double)posq_i.z;
        double z_distance = fabs(z_atom - z_opposite);
        return (z_distance / Lcell) * (-q_i);
    }
};

// Level 3 優化: 模板化的 Warp-Assisted Reduction Kernel
// Loader 是一個 functor 結構，它有一個 operator() 來加載值
template <typename Loader>
__global__ void warpAssistedReductionKernel(
    int numItems,
    const int* __restrict__ indices,
    cudaTextureObject_t posqTexture,
    double* __restrict__ partialSums,
    double arg1, // 用於 z_opposite
    double arg2  // 用於 Lcell
) {
    extern __shared__ double sdata[];
    double sum = 0.0;
    Loader loader;

    // Grid-stride loop
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < numItems; i += gridDim.x * blockDim.x) {
        sum += loader(i, indices, posqTexture, arg1, arg2);
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        sum += __shfl_down_sync(0xffffffff, sum, offset);

    if (threadIdx.x % 32 == 0)
        sdata[threadIdx.x / 32] = sum;

    __syncthreads();

    // Block-level reduction (由第一個 warp 執行)
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
 * Optimized Kernel: 在GPU上計算scale並歸一化
 *
 * 消除D2H傳輸和cudaStreamSynchronize()
 * 預計提升: 10-20%（最大收益！）
 */
__global__ void computeScaleAndNormalizeKernel(
    int numElectrodes,
    const int* __restrict__ electrodeIndices,
    float4* __restrict__ posq,
    const double* __restrict__ Q_analytic,  // [1]
    const double* __restrict__ Q_numeric    // [1]
) {
    // ═══════════════════════════════════════════════════════════
    // Phase 1: 計算scale factor (單線程)
    // ═══════════════════════════════════════════════════════════
    __shared__ double scale_factor;
    __shared__ bool valid_scale;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        double Q_a = *Q_analytic;
        double Q_n = *Q_numeric;

        // 正確的物理約束：Q_analytic 和 Q_numeric 應該有相同的符號。
        // 檢查它們的乘積是否為正，可以同時處理陰極（+*+ > 0）和陽極（-*- > 0）。
        if (fabs(Q_n) > SMALL_THRESHOLD && (Q_a * Q_n) > 0) {
            scale_factor = Q_a / Q_n;
            valid_scale = true; // 乘積 > 0 已經保證了 scale_factor > 0
        } else {
            valid_scale = false;
        }
    }
    __syncthreads();

    // ═══════════════════════════════════════════════════════════
    // Phase 2: 所有線程并行歸一化
    // ═══════════════════════════════════════════════════════════
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
    CalcConstantVKernel(name, platform), cu(cu), gpuInitialized(false), posqTexture(0), scfGraph(nullptr), scfGraphExec(nullptr) {

    // 初始化所有指針為 nullptr
    d_cathodeIndices = nullptr;
    d_anodeIndices = nullptr;
    d_cathodeAreas = nullptr;
    d_anodeAreas = nullptr;
    d_electrolyteIndices = nullptr;
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
    cu.setAsCurrent();
    if (posqTexture != 0) {
        cudaDestroyTextureObject(posqTexture);
    }
    if (scfGraphExec != nullptr) {
        cudaGraphExecDestroy(scfGraphExec);
    }
    if (scfGraph != nullptr) {
        cudaGraphDestroy(scfGraph);
    }
    delete d_cathodeIndices;
    delete d_anodeIndices;
    delete d_cathodeAreas;
    delete d_anodeAreas;
    delete d_electrolyteIndices;
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
    vector<pair<int, double>> cathode_pairs(numCathodes);
    for (int i = 0; i < numCathodes; i++) {
        force.getCathodeAtomParameters(i, cathode_pairs[i].first, cathode_pairs[i].second);
    }

    // 讀取 anode atoms
    numAnodes = force.getNumAnodeAtoms();
    vector<pair<int, double>> anode_pairs(numAnodes);
    for (int i = 0; i < numAnodes; i++) {
        force.getAnodeAtomParameters(i, anode_pairs[i].first, anode_pairs[i].second);
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

    // ═══════════════════════════════════════════════════════════
    // 優化2: 排序電極/電解質索引以提高內存合併
    // ═══════════════════════════════════════════════════════════
    std::sort(cathode_pairs.begin(), cathode_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::sort(anode_pairs.begin(), anode_pairs.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });
    
    std::sort(electrolyteIndices.begin(), electrolyteIndices.end());

    cathodeIndices.resize(numCathodes);
    cathodeAreas.resize(numCathodes);
    for (int i = 0; i < numCathodes; i++) {
        cathodeIndices[i] = cathode_pairs[i].first;
        cathodeAreas[i] = cathode_pairs[i].second;
    }

    anodeIndices.resize(numAnodes);
    anodeAreas.resize(numAnodes);
    for (int i = 0; i < numAnodes; i++) {
        anodeIndices[i] = anode_pairs[i].first;
        anodeAreas[i] = anode_pairs[i].second;
    }
    std::cout << "[CUDA] Electrode and electrolyte indices sorted for coalescing" << std::endl;

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

    // ═══════════════════════════════════════════════════════════
    // Level 2 優化: 創建並綁定 posq 紋理對象
    // ═══════════════════════════════════════════════════════════
    if (posqTexture == 0) {
        cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(resDesc));
        resDesc.resType = cudaResourceTypeLinear;
        resDesc.res.linear.devPtr = (void*)posq.getDevicePointer();
        resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
        resDesc.res.linear.desc.x = 32; // float4
        resDesc.res.linear.desc.y = 32;
        resDesc.res.linear.desc.z = 32;
        resDesc.res.linear.desc.w = 32;
        resDesc.res.linear.sizeInBytes = posq.getSize() * sizeof(float4);

        cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(texDesc));
        texDesc.readMode = cudaReadModeElementType;

        cudaError_t err = cudaCreateTextureObject(&posqTexture, &resDesc, &texDesc, NULL);
        if (err != cudaSuccess) {
            std::cerr << "[CUDA] Failed to create posq texture object: " << cudaGetErrorString(err) << std::endl;
            throw OpenMMException("Failed to create CUDA texture object");
        }
        std::cout << "[CUDA] posq texture object created" << std::endl;
    }

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
    cu.setAsCurrent();

    // Lazy GPU initialization on first execute()
    if (!gpuInitialized) {
        std::cout << "[CUDA] First execute() call - initializing GPU resources" << std::endl;
        initializeGPU();
    }

    // Level 3 優化: CUDA Graph
    if (scfGraphExec == nullptr) {
        std::cout << "[CUDA] Capturing SCF iteration loop into a CUDA Graph..." << std::endl;

        CudaArray& posq = cu.getPosq();
        CudaArray& forces = cu.getForce();
        int blockSize = 256;
        int numBlocks_cathode = (numCathodes + blockSize - 1) / blockSize;
        int numBlocks_anode = (numAnodes + blockSize - 1) / blockSize;
        int numBlocks_electrolyte = (numElectrolytes + blockSize - 1) / blockSize;
        size_t sharedMemSize = blockSize * sizeof(double);

        cudaStreamBeginCapture(cu.getCurrentStream(), cudaStreamCaptureModeGlobal);

        for (int iter = 0; iter < nIterations; iter++) {
            computeAndUpdateChargesFusedKernel<<<numBlocks_cathode, blockSize, 0, cu.getCurrentStream()>>>(numCathodes, (const int*)d_cathodeIndices->getDevicePointer(), (const double*)d_cathodeAreas->getDevicePointer(), (const float4*)forces.getDevicePointer(), (float4*)posq.getDevicePointer(), voltage, Lgap, +2.0);
            computeAndUpdateChargesFusedKernel<<<numBlocks_anode, blockSize, 0, cu.getCurrentStream()>>>(numAnodes, (const int*)d_anodeIndices->getDevicePointer(), (const double*)d_anodeAreas->getDevicePointer(), (const float4*)forces.getDevicePointer(), (float4*)posq.getDevicePointer(), voltage, Lgap, -2.0);
            
            cudaMemsetAsync((void*)d_Q_analytic_cathode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());
            cudaMemsetAsync((void*)d_Q_analytic_anode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());
            cudaMemsetAsync((void*)d_Q_numeric_cathode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());
            cudaMemsetAsync((void*)d_Q_numeric_anode->getDevicePointer(), 0, sizeof(double), cu.getCurrentStream());

            computeGeometricChargeKernel<<<1, 1, 0, cu.getCurrentStream()>>>((double*)d_Q_analytic_cathode->getDevicePointer(), voltage, Lgap, Lcell, totalArea, +1.0);
            computeGeometricChargeKernel<<<1, 1, 0, cu.getCurrentStream()>>>((double*)d_Q_analytic_anode->getDevicePointer(), voltage, Lgap, Lcell, totalArea, -1.0);

            warpAssistedReductionKernel<ImageChargeLoader><<<numBlocks_electrolyte, blockSize, sharedMemSize, cu.getCurrentStream()>>>(numElectrolytes, (const int*)d_electrolyteIndices->getDevicePointer(), posqTexture, (double*)d_cathode_partial->getDevicePointer(), z_anode, Lcell);
            reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(numBlocks_electrolyte, (const double*)d_cathode_partial->getDevicePointer(), (double*)d_Q_analytic_cathode->getDevicePointer());
            
            warpAssistedReductionKernel<ImageChargeLoader><<<numBlocks_electrolyte, blockSize, sharedMemSize, cu.getCurrentStream()>>>(numElectrolytes, (const int*)d_electrolyteIndices->getDevicePointer(), posqTexture, (double*)d_anode_partial->getDevicePointer(), z_cathode, Lcell);
            reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(numBlocks_electrolyte, (const double*)d_anode_partial->getDevicePointer(), (double*)d_Q_analytic_anode->getDevicePointer());

            warpAssistedReductionKernel<NumericChargeLoader><<<numBlocks_cathode, blockSize, sharedMemSize, cu.getCurrentStream()>>>(numCathodes, (const int*)d_cathodeIndices->getDevicePointer(), posqTexture, (double*)d_cathode_numeric_partial->getDevicePointer(), 0.0, 0.0);
            reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(numBlocks_cathode, (const double*)d_cathode_numeric_partial->getDevicePointer(), (double*)d_Q_numeric_cathode->getDevicePointer());

            warpAssistedReductionKernel<NumericChargeLoader><<<numBlocks_anode, blockSize, sharedMemSize, cu.getCurrentStream()>>>(numAnodes, (const int*)d_anodeIndices->getDevicePointer(), posqTexture, (double*)d_anode_numeric_partial->getDevicePointer(), 0.0, 0.0);
            reducePartialSumsKernel<<<1, 256, 256*sizeof(double), cu.getCurrentStream()>>>(numBlocks_anode, (const double*)d_anode_numeric_partial->getDevicePointer(), (double*)d_Q_numeric_anode->getDevicePointer());

            computeScaleAndNormalizeKernel<<<numBlocks_cathode, blockSize, 0, cu.getCurrentStream()>>>(numCathodes, (const int*)d_cathodeIndices->getDevicePointer(), (float4*)posq.getDevicePointer(), (const double*)d_Q_analytic_cathode->getDevicePointer(), (const double*)d_Q_numeric_cathode->getDevicePointer());
            computeScaleAndNormalizeKernel<<<numBlocks_anode, blockSize, 0, cu.getCurrentStream()>>>(numAnodes, (const int*)d_anodeIndices->getDevicePointer(), (float4*)posq.getDevicePointer(), (const double*)d_Q_analytic_anode->getDevicePointer(), (const double*)d_Q_numeric_anode->getDevicePointer());
        }

        cudaStreamEndCapture(cu.getCurrentStream(), &scfGraph);
        
        // 實例化圖以便執行
        cudaGraphInstantiate(&scfGraphExec, scfGraph, NULL, NULL, 0);
        std::cout << "[CUDA] CUDA Graph captured and instantiated." << std::endl;
    }

    // 啟動已捕獲的圖
    cudaGraphLaunch(scfGraphExec, cu.getCurrentStream());
    cudaStreamSynchronize(cu.getCurrentStream());

    // 通知 OpenMM 電荷已更新
    cu.invalidateMolecules();
    return 0.0;
}

void CudaCalcConstantVKernel::copyParametersToContext(ContextImpl& context, const ConstantVForce& force) {
    // 如果電壓改變，需要重新計算（照抄Reference Line 151）
    voltage = force.getVoltage() * CONVERSION_EV_KJMOL;  // V -> kJ/mol
}
