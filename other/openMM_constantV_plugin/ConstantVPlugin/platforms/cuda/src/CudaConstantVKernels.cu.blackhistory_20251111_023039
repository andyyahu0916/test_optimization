#include "CudaConstantVKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/cuda/CudaNonbondedUtilities.h"
#include <cuda.h>
#include <stdexcept> // 為了 std::runtime_error

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

// Coulomb constant in OpenMM units: kJ/mol · nm / e²
static const double COULOMB_CONSTANT = 138.935456;

// ===========================================================================
// CUDA Kernels
// ===========================================================================

/**
 * Kernel 1: Calculate electric field E_f[i] at each electrode atom from electrolyte.
 *
 * E_f[i] = Σ_j (k * q_f[j] / r_ij)
 *
 * [!!--- 警告：此核心物理上是錯誤的，參見 SOP 階段二 ---!!]
 */
__global__ void calculateEfKernel(
    int N,
    int M,
    const int* __restrict__ electrodeAtomIndices,
    const int* __restrict__ electrolyteAtomIndices,
    const double* __restrict__ fixedCharges, // 您的插件使用 double
    const float4* __restrict__ posq,         // OpenMM 儲存 float
    double* __restrict__ Ef
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    int electrodeIndex = electrodeAtomIndices[i];
    float3 electrodePos = make_float3(posq[electrodeIndex].x, posq[electrodeIndex].y, posq[electrodeIndex].z);

    double sum_Ef = 0.0;
    for (int j_base = 0; j_base < M; j_base++) {
        int electrolyteIndex = electrolyteAtomIndices[j_base];
        float3 electrolytePos = make_float3(posq[electrolyteIndex].x, posq[electrolyteIndex].y, posq[electrolyteIndex].z);
        
        // 使用 fixedCharges 陣列
        double q_j = fixedCharges[j_base];

        float dx = electrodePos.x - electrolytePos.x;
        float dy = electrodePos.y - electrolytePos.y;
        float dz = electrodePos.z - electrolytePos.z;
        float r2 = dx*dx + dy*dy + dz*dz;
        
        // 避免 r=0
        if (r2 > 1e-6) {
            float r = sqrt(r2);
            sum_Ef += COULOMB_CONSTANT * q_j / r;
        }
    }
    Ef[i] = sum_Ef;
}

/**
 * Kernel 2: Scatter-write computed charges.
 */
__global__ void scatterWriteChargesKernel(
    int N,
    const double* __restrict__ q_e,                 // [N] - Computed charges (double)
    const int* __restrict__ electrodeAtomIndices,  // [N] - Global particle indices
    float4* __restrict__ posq                      // [NumParticles] - Global pos+charge array (float)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N)
        return;

    int globalIndex = electrodeAtomIndices[i];
    posq[globalIndex].w = (float)q_e[i]; // Double -> Float 轉換
}


// ===========================================================================
// Kernel 實作
// ===========================================================================

CudaCalcConstantVKernel::CudaCalcConstantVKernel(string name, const Platform& platform, CudaContext& cu) :
    CalcConstantVKernel(name, platform), cu(cu), cublasHandle(NULL), N(0), M(0),
    d_electrodeAtomIndices(NULL), d_targetPotentials(NULL), d_electrolyteAtomIndices(NULL),
    d_fixedCharges(NULL), d_invCapMatrix(NULL), d_Ef(NULL), d_b(NULL), d_q_e(NULL) {

    cublasStatus_t status = cublasCreate(&cublasHandle);
    if (status != CUBLAS_STATUS_SUCCESS)
        throw OpenMMException("Failed to create cuBLAS handle");
}

CudaCalcConstantVKernel::~CudaCalcConstantVKernel() {
    if (cublasHandle)
        cublasDestroy(cublasHandle);
    delete d_electrodeAtomIndices;
    delete d_targetPotentials;
    delete d_electrolyteAtomIndices;
    delete d_fixedCharges;
    delete d_invCapMatrix;
    delete d_Ef;
    delete d_b;
    delete d_q_e;
}

void CudaCalcConstantVKernel::initialize(const System& system, const ConstantVForce& force) {
    N = force.getNumElectrodeAtoms();
    M = force.getNumElectrolyteAtoms();

    vector<int> electrodeAtomIndices(N);
    vector<double> targetPotentials(N);
    for (int i = 0; i < N; i++) {
        force.getElectrodeAtomParameters(i, electrodeAtomIndices[i], targetPotentials[i]);
    }

    vector<int> electrolyteAtomIndices(M);
    vector<double> fixedCharges(M);
    for (int i = 0; i < M; i++) {
        force.getElectrolyteAtomParameters(i, electrolyteAtomIndices[i], fixedCharges[i]);
    }
    
    // **FIX 1**: 修正 'too many arguments' 錯誤
    // 假設 force.getInverseCapacitanceMatrix() 傳回一個 vector
    vector<double> invCapMatrix = force.getInverseCapacitanceMatrix();
    if (invCapMatrix.size() != (size_t)N*N) {
         throw OpenMMException("CudaCalcConstantVKernel::initialize: C_inv matrix size mismatch.");
    }

    d_electrodeAtomIndices = CudaArray::create<int>(cu, N, "cv_electrodeIndices");
    d_targetPotentials = CudaArray::create<double>(cu, N, "cv_targetPotentials");
    d_electrolyteAtomIndices = CudaArray::create<int>(cu, M, "cv_electrolyteIndices");
    d_fixedCharges = CudaArray::create<double>(cu, M, "cv_fixedCharges");
    d_invCapMatrix = CudaArray::create<double>(cu, N*N, "cv_invCapMatrix");
    d_Ef = CudaArray::create<double>(cu, N, "cv_buf_Ef");
    d_b = CudaArray::create<double>(cu, N, "cv_buf_b");
    d_q_e = CudaArray::create<double>(cu, N, "cv_buf_q_e");

    d_electrodeAtomIndices->upload(electrodeAtomIndices);
    d_targetPotentials->upload(targetPotentials);
    d_electrolyteAtomIndices->upload(electrolyteAtomIndices);
    d_fixedCharges->upload(fixedCharges);
    d_invCapMatrix->upload(invCapMatrix);
}

/**
 * Execute the kernel.
 */
double CudaCalcConstantVKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {

    int blockSize = 256;
    int numBlocks_N = (N + blockSize - 1) / blockSize;
    
    // [!!--- 警告：此核心物理上是錯誤的，參見 SOP 階段二 ---!!]
    calculateEfKernel<<<numBlocks_N, blockSize, 0, cu.getCurrentStream()>>>(
        N,
        M,
        (const int*)d_electrodeAtomIndices->getDevicePointer(),
        (const int*)d_electrolyteAtomIndices->getDevicePointer(),
        (const double*)d_fixedCharges->getDevicePointer(),
        (const float4*)cu.getPosq().getDevicePointer(), 
        (double*)d_Ef->getDevicePointer()
    );

    cudaMemcpyAsync(
        (void*)d_b->getDevicePointer(),
        (const void*)d_targetPotentials->getDevicePointer(),
        N * sizeof(double),
        cudaMemcpyDeviceToDevice,
        cu.getCurrentStream()
    );
    
    const double alpha = -1.0;
    
    cublasStatus_t daxpy_result = cublasDaxpy(
        cublasHandle,
        N,
        &alpha,
        (const double*)d_Ef->getDevicePointer(), 1,
        (double*)d_b->getDevicePointer(), 1
    );
    if (daxpy_result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS daxpy failed in execute");
    }

    const double beta = 0.0;
    const double alpha_gemv = 1.0;
    cublasStatus_t dgemv_result = cublasDgemv(
        cublasHandle,
        CUBLAS_OP_N, // No transpose
        N, N,
        &alpha_gemv,
        (const double*)d_invCapMatrix->getDevicePointer(), N,
        (const double*)d_b->getDevicePointer(), 1,
        &beta,
        (double*)d_q_e->getDevicePointer(), 1
    );
    if (dgemv_result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("cuBLAS dgemv failed in execute");
    }
    
    CudaArray& posq = cu.getPosq();

    scatterWriteChargesKernel<<<numBlocks_N, blockSize, 0, cu.getCurrentStream()>>>(
        N,
        (const double*)d_q_e->getDevicePointer(),
        (const int*)d_electrodeAtomIndices->getDevicePointer(),
        (float4*)posq.getDevicePointer()
    );

    cu.invalidateMolecules();

    return 0.0;
}

void CudaCalcConstantVKernel::copyParametersToContext(ContextImpl& context, const ConstantVForce& force) {
    vector<double> targetPotentials(N);
    for (int i = 0; i < N; i++) {
        int particle;
        double potential;
        force.getElectrodeAtomParameters(i, particle, potential);
        targetPotentials[i] = potential;
    }
    d_targetPotentials->upload(targetPotentials);

    vector<double> invCapMatrix = force.getInverseCapacitanceMatrix();
    if (invCapMatrix.size() == (size_t)N*N) {
        d_invCapMatrix->upload(invCapMatrix);
    }
}
