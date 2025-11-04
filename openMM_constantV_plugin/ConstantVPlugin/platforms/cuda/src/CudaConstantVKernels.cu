#include "CudaConstantVKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/NonbondedForce.h"
#include "openmm/System.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/cuda/CudaNonbondedUtilities.h"
#include <cuda.h>

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

// Coulomb constant in OpenMM units: kJ/mol · nm / e²
static const double COULOMB_CONSTANT = 138.935456;

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * Kernel 1: Calculate electric field E_f[i] at each electrode atom from electrolyte.
 *
 * E_f[i] = Σ_j (k * q_f[j] / r_ij)
 *
 * Each thread handles one electrode atom (index i).
 */
__global__ void calculateEfKernel(
    int N,
    int M,
    const int* __restrict__ electrodeAtomIndices,
    const int* __restrict__ electrolyteAtomIndices,
    const double* __restrict__ fixedCharges,
    const float4* __restrict__ posq,  // OpenMM's position array (x, y, z, q)
    double* __restrict__ Ef
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int electrode_idx = electrodeAtomIndices[i];
    float4 pos_i = posq[electrode_idx];

    double sum = 0.0;

    // Loop over all electrolyte atoms
    for (int j = 0; j < M; j++) {
        int electrolyte_idx = electrolyteAtomIndices[j];
        float4 pos_j = posq[electrolyte_idx];

        // Calculate distance
        double dx = pos_i.x - pos_j.x;
        double dy = pos_i.y - pos_j.y;
        double dz = pos_i.z - pos_j.z;
        double r_squared = dx*dx + dy*dy + dz*dz;

        if (r_squared > 0.0) {
            double r_inv = rsqrt(r_squared);  // 1/sqrt(r^2) = 1/r
            sum += COULOMB_CONSTANT * fixedCharges[j] * r_inv;
        }
    }

    Ef[i] = sum;
}

/**
 * Kernel 2: Calculate b[i] = V[i] - E_f[i]
 *
 * Simple element-wise subtraction.
 */
__global__ void calculateBKernel(
    int N,
    const double* __restrict__ targetPotentials,
    const double* __restrict__ Ef,
    double* __restrict__ b
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    b[i] = targetPotentials[i] - Ef[i];
}

/**
 * Kernel 3: Update NonbondedForce charges directly in GPU memory.
 *
 * This kernel writes the computed electrode charges q_e[i] directly
 * into the GPU memory location where NonbondedForce stores particle charges.
 *
 * WARNING: This requires direct access to CUDA platform's internal charge array.
 * The posq array stores (x, y, z, q) as float4, where q is the charge.
 */
__global__ void updateChargesKernel(
    int N,
    const int* __restrict__ electrodeAtomIndices,
    const double* __restrict__ q_e,
    float4* __restrict__ posq  // Will modify the .w component (charge)
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    int atom_idx = electrodeAtomIndices[i];

    // Read current posq
    float4 current = posq[atom_idx];

    // Update charge (w component)
    current.w = (float)q_e[i];

    // Write back
    posq[atom_idx] = current;
}

// ============================================================================
// Kernel Implementation
// ============================================================================

CudaCalcConstantVKernel::CudaCalcConstantVKernel(string name, const Platform& platform, CudaContext& cu) :
        CalcConstantVKernel(name, platform), cu(cu), cublasHandle(nullptr),
        N(0), M(0),
        d_electrodeAtomIndices(nullptr), d_targetPotentials(nullptr),
        d_electrolyteAtomIndices(nullptr), d_fixedCharges(nullptr),
        d_invCapMatrix(nullptr),
        d_Ef(nullptr), d_b(nullptr), d_q_e(nullptr),
        nonbondedForce(nullptr) {
}

CudaCalcConstantVKernel::~CudaCalcConstantVKernel() {
    if (cublasHandle != nullptr) {
        cublasDestroy(cublasHandle);
    }

    // CudaArray cleanup is automatic (RAII)
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

    if (N == 0 || M == 0) {
        throw OpenMMException("ConstantVForce requires at least one electrode and one electrolyte atom");
    }

    // Initialize cuBLAS
    cublasStatus_t status = cublasCreate(&cublasHandle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        throw OpenMMException("Failed to initialize cuBLAS");
    }

    // Allocate and upload static data to GPU
    vector<int> electrodeAtomIndices(N);
    vector<double> targetPotentials(N);
    for (int i = 0; i < N; i++) {
        int particle;
        double potential;
        force.getElectrodeAtomParameters(i, particle, potential);
        electrodeAtomIndices[i] = particle;
        targetPotentials[i] = potential;
    }

    vector<int> electrolyteAtomIndices(M);
    vector<double> fixedCharges(M);
    for (int i = 0; i < M; i++) {
        int particle;
        double charge;
        force.getElectrolyteAtomParameters(i, particle, charge);
        electrolyteAtomIndices[i] = particle;
        fixedCharges[i] = charge;
    }

    const vector<double>& invCapMatrix = force.getInverseCapacitanceMatrix();
    if (invCapMatrix.size() != N * N) {
        throw OpenMMException("Inverse capacitance matrix size mismatch");
    }

    // Allocate GPU arrays for static data
    d_electrodeAtomIndices = CudaArray::create<int>(cu, N, "electrodeAtomIndices");
    d_targetPotentials = CudaArray::create<double>(cu, N, "targetPotentials");
    d_electrolyteAtomIndices = CudaArray::create<int>(cu, M, "electrolyteAtomIndices");
    d_fixedCharges = CudaArray::create<double>(cu, M, "fixedCharges");
    d_invCapMatrix = CudaArray::create<double>(cu, N * N, "invCapMatrix");

    // Upload static data to GPU (once)
    d_electrodeAtomIndices->upload(electrodeAtomIndices);
    d_targetPotentials->upload(targetPotentials);
    d_electrolyteAtomIndices->upload(electrolyteAtomIndices);
    d_fixedCharges->upload(fixedCharges);
    d_invCapMatrix->upload(invCapMatrix);

    // Allocate intermediate buffers
    d_Ef = CudaArray::create<double>(cu, N, "Ef");
    d_b = CudaArray::create<double>(cu, N, "b");
    d_q_e = CudaArray::create<double>(cu, N, "q_e");

    // Find NonbondedForce
    for (int i = 0; i < system.getNumForces(); i++) {
        const NonbondedForce* nbForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
        if (nbForce != nullptr) {
            nonbondedForce = const_cast<NonbondedForce*>(nbForce);
            break;
        }
    }

    if (nonbondedForce == nullptr) {
        throw OpenMMException("ConstantVForce requires a NonbondedForce in the System");
    }
}

double CudaCalcConstantVKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // Get CUDA position array from context
    CUdeviceptr d_posq = cu.getPosq().getDevicePointer();

    // Block/grid dimensions for kernels
    int blockSize = 256;
    int numBlocks_N = (N + blockSize - 1) / blockSize;

    // ========================================================================
    // Step 1: Positions are already on GPU (d_posq)
    // ========================================================================

    // ========================================================================
    // Step 2: Calculate E_f[i] = Σ_j (k * q_f[j] / r_ij)
    // ========================================================================
    calculateEfKernel<<<numBlocks_N, blockSize>>>(
        N, M,
        (const int*)d_electrodeAtomIndices->getDevicePointer(),
        (const int*)d_electrolyteAtomIndices->getDevicePointer(),
        (const double*)d_fixedCharges->getDevicePointer(),
        (const float4*)d_posq,
        (double*)d_Ef->getDevicePointer()
    );

    // ========================================================================
    // Step 3: Calculate b = V - E_f
    // ========================================================================
    calculateBKernel<<<numBlocks_N, blockSize>>>(
        N,
        (const double*)d_targetPotentials->getDevicePointer(),
        (const double*)d_Ef->getDevicePointer(),
        (double*)d_b->getDevicePointer()
    );

    // ========================================================================
    // Step 4: Matrix multiply q_e = C_inv * b (using cuBLAS)
    // ========================================================================
    // cublasDgemv: y = alpha * A * x + beta * y
    // We want: q_e = 1.0 * C_inv * b + 0.0 * q_e
    const double alpha = 1.0;
    const double beta = 0.0;

    cublasStatus_t status = cublasDgemv(
        cublasHandle,
        CUBLAS_OP_N,  // No transpose
        N,            // Rows of C_inv
        N,            // Cols of C_inv
        &alpha,
        (const double*)d_invCapMatrix->getDevicePointer(),  // A (C_inv)
        N,            // Leading dimension of A
        (const double*)d_b->getDevicePointer(),             // x (b)
        1,            // Increment for x
        &beta,
        (double*)d_q_e->getDevicePointer(),                 // y (q_e)
        1             // Increment for y
    );

    if (status != CUBLAS_STATUS_SUCCESS) {
        throw OpenMMException("cuBLAS matrix-vector multiply failed");
    }

    // ========================================================================
    // Step 5: Update charges in posq array (DIRECTLY on GPU - no CPU transfer)
    // ========================================================================
    updateChargesKernel<<<numBlocks_N, blockSize>>>(
        N,
        (const int*)d_electrodeAtomIndices->getDevicePointer(),
        (const double*)d_q_e->getDevicePointer(),
        (float4*)d_posq
    );

    // Ensure kernel completes
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        throw OpenMMException(std::string("CUDA error in updateChargesKernel: ") + cudaGetErrorString(err));
    }

    // ========================================================================
    // Step 5: Download q_e and update NonbondedForce
    // ========================================================================
    // NOTE: This requires 1 download + 1 upload per timestep.
    // Original Python: 8 transfers (4 iterations × 2 transfers)
    // This CUDA version: 2 transfers (4× better)
    // All heavy computation (E_f, cuBLAS) stays on GPU.
    //
    // TODO: Investigate zero-transfer solution by directly accessing
    // NonbondedForce's internal CUDA charge array.

    vector<double> q_e_host(N);
    d_q_e->download(q_e_host);

    vector<int> electrodeAtomIndices_host(N);
    d_electrodeAtomIndices->download(electrodeAtomIndices_host);

    for (int i = 0; i < N; i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(electrodeAtomIndices_host[i], charge, sigma, epsilon);
        nonbondedForce->setParticleParameters(electrodeAtomIndices_host[i], q_e_host[i], sigma, epsilon);
    }

    nonbondedForce->updateParametersInContext(context.getOwner());

    return 0.0;  // No energy contribution from this force
}

void CudaCalcConstantVKernel::copyParametersToContext(ContextImpl& context, const ConstantVForce& force) {
    // Re-upload target potentials (in case user changed them)
    vector<double> targetPotentials(N);
    for (int i = 0; i < N; i++) {
        int particle;
        double potential;
        force.getElectrodeAtomParameters(i, particle, potential);
        targetPotentials[i] = potential;
    }
    d_targetPotentials->upload(targetPotentials);

    // Re-upload inverse capacitance matrix (in case user changed it)
    const vector<double>& invCapMatrix = force.getInverseCapacitanceMatrix();
    d_invCapMatrix->upload(invCapMatrix);
}
