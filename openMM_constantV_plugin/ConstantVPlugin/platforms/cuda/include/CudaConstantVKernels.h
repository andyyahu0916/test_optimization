#ifndef CUDA_CONSTANTV_KERNELS_H_
#define CUDA_CONSTANTV_KERNELS_H_

#include "ConstantVKernels.h"
#include "openmm/NonbondedForce.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <cublas_v2.h>
#include <vector>

namespace ConstantVPlugin {

/**
 * CUDA implementation of CalcConstantVKernel.
 *
 * Single-pass algorithm: q_e = C_inv * (V - E_f)
 * where E_f is the electric field from electrolyte atoms.
 */
class CudaCalcConstantVKernel : public CalcConstantVKernel {
public:
    CudaCalcConstantVKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu);
    ~CudaCalcConstantVKernel();

    /**
     * Initialize the kernel.
     */
    void initialize(const OpenMM::System& system, const ConstantVForce& force);

    /**
     * Execute the kernel - computes electrode charges and updates NonbondedForce.
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

    /**
     * Copy changed parameters over to a context.
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const ConstantVForce& force);

private:
    OpenMM::CudaContext& cu;
    cublasHandle_t cublasHandle;

    // System dimensions
    int N;  // number of electrode atoms
    int M;  // number of electrolyte atoms

    // GPU arrays - static data (allocated in initialize(), immutable during simulation)
    OpenMM::CudaArray* d_electrodeAtomIndices;    // [N] - electrode particle indices
    OpenMM::CudaArray* d_targetPotentials;        // [N] - target potentials (kJ/mol)
    OpenMM::CudaArray* d_electrolyteAtomIndices;  // [M] - electrolyte particle indices
    OpenMM::CudaArray* d_fixedCharges;            // [M] - fixed charges (e)
    OpenMM::CudaArray* d_invCapMatrix;            // [N*N] - inverse capacitance matrix

    // GPU arrays - intermediate buffers (allocated in initialize(), rewritten each step)
    OpenMM::CudaArray* d_Ef;   // [N] - electric field at electrodes from electrolyte
    OpenMM::CudaArray* d_b;    // [N] - V - E_f
    OpenMM::CudaArray* d_q_e;  // [N] - computed electrode charges

    // Pointer to NonbondedForce for charge updates
    OpenMM::NonbondedForce* nonbondedForce;
};

} // namespace ConstantVPlugin

#endif /*CUDA_CONSTANTV_KERNELS_H_*/
