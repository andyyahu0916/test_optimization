#ifndef CUDA_CONSTANTV_KERNELS_H_
#define CUDA_CONSTANTV_KERNELS_H_

#include "ConstantVKernels.h"
#include "openmm/NonbondedForce.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>

namespace ConstantVPlugin {

/**
 * CUDA implementation of CalcConstantVKernel.
 *
 * Zero-transfer architecture: Direct translation of Reference platform SCF iteration.
 * All data stays on GPU, only transfer 4 doubles per iteration for Green's Reciprocity.
 *
 * Physics: Maxwell boundary conditions + Green's Reciprocity Theorem
 * Based on professor's original Python code (John Pople lineage, ab initio/first principles)
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
     * Execute the kernel - computes electrode charges via SCF iteration and updates NonbondedForce.
     */
    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy);

    /**
     * Copy changed parameters over to a context.
     */
    void copyParametersToContext(OpenMM::ContextImpl& context, const ConstantVForce& force);

private:
    OpenMM::CudaContext& cu;

    // System dimensions
    int numCathodes;      // Number of cathode atoms
    int numAnodes;        // Number of anode atoms
    int numElectrolytes;  // Number of electrolyte atoms

    // Physical parameters
    double voltage;       // Applied voltage (V)
    double Lgap;          // Gap between electrodes (nm)
    double Lcell;         // Cell size in Z direction (nm)
    double totalArea;     // Total electrode area (nm^2)
    double z_cathode;     // Cathode Z position (nm)
    double z_anode;       // Anode Z position (nm)
    int nIterations;      // Number of SCF iterations

    // GPU arrays - electrode data (immutable during simulation)
    OpenMM::CudaArray* d_cathodeIndices;    // [numCathodes] - cathode particle indices
    OpenMM::CudaArray* d_cathodeAreas;      // [numCathodes] - area per cathode atom (nm^2)
    OpenMM::CudaArray* d_anodeIndices;      // [numAnodes] - anode particle indices
    OpenMM::CudaArray* d_anodeAreas;        // [numAnodes] - area per anode atom (nm^2)
    OpenMM::CudaArray* d_electrolyteIndices; // [numElectrolytes] - electrolyte particle indices

    // GPU arrays - Green's Reciprocity (analytic charge calculation)
    OpenMM::CudaArray* d_Q_analytic_cathode;  // [1] - analytic total charge for cathode
    OpenMM::CudaArray* d_Q_analytic_anode;    // [1] - analytic total charge for anode
    OpenMM::CudaArray* d_Q_numeric_cathode;   // [1] - numeric total charge for cathode
    OpenMM::CudaArray* d_Q_numeric_anode;     // [1] - numeric total charge for anode

    // GPU arrays - parallel reduction buffers
    OpenMM::CudaArray* d_cathode_partial;     // [numBlocks] - partial sums for cathode image charge
    OpenMM::CudaArray* d_anode_partial;       // [numBlocks] - partial sums for anode image charge
    OpenMM::CudaArray* d_cathode_numeric_partial; // [numBlocks] - partial sums for cathode numeric charge
    OpenMM::CudaArray* d_anode_numeric_partial;   // [numBlocks] - partial sums for anode numeric charge

    // Pointer to NonbondedForce for charge updates
    OpenMM::NonbondedForce* nonbondedForce;

    // Level 2 Optimization: Texture object for posq array
    cudaTextureObject_t posqTexture;

    // Level 3 Optimization: CUDA Graph for SCF loop
    cudaGraph_t scfGraph;
    cudaGraphExec_t scfGraphExec;

    // Lazy initialization flag
    bool gpuInitialized;

    // CPU-side atom data (stored during initialize(), used in initializeGPU())
    std::vector<int> cathodeIndices;
    std::vector<double> cathodeAreas;
    std::vector<int> anodeIndices;
    std::vector<double> anodeAreas;
    std::vector<int> electrolyteIndices;

private:
    void initializeGPU();  // Defer GPU allocation to first execute()
};

} // namespace ConstantVPlugin

#endif /*CUDA_CONSTANTV_KERNELS_H_*/
