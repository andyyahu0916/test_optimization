#ifndef CUDA_CONSTANTV_KERNELS_H_
#define CUDA_CONSTANTV_KERNELS_H_

#include "ConstantVKernels.h"
#include "openmm/NonbondedForce.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>
#include <string>

namespace ConstantVPlugin {

/**
 * Struct to hold Buckyball/Nanotube data for CUDA.
 * We calculate static geometry (normals, areas) on CPU during initialization
 * and upload to GPU arrays.
 */
struct CudaConductorData {
    std::string electrodeType; // "cathode" or "anode"
    double voltage;            // kJ/mol
    double closeThreshold;     // nm
    bool closeToElectrode;     // Determined at init
    int contactAtomIndex;      // Determined at init
    double dr_center_contact;  // Determined at init

    // For Nanotubes
    double length;
    std::vector<double> axis;

    // Geometry parameters
    double radius;
    double area_atom;
    std::vector<double> r_center; // [3]

    // GPU Arrays (Owned by this struct, must be deleted)
    OpenMM::CudaArray* d_virtualAtomIndices; // [Natoms] int
    OpenMM::CudaArray* d_realAtomIndices;    // [Natoms] int
    OpenMM::CudaArray* d_normals;            // [3*Natoms] double (nx, ny, nz interleaved)

    CudaConductorData() : d_virtualAtomIndices(nullptr), d_realAtomIndices(nullptr), d_normals(nullptr) {}

    ~CudaConductorData() {
        delete d_virtualAtomIndices;
        delete d_realAtomIndices;
        delete d_normals;
    }
};

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

    // GPU arrays - SCF iteration working data (rewritten each iteration)
    OpenMM::CudaArray* d_Ez_cathode;        // [numCathodes] - Ez_external for cathode atoms
    OpenMM::CudaArray* d_Ez_anode;          // [numAnodes] - Ez_external for anode atoms

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

    // Buffer for Step 2 Charge Transfer (Zero Transfer Architecture)
    // WARNING: This buffer size is 1.
    // All kernels writing to this buffer MUST be serialized in the same CUDA stream.
    // Do not use concurrent streams for different conductors without expanding this buffer.
    OpenMM::CudaArray* d_contactForceBuffer;

    // Pointer to NonbondedForce for charge updates
    OpenMM::NonbondedForce* nonbondedForce;

    // Lazy initialization flag
    bool gpuInitialized;

    // CPU-side atom data (stored during initialize(), used in initializeGPU())
    std::vector<int> cathodeIndices;
    std::vector<double> cathodeAreas;
    std::vector<int> anodeIndices;
    std::vector<double> anodeAreas;
    std::vector<int> electrolyteIndices;

    // Complex Conductors (Buckyballs/Nanotubes)
    // We store pointers to allow CudaConductorData to manage its own CudaArrays
    std::vector<CudaConductorData*> buckyballs;
    std::vector<CudaConductorData*> nanotubes;

    // Temporary storage for initialization (cleared after init)
    struct InitConductorData {
        std::vector<int> virtualAtoms;
        std::vector<int> realAtoms;
        std::string electrodeType;
        double voltage;
        std::vector<double> axis; // For nanotubes
    };
    std::vector<InitConductorData> buckyballInitData;
    std::vector<InitConductorData> nanotubeInitData;
    std::vector<int> virtualSiteIndices; // Cached list of all virtual-site atoms (flat + conductors)
    bool virtualLJApplied;

private:
    void initializeGPU();  // Defer GPU allocation to first execute()
    void enforceVirtualSiteParameters(OpenMM::ContextImpl& context);

    // Helper methods for conductor initialization (CPU side logic)
    void initializeBuckyballGeometry(CudaConductorData* conductor, const std::vector<OpenMM::Vec3>& positions);
    void initializeNanotubeGeometry(CudaConductorData* conductor, const std::vector<OpenMM::Vec3>& positions, const OpenMM::Vec3 boxVectors[3]);
    void findContactNeighborConductor(CudaConductorData* conductor, const std::vector<OpenMM::Vec3>& positions);
    void projectOrthogonalToAxis(const double vec_in[3], const double axis[3], double vec_out[3]);
};

/**
 * CUDA implementation of IntegrateConstantVStepKernel.
 * Coordinates SCF iteration + force calculation + Verlet integration.
 */
class CudaIntegrateConstantVStepKernel : public IntegrateConstantVStepKernel {
public:
    CudaIntegrateConstantVStepKernel(std::string name, const OpenMM::Platform& platform, OpenMM::CudaContext& cu);

    void initialize(const OpenMM::System& system, const ConstantVIntegrator& integrator) override;

    void execute(OpenMM::ContextImpl& context, const ConstantVIntegrator& integrator) override;

    double computeKineticEnergy(OpenMM::ContextImpl& context, const ConstantVIntegrator& integrator) override;

private:
    OpenMM::CudaContext& cu;
    int scf_frequency;
    double prevStepSize;
    OpenMM::Kernel calcConstantVKernel;  // SCF iteration kernel
    bool kernelInitialized;  // Track whether calcConstantVKernel has been initialized
};

} // namespace ConstantVPlugin

#endif /*CUDA_CONSTANTV_KERNELS_H_*/
