#ifndef CUDA_CONSTANTV_KERNELS_H_
#define CUDA_CONSTANTV_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                    CUDA Platform Implementation                            *
 * -------------------------------------------------------------------------- *
 * CUDA-specific kernel implementations for ConstantV integration            *
 * -------------------------------------------------------------------------- */

#include "openmm/ConstantVKernels.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include <vector>

namespace OpenMM {

/**
 * CUDA implementation of CalcConstantVKernel.
 *
 * This kernel wrapper manages GPU memory and calls the CUDA kernels
 * defined in constantVDrudeLangevin.cu.
 */
class CudaCalcConstantVKernel : public CalcConstantVKernel {
public:
    CudaCalcConstantVKernel(std::string name, const Platform& platform, CudaContext& cu);
    ~CudaCalcConstantVKernel();

    void initialize(const System& system,
                   const std::vector<int>& cathodeAtomIndices,
                   const std::vector<double>& cathodeAreas,
                   const std::vector<int>& anodeAtomIndices,
                   const std::vector<double>& anodeAreas,
                   const std::vector<int>& electrolyteAtomIndices,
                   const std::vector<double>& electrolyteCharges,
                   double voltage,
                   double Lgap,
                   double Lcell,
                   double totalArea,
                   double z_cathode,
                   double z_anode,
                   int nIterations);

    void addBuckyballConductor(const std::vector<int>& virtualAtomIndices,
                               const std::vector<int>& realAtomIndices,
                               const std::string& electrodeType,
                               double voltage,
                               const Vec3& center,
                               double radius,
                               const std::vector<Vec3>& normalVectors,
                               double areaPerAtom,
                               int contactAtomIndex,
                               double contactDistance);

    void addNanotubeConductor(const std::vector<int>& virtualAtomIndices,
                              const std::vector<int>& realAtomIndices,
                              const std::string& electrodeType,
                              double voltage,
                              const Vec3& center,
                              const Vec3& axis,
                              double radius,
                              double length,
                              const std::vector<Vec3>& normalVectors,
                              double areaPerAtom,
                              int contactAtomIndex,
                              double contactDistance);

    double execute(ContextImpl& context, bool includeForces,
                  bool includeEnergy, int groups);

    void updateParameters(ContextImpl& context, const ConstantVForce& force);

private:
    CudaContext& cu;
    bool hasInitialized;

    // GPU arrays for electrode data
    CudaArray* cathodeIndicesGPU;
    CudaArray* cathodeAreasGPU;
    CudaArray* anodeIndicesGPU;
    CudaArray* anodeAreasGPU;
    CudaArray* electrolyteIndicesGPU;

    // ElectrodeData struct on GPU (single struct)
    CudaArray* electrodeDataGPU;

    // Parameters
    int numCathodeAtoms;
    int numAnodeAtoms;
    int numElectrolyteAtoms;
    double voltage;
    double Lgap;
    double Lcell;
    double totalArea;
    double z_cathode;
    double z_anode;
    int nIterations;
};

/**
 * CUDA implementation of IntegrateConstantVDrudeLangevinStepKernel.
 *
 * This kernel integrates the equations of motion for Drude oscillators
 * with constant voltage boundary conditions.
 */
class CudaIntegrateConstantVDrudeLangevinStepKernel : public KernelImpl {
public:
    CudaIntegrateConstantVDrudeLangevinStepKernel(std::string name, const Platform& platform, CudaContext& cu);
    ~CudaIntegrateConstantVDrudeLangevinStepKernel();

    void initialize(const System& system, const ConstantVDrudeLangevinIntegrator& integrator);
    void execute(ContextImpl& context, const ConstantVDrudeLangevinIntegrator& integrator);

private:
    CudaContext& cu;
    bool hasInitialized;

    // GPU arrays for electrode data
    CudaArray* cathodeIndicesGPU;
    CudaArray* cathodeAreasGPU;
    CudaArray* anodeIndicesGPU;
    CudaArray* anodeAreasGPU;
    CudaArray* electrolyteIndicesGPU;

    // ElectrodeData struct on GPU
    CudaArray* electrodeDataGPU;

    // GPU arrays for Drude particle data
    CudaArray* pairParticlesGPU;      // int2 array
    CudaArray* normalParticlesGPU;    // int array
    CudaArray* drudeDataGPU;          // DrudeParticleData struct

    // Integration intermediate arrays
    CudaArray* posDeltaGPU;           // float4 array for position delta

    // Parameters
    int numCathodeAtoms;
    int numAnodeAtoms;
    int numElectrolyteAtoms;
    int numDrudePairs;
    int numNormalParticles;
    double voltage;
    double Lgap;
    double Lcell;
    double totalArea;
    double z_cathode;
    double z_anode;
    int scfIterations;
    int scfFrequency;
    int stepCount;
    double maxDrudeDistance;
};

} // namespace OpenMM

#endif // CUDA_CONSTANTV_KERNELS_H_
