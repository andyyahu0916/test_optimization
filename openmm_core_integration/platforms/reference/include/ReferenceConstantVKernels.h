#ifndef REFERENCE_CONSTANTV_KERNELS_H_
#define REFERENCE_CONSTANTV_KERNELS_H_

/* -------------------------------------------------------------------------- *
 *                  Reference Platform Implementation                         *
 * -------------------------------------------------------------------------- *
 * Reference (CPU) kernel implementations for ConstantV integration          *
 * -------------------------------------------------------------------------- */

#include "openmm/ConstantVKernels.h"
#include "openmm/Platform.h"
#include "openmm/internal/ContextImpl.h"
#include "ReferenceConstantVDrudeLangevinDynamics.h"
#include <vector>

namespace OpenMM {

/**
 * Reference (CPU) implementation of CalcConstantVKernel.
 */
class ReferenceCalcConstantVKernel : public CalcConstantVKernel {
public:
    ReferenceCalcConstantVKernel(std::string name, const Platform& platform);
    ~ReferenceCalcConstantVKernel();

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
    // Electrode data
    std::vector<int> cathodeIndices;
    std::vector<double> cathodeAreas;
    std::vector<int> anodeIndices;
    std::vector<double> anodeAreas;
    std::vector<int> electrolyteIndices;
    std::vector<double> electrolyteCharges;
    std::vector<double> cathodeCharges;
    std::vector<double> anodeCharges;

    // Parameters
    double voltage;
    double Lgap;
    double Lcell;
    double totalArea;
    double z_cathode;
    double z_anode;
    int nIterations;

    // Helper method to run SCF
    void runSCF(const std::vector<Vec3>& positions);
};

/**
 * Reference implementation of IntegrateConstantVDrudeLangevinStepKernel.
 */
class ReferenceIntegrateConstantVDrudeLangevinStepKernel : public KernelImpl {
public:
    ReferenceIntegrateConstantVDrudeLangevinStepKernel(std::string name, const Platform& platform);
    ~ReferenceIntegrateConstantVDrudeLangevinStepKernel();

    void initialize(const System& system, const ConstantVDrudeLangevinIntegrator& integrator);
    void execute(ContextImpl& context, const ConstantVDrudeLangevinIntegrator& integrator);

private:
    ReferenceConstantVDrudeLangevinDynamics* dynamics;
    int stepCount;
};

} // namespace OpenMM

#endif // REFERENCE_CONSTANTV_KERNELS_H_
