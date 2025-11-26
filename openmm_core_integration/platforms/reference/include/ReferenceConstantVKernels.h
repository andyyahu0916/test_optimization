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
    // Conductor data structure
    struct BuckyballConductor {
        std::vector<int> virtualIndices;
        std::vector<int> realIndices;
        std::vector<Vec3> normals;
        double areaPerAtom;
        double radius;
        Vec3 center;
        int contactAtomIndex;
        double contactDistance;
        double voltage_kjmol;
        char electrodeType;  // 'c' or 'a'
        std::vector<double> charges;  // Electrode charges
    };

    struct NanotubeConductor {
        std::vector<int> virtualIndices;
        std::vector<int> realIndices;
        std::vector<Vec3> normals;
        double areaPerAtom;
        Vec3 axis;
        Vec3 center;
        int contactAtomIndex;
        double contactDistance;
        double voltage_kjmol;
        char electrodeType;  // 'c' or 'a'
        std::vector<double> charges;  // Electrode charges
    };

    // Electrode data
    std::vector<int> cathodeIndices;
    std::vector<double> cathodeAreas;
    std::vector<int> anodeIndices;
    std::vector<double> anodeAreas;
    std::vector<int> electrolyteIndices;
    std::vector<double> electrolyteCharges;
    std::vector<double> cathodeCharges;
    std::vector<double> anodeCharges;

    // Conductor data
    std::vector<BuckyballConductor> buckyballs;
    std::vector<NanotubeConductor> nanotubes;

    // Parameters
    double voltage;
    double Lgap;
    double Lcell;
    double totalArea;
    double z_cathode;
    double z_anode;
    int nIterations;

    // ═══════════════════════════════════════════════════════════════════════
    // SCF Helper Methods (E-field method matching Python/CUDA)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Run SCF iteration using E-field method.
     * @param positions  Current atom positions
     * @param forces     Current forces (needed for Ez = F_z / q calculation)
     */
    void runSCF(const std::vector<Vec3>& positions, const std::vector<Vec3>& forces);

    /**
     * Legacy runSCF without forces (backward compatible, uses zero forces).
     */
    void runSCF(const std::vector<Vec3>& positions);

    /**
     * Compute Q_analytic using Green's Reciprocity.
     * Includes image charge contributions from electrolyte.
     * @param positions   Current atom positions
     * @param isCathode   True for cathode, false for anode
     * @return Q_analytic
     */
    double computeAnalyticCharge(const std::vector<Vec3>& positions, bool isCathode) const;

    /**
     * Scale electrode charges to match analytic normalization.
     * @param charges     Electrode charges (modified in place)
     * @param Q_analytic  Target analytic charge sum
     */
    void scaleChargesAnalytic(std::vector<double>& charges, double Q_analytic) const;
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
