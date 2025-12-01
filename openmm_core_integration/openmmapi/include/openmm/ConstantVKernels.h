#ifndef OPENMM_CONSTANTVKERNELS_H_
#define OPENMM_CONSTANTVKERNELS_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVKernels - Platform-specific kernel interfaces                     *
 * Copyright (c) 2025 - Present                                               *
 * -------------------------------------------------------------------------- */

#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include "openmm/Vec3.h"
#include <string>
#include <vector>

namespace OpenMM {

// Forward declarations
class ConstantVForce;
class ContextImpl;

/**
 * CalcConstantVKernel is the abstract interface for platform-specific
 * implementations of the constant voltage SCF solver.
 *
 * Each platform (CUDA, Reference, OpenCL) provides a derived class that
 * implements these methods using platform-specific code:
 *
 * - **CUDA**: GPU-accelerated SCF with warp-level reductions, template
 *             specialization, and kernel fusion
 * - **Reference**: CPU-based SCF with simple nested loops
 * - **OpenCL**: Cross-vendor GPU support with similar optimizations to CUDA
 *
 * Algorithm Overview (matching professor's Poisson_solver_fixed_voltage):
 *
 * 1. **Initial Charge Distribution**:
 *    - Cathode atoms start at zero charge
 *    - Anode atoms start at zero charge
 *    - Electrolyte atoms have fixed charges
 *
 * 2. **SCF Iteration Loop** (N iterations, typically 4):
 *    For each iteration:
 *    a. **Compute Electrode Potentials**:
 *       - phi_cathode = sum over all charges of (q_j / r_ij) for cathode atoms
 *       - phi_anode = sum over all charges of (q_j / r_ij) for anode atoms
 *       where r_ij = distance from atom i to charge j
 *
 *    b. **Apply Maxwell Boundary Conditions**:
 *       - Target: phi_cathode_avg = V_cathode
 *       - Target: phi_anode_avg = V_anode
 *
 *    c. **Compute Required Charge Deltas**:
 *       - dq_cathode = (V_cathode - phi_cathode_avg) * area_cathode * epsilon0
 *       - dq_anode = (V_anode - phi_anode_avg) * area_anode * epsilon0
 *
 *    d. **Enforce Green's Reciprocity** (Global Charge Conservation):
 *       - Q_total = sum(q_cathode) + sum(q_anode) + sum(q_electrolyte)
 *       - correction = -Q_total / (N_cathode + N_anode)
 *       - Apply correction uniformly to electrode atoms
 *
 *    e. **Update Electrode Charges**:
 *       - q_cathode[i] += dq_cathode / N_cathode + correction
 *       - q_anode[i] += dq_anode / N_anode + correction
 *
 * 3. **Force Calculation**:
 *    - Coulomb forces between all charged atoms
 *    - F_ij = k_e * q_i * q_j * r_ij / |r_ij|^3
 *
 * 4. **Energy Calculation**:
 *    - U = (1/2) * sum_ij (k_e * q_i * q_j / r_ij)
 *
 * Corresponds to: MM_classes.py::Poisson_solver_fixed_voltage (Lines 287-374)
 */
class CalcConstantVKernel : public KernelImpl {
public:
    CalcConstantVKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {}
    
    static std::string Name() {
        return "CalcConstantV";
    }

    /**
     * Initialize the kernel with system and electrode data.
     *
     * @param system                 OpenMM System object
     * @param cathodeAtomIndices     Indices of cathode atoms
     * @param cathodeAreas           Area per cathode atom (nm^2)
     * @param anodeAtomIndices       Indices of anode atoms
     * @param anodeAreas             Area per anode atom (nm^2)
     * @param electrolyteAtomIndices Indices of electrolyte atoms
     * @param electrolyteCharges     Fixed charges of electrolyte atoms (e)
     * @param voltage                Applied voltage (V)
     * @param Lgap                   Electrode gap (nm)
     * @param Lcell                  Cell size in Z (nm)
     * @param totalArea              Total electrode area (nm^2)
     * @param z_cathode              Cathode Z position (nm)
     * @param z_anode                Anode Z position (nm)
     * @param nIterations            Number of SCF iterations
     */
    virtual void initialize(const System& system,
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
                           int nIterations) = 0;

    /**
     * Add Buckyball conductor to the kernel.
     *
     * @param virtualAtomIndices  Indices of virtual atoms (image layer)
     * @param realAtomIndices     Indices of real atoms (physical conductor)
     * @param electrodeType       "cathode" or "anode"
     * @param voltage             Applied voltage (V)
     * @param center              Sphere center (nm)
     * @param radius              Sphere radius (nm)
     * @param normalVectors       Normal vectors for each virtual atom
     * @param areaPerAtom         Area per virtual atom (nm^2)
     * @param contactAtomIndex    Index of closest electrode atom
     * @param contactDistance     Distance to closest electrode atom (nm)
     */
    virtual void addBuckyballConductor(const std::vector<int>& virtualAtomIndices,
                                       const std::vector<int>& realAtomIndices,
                                       const std::string& electrodeType,
                                       double voltage,
                                       const Vec3& center,
                                       double radius,
                                       const std::vector<Vec3>& normalVectors,
                                       double areaPerAtom,
                                       int contactAtomIndex,
                                       double contactDistance) = 0;

    /**
     * Add Nanotube conductor to the kernel.
     *
     * @param virtualAtomIndices  Indices of virtual atoms (image layer)
     * @param realAtomIndices     Indices of real atoms (physical conductor)
     * @param electrodeType       "cathode" or "anode"
     * @param voltage             Applied voltage (V)
     * @param center              Nanotube center (nm)
     * @param axis                Nanotube axis (normalized)
     * @param radius              Nanotube radius (nm)
     * @param length              Nanotube length (nm)
     * @param normalVectors       Normal vectors for each virtual atom
     * @param areaPerAtom         Area per virtual atom (nm^2)
     * @param contactAtomIndex    Index of closest electrode atom
     * @param contactDistance     Distance to closest electrode atom (nm)
     */
    virtual void addNanotubeConductor(const std::vector<int>& virtualAtomIndices,
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
                                      double contactDistance) = 0;

    /**
     * Execute the SCF solver and calculate forces/energy.
     *
     * @param context        OpenMM context
     * @param includeForces  Whether to calculate forces
     * @param includeEnergy  Whether to calculate energy
     * @param groups         Force groups to include
     * @return               Total electrostatic energy (kJ/mol)
     */
    virtual double execute(ContextImpl& context, bool includeForces,
                          bool includeEnergy, int groups) = 0;

    /**
     * Update parameters in an existing context.
     * Called when Force parameters change.
     *
     * @param context  OpenMM context
     * @param force    Updated ConstantVForce object
     */
    virtual void updateParameters(ContextImpl& context, const ConstantVForce& force) = 0;
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVKERNELS_H_
