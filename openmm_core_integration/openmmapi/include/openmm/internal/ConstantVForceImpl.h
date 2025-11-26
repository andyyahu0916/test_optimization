#ifndef OPENMM_CONSTANTVFORCEIMPL_H_
#define OPENMM_CONSTANTVFORCEIMPL_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVForceImpl - Internal implementation of ConstantVForce             *
 * Copyright (c) 2025 - Present                                               *
 * -------------------------------------------------------------------------- */

#include "openmm/ConstantVForce.h"
#include "openmm/internal/ForceImpl.h"
#include "openmm/Kernel.h"
#include <map>
#include <string>

namespace OpenMM {

/**
 * ConstantVForceImpl is the internal implementation of ConstantVForce.
 *
 * This class bridges between the public Force API and platform-specific kernels.
 * It handles:
 * - Initialization of conductor geometry (Buckyball/Nanotube)
 * - Creation and execution of platform-specific SCF kernels
 * - Parameter updates in existing contexts
 * - Force and energy calculations
 *
 * Geometry Initialization Pipeline:
 * 1. For each Buckyball conductor:
 *    - Gather virtual atom positions
 *    - Compute center (average position)
 *    - Compute radius (average distance from center)
 *    - Compute normal vectors (radial directions)
 *    - Compute area per atom (4πr²/N)
 *    - Find contact electrode atom (nearest to center)
 *
 * 2. For each Nanotube conductor:
 *    - Gather virtual atom positions
 *    - Compute center (average position)
 *    - Project positions onto plane perpendicular to axis
 *    - Compute radius (average radial distance)
 *    - Compute normal vectors (radial, perpendicular to axis)
 *    - Compute length (from box vectors)
 *    - Compute area per atom (2πrL/N)
 *    - Find contact electrode atom (nearest to center)
 *
 * 3. Pass all geometry data to platform-specific kernel
 */
class OPENMM_EXPORT ConstantVForceImpl : public ForceImpl {
public:
    ConstantVForceImpl(const ConstantVForce& owner);

    ~ConstantVForceImpl();

    /**
     * Initialize the implementation.
     * Called once when the Context is created.
     */
    void initialize(ContextImpl& context);

    /**
     * Get the Force object this implementation belongs to.
     */
    const ConstantVForce& getOwner() const {
        return owner;
    }

    /**
     * Calculate forces and energy.
     * Called every time step by the Context.
     *
     * This method:
     * 1. Gets current atomic positions from context
     * 2. Updates conductor geometry if needed
     * 3. Invokes platform-specific SCF kernel
     * 4. Applies electrode forces based on updated charges
     * 5. Returns total electrostatic energy
     */
    void calcForce(ContextImpl& context, const std::vector<Vec3>& positions,
                   std::vector<Vec3>& forces);

    /**
     * Get the total energy (kJ/mol).
     * Called by calcForce() after kernel execution.
     */
    double calcForcesAndEnergy(ContextImpl& context, bool includeForces,
                               bool includeEnergy, int groups);

    /**
     * Get all force field parameters as a map.
     * Used for serialization and state comparison.
     */
    std::map<std::string, double> getDefaultParameters();

    /**
     * Get force field parameter names.
     */
    std::vector<std::string> getKernelNames();

    /**
     * Update parameters in an existing context.
     * Called when user modifies Force parameters after Context creation.
     *
     * This triggers:
     * - Recomputation of conductor geometry
     * - Update of kernel-side electrode data
     * - Reinitialize SCF solver state
     */
    void updateParametersInContext(ContextImpl& context);

private:
    const ConstantVForce& owner;  // Reference to public Force object
    Kernel kernel;                 // Platform-specific CalcConstantVKernel

    /**
     * Initialize Buckyball conductor geometry.
     * Called during initialize() for each Buckyball.
     *
     * @param context          OpenMM context
     * @param conductorIndex   Index in owner's buckyballConductors vector
     * @param positions        Current atomic positions
     */
    void initializeBuckyballGeometry(ContextImpl& context,
                                     int conductorIndex,
                                     const std::vector<Vec3>& positions);

    /**
     * Initialize Nanotube conductor geometry.
     * Called during initialize() for each Nanotube.
     *
     * @param context          OpenMM context
     * @param conductorIndex   Index in owner's nanotubeConductors vector
     * @param positions        Current atomic positions
     */
    void initializeNanotubeGeometry(ContextImpl& context,
                                    int conductorIndex,
                                    const std::vector<Vec3>& positions);

    /**
     * Update conductor geometry based on current positions.
     * Called during updateParametersInContext() or when atoms move significantly.
     *
     * @param context   OpenMM context
     */
    void updateConductorGeometry(ContextImpl& context);
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVFORCEIMPL_H_
