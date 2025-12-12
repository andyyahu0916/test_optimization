#ifndef CONSTANTV_DRUDELANGEVIN_INTEGRATOR_H_
#define CONSTANTV_DRUDELANGEVIN_INTEGRATOR_H_

/* -------------------------------------------------------------------------- *
 *                          ConstantV OpenMM Plugin                           *
 * -------------------------------------------------------------------------- *
 * This is an OpenMM plugin for constant voltage simulations with Drude       *
 * polarizable force fields.                                                  *
 *                                                                            *
 * Copyright (c) 2025 - Present                                               *
 * Authors: Based on OpenMM's DrudeLangevinIntegrator                         *
 * -------------------------------------------------------------------------- */

#include "ConstantVForce.h"
#include "openmm/DrudeIntegrator.h"
#include "openmm/DrudeLangevinIntegrator.h"
#include "openmm/Kernel.h"
#include <vector>

namespace ConstantVPlugin {

/**
 * ConstantVDrudeLangevinIntegrator implements dual-temperature Langevin dynamics
 * for polarizable Drude oscillators under constant voltage conditions.
 *
 * This integrator combines:
 * - OpenMM's DrudeLangevinIntegrator: Dual-temperature Langevin thermostat
 *   (typically 300K for real atoms, 1K for Drude particles)
 * - ConstantVPlugin's SCF solver: Electrode charge updates via Maxwell boundary
 *   conditions and Green's Reciprocity theorem
 *
 * Physical Rationale:
 * - Drude particles represent electronic polarization via charged virtual sites
 *   connected to parent atoms by harmonic springs
 * - Without proper thermostating, Drude particles can "fly away" (polarization
 *   catastrophe) due to numerical instability
 * - Dual-temperature Langevin applies strong damping (1K, high friction) to
 *   Drude internal degrees of freedom while allowing realistic dynamics (300K,
 *   normal friction) for center-of-mass motion
 * - Electrode charges must be updated self-consistently as atomic positions
 *   change, requiring periodic SCF iterations
 *
 * Design Philosophy:
 * - **Composition over Inheritance**: Reuses OpenMM's IntegrateDrudeLangevinStepKernel
 *   (100% validated, platform-optimized) rather than reimplementing Drude integration
 * - **Separation of Concerns**: SCF solver (CalcConstantVKernel) handles charge
 *   updates; Drude Langevin kernel handles integration
 * - **Zero Physics Reinvention**: When voltage = 0V, behavior is identical to
 *   standard DrudeLangevinIntegrator
 *
 * Example Usage:
 * \code
 * ConstantVDrudeLangevinIntegrator integrator(
 *     300.0,  // temperature (K) - real atoms
 *     1.0,    // friction (1/ps) - real atoms
 *     1.0,    // drudeTemperature (K) - Drude particles (CRITICAL: must be low!)
 *     20.0,   // drudeFriction (1/ps) - Drude particles (high friction for stability)
 *     0.001   // timestep (ps)
 * );
 *
 * integrator.setVoltage(1.0);  // 1.0 V
 * integrator.setNumSCFIterations(4);
 * integrator.setSCFFrequency(1);  // Update charges every MD step
 *
 * // Add electrode atoms (same API as ConstantVIntegrator)
 * for (int i : cathode_atoms)
 *     integrator.addCathodeAtom(i, area_per_atom);
 * for (int i : anode_atoms)
 *     integrator.addAnodeAtom(i, area_per_atom);
 * for (int i : electrolyte_atoms)
 *     integrator.addElectrolyteAtom(i, charge);
 *
 * // Set geometry (auto-configured via Python helpers)
 * integrator.setLgap(gap);
 * integrator.setLcell(cell);
 * integrator.setTotalArea(area);
 * integrator.setZCathode(z_cath);
 * integrator.setZAnode(z_anod);
 *
 * Context context(system, integrator);
 * integrator.step(1000000);  // 1 ns simulation
 * \endcode
 */
class ConstantVDrudeLangevinIntegrator : public OpenMM::DrudeIntegrator {
public:
    /**
     * Create a ConstantVDrudeLangevinIntegrator.
     *
     * @param temperature          temperature of the main heat bath applied to
     *                             center-of-mass motion (K)
     * @param frictionCoeff        friction coefficient for center-of-mass motion (1/ps)
     * @param drudeTemperature     temperature of the heat bath applied to internal
     *                             coordinates of Drude particles (K). Typically 1K
     *                             to prevent polarization catastrophe.
     * @param drudeFrictionCoeff   friction coefficient for Drude internal coordinates (1/ps).
     *                             Typically 20.0 (much higher than real atoms) for
     *                             strong damping.
     * @param stepSize             integration timestep (ps)
     */
    ConstantVDrudeLangevinIntegrator(
        double temperature,
        double frictionCoeff,
        double drudeTemperature,
        double drudeFrictionCoeff,
        double stepSize
    );

    /**
     * Destructor.
     * Cleans up internal DrudeLangevinIntegrator delegate to prevent memory leak.
     */
    ~ConstantVDrudeLangevinIntegrator();

    // ═══════════════════════════════════════════════════════════
    // Langevin Parameters (from DrudeLangevinIntegrator)
    // ═══════════════════════════════════════════════════════════

    /**
     * Get the temperature of the main heat bath (K).
     */
    double getTemperature() const {
        return temperature;
    }

    /**
     * Set the temperature of the main heat bath (K).
     */
    void setTemperature(double temp);

    /**
     * Get the friction coefficient for center-of-mass motion (1/ps).
     */
    double getFriction() const {
        return friction;
    }

    /**
     * Set the friction coefficient for center-of-mass motion (1/ps).
     */
    void setFriction(double coeff);

    /**
     * Get the friction coefficient for Drude internal coordinates (1/ps).
     */
    double getDrudeFriction() const {
        return drudeFriction;
    }

    /**
     * Set the friction coefficient for Drude internal coordinates (1/ps).
     */
    void setDrudeFriction(double coeff);

    // ═══════════════════════════════════════════════════════════
    // Constant Voltage Parameters
    // ═══════════════════════════════════════════════════════════

    /**
     * Get the applied voltage (V).
     */
    double getVoltage() const {
        return voltage;
    }

    /**
     * Set the applied voltage (V).
     */
    void setVoltage(double v) {
        voltage = v;
    }

    /**
     * Get the number of SCF iterations per charge update.
     */
    int getNumSCFIterations() const {
        return nIterations;
    }

    /**
     * Set the number of SCF iterations per charge update.
     * Must be >= 1.
     */
    void setNumSCFIterations(int n);

    /**
     * Get the SCF frequency (charge updates every N MD steps).
     */
    int getSCFFrequency() const {
        return scfFrequency;
    }

    /**
     * Set the SCF frequency (charge updates every N MD steps).
     * Must be >= 1. Default is 1 (update every step).
     */
    void setSCFFrequency(int freq);

    // ═══════════════════════════════════════════════════════════
    // Electrode Atom Management
    // ═══════════════════════════════════════════════════════════

    /**
     * Add a cathode atom.
     *
     * @param particle   particle index
     * @param area       area per atom (nm^2)
     */
    void addCathodeAtom(int particle, double area);

    /**
     * Add an anode atom.
     *
     * @param particle   particle index
     * @param area       area per atom (nm^2)
     */
    void addAnodeAtom(int particle, double area);

    /**
     * Add an electrolyte atom.
     *
     * @param particle   particle index
     * @param charge     initial charge (elementary charge units)
     */
    void addElectrolyteAtom(int particle, double charge);

    /**
     * Get the number of cathode atoms.
     */
    int getNumCathodeAtoms() const {
        return cathodeAtomIndices.size();
    }

    /**
     * Get the number of anode atoms.
     */
    int getNumAnodeAtoms() const {
        return anodeAtomIndices.size();
    }

    /**
     * Get the number of electrolyte atoms.
     */
    int getNumElectrolyteAtoms() const {
        return electrolyteAtomIndices.size();
    }

    // ═══════════════════════════════════════════════════════════
    // Geometry Parameters
    // ═══════════════════════════════════════════════════════════

    /**
     * Set the gap between electrodes (nm).
     */
    void setLgap(double gap) {
        Lgap = gap;
    }

    /**
     * Set the cell size in Z direction (nm).
     */
    void setLcell(double cell) {
        Lcell = cell;
    }

    /**
     * Set the total electrode area (nm^2).
     */
    void setTotalArea(double area) {
        totalArea = area;
    }

    /**
     * Set the cathode Z position (nm).
     */
    void setZCathode(double z) {
        z_cathode = z;
    }

    /**
     * Set the anode Z position (nm).
     */
    void setZAnode(double z) {
        z_anode = z;
    }

    /**
     * Get the gap between electrodes (nm).
     */
    double getLgap() const {
        return Lgap;
    }

    /**
     * Get the cell size in Z direction (nm).
     */
    double getLcell() const {
        return Lcell;
    }

    /**
     * Get the total electrode area (nm^2).
     */
    double getTotalArea() const {
        return totalArea;
    }

    /**
     * Get the cathode Z position (nm).
     */
    double getZCathode() const {
        return z_cathode;
    }

    /**
     * Get the anode Z position (nm).
     */
    double getZAnode() const {
        return z_anode;
    }

    // ═══════════════════════════════════════════════════════════
    // Internal Data Access (for kernel initialization)
    // ═══════════════════════════════════════════════════════════

    /**
     * Get cathode atom indices (read-only).
     */
    const std::vector<int>& getCathodeAtomIndices() const {
        return cathodeAtomIndices;
    }

    /**
     * Get cathode areas (read-only).
     */
    const std::vector<double>& getCathodeAreas() const {
        return cathodeAreas;
    }

    /**
     * Get anode atom indices (read-only).
     */
    const std::vector<int>& getAnodeAtomIndices() const {
        return anodeAtomIndices;
    }

    /**
     * Get anode areas (read-only).
     */
    const std::vector<double>& getAnodeAreas() const {
        return anodeAreas;
    }

    /**
     * Get electrolyte atom indices (read-only).
     */
    const std::vector<int>& getElectrolyteAtomIndices() const {
        return electrolyteAtomIndices;
    }

    /**
     * Get electrolyte charges (read-only).
     */
    const std::vector<double>& getElectrolyteCharges() const {
        return electrolyteCharges;
    }

    // ═══════════════════════════════════════════════════════════
    // Integrator Interface (required by OpenMM::Integrator)
    // ═══════════════════════════════════════════════════════════

    /**
     * Advance the simulation through time by taking a series of time steps.
     *
     * @param steps   the number of time steps to take
     */
    void step(int steps) override;

    /**
     * Compute the total kinetic energy of the system.
     *
     * @return the kinetic energy (kJ/mol)
     */
    double computeKineticEnergy() override;

protected:
    /**
     * Initialize the integrator.
     * Called by Context when it is created or reinitialized.
     */
    void initialize(OpenMM::ContextImpl& context) override;

    /**
     * Clean up the integrator.
     * Called by Context when it is destroyed or reinitialized.
     */
    void cleanup() override;

    /**
     * Get the names of all kernels used by this integrator.
     */
    std::vector<std::string> getKernelNames() override;

private:
    // Langevin parameters
    double temperature;
    double friction;
    double drudeFriction;

    // Constant voltage parameters
    double voltage;
    int nIterations;      // SCF iterations per charge update
    int scfFrequency;     // Update charges every N MD steps

    // Geometry parameters
    double Lgap, Lcell, totalArea, z_cathode, z_anode;

    // Electrode atom data
    std::vector<int> cathodeAtomIndices;
    std::vector<double> cathodeAreas;
    std::vector<int> anodeAtomIndices;
    std::vector<double> anodeAreas;
    std::vector<int> electrolyteAtomIndices;
    std::vector<double> electrolyteCharges;

    // Kernels
    OpenMM::Kernel drudeLangevinKernel;  // OpenMM's IntegrateDrudeLangevinStepKernel
    OpenMM::Kernel calcConstantVKernel;  // ConstantVPlugin's CalcConstantVKernel

    // Internal state
    int stepCount;  // Track MD steps for SCF frequency

    // Internal DrudeLangevinIntegrator for kernel delegation
    // The Drude Langevin kernel requires a DrudeLangevinIntegrator& reference,
    // so we maintain an internal instance that we keep in sync with our parameters
    OpenMM::DrudeLangevinIntegrator* drudeLangevinDelegate;
};

} // namespace ConstantVPlugin

#endif /*CONSTANTV_DRUDELANGEVIN_INTEGRATOR_H_*/
