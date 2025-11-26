#ifndef OPENMM_CONSTANTVINTEGRATOR_H_
#define OPENMM_CONSTANTVINTEGRATOR_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVIntegrator - Velocity Verlet with constant voltage                *
 * Copyright (c) 2025 - Present                                               *
 * -------------------------------------------------------------------------- */

#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include <vector>

namespace OpenMM {

/**
 * ConstantVIntegrator implements velocity Verlet integration with periodic
 * self-consistent field (SCF) charge updates for constant voltage boundary
 * conditions.
 *
 * Algorithm (每个时间步):
 * 1. v(t+Δt/2) = v(t) + (F(t)/m) * Δt/2
 * 2. x(t+Δt) = x(t) + v(t+Δt/2) * Δt
 * 3. IF (step % scf_frequency == 0):
 *    - Run SCF iterations to update electrode charges
 * 4. Compute F(t+Δt) using updated charges
 * 5. v(t+Δt) = v(t+Δt/2) + (F(t+Δt)/m) * Δt/2
 *
 * This is the simplest integrator for constant voltage simulations, suitable for:
 * - NVE ensemble (microcanonical)
 * - Rigid water models (no need for thermostats)
 * - Testing and validation (deterministic, reversible)
 *
 * For polarizable force fields (Drude oscillators), use ConstantVDrudeLangevinIntegrator instead.
 *
 * Example Usage:
 * \code
 * ConstantVIntegrator integrator(0.001);  // 1 fs timestep
 *
 * // Set voltage and geometry
 * integrator.setVoltage(1.0);             // 1.0 V
 * integrator.setLgap(gap);
 * integrator.setLcell(cell);
 * integrator.setTotalArea(area);
 * integrator.setZCathode(z_cath);
 * integrator.setZAnode(z_anod);
 *
 * // Add electrode atoms
 * for (int i : cathode_atoms)
 *     integrator.addCathodeAtom(i, area_per_atom);
 * for (int i : anode_atoms)
 *     integrator.addAnodeAtom(i, area_per_atom);
 * for (int i : electrolyte_atoms)
 *     integrator.addElectrolyteAtom(i, charge);
 *
 * // Set SCF parameters
 * integrator.setNumSCFIterations(4);      // 4 iterations per SCF update
 * integrator.setSCFFrequency(1);          // Update every MD step
 *
 * Context context(system, integrator);
 * integrator.step(1000000);               // 1 ns simulation
 * \endcode
 */
class OPENMM_EXPORT ConstantVIntegrator : public Integrator {
public:
    /**
     * Create a ConstantVIntegrator.
     *
     * @param stepSize   integration timestep (ps)
     */
    explicit ConstantVIntegrator(double stepSize);

    /**
     * Destructor.
     */
    ~ConstantVIntegrator();

    // ═══════════════════════════════════════════════════════════
    // Physical Parameters
    // ═══════════════════════════════════════════════════════════

    /**
     * Get the applied voltage (V).
     */
    double getVoltage() const {
        return voltageVolts;
    }

    /**
     * Set the applied voltage (V).
     */
    void setVoltage(double voltage);

    /**
     * Get the electrode gap (nm).
     */
    double getLgap() const {
        return Lgap;
    }

    /**
     * Set the electrode gap (nm).
     * Must be positive.
     */
    void setLgap(double gap);

    /**
     * Get the cell size in Z direction (nm).
     */
    double getLcell() const {
        return Lcell;
    }

    /**
     * Set the cell size in Z direction (nm).
     * Must be positive.
     */
    void setLcell(double cell);

    /**
     * Get the total electrode area (nm^2).
     */
    double getTotalArea() const {
        return totalArea;
    }

    /**
     * Set the total electrode area (nm^2).
     * Must be positive.
     */
    void setTotalArea(double area);

    /**
     * Get the cathode Z position (nm).
     */
    double getZCathode() const {
        return z_cathode;
    }

    /**
     * Set the cathode Z position (nm).
     */
    void setZCathode(double z) {
        z_cathode = z;
    }

    /**
     * Get the anode Z position (nm).
     */
    double getZAnode() const {
        return z_anode;
    }

    /**
     * Set the anode Z position (nm).
     */
    void setZAnode(double z) {
        z_anode = z;
    }

    // ═══════════════════════════════════════════════════════════
    // SCF Parameters
    // ═══════════════════════════════════════════════════════════

    /**
     * Get the number of SCF iterations per charge update.
     */
    int getNumSCFIterations() const {
        return nIterations;
    }

    /**
     * Set the number of SCF iterations per charge update.
     * Must be >= 1. Default is 4.
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
     * @return           index of the cathode atom
     */
    int addCathodeAtom(int particle, double area);

    /**
     * Get cathode atom parameters.
     *
     * @param index      cathode atom index
     * @param particle   [out] particle index
     * @param area       [out] area per atom (nm^2)
     */
    void getCathodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * Set cathode atom parameters.
     *
     * @param index      cathode atom index
     * @param particle   particle index
     * @param area       area per atom (nm^2)
     */
    void setCathodeAtomParameters(int index, int particle, double area);

    /**
     * Get the number of cathode atoms.
     */
    int getNumCathodeAtoms() const {
        return cathodeAtoms.size();
    }

    /**
     * Add an anode atom.
     *
     * @param particle   particle index
     * @param area       area per atom (nm^2)
     * @return           index of the anode atom
     */
    int addAnodeAtom(int particle, double area);

    /**
     * Get anode atom parameters.
     *
     * @param index      anode atom index
     * @param particle   [out] particle index
     * @param area       [out] area per atom (nm^2)
     */
    void getAnodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * Set anode atom parameters.
     *
     * @param index      anode atom index
     * @param particle   particle index
     * @param area       area per atom (nm^2)
     */
    void setAnodeAtomParameters(int index, int particle, double area);

    /**
     * Get the number of anode atoms.
     */
    int getNumAnodeAtoms() const {
        return anodeAtoms.size();
    }

    /**
     * Add an electrolyte atom.
     *
     * @param particle   particle index
     * @param charge     fixed charge (elementary charge units)
     * @return           index of the electrolyte atom
     */
    int addElectrolyteAtom(int particle, double charge);

    /**
     * Get electrolyte atom parameters.
     *
     * @param index      electrolyte atom index
     * @param particle   [out] particle index
     * @param charge     [out] fixed charge (elementary charge units)
     */
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;

    /**
     * Set electrolyte atom parameters.
     *
     * @param index      electrolyte atom index
     * @param particle   particle index
     * @param charge     fixed charge (elementary charge units)
     */
    void setElectrolyteAtomParameters(int index, int particle, double charge);

    /**
     * Get the number of electrolyte atoms.
     */
    int getNumElectrolyteAtoms() const {
        return electrolyteAtoms.size();
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

protected:
    /**
     * Initialize the integrator.
     * Called by Context when it is created or reinitialized.
     */
    void initialize(ContextImpl& context) override;

    /**
     * Clean up the integrator.
     * Called by Context when it is destroyed or reinitialized.
     */
    void cleanup() override;

    /**
     * Get the names of all kernels used by this integrator.
     */
    std::vector<std::string> getKernelNames() override;

    /**
     * Compute the total kinetic energy of the system.
     *
     * @return the kinetic energy (kJ/mol)
     */
    double computeKineticEnergy() override;

private:
    // Physical parameters
    double voltageVolts;    // Applied voltage (V)
    double voltageKjMol;    // Applied voltage (kJ/mol, internal)
    double Lgap;            // Electrode gap (nm)
    double Lcell;           // Cell size in Z (nm)
    double totalArea;       // Total electrode area (nm^2)
    double z_cathode;       // Cathode Z position (nm)
    double z_anode;         // Anode Z position (nm)

    // SCF parameters
    int nIterations;        // SCF iterations per charge update
    int scfFrequency;       // Update charges every N MD steps

    // Electrode atom data
    struct CathodeAtomInfo {
        int particle;
        double area;
        CathodeAtomInfo(int p, double a) : particle(p), area(a) {}
    };
    struct AnodeAtomInfo {
        int particle;
        double area;
        AnodeAtomInfo(int p, double a) : particle(p), area(a) {}
    };
    struct ElectrolyteAtomInfo {
        int particle;
        double charge;
        ElectrolyteAtomInfo(int p, double c) : particle(p), charge(c) {}
    };

    std::vector<CathodeAtomInfo> cathodeAtoms;
    std::vector<AnodeAtomInfo> anodeAtoms;
    std::vector<ElectrolyteAtomInfo> electrolyteAtoms;

    // Kernels
    Kernel verletKernel;        // IntegrateVerletStepKernel
    Kernel calcConstantVKernel;  // CalcConstantVKernel

    // Internal state
    int stepCount;  // Track MD steps for SCF frequency
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVINTEGRATOR_H_
