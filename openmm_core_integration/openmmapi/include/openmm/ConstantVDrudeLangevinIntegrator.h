/* -------------------------------------------------------------------------- *
 *                          OpenMM - Native ConstantV                        *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM Core API.                                      *
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_CONSTANTVDRUDELANGEVININTEGRATOR_H_
#define OPENMM_CONSTANTVDRUDELANGEVININTEGRATOR_H_

#include "openmm/DrudeLangevinIntegrator.h"
#include "openmm/Vec3.h"
#include "openmm/internal/windowsExport.h"
#include <vector>
#include <string>

namespace OpenMM {

/**
 * ═══════════════════════════════════════════════════════════════════════════
 * ConstantVDrudeLangevinIntegrator - Native Core Implementation
 * ═══════════════════════════════════════════════════════════════════════════
 *
 * This integrator extends DrudeLangevinIntegrator with built-in support for
 * fixed-voltage electrodes using the Self-Consistent Field (SCF) method.
 *
 * Unlike the Plugin approach (ConstantVForce), this integrator directly
 * modifies particle charges during the integration step, eliminating:
 *   - Force Group overhead (no need to exclude groups)
 *   - Context update overhead (no updateParametersInContext() calls)
 *   - Virtual Site workarounds (charges updated directly in kernel)
 *
 * Physical Algorithm (Professor's Method):
 * ----------------------------------------
 * At each MD step, BEFORE integrating velocities/positions:
 *   1. Compute external field: E_ext = F / q_old (from NonbondedForce)
 *   2. Update electrode charges: q_new = 2/(4π) * area * (V/Lgap + E_ext)
 *   3. Apply Green's Reciprocity: Scale charges to match analytic total
 *   4. (Repeat SCF iterations as configured)
 *   5. Integrate dynamics with updated charges
 *
 * Memory Layout (Zero-Copy CUDA):
 * --------------------------------
 * All electrode metadata is stored in a single GPU-resident struct:
 *   - Cathode/Anode atom indices (sorted for coalesced access)
 *   - Per-atom area values
 *   - Electrolyte indices
 *   - Conductor geometry (Buckyball/Nanotube)
 *
 * This data is uploaded ONCE during Context initialization and never copied
 * back to CPU during simulation.
 *
 * References:
 *   - MM_classes.py::Poisson_solver_fixed_voltage() (SCF loop)
 *   - CudaConstantVKernels.cu (Optimized implementation)
 *   - ReferenceConstantVKernels.cpp (Verified double-precision logic)
 *
 * Thread Safety: NOT thread-safe. Create one integrator per Context.
 *
 * Performance:
 *   - Reference Platform: ~100 µs/SCF iteration (N=1000 atoms)
 *   - CUDA Platform: ~5 µs/SCF iteration (N=1000 atoms, RTX 4090)
 */
class OPENMM_EXPORT ConstantVDrudeLangevinIntegrator : public DrudeLangevinIntegrator {
public:
    /**
     * Create a ConstantVDrudeLangevinIntegrator.
     *
     * @param temperature          Temperature of the main heat bath (K)
     * @param frictionCoeff        Friction coefficient for ordinary particles (1/ps)
     * @param drudeTemperature     Temperature of the heat bath applied to Drude particles (K)
     * @param drudeFrictionCoeff   Friction coefficient for Drude particles (1/ps)
     * @param stepSize             Step size with which to integrate (ps)
     * @param voltage              Applied voltage between electrodes (Volts)
     * @param Lgap                 Vacuum gap between electrodes (nm)
     * @param Lcell                Physical distance between electrodes (nm)
     * @param scfIterations        Number of SCF iterations per MD step (default: 4)
     */
    ConstantVDrudeLangevinIntegrator(
        double temperature,
        double frictionCoeff,
        double drudeTemperature,
        double drudeFrictionCoeff,
        double stepSize,
        double voltage,
        double Lgap,
        double Lcell,
        int scfIterations = 4
    );

    /**
     * Destructor.
     */
    ~ConstantVDrudeLangevinIntegrator();

    // ═══════════════════════════════════════════════════════════════════════
    // Electrode Configuration
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Add a cathode electrode atom.
     *
     * @param particle  Particle index for cathode atom
     * @param area      Surface area for this atom (nm²)
     */
    void addCathodeAtom(int particle, double area);

    /**
     * Get the number of cathode atoms.
     */
    int getNumCathodeAtoms() const {
        return cathodeIndices.size();
    }

    /**
     * Get cathode atom parameters.
     *
     * @param index     Atom index in cathode list
     * @param particle  [out] Particle index
     * @param area      [out] Surface area (nm²)
     */
    void getCathodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * Add an anode electrode atom.
     *
     * @param particle  Particle index for anode atom
     * @param area      Surface area for this atom (nm²)
     */
    void addAnodeAtom(int particle, double area);

    /**
     * Get the number of anode atoms.
     */
    int getNumAnodeAtoms() const {
        return anodeIndices.size();
    }

    /**
     * Get anode atom parameters.
     *
     * @param index     Atom index in anode list
     * @param particle  [out] Particle index
     * @param area      [out] Surface area (nm²)
     */
    void getAnodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * Add an electrolyte atom (for Green's Reciprocity image charge term).
     *
     * Electrolyte atoms contribute to the total electrode charge via:
     *   Q_image += (z_distance / Lcell) * (-q_electrolyte)
     *
     * @param particle  Particle index for electrolyte atom
     * @param charge    Fixed charge of this atom (elementary charge units)
     */
    void addElectrolyteAtom(int particle, double charge);

    /**
     * Get the number of electrolyte atoms.
     */
    int getNumElectrolyteAtoms() const {
        return electrolyteIndices.size();
    }

    /**
     * Add Buckyball conductor (spherical conductor near electrode).
     *
     * Physical Requirements:
     *   - Must have both virtual and real atom layers
     *   - Virtual layer: Used for electrostatics (Maxwell BC)
     *   - Real layer: Used for VDW/steric interactions
     *
     * @param virtualIndices   Virtual layer atom indices
     * @param realIndices      Real layer atom indices
     * @param electrodeType    "cathode" or "anode" (which electrode it contacts)
     * @param voltage          Applied voltage for this conductor (V)
     */
    void addBuckyballConductor(
        const std::vector<int>& virtualIndices,
        const std::vector<int>& realIndices,
        const std::string& electrodeType,
        double voltage
    );

    /**
     * Add Nanotube conductor (cylindrical conductor near electrode).
     *
     * @param virtualIndices   Virtual layer atom indices
     * @param realIndices      Real layer atom indices
     * @param electrodeType    "cathode" or "anode"
     * @param voltage          Applied voltage for this conductor (V)
     * @param axis             Unit vector along nanotube axis [ax, ay, az]
     */
    void addNanotubeConductor(
        const std::vector<int>& virtualIndices,
        const std::vector<int>& realIndices,
        const std::string& electrodeType,
        double voltage,
        const Vec3& axis
    );

    /**
     * Get the number of Buckyball conductors.
     */
    int getNumBuckyballConductors() const {
        return buckyballs.size();
    }

    /**
     * Get the number of Nanotube conductors.
     */
    int getNumNanotubeConductors() const {
        return nanotubes.size();
    }

    /**
     * Get Buckyball conductor parameters (FIX P2-3: needed for GPU upload).
     *
     * @param index            Conductor index
     * @param virtualIndices   [out] Virtual layer atom indices
     * @param realIndices      [out] Real layer atom indices
     * @param electrodeType    [out] "cathode" or "anode"
     * @param voltage          [out] Applied voltage (V)
     */
    void getBuckyballConductorParameters(int index,
                                        std::vector<int>& virtualIndices,
                                        std::vector<int>& realIndices,
                                        std::string& electrodeType,
                                        double& voltage) const;

    /**
     * Get Nanotube conductor parameters (FIX P2-3: needed for GPU upload).
     *
     * @param index            Conductor index
     * @param virtualIndices   [out] Virtual layer atom indices
     * @param realIndices      [out] Real layer atom indices
     * @param electrodeType    [out] "cathode" or "anode"
     * @param voltage          [out] Applied voltage (V)
     * @param axis             [out] Unit vector along nanotube axis
     */
    void getNanotubeConductorParameters(int index,
                                       std::vector<int>& virtualIndices,
                                       std::vector<int>& realIndices,
                                       std::string& electrodeType,
                                       double& voltage,
                                       Vec3& axis) const;

    // ═══════════════════════════════════════════════════════════════════════
    // System Geometry Parameters
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Set total electrode area (nm²).
     *
     * This is used in the analytic charge formula:
     *   Q_analytic = sign/(4π) * area * (V/Lgap + V/Lcell) * conversion
     */
    void setTotalArea(double area) { totalArea = area; }

    /**
     * Get total electrode area.
     */
    double getTotalArea() const { return totalArea; }

    /**
     * Set cathode Z position (nm).
     *
     * Required for Green's Reciprocity image charge calculation.
     */
    void setZCathode(double z) { z_cathode = z; }

    /**
     * Get cathode Z position.
     */
    double getZCathode() const { return z_cathode; }

    /**
     * Set anode Z position (nm).
     */
    void setZAnode(double z) { z_anode = z; }

    /**
     * Get anode Z position.
     */
    double getZAnode() const { return z_anode; }

    // ═══════════════════════════════════════════════════════════════════════
    // Voltage and SCF Parameters
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Set applied voltage (Volts).
     */
    void setVoltage(double v) { voltage = v; }

    /**
     * Get applied voltage.
     */
    double getVoltage() const { return voltage; }

    /**
     * Set vacuum gap distance (nm).
     */
    void setLgap(double gap) { Lgap = gap; }

    /**
     * Get vacuum gap distance.
     */
    double getLgap() const { return Lgap; }

    /**
     * Set electrode cell distance (nm).
     */
    void setLcell(double cell) { Lcell = cell; }

    /**
     * Get electrode cell distance.
     */
    double getLcell() const { return Lcell; }

    /**
     * Set number of SCF iterations per MD step.
     *
     * Higher values improve charge convergence but increase cost.
     * Typical values: 2-8 (professor's default: 4)
     */
    void setNumSCFIterations(int n) { scfIterations = n; }

    /**
     * Get number of SCF iterations.
     */
    int getNumSCFIterations() const { return scfIterations; }

    /**
     * Set SCF frequency (charge updates every N MD steps).
     *
     * Must be >= 1. Default is 1 (update every step).
     */
    void setSCFFrequency(int freq) { scfFrequency = freq; }

    /**
     * Get SCF frequency.
     */
    int getSCFFrequency() const { return scfFrequency; }

    // ═══════════════════════════════════════════════════════════════════════
    // Query Methods
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Get current electrode charges (after last SCF convergence).
     *
     * This method queries the NonbondedForce parameters to retrieve
     * the current electrode charges.
     *
     * @param cathodeCharges   Output: Cathode charges (e)
     * @param anodeCharges     Output: Anode charges (e)
     */
    void getElectrodeCharges(
        std::vector<double>& cathodeCharges,
        std::vector<double>& anodeCharges
    ) const;

    /**
     * Get total cathode charge (sum of all cathode atom charges).
     */
    double getTotalCathodeCharge() const;

    /**
     * Get total anode charge.
     */
    double getTotalAnodeCharge() const;

    // ═══════════════════════════════════════════════════════════════════════
    // Integrator Interface (Override DrudeLangevinIntegrator)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Advance the simulation through time by one step.
     *
     * SCF Charge Update Sequence:
     * ---------------------------
     * 1. Call kernel->updateElectrodeCharges() (SCF loop)
     * 2. Call parent DrudeLangevinIntegrator::step() (integrate dynamics)
     *
     * This ensures charges are equilibrated BEFORE positions/velocities update.
     */
    void step(int steps) override;

    /**
     * Get the names of all kernels used by this integrator.
     */
    std::vector<std::string> getKernelNames() override;

    /**
     * Clean up the integrator.
     * Called by Context when it is destroyed or reinitialized.
     */
    void cleanup() override;

protected:
    /**
     * Internal: Initialize electrode geometry (called from Context creation).
     *
     * This method is invoked by ContextImpl::initialize() to set up
     * platform-specific data structures.
     */
    void initialize(ContextImpl& context) override;

private:
    // Voltage parameters
    double voltage;        // Applied voltage (V)
    double Lgap;           // Vacuum gap (nm)
    double Lcell;          // Electrode spacing (nm)
    double totalArea;      // Total electrode area (nm²)
    double z_cathode;      // Cathode Z position (nm)
    double z_anode;        // Anode Z position (nm)

    // SCF control
    int scfIterations;     // Number of SCF iterations per step
    int scfFrequency;      // Update charges every N MD steps

    // Electrode atom data
    std::vector<int> cathodeIndices;
    std::vector<double> cathodeAreas;
    std::vector<int> anodeIndices;
    std::vector<double> anodeAreas;
    std::vector<int> electrolyteIndices;
    std::vector<double> electrolyteCharges;

    // Conductor data (Buckyball/Nanotube)
    struct ConductorData {
        std::vector<int> virtualIndices;
        std::vector<int> realIndices;
        std::string electrodeType;
        double voltage;
        Vec3 axis;  // Only for Nanotube
    };
    std::vector<ConductorData> buckyballs;
    std::vector<ConductorData> nanotubes;

    // Internal flags
    bool electrodesInitialized;
    int stepCount;  // Track number of steps taken

    // Platform-specific kernel (created in initialize())
    Kernel stepKernel;  // Kernel for IntegrateConstantVDrudeLangevinStep
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVDRUDELANGEVININTEGRATOR_H_
