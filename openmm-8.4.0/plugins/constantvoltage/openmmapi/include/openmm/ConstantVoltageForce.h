/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * https://openmm.org                                                         *
 *                                                                            *
 * Copyright (c) 2024 Stanford University and the Authors.                    *
 * Authors: Andy (Constant Voltage Integration)                               *
 * Contributors: Prof. McDaniel (Original Algorithm)                          *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#ifndef OPENMM_CONSTANTVOLTAGEFORCE_H_
#define OPENMM_CONSTANTVOLTAGEFORCE_H_

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include <vector>
#include <string>
#include "internal/windowsExportConstantV.h"

namespace OpenMM {

/**
 * ConstantVoltageForce is a data container for constant voltage electrochemical
 * simulations. It stores electrode atom indices, areas, and geometric parameters
 * needed for the SCF charge update algorithm.
 *
 * This class does NOT directly compute forces. Instead, it provides data to the
 * ConstantVDrudeLangevinIntegrator, which performs the actual SCF charge updates.
 *
 * ALGORITHM REFERENCE:
 * Based on Prof. McDaniel's Fixed-Voltage MD implementation:
 * - MM_classes.py: Poisson_solver_fixed_voltage()
 * - Fixed_Voltage_routines.py: Electrode_Virtual, Buckyball_Virtual, Nanotube_Virtual
 *
 * USAGE:
 * 1. Create a ConstantVoltageForce and add electrode atoms
 * 2. Add it to the System
 * 3. Use ConstantVDrudeLangevinIntegrator (which reads data from this Force)
 *
 * SERIALIZATION:
 * This class supports XML serialization for checkpoint/restart capability.
 */
class OPENMM_EXPORT_CONSTANTV ConstantVoltageForce : public Force {
public:
    /**
     * Create a ConstantVoltageForce.
     */
    ConstantVoltageForce();

    // ═══════════════════════════════════════════════════════════════════════
    // Flat Electrode Atoms
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Add a cathode atom.
     *
     * @param particle   the index of the particle in the System
     * @param area       the area per atom (nm²) for charge calculation
     * @return the index of the cathode atom that was added
     */
    int addCathodeAtom(int particle, double area);

    /**
     * Get the number of cathode atoms.
     */
    int getNumCathodeAtoms() const { return cathodeParticles.size(); }

    /**
     * Get parameters for a cathode atom.
     *
     * @param index          the index of the cathode atom
     * @param[out] particle  the particle index in the System
     * @param[out] area      the area per atom (nm²)
     */
    void getCathodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * Set parameters for a cathode atom.
     */
    void setCathodeAtomParameters(int index, int particle, double area);

    /**
     * Add an anode atom.
     *
     * @param particle   the index of the particle in the System
     * @param area       the area per atom (nm²) for charge calculation
     * @return the index of the anode atom that was added
     */
    int addAnodeAtom(int particle, double area);

    /**
     * Get the number of anode atoms.
     */
    int getNumAnodeAtoms() const { return anodeParticles.size(); }

    /**
     * Get parameters for an anode atom.
     */
    void getAnodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * Set parameters for an anode atom.
     */
    void setAnodeAtomParameters(int index, int particle, double area);

    // ═══════════════════════════════════════════════════════════════════════
    // Electrolyte Atoms (for Green's Reciprocity image charge calculation)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Add an electrolyte atom. These are used for the analytic charge
     * calculation via Green's reciprocity.
     *
     * @param particle  the index of the particle in the System
     * @return the index of the electrolyte atom that was added
     */
    int addElectrolyteAtom(int particle);

    /**
     * Get the number of electrolyte atoms.
     */
    int getNumElectrolyteAtoms() const { return electrolyteParticles.size(); }

    /**
     * Get the particle index for an electrolyte atom.
     */
    int getElectrolyteAtomParticle(int index) const;

    // ═══════════════════════════════════════════════════════════════════════
    // Conductor Support: Buckyballs
    // Ref: Fixed_Voltage_routines.py::Buckyball_Virtual
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Add a Buckyball conductor.
     *
     * @param virtualParticles  indices of virtual (charge-carrying) atoms
     * @param realParticles     indices of real (VDW) atoms
     * @param electrodeType     "cathode" or "anode"
     * @return the index of the conductor that was added
     */
    int addBuckyballConductor(const std::vector<int>& virtualParticles,
                              const std::vector<int>& realParticles,
                              const std::string& electrodeType);

    /**
     * Get the number of Buckyball conductors.
     */
    int getNumBuckyballConductors() const { return buckyballVirtualParticles.size(); }

    /**
     * Get parameters for a Buckyball conductor.
     */
    void getBuckyballConductorParameters(int index,
                                         std::vector<int>& virtualParticles,
                                         std::vector<int>& realParticles,
                                         std::string& electrodeType) const;

    // ═══════════════════════════════════════════════════════════════════════
    // Conductor Support: Nanotubes
    // Ref: Fixed_Voltage_routines.py::Nanotube_Virtual
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Add a Nanotube conductor.
     *
     * @param virtualParticles  indices of virtual (charge-carrying) atoms
     * @param realParticles     indices of real (VDW) atoms
     * @param electrodeType     "cathode" or "anode"
     * @param axis              the nanotube axis direction (normalized)
     * @return the index of the conductor that was added
     */
    int addNanotubeConductor(const std::vector<int>& virtualParticles,
                             const std::vector<int>& realParticles,
                             const std::string& electrodeType,
                             const Vec3& axis);

    /**
     * Get the number of Nanotube conductors.
     */
    int getNumNanotubeConductors() const { return nanotubeVirtualParticles.size(); }

    /**
     * Get parameters for a Nanotube conductor.
     */
    void getNanotubeConductorParameters(int index,
                                        std::vector<int>& virtualParticles,
                                        std::vector<int>& realParticles,
                                        std::string& electrodeType,
                                        Vec3& axis) const;

    // ═══════════════════════════════════════════════════════════════════════
    // System Parameters
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Set the applied voltage (in Volts).
     * Internally converted to kJ/mol for calculations.
     */
    void setVoltage(double voltage);

    /**
     * Get the applied voltage (in Volts).
     */
    double getVoltage() const { return voltage; }

    /**
     * Set the gap distance between electrodes (in nm).
     * This is Lgap in the original algorithm.
     */
    void setLgap(double lgap);

    /**
     * Get the gap distance (in nm).
     */
    double getLgap() const { return Lgap; }

    /**
     * Set the electrochemical cell length (in nm).
     * This is Lcell in the original algorithm.
     */
    void setLcell(double lcell);

    /**
     * Get the cell length (in nm).
     */
    double getLcell() const { return Lcell; }

    /**
     * Set the total electrode area (in nm²).
     */
    void setTotalArea(double area);

    /**
     * Get the total electrode area (in nm²).
     */
    double getTotalArea() const { return totalArea; }

    /**
     * Set the z-positions of the electrodes (in nm).
     * Used for Green's reciprocity image charge calculation.
     */
    void setElectrodeZPositions(double zCathode, double zAnode);

    /**
     * Get the cathode z-position (in nm).
     */
    double getZCathode() const { return zCathode; }

    /**
     * Get the anode z-position (in nm).
     */
    double getZAnode() const { return zAnode; }

    // ═══════════════════════════════════════════════════════════════════════
    // SCF Parameters
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Set the number of SCF iterations per update.
     * Default: 4 (matching original Python implementation)
     */
    void setNumSCFIterations(int n) { numSCFIterations = n; }

    /**
     * Get the number of SCF iterations.
     */
    int getNumSCFIterations() const { return numSCFIterations; }

    /**
     * Set the SCF update frequency (in timesteps).
     * Default: 200 (matching original Python implementation)
     */
    void setSCFFrequency(int freq) { scfFrequency = freq; }

    /**
     * Get the SCF update frequency.
     */
    int getSCFFrequency() const { return scfFrequency; }

    /**
     * Set the small charge threshold for numerical stability.
     * Default: 1e-6 (matching original Python implementation)
     */
    void setSmallThreshold(double threshold) { smallThreshold = threshold; }

    /**
     * Get the small charge threshold.
     */
    double getSmallThreshold() const { return smallThreshold; }

    // ═══════════════════════════════════════════════════════════════════════
    // Force Interface
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Returns whether this force uses periodic boundary conditions.
     */
    bool usesPeriodicBoundaryConditions() const override { return true; }

protected:
    ForceImpl* createImpl() const override;

private:
    // Flat electrode atoms
    std::vector<int> cathodeParticles;
    std::vector<double> cathodeAreas;
    std::vector<int> anodeParticles;
    std::vector<double> anodeAreas;

    // Electrolyte atoms
    std::vector<int> electrolyteParticles;

    // Buckyball conductors
    std::vector<std::vector<int>> buckyballVirtualParticles;
    std::vector<std::vector<int>> buckyballRealParticles;
    std::vector<std::string> buckyballElectrodeTypes;

    // Nanotube conductors
    std::vector<std::vector<int>> nanotubeVirtualParticles;
    std::vector<std::vector<int>> nanotubeRealParticles;
    std::vector<std::string> nanotubeElectrodeTypes;
    std::vector<Vec3> nanotubeAxes;

    // System parameters
    double voltage;      // Volts
    double Lgap;         // nm
    double Lcell;        // nm
    double totalArea;    // nm²
    double zCathode;     // nm
    double zAnode;       // nm

    // SCF parameters
    int numSCFIterations;
    int scfFrequency;
    double smallThreshold;
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVOLTAGEFORCE_H_
