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

#ifndef OPENMM_CONSTANTVDRUDELANGEVININTEGRATOR_H_
#define OPENMM_CONSTANTVDRUDELANGEVININTEGRATOR_H_

#include "openmm/Integrator.h"
#include "openmm/Kernel.h"
#include "internal/windowsExportConstantV.h"
#include <vector>

namespace OpenMM {

/**
 * ConstantVDrudeLangevinIntegrator implements constant voltage electrochemical
 * MD with Drude oscillator polarizability.
 *
 * This integrator combines:
 * 1. SCF electrode charge updates (Self-Consistent Field)
 * 2. Drude Langevin dynamics with dual thermostats
 *
 * ALGORITHM OVERVIEW (per timestep):
 * ----------------------------------
 * if (step % scfFrequency == 0):
 *     for i in numSCFIterations:
 *         Update cathode charges: q = (2/4π) × area × (V/L + Ez) × K
 *         Update anode charges: q = -(2/4π) × area × (V/L + Ez) × K
 *         Update conductor charges (Buckyball/Nanotube)
 *         Recalculate forces
 *         Scale charges to match Green's reciprocity analytic values
 *
 * Drude Langevin Integration:
 *     v_center += 0.5×dt×(F_center/m + frictionCorrection + randomNoise)
 *     v_drude += 0.5×dt×(F_drude/m_drude + drudeFrictionCorrection + drudeNoise)
 *     x += dt × v
 *     Apply constraints
 *     Recalculate forces
 *     v_center += 0.5×dt×(F_center/m + frictionCorrection + randomNoise)
 *     v_drude += 0.5×dt×(F_drude/m_drude + drudeFrictionCorrection + drudeNoise)
 *     Apply Drude hard wall constraint
 *
 * REFERENCE:
 * Prof. McDaniel's MM_classes.py, Fixed_Voltage_routines.py
 *
 * USAGE:
 * 1. Add ConstantVoltageForce to System (with electrode atom data)
 * 2. Create ConstantVDrudeLangevinIntegrator
 * 3. Run simulation - integrator automatically finds Force and performs SCF
 */
class OPENMM_EXPORT_CONSTANTV ConstantVDrudeLangevinIntegrator : public Integrator {
public:
    /**
     * Create a ConstantVDrudeLangevinIntegrator.
     *
     * @param temperature     the temperature of the main heat bath (K)
     * @param frictionCoeff   friction coefficient for main particles (1/ps)
     * @param drudeTemperature the temperature of the Drude heat bath (K)
     * @param drudeFriction    friction coefficient for Drude particles (1/ps)
     * @param stepSize         integration timestep (ps)
     */
    ConstantVDrudeLangevinIntegrator(double temperature, double frictionCoeff,
                                     double drudeTemperature, double drudeFriction,
                                     double stepSize);

    // ═══════════════════════════════════════════════════════════════════════
    // Temperature & Friction (Drude Langevin)
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Get the temperature of the main heat bath (K).
     */
    double getTemperature() const { return temperature; }

    /**
     * Set the temperature of the main heat bath (K).
     */
    void setTemperature(double temp);

    /**
     * Get the friction coefficient for main particles (1/ps).
     */
    double getFriction() const { return friction; }

    /**
     * Set the friction coefficient for main particles (1/ps).
     */
    void setFriction(double coeff);

    /**
     * Get the temperature of the Drude heat bath (K).
     */
    double getDrudeTemperature() const { return drudeTemperature; }

    /**
     * Set the temperature of the Drude heat bath (K).
     */
    void setDrudeTemperature(double temp);

    /**
     * Get the friction coefficient for Drude particles (1/ps).
     */
    double getDrudeFriction() const { return drudeFriction; }

    /**
     * Set the friction coefficient for Drude particles (1/ps).
     */
    void setDrudeFriction(double coeff);

    /**
     * Get the maximum distance a Drude particle can move from its parent (nm).
     * This implements the "hard wall" constraint.
     */
    double getMaxDrudeDistance() const { return maxDrudeDistance; }

    /**
     * Set the maximum Drude distance (nm). Set to 0 to disable.
     */
    void setMaxDrudeDistance(double distance);

    // ═══════════════════════════════════════════════════════════════════════
    // Random Number Generator
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Get the random number seed.
     */
    int getRandomNumberSeed() const { return randomNumberSeed; }

    /**
     * Set the random number seed.
     */
    void setRandomNumberSeed(int seed) { randomNumberSeed = seed; }

    // ═══════════════════════════════════════════════════════════════════════
    // Integrator Interface
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Advance the simulation by a specified number of time steps.
     */
    void step(int steps) override;

protected:
    /**
     * Initialize the integrator. Called when the Context is created.
     */
    void initialize(ContextImpl& context) override;

    /**
     * Clean up the integrator.
     */
    void cleanup() override;

    /**
     * Get the names of all kernels used by this integrator.
     */
    std::vector<std::string> getKernelNames() override;

    /**
     * Compute kinetic energy at the current time.
     */
    double computeKineticEnergy() override;

private:
    // Drude Langevin parameters
    double temperature;
    double friction;
    double drudeTemperature;
    double drudeFriction;
    double maxDrudeDistance;
    int randomNumberSeed;

    // Kernel
    Kernel kernel;
    bool hasInitializedKernel;
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVDRUDELANGEVININTEGRATOR_H_
