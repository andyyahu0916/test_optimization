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

#ifndef OPENMM_CONSTANTVOLTAGE_KERNELS_H_
#define OPENMM_CONSTANTVOLTAGE_KERNELS_H_

#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace OpenMM {

class ConstantVoltageForce;
class ConstantVDrudeLangevinIntegrator;
class ContextImpl;

/**
 * This kernel is invoked by ConstantVoltageForce to initialize data on the GPU.
 * It also provides the SCF charge update functionality that can be called by
 * the Integrator.
 */
class CalcConstantVoltageForceKernel : public KernelImpl {
public:
    static std::string Name() {
        return "CalcConstantVoltageForce";
    }
    CalcConstantVoltageForceKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {}

    /**
     * Initialize the kernel.
     *
     * @param system   the System this kernel will be applied to
     * @param force    the ConstantVoltageForce this kernel will be used for
     */
    virtual void initialize(const System& system, const ConstantVoltageForce& force) = 0;

    /**
     * Execute the kernel to calculate forces and energy (returns 0, as this
     * Force doesn't compute forces - it only provides data for the Integrator).
     */
    virtual double execute(ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    /**
     * Perform SCF charge update on electrode atoms. This is called by the
     * Integrator before the MD step.
     *
     * GPU-NATIVE: All computation happens on GPU with zero CPU-GPU transfers.
     */
    virtual void updateElectrodeCharges(ContextImpl& context) = 0;

    /**
     * Get the total charge on cathode atoms (for diagnostics).
     */
    virtual double getTotalCathodeCharge(ContextImpl& context) = 0;

    /**
     * Get the total charge on anode atoms (for diagnostics).
     */
    virtual double getTotalAnodeCharge(ContextImpl& context) = 0;
};

/**
 * This kernel is invoked by ConstantVDrudeLangevinIntegrator to take one time step.
 * It combines SCF charge updates with Drude Langevin dynamics integration.
 */
class IntegrateConstantVDrudeLangevinStepKernel : public KernelImpl {
public:
    static std::string Name() {
        return "IntegrateConstantVDrudeLangevinStep";
    }
    IntegrateConstantVDrudeLangevinStepKernel(std::string name, const Platform& platform) : KernelImpl(name, platform) {}

    /**
     * Initialize the kernel.
     *
     * @param system      the System this kernel will be applied to
     * @param integrator  the ConstantVDrudeLangevinIntegrator this kernel will be used for
     */
    virtual void initialize(const System& system, const ConstantVDrudeLangevinIntegrator& integrator) = 0;

    /**
     * Execute the kernel to take one timestep.
     *
     * EXECUTION FLOW:
     * 1. If (step % scfFrequency == 0): Call SCF charge update
     * 2. Drude Langevin velocity update (part 1)
     * 3. Position update
     * 4. Apply constraints
     * 5. Recalculate forces
     * 6. Drude Langevin velocity update (part 2)
     * 7. Apply Drude hard wall constraints
     */
    virtual void execute(ContextImpl& context, const ConstantVDrudeLangevinIntegrator& integrator) = 0;

    /**
     * Compute the kinetic energy.
     */
    virtual double computeKineticEnergy(ContextImpl& context, const ConstantVDrudeLangevinIntegrator& integrator) = 0;
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVOLTAGE_KERNELS_H_
