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

#ifndef OPENMM_REFERENCE_CONSTANTVOLTAGE_KERNELS_H_
#define OPENMM_REFERENCE_CONSTANTVOLTAGE_KERNELS_H_

#include "openmm/ConstantVoltageKernels.h"
#include "openmm/ConstantVoltageForce.h"

namespace OpenMM {

/**
 * This kernel is invoked by ConstantVoltageForce to handle electrode data.
 */
class ReferenceCalcConstantVoltageForceKernel : public CalcConstantVoltageForceKernel {
public:
    ReferenceCalcConstantVoltageForceKernel(std::string name, const Platform& platform)
        : CalcConstantVoltageForceKernel(name, platform) {}
    ~ReferenceCalcConstantVoltageForceKernel();

    void initialize(const System& system, const ConstantVoltageForce& force) override;
    double execute(ContextImpl& context, bool includeForces, bool includeEnergy) override;
    void updateElectrodeCharges(ContextImpl& context) override;
    double getTotalCathodeCharge(ContextImpl& context) override;
    double getTotalAnodeCharge(ContextImpl& context) override;

private:
    const ConstantVoltageForce* force;
    int numCathodes;
    int numAnodes;
    std::vector<int> cathodeIndices;
    std::vector<double> cathodeAreas;
    std::vector<int> anodeIndices;
    std::vector<double> anodeAreas;
    std::vector<int> electrolyteIndices;
    double voltage_kjmol;
    double Lgap;
    double Lcell;
    double totalArea;
    double zCathode;
    double zAnode;
    double smallThreshold;
    int numSCFIterations;
};

/**
 * This kernel is invoked by ConstantVDrudeLangevinIntegrator to take one time step.
 */
class ReferenceIntegrateConstantVDrudeLangevinStepKernel : public IntegrateConstantVDrudeLangevinStepKernel {
public:
    ReferenceIntegrateConstantVDrudeLangevinStepKernel(std::string name, const Platform& platform)
        : IntegrateConstantVDrudeLangevinStepKernel(name, platform), forceKernel(nullptr) {}
    ~ReferenceIntegrateConstantVDrudeLangevinStepKernel();

    void initialize(const System& system, const ConstantVDrudeLangevinIntegrator& integrator) override;
    void execute(ContextImpl& context, const ConstantVDrudeLangevinIntegrator& integrator) override;
    double computeKineticEnergy(ContextImpl& context, const ConstantVDrudeLangevinIntegrator& integrator) override;

private:
    int numParticles;
    int stepCount;
    int scfFrequency;
    int numSCFIterations;
    double maxDrudeDistance;
    ReferenceCalcConstantVoltageForceKernel* forceKernel;

    // Drude particle data
    std::vector<int> drudeParentIndices;  // For each Drude particle, which is its parent
    std::vector<int> normalParticleIndices;
};

} // namespace OpenMM

#endif // OPENMM_REFERENCE_CONSTANTVOLTAGE_KERNELS_H_
