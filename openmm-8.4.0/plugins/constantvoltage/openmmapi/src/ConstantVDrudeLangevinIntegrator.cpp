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

#include "openmm/ConstantVDrudeLangevinIntegrator.h"
#include "openmm/ConstantVoltageKernels.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace OpenMM;

ConstantVDrudeLangevinIntegrator::ConstantVDrudeLangevinIntegrator(
    double temperature, double frictionCoeff,
    double drudeTemperature, double drudeFriction,
    double stepSize)
    : temperature(temperature), friction(frictionCoeff),
      drudeTemperature(drudeTemperature), drudeFriction(drudeFriction),
      maxDrudeDistance(0.02),  // 0.02 nm default, same as DrudeLangevinIntegrator
      randomNumberSeed(0),
      hasInitializedKernel(false)
{
    setStepSize(stepSize);
    setConstraintTolerance(1e-5);
}

void ConstantVDrudeLangevinIntegrator::setTemperature(double temp) {
    if (temp < 0)
        throw OpenMMException("ConstantVDrudeLangevinIntegrator: Temperature cannot be negative");
    temperature = temp;
}

void ConstantVDrudeLangevinIntegrator::setFriction(double coeff) {
    if (coeff < 0)
        throw OpenMMException("ConstantVDrudeLangevinIntegrator: Friction cannot be negative");
    friction = coeff;
}

void ConstantVDrudeLangevinIntegrator::setDrudeTemperature(double temp) {
    if (temp < 0)
        throw OpenMMException("ConstantVDrudeLangevinIntegrator: Drude temperature cannot be negative");
    drudeTemperature = temp;
}

void ConstantVDrudeLangevinIntegrator::setDrudeFriction(double coeff) {
    if (coeff < 0)
        throw OpenMMException("ConstantVDrudeLangevinIntegrator: Drude friction cannot be negative");
    drudeFriction = coeff;
}

void ConstantVDrudeLangevinIntegrator::setMaxDrudeDistance(double distance) {
    if (distance < 0)
        throw OpenMMException("ConstantVDrudeLangevinIntegrator: Max Drude distance cannot be negative");
    maxDrudeDistance = distance;
}

void ConstantVDrudeLangevinIntegrator::initialize(ContextImpl& contextRef) {
    if (hasInitializedKernel) {
        throw OpenMMException("ConstantVDrudeLangevinIntegrator: This integrator has already been initialized");
    }
    context = &contextRef;
    kernel = context->getPlatform().createKernel(IntegrateConstantVDrudeLangevinStepKernel::Name(), *context);
    kernel.getAs<IntegrateConstantVDrudeLangevinStepKernel>().initialize(context->getSystem(), *this);
    hasInitializedKernel = true;
}

void ConstantVDrudeLangevinIntegrator::cleanup() {
    hasInitializedKernel = false;
}

std::vector<std::string> ConstantVDrudeLangevinIntegrator::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(IntegrateConstantVDrudeLangevinStepKernel::Name());
    return names;
}

void ConstantVDrudeLangevinIntegrator::step(int steps) {
    if (!hasInitializedKernel) {
        throw OpenMMException("ConstantVDrudeLangevinIntegrator: This integrator has not been initialized. "
                              "Create a Context with this integrator first.");
    }
    for (int i = 0; i < steps; ++i) {
        context->updateContextState();
        context->calcForcesAndEnergy(true, false);
        kernel.getAs<IntegrateConstantVDrudeLangevinStepKernel>().execute(*context, *this);
    }
}

double ConstantVDrudeLangevinIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateConstantVDrudeLangevinStepKernel>().computeKineticEnergy(*context, *this);
}
