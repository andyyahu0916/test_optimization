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

#include "openmm/internal/ConstantVoltageForceImpl.h"
#include "openmm/ConstantVoltageKernels.h"
#include "openmm/internal/ContextImpl.h"

using namespace OpenMM;

ConstantVoltageForceImpl::ConstantVoltageForceImpl(const ConstantVoltageForce& owner)
    : owner(owner) {
}

ConstantVoltageForceImpl::~ConstantVoltageForceImpl() {
}

void ConstantVoltageForceImpl::initialize(ContextImpl& context) {
    kernel = context.getPlatform().createKernel(CalcConstantVoltageForceKernel::Name(), context);
    kernel.getAs<CalcConstantVoltageForceKernel>().initialize(context.getSystem(), owner);
}

double ConstantVoltageForceImpl::calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) {
    // ConstantVoltageForce doesn't directly compute forces - it provides data
    // to the Integrator which handles SCF charge updates
    return kernel.getAs<CalcConstantVoltageForceKernel>().execute(context, includeForces, includeEnergy);
}

std::vector<std::string> ConstantVoltageForceImpl::getKernelNames() {
    std::vector<std::string> names;
    names.push_back(CalcConstantVoltageForceKernel::Name());
    return names;
}
