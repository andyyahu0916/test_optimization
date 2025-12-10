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

#ifndef OPENMM_CONSTANTVOLTAGEFORCEIMPL_H_
#define OPENMM_CONSTANTVOLTAGEFORCEIMPL_H_

#include "openmm/internal/ForceImpl.h"
#include "openmm/ConstantVoltageForce.h"
#include "openmm/Kernel.h"
#include <string>

namespace OpenMM {

/**
 * This is the internal implementation of ConstantVoltageForce.
 */
class OPENMM_EXPORT_CONSTANTV ConstantVoltageForceImpl : public ForceImpl {
public:
    ConstantVoltageForceImpl(const ConstantVoltageForce& owner);
    ~ConstantVoltageForceImpl();

    void initialize(ContextImpl& context) override;

    const ConstantVoltageForce& getOwner() const { return owner; }

    void updateContextState(ContextImpl& context, bool& forcesInvalid) override {}

    double calcForcesAndEnergy(ContextImpl& context, bool includeForces, bool includeEnergy, int groups) override;

    std::map<std::string, double> getDefaultParameters() override {
        return std::map<std::string, double>();
    }

    std::vector<std::string> getKernelNames() override;

private:
    const ConstantVoltageForce& owner;
    Kernel kernel;
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVOLTAGEFORCEIMPL_H_
