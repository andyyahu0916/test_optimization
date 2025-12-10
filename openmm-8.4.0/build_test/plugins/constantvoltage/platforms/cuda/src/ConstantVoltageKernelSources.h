#ifndef OPENMM_CONSTANTV_KERNEL_SOURCES_H_
#define OPENMM_CONSTANTV_KERNEL_SOURCES_H_

/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * https://openmm.org                                                         *
 *                                                                            *
 * Copyright (c) 2024 Stanford University and the Authors.                    *
 * Authors: Andy (Constant Voltage Integration)                               *
 * -------------------------------------------------------------------------- */

#include <string>

namespace OpenMM {

/**
 * This class is a central holding place for the source code of ConstantVoltage CUDA kernels.
 * The CMake build script inserts declarations into it based on the .cu files in the
 * kernels subfolder.
 */

class ConstantVoltageKernelSources {
public:
static const std::string conductorCharge;
static const std::string constantVoltage;
static const std::string drudeLangevin;

};

} // namespace OpenMM

#endif /*OPENMM_CONSTANTV_KERNEL_SOURCES_H_*/
