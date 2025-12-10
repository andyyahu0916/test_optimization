/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * https://openmm.org                                                         *
 *                                                                            *
 * Copyright (c) 2024 Stanford University and the Authors.                    *
 * Authors: Andy (Constant Voltage Integration)                               *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and/or its documentation without restriction.        *
 * -------------------------------------------------------------------------- */

#include "ReferenceConstantVoltageKernels.h"
#include "openmm/Platform.h"
#include "openmm/KernelFactory.h"
#include "openmm/internal/ContextImpl.h"

using namespace OpenMM;

extern "C" OPENMM_EXPORT_CONSTANTV void registerConstantVoltagePlatforms() {
}

extern "C" OPENMM_EXPORT_CONSTANTV void registerConstantVoltageReferenceKernelFactories();

class ReferenceConstantVoltageKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const override {
        if (name == CalcConstantVoltageForceKernel::Name())
            return new ReferenceCalcConstantVoltageForceKernel(name, platform);
        if (name == IntegrateConstantVDrudeLangevinStepKernel::Name())
            return new ReferenceIntegrateConstantVDrudeLangevinStepKernel(name, platform);
        throw OpenMMException("ConstantVoltage: Tried to create unknown kernel: " + name);
    }
};

extern "C" OPENMM_EXPORT_CONSTANTV void registerConstantVoltageReferenceKernelFactories() {
    Platform& platform = Platform::getPlatformByName("Reference");
    ReferenceConstantVoltageKernelFactory* factory = new ReferenceConstantVoltageKernelFactory();
    platform.registerKernelFactory(CalcConstantVoltageForceKernel::Name(), factory);
    platform.registerKernelFactory(IntegrateConstantVDrudeLangevinStepKernel::Name(), factory);
}
