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

#include "CudaConstantVoltageKernels.h"
#include "openmm/Platform.h"
#include "openmm/KernelFactory.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaPlatform.h"
#include "openmm/internal/ContextImpl.h"

using namespace OpenMM;

extern "C" OPENMM_EXPORT_CONSTANTV void registerConstantVoltageCudaKernelFactories();

class CudaConstantVoltageKernelFactory : public KernelFactory {
public:
    KernelImpl* createKernelImpl(std::string name, const Platform& platform, ContextImpl& context) const override {
        CudaContext& cu = *static_cast<CudaPlatform::PlatformData*>(context.getPlatformData())->contexts[0];
        if (name == CalcConstantVoltageForceKernel::Name())
            return new CudaCalcConstantVoltageForceKernel(name, platform, cu);
        if (name == IntegrateConstantVDrudeLangevinStepKernel::Name())
            return new CudaIntegrateConstantVDrudeLangevinStepKernel(name, platform, cu);
        throw OpenMMException("ConstantVoltage CUDA: Tried to create unknown kernel: " + name);
    }
};

extern "C" OPENMM_EXPORT_CONSTANTV void registerConstantVoltageCudaKernelFactories() {
    try {
        Platform& platform = Platform::getPlatformByName("CUDA");
        CudaConstantVoltageKernelFactory* factory = new CudaConstantVoltageKernelFactory();
        platform.registerKernelFactory(CalcConstantVoltageForceKernel::Name(), factory);
        platform.registerKernelFactory(IntegrateConstantVDrudeLangevinStepKernel::Name(), factory);
    }
    catch (std::exception& ex) {
        // CUDA platform not available, skip registration
    }
}
