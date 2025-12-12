#ifndef CONSTANTV_KERNELS_H_
#define CONSTANTV_KERNELS_H_

#include "ConstantVForce.h"
#include "ConstantVIntegrator.h"
#include "openmm/KernelImpl.h"
#include "openmm/Platform.h"
#include "openmm/System.h"
#include <string>

namespace ConstantVPlugin {

/**
 * This kernel updates electrode charges. It does NOT compute forces or energy.
 * (旧版：用Force实现，已废弃)
 */
class CalcConstantVKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "CalcConstantV";
    }

    CalcConstantVKernel(std::string name, const OpenMM::Platform& platform) :
        OpenMM::KernelImpl(name, platform) {
    }

    /**
     * Initialize the kernel.
     */
    virtual void initialize(const OpenMM::System& system, const ConstantVForce& force) = 0;

    /**
     * Execute the kernel: update electrode charges.
     *
     * @return 0.0 (this force does not contribute to energy)
     */
    virtual double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) = 0;

    /**
     * Copy changed parameters to the context.
     */
    virtual void copyParametersToContext(OpenMM::ContextImpl& context, const ConstantVForce& force) = 0;
};

// ═══════════════════════════════════════════════════════════
// Integrator Kernel（新版：推荐使用）
// ═══════════════════════════════════════════════════════════

/**
 * ConstantVIntegrator的Kernel接口
 * 在execute()中实现SCF迭代（参考DrudeSCFIntegrator）
 */
class IntegrateConstantVStepKernel : public OpenMM::KernelImpl {
public:
    static std::string Name() {
        return "IntegrateConstantVStep";
    }

    IntegrateConstantVStepKernel(std::string name, const OpenMM::Platform& platform) :
        OpenMM::KernelImpl(name, platform) {
    }

    /**
     * 初始化Kernel
     */
    virtual void initialize(const OpenMM::System& system, const ConstantVIntegrator& integrator) = 0;

    /**
     * 执行一个积分步（包含SCF迭代）
     */
    virtual void execute(OpenMM::ContextImpl& context, const ConstantVIntegrator& integrator) = 0;

    /**
     * 计算动能
     */
    virtual double computeKineticEnergy(OpenMM::ContextImpl& context, const ConstantVIntegrator& integrator) = 0;
};

} // namespace ConstantVPlugin

#endif // CONSTANTV_KERNELS_H_
