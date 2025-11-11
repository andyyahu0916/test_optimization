#ifndef REFERENCE_CONSTANTV_KERNELS_H_
#define REFERENCE_CONSTANTV_KERNELS_H_

#include "ConstantVKernels.h"
#include "openmm/Platform.h"
#include "openmm/NonbondedForce.h"
#include <vector>

namespace ConstantVPlugin {

/**
 * Reference platform implementation (CPU, golden standard).
 */
class ReferenceCalcConstantVKernel : public CalcConstantVKernel {
public:
    ReferenceCalcConstantVKernel(std::string name, const OpenMM::Platform& platform) :
        CalcConstantVKernel(name, platform), nonbondedForce(nullptr) {
    }

    void initialize(const OpenMM::System& system, const ConstantVForce& force) override;

    double execute(OpenMM::ContextImpl& context, bool includeForces, bool includeEnergy) override;

    void copyParametersToContext(OpenMM::ContextImpl& context, const ConstantVForce& force) override;

private:
    // ═══════════════════════════════════════════════════════════
    // 教授算法需要的成员变量
    // 翻译自: MM_classes.py + Fixed_Voltage_routines.py
    // ═══════════════════════════════════════════════════════════

    // 电极分类（对应教授的Cathode/Anode）
    std::vector<int> cathodeAtomIndices;  // self.Cathode.electrode_atoms
    std::vector<int> anodeAtomIndices;    // self.Anode.electrode_atoms

    // 电解质原子（用于Green's reciprocity镜像电荷）
    std::vector<int> electrolyteAtomIndices;  // MMsys.electrolyte_atom_indices
    std::vector<double> electrolyteCharges;   // 固定电荷

    // 系统几何参数（对应MMsys成员）
    double voltage;     // self.Cathode.Voltage (kJ/mol, 已转换)
    double Lgap;        // self.Lgap (nm)
    double Lcell;       // self.Lcell (nm)
    double totalArea;   // self.Cathode.sheet_area (nm^2)

    // 电极位置（用于镜像电荷计算）
    double z_cathode;   // self.Cathode.z_pos
    double z_anode;     // self.Anode.z_pos

    // 每个原子的面积（对应area_atom）
    std::vector<double> areaPerAtom;  // self.Cathode.area_atom

    // SCF参数
    int nIterations;    // Niterations（默认4）

    // 当前电荷（用于迭代，对应atom.charge）
    std::vector<double> currentCharges;

    // 解析总电荷（Green's reciprocity）
    double Q_analytic_cathode;  // self.Cathode.Q_analytic
    double Q_analytic_anode;    // self.Anode.Q_analytic

    // NonbondedForce引用
    OpenMM::NonbondedForce* nonbondedForce;
    std::vector<double> particleSigmas;
    std::vector<double> particleEpsilons;

    // Lazy initialization flag (deferred from initialize() to execute())
    bool chargesInitialized;

    // ═══════════════════════════════════════════════════════════
    // 辅助函数（对应教授的类方法）
    // ═══════════════════════════════════════════════════════════

    /**
     * 计算电极的解析总电荷（Green's reciprocity theorem）
     * 翻译自: Fixed_Voltage_routines.py::compute_Electrode_charge_analytic (318-345行)
     *
     * @param electrodeAtomIndices  电极原子索引
     * @param positions             所有原子位置
     * @param electrodeType         "cathode" or "anode"
     * @param z_opposite            对面电极的z位置
     * @param Q_analytic            输出：解析总电荷
     */
    void computeElectrodeChargeAnalytic(
        const std::vector<int>& electrodeAtomIndices,
        const std::vector<OpenMM::Vec3>& positions,
        const std::string& electrodeType,
        double z_opposite,
        double& Q_analytic
    );

    /**
     * 缩放电荷到解析归一化（Green's reciprocity correction）
     * 翻译自: Fixed_Voltage_routines.py::Scale_charges_analytic (354-372行)
     *
     * @param electrodeAtomIndices  电极原子索引
     * @param Q_analytic            解析总电荷
     * @param printFlag             是否打印（调试用）
     */
    void scaleChargesAnalytic(
        const std::vector<int>& electrodeAtomIndices,
        double Q_analytic,
        bool printFlag = false
    );

private:
    /**
     * Initialize electrode charges (deferred from initialize() to first execute())
     * This follows OpenMM plugin contract: initialize() must be side-effect free
     * @param context  OpenMM context (needed for updateParametersInContext)
     */
    void initializeElectrodeCharges(OpenMM::ContextImpl& context);
};

// ═══════════════════════════════════════════════════════════
// Integrator Kernel（新版：推荐使用）
// ═══════════════════════════════════════════════════════════

/**
 * Reference平台的ConstantVIntegrator Kernel实现
 *
 * 在execute()中实现教授的SCF算法：
 * - 翻译自: MM_classes.py::Poisson_solver_fixed_voltage (Line 287-374)
 */
class ReferenceIntegrateConstantVStepKernel : public IntegrateConstantVStepKernel {
public:
    ReferenceIntegrateConstantVStepKernel(std::string name, const OpenMM::Platform& platform) :
        IntegrateConstantVStepKernel(name, platform), nonbondedForce(nullptr) {
    }

    void initialize(const OpenMM::System& system, const ConstantVIntegrator& integrator) override;

    void execute(OpenMM::ContextImpl& context, const ConstantVIntegrator& integrator) override;

    double computeKineticEnergy(OpenMM::ContextImpl& context, const ConstantVIntegrator& integrator) override;

private:
    // ═══════════════════════════════════════════════════════════
    // 教授算法需要的成员变量
    // 翻译自: MM_classes.py + Fixed_Voltage_routines.py
    // ═══════════════════════════════════════════════════════════

    // 粒子质量（用于动能计算和Verlet积分）
    std::vector<double> particleInvMass;

    // 电极分类（对应教授的Cathode/Anode）
    std::vector<int> cathodeAtomIndices;      // Cathode.electrode_atoms
    std::vector<double> cathodeAreas;         // 每个阴极原子的面积

    std::vector<int> anodeAtomIndices;        // Anode.electrode_atoms
    std::vector<double> anodeAreas;           // 每个阳极原子的面积

    // 电解质原子（用于Green's reciprocity镜像电荷）
    std::vector<int> electrolyteAtomIndices;  // electrolyte_atom_indices
    std::vector<double> electrolyteCharges;   // 固定电荷

    // 系统几何参数（对应MMsys成员）
    double voltage;     // Cathode.Voltage (kJ/mol, 已转换)
    double Lgap;        // MMsys.Lgap (nm)
    double Lcell;       // MMsys.Lcell (nm)
    double totalArea;   // Cathode.sheet_area (nm^2)
    double z_cathode;   // Cathode.z_pos
    double z_anode;     // Anode.z_pos

    // SCF参数
    int nIterations;        // Niterations（默认4）
    int scf_frequency;      // SCF更新频率

    // 当前电荷（用于迭代，对应atom.charge）
    std::vector<double> currentCharges;

    // 解析总电荷（Green's reciprocity）
    double Q_analytic_cathode;  // Cathode.Q_analytic
    double Q_analytic_anode;    // Anode.Q_analytic

    // NonbondedForce引用
    OpenMM::NonbondedForce* nonbondedForce;

    // ═══════════════════════════════════════════════════════════
    // 辅助函数（对应教授的类方法）
    // ═══════════════════════════════════════════════════════════

    /**
     * 计算电极的解析总电荷（Green's reciprocity theorem）
     * 翻译自: Fixed_Voltage_routines.py::compute_Electrode_charge_analytic (318-345行)
     */
    void computeElectrodeChargeAnalytic(
        const std::vector<int>& electrodeAtomIndices,
        const std::vector<OpenMM::Vec3>& positions,
        const std::string& electrodeType,
        double z_opposite,
        double& Q_analytic
    );

    /**
     * 缩放电荷到解析归一化（Green's reciprocity correction）
     * 翻译自: Fixed_Voltage_routines.py::Scale_charges_analytic (354-372行)
     */
    void scaleChargesAnalytic(
        const std::vector<int>& electrodeAtomIndices,
        double Q_analytic
    );

    /**
     * SCF迭代主循环
     * 翻译自: MM_classes.py::Poisson_solver_fixed_voltage (Line 310-365)
     */
    void scf_iteration(OpenMM::ContextImpl& context);
};

} // namespace ConstantVPlugin

#endif // REFERENCE_CONSTANTV_KERNELS_H_
