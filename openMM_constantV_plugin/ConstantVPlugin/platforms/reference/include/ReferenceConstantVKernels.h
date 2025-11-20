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

    // Buckyball导体（对应Buckyball_Virtual class）
    // Fixed_Voltage_routines.py:391-473
    struct BuckyballConductor {
        std::vector<int> virtualAtomIndices;   // 虚拟层原子
        std::vector<int> realAtomIndices;      // 真实层原子
        std::string electrodeType;             // "cathode" or "anode"
        double voltageKjMol;                   // 电压 (kJ/mol)
        double r_center[3];                    // 球心位置 (nm)
        double radius;                         // 球半径 (nm)
        double area_atom;                      // 每原子面积 (nm^2)
        std::vector<double> normalVectors;     // 表面法向量 [nx0,ny0,nz0, nx1,ny1,nz1, ...]
        int contactAtomIndex;                  // 最近接触电极原子
        double dr_center_contact;              // 球心到接触原子距离 (nm)
        bool closeToElectrode;                 // 是否靠近主电极
        double closeThreshold;                 // 接近阈值 (nm)
    };
    std::vector<BuckyballConductor> buckyballConductors;

    // Nanotube导体（对应Nanotube_Virtual class）
    // Fixed_Voltage_routines.py:482-589
    struct NanotubeConductor {
        std::vector<int> virtualAtomIndices;   // 虚拟层原子
        std::vector<int> realAtomIndices;      // 真实层原子
        std::string electrodeType;             // "cathode" or "anode"
        double voltageKjMol;                   // 电压 (kJ/mol)
        double axis[3];                        // 纳米管轴向单位向量
        double r_center[3];                    // 中心位置 (nm)
        double radius;                         // 半径 (nm)
        double length;                         // 长度 (nm) = box 'a' vector
        double area_atom;                      // 每原子面积 (nm^2) = 2π*r*L/N
        std::vector<double> normalVectors;     // 径向法向量 [nx0,ny0,nz0, nx1,ny1,nz1, ...]
        int contactAtomIndex;                  // 最近接触电极原子
        double dr_center_contact;              // 径向距离 (nm)
        bool closeToElectrode;                 // 是否靠近主电极
        double closeThreshold;                 // 接近阈值 (nm)
    };
    std::vector<NanotubeConductor> nanotubeConductors;

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

    /**
     * Initialize Buckyball geometry (center, radius, normals)
     * 翻译自: Buckyball_Virtual.__init__ (Line 424-457)
     * @param conductor     Buckyball导体对象
     * @param positions     所有原子位置
     */
    void initializeBuckyballGeometry(BuckyballConductor& conductor,
                                      const std::vector<OpenMM::Vec3>& positions);

    /**
     * Find contact neighbor conductor
     * 翻译自: Conductor_Virtual.find_contact_neighbor_conductor (Line 177-227)
     * @param conductor     Buckyball导体对象
     * @param positions     所有原子位置
     */
    void findContactNeighborConductor(BuckyballConductor& conductor,
                                       const std::vector<OpenMM::Vec3>& positions);

    /**
     * Numerical charge calculation for conductors
     * 翻译自: MM.Numerical_charge_Conductor (Line 388-497)
     * @param conductor     Buckyball导体对象
     * @param forces        所有原子受力
     * @param context       OpenMM上下文
     */
    void numericalChargeConductor(BuckyballConductor& conductor,
                                   const std::vector<OpenMM::Vec3>& forces,
                                   OpenMM::ContextImpl& context);

    /**
     * Initialize Nanotube geometry (center, radius, length, axis, normals)
     * 翻译自: Nanotube_Virtual.__init__ (Line 517-572)
     * @param conductor     Nanotube导体对象
     * @param positions     所有原子位置
     * @param boxVectors    盒子向量（获取长度）
     */
    void initializeNanotubeGeometry(NanotubeConductor& conductor,
                                    const std::vector<OpenMM::Vec3>& positions,
                                    const OpenMM::Vec3 boxVectors[3]);

    /**
     * Project vector orthogonal to nanotube axis
     * 翻译自: Nanotube_Virtual.project_orthogonal_to_axis (Line 576-579)
     * vec_out = vec_in - axis * dot(vec_in, axis)
     * @param vec_in       输入向量
     * @param axis         纳米管轴向
     * @param vec_out      输出向量（垂直分量）
     */
    void projectOrthogonalToAxis(const double vec_in[3],
                                 const double axis[3],
                                 double vec_out[3]);

    /**
     * Numerical charge for Nanotube conductor
     * Same as Buckyball but uses cylindrical geometry
     * @param conductor     Nanotube导体对象
     * @param forces        所有原子受力
     * @param context       OpenMM context
     */
    void numericalChargeNanotube(NanotubeConductor& conductor,
                                 const std::vector<OpenMM::Vec3>& forces,
                                   OpenMM::ContextImpl& context);
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

    // Buckyball导体（对应Buckyball_Virtual class）
    // Fixed_Voltage_routines.py:391-473
    struct BuckyballConductor {
        std::vector<int> virtualAtomIndices;   // 虚拟层原子
        std::vector<int> realAtomIndices;      // 真实层原子
        std::string electrodeType;             // "cathode" or "anode"
        double voltageKjMol;                   // 电压 (kJ/mol)
        double r_center[3];                    // 球心位置 (nm)
        double radius;                         // 球半径 (nm)
        double area_atom;                      // 每原子面积 (nm^2)
        std::vector<double> normalVectors;     // 表面法向量 [nx0,ny0,nz0, nx1,ny1,nz1, ...]
        int contactAtomIndex;                  // 最近接触电极原子
        double dr_center_contact;              // 球心到接触原子距离 (nm)
        bool closeToElectrode;                 // 是否靠近主电极
        double closeThreshold;                 // 接近阈值 (nm)
    };
    std::vector<BuckyballConductor> buckyballConductors;

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

    /**
     * Initialize Buckyball geometry (center, radius, normals)
     * 翻译自: Buckyball_Virtual.__init__ (Line 424-457)
     */
    void initializeBuckyballGeometry(BuckyballConductor& conductor,
                                      const std::vector<OpenMM::Vec3>& positions);

    /**
     * Find contact neighbor conductor
     * 翻译自: Conductor_Virtual.find_contact_neighbor_conductor (Line 177-227)
     */
    void findContactNeighborConductor(BuckyballConductor& conductor,
                                       const std::vector<OpenMM::Vec3>& positions);

    /**
     * Numerical charge calculation for conductors
     * 翻译自: MM.Numerical_charge_Conductor (Line 388-497)
     */
    void numericalChargeConductor(BuckyballConductor& conductor,
                                   const std::vector<OpenMM::Vec3>& forces,
                                   OpenMM::ContextImpl& context);
};

} // namespace ConstantVPlugin

#endif // REFERENCE_CONSTANTV_KERNELS_H_
