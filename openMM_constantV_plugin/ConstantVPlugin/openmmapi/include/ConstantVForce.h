#ifndef OPENMM_CONSTANTVFORCE_H_
#define OPENMM_CONSTANTVFORCE_H_

#include "openmm/Force.h"
#include <vector>
#include <string>

namespace ConstantVPlugin {

/**
 * ConstantVForce 使用自洽场（SCF）迭代更新电极电荷
 *
 * 算法（教授方法）：
 *   for iter in range(Niterations):
 *       1. 从OpenMM获取力（包含所有交互）
 *       2. 计算Ez = F_z / q_old
 *       3. 更新电荷：q = 2/(4π) * area * (V/Lgap + Ez)
 *       4. Green's reciprocity校正
 *       5. 更新OpenMM context
 *
 * 翻译自: MM_classes.py::Poisson_solver_fixed_voltage
 */
class ConstantVForce : public OpenMM::Force {
public:
    ConstantVForce();

    // ═══════════════════════════════════════════════════════════
    // 电极设置（对应教授的Cathode/Anode）
    // ═══════════════════════════════════════════════════════════

    /**
     * 添加阴极原子
     * @param particle   原子全局索引
     * @param area       该原子的表面积 (nm^2)
     * @return 阴极原子列表中的索引
     */
    int addCathodeAtom(int particle, double area);

    /**
     * 添加阳极原子
     * @param particle   原子全局索引
     * @param area       该原子的表面积 (nm^2)
     * @return 阳极原子列表中的索引
     */
    int addAnodeAtom(int particle, double area);

    /**
     * 获取阴极原子数
     */
    int getNumCathodeAtoms() const {
        return cathodeAtomIndices.size();
    }

    /**
     * 获取阳极原子数
     */
    int getNumAnodeAtoms() const {
        return anodeAtomIndices.size();
    }

    /**
     * 获取阴极原子参数
     */
    void getCathodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * 获取阳极原子参数
     */
    void getAnodeAtomParameters(int index, int& particle, double& area) const;

    // ═══════════════════════════════════════════════════════════
    // 电解质设置（用于Green's reciprocity镜像电荷）
    // ═══════════════════════════════════════════════════════════

    /**
     * 添加电解质原子（固定电荷）
     * @param particle   原子全局索引
     * @param charge     固定电荷（e）
     * @return 电解质原子列表中的索引
     */
    int addElectrolyteAtom(int particle, double charge);

    /**
     * 获取电解质原子数
     */
    int getNumElectrolyteAtoms() const {
        return electrolyteAtomIndices.size();
    }

    /**
     * 获取电解质原子参数
     */
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;

    // ═══════════════════════════════════════════════════════════
    // 导体设置（Conductor support - Buckyball, Nanotube）
    // 对应: Buckyball_Virtual, Nanotube_Virtual classes
    // ═══════════════════════════════════════════════════════════

    /**
     * 添加Buckyball导体（球形导体）
     *
     * 对应: Buckyball_Virtual class (Fixed_Voltage_routines.py:391-473)
     *
     * @param virtualAtoms     虚拟层原子索引列表（用于静电）
     * @param realAtoms        真实层原子索引列表（用于VDW/steric）
     * @param electrodeType    "cathode" 或 "anode"
     * @param voltage          施加电压（V）
     * @return Buckyball导体列表中的索引
     */
    int addBuckyballConductor(const std::vector<int>& virtualAtoms,
                               const std::vector<int>& realAtoms,
                               const std::string& electrodeType,
                               double voltage);

    /**
     * 获取Buckyball导体数量
     */
    int getNumBuckyballConductors() const {
        return buckyballConductors.size();
    }

    /**
     * 获取Buckyball导体参数
     */
    void getBuckyballConductorParameters(int index,
                                          std::vector<int>& virtualAtoms,
                                          std::vector<int>& realAtoms,
                                          std::string& electrodeType,
                                          double& voltage) const;

    // ═══════════════════════════════════════════════════════════
    // Nanotube Conductor API
    // 对应: Nanotube_Virtual class (Fixed_Voltage_routines.py:482-589)
    // ═══════════════════════════════════════════════════════════

    /**
     * 添加碳纳米管导体（圆柱形）
     * 对应: Nanotube_Virtual(electrode_identifier, electrode_type, Voltage, MMsys, chain_flag, exclude_element, axis)
     *
     * 纳米管使用圆柱几何，法向量为径向（垂直于轴）
     * 面积: 2π × radius × length / Natoms
     *
     * @param virtualAtoms     虚拟层原子索引列表（用于静电）
     * @param realAtoms        真实层原子索引列表（用于VDW/steric）
     * @param electrodeType    "cathode" 或 "anode"
     * @param voltage          施加电压（V）
     * @param axis             纳米管轴向单位向量 [ax, ay, az] (e.g., [1,0,0] for x-axis)
     * @return Nanotube导体列表中的索引
     */
    int addNanotubeConductor(const std::vector<int>& virtualAtoms,
                             const std::vector<int>& realAtoms,
                             const std::string& electrodeType,
                             double voltage,
                             const std::vector<double>& axis);

    /**
     * 获取Nanotube导体数量
     */
    int getNumNanotubeConductors() const {
        return nanotubeConductors.size();
    }

    /**
     * 获取Nanotube导体参数
     */
    void getNanotubeConductorParameters(int index,
                                        std::vector<int>& virtualAtoms,
                                        std::vector<int>& realAtoms,
                                        std::string& electrodeType,
                                        double& voltage,
                                        std::vector<double>& axis) const;

    // ═══════════════════════════════════════════════════════════
    // 系统几何参数（对应MMsys.Lgap, Lcell等）
    // ═══════════════════════════════════════════════════════════

    /**
     * 设置电压（V）
     * 对应: self.Cathode.Voltage
     * 注意：内部会转换成kJ/mol（乘以96.487）
     */
    void setVoltage(double voltage);
    double getVoltage() const { return voltageVolts; }

    /**
     * 设置真空间隙 Lgap (nm)
     * 对应: MMsys.Lgap
     */
    void setLgap(double gap);
    double getLgap() const { return Lgap; }

    /**
     * 设置电极间距 Lcell (nm)
     * 对应: MMsys.Lcell
     */
    void setLcell(double cell);
    double getLcell() const { return Lcell; }

    /**
     * 设置电极总面积 (nm^2)
     * 对应: self.Cathode.sheet_area
     */
    void setTotalArea(double area);
    double getTotalArea() const { return totalArea; }

    /**
     * 设置阴极z位置 (nm)
     * 对应: self.Cathode.z_pos
     */
    void setZCathode(double z);
    double getZCathode() const { return z_cathode; }

    /**
     * 设置阳极z位置 (nm)
     * 对应: self.Anode.z_pos
     */
    void setZAnode(double z);
    double getZAnode() const { return z_anode; }

    // ═══════════════════════════════════════════════════════════
    // SCF参数
    // ═══════════════════════════════════════════════════════════

    /**
     * 设置SCF迭代次数
     * 对应: Niterations（教授默认4）
     */
    void setNumIterations(int n);
    int getNumIterations() const { return nIterations; }

    bool usesPeriodicBoundaryConditions() const override {
        return true;
    }

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    // 电极原子信息
    class CathodeAtomInfo;
    class AnodeAtomInfo;
    class ElectrolyteAtomInfo;

    // 导体信息（Conductor support）
    class BuckyballConductorInfo;
    class NanotubeConductorInfo;

    std::vector<CathodeAtomInfo> cathodeAtoms;
    std::vector<int> cathodeAtomIndices;
    std::vector<double> cathodeAreas;

    std::vector<AnodeAtomInfo> anodeAtoms;
    std::vector<int> anodeAtomIndices;
    std::vector<double> anodeAreas;

    std::vector<ElectrolyteAtomInfo> electrolyteAtoms;
    std::vector<int> electrolyteAtomIndices;
    std::vector<double> electrolyteCharges;

    // 导体信息
    std::vector<BuckyballConductorInfo> buckyballConductors;
    std::vector<NanotubeConductorInfo> nanotubeConductors;

    // 系统参数（对应教授的MMsys成员）
    double voltageVolts;    // 输入电压（V）
    double voltageKjMol;    // 内部使用（kJ/mol）
    double Lgap;            // 真空间隙 (nm)
    double Lcell;           // 电极间距 (nm)
    double totalArea;       // 电极总面积 (nm^2)
    double z_cathode;       // 阴极z位置 (nm)
    double z_anode;         // 阳极z位置 (nm)

    // SCF参数
    int nIterations;        // 迭代次数（默认4）
};

/**
 * 阴极原子信息
 */
class ConstantVForce::CathodeAtomInfo {
public:
    int particle;
    double area;  // 该原子的表面积 (nm^2)
    CathodeAtomInfo() : particle(-1), area(0.0) {}
    CathodeAtomInfo(int particle, double area) : particle(particle), area(area) {}
};

/**
 * 阳极原子信息
 */
class ConstantVForce::AnodeAtomInfo {
public:
    int particle;
    double area;  // 该原子的表面积 (nm^2)
    AnodeAtomInfo() : particle(-1), area(0.0) {}
    AnodeAtomInfo(int particle, double area) : particle(particle), area(area) {}
};

/**
 * 电解质原子信息
 */
class ConstantVForce::ElectrolyteAtomInfo {
public:
    int particle;
    double charge;  // 固定电荷 (e)
    ElectrolyteAtomInfo() : particle(-1), charge(0.0) {}
    ElectrolyteAtomInfo(int particle, double charge) : particle(particle), charge(charge) {}
};

/**
 * Buckyball导体信息
 *
 * 对应: Buckyball_Virtual class (Fixed_Voltage_routines.py:391-473)
 *
 * 关键物理概念：
 * - 虚拟层（virtual）：用于静电计算，通过镜像电荷满足Maxwell边界条件
 * - 真实层（real）：用于VDW/steric交互，防止离子穿透
 * - 几何参数：球心(r_center)、半径(radius)、表面法向量(normal vectors)
 * - 边界条件：球面上法向电场为零（通过镜像电荷实现）
 */
class ConstantVForce::BuckyballConductorInfo {
public:
    // 原子列表
    std::vector<int> virtualAtomIndices;  // 虚拟层原子索引（对应 electrode_atoms）
    std::vector<int> realAtomIndices;     // 真实层原子索引（对应 electrode_atoms_real）

    // 电极类型和电压
    std::string electrodeType;  // "cathode" 或 "anode"
    double voltageVolts;        // 输入电压（V）
    double voltageKjMol;        // 内部使用（kJ/mol，= voltageVolts * 96.487）

    // 几何参数（将在初始化时计算）
    double r_center[3];         // 球心位置 (nm) - Line 428-436
    double radius;              // 球半径 (nm) - Line 440-446
    double area_atom;           // 每原子面积 (nm^2) - Line 447

    // 每个原子的表面法向量（在初始化时计算）
    // 对应: atom.nx, atom.ny, atom.nz (Line 451-456)
    std::vector<double> normalVectors;  // 扁平化: [nx0,ny0,nz0, nx1,ny1,nz1, ...]

    // 接触导体信息（用于电荷转移计算）
    // 对应: find_contact_neighbor_conductor (Line 459)
    int contactAtomIndex;              // 最近的接触电极原子索引
    double dr_center_contact;          // 球心到接触原子的距离 (nm)
    bool closeToElectrode;             // 是否靠近主电极（vs另一个导体）
    double closeThreshold;             // 接近阈值 (nm)，默认1.5

    // 默认构造函数
    BuckyballConductorInfo() :
        electrodeType(""),
        voltageVolts(0.0),
        voltageKjMol(0.0),
        radius(0.0),
        area_atom(0.0),
        contactAtomIndex(-1),
        dr_center_contact(0.0),
        closeToElectrode(true),
        closeThreshold(1.5)
    {
        r_center[0] = r_center[1] = r_center[2] = 0.0;
    }

    // 完整构造函数
    BuckyballConductorInfo(const std::vector<int>& virtualAtoms,
                            const std::vector<int>& realAtoms,
                            const std::string& type,
                            double voltage) :
        virtualAtomIndices(virtualAtoms),
        realAtomIndices(realAtoms),
        electrodeType(type),
        voltageVolts(voltage),
        voltageKjMol(voltage * 96.487),  // conversion_eV_Kjmol
        radius(0.0),
        area_atom(0.0),
        contactAtomIndex(-1),
        dr_center_contact(0.0),
        closeToElectrode(true),
        closeThreshold(1.5)
    {
        r_center[0] = r_center[1] = r_center[2] = 0.0;
        // normalVectors will be initialized later (3 * virtualAtoms.size())
    }
};

/**
 * ═══════════════════════════════════════════════════════════════
 * NanotubeConductorInfo - 碳纳米管导体信息
 * 对应: Nanotube_Virtual class (Fixed_Voltage_routines.py:482-589)
 * ═══════════════════════════════════════════════════════════════
 *
 * 关键物理概念：
 * - 圆柱几何：轴向(axis)、半径(radius)、长度(length)
 * - 法向量：径向(radial direction)，垂直于轴
 * - 面积: 2π × radius × length / Natoms （圆柱侧面积）
 * - project_orthogonal_to_axis: vec_out = vec_in - axis * dot(vec_in, axis)
 * - 边界条件：圆柱面上径向电场为零
 */
class ConstantVForce::NanotubeConductorInfo {
public:
    // 原子列表
    std::vector<int> virtualAtomIndices;  // 虚拟层原子索引（对应 electrode_atoms）
    std::vector<int> realAtomIndices;     // 真实层原子索引（对应 electrode_atoms_real）

    // 电极类型和电压
    std::string electrodeType;  // "cathode" 或 "anode"
    double voltageVolts;        // 输入电压（V）
    double voltageKjMol;        // 内部使用（kJ/mol，= voltageVolts * 96.487）

    // 圆柱几何参数（将在初始化时计算）
    double axis[3];             // 纳米管轴向单位向量 (Line 497, 577-578)
    double r_center[3];         // 中心位置 (nm) (Line 521-529)
    double radius;              // 半径 (nm) (Line 541-556)
    double length;              // 长度 (nm) = box 'a' vector length (Line 532-536)
    double area_atom;           // 每原子面积 (nm^2) = 2π*r*L/N (Line 561)

    // 每个原子的径向法向量（在初始化时计算）
    // 法向量 = radial direction perpendicular to axis
    // 对应: atom.nx, atom.ny, atom.nz (Line 558)
    std::vector<double> normalVectors;  // 扁平化: [nx0,ny0,nz0, nx1,ny1,nz1, ...]

    // 接触导体信息（用于电荷转移计算）
    // 对应: find_contact_neighbor_conductor (Line 564)
    int contactAtomIndex;              // 最近的接触电极原子索引
    double dr_center_contact;          // 径向距离 (nm) (Line 567-570)
    bool closeToElectrode;             // 是否靠近主电极
    double closeThreshold;             // 接近阈值 (nm)，默认1.5

    // 默认构造函数
    NanotubeConductorInfo() :
        electrodeType(""),
        voltageVolts(0.0),
        voltageKjMol(0.0),
        radius(0.0),
        length(0.0),
        area_atom(0.0),
        contactAtomIndex(-1),
        dr_center_contact(0.0),
        closeToElectrode(true),
        closeThreshold(1.5)
    {
        axis[0] = axis[1] = axis[2] = 0.0;
        r_center[0] = r_center[1] = r_center[2] = 0.0;
    }

    // 完整构造函数
    NanotubeConductorInfo(const std::vector<int>& virtualAtoms,
                          const std::vector<int>& realAtoms,
                          const std::string& type,
                          double voltage,
                          const std::vector<double>& axisVec) :
        virtualAtomIndices(virtualAtoms),
        realAtomIndices(realAtoms),
        electrodeType(type),
        voltageVolts(voltage),
        voltageKjMol(voltage * 96.487),  // conversion_eV_Kjmol
        radius(0.0),
        length(0.0),
        area_atom(0.0),
        contactAtomIndex(-1),
        dr_center_contact(0.0),
        closeToElectrode(true),
        closeThreshold(1.5)
    {
        // 复制axis向量（应该是单位向量）
        if (axisVec.size() == 3) {
            axis[0] = axisVec[0];
            axis[1] = axisVec[1];
            axis[2] = axisVec[2];
        } else {
            axis[0] = 1.0; axis[1] = 0.0; axis[2] = 0.0;  // default x-axis
        }
        r_center[0] = r_center[1] = r_center[2] = 0.0;
        // normalVectors will be initialized later (3 * virtualAtoms.size())
    }
};

} // namespace ConstantVPlugin

#endif // OPENMM_CONSTANTVFORCE_H_
