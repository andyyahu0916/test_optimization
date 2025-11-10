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

    std::vector<CathodeAtomInfo> cathodeAtoms;
    std::vector<int> cathodeAtomIndices;
    std::vector<double> cathodeAreas;

    std::vector<AnodeAtomInfo> anodeAtoms;
    std::vector<int> anodeAtomIndices;
    std::vector<double> anodeAreas;

    std::vector<ElectrolyteAtomInfo> electrolyteAtoms;
    std::vector<int> electrolyteAtomIndices;
    std::vector<double> electrolyteCharges;

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

} // namespace ConstantVPlugin

#endif // OPENMM_CONSTANTVFORCE_H_
