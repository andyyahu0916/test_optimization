#ifndef OPENMM_CONSTANTVINTEGRATOR_H_
#define OPENMM_CONSTANTVINTEGRATOR_H_

#include "openmm/Integrator.h"
#include "openmm/Kernel.h"

namespace ConstantVPlugin {

/**
 * 恒电压Verlet积分器
 *
 * 在每个时间步执行SCF自洽场迭代，更新电极电荷以满足恒电压边界条件。
 * 基于教授的算法：MM_classes.py::Poisson_solver_fixed_voltage (Line 287-374)
 *
 * 参考：DrudeSCFIntegrator的实现
 */
class ConstantVIntegrator : public OpenMM::Integrator {
public:
    /**
     * 创建ConstantVIntegrator
     *
     * @param stepSize       时间步长（picoseconds）
     */
    ConstantVIntegrator(double stepSize);

    // ═══════════════════════════════════════════════════════════
    // 物理参数（对应教授代码）
    // ═══════════════════════════════════════════════════════════

    /**
     * 设置电压（Volts）
     * 对应：教授的Cathode.Voltage / Anode.Voltage
     */
    void setVoltage(double voltage);
    double getVoltage() const { return voltageVolts; }

    /**
     * 设置真空间隙Lgap（nanometers）
     * 对应：教授的MMsys.Lgap
     */
    void setLgap(double gap);
    double getLgap() const { return Lgap; }

    /**
     * 设置电池厚度Lcell（nanometers）
     * 对应：教授的MMsys.Lcell
     */
    void setLcell(double cell);
    double getLcell() const { return Lcell; }

    /**
     * 设置电极总面积（nm^2）
     * 对应：教授的Cathode.area / Anode.area
     */
    void setTotalArea(double area);
    double getTotalArea() const { return totalArea; }

    /**
     * 设置阴极z位置（nanometers）
     */
    void setZCathode(double z);
    double getZCathode() const { return z_cathode; }

    /**
     * 设置阳极z位置（nanometers）
     */
    void setZAnode(double z);
    double getZAnode() const { return z_anode; }

    // ═══════════════════════════════════════════════════════════
    // SCF参数
    // ═══════════════════════════════════════════════════════════

    /**
     * 设置SCF迭代次数
     * 对应：教授的Niterations参数（默认4）
     */
    void setNumSCFIterations(int n);
    int getNumSCFIterations() const { return nIterations; }

    /**
     * 设置SCF更新频率（每N步做一次SCF）
     * 对应：教授的freq_charge_update_fs参数
     */
    void setSCFFrequency(int freq);
    int getSCFFrequency() const { return scf_frequency; }

    // ═══════════════════════════════════════════════════════════
    // 电极原子设置
    // ═══════════════════════════════════════════════════════════

    /**
     * 添加阴极原子
     *
     * @param particle   粒子索引
     * @param area       该原子的面积（nm^2）
     */
    int addCathodeAtom(int particle, double area);
    int getNumCathodeAtoms() const { return cathodeAtoms.size(); }
    void getCathodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * 添加阳极原子
     */
    int addAnodeAtom(int particle, double area);
    int getNumAnodeAtoms() const { return anodeAtoms.size(); }
    void getAnodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * 添加电解质原子
     *
     * @param particle   粒子索引
     * @param charge     固定电荷（elementary charge）
     */
    int addElectrolyteAtom(int particle, double charge);
    int getNumElectrolyteAtoms() const { return electrolyteAtoms.size(); }
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;

    // ═══════════════════════════════════════════════════════════
    // Integrator接口实现
    // ═══════════════════════════════════════════════════════════

    /**
     * 推进模拟
     */
    void step(int steps) override;

protected:
    void initialize(OpenMM::ContextImpl& context) override;
    void cleanup() override;
    std::vector<std::string> getKernelNames() override;
    double computeKineticEnergy() override;

private:
    // 物理参数
    double voltageVolts;    // 电压（V）
    double voltageKjMol;    // 电压（kJ/mol，内部使用）
    double Lgap;            // 真空间隙（nm）
    double Lcell;           // 电池厚度（nm）
    double totalArea;       // 电极总面积（nm^2）
    double z_cathode;       // 阴极z位置（nm）
    double z_anode;         // 阳极z位置（nm）

    // SCF参数
    int nIterations;        // SCF迭代次数
    int scf_frequency;      // SCF更新频率

    // 电极原子
    struct CathodeAtomInfo {
        int particle;
        double area;
        CathodeAtomInfo(int p, double a) : particle(p), area(a) {}
    };
    struct AnodeAtomInfo {
        int particle;
        double area;
        AnodeAtomInfo(int p, double a) : particle(p), area(a) {}
    };
    struct ElectrolyteAtomInfo {
        int particle;
        double charge;
        ElectrolyteAtomInfo(int p, double c) : particle(p), charge(c) {}
    };

    std::vector<CathodeAtomInfo> cathodeAtoms;
    std::vector<AnodeAtomInfo> anodeAtoms;
    std::vector<ElectrolyteAtomInfo> electrolyteAtoms;

    OpenMM::Kernel kernel;
};

} // namespace ConstantVPlugin

#endif /*OPENMM_CONSTANTVINTEGRATOR_H_*/
