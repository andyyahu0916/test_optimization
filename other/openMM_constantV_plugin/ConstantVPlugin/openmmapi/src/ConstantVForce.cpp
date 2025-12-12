#include "ConstantVForce.h"
#include "internal/ConstantVForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

// 转换常数（对应教授的Fixed_Voltage_routines.py:38）
static constexpr double CONVERSION_EV_KJMOL = 96.487;

// ═══════════════════════════════════════════════════════════
// 构造函数
// ═══════════════════════════════════════════════════════════

ConstantVForce::ConstantVForce() :
    voltageVolts(0.0),
    voltageKjMol(0.0),
    Lgap(0.0),
    Lcell(0.0),
    totalArea(0.0),
    z_cathode(0.0),
    z_anode(0.0),
    nIterations(4)  // 教授默认4次SCF迭代
{
    // 🔥 CRITICAL: Assign to force group 31 to prevent infinite recursion
    // - CudaCalcConstantVKernel::execute() internally calls calcForcesAndEnergy()
    // - If ConstantVForce is in the integration force groups, it would re-trigger itself
    // - Solution: Use a dedicated group (31) that is always masked out internally
    // - See CudaConstantVKernels.cu Line 805-807 for the masking logic
    setForceGroup(31);
}

// ═══════════════════════════════════════════════════════════
// 阴极原子方法
// ═══════════════════════════════════════════════════════════

int ConstantVForce::addCathodeAtom(int particle, double area) {
    cathodeAtoms.push_back(CathodeAtomInfo(particle, area));
    cathodeAtomIndices.push_back(particle);
    cathodeAreas.push_back(area);
    return cathodeAtoms.size() - 1;
}

void ConstantVForce::getCathodeAtomParameters(int index, int& particle, double& area) const {
    particle = cathodeAtoms[index].particle;
    area = cathodeAtoms[index].area;
}

// ═══════════════════════════════════════════════════════════
// 阳极原子方法
// ═══════════════════════════════════════════════════════════

int ConstantVForce::addAnodeAtom(int particle, double area) {
    anodeAtoms.push_back(AnodeAtomInfo(particle, area));
    anodeAtomIndices.push_back(particle);
    anodeAreas.push_back(area);
    return anodeAtoms.size() - 1;
}

void ConstantVForce::getAnodeAtomParameters(int index, int& particle, double& area) const {
    particle = anodeAtoms[index].particle;
    area = anodeAtoms[index].area;
}

// ═══════════════════════════════════════════════════════════
// 电解质原子方法
// ═══════════════════════════════════════════════════════════

int ConstantVForce::addElectrolyteAtom(int particle, double charge) {
    electrolyteAtoms.push_back(ElectrolyteAtomInfo(particle, charge));
    electrolyteAtomIndices.push_back(particle);
    electrolyteCharges.push_back(charge);
    return electrolyteAtoms.size() - 1;
}

void ConstantVForce::getElectrolyteAtomParameters(int index, int& particle, double& charge) const {
    particle = electrolyteAtoms[index].particle;
    charge = electrolyteAtoms[index].charge;
}

// ═══════════════════════════════════════════════════════════
// Buckyball导体方法
// 对应: Buckyball_Virtual class (Fixed_Voltage_routines.py:391-473)
// ═══════════════════════════════════════════════════════════

int ConstantVForce::addBuckyballConductor(const std::vector<int>& virtualAtoms,
                                            const std::vector<int>& realAtoms,
                                            const std::string& electrodeType,
                                            double voltage) {
    // 验证输入（对应Python Original的Line 398-403）
    if (electrodeType != "cathode" && electrodeType != "anode") {
        throw OpenMMException("ConstantVForce::addBuckyballConductor: electrode_type must be 'cathode' or 'anode'");
    }

    if (virtualAtoms.empty()) {
        throw OpenMMException("ConstantVForce::addBuckyballConductor: virtualAtoms list cannot be empty");
    }

    if (realAtoms.empty()) {
        throw OpenMMException("ConstantVForce::addBuckyballConductor: realAtoms list cannot be empty (must input both virtual and real electrode atoms for BuckyBall)");
    }

    // 创建BuckyballConductorInfo对象（对应Python的__init__）
    BuckyballConductorInfo conductor(virtualAtoms, realAtoms, electrodeType, voltage);

    buckyballConductors.push_back(conductor);
    return buckyballConductors.size() - 1;
}

void ConstantVForce::getBuckyballConductorParameters(int index,
                                                       std::vector<int>& virtualAtoms,
                                                       std::vector<int>& realAtoms,
                                                       std::string& electrodeType,
                                                       double& voltage) const {
    const BuckyballConductorInfo& conductor = buckyballConductors[index];
    virtualAtoms = conductor.virtualAtomIndices;
    realAtoms = conductor.realAtomIndices;
    electrodeType = conductor.electrodeType;
    voltage = conductor.voltageVolts;
}

// ═══════════════════════════════════════════════════════════
// Nanotube Conductor 方法
// 对应: Nanotube_Virtual class (Fixed_Voltage_routines.py:482-589)
// ═══════════════════════════════════════════════════════════

int ConstantVForce::addNanotubeConductor(const std::vector<int>& virtualAtoms,
                                          const std::vector<int>& realAtoms,
                                          const std::string& electrodeType,
                                          double voltage,
                                          const std::vector<double>& axis) {
    // 验证输入（对应Python Original的Line 489-494）
    if (electrodeType != "cathode" && electrodeType != "anode") {
        throw OpenMMException("ConstantVForce::addNanotubeConductor: electrode_type must be 'cathode' or 'anode'");
    }

    if (virtualAtoms.empty()) {
        throw OpenMMException("ConstantVForce::addNanotubeConductor: virtualAtoms list cannot be empty");
    }

    if (realAtoms.empty()) {
        throw OpenMMException("ConstantVForce::addNanotubeConductor: realAtoms list cannot be empty (must input both virtual and real electrode atoms for Nanotube)");
    }

    if (axis.size() != 3) {
        throw OpenMMException("ConstantVForce::addNanotubeConductor: axis must be a 3-element vector [ax, ay, az]");
    }

    // 验证axis是单位向量（允许小误差）
    double axisNorm = sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (fabs(axisNorm - 1.0) > 0.01) {
        throw OpenMMException("ConstantVForce::addNanotubeConductor: axis must be a unit vector (norm ~ 1.0)");
    }

    // 创建NanotubeConductorInfo对象（对应Python的__init__）
    NanotubeConductorInfo conductor(virtualAtoms, realAtoms, electrodeType, voltage, axis);

    nanotubeConductors.push_back(conductor);
    return nanotubeConductors.size() - 1;
}

void ConstantVForce::getNanotubeConductorParameters(int index,
                                                     std::vector<int>& virtualAtoms,
                                                     std::vector<int>& realAtoms,
                                                     std::string& electrodeType,
                                                     double& voltage,
                                                     std::vector<double>& axis) const {
    const NanotubeConductorInfo& conductor = nanotubeConductors[index];
    virtualAtoms = conductor.virtualAtomIndices;
    realAtoms = conductor.realAtomIndices;
    electrodeType = conductor.electrodeType;
    voltage = conductor.voltageVolts;
    axis.resize(3);
    axis[0] = conductor.axis[0];
    axis[1] = conductor.axis[1];
    axis[2] = conductor.axis[2];
}

// ═══════════════════════════════════════════════════════════
// 系统几何参数方法
// ═══════════════════════════════════════════════════════════

void ConstantVForce::setVoltage(double voltage) {
    voltageVolts = voltage;
    // 转换成kJ/mol（对应教授的Electrode_Virtual.__init__:88）
    // self.Voltage = Voltage * conversion_eV_Kjmol
    voltageKjMol = voltage * CONVERSION_EV_KJMOL;
}

void ConstantVForce::setLgap(double gap) {
    Lgap = gap;
}

void ConstantVForce::setLcell(double cell) {
    Lcell = cell;
}

void ConstantVForce::setTotalArea(double area) {
    totalArea = area;
}

void ConstantVForce::setZCathode(double z) {
    z_cathode = z;
}

void ConstantVForce::setZAnode(double z) {
    z_anode = z;
}

// ═══════════════════════════════════════════════════════════
// SCF参数方法
// ═══════════════════════════════════════════════════════════

void ConstantVForce::setNumIterations(int n) {
    if (n < 1)
        throw OpenMMException("ConstantVForce: number of iterations must be >= 1");
    nIterations = n;
}

// ═══════════════════════════════════════════════════════════
// ForceImpl创建
// ═══════════════════════════════════════════════════════════

ForceImpl* ConstantVForce::createImpl() const {
    return new ConstantVForceImpl(*this);
}
