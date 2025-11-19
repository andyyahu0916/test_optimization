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
