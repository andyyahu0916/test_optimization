#include "ConstantVIntegrator.h"
#include "ConstantVKernels.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

// 转换常数（对应教授的Fixed_Voltage_routines.py:38）
static constexpr double CONVERSION_EV_KJMOL = 96.487;

// ═══════════════════════════════════════════════════════════
// 构造函数
// ═══════════════════════════════════════════════════════════

ConstantVIntegrator::ConstantVIntegrator(double stepSize) :
    voltageVolts(0.0),
    voltageKjMol(0.0),
    Lgap(0.0),
    Lcell(0.0),
    totalArea(0.0),
    z_cathode(0.0),
    z_anode(0.0),
    nIterations(4),       // 教授默认4次SCF迭代
    scf_frequency(1)      // 默认每步都做SCF
{
    setStepSize(stepSize);
    setConstraintTolerance(1e-5);
}

// ═══════════════════════════════════════════════════════════
// 物理参数方法
// ═══════════════════════════════════════════════════════════

void ConstantVIntegrator::setVoltage(double voltage) {
    voltageVolts = voltage;
    voltageKjMol = voltage * CONVERSION_EV_KJMOL;
}

void ConstantVIntegrator::setLgap(double gap) {
    Lgap = gap;
}

void ConstantVIntegrator::setLcell(double cell) {
    Lcell = cell;
}

void ConstantVIntegrator::setTotalArea(double area) {
    totalArea = area;
}

void ConstantVIntegrator::setZCathode(double z) {
    z_cathode = z;
}

void ConstantVIntegrator::setZAnode(double z) {
    z_anode = z;
}

// ═══════════════════════════════════════════════════════════
// SCF参数方法
// ═══════════════════════════════════════════════════════════

void ConstantVIntegrator::setNumSCFIterations(int n) {
    if (n < 1)
        throw OpenMMException("ConstantVIntegrator: number of SCF iterations must be >= 1");
    nIterations = n;
}

void ConstantVIntegrator::setSCFFrequency(int freq) {
    if (freq < 1)
        throw OpenMMException("ConstantVIntegrator: SCF frequency must be >= 1");
    scf_frequency = freq;
}

// ═══════════════════════════════════════════════════════════
// 电极原子方法
// ═══════════════════════════════════════════════════════════

int ConstantVIntegrator::addCathodeAtom(int particle, double area) {
    cathodeAtoms.push_back(CathodeAtomInfo(particle, area));
    return cathodeAtoms.size() - 1;
}

void ConstantVIntegrator::getCathodeAtomParameters(int index, int& particle, double& area) const {
    particle = cathodeAtoms[index].particle;
    area = cathodeAtoms[index].area;
}

int ConstantVIntegrator::addAnodeAtom(int particle, double area) {
    anodeAtoms.push_back(AnodeAtomInfo(particle, area));
    return anodeAtoms.size() - 1;
}

void ConstantVIntegrator::getAnodeAtomParameters(int index, int& particle, double& area) const {
    particle = anodeAtoms[index].particle;
    area = anodeAtoms[index].area;
}

int ConstantVIntegrator::addElectrolyteAtom(int particle, double charge) {
    electrolyteAtoms.push_back(ElectrolyteAtomInfo(particle, charge));
    return electrolyteAtoms.size() - 1;
}

void ConstantVIntegrator::getElectrolyteAtomParameters(int index, int& particle, double& charge) const {
    particle = electrolyteAtoms[index].particle;
    charge = electrolyteAtoms[index].charge;
}

// ═══════════════════════════════════════════════════════════
// Integrator接口实现
// ═══════════════════════════════════════════════════════════

void ConstantVIntegrator::initialize(ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMMException("This Integrator is already bound to a context");
    context = &contextRef;
    owner = &contextRef.getOwner();
    kernel = context->getPlatform().createKernel(IntegrateConstantVStepKernel::Name(), contextRef);
    kernel.getAs<IntegrateConstantVStepKernel>().initialize(contextRef.getSystem(), *this);
}

void ConstantVIntegrator::cleanup() {
    kernel = Kernel();
}

vector<string> ConstantVIntegrator::getKernelNames() {
    vector<string> names;
    names.push_back(IntegrateConstantVStepKernel::Name());
    return names;
}

double ConstantVIntegrator::computeKineticEnergy() {
    return kernel.getAs<IntegrateConstantVStepKernel>().computeKineticEnergy(*context, *this);
}

void ConstantVIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMMException("This Integrator is not bound to a context!");

    for (int i = 0; i < steps; ++i) {
        context->updateContextState();
        // 不在这里计算力！让execute()负责在正确的时机计算
        kernel.getAs<IntegrateConstantVStepKernel>().execute(*context, *this);
    }
}
