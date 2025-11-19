#include "ReferenceConstantVKernels.h"
#include "ConstantVForce.h"
#include "ConstantVIntegrator.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/reference/RealVec.h"
#include "openmm/reference/ReferencePlatform.h"
#include "ReferenceConstraints.h"
#include "ReferenceVirtualSites.h"
#include <cmath>
#include <iostream>

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

// ═══════════════════════════════════════════════════════════
// 辅助函数（从Reference Platform获取数据）
// ═══════════════════════════════════════════════════════════

static vector<Vec3>& extractPositions(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->positions;
}

static vector<Vec3>& extractVelocities(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->velocities;
}

static vector<Vec3>& extractForces(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->forces;
}

static ReferenceConstraints& extractConstraints(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->constraints;
}

static ReferenceVirtualSites& extractVirtualSites(ContextImpl& context) {
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    return *data->virtualSites;
}

// ═══════════════════════════════════════════════════════════
// 常数定义（教授算法）
// 翻译自: Fixed_Voltage_routines.py::36-38
// ═══════════════════════════════════════════════════════════

// Line 36: conversion_nmBohr = 18.8973
static constexpr double CONVERSION_NMBOHR = 18.8973;

// Line 37: conversion_KjmolNm_Au = conversion_nmBohr / 2625.5
static constexpr double CONVERSION_KJMOLNM_AU = CONVERSION_NMBOHR / 2625.5;  // = 0.00719924...

// Line 38: conversion_eV_Kjmol = 96.487
static constexpr double CONVERSION_EV_KJMOL = 96.487;

// Threshold (MM_classes.py:48)
// self.small_threshold = 1e-6
static constexpr double SMALL_THRESHOLD = 1e-6;

// ═══════════════════════════════════════════════════════════
// initialize() - 缓存参数
// 注意：这个函数暂时保留原有逻辑，等Force类修改后再更新
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::initialize(const System& system, const ConstantVForce& force) {
    // Initialize lazy flag
    chargesInitialized = false;
    // ═══════════════════════════════════════════════════════════
    // 从Force获取参数（教授算法需要的所有参数）
    // ═══════════════════════════════════════════════════════════

    // 查找NonbondedForce
    nonbondedForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        const NonbondedForce* nbForce = dynamic_cast<const NonbondedForce*>(&system.getForce(i));
        if (nbForce != nullptr) {
            nonbondedForce = const_cast<NonbondedForce*>(nbForce);
            break;
        }
    }

    if (nonbondedForce == nullptr)
        throw OpenMMException("ConstantVForce: NonbondedForce not found");

    // 缓存sigma和epsilon
    int numParticles = system.getNumParticles();
    particleSigmas.resize(numParticles);
    particleEpsilons.resize(numParticles);
    currentCharges.resize(numParticles, 0.0);

    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);
        particleSigmas[i] = sigma;
        particleEpsilons[i] = epsilon;
        currentCharges[i] = charge;  // 初始化当前电荷
    }

    // ───────────────────────────────────────────────────────
    // 从Force获取阴极原子
    // ───────────────────────────────────────────────────────
    int numCathode = force.getNumCathodeAtoms();
    cathodeAtomIndices.resize(numCathode);
    areaPerAtom.resize(numCathode);

    for (int i = 0; i < numCathode; i++) {
        int particle;
        double area;
        force.getCathodeAtomParameters(i, particle, area);
        cathodeAtomIndices[i] = particle;
        areaPerAtom[i] = area;
    }

    // ───────────────────────────────────────────────────────
    // 从Force获取阳极原子
    // ───────────────────────────────────────────────────────
    int numAnode = force.getNumAnodeAtoms();
    anodeAtomIndices.resize(numAnode);
    // areaPerAtom继续追加（阴极在前，阳极在后）
    int cathodeSize = areaPerAtom.size();
    areaPerAtom.resize(cathodeSize + numAnode);

    for (int i = 0; i < numAnode; i++) {
        int particle;
        double area;
        force.getAnodeAtomParameters(i, particle, area);
        anodeAtomIndices[i] = particle;
        areaPerAtom[cathodeSize + i] = area;
    }

    // ───────────────────────────────────────────────────────
    // 从Force获取电解质原子
    // ───────────────────────────────────────────────────────
    int numElectrolyte = force.getNumElectrolyteAtoms();
    electrolyteAtomIndices.resize(numElectrolyte);
    electrolyteCharges.resize(numElectrolyte);

    for (int i = 0; i < numElectrolyte; i++) {
        int particle;
        double charge;
        force.getElectrolyteAtomParameters(i, particle, charge);
        electrolyteAtomIndices[i] = particle;
        electrolyteCharges[i] = charge;
    }

    // ───────────────────────────────────────────────────────
    // 从Force获取系统几何参数
    // ───────────────────────────────────────────────────────
    voltage = force.getVoltage() * 96.487;  // V -> kJ/mol（完全照抄教授的转换）
    Lgap = force.getLgap();
    Lcell = force.getLcell();
    totalArea = force.getTotalArea();
    z_cathode = force.getZCathode();
    z_anode = force.getZAnode();

    // ───────────────────────────────────────────────────────
    // 从Force获取SCF参数
    // ───────────────────────────────────────────────────────
    nIterations = force.getNumIterations();

    // ═══════════════════════════════════════════════════════════
    // OpenMM Plugin Contract: initialize() must be side-effect free!
    // Charge initialization is DEFERRED to first execute() call
    // (See initializeElectrodeCharges() below)
    // ═══════════════════════════════════════════════════════════
}

// ═══════════════════════════════════════════════════════════
// initializeElectrodeCharges() - 初始化電極電荷（延遲到execute()）
// 對應Python: initialize_Charge() in Fixed_Voltage_routines.py:278-303
// OpenMM Plugin Contract: This must be called from execute(), NOT initialize()!
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::initializeElectrodeCharges(ContextImpl& context) {
    // Line 286-288: 檢查電壓是否很小
    bool flag_small = false;
    if (fabs(voltage) < 0.01) {
        std::cout << "[Reference] Adding small value to initial charges for small Voltage input..." << std::endl;
        flag_small = true;
    }

    // Line 291-300: 陰極初始電荷（sign=+1.0）
    for (size_t i = 0; i < cathodeAtomIndices.size(); i++) {
        int atomIdx = cathodeAtomIndices[i];
        // Line 293: 計算初始電荷
        double q_i = 1.0 / (4.0 * M_PI) * areaPerAtom[i] *
                     (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOLNM_AU;
        // Line 294-296: 如果電壓很小，添加小值防止數值為零
        if (flag_small) {
            q_i = q_i + SMALL_THRESHOLD;  // Cathode為正
        }
        // Line 299-300: 設置電荷和LJ參數
        currentCharges[atomIdx] = q_i;
        nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
    }

    // Line 291-300: 陽極初始電荷（sign=-1.0）
    for (size_t i = 0; i < anodeAtomIndices.size(); i++) {
        int atomIdx = anodeAtomIndices[i];
        // Line 293: 計算初始電荷（負號）
        double q_i = -1.0 / (4.0 * M_PI) * areaPerAtom[cathodeAtomIndices.size() + i] *
                     (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOLNM_AU;
        // Line 294-296: 如果電壓很小，添加小值防止數值為零
        if (flag_small) {
            q_i = q_i - SMALL_THRESHOLD;  // Anode為負
        }
        // Line 299-300: 設置電荷和LJ參數
        currentCharges[atomIdx] = q_i;
        nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
    }

    // ✅ CRITICAL: Update OpenMM's internal state!
    // Without this, NonbondedForce won't see the updated charges
    nonbondedForce->updateParametersInContext(context.getOwner());

    chargesInitialized = true;
    std::cout << "[Reference] Electrode charges initialized and context updated" << std::endl;
}

// ═══════════════════════════════════════════════════════════
// computeElectrodeChargeAnalytic()
// 翻译自: Fixed_Voltage_routines.py::compute_Electrode_charge_analytic (318-345行)
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::computeElectrodeChargeAnalytic(
    const vector<int>& electrodeAtomIndices,
    const vector<Vec3>& positions,
    const string& electrodeType,
    double z_opposite,
    double& Q_analytic
) {
    // Line 319-322: 确定符号（完全照抄）
    double sign = 1.0;
    if (electrodeType == "anode") {
        sign = -1.0;
    }

    // Line 324-325: 几何贡献（完全照抄公式）
    // self.Q_analytic = sign / (4.0 * numpy.pi) * self.sheet_area *
    //                   (self.Voltage / MMsys.Lgap + self.Voltage / MMsys.Lcell) *
    //                   conversion_KjmolNm_Au
    Q_analytic = sign / (4.0 * M_PI) * totalArea *
                 (voltage / Lgap + voltage / Lcell) *
                 CONVERSION_KJMOLNM_AU;

    // Line 327-333: 电解质镜像电荷贡献（完全照抄）
    // for index in MMsys.electrolyte_atom_indices:
    //     (q_i, sig, eps) = MMsys.nbondedForce.getParticleParameters(index)
    //     z_atom = positions[index][2]._value
    //     z_distance = abs(z_atom - z_opposite)
    //     self.Q_analytic += (z_distance / MMsys.Lcell) * (-q_i._value)
    for (size_t i = 0; i < electrolyteAtomIndices.size(); i++) {
        int index = electrolyteAtomIndices[i];
        // 修復Bug #4: 每次從NonbondedForce實時讀取電荷（極化模擬中會變化）
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(index, charge, sigma, epsilon);
        double q_i = charge;
        double z_atom = positions[index][2];    // OpenMM Vec3的z分量
        double z_distance = fabs(z_atom - z_opposite);
        // 完全照抄Python公式
        Q_analytic += (z_distance / Lcell) * (-q_i);
    }

    // Line 335-344: 导体贡献（第一版跳过）
    // TODO: 实现Buckyball/Nanotube支持
}

// ═══════════════════════════════════════════════════════════
// scaleChargesAnalytic()
// 翻译自: Fixed_Voltage_routines.py::Scale_charges_analytic (354-372行)
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::scaleChargesAnalytic(
    const vector<int>& electrodeAtomIndices,
    double Q_analytic,
    bool printFlag
) {
    // Line 355-356: 计算数值总电荷（完全照抄）
    // Q_numeric = self.get_total_charge()
    double Q_numeric = 0.0;
    for (int atomIdx : electrodeAtomIndices) {
        Q_numeric += currentCharges[atomIdx];
    }

    // Line 358-359: 打印（可选，完全照抄）
    if (printFlag) {
        cout << "Q_numeric = " << Q_numeric
             << ", Q_analytic = " << Q_analytic << endl;
    }

    // Line 361-364: 计算缩放因子，防止除零（完全照抄）
    // scale_factor = -1
    // if abs(Q_numeric) > MMsys.small_threshold:
    //     scale_factor = self.Q_analytic / Q_numeric
    double scale_factor = -1.0;
    if (fabs(Q_numeric) > SMALL_THRESHOLD) {
        scale_factor = Q_analytic / Q_numeric;
    }

    // Line 366-371: 缩放所有电极电荷（完全照抄）
    // if scale_factor > 0.0:
    //     for atom in self.electrode_atoms:
    //         atom.charge = atom.charge * scale_factor
    //         MMsys.nbondedForce.setParticleParameters(atom.atom_index, atom.charge, 1.0, 0.0)
    if (scale_factor > 0.0) {
        for (int atomIdx : electrodeAtomIndices) {
            currentCharges[atomIdx] = currentCharges[atomIdx] * scale_factor;
            nonbondedForce->setParticleParameters(
                atomIdx,
                currentCharges[atomIdx],
                1.0,
                0.0
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════
// execute() - 主循环（SCF迭代）
// 翻译自: MM_classes.py::Poisson_solver_fixed_voltage (287-374行)
// ═══════════════════════════════════════════════════════════

double ReferenceCalcConstantVKernel::execute(
    ContextImpl& context,
    bool includeForces,
    bool includeEnergy
) {
    // Lazy initialization on first execute() call
    // This follows OpenMM plugin contract: initialize() must be side-effect free
    if (!chargesInitialized) {
        std::cout << "[Reference] First execute() call - initializing electrode charges" << std::endl;
        initializeElectrodeCharges(context);  // Pass context for updateParametersInContext()
    }

    const int N_cathode = cathodeAtomIndices.size();
    const int N_anode = anodeAtomIndices.size();

    std::cout << "[Reference Debug] execute() called, N_cathode=" << N_cathode
              << ", N_anode=" << N_anode << std::endl;

    if (N_cathode == 0 && N_anode == 0)
        return 0.0;

    // ═══════════════════════════════════════════════════════════
    // 阶段0：计算解析总电荷（Green's Reciprocity）
    // Line 295-300
    // ═══════════════════════════════════════════════════════════

    // Line 295-297: 获取位置（完全照抄）
    // state = self.simmd.context.getState(getEnergy=False, getForces=False,
    //                                    getVelocities=False, getPositions=True)
    // positions = state.getPositions()
    std::cout << "[Reference Debug] Phase 0: Getting positions..." << std::endl;
    vector<RealVec>& positions = extractPositions(context);
    std::cout << "[Reference Debug] Positions obtained, size=" << positions.size() << std::endl;

    // Line 298-300: 计算解析总电荷（完全照抄）
    // self.Cathode.compute_Electrode_charge_analytic(self, positions, self.Conductor_list,
    //                                                z_opposite=self.Anode.z_pos)
    // self.Anode.compute_Electrode_charge_analytic(self, positions, self.Conductor_list,
    //                                              z_opposite=self.Cathode.z_pos)
    std::cout << "[Reference Debug] Computing cathode analytic charge..." << std::endl;
    computeElectrodeChargeAnalytic(
        cathodeAtomIndices, positions, "cathode",
        z_anode, Q_analytic_cathode
    );
    std::cout << "[Reference Debug] Cathode Q_analytic=" << Q_analytic_cathode << std::endl;

    std::cout << "[Reference Debug] Computing anode analytic charge..." << std::endl;
    computeElectrodeChargeAnalytic(
        anodeAtomIndices, positions, "anode",
        z_cathode, Q_analytic_anode
    );
    std::cout << "[Reference Debug] Anode Q_analytic=" << Q_analytic_anode << std::endl;

    // ═══════════════════════════════════════════════════════════
    // 阶段1：SCF迭代主循环
    // Line 310-365
    // ═══════════════════════════════════════════════════════════

    // Line 310: 开始SCF迭代（完全照抄）
    // for i_iter in range(Niterations):
    std::cout << "[Reference Debug] Starting SCF iterations (nIterations=" << nIterations << ")..." << std::endl;
    for (int iter = 0; iter < nIterations; iter++) {
        std::cout << "[Reference Debug] SCF iteration " << iter << "/" << nIterations << std::endl;

        // ───────────────────────────────────────────────────────
        // Line 313-314: 获取力（完全照抄）
        // ───────────────────────────────────────────────────────
        // state = self.simmd.context.getState(getEnergy=True, getForces=True,
        //                                    getVelocities=False, getPositions=True)
        // forces = state.getForces()
        std::cout << "[Reference Debug] Getting state (forces+positions)..." << std::endl;
        State state = context.getOwner().getState(State::Forces | State::Positions);
        std::cout << "[Reference Debug] State obtained" << std::endl;
        const vector<Vec3>& forces = state.getForces();
        std::cout << "[Reference Debug] Forces size=" << forces.size() << std::endl;

        // ═══════════════════════════════════════════════════════
        // Line 321-335: 更新阴极电荷（完全照抄）
        // ═══════════════════════════════════════════════════════

        // for atom in self.Cathode.electrode_atoms:
        std::cout << "[Reference Debug] Updating cathode charges..." << std::endl;
        for (size_t i = 0; i < cathodeAtomIndices.size(); i++) {
            int atomIdx = cathodeAtomIndices[i];

            // Line 324-325: 获取旧电荷（完全照抄）
            // index = atom.atom_index
            // q_i_old = atom.charge
            double q_i_old = currentCharges[atomIdx];

            // Line 327: 从力计算电场，防止除零（完全照抄）
            // Ez_external = (forces[index][2]._value / q_i_old) if abs(q_i_old) > (0.9*self.small_threshold) else 0.
            double Ez_external = 0.0;
            if (fabs(q_i_old) > (0.9 * SMALL_THRESHOLD)) {
                Ez_external = forces[atomIdx][2] / q_i_old;
            }

            // Line 330: 边界条件求解新电荷（完全照抄公式）
            // q_i = 2.0 / (4.0 * numpy.pi) * self.Cathode.area_atom *
            //       (self.Cathode.Voltage / self.Lgap + Ez_external) * conversion_KjmolNm_Au
            double q_i = 2.0 / (4.0 * M_PI) * areaPerAtom[i] *
                        (voltage / Lgap + Ez_external) *
                        CONVERSION_KJMOLNM_AU;

            // Line 332-333: 防止电荷归零（完全照抄）
            // if abs(q_i) < self.small_threshold:
            //     q_i = self.small_threshold  # Cathode为正
            if (fabs(q_i) < SMALL_THRESHOLD) {
                q_i = SMALL_THRESHOLD;  // Cathode为正
            }

            // Line 334-335: 更新（完全照抄）
            // atom.charge = q_i
            // self.nbondedForce.setParticleParameters(index, q_i, 1.0, 0.0)
            currentCharges[atomIdx] = q_i;
            nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
        }

        // ═══════════════════════════════════════════════════════
        // Line 337-350: 更新阳极电荷（完全照抄，符号相反）
        // ═══════════════════════════════════════════════════════

        // for atom in self.Anode.electrode_atoms:
        std::cout << "[Reference Debug] Updating anode charges..." << std::endl;
        for (size_t i = 0; i < anodeAtomIndices.size(); i++) {
            int atomIdx = anodeAtomIndices[i];

            // Line 339-340: 获取旧电荷（完全照抄）
            // index = atom.atom_index
            // q_i_old = atom.charge
            double q_i_old = currentCharges[atomIdx];

            // Line 342: 从力计算电场，防止除零（完全照抄）
            // Ez_external = (forces[index][2]._value / q_i_old) if abs(q_i_old) > (0.9*self.small_threshold) else 0.
            double Ez_external = 0.0;
            if (fabs(q_i_old) > (0.9 * SMALL_THRESHOLD)) {
                Ez_external = forces[atomIdx][2] / q_i_old;
            }

            // Line 345: 边界条件（注意：-2.0不是2.0，完全照抄）
            // q_i = -2.0 / (4.0 * numpy.pi) * self.Anode.area_atom *
            //       (self.Anode.Voltage / self.Lgap + Ez_external) * conversion_KjmolNm_Au
            double q_i = -2.0 / (4.0 * M_PI) * areaPerAtom[cathodeAtomIndices.size() + i] *
                        (voltage / Lgap + Ez_external) *
                        CONVERSION_KJMOLNM_AU;

            // Line 347-348: 防止电荷归零（完全照抄）
            // if abs(q_i) < self.small_threshold:
            //     q_i = -1.0 * self.small_threshold  # Anode为负
            if (fabs(q_i) < SMALL_THRESHOLD) {
                q_i = -1.0 * SMALL_THRESHOLD;  // Anode为负
            }

            // Line 349-350: 更新（完全照抄）
            // atom.charge = q_i
            // self.nbondedForce.setParticleParameters(index, q_i, 1.0, 0.0)
            currentCharges[atomIdx] = q_i;
            nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
        }

        // ───────────────────────────────────────────────────────
        // Line 352-360: Conductor处理（第一版跳过）
        // ───────────────────────────────────────────────────────
        // TODO: 实现Buckyball/Nanotube支持

        // ═══════════════════════════════════════════════════════
        // Line 362-363: Green's校正（完全照抄）
        // ═══════════════════════════════════════════════════════
        // self.Scale_charges_analytic_general()
        std::cout << "[Reference Debug] Scaling charges (Green's correction)..." << std::endl;
        scaleChargesAnalytic(cathodeAtomIndices, Q_analytic_cathode, false);
        scaleChargesAnalytic(anodeAtomIndices, Q_analytic_anode, false);
        std::cout << "[Reference Debug] Charges scaled" << std::endl;

        // ═══════════════════════════════════════════════════════
        // Line 365: 更新OpenMM context（完全照抄）
        // ═══════════════════════════════════════════════════════
        // self.nbondedForce.updateParametersInContext(self.simmd.context)
        std::cout << "[Reference Debug] Calling updateParametersInContext..." << std::endl;
        nonbondedForce->updateParametersInContext(context.getOwner());
        std::cout << "[Reference Debug] updateParametersInContext completed" << std::endl;
    }
    std::cout << "[Reference Debug] SCF iterations completed" << std::endl;

    // ───────────────────────────────────────────────────────
    // Line 367-368: 最后一次打印（完全照抄）
    // ───────────────────────────────────────────────────────
    // self.Scale_charges_analytic_general(print_flag=True)
    std::cout << "[Reference Debug] Final scaling with print..." << std::endl;
    scaleChargesAnalytic(cathodeAtomIndices, Q_analytic_cathode, true);
    scaleChargesAnalytic(anodeAtomIndices, Q_analytic_anode, true);

    std::cout << "[Reference Debug] execute() finished successfully" << std::endl;
    return 0.0;  // 不贡献能量
}

// ═══════════════════════════════════════════════════════════
// copyParametersToContext() - 参数更新
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::copyParametersToContext(ContextImpl& context, const ConstantVForce& force) {
    // TODO: 等ConstantVForce修改后实现
}

// ═══════════════════════════════════════════════════════════
// ReferenceIntegrateConstantVStepKernel 实现
// 翻译自: MM_classes.py::Poisson_solver_fixed_voltage
// ═══════════════════════════════════════════════════════════

void ReferenceIntegrateConstantVStepKernel::initialize(
    const System& system,
    const ConstantVIntegrator& integrator
) {
    int numParticles = system.getNumParticles();
    particleInvMass.resize(numParticles);
    currentCharges.resize(numParticles, 0.0);

    for (int i = 0; i < numParticles; i++) {
        double mass = system.getParticleMass(i);
        particleInvMass[i] = (mass == 0.0 ? 0.0 : 1.0/mass);
    }

    // 从Integrator获取参数
    voltage = integrator.getVoltage() * CONVERSION_EV_KJMOL;  // 转换到kJ/mol
    Lgap = integrator.getLgap();
    Lcell = integrator.getLcell();
    totalArea = integrator.getTotalArea();
    z_cathode = integrator.getZCathode();
    z_anode = integrator.getZAnode();
    nIterations = integrator.getNumSCFIterations();
    scf_frequency = integrator.getSCFFrequency();

    // 获取电极原子
    int numCathode = integrator.getNumCathodeAtoms();
    cathodeAtomIndices.resize(numCathode);
    cathodeAreas.resize(numCathode);
    for (int i = 0; i < numCathode; i++) {
        int particle;
        double area;
        integrator.getCathodeAtomParameters(i, particle, area);
        cathodeAtomIndices[i] = particle;
        cathodeAreas[i] = area;
    }

    int numAnode = integrator.getNumAnodeAtoms();
    anodeAtomIndices.resize(numAnode);
    anodeAreas.resize(numAnode);
    for (int i = 0; i < numAnode; i++) {
        int particle;
        double area;
        integrator.getAnodeAtomParameters(i, particle, area);
        anodeAtomIndices[i] = particle;
        anodeAreas[i] = area;
    }

    int numElectrolyte = integrator.getNumElectrolyteAtoms();
    electrolyteAtomIndices.resize(numElectrolyte);
    electrolyteCharges.resize(numElectrolyte);
    for (int i = 0; i < numElectrolyte; i++) {
        int particle;
        double charge;
        integrator.getElectrolyteAtomParameters(i, particle, charge);
        electrolyteAtomIndices[i] = particle;
        electrolyteCharges[i] = charge;
    }

    // 找到NonbondedForce
    nonbondedForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        const Force& force = system.getForce(i);
        if (dynamic_cast<const NonbondedForce*>(&force) != nullptr) {
            // 需要const_cast因为我们要修改电荷参数
            nonbondedForce = const_cast<NonbondedForce*>(dynamic_cast<const NonbondedForce*>(&force));
            break;
        }
    }

    if (nonbondedForce == nullptr)
        throw OpenMMException("ConstantVIntegrator: NonbondedForce not found in System");

    // 从NonbondedForce初始化currentCharges（修復Bug #3）
    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);
        currentCharges[i] = charge;
    }

    // ═══════════════════════════════════════════════════════════
    // 修復Bug #6: 計算並設置初始電荷
    // 對應Python: initialize_Charge() in Fixed_Voltage_routines.py:278-303
    // ═══════════════════════════════════════════════════════════

    // Line 286-288: 檢查電壓是否很小
    bool flag_small = false;
    if (fabs(voltage) < 0.01) {
        std::cout << "adding small value to initial charges in initialize_Charge routine for small Voltage input..." << std::endl;
        flag_small = true;
    }

    // Line 291-300: 陰極初始電荷（sign=+1.0）
    for (size_t i = 0; i < cathodeAtomIndices.size(); i++) {
        int atomIdx = cathodeAtomIndices[i];
        // Line 293: 計算初始電荷
        double q_i = 1.0 / (4.0 * M_PI) * cathodeAreas[i] *
                     (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOLNM_AU;
        // Line 294-296: 如果電壓很小，添加小值防止數值為零
        if (flag_small) {
            q_i = q_i + SMALL_THRESHOLD;  // Cathode為正
        }
        // Line 299-300: 設置電荷和LJ參數
        currentCharges[atomIdx] = q_i;
        nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
    }

    // Line 291-300: 陽極初始電荷（sign=-1.0）
    for (size_t i = 0; i < anodeAtomIndices.size(); i++) {
        int atomIdx = anodeAtomIndices[i];
        // Line 293: 計算初始電荷（負號）
        double q_i = -1.0 / (4.0 * M_PI) * anodeAreas[i] *
                     (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOLNM_AU;
        // Line 294-296: 如果電壓很小，添加小值防止數值為零
        if (flag_small) {
            q_i = q_i - SMALL_THRESHOLD;  // Anode為負
        }
        // Line 299-300: 設置電荷和LJ參數
        currentCharges[atomIdx] = q_i;
        nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
    }
}

// ═══════════════════════════════════════════════════════════
// execute() - 执行一个积分步
// ═══════════════════════════════════════════════════════════

void ReferenceIntegrateConstantVStepKernel::execute(
    ContextImpl& context,
    const ConstantVIntegrator& integrator
) {
    // ═══════════════════════════════════════════════════════════
    // 教授的顺序: 先SCF，后MD积分（参考run_openMM_refactored.py Line 242-244）
    // MMsys.Poisson_solver_fixed_voltage(Niterations=4)  # 先更新电荷
    // MMsys.simmd.step(freq_charge_update_fs)            # 后积分（自动计算新力）
    // ═══════════════════════════════════════════════════════════

    // 步骤1: 每scf_frequency步做一次SCF（在积分之前！）
    int stepCount = context.getStepCount();
    if (stepCount % scf_frequency == 0) {
        scf_iteration(context);  // 更新电荷，但最后一次迭代后力是旧的
    }

    // 步骤2: 重新计算力（使用最新电荷）
    // 对应教授的 simmd.step() 内部会自动用新电荷计算力
    // ⭐ CRITICAL: Exclude ConstantVForce (Group 31) to prevent double SCF execution
    int forceGroups = context.getIntegrator().getIntegrationForceGroups();
    forceGroups &= ~(1U << 31);  // Exclude Group 31 (ConstantVForce)
    context.calcForcesAndEnergy(true, false, forceGroups);

    // 步骤3: Verlet积分（参考DrudeSCFIntegrator）
    vector<Vec3>& pos = extractPositions(context);
    vector<Vec3>& vel = extractVelocities(context);
    vector<Vec3>& force = extractForces(context);

    int numParticles = particleInvMass.size();
    double dt = integrator.getStepSize();

    // 更新速度和位置
    for (int i = 0; i < numParticles; i++) {
        if (particleInvMass[i] != 0.0) {
            vel[i] += force[i] * particleInvMass[i] * dt;
            pos[i] += vel[i] * dt;
        }
    }

    // 应用约束（如果有）
    extractConstraints(context).apply(pos, pos, particleInvMass, integrator.getConstraintTolerance());

    // 更新时间和步数
    ReferencePlatform::PlatformData* data = reinterpret_cast<ReferencePlatform::PlatformData*>(context.getPlatformData());
    data->time += integrator.getStepSize();
    data->stepCount++;
}

// ═══════════════════════════════════════════════════════════
// scf_iteration() - SCF主循环
// 翻译自: MM_classes.py::Poisson_solver_fixed_voltage (Line 310-365)
// ═══════════════════════════════════════════════════════════

void ReferenceIntegrateConstantVStepKernel::scf_iteration(ContextImpl& context) {
    // Line 296-300: 获取位置，计算解析电荷
    vector<Vec3>& positions = extractPositions(context);

    computeElectrodeChargeAnalytic(
        cathodeAtomIndices, positions, "cathode",
        z_anode, Q_analytic_cathode
    );
    computeElectrodeChargeAnalytic(
        anodeAtomIndices, positions, "anode",
        z_cathode, Q_analytic_anode
    );

    // Line 310: 开始SCF迭代
    for (int iter = 0; iter < nIterations; iter++) {
        // Line 313-314: 获取力（在Integrator中调用是安全的！）
        context.calcForcesAndEnergy(true, false, context.getIntegrator().getIntegrationForceGroups());
        vector<Vec3>& forces = extractForces(context);

        // ═══════════════════════════════════════════════════════════
        // Line 323-335: 更新阴极电荷（完全照抄）
        // ═══════════════════════════════════════════════════════════
        for (size_t i = 0; i < cathodeAtomIndices.size(); i++) {
            int atomIdx = cathodeAtomIndices[i];
            double q_i_old = currentCharges[atomIdx];

            // Line 327: Ez从力计算，防止除零（0.9不是1.0！）
            double Ez_external = 0.0;
            if (fabs(q_i_old) > (0.9 * SMALL_THRESHOLD)) {
                Ez_external = forces[atomIdx][2] / q_i_old;
            }

            // Line 330: 边界条件（阴极：+2.0）
            double q_i = 2.0 / (4.0 * M_PI) * cathodeAreas[i] *
                        (voltage / Lgap + Ez_external) *
                        CONVERSION_KJMOLNM_AU;

            // Line 332-333: 防归零（阴极：+SMALL_THRESHOLD）
            if (fabs(q_i) < SMALL_THRESHOLD) {
                q_i = SMALL_THRESHOLD;
            }

            currentCharges[atomIdx] = q_i;
            nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
        }

        // ═══════════════════════════════════════════════════════════
        // Line 338-350: 更新阳极电荷（完全照抄）
        // ═══════════════════════════════════════════════════════════
        for (size_t i = 0; i < anodeAtomIndices.size(); i++) {
            int atomIdx = anodeAtomIndices[i];
            double q_i_old = currentCharges[atomIdx];

            // Line 342: Ez从力计算，防止除零
            double Ez_external = 0.0;
            if (fabs(q_i_old) > (0.9 * SMALL_THRESHOLD)) {
                Ez_external = forces[atomIdx][2] / q_i_old;
            }

            // Line 345: 边界条件（阳极：-2.0）
            double q_i = -2.0 / (4.0 * M_PI) * anodeAreas[i] *
                        (voltage / Lgap + Ez_external) *
                        CONVERSION_KJMOLNM_AU;

            // Line 347-348: 防归零（阳极：-1.0*SMALL_THRESHOLD）
            if (fabs(q_i) < SMALL_THRESHOLD) {
                q_i = -1.0 * SMALL_THRESHOLD;
            }

            currentCharges[atomIdx] = q_i;
            nonbondedForce->setParticleParameters(atomIdx, q_i, 1.0, 0.0);
        }

        // Line 362-365: Green's校正 + 更新Context
        scaleChargesAnalytic(cathodeAtomIndices, Q_analytic_cathode);
        scaleChargesAnalytic(anodeAtomIndices, Q_analytic_anode);
        nonbondedForce->updateParametersInContext(context.getOwner());
    }
}

// ═══════════════════════════════════════════════════════════
// computeElectrodeChargeAnalytic() - 计算解析电荷
// 翻译自: Fixed_Voltage_routines.py (Line 318-345)
// ═══════════════════════════════════════════════════════════

void ReferenceIntegrateConstantVStepKernel::computeElectrodeChargeAnalytic(
    const vector<int>& electrodeAtomIndices,
    const vector<Vec3>& positions,
    const string& electrodeType,
    double z_opposite,
    double& Q_analytic
) {
    // Line 319-322: 符号判断
    double sign = 1.0;
    if (electrodeType == "anode") {
        sign = -1.0;
    }

    // Line 324-325: 几何贡献
    Q_analytic = sign / (4.0 * M_PI) * totalArea *
                 (voltage / Lgap + voltage / Lcell) *
                 CONVERSION_KJMOLNM_AU;

    // Line 327-333: 镜像电荷贡献
    for (size_t i = 0; i < electrolyteAtomIndices.size(); i++) {
        int index = electrolyteAtomIndices[i];
        // 修復Bug #4: 每次從NonbondedForce實時讀取電荷（極化模擬中會變化）
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(index, charge, sigma, epsilon);
        double q_i = charge;
        double z_atom = positions[index][2];
        double z_distance = fabs(z_atom - z_opposite);
        Q_analytic += (z_distance / Lcell) * (-q_i);
    }
}

// ═══════════════════════════════════════════════════════════
// scaleChargesAnalytic() - Green's校正
// 翻译自: Fixed_Voltage_routines.py (Line 354-372)
// ═══════════════════════════════════════════════════════════

void ReferenceIntegrateConstantVStepKernel::scaleChargesAnalytic(
    const vector<int>& electrodeAtomIndices,
    double Q_analytic
) {
    // Line 355-356: 计算数值总电荷
    double Q_numeric = 0.0;
    for (int atomIdx : electrodeAtomIndices) {
        Q_numeric += currentCharges[atomIdx];
    }

    // Line 361-364: 计算缩放因子，防止除零
    double scale_factor = -1.0;
    if (fabs(Q_numeric) > SMALL_THRESHOLD) {
        scale_factor = Q_analytic / Q_numeric;
    }

    // Line 366-371: 缩放所有电极电荷
    if (scale_factor > 0.0) {
        for (int atomIdx : electrodeAtomIndices) {
            currentCharges[atomIdx] = currentCharges[atomIdx] * scale_factor;
            nonbondedForce->setParticleParameters(atomIdx, currentCharges[atomIdx], 1.0, 0.0);
        }
    }
}

// ═══════════════════════════════════════════════════════════
// computeKineticEnergy() - 计算动能
// ═══════════════════════════════════════════════════════════

double ReferenceIntegrateConstantVStepKernel::computeKineticEnergy(
    ContextImpl& context,
    const ConstantVIntegrator& integrator
) {
    vector<Vec3>& vel = extractVelocities(context);
    double energy = 0.0;
    for (size_t i = 0; i < particleInvMass.size(); i++) {
        if (particleInvMass[i] != 0.0) {
            double v2 = vel[i].dot(vel[i]);
            energy += v2 / particleInvMass[i];
        }
    }
    return 0.5 * energy;
}
