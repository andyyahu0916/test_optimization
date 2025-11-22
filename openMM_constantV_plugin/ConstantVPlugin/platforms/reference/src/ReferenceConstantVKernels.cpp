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
    // 从Force获取Buckyball导体
    // 对应: Buckyball_Virtual class
    // ───────────────────────────────────────────────────────
    int numBuckyballs = force.getNumBuckyballConductors();
    buckyballConductors.resize(numBuckyballs);

    for (int i = 0; i < numBuckyballs; i++) {
        std::vector<int> virtualAtoms, realAtoms;
        std::string electrodeType;
        double voltage;
        force.getBuckyballConductorParameters(i, virtualAtoms, realAtoms, electrodeType, voltage);

        BuckyballConductor& conductor = buckyballConductors[i];
        conductor.virtualAtomIndices = virtualAtoms;
        conductor.realAtomIndices = realAtoms;
        conductor.electrodeType = electrodeType;
        conductor.voltageKjMol = voltage * 96.487;  // V -> kJ/mol
        conductor.closeThreshold = 1.5;  // nm (默认值，对应Python Line 100)
        conductor.closeToElectrode = true;
        conductor.contactAtomIndex = -1;
        conductor.dr_center_contact = 0.0;
        // 几何参数将在initializeElectrodeCharges()中计算
    }

    // ───────────────────────────────────────────────────────
    // 从Force获取Nanotube导体
    // 对应: Nanotube_Virtual class
    // ───────────────────────────────────────────────────────
    int numNanotubes = force.getNumNanotubeConductors();
    nanotubeConductors.resize(numNanotubes);

    for (int i = 0; i < numNanotubes; i++) {
        std::vector<int> virtualAtoms, realAtoms;
        std::string electrodeType;
        double voltage;
        std::vector<double> axis;
        force.getNanotubeConductorParameters(i, virtualAtoms, realAtoms, electrodeType, voltage, axis);

        NanotubeConductor& conductor = nanotubeConductors[i];
        conductor.virtualAtomIndices = virtualAtoms;
        conductor.realAtomIndices = realAtoms;
        conductor.electrodeType = electrodeType;
        conductor.voltageKjMol = voltage * 96.487;  // V -> kJ/mol
        conductor.axis[0] = axis[0];
        conductor.axis[1] = axis[1];
        conductor.axis[2] = axis[2];
        conductor.closeThreshold = 1.5;  // nm
        conductor.closeToElectrode = true;
        conductor.contactAtomIndex = -1;
        conductor.dr_center_contact = 0.0;
        // 几何参数将在initializeElectrodeCharges()中计算
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

    // ───────────────────────────────────────────────────────
    // Initialize Buckyball conductors (对应 Buckyball_Virtual.__init__)
    // ───────────────────────────────────────────────────────
    if (!buckyballConductors.empty()) {
        // Get positions for geometry calculation
        vector<Vec3>& positions = extractPositions(context);

        for (BuckyballConductor& conductor : buckyballConductors) {
            // Line 424-457: Initialize geometry (center, radius, normals)
            initializeBuckyballGeometry(conductor, positions);

            // Line 459: Find contact neighbor conductor
            findContactNeighborConductor(conductor, positions);
        }

        std::cout << "[Reference] Initialized " << buckyballConductors.size()
                  << " Buckyball conductor(s)" << std::endl;
    }

    // ───────────────────────────────────────────────────────
    // Initialize Nanotube conductors (对应 Nanotube_Virtual.__init__)
    // ───────────────────────────────────────────────────────
    if (!nanotubeConductors.empty()) {
        // Get positions and box vectors for geometry calculation
        vector<Vec3>& positions = extractPositions(context);
        Vec3 boxVectors[3];
        context.getOwner().getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);

        for (NanotubeConductor& conductor : nanotubeConductors) {
            // Line 517-572: Initialize geometry (center, radius, length, normals)
            initializeNanotubeGeometry(conductor, positions, boxVectors);

            // Line 564: Find contact neighbor conductor
            // TODO: Implement findContactNeighborNanotube if needed
            // For now skip - most Nanotubes don't have contact neighbors
        }

        std::cout << "[Reference] Initialized " << nanotubeConductors.size()
                  << " Nanotube conductor(s)" << std::endl;
    }

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
// initializeBuckyballGeometry()
// 翻译自: Buckyball_Virtual.__init__ (Line 424-457)
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::initializeBuckyballGeometry(
    BuckyballConductor& conductor,
    const vector<Vec3>& positions
) {
    int Natoms = conductor.virtualAtomIndices.size();

    // Line 428-436: Find center of buckyball (完全照抄)
    // self.r_center = [ 0.0 , 0.0 , 0.0 ] # in nm
    // for atom in self.electrode_atoms:
    //     self.r_center[0] += positions[atom.atom_index][0]._value
    //     self.r_center[1] += positions[atom.atom_index][1]._value
    //     self.r_center[2] += positions[atom.atom_index][2]._value
    // self.r_center[0] = self.r_center[0] / self.Natoms
    // self.r_center[1] = self.r_center[1] / self.Natoms
    // self.r_center[2] = self.r_center[2] / self.Natoms
    conductor.r_center[0] = 0.0;
    conductor.r_center[1] = 0.0;
    conductor.r_center[2] = 0.0;

    for (int atomIdx : conductor.virtualAtomIndices) {
        conductor.r_center[0] += positions[atomIdx][0];
        conductor.r_center[1] += positions[atomIdx][1];
        conductor.r_center[2] += positions[atomIdx][2];
    }

    conductor.r_center[0] /= Natoms;
    conductor.r_center[1] /= Natoms;
    conductor.r_center[2] /= Natoms;

    // Line 439-446: compute area per atom, get radius from first atom (完全照抄)
    // self.radius=0.0
    // for atom in self.electrode_atoms:
    //     rx = positions[atom.atom_index][0]._value - self.r_center[0]
    //     ry = positions[atom.atom_index][1]._value - self.r_center[1]
    //     rz = positions[atom.atom_index][2]._value - self.r_center[2]
    //     self.radius = sqrt( rx**2 + ry**2 + rz**2 )
    //     break
    // self.area_atom = 4.0 * numpy.pi * self.radius**2 / self.Natoms
    if (Natoms > 0) {
        int firstAtom = conductor.virtualAtomIndices[0];
        double rx = positions[firstAtom][0] - conductor.r_center[0];
        double ry = positions[firstAtom][1] - conductor.r_center[1];
        double rz = positions[firstAtom][2] - conductor.r_center[2];
        conductor.radius = sqrt(rx*rx + ry*ry + rz*rz);
    }

    conductor.area_atom = 4.0 * M_PI * conductor.radius * conductor.radius / Natoms;

    // Line 450-456: calculate surface normal vector at each atom (完全照抄)
    // for atom in self.electrode_atoms:
    //     nx = positions[atom.atom_index][0]._value - self.r_center[0]
    //     ny = positions[atom.atom_index][1]._value - self.r_center[1]
    //     nz = positions[atom.atom_index][2]._value - self.r_center[2]
    //     norm = sqrt( nx**2 + ny**2 + nz**2)
    //     atom.nx = nx / norm ; atom.ny = ny / norm ; atom.nz = nz / norm
    conductor.normalVectors.resize(3 * Natoms);

    for (size_t i = 0; i < conductor.virtualAtomIndices.size(); i++) {
        int atomIdx = conductor.virtualAtomIndices[i];
        double nx = positions[atomIdx][0] - conductor.r_center[0];
        double ny = positions[atomIdx][1] - conductor.r_center[1];
        double nz = positions[atomIdx][2] - conductor.r_center[2];
        double norm = sqrt(nx*nx + ny*ny + nz*nz);

        conductor.normalVectors[3*i + 0] = nx / norm;
        conductor.normalVectors[3*i + 1] = ny / norm;
        conductor.normalVectors[3*i + 2] = nz / norm;
    }

    std::cout << "[Reference] Buckyball geometry initialized: r_center=("
              << conductor.r_center[0] << "," << conductor.r_center[1] << "," << conductor.r_center[2]
              << "), radius=" << conductor.radius
              << ", area_atom=" << conductor.area_atom << std::endl;
}

// ═══════════════════════════════════════════════════════════
// initializeNanotubeGeometry()
// 翻译自: Nanotube_Virtual.__init__ (Line 517-572)
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::initializeNanotubeGeometry(
    NanotubeConductor& conductor,
    const vector<Vec3>& positions,
    const Vec3 boxVectors[3]
) {
    int Natoms = conductor.virtualAtomIndices.size();

    // Line 521-529: Find center of nanotube (完全照抄)
    // self.r_center = [ 0.0 , 0.0 , 0.0 ] # in nm
    // for atom in self.electrode_atoms:
    //     self.r_center[0] += positions[atom.atom_index][0]._value
    //     self.r_center[1] += positions[atom.atom_index][1]._value
    //     self.r_center[2] += positions[atom.atom_index][2]._value
    // self.r_center[0] = self.r_center[0] / self.Natoms
    // self.r_center[1] = self.r_center[1] / self.Natoms
    // self.r_center[2] = self.r_center[2] / self.Natoms
    conductor.r_center[0] = 0.0;
    conductor.r_center[1] = 0.0;
    conductor.r_center[2] = 0.0;

    for (int atomIdx : conductor.virtualAtomIndices) {
        conductor.r_center[0] += positions[atomIdx][0];
        conductor.r_center[1] += positions[atomIdx][1];
        conductor.r_center[2] += positions[atomIdx][2];
    }

    conductor.r_center[0] /= Natoms;
    conductor.r_center[1] /= Natoms;
    conductor.r_center[2] /= Natoms;

    // Line 532-536: Assume nanotube length = box 'a' vector length (完全照抄)
    // print( 'WARNING:  Assuming Nanotube length is equal to length of "a" box vector.  Need to modify code if this is not the case!')
    // boxVecs = MMsys.simmd.topology.getPeriodicBoxVectors()
    // self.length = boxVecs[0][0] / nanometer
    conductor.length = boxVectors[0][0];
    std::cout << "[Reference] WARNING: Assuming Nanotube length = box 'a' vector length ("
              << conductor.length << " nm)" << std::endl;

    // Line 539-558: Compute radial vector at each atom, get radius (完全照抄)
    // Make sure radius is approximately the same for all atoms
    // radius_threshold=0.001
    // self.radius= -1.0
    // for atom in self.electrode_atoms:
    //     dr = [0] * 3
    //     for i in range(3):
    //         dr[i] = positions[atom.atom_index][i]._value - self.r_center[i]
    //     # project out radial component
    //     radial_vector =  self.project_orthogonal_to_axis( numpy.asarray(dr) )
    //     radius = sqrt( radial_vector[0]**2 + radial_vector[1]**2 + radial_vector[2]**2 )
    //     # check that radius matches stored value for nanotube
    //     if self.radius < 0:
    //         self.radius = radius
    //     else:
    //         if abs( self.radius - radius ) > radius_threshold :
    //             print( atom.atom_index , radius , self.radius )
    //             print( 'different radius for atoms in nanotube, something is wrong!')
    //             sys.exit()
    //     # store radial vector for atom
    //     atom.nx = radial_vector[0] / radius ; atom.ny = radial_vector[1] / radius ; atom.nz = radial_vector[2] / radius ;

    double radius_threshold = 0.001;
    conductor.radius = -1.0;
    conductor.normalVectors.resize(3 * Natoms);

    for (size_t i = 0; i < conductor.virtualAtomIndices.size(); i++) {
        int atomIdx = conductor.virtualAtomIndices[i];

        // dr = position - center
        double dr[3];
        dr[0] = positions[atomIdx][0] - conductor.r_center[0];
        dr[1] = positions[atomIdx][1] - conductor.r_center[1];
        dr[2] = positions[atomIdx][2] - conductor.r_center[2];

        // Project out radial component (perpendicular to axis)
        double radial_vector[3];
        projectOrthogonalToAxis(dr, conductor.axis, radial_vector);

        double radius = sqrt(radial_vector[0]*radial_vector[0] +
                           radial_vector[1]*radial_vector[1] +
                           radial_vector[2]*radial_vector[2]);

        // Check radius consistency
        if (conductor.radius < 0) {
            conductor.radius = radius;
        } else {
            if (fabs(conductor.radius - radius) > radius_threshold) {
                std::cerr << "[Reference] ERROR: Atom " << atomIdx
                          << " has different radius (" << radius
                          << " vs " << conductor.radius << ")" << std::endl;
                throw OpenMMException("Different radius for atoms in nanotube!");
            }
        }

        // Store radial normal vector (normalized)
        conductor.normalVectors[3*i + 0] = radial_vector[0] / radius;
        conductor.normalVectors[3*i + 1] = radial_vector[1] / radius;
        conductor.normalVectors[3*i + 2] = radial_vector[2] / radius;
    }

    // Line 561: Compute area per atom (圆柱侧面积)
    // self.area_atom = 2.0 * numpy.pi * self.radius * self.length / self.Natoms
    conductor.area_atom = 2.0 * M_PI * conductor.radius * conductor.length / Natoms;

    std::cout << "[Reference] Nanotube geometry initialized: r_center=("
              << conductor.r_center[0] << "," << conductor.r_center[1] << "," << conductor.r_center[2]
              << "), radius=" << conductor.radius
              << ", length=" << conductor.length
              << ", area_atom=" << conductor.area_atom << std::endl;
}

// ═══════════════════════════════════════════════════════════
// projectOrthogonalToAxis()
// 翻译自: Nanotube_Virtual.project_orthogonal_to_axis (Line 576-579)
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::projectOrthogonalToAxis(
    const double vec_in[3],
    const double axis[3],
    double vec_out[3]
) {
    // Line 577-578: Project out component parallel to axis (完全照抄)
    // axis_local = numpy.asarray( self.axis )
    // vec_out = vec_in - axis_local * numpy.dot( vec_in , axis_local )
    double dot_product = vec_in[0]*axis[0] + vec_in[1]*axis[1] + vec_in[2]*axis[2];

    vec_out[0] = vec_in[0] - axis[0] * dot_product;
    vec_out[1] = vec_in[1] - axis[1] * dot_product;
    vec_out[2] = vec_in[2] - axis[2] * dot_product;
}

// ═══════════════════════════════════════════════════════════
// findContactNeighborConductor()
// 翻译自: Conductor_Virtual.find_contact_neighbor_conductor (Line 177-227)
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::findContactNeighborConductor(
    BuckyballConductor& conductor,
    const vector<Vec3>& positions
) {
    // Line 181-184: Find Cathode/Anode contact atom (完全照抄)
    // if self.electrode_type == "cathode":
    //     Electrode_contact = MMsys.Cathode
    // else:
    //     Electrode_contact = MMsys.Anode
    const vector<int>* electrodeContact = nullptr;
    if (conductor.electrodeType == "cathode") {
        electrodeContact = &cathodeAtomIndices;
    } else {
        electrodeContact = &anodeAtomIndices;
    }

    // Line 186-193: find contact atom based on closest distance to r_center (完全照抄)
    // min_dist = 10.0 # something large...
    // for atom in Electrode_contact.electrode_atoms:
    //     dr_atom = numpy.sqrt( ( r_center[0] - positions[atom.atom_index][0]._value )**2 +
    //                           ( r_center[1] - positions[atom.atom_index][1]._value )**2 +
    //                           ( r_center[2] - positions[atom.atom_index][2]._value )**2 )
    //     if dr_atom < min_dist:
    //         self.Electrode_contact_atom = atom
    //         min_dist = dr_atom
    double min_dist = 10.0;  // something large
    conductor.contactAtomIndex = -1;

    for (int atomIdx : *electrodeContact) {
        double dx = conductor.r_center[0] - positions[atomIdx][0];
        double dy = conductor.r_center[1] - positions[atomIdx][1];
        double dz = conductor.r_center[2] - positions[atomIdx][2];
        double dr_atom = sqrt(dx*dx + dy*dy + dz*dz);

        if (dr_atom < min_dist) {
            conductor.contactAtomIndex = atomIdx;
            min_dist = dr_atom;
        }
    }

    // Line 195-198: We are likely done here (完全照抄)
    // if  min_dist < self.close_conductor_threshold :
    //     self.dr_center_contact = min_dist
    //     return False  # indicates that dr_vector isn't returned
    if (min_dist < conductor.closeThreshold) {
        conductor.dr_center_contact = min_dist;
        conductor.closeToElectrode = true;
        std::cout << "[Reference] Buckyball contact found: atomIdx=" << conductor.contactAtomIndex
                  << ", distance=" << min_dist << " nm" << std::endl;
        return;
    }

    // Line 200-227: conductor is in contact with another conductor (第一版跳过)
    // TODO: 实现多导体链接支持
    conductor.closeToElectrode = false;
    std::cout << "[Reference] Warning: Buckyball not close to primary electrode (dist=" << min_dist
              << " > threshold=" << conductor.closeThreshold << ")" << std::endl;
}

// ═══════════════════════════════════════════════════════════
// numericalChargeConductor()
// 翻译自: MM.Numerical_charge_Conductor (Line 388-497)
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::numericalChargeConductor(
    BuckyballConductor& conductor,
    const vector<Vec3>& forces,
    ContextImpl& context
) {
    // ═══════════════════════════════════════════════════════════
    // Step 1: Image charges on Conductor (Line 390-422)
    // Project Efield to surface normal vector
    // ═══════════════════════════════════════════════════════════

    // Line 396-420: Images charges are set on 'Virtual' atoms (完全照抄)
    // for atom in Conductor.electrode_atoms:
    //     index = atom.atom_index
    //     (q_i_quantity, sig, eps) = self.nbondedForce.getParticleParameters(index)
    //     q_i = q_i_quantity._value
    //
    //     E_external=[]
    //     if abs(q_i) > (0.9*self.small_threshold):
    //         E_external.append( forces[index][0]._value / q_i ) # Ex
    //         E_external.append( forces[index][1]._value / q_i ) # Ey
    //         E_external.append( forces[index][2]._value / q_i ) # Ez
    //
    //         # project out normal
    //         En_external = numpy.dot( numpy.array( E_external ) , numpy.array( [ atom.nx , atom.ny , atom.nz ] ) )
    //         # now solve for surface charge
    //         q_i = 2.0 / ( 4.0 * numpy.pi ) * Conductor.area_atom * En_external * conversion_KjmolNm_Au
    //     else:
    //         q_i = self.small_threshold  # Cathode, make positive
    //
    //     atom.charge = q_i
    //     self.nbondedForce.setParticleParameters(index, atom.charge, sig , eps)

    for (size_t i = 0; i < conductor.virtualAtomIndices.size(); i++) {
        int atomIdx = conductor.virtualAtomIndices[i];
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(atomIdx, charge, sigma, epsilon);
        double q_i = charge;

        double nx = conductor.normalVectors[3*i + 0];
        double ny = conductor.normalVectors[3*i + 1];
        double nz = conductor.normalVectors[3*i + 2];

        if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
            // E_external = F / q
            double Ex = forces[atomIdx][0] / q_i;
            double Ey = forces[atomIdx][1] / q_i;
            double Ez = forces[atomIdx][2] / q_i;

            // project out normal component
            double En_external = Ex * nx + Ey * ny + Ez * nz;

            // solve for surface charge (完全照抄公式)
            q_i = 2.0 / (4.0 * M_PI) * conductor.area_atom * En_external * CONVERSION_KJMOLNM_AU;
        } else {
            q_i = SMALL_THRESHOLD;  // prevent zero charge
        }

        currentCharges[atomIdx] = q_i;
        nonbondedForce->setParticleParameters(atomIdx, q_i, sigma, epsilon);
    }

    // Line 424-426: Update context and get new forces (完全照抄)
    // self.nbondedForce.updateParametersInContext(self.simmd.context)
    // state = self.simmd.context.getState(getEnergy=True,getForces=True,getVelocities=False,getPositions=True)
    // forces = state.getForces()
    nonbondedForce->updateParametersInContext(context.getOwner());

    // Recompute forces after Step 1 charge update
    context.calcForcesAndEnergy(true, false, -1);
    vector<Vec3>& forcesNew = extractForces(context);

    // ═══════════════════════════════════════════════════════════
    // Step 2: Charge transfer to Conductor (Line 429-495)
    // Distribute uniformly on atoms
    // ═══════════════════════════════════════════════════════════

    // Line 435-439: index of close contact atom (完全照抄)
    // conductor_atom = Conductor.Electrode_contact_atom
    // conductor_atom_index = conductor_atom.atom_index
    // (q_i_quantity, sig, eps) = self.nbondedForce.getParticleParameters(conductor_atom_index)
    // q_i = q_i_quantity._value
    if (conductor.contactAtomIndex < 0) {
        std::cout << "[Reference] Warning: No contact atom found for Buckyball, skipping Step 2" << std::endl;
        return;
    }

    int conductorAtomIndex = conductor.contactAtomIndex;
    double charge, sigma, epsilon;
    nonbondedForce->getParticleParameters(conductorAtomIndex, charge, sigma, epsilon);
    double q_i = charge;

    // Need to find the normal vector for this contact atom
    // The contact atom is on the electrode (cathode/anode), NOT on the conductor
    // So we need to compute its normal vector pointing from electrode to conductor
    // For flat electrode, normal is simply (0, 0, ±1)
    double conductor_atom_nx, conductor_atom_ny, conductor_atom_nz;

    // Determine normal based on electrode type
    if (conductor.electrodeType == "cathode") {
        // Cathode normal points in +z direction (towards anode)
        conductor_atom_nx = 0.0;
        conductor_atom_ny = 0.0;
        conductor_atom_nz = 1.0;
    } else {
        // Anode normal points in -z direction (towards cathode)
        conductor_atom_nx = 0.0;
        conductor_atom_ny = 0.0;
        conductor_atom_nz = -1.0;
    }

    // Line 441-452: get field normal to surface (完全照抄)
    // E_external=[]
    // if abs(q_i) > (0.9*self.small_threshold):
    //     E_external.append( forces[conductor_atom_index][0]._value / q_i ) # Ex
    //     E_external.append( forces[conductor_atom_index][1]._value / q_i ) # Ey
    //     E_external.append( forces[conductor_atom_index][2]._value / q_i ) # Ez
    //     En_external = numpy.dot( numpy.array( E_external ) , numpy.array( [ conductor_atom.nx , conductor_atom.ny , conductor_atom.nz ] ) )
    // else:
    //     En_external = 0.0
    double En_external = 0.0;

    if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
        double Ex = forcesNew[conductorAtomIndex][0] / q_i;
        double Ey = forcesNew[conductorAtomIndex][1] / q_i;
        double Ez = forcesNew[conductorAtomIndex][2] / q_i;

        // project out normal
        En_external = Ex * conductor_atom_nx + Ey * conductor_atom_ny + Ez * conductor_atom_nz;
    }

    // Line 455-466: boundary condition depends on whether contact is with Electrode or another conductor (完全照抄)
    // if Conductor.close_conductor_Electrode :
    //     dE_conductor = - ( En_external + self.Cathode.Voltage / self.Lgap / 2.0 ) * conversion_KjmolNm_Au
    // else :
    //     dE_conductor = - En_external * conversion_KjmolNm_Au
    double dE_conductor;

    if (conductor.closeToElectrode) {
        // Line 462: Electrostatics boundary condition (完全照抄)
        // dE_conductor = -( Eext + dV/2L )
        dE_conductor = -(En_external + voltage / Lgap / 2.0) * CONVERSION_KJMOLNM_AU;
    } else {
        // Line 466: Another conductor contact (完全照抄)
        dE_conductor = -En_external * CONVERSION_KJMOLNM_AU;
    }

    // Line 469-473: Charge depends on geometry of conductor (完全照抄)
    // if type(Conductor).__name__ == "Buckyball_Virtual" :
    //     sign=-1.0
    //     dQ_conductor =  sign * dE_conductor * Conductor.dr_center_contact**2
    double sign = -1.0;
    double dQ_conductor = sign * dE_conductor * conductor.dr_center_contact * conductor.dr_center_contact;

    // Line 486-495: per atom charge and ADD to Conductor (完全照抄)
    // dq_atom = dQ_conductor / Conductor.Natoms
    // for atom in Conductor.electrode_atoms:
    //     index = atom.atom_index
    //     (q_i_quantity, sig, eps) = self.nbondedForce.getParticleParameters(index)
    //     q_i = q_i_quantity._value +  dq_atom
    //     atom.charge = q_i
    //     self.nbondedForce.setParticleParameters(index, q_i, sig , eps)
    int Natoms = conductor.virtualAtomIndices.size();
    double dq_atom = dQ_conductor / Natoms;

    for (int atomIdx : conductor.virtualAtomIndices) {
        double charge_old, sig, eps;
        nonbondedForce->getParticleParameters(atomIdx, charge_old, sig, eps);
        double q_i_new = charge_old + dq_atom;  // ADD dq_atom (完全照抄)

        currentCharges[atomIdx] = q_i_new;
        nonbondedForce->setParticleParameters(atomIdx, q_i_new, sig, eps);
    }

    std::cout << "[Reference] Buckyball charge transfer: dQ=" << dQ_conductor
              << ", dq_atom=" << dq_atom << ", En_external=" << En_external << std::endl;
}

// ═══════════════════════════════════════════════════════════
// numericalChargeNanotube()
// Same as Buckyball but uses cylindrical geometry
// 翻译自: MM.Numerical_charge_Conductor (Line 388-497) with Nanotube geometry
// ═══════════════════════════════════════════════════════════

void ReferenceCalcConstantVKernel::numericalChargeNanotube(
    NanotubeConductor& conductor,
    const vector<Vec3>& forces,
    ContextImpl& context
) {
    // ═══════════════════════════════════════════════════════════
    // Step 1: Image charges on Nanotube (Line 390-422)
    // Uses radial normal vectors (perpendicular to axis)
    // ═══════════════════════════════════════════════════════════

    for (size_t i = 0; i < conductor.virtualAtomIndices.size(); i++) {
        int atomIdx = conductor.virtualAtomIndices[i];
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(atomIdx, charge, sigma, epsilon);
        double q_i = charge;

        // Get radial normal vector (already computed in initializeNanotubeGeometry)
        double nx = conductor.normalVectors[3*i + 0];
        double ny = conductor.normalVectors[3*i + 1];
        double nz = conductor.normalVectors[3*i + 2];

        if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
            // E_external = F / q
            double Ex = forces[atomIdx][0] / q_i;
            double Ey = forces[atomIdx][1] / q_i;
            double Ez = forces[atomIdx][2] / q_i;

            // Project onto radial normal (perpendicular to nanotube axis)
            double En_external = Ex * nx + Ey * ny + Ez * nz;

            // Solve for surface charge using cylindrical area
            // area_atom = 2π × radius × length / Natoms (Line 561 in Original)
            q_i = 2.0 / (4.0 * M_PI) * conductor.area_atom * En_external * CONVERSION_KJMOLNM_AU;
        } else {
            q_i = SMALL_THRESHOLD;  // prevent zero charge
        }

        currentCharges[atomIdx] = q_i;
        nonbondedForce->setParticleParameters(atomIdx, q_i, sigma, epsilon);
    }

    // Update context and get new forces
    nonbondedForce->updateParametersInContext(context.getOwner());
    context.calcForcesAndEnergy(true, false, -1);
    vector<Vec3>& forcesNew = extractForces(context);

    // ═══════════════════════════════════════════════════════════
    // Step 2: Charge transfer to Nanotube (Line 429-495)
    // Same logic as Buckyball but using cylindrical geometry
    // ═══════════════════════════════════════════════════════════

    if (conductor.contactAtomIndex < 0) {
        std::cout << "[Reference] Warning: No contact atom found for Nanotube, skipping Step 2" << std::endl;
        return;
    }

    int conductorAtomIndex = conductor.contactAtomIndex;
    double charge, sigma, epsilon;
    nonbondedForce->getParticleParameters(conductorAtomIndex, charge, sigma, epsilon);
    double q_i = charge;

    // Normal vector for contact atom on electrode
    double conductor_atom_nx, conductor_atom_ny, conductor_atom_nz;

    if (conductor.electrodeType == "cathode") {
        conductor_atom_nx = 0.0;
        conductor_atom_ny = 0.0;
        conductor_atom_nz = 1.0;
    } else {
        conductor_atom_nx = 0.0;
        conductor_atom_ny = 0.0;
        conductor_atom_nz = -1.0;
    }

    // Get field normal to surface
    double En_external = 0.0;

    if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
        double Ex = forcesNew[conductorAtomIndex][0] / q_i;
        double Ey = forcesNew[conductorAtomIndex][1] / q_i;
        double Ez = forcesNew[conductorAtomIndex][2] / q_i;

        En_external = Ex * conductor_atom_nx + Ey * conductor_atom_ny + Ez * conductor_atom_nz;
    }

    // Boundary condition
    double dE_conductor;

    if (conductor.closeToElectrode) {
        dE_conductor = -(En_external + voltage / Lgap / 2.0) * CONVERSION_KJMOLNM_AU;
    } else {
        dE_conductor = -En_external * CONVERSION_KJMOLNM_AU;
    }

    // Charge transfer (cylindrical geometry)
    // For Nanotube: same formula as Buckyball (sign=-1.0)
    double sign = -1.0;
    double dQ_conductor = sign * dE_conductor * conductor.dr_center_contact * conductor.length / 2.0;

    // Distribute charge uniformly on all virtual atoms
    int Natoms = conductor.virtualAtomIndices.size();
    double dq_atom = dQ_conductor / Natoms;

    for (int atomIdx : conductor.virtualAtomIndices) {
        double charge_old, sig, eps;
        nonbondedForce->getParticleParameters(atomIdx, charge_old, sig, eps);
        double q_i_new = charge_old + dq_atom;  // ADD dq_atom

        currentCharges[atomIdx] = q_i_new;
        nonbondedForce->setParticleParameters(atomIdx, q_i_new, sig, eps);
    }

    std::cout << "[Reference] Nanotube charge transfer: dQ=" << dQ_conductor
              << ", dq_atom=" << dq_atom << ", En_external=" << En_external << std::endl;
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

        // ═══════════════════════════════════════════════════════
        // Line 352-361: Conductor处理（完全照抄）
        // ═══════════════════════════════════════════════════════
        // if self.Conductor_list:
        //     for Conductor in self.Conductor_list:
        //         self.Numerical_charge_Conductor( Conductor , forces )
        //     self.nbondedForce.updateParametersInContext(self.simmd.context)
        //     self.Cathode.compute_Electrode_charge_analytic( self , positions , self.Conductor_list, z_opposite = self.Anode.z_pos )
        //     self.Anode.compute_Electrode_charge_analytic( self , positions , self.Conductor_list, z_opposite = self.Cathode.z_pos )

        if (!buckyballConductors.empty()) {
            std::cout << "[Reference Debug] Processing " << buckyballConductors.size() << " Buckyball conductor(s)..." << std::endl;

            // Line 354-355: Process each conductor (完全照抄)
            for (BuckyballConductor& conductor : buckyballConductors) {
                numericalChargeConductor(conductor, forces, context);
            }

            // Line 357: Update context after conductor charge updates (完全照抄)
            nonbondedForce->updateParametersInContext(context.getOwner());

            // Line 358-360: Recompute Q_analytic because conductors are "part of electrolyte" (完全照抄)
            // Get fresh positions after conductor updates
            const vector<Vec3>& positionsUpdated = state.getPositions();

            computeElectrodeChargeAnalytic(
                cathodeAtomIndices, positionsUpdated, "cathode",
                z_anode, Q_analytic_cathode
            );

            computeElectrodeChargeAnalytic(
                anodeAtomIndices, positionsUpdated, "anode",
                z_cathode, Q_analytic_anode
            );

            std::cout << "[Reference Debug] After conductor processing: Q_analytic_cathode="
                      << Q_analytic_cathode << ", Q_analytic_anode=" << Q_analytic_anode << std::endl;
        }

        // ═══════════════════════════════════════════════════════
        // Process Nanotube conductors (same pattern as Buckyball)
        // ═══════════════════════════════════════════════════════
        if (!nanotubeConductors.empty()) {
            std::cout << "[Reference Debug] Processing " << nanotubeConductors.size() << " Nanotube conductor(s)..." << std::endl;

            // Process each Nanotube conductor
            for (NanotubeConductor& conductor : nanotubeConductors) {
                numericalChargeNanotube(conductor, forces, context);
            }

            // Update context after conductor charge updates
            nonbondedForce->updateParametersInContext(context.getOwner());

            // Recompute Q_analytic because conductors are "part of electrolyte"
            const vector<Vec3>& positionsUpdated = state.getPositions();

            computeElectrodeChargeAnalytic(
                cathodeAtomIndices, positionsUpdated, "cathode",
                z_anode, Q_analytic_cathode
            );

            computeElectrodeChargeAnalytic(
                anodeAtomIndices, positionsUpdated, "anode",
                z_cathode, Q_analytic_anode
            );

            std::cout << "[Reference Debug] After Nanotube processing: Q_analytic_cathode="
                      << Q_analytic_cathode << ", Q_analytic_anode=" << Q_analytic_anode << std::endl;
        }

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

    // ───────────────────────────────────────────────────────
    // 找到ConstantVForce（用于加载Buckyball数据）
    // ───────────────────────────────────────────────────────
    const ConstantVForce* constantVForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        const Force& force = system.getForce(i);
        const ConstantVForce* cvForce = dynamic_cast<const ConstantVForce*>(&force);
        if (cvForce != nullptr) {
            constantVForce = cvForce;
            break;
        }
    }

    // ───────────────────────────────────────────────────────
    // 从ConstantVForce加载Buckyball导体（如果存在）
    // ───────────────────────────────────────────────────────
    if (constantVForce != nullptr) {
        int numBuckyballs = constantVForce->getNumBuckyballConductors();
        buckyballConductors.resize(numBuckyballs);

        for (int i = 0; i < numBuckyballs; i++) {
            std::vector<int> virtualAtoms, realAtoms;
            std::string electrodeType;
            double voltageV;
            constantVForce->getBuckyballConductorParameters(i, virtualAtoms, realAtoms, electrodeType, voltageV);

            BuckyballConductor& conductor = buckyballConductors[i];
            conductor.virtualAtomIndices = virtualAtoms;
            conductor.realAtomIndices = realAtoms;
            conductor.electrodeType = electrodeType;
            conductor.voltageKjMol = voltageV * CONVERSION_EV_KJMOL;
            conductor.closeThreshold = 1.5;  // nm
            conductor.closeToElectrode = true;
            conductor.contactAtomIndex = -1;
            conductor.dr_center_contact = 0.0;
            // 几何参数将在第一次execute时计算
        }

        if (numBuckyballs > 0) {
            std::cout << "[Reference Integrator] Loaded " << numBuckyballs << " Buckyball conductor(s) from Force" << std::endl;
        }

        // Nanotube conductors
        int numNanotubes = constantVForce->getNumNanotubeConductors();
        nanotubeConductors.resize(numNanotubes);

        for (int i = 0; i < numNanotubes; i++) {
            std::vector<int> virtualAtoms, realAtoms;
            std::string electrodeType;
            double voltageV;
            std::vector<double> axis;
            constantVForce->getNanotubeConductorParameters(i, virtualAtoms, realAtoms, electrodeType, voltageV, axis);

            NanotubeConductor& conductor = nanotubeConductors[i];
            conductor.virtualAtomIndices = virtualAtoms;
            conductor.realAtomIndices = realAtoms;
            conductor.electrodeType = electrodeType;
            conductor.voltageKjMol = voltageV * CONVERSION_EV_KJMOL;
            conductor.axis[0] = axis[0];
            conductor.axis[1] = axis[1];
            conductor.axis[2] = axis[2];
            conductor.closeThreshold = 1.5;  // nm
            conductor.closeToElectrode = true;
            conductor.contactAtomIndex = -1;
            conductor.dr_center_contact = 0.0;
            // 几何参数将在第一次execute时计算
        }

        if (numNanotubes > 0) {
            std::cout << "[Reference Integrator] Loaded " << numNanotubes << " Nanotube conductor(s) from Force" << std::endl;
        }
    }

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

    // ───────────────────────────────────────────────────────
    // Initialize Buckyball conductors geometry
    // (Note: This happens in initialize(), not in first execute())
    // ───────────────────────────────────────────────────────
    // We need a ContextImpl to get positions, but we don't have it in initialize()
    // So Buckyball geometry will be initialized in first execute() instead
    // (Flag will be added to track this)
}

// ═══════════════════════════════════════════════════════════
// Integrator Buckyball helper methods (identical to CalcKernel)
// ═══════════════════════════════════════════════════════════

void ReferenceIntegrateConstantVStepKernel::initializeBuckyballGeometry(
    BuckyballConductor& conductor,
    const vector<Vec3>& positions
) {
    // Identical implementation to CalcKernel - Line 327-400
    int Natoms = conductor.virtualAtomIndices.size();

    conductor.r_center[0] = 0.0;
    conductor.r_center[1] = 0.0;
    conductor.r_center[2] = 0.0;

    for (int atomIdx : conductor.virtualAtomIndices) {
        conductor.r_center[0] += positions[atomIdx][0];
        conductor.r_center[1] += positions[atomIdx][1];
        conductor.r_center[2] += positions[atomIdx][2];
    }

    conductor.r_center[0] /= Natoms;
    conductor.r_center[1] /= Natoms;
    conductor.r_center[2] /= Natoms;

    if (Natoms > 0) {
        int firstAtom = conductor.virtualAtomIndices[0];
        double rx = positions[firstAtom][0] - conductor.r_center[0];
        double ry = positions[firstAtom][1] - conductor.r_center[1];
        double rz = positions[firstAtom][2] - conductor.r_center[2];
        conductor.radius = sqrt(rx*rx + ry*ry + rz*rz);
    }

    conductor.area_atom = 4.0 * M_PI * conductor.radius * conductor.radius / Natoms;

    conductor.normalVectors.resize(3 * Natoms);

    for (size_t i = 0; i < conductor.virtualAtomIndices.size(); i++) {
        int atomIdx = conductor.virtualAtomIndices[i];
        double nx = positions[atomIdx][0] - conductor.r_center[0];
        double ny = positions[atomIdx][1] - conductor.r_center[1];
        double nz = positions[atomIdx][2] - conductor.r_center[2];
        double norm = sqrt(nx*nx + ny*ny + nz*nz);

        conductor.normalVectors[3*i + 0] = nx / norm;
        conductor.normalVectors[3*i + 1] = ny / norm;
        conductor.normalVectors[3*i + 2] = nz / norm;
    }
}

void ReferenceIntegrateConstantVStepKernel::findContactNeighborConductor(
    BuckyballConductor& conductor,
    const vector<Vec3>& positions
) {
    // Identical implementation to CalcKernel - Line 407-464
    const vector<int>* electrodeContact = nullptr;
    if (conductor.electrodeType == "cathode") {
        electrodeContact = &cathodeAtomIndices;
    } else {
        electrodeContact = &anodeAtomIndices;
    }

    double min_dist = 10.0;
    conductor.contactAtomIndex = -1;

    for (int atomIdx : *electrodeContact) {
        double dx = conductor.r_center[0] - positions[atomIdx][0];
        double dy = conductor.r_center[1] - positions[atomIdx][1];
        double dz = conductor.r_center[2] - positions[atomIdx][2];
        double dr_atom = sqrt(dx*dx + dy*dy + dz*dz);

        if (dr_atom < min_dist) {
            conductor.contactAtomIndex = atomIdx;
            min_dist = dr_atom;
        }
    }

    if (min_dist < conductor.closeThreshold) {
        conductor.dr_center_contact = min_dist;
        conductor.closeToElectrode = true;
        return;
    }

    conductor.closeToElectrode = false;
}

void ReferenceIntegrateConstantVStepKernel::numericalChargeConductor(
    BuckyballConductor& conductor,
    const vector<Vec3>& forces,
    ContextImpl& context
) {
    // Identical implementation to CalcKernel - Line 471-690
    // Step 1: Image charges
    for (size_t i = 0; i < conductor.virtualAtomIndices.size(); i++) {
        int atomIdx = conductor.virtualAtomIndices[i];
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(atomIdx, charge, sigma, epsilon);
        double q_i = charge;

        double nx = conductor.normalVectors[3*i + 0];
        double ny = conductor.normalVectors[3*i + 1];
        double nz = conductor.normalVectors[3*i + 2];

        if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
            double Ex = forces[atomIdx][0] / q_i;
            double Ey = forces[atomIdx][1] / q_i;
            double Ez = forces[atomIdx][2] / q_i;

            double En_external = Ex * nx + Ey * ny + Ez * nz;

            q_i = 2.0 / (4.0 * M_PI) * conductor.area_atom * En_external * CONVERSION_KJMOLNM_AU;
        } else {
            q_i = SMALL_THRESHOLD;
        }

        currentCharges[atomIdx] = q_i;
        nonbondedForce->setParticleParameters(atomIdx, q_i, sigma, epsilon);
    }

    nonbondedForce->updateParametersInContext(context.getOwner());

    // Recompute forces
    context.calcForcesAndEnergy(true, false, -1);
    vector<Vec3>& forcesNew = extractForces(context);

    // Step 2: Charge transfer
    if (conductor.contactAtomIndex < 0) {
        return;
    }

    int conductorAtomIndex = conductor.contactAtomIndex;
    double charge, sigma, epsilon;
    nonbondedForce->getParticleParameters(conductorAtomIndex, charge, sigma, epsilon);
    double q_i = charge;

    double conductor_atom_nx, conductor_atom_ny, conductor_atom_nz;

    if (conductor.electrodeType == "cathode") {
        conductor_atom_nx = 0.0;
        conductor_atom_ny = 0.0;
        conductor_atom_nz = 1.0;
    } else {
        conductor_atom_nx = 0.0;
        conductor_atom_ny = 0.0;
        conductor_atom_nz = -1.0;
    }

    double En_external = 0.0;

    if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
        double Ex = forcesNew[conductorAtomIndex][0] / q_i;
        double Ey = forcesNew[conductorAtomIndex][1] / q_i;
        double Ez = forcesNew[conductorAtomIndex][2] / q_i;

        En_external = Ex * conductor_atom_nx + Ey * conductor_atom_ny + Ez * conductor_atom_nz;
    }

    double dE_conductor;

    if (conductor.closeToElectrode) {
        dE_conductor = -(En_external + voltage / Lgap / 2.0) * CONVERSION_KJMOLNM_AU;
    } else {
        dE_conductor = -En_external * CONVERSION_KJMOLNM_AU;
    }

    double sign = -1.0;
    double dQ_conductor = sign * dE_conductor * conductor.dr_center_contact * conductor.length / 2.0;

    int Natoms = conductor.virtualAtomIndices.size();
    double dq_atom = dQ_conductor / Natoms;

    for (int atomIdx : conductor.virtualAtomIndices) {
        double charge_old, sig, eps;
        nonbondedForce->getParticleParameters(atomIdx, charge_old, sig, eps);
        double q_i_new = charge_old + dq_atom;

        currentCharges[atomIdx] = q_i_new;
        nonbondedForce->setParticleParameters(atomIdx, q_i_new, sig, eps);
    }
}

// ═══════════════════════════════════════════════════════════
// initializeNanotubeGeometry() - Initialize Nanotube geometry
// (Integrator kernel version - same as CalcKernel)
// ═══════════════════════════════════════════════════════════

void ReferenceIntegrateConstantVStepKernel::initializeNanotubeGeometry(
    NanotubeConductor& conductor,
    const vector<Vec3>& positions,
    const Vec3 boxVectors[3]
) {
    // Compute center
    conductor.r_center[0] = 0.0;
    conductor.r_center[1] = 0.0;
    conductor.r_center[2] = 0.0;

    for (int idx : conductor.virtualAtomIndices) {
        conductor.r_center[0] += positions[idx][0];
        conductor.r_center[1] += positions[idx][1];
        conductor.r_center[2] += positions[idx][2];
    }

    int N = conductor.virtualAtomIndices.size();
    conductor.r_center[0] /= N;
    conductor.r_center[1] /= N;
    conductor.r_center[2] /= N;

    // Compute radius (radial distance from axis)
    conductor.radius = 0.0;
    for (int idx : conductor.virtualAtomIndices) {
        double dx = positions[idx][0] - conductor.r_center[0];
        double dy = positions[idx][1] - conductor.r_center[1];
        double dz = positions[idx][2] - conductor.r_center[2];

        double proj_x = dx - conductor.axis[0] * (conductor.axis[0]*dx + conductor.axis[1]*dy + conductor.axis[2]*dz);
        double proj_y = dy - conductor.axis[1] * (conductor.axis[0]*dx + conductor.axis[1]*dy + conductor.axis[2]*dz);
        double proj_z = dz - conductor.axis[2] * (conductor.axis[0]*dx + conductor.axis[1]*dy + conductor.axis[2]*dz);

        double r = sqrt(proj_x*proj_x + proj_y*proj_y + proj_z*proj_z);
        conductor.radius += r;
    }
    conductor.radius /= N;

    // Length from box 'a' vector
    conductor.length = sqrt(boxVectors[0][0]*boxVectors[0][0] +
                           boxVectors[0][1]*boxVectors[0][1] +
                           boxVectors[0][2]*boxVectors[0][2]);

    // Area per atom: 2π × r × L / N
    conductor.area_atom = 2.0 * M_PI * conductor.radius * conductor.length / N;

    // Radial normal vectors
    conductor.normalVectors.resize(3 * N);
    for (size_t i = 0; i < conductor.virtualAtomIndices.size(); i++) {
        int idx = conductor.virtualAtomIndices[i];

        double dx = positions[idx][0] - conductor.r_center[0];
        double dy = positions[idx][1] - conductor.r_center[1];
        double dz = positions[idx][2] - conductor.r_center[2];

        // Project orthogonal to axis (radial vector)
        double dot_product = dx*conductor.axis[0] + dy*conductor.axis[1] + dz*conductor.axis[2];
        double radial_x = dx - conductor.axis[0] * dot_product;
        double radial_y = dy - conductor.axis[1] * dot_product;
        double radial_z = dz - conductor.axis[2] * dot_product;

        // Normalize
        double norm = sqrt(radial_x*radial_x + radial_y*radial_y + radial_z*radial_z);
        if (norm > 1e-10) {
            conductor.normalVectors[3*i + 0] = radial_x / norm;
            conductor.normalVectors[3*i + 1] = radial_y / norm;
            conductor.normalVectors[3*i + 2] = radial_z / norm;
        } else {
            conductor.normalVectors[3*i + 0] = 1.0;
            conductor.normalVectors[3*i + 1] = 0.0;
            conductor.normalVectors[3*i + 2] = 0.0;
        }
    }
}

// ═══════════════════════════════════════════════════════════
// numericalChargeNanotube() - Numerical charging for Nanotube
// (Integrator kernel version - same as CalcKernel)
// ═══════════════════════════════════════════════════════════

void ReferenceIntegrateConstantVStepKernel::numericalChargeNanotube(
    NanotubeConductor& conductor,
    const vector<Vec3>& forces,
    ContextImpl& context
) {
    // Step 1: Image charges on Nanotube using radial normals
    for (size_t i = 0; i < conductor.virtualAtomIndices.size(); i++) {
        int atomIdx = conductor.virtualAtomIndices[i];
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(atomIdx, charge, sigma, epsilon);
        double q_i = charge;

        // Get radial normal vector (perpendicular to axis)
        double nx = conductor.normalVectors[3*i + 0];
        double ny = conductor.normalVectors[3*i + 1];
        double nz = conductor.normalVectors[3*i + 2];

        if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
            double Ex = forces[atomIdx][0] / q_i;
            double Ey = forces[atomIdx][1] / q_i;
            double Ez = forces[atomIdx][2] / q_i;

            // Project onto radial normal (not axial!)
            double En_external = Ex * nx + Ey * ny + Ez * nz;

            // Solve for surface charge using cylindrical area
            q_i = 2.0 / (4.0 * M_PI) * conductor.area_atom * En_external * CONVERSION_KJMOLNM_AU;
        } else {
            q_i = SMALL_THRESHOLD;
        }

        currentCharges[atomIdx] = q_i;
        nonbondedForce->setParticleParameters(atomIdx, q_i, sigma, epsilon);
    }

    // Update and recompute forces
    nonbondedForce->updateParametersInContext(context.getOwner());
    context.calcForcesAndEnergy(true, false, -1);
    vector<Vec3>& forcesNew = extractForces(context);

    // Step 2: Charge transfer to Nanotube (same logic as Buckyball)
    if (conductor.contactAtomIndex < 0) {
        return;  // No contact atom
    }

    int conductorAtomIndex = conductor.contactAtomIndex;
    double charge, sigma, epsilon;
    nonbondedForce->getParticleParameters(conductorAtomIndex, charge, sigma, epsilon);
    double q_i = charge;

    // Flat electrode normal
    double conductor_atom_nx, conductor_atom_ny, conductor_atom_nz;
    if (conductor.electrodeType == "cathode") {
        conductor_atom_nx = 0.0;
        conductor_atom_ny = 0.0;
        conductor_atom_nz = 1.0;
    } else {
        conductor_atom_nx = 0.0;
        conductor_atom_ny = 0.0;
        conductor_atom_nz = -1.0;
    }

    double En_external = 0.0;
    if (fabs(q_i) > (0.9 * SMALL_THRESHOLD)) {
        double Ex = forcesNew[conductorAtomIndex][0] / q_i;
        double Ey = forcesNew[conductorAtomIndex][1] / q_i;
        double Ez = forcesNew[conductorAtomIndex][2] / q_i;

        En_external = Ex * conductor_atom_nx + Ey * conductor_atom_ny + Ez * conductor_atom_nz;
    }

    double dE_conductor;
    if (conductor.closeToElectrode) {
        dE_conductor = -(En_external + voltage / Lgap / 2.0) * CONVERSION_KJMOLNM_AU;
    } else {
        dE_conductor = -En_external * CONVERSION_KJMOLNM_AU;
    }

    double sign = -1.0;
    double dQ_conductor = sign * dE_conductor * conductor.dr_center_contact * conductor.dr_center_contact;

    int Natoms = conductor.virtualAtomIndices.size();
    double dq_atom = dQ_conductor / Natoms;

    // Distribute charge uniformly
    for (int atomIdx : conductor.virtualAtomIndices) {
        double charge_old, sig, eps;
        nonbondedForce->getParticleParameters(atomIdx, charge_old, sig, eps);
        double q_i_new = charge_old + dq_atom;

        currentCharges[atomIdx] = q_i_new;
        nonbondedForce->setParticleParameters(atomIdx, q_i_new, sig, eps);
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

    // ───────────────────────────────────────────────────────
    // Initialize Buckyball geometry on first call
    // (Need positions from context, so must be done here)
    // ───────────────────────────────────────────────────────
    static bool buckyballInitialized = false;
    if (!buckyballInitialized && !buckyballConductors.empty()) {
        for (BuckyballConductor& conductor : buckyballConductors) {
            initializeBuckyballGeometry(conductor, positions);
            findContactNeighborConductor(conductor, positions);
        }
        buckyballInitialized = true;
        std::cout << "[Reference Integrator] Buckyball conductors initialized" << std::endl;
    }

    static bool nanotubeInitialized = false;
    if (!nanotubeInitialized && !nanotubeConductors.empty()) {
        Vec3 boxVectors[3];
        context.getPeriodicBoxVectors(boxVectors[0], boxVectors[1], boxVectors[2]);

        for (NanotubeConductor& conductor : nanotubeConductors) {
            initializeNanotubeGeometry(conductor, positions, boxVectors);
            // Skip findContactNeighbor for now
        }
        nanotubeInitialized = true;
        std::cout << "[Reference Integrator] Nanotube conductors initialized" << std::endl;
    }

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

        // ═══════════════════════════════════════════════════════
        // Line 352-361: Conductor处理（完全照抄）
        // ═══════════════════════════════════════════════════════
        // if self.Conductor_list:
        //     for Conductor in self.Conductor_list:
        //         self.Numerical_charge_Conductor( Conductor , forces )
        //     self.nbondedForce.updateParametersInContext(self.simmd.context)
        //     self.Cathode.compute_Electrode_charge_analytic(...)
        //     self.Anode.compute_Electrode_charge_analytic(...)

        if (!buckyballConductors.empty()) {
            // Line 354-355: Process each conductor
            for (BuckyballConductor& conductor : buckyballConductors) {
                numericalChargeConductor(conductor, forces, context);
            }

            // Line 357: Update context after conductor charge updates
            nonbondedForce->updateParametersInContext(context.getOwner());

            // Line 358-360: Recompute Q_analytic (conductors are "part of electrolyte")
            computeElectrodeChargeAnalytic(
                cathodeAtomIndices, positions, "cathode",
                z_anode, Q_analytic_cathode
            );

            computeElectrodeChargeAnalytic(
                anodeAtomIndices, positions, "anode",
                z_cathode, Q_analytic_anode
            );
        }

        // ═══════════════════════════════════════════════════════
        // Process Nanotube conductors (same pattern as Buckyball)
        // ═══════════════════════════════════════════════════════
        if (!nanotubeConductors.empty()) {
            // Process each Nanotube conductor
            for (NanotubeConductor& conductor : nanotubeConductors) {
                numericalChargeNanotube(conductor, forces, context);
            }

            // Update context after conductor charge updates
            nonbondedForce->updateParametersInContext(context.getOwner());

            // Recompute Q_analytic (conductors are "part of electrolyte")
            computeElectrodeChargeAnalytic(
                cathodeAtomIndices, positions, "cathode",
                z_anode, Q_analytic_cathode
            );

            computeElectrodeChargeAnalytic(
                anodeAtomIndices, positions, "anode",
                z_cathode, Q_analytic_anode
            );
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
