/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVIntegrator implementation                                         *
 * Copyright (c) 2025 - Present                                               *
 * -------------------------------------------------------------------------- */

#include "openmm/ConstantVIntegrator.h"
#include "openmm/ConstantVKernels.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/KernelFactory.h"
#include <string>

using namespace OpenMM;
using std::vector;
using std::string;

ConstantVIntegrator::ConstantVIntegrator(double stepSize) :
    voltageVolts(0.0),
    voltageKjMol(0.0),
    Lgap(0.0),
    Lcell(0.0),
    totalArea(0.0),
    z_cathode(0.0),
    z_anode(0.0),
    nIterations(4),      // Default: 4 SCF iterations
    scfFrequency(1),     // Default: update every step
    stepCount(0)
{
    setStepSize(stepSize);
}

ConstantVIntegrator::~ConstantVIntegrator() {
}

// ═══════════════════════════════════════════════════════════════════════════
// Physical Parameters
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVIntegrator::setVoltage(double voltage) {
    voltageVolts = voltage;
    voltageKjMol = voltage * 96.487;  // conversion_eV_to_kJmol
}

void ConstantVIntegrator::setLgap(double gap) {
    if (gap <= 0)
        throw OpenMMException("ConstantVIntegrator: Lgap must be positive");
    Lgap = gap;
}

void ConstantVIntegrator::setLcell(double cell) {
    if (cell <= 0)
        throw OpenMMException("ConstantVIntegrator: Lcell must be positive");
    Lcell = cell;
}

void ConstantVIntegrator::setTotalArea(double area) {
    if (area <= 0)
        throw OpenMMException("ConstantVIntegrator: totalArea must be positive");
    totalArea = area;
}

// ═══════════════════════════════════════════════════════════════════════════
// SCF Parameters
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVIntegrator::setNumSCFIterations(int n) {
    if (n < 1)
        throw OpenMMException("ConstantVIntegrator: number of iterations must be at least 1");
    nIterations = n;
}

void ConstantVIntegrator::setSCFFrequency(int freq) {
    if (freq < 1)
        throw OpenMMException("ConstantVIntegrator: SCF frequency must be at least 1");
    scfFrequency = freq;
}

// ═══════════════════════════════════════════════════════════════════════════
// Cathode Atom Management
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVIntegrator::addCathodeAtom(int particle, double area) {
    cathodeAtoms.push_back(CathodeAtomInfo(particle, area));
    return cathodeAtoms.size() - 1;
}

void ConstantVIntegrator::getCathodeAtomParameters(int index, int& particle, double& area) const {
    if (index < 0 || index >= (int)cathodeAtoms.size())
        throw OpenMMException("ConstantVIntegrator: cathode atom index out of range");
    particle = cathodeAtoms[index].particle;
    area = cathodeAtoms[index].area;
}

void ConstantVIntegrator::setCathodeAtomParameters(int index, int particle, double area) {
    if (index < 0 || index >= (int)cathodeAtoms.size())
        throw OpenMMException("ConstantVIntegrator: cathode atom index out of range");
    cathodeAtoms[index].particle = particle;
    cathodeAtoms[index].area = area;
}

// ═══════════════════════════════════════════════════════════════════════════
// Anode Atom Management
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVIntegrator::addAnodeAtom(int particle, double area) {
    anodeAtoms.push_back(AnodeAtomInfo(particle, area));
    return anodeAtoms.size() - 1;
}

void ConstantVIntegrator::getAnodeAtomParameters(int index, int& particle, double& area) const {
    if (index < 0 || index >= (int)anodeAtoms.size())
        throw OpenMMException("ConstantVIntegrator: anode atom index out of range");
    particle = anodeAtoms[index].particle;
    area = anodeAtoms[index].area;
}

void ConstantVIntegrator::setAnodeAtomParameters(int index, int particle, double area) {
    if (index < 0 || index >= (int)anodeAtoms.size())
        throw OpenMMException("ConstantVIntegrator: anode atom index out of range");
    anodeAtoms[index].particle = particle;
    anodeAtoms[index].area = area;
}

// ═══════════════════════════════════════════════════════════════════════════
// Electrolyte Atom Management
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVIntegrator::addElectrolyteAtom(int particle, double charge) {
    electrolyteAtoms.push_back(ElectrolyteAtomInfo(particle, charge));
    return electrolyteAtoms.size() - 1;
}

void ConstantVIntegrator::getElectrolyteAtomParameters(int index, int& particle, double& charge) const {
    if (index < 0 || index >= (int)electrolyteAtoms.size())
        throw OpenMMException("ConstantVIntegrator: electrolyte atom index out of range");
    particle = electrolyteAtoms[index].particle;
    charge = electrolyteAtoms[index].charge;
}

void ConstantVIntegrator::setElectrolyteAtomParameters(int index, int particle, double charge) {
    if (index < 0 || index >= (int)electrolyteAtoms.size())
        throw OpenMMException("ConstantVIntegrator: electrolyte atom index out of range");
    electrolyteAtoms[index].particle = particle;
    electrolyteAtoms[index].charge = charge;
}

// ═══════════════════════════════════════════════════════════════════════════
// Integrator Interface Implementation
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVIntegrator::initialize(ContextImpl& context) {
    // Create Verlet integration kernel
    verletKernel = context.getPlatform().createKernel("IntegrateVerletStep", context);

    // Create ConstantV SCF kernel
    calcConstantVKernel = context.getPlatform().createKernel(CalcConstantVKernel::Name(), context);

    // Extract electrode data into arrays
    vector<int> cathodeIndices, anodeIndices, electrolyteIndices;
    vector<double> cathodeAreas, anodeAreas, electrolyteCharges;

    cathodeIndices.reserve(cathodeAtoms.size());
    cathodeAreas.reserve(cathodeAtoms.size());
    for (const auto& atom : cathodeAtoms) {
        cathodeIndices.push_back(atom.particle);
        cathodeAreas.push_back(atom.area);
    }

    anodeIndices.reserve(anodeAtoms.size());
    anodeAreas.reserve(anodeAtoms.size());
    for (const auto& atom : anodeAtoms) {
        anodeIndices.push_back(atom.particle);
        anodeAreas.push_back(atom.area);
    }

    electrolyteIndices.reserve(electrolyteAtoms.size());
    electrolyteCharges.reserve(electrolyteAtoms.size());
    for (const auto& atom : electrolyteAtoms) {
        electrolyteIndices.push_back(atom.particle);
        electrolyteCharges.push_back(atom.charge);
    }

    // Initialize CalcConstantVKernel
    CalcConstantVKernel& calcKernel = dynamic_cast<CalcConstantVKernel&>(calcConstantVKernel.getImpl());
    calcKernel.initialize(
        context.getSystem(),
        cathodeIndices,
        cathodeAreas,
        anodeIndices,
        anodeAreas,
        electrolyteIndices,
        electrolyteCharges,
        voltageVolts,
        Lgap,
        Lcell,
        totalArea,
        z_cathode,
        z_anode,
        nIterations
    );

    // Reset step counter
    stepCount = 0;
}

void ConstantVIntegrator::cleanup() {
    verletKernel = Kernel();
    calcConstantVKernel = Kernel();
}

vector<string> ConstantVIntegrator::getKernelNames() {
    vector<string> names;
    names.push_back("IntegrateVerletStep");
    names.push_back(CalcConstantVKernel::Name());
    return names;
}

void ConstantVIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMMException("ConstantVIntegrator::step() called without a Context");

    for (int i = 0; i < steps; ++i) {
        // Step 1: Velocity half-step + position update (handled by Verlet kernel)
        // This also computes forces at the new position
        dynamic_cast<KernelImpl&>(verletKernel.getImpl()).execute(*context, *this);

        // Step 2: Check if we need to update electrode charges
        if ((stepCount % scfFrequency) == 0) {
            CalcConstantVKernel& calcKernel = dynamic_cast<CalcConstantVKernel&>(calcConstantVKernel.getImpl());
            calcKernel.execute(*context, true, false, -1);
        }

        // Step 3: Velocity second half-step (handled by Verlet kernel)
        // (This is already included in the Verlet kernel execution above)

        stepCount++;
    }
}

double ConstantVIntegrator::computeKineticEnergy() {
    return dynamic_cast<ContextImpl&>(*context).calcKineticEnergy();
}
