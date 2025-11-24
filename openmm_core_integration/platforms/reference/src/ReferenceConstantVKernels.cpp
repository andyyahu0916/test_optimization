/* -------------------------------------------------------------------------- *
 *                Reference ConstantV Kernel Implementation                   *
 * -------------------------------------------------------------------------- */

#include "ReferenceConstantVKernels.h"
#include "ReferenceConstantVDrudeLangevinDynamics.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Context.h"
#include <cmath>

using namespace OpenMM;
using namespace std;

// Physical constants
static const double EPSILON_0 = 8.854187817e-12;  // F/m
static const double E_CHARGE = 1.602176634e-19;   // C
static const double conversion_eV_to_kJmol = 96.487;

// ═══════════════════════════════════════════════════════════════════════════
// ReferenceCalcConstantVKernel
// ═══════════════════════════════════════════════════════════════════════════

ReferenceCalcConstantVKernel::ReferenceCalcConstantVKernel(string name, const Platform& platform) :
    CalcConstantVKernel(name, platform),
    voltage(0.0),
    Lgap(0.0),
    Lcell(0.0),
    totalArea(0.0),
    z_cathode(0.0),
    z_anode(0.0),
    nIterations(4)
{
}

ReferenceCalcConstantVKernel::~ReferenceCalcConstantVKernel() {
}

void ReferenceCalcConstantVKernel::initialize(
    const System& system,
    const vector<int>& cathodeAtomIndices,
    const vector<double>& cathodeAreas,
    const vector<int>& anodeAtomIndices,
    const vector<double>& anodeAreas,
    const vector<int>& electrolyteAtomIndices,
    const vector<double>& electrolyteCharges,
    double voltage,
    double Lgap,
    double Lcell,
    double totalArea,
    double z_cathode,
    double z_anode,
    int nIterations)
{
    this->cathodeIndices = cathodeAtomIndices;
    this->cathodeAreas = cathodeAreas;
    this->anodeIndices = anodeAtomIndices;
    this->anodeAreas = anodeAreas;
    this->electrolyteIndices = electrolyteAtomIndices;
    this->electrolyteCharges = electrolyteCharges;
    this->voltage = voltage * conversion_eV_to_kJmol;  // V to kJ/mol
    this->Lgap = Lgap;
    this->Lcell = Lcell;
    this->totalArea = totalArea;
    this->z_cathode = z_cathode;
    this->z_anode = z_anode;
    this->nIterations = nIterations;

    // Initialize charges
    cathodeCharges.resize(cathodeIndices.size(), 0.0);
    anodeCharges.resize(anodeIndices.size(), 0.0);
}

void ReferenceCalcConstantVKernel::addBuckyballConductor(
    const vector<int>& virtualAtomIndices,
    const vector<int>& realAtomIndices,
    const string& electrodeType,
    double voltage,
    const Vec3& center,
    double radius,
    const vector<Vec3>& normalVectors,
    double areaPerAtom,
    int contactAtomIndex,
    double contactDistance)
{
    // TODO: Implement Buckyball conductor support
    throw OpenMMException("Buckyball conductors not yet implemented in Reference platform");
}

void ReferenceCalcConstantVKernel::addNanotubeConductor(
    const vector<int>& virtualAtomIndices,
    const vector<int>& realAtomIndices,
    const string& electrodeType,
    double voltage,
    const Vec3& center,
    const Vec3& axis,
    double radius,
    double length,
    const vector<Vec3>& normalVectors,
    double areaPerAtom,
    int contactAtomIndex,
    double contactDistance)
{
    // TODO: Implement Nanotube conductor support
    throw OpenMMException("Nanotube conductors not yet implemented in Reference platform");
}

void ReferenceCalcConstantVKernel::runSCF(const vector<Vec3>& positions) {
    const int numCathodes = cathodeIndices.size();
    const int numAnodes = anodeIndices.size();
    const int numElectrolytes = electrolyteIndices.size();

    if (numCathodes == 0 && numAnodes == 0)
        return;

    // SCF iteration loop
    for (int iter = 0; iter < nIterations; iter++) {
        // Compute electrode potentials
        double phi_cathode_sum = 0.0;
        double phi_anode_sum = 0.0;

        // Cathode potentials
        for (int i = 0; i < numCathodes; i++) {
            Vec3 pos_i = positions[cathodeIndices[i]];
            double phi_i = 0.0;

            // Contribution from other cathode atoms
            for (int j = 0; j < numCathodes; j++) {
                if (i != j) {
                    Vec3 pos_j = positions[cathodeIndices[j]];
                    Vec3 rij = pos_i - pos_j;
                    double r = sqrt(rij.dot(rij));
                    if (r > 1e-6)
                        phi_i += cathodeCharges[j] / r;
                }
            }

            // Contribution from anode atoms
            for (int j = 0; j < numAnodes; j++) {
                Vec3 pos_j = positions[anodeIndices[j]];
                Vec3 rij = pos_i - pos_j;
                double r = sqrt(rij.dot(rij));
                if (r > 1e-6)
                    phi_i += anodeCharges[j] / r;
            }

            // Contribution from electrolyte atoms
            for (int j = 0; j < numElectrolytes; j++) {
                Vec3 pos_j = positions[electrolyteIndices[j]];
                Vec3 rij = pos_i - pos_j;
                double r = sqrt(rij.dot(rij));
                if (r > 1e-6)
                    phi_i += electrolyteCharges[j] / r;
            }

            phi_cathode_sum += phi_i;
        }

        // Anode potentials
        for (int i = 0; i < numAnodes; i++) {
            Vec3 pos_i = positions[anodeIndices[i]];
            double phi_i = 0.0;

            // Contribution from cathode atoms
            for (int j = 0; j < numCathodes; j++) {
                Vec3 pos_j = positions[cathodeIndices[j]];
                Vec3 rij = pos_i - pos_j;
                double r = sqrt(rij.dot(rij));
                if (r > 1e-6)
                    phi_i += cathodeCharges[j] / r;
            }

            // Contribution from other anode atoms
            for (int j = 0; j < numAnodes; j++) {
                if (i != j) {
                    Vec3 pos_j = positions[anodeIndices[j]];
                    Vec3 rij = pos_i - pos_j;
                    double r = sqrt(rij.dot(rij));
                    if (r > 1e-6)
                        phi_i += anodeCharges[j] / r;
                }
            }

            // Contribution from electrolyte atoms
            for (int j = 0; j < numElectrolytes; j++) {
                Vec3 pos_j = positions[electrolyteIndices[j]];
                Vec3 rij = pos_i - pos_j;
                double r = sqrt(rij.dot(rij));
                if (r > 1e-6)
                    phi_i += electrolyteCharges[j] / r;
            }

            phi_anode_sum += phi_i;
        }

        // Compute average potentials
        double phi_cathode_avg = (numCathodes > 0) ? phi_cathode_sum / numCathodes : 0.0;
        double phi_anode_avg = (numAnodes > 0) ? phi_anode_sum / numAnodes : 0.0;

        // Target potentials (voltage already in kJ/mol)
        double V_cathode = -voltage / 2.0;
        double V_anode = voltage / 2.0;

        // Compute charge deltas
        double dq_cathode = (V_cathode - phi_cathode_avg) * totalArea * EPSILON_0 / (numCathodes + 1e-10);
        double dq_anode = (V_anode - phi_anode_avg) * totalArea * EPSILON_0 / (numAnodes + 1e-10);

        // Green's Reciprocity: enforce charge conservation
        double Q_total = 0.0;
        for (int i = 0; i < numCathodes; i++)
            Q_total += cathodeCharges[i];
        for (int i = 0; i < numAnodes; i++)
            Q_total += anodeCharges[i];
        for (int i = 0; i < numElectrolytes; i++)
            Q_total += electrolyteCharges[i];

        double correction = -Q_total / (numCathodes + numAnodes + 1e-10);

        // Update charges
        for (int i = 0; i < numCathodes; i++)
            cathodeCharges[i] += dq_cathode / numCathodes + correction;

        for (int i = 0; i < numAnodes; i++)
            anodeCharges[i] += dq_anode / numAnodes + correction;
    }
}

double ReferenceCalcConstantVKernel::execute(ContextImpl& context, bool includeForces,
                                             bool includeEnergy, int groups)
{
    // Get positions
    vector<Vec3> positions;
    context.getPositions(positions);

    // Run SCF to update electrode charges
    runSCF(positions);

    // Compute electrostatic energy (simplified)
    double energy = 0.0;
    if (includeEnergy) {
        for (size_t i = 0; i < cathodeIndices.size(); i++) {
            for (size_t j = i+1; j < cathodeIndices.size(); j++) {
                Vec3 rij = positions[cathodeIndices[i]] - positions[cathodeIndices[j]];
                double r = sqrt(rij.dot(rij));
                if (r > 1e-6)
                    energy += cathodeCharges[i] * cathodeCharges[j] / r;
            }
        }

        for (size_t i = 0; i < anodeIndices.size(); i++) {
            for (size_t j = i+1; j < anodeIndices.size(); j++) {
                Vec3 rij = positions[anodeIndices[i]] - positions[anodeIndices[j]];
                double r = sqrt(rij.dot(rij));
                if (r > 1e-6)
                    energy += anodeCharges[i] * anodeCharges[j] / r;
            }
        }

        for (size_t i = 0; i < cathodeIndices.size(); i++) {
            for (size_t j = 0; j < anodeIndices.size(); j++) {
                Vec3 rij = positions[cathodeIndices[i]] - positions[anodeIndices[j]];
                double r = sqrt(rij.dot(rij));
                if (r > 1e-6)
                    energy += cathodeCharges[i] * anodeCharges[j] / r;
            }
        }

        energy *= 0.5;  // Factor of 1/2 from double counting
    }

    // TODO: Compute forces if includeForces is true

    return energy;
}

void ReferenceCalcConstantVKernel::updateParameters(ContextImpl& context, const ConstantVForce& force)
{
    voltage = force.getVoltage() * conversion_eV_to_kJmol;
    Lgap = force.getLgap();
    Lcell = force.getLcell();
    totalArea = force.getTotalArea();
    z_cathode = force.getZCathode();
    z_anode = force.getZAnode();
    nIterations = force.getNumIterations();
}

// ═══════════════════════════════════════════════════════════════════════════
// ReferenceIntegrateConstantVDrudeLangevinStepKernel
// ═══════════════════════════════════════════════════════════════════════════

ReferenceIntegrateConstantVDrudeLangevinStepKernel::ReferenceIntegrateConstantVDrudeLangevinStepKernel(
    string name, const Platform& platform) :
    KernelImpl(name, platform),
    dynamics(nullptr),
    stepCount(0)
{
}

ReferenceIntegrateConstantVDrudeLangevinStepKernel::~ReferenceIntegrateConstantVDrudeLangevinStepKernel() {
    if (dynamics)
        delete dynamics;
}

void ReferenceIntegrateConstantVDrudeLangevinStepKernel::initialize(
    const System& system,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    // Create dynamics object
    dynamics = new ReferenceConstantVDrudeLangevinDynamics(
        system.getNumParticles(),
        integrator.getStepSize(),
        integrator.getFriction(),
        integrator.getTemperature()
    );

    // Set electrode parameters
    for (int i = 0; i < integrator.getNumCathodeAtoms(); i++) {
        int particle;
        double area;
        integrator.getCathodeAtomParameters(i, particle, area);
        dynamics->addCathodeAtom(particle, area);
    }

    for (int i = 0; i < integrator.getNumAnodeAtoms(); i++) {
        int particle;
        double area;
        integrator.getAnodeAtomParameters(i, particle, area);
        dynamics->addAnodeAtom(particle, area);
    }

    for (int i = 0; i < integrator.getNumElectrolyteAtoms(); i++) {
        int particle;
        double charge;
        integrator.getElectrolyteAtomParameters(i, particle, charge);
        dynamics->addElectrolyteAtom(particle, charge);
    }

    dynamics->setVoltage(integrator.getVoltage());
    dynamics->setLgap(integrator.getLgap());
    dynamics->setLcell(integrator.getLcell());
    dynamics->setNumSCFIterations(integrator.getNumSCFIterations());
}

void ReferenceIntegrateConstantVDrudeLangevinStepKernel::execute(
    ContextImpl& context,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    // Get positions and velocities
    vector<Vec3> positions, velocities;
    context.getPositions(positions);
    context.getVelocities(velocities);

    // Check if we need to update charges
    if (stepCount % integrator.getSCFFrequency() == 0) {
        dynamics->updateElectrodeCharges(positions);
    }

    // Perform integration step
    vector<Vec3> forces(context.getSystem().getNumParticles());
    dynamics->update(context, positions, velocities, forces, integrator.getStepSize());

    // Update context
    context.setPositions(positions);
    context.setVelocities(velocities);

    stepCount++;
}
