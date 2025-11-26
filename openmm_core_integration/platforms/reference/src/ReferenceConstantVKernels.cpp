/* -------------------------------------------------------------------------- *
 *                Reference ConstantV Kernel Implementation                   *
 * -------------------------------------------------------------------------- *
 * IMPORTANT: This implementation uses the ELECTRIC FIELD method, matching    *
 * the original Python implementation (MM_classes.py) and CUDA kernel.        *
 * -------------------------------------------------------------------------- */

#include "ReferenceConstantVKernels.h"
#include "ReferenceConstantVDrudeLangevinDynamics.h"
#include "openmm/OpenMMException.h"
#include "openmm/NonbondedForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/Context.h"
#include <cmath>

using namespace OpenMM;
using namespace std;

// ═══════════════════════════════════════════════════════════════════════════
// Physical Constants (matching Fixed_Voltage_routines.py)
// ═══════════════════════════════════════════════════════════════════════════

static const double CONVERSION_NM_TO_BOHR = 18.8973;
static const double CONVERSION_KJMOL_NM_TO_AU = CONVERSION_NM_TO_BOHR / 2625.5;  // ≈ 0.00719924
static const double CONVERSION_EV_TO_KJMOL = 96.487;
static const double FOUR_PI = 12.566370614359172;
static const double SMALL_THRESHOLD = 1e-6;

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
    this->voltage = voltage * CONVERSION_EV_TO_KJMOL;  // V to kJ/mol
    this->Lgap = Lgap;
    this->Lcell = Lcell;
    this->totalArea = totalArea;
    this->z_cathode = z_cathode;
    this->z_anode = z_anode;
    this->nIterations = nIterations;

    // Initialize charges with small non-zero value (prevents division by zero in E-field calc)
    cathodeCharges.resize(cathodeIndices.size(), SMALL_THRESHOLD);
    anodeCharges.resize(anodeIndices.size(), -SMALL_THRESHOLD);
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
    BuckyballConductor bucky;
    bucky.virtualIndices = virtualAtomIndices;
    bucky.realIndices = realAtomIndices;
    bucky.normals = normalVectors;
    bucky.areaPerAtom = areaPerAtom;
    bucky.radius = radius;
    bucky.center = center;
    bucky.contactAtomIndex = contactAtomIndex;
    bucky.contactDistance = contactDistance;
    bucky.voltage_kjmol = voltage * CONVERSION_EV_TO_KJMOL;
    bucky.electrodeType = (electrodeType == "cathode") ? 'c' : 'a';
    bucky.charges.resize(virtualAtomIndices.size(), 0.0);

    buckyballs.push_back(bucky);
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
    NanotubeConductor tube;
    tube.virtualIndices = virtualAtomIndices;
    tube.realIndices = realAtomIndices;
    tube.normals = normalVectors;
    tube.areaPerAtom = areaPerAtom;
    tube.axis = axis;
    tube.center = center;
    tube.contactAtomIndex = contactAtomIndex;
    tube.contactDistance = contactDistance;
    tube.voltage_kjmol = voltage * CONVERSION_EV_TO_KJMOL;
    tube.electrodeType = (electrodeType == "cathode") ? 'c' : 'a';
    tube.charges.resize(virtualAtomIndices.size(), 0.0);

    nanotubes.push_back(tube);
}

/**
 * Compute Q_analytic for an electrode using Green's Reciprocity.
 *
 * Q_analytic = ±1/(4π) × Area × (V/Lgap + V/Lcell) × K_au
 *              + Σ_electrolyte (z_distance / Lcell) × (-q_i)
 *
 * This matches Fixed_Voltage_routines.py::compute_Electrode_charge_analytic()
 */
double ReferenceCalcConstantVKernel::computeAnalyticCharge(
    const vector<Vec3>& positions,
    bool isCathode) const
{
    // Sign: +1 for cathode, -1 for anode
    double sign = isCathode ? 1.0 : -1.0;

    // z_opposite: for cathode, use anode z; for anode, use cathode z
    double z_opposite = isCathode ? z_anode : z_cathode;

    // ═══════════════════════════════════════════════════════════════════════
    // Geometric contribution: ±1/(4π) × Area × (V/Lgap + V/Lcell) × K_au
    // Corresponds to: Fixed_Voltage_routines.py L322-325
    // ═══════════════════════════════════════════════════════════════════════

    double Q_analytic = sign / FOUR_PI * totalArea *
                        (voltage / Lgap + voltage / Lcell) * CONVERSION_KJMOL_NM_TO_AU;

    // ═══════════════════════════════════════════════════════════════════════
    // Image charge contribution from electrolyte atoms
    // Σ (z_distance / Lcell) × (-q_i)
    // Corresponds to: Fixed_Voltage_routines.py L333-338
    // ═══════════════════════════════════════════════════════════════════════

    for (size_t i = 0; i < electrolyteIndices.size(); i++) {
        int idx = electrolyteIndices[i];
        double z_atom = positions[idx][2];  // z component
        double z_distance = fabs(z_atom - z_opposite);
        double q_i = electrolyteCharges[i];

        Q_analytic += (z_distance / Lcell) * (-q_i);
    }

    // ═══════════════════════════════════════════════════════════════════════
    // Image charge contribution from conductors
    // Conductors are treated as "part of electrolyte" for flat electrodes
    // Corresponds to: Fixed_Voltage_routines.py L340-348
    // ═══════════════════════════════════════════════════════════════════════

    for (const auto& bucky : buckyballs) {
        for (size_t i = 0; i < bucky.virtualIndices.size(); i++) {
            int idx = bucky.virtualIndices[i];
            double z_atom = positions[idx][2];
            double z_distance = fabs(z_atom - z_opposite);
            double q_i = bucky.charges[i];

            Q_analytic += (z_distance / Lcell) * (-q_i);
        }
    }

    for (const auto& tube : nanotubes) {
        for (size_t i = 0; i < tube.virtualIndices.size(); i++) {
            int idx = tube.virtualIndices[i];
            double z_atom = positions[idx][2];
            double z_distance = fabs(z_atom - z_opposite);
            double q_i = tube.charges[i];

            Q_analytic += (z_distance / Lcell) * (-q_i);
        }
    }

    return Q_analytic;
}

/**
 * Scale electrode charges to match analytic normalization.
 *
 * scale_factor = Q_analytic / Q_numeric
 * q_scaled = q_numeric × scale_factor
 *
 * This matches Fixed_Voltage_routines.py::Scale_charges_analytic()
 */
void ReferenceCalcConstantVKernel::scaleChargesAnalytic(
    vector<double>& charges,
    double Q_analytic) const
{
    // Sum numeric charges
    double Q_numeric = 0.0;
    for (double q : charges)
        Q_numeric += q;

    // Compute scale factor (avoid division by zero)
    double scale_factor = 1.0;
    if (fabs(Q_numeric) > SMALL_THRESHOLD) {
        scale_factor = Q_analytic / Q_numeric;
    }

    // Apply scaling
    for (double& q : charges) {
        q *= scale_factor;
    }
}

/**
 * Run SCF iteration using the ELECTRIC FIELD method.
 *
 * This matches the original Python implementation in MM_classes.py:
 *   1. Compute Ez_external = F_z / q_old (electric field from forces)
 *   2. Update charge: q_new = 2/(4π) × area × (V/Lgap + Ez_external) × K_au
 *   3. Scale to analytic normalization
 *
 * CRITICAL: This uses E-field (not potential) to match Python/CUDA.
 */
void ReferenceCalcConstantVKernel::runSCF(
    const vector<Vec3>& positions,
    const vector<Vec3>& forces)
{
    const int numCathodes = cathodeIndices.size();
    const int numAnodes = anodeIndices.size();

    if (numCathodes == 0 && numAnodes == 0)
        return;

    // ═══════════════════════════════════════════════════════════════════════
    // Step 1: Compute Q_analytic for both electrodes (includes image charges)
    // Corresponds to: MM_classes.py L700-701
    // ═══════════════════════════════════════════════════════════════════════

    double Q_analytic_cathode = computeAnalyticCharge(positions, true);   // isCathode = true
    double Q_analytic_anode = computeAnalyticCharge(positions, false);    // isCathode = false

    // ═══════════════════════════════════════════════════════════════════════
    // Step 2: SCF iteration loop
    // Corresponds to: MM_classes.py L704-768
    // ═══════════════════════════════════════════════════════════════════════

    for (int iter = 0; iter < nIterations; iter++) {

        // ═══════════════════════════════════════════════════════════════════
        // Step 2a: Update cathode charges using E-field method
        // Corresponds to: MM_classes.py L724-742
        // q_i = 2/(4π) × area × (V/Lgap + Ez_external) × K_au
        // ═══════════════════════════════════════════════════════════════════

        for (int i = 0; i < numCathodes; i++) {
            int idx = cathodeIndices[i];
            double q_old = cathodeCharges[i];
            double area = cathodeAreas[i];

            // Compute Ez_external from force
            // Ez = F_z / q (in kJ/mol/nm units)
            double Ez_external = 0.0;
            if (fabs(q_old) > 0.9 * SMALL_THRESHOLD) {
                Ez_external = forces[idx][2] / q_old;  // F_z / q
            }

            // Update charge (positive for cathode)
            // Corresponds to: MM_classes.py L738
            double factor = 2.0 / FOUR_PI * CONVERSION_KJMOL_NM_TO_AU;
            double V_over_Lgap = voltage / Lgap;
            double q_new = factor * area * (V_over_Lgap + Ez_external);

            // Don't allow charge to go below threshold
            if (fabs(q_new) < SMALL_THRESHOLD) {
                q_new = SMALL_THRESHOLD;  // Positive for cathode
            }

            cathodeCharges[i] = q_new;
        }

        // ═══════════════════════════════════════════════════════════════════
        // Step 2b: Update anode charges using E-field method
        // Corresponds to: MM_classes.py L744-760
        // q_i = -2/(4π) × area × (V/Lgap + Ez_external) × K_au (negative sign for anode)
        // ═══════════════════════════════════════════════════════════════════

        for (int i = 0; i < numAnodes; i++) {
            int idx = anodeIndices[i];
            double q_old = anodeCharges[i];
            double area = anodeAreas[i];

            // Compute Ez_external from force
            double Ez_external = 0.0;
            if (fabs(q_old) > 0.9 * SMALL_THRESHOLD) {
                Ez_external = forces[idx][2] / q_old;
            }

            // Update charge (negative for anode)
            // Corresponds to: MM_classes.py L754
            double factor = 2.0 / FOUR_PI * CONVERSION_KJMOL_NM_TO_AU;
            double V_over_Lgap = voltage / Lgap;
            double q_new = -factor * area * (V_over_Lgap + Ez_external);  // Note: negative sign

            // Don't allow charge to go below threshold
            if (fabs(q_new) < SMALL_THRESHOLD) {
                q_new = -SMALL_THRESHOLD;  // Negative for anode
            }

            anodeCharges[i] = q_new;
        }

        // ═══════════════════════════════════════════════════════════════════
        // Step 2c: Update Buckyball conductor charges (if any)
        // Corresponds to: MM_classes.py::Numerical_charge_Conductor()
        // ═══════════════════════════════════════════════════════════════════

        for (auto& bucky : buckyballs) {
            for (size_t i = 0; i < bucky.virtualIndices.size(); i++) {
                int idx = bucky.virtualIndices[i];
                double q_old = bucky.charges[i];
                Vec3 normal = bucky.normals[i];

                // Compute normal component of E-field
                double E_n_external = 0.0;
                if (fabs(q_old) > 0.9 * SMALL_THRESHOLD) {
                    double F_n = forces[idx][0] * normal[0] +
                                 forces[idx][1] * normal[1] +
                                 forces[idx][2] * normal[2];
                    E_n_external = F_n / q_old;
                }

                // Update charge
                double factor = 2.0 / FOUR_PI * CONVERSION_KJMOL_NM_TO_AU;
                double q_new = factor * bucky.areaPerAtom * E_n_external;

                if (fabs(q_new) < SMALL_THRESHOLD) {
                    q_new = SMALL_THRESHOLD;
                }

                bucky.charges[i] = q_new;
            }
        }

        // ═══════════════════════════════════════════════════════════════════
        // Step 2d: Recompute Q_analytic if conductors present
        // Corresponds to: MM_classes.py L764-766
        // ═══════════════════════════════════════════════════════════════════

        if (!buckyballs.empty() || !nanotubes.empty()) {
            Q_analytic_cathode = computeAnalyticCharge(positions, true);
            Q_analytic_anode = computeAnalyticCharge(positions, false);
        }

        // ═══════════════════════════════════════════════════════════════════
        // Step 2e: Scale charges to match analytic normalization
        // Corresponds to: MM_classes.py L768 (Scale_charges_analytic_general)
        // ═══════════════════════════════════════════════════════════════════

        if (buckyballs.empty() && nanotubes.empty()) {
            // No conductors: scale each electrode independently
            scaleChargesAnalytic(cathodeCharges, Q_analytic_cathode);
            scaleChargesAnalytic(anodeCharges, Q_analytic_anode);
        } else {
            // With conductors: scale cathode + conductors together
            // Anode scaled independently
            // Corresponds to: MM_classes.py::Scale_charges_analytic_general() L527-545

            // Sum cathode + conductor charges
            double Q_cathode_plus_cond = 0.0;
            for (double q : cathodeCharges)
                Q_cathode_plus_cond += q;
            for (const auto& bucky : buckyballs) {
                for (double q : bucky.charges)
                    Q_cathode_plus_cond += q;
            }
            for (const auto& tube : nanotubes) {
                for (double q : tube.charges)
                    Q_cathode_plus_cond += q;
            }

            // Compute scale factor (use -Q_analytic_anode for cathode side)
            double scale_cathode = 1.0;
            if (fabs(Q_cathode_plus_cond) > SMALL_THRESHOLD) {
                scale_cathode = (-Q_analytic_anode) / Q_cathode_plus_cond;
            }

            // Apply to cathode
            for (double& q : cathodeCharges)
                q *= scale_cathode;

            // Apply to conductors
            for (auto& bucky : buckyballs) {
                for (double& q : bucky.charges)
                    q *= scale_cathode;
            }
            for (auto& tube : nanotubes) {
                for (double& q : tube.charges)
                    q *= scale_cathode;
            }

            // Scale anode independently
            scaleChargesAnalytic(anodeCharges, Q_analytic_anode);
        }
    }
}

/**
 * Legacy runSCF without forces (for backward compatibility).
 * Uses zero forces, which gives V/Lgap term only.
 */
void ReferenceCalcConstantVKernel::runSCF(const vector<Vec3>& positions) {
    // Create zero forces
    vector<Vec3> zeroForces(positions.size(), Vec3(0, 0, 0));
    runSCF(positions, zeroForces);
}

double ReferenceCalcConstantVKernel::execute(ContextImpl& context, bool includeForces,
                                             bool includeEnergy, int groups)
{
    // Get positions and forces
    vector<Vec3> positions;
    vector<Vec3> forces;
    context.getPositions(positions);
    
    // Compute forces from all force groups (needed for E-field calculation)
    context.calcForcesAndEnergy(true, false, groups);
    forces.resize(context.getSystem().getNumParticles());
    context.getForces(forces);

    // Run SCF to update electrode charges using E-field method
    runSCF(positions, forces);

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
    voltage = force.getVoltage() * CONVERSION_EV_TO_KJMOL;
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
    stepCount(0),
    nonbondedForce(nullptr)
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
    dynamics->setTotalArea(integrator.getTotalArea());
    dynamics->setZCathode(integrator.getZCathode());
    dynamics->setZAnode(integrator.getZAnode());
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

    initializeChargeCache(context);

    vector<Vec3> forces(context.getSystem().getNumParticles(), Vec3(0, 0, 0));

    // Check if we need to update charges (SCF loop)
    if (stepCount % integrator.getSCFFrequency() == 0) {
        context.calcForcesAndEnergy(true, false, -1);
        context.getForces(forces);

        dynamics->updateElectrodeCharges(positions, forces, cachedCharges);
        applyChargeUpdates(context);

        // Recompute forces after charge update to keep dynamics consistent
        context.calcForcesAndEnergy(true, false, -1);
        context.getForces(forces);
    }

    // Perform integration step with (possibly) refreshed forces
    dynamics->update(context, positions, velocities, forces, integrator.getStepSize());

    // Update context
    context.setPositions(positions);
    context.setVelocities(velocities);

    stepCount++;
}

void ReferenceIntegrateConstantVDrudeLangevinStepKernel::initializeChargeCache(ContextImpl& context) {
    if (nonbondedForce != nullptr)
        return;

    const System& system = context.getSystem();
    for (int i = 0; i < system.getNumForces(); i++) {
        const Force& force = system.getForce(i);
        const NonbondedForce* candidate = dynamic_cast<const NonbondedForce*>(&force);
        if (candidate != nullptr) {
            nonbondedForce = const_cast<NonbondedForce*>(candidate);
            break;
        }
    }

    if (nonbondedForce == nullptr)
        throw OpenMMException("ConstantVDrudeLangevinIntegrator requires a NonbondedForce in the System");

    int numParticles = nonbondedForce->getNumParticles();
    cachedCharges.resize(numParticles);
    cachedSigma.resize(numParticles);
    cachedEpsilon.resize(numParticles);
    for (int i = 0; i < numParticles; i++) {
        double charge, sigma, epsilon;
        nonbondedForce->getParticleParameters(i, charge, sigma, epsilon);
        cachedCharges[i] = charge;
        cachedSigma[i] = sigma;
        cachedEpsilon[i] = epsilon;
    }
}

void ReferenceIntegrateConstantVDrudeLangevinStepKernel::applyChargeUpdates(ContextImpl& context) {
    if (nonbondedForce == nullptr)
        return;

    for (int i = 0; i < (int) cachedCharges.size(); i++)
        nonbondedForce->setParticleParameters(i, cachedCharges[i], cachedSigma[i], cachedEpsilon[i]);

    nonbondedForce->updateParametersInContext(context);
}
