#include "ConstantVDrudeLangevinIntegrator.h"
#include "ConstantVForce.h"
#include "ConstantVKernels.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/DrudeLangevinIntegrator.h"
#include "openmm/DrudeKernels.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/internal/DrudeHelpers.h"
#include <cmath>
#include <string>

using namespace ConstantVPlugin;
using namespace OpenMM;
using namespace std;

// ═══════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════

ConstantVDrudeLangevinIntegrator::ConstantVDrudeLangevinIntegrator(
    double temperature,
    double frictionCoeff,
    double drudeTemperature,
    double drudeFrictionCoeff,
    double stepSize
) : DrudeIntegrator(stepSize),
    temperature(temperature),
    friction(frictionCoeff),
    drudeFriction(drudeFrictionCoeff),
    voltage(0.0),
    nIterations(4),
    scfFrequency(1),
    Lgap(0.0),
    Lcell(0.0),
    totalArea(0.0),
    z_cathode(0.0),
    z_anode(0.0),
    stepCount(0)
{
    setDrudeTemperature(drudeTemperature);
    setMaxDrudeDistance(0.02);  // Default from DrudeLangevinIntegrator
    setConstraintTolerance(1e-5);
    setRandomNumberSeed(0);

    // Create internal DrudeLangevinIntegrator delegate
    // This is needed because IntegrateDrudeLangevinStepKernel requires a
    // DrudeLangevinIntegrator& reference (not just DrudeIntegrator)
    drudeLangevinDelegate = new DrudeLangevinIntegrator(
        temperature, frictionCoeff, drudeTemperature, drudeFrictionCoeff, stepSize
    );
}

// ═══════════════════════════════════════════════════════════
// Destructor
// ═══════════════════════════════════════════════════════════

ConstantVDrudeLangevinIntegrator::~ConstantVDrudeLangevinIntegrator() {
    // CRITICAL: Clean up delegate to prevent memory leak
    // This is called automatically when the integrator is destroyed
    // (e.g., when Python GC collects the object)
    cleanup();
}

// ═══════════════════════════════════════════════════════════
// Langevin Parameter Setters
// ═══════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::setTemperature(double temp) {
    if (temp < 0)
        throw OpenMMException("Temperature cannot be negative");
    temperature = temp;
    drudeLangevinDelegate->setTemperature(temp);
}

void ConstantVDrudeLangevinIntegrator::setFriction(double coeff) {
    if (coeff < 0)
        throw OpenMMException("Friction coefficient cannot be negative");
    friction = coeff;
    drudeLangevinDelegate->setFriction(coeff);
}

void ConstantVDrudeLangevinIntegrator::setDrudeFriction(double coeff) {
    if (coeff < 0)
        throw OpenMMException("Drude friction coefficient cannot be negative");
    drudeFriction = coeff;
    drudeLangevinDelegate->setDrudeFriction(coeff);
}

// ═══════════════════════════════════════════════════════════
// Constant Voltage Parameter Setters
// ═══════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::setNumSCFIterations(int n) {
    if (n < 1)
        throw OpenMMException("Number of SCF iterations must be >= 1");
    nIterations = n;
}

void ConstantVDrudeLangevinIntegrator::setSCFFrequency(int freq) {
    if (freq < 1)
        throw OpenMMException("SCF frequency must be >= 1");
    scfFrequency = freq;
}

// ═══════════════════════════════════════════════════════════
// Electrode Atom Management
// ═══════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::addCathodeAtom(int particle, double area) {
    cathodeAtomIndices.push_back(particle);
    cathodeAreas.push_back(area);
}

void ConstantVDrudeLangevinIntegrator::addAnodeAtom(int particle, double area) {
    anodeAtomIndices.push_back(particle);
    anodeAreas.push_back(area);
}

void ConstantVDrudeLangevinIntegrator::addElectrolyteAtom(int particle, double charge) {
    electrolyteAtomIndices.push_back(particle);
    electrolyteCharges.push_back(charge);
}

// ═══════════════════════════════════════════════════════════
// Integrator Initialization
// ═══════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::initialize(ContextImpl& contextRef) {
    if (owner != NULL && &contextRef.getOwner() != owner)
        throw OpenMMException("This Integrator is already bound to a context");

    // ───────────────────────────────────────────────────────
    // 1. Get DrudeForce (required for Drude Langevin kernel)
    // ───────────────────────────────────────────────────────
    const DrudeForce* drudeForce = getDrudeForce(contextRef);
    if (drudeForce == NULL)
        throw OpenMMException("ConstantVDrudeLangevinIntegrator requires a DrudeForce");

    context = &contextRef;
    owner = &contextRef.getOwner();

    // ───────────────────────────────────────────────────────
    // 2. Create OpenMM's Drude Langevin kernel (100% reuse!)
    // ───────────────────────────────────────────────────────
    // This kernel implements dual-temperature Langevin dynamics:
    // - Real atoms: Langevin at 'temperature' with 'friction'
    // - Drude particles: Langevin at 'drudeTemperature' with 'drudeFriction'
    // We delegate all integration to OpenMM's validated implementation.
    //
    // Note: We pass drudeLangevinDelegate instead of *this because
    // IntegrateDrudeLangevinStepKernel requires a DrudeLangevinIntegrator&
    drudeLangevinKernel = context->getPlatform().createKernel(
        IntegrateDrudeLangevinStepKernel::Name(), contextRef
    );
    drudeLangevinKernel.getAs<IntegrateDrudeLangevinStepKernel>()
        .initialize(contextRef.getSystem(), *drudeLangevinDelegate, *drudeForce);

    // ───────────────────────────────────────────────────────
    // 3. Create ConstantV SCF kernel
    // ───────────────────────────────────────────────────────
    // We need to pass electrode data to the kernel. To keep the kernel
    // interface clean (it expects a ConstantVForce object), we create a
    // temporary ConstantVForce with our data.
    ConstantVForce tempForce;
    tempForce.setVoltage(voltage);
    tempForce.setNumIterations(nIterations);
    tempForce.setLgap(Lgap);
    tempForce.setLcell(Lcell);
    tempForce.setTotalArea(totalArea);
    tempForce.setZCathode(z_cathode);
    tempForce.setZAnode(z_anode);

    // Add electrode atoms
    for (size_t i = 0; i < cathodeAtomIndices.size(); i++)
        tempForce.addCathodeAtom(cathodeAtomIndices[i], cathodeAreas[i]);
    for (size_t i = 0; i < anodeAtomIndices.size(); i++)
        tempForce.addAnodeAtom(anodeAtomIndices[i], anodeAreas[i]);
    for (size_t i = 0; i < electrolyteAtomIndices.size(); i++)
        tempForce.addElectrolyteAtom(electrolyteAtomIndices[i], electrolyteCharges[i]);

    // Create and initialize the SCF kernel
    calcConstantVKernel = context->getPlatform().createKernel(
        CalcConstantVKernel::Name(), contextRef
    );
    calcConstantVKernel.getAs<CalcConstantVKernel>()
        .initialize(contextRef.getSystem(), tempForce);
}

// ═══════════════════════════════════════════════════════════
// Integrator Step Function
// ═══════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::step(int steps) {
    if (context == NULL)
        throw OpenMMException("This Integrator is not bound to a context!");

    for (int i = 0; i < steps; ++i) {
        // ───────────────────────────────────────────────────────
        // 1. SCF Charge Update (every scfFrequency MD steps)
        // ───────────────────────────────────────────────────────
        if (stepCount % scfFrequency == 0) {
            // ⭐ CRITICAL: Execute SCF solver to update electrode charges
            // The SCF kernel:
            // 1. Calculates forces to get electric fields (Ez_external)
            // 2. Updates electrode charges via Maxwell boundary conditions
            // 3. Applies Green's Reciprocity normalization
            // 4. Calls invalidateMolecules() so next force calc sees new charges
            //
            // After SCF completes, electrode charges in NonbondedForce have changed,
            // which means forces will be different. We DON'T need to manually
            // recalculate forces here because the Drude Langevin kernel will
            // call calcForcesAndEnergy() below, which will use the updated charges.
            calcConstantVKernel.getAs<CalcConstantVKernel>()
                .execute(*context, true, false);
        }

        // ───────────────────────────────────────────────────────
        // 2. Drude Langevin Integration (every step, 100% OpenMM)
        // ───────────────────────────────────────────────────────
        // This is identical to DrudeLangevinIntegrator::step()
        // - updateContextState(): Update any time-dependent parameters
        // - calcForcesAndEnergy(): Compute forces using current charges
        //   (including updated electrode charges from SCF if this was an SCF step)
        // - execute(): Perform one Langevin integration step with dual-temperature
        //
        // ⭐ CRITICAL: Exclude ConstantVForce (Group 31) to prevent double SCF execution
        // Without this, calcForcesAndEnergy() would trigger ConstantVForce again,
        // causing SCF to run twice (once explicitly above, once implicitly here)
        context->updateContextState();
        int forceGroups = getIntegrationForceGroups();
        forceGroups &= ~(1U << 31);  // Exclude Group 31 (ConstantVForce)
        context->calcForcesAndEnergy(true, false, forceGroups);
        drudeLangevinKernel.getAs<IntegrateDrudeLangevinStepKernel>()
            .execute(*context, *drudeLangevinDelegate);

        stepCount++;
    }
}

// ═══════════════════════════════════════════════════════════
// Kinetic Energy Computation
// ═══════════════════════════════════════════════════════════

double ConstantVDrudeLangevinIntegrator::computeKineticEnergy() {
    // Delegate to Drude Langevin kernel (it correctly handles Drude KE)
    return drudeLangevinKernel.getAs<IntegrateDrudeLangevinStepKernel>()
        .computeKineticEnergy(*context, *drudeLangevinDelegate);
}

// ═══════════════════════════════════════════════════════════
// Cleanup
// ═══════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::cleanup() {
    drudeLangevinKernel = Kernel();
    calcConstantVKernel = Kernel();
    if (drudeLangevinDelegate != NULL) {
        delete drudeLangevinDelegate;
        drudeLangevinDelegate = NULL;
    }
}

// ═══════════════════════════════════════════════════════════
// Kernel Names
// ═══════════════════════════════════════════════════════════

vector<string> ConstantVDrudeLangevinIntegrator::getKernelNames() {
    vector<string> names;
    names.push_back(IntegrateDrudeLangevinStepKernel::Name());
    names.push_back(CalcConstantVKernel::Name());
    return names;
}
