/* -------------------------------------------------------------------------- *
 *                          OpenMM - Native ConstantV                        *
 * -------------------------------------------------------------------------- *
 * Implementation of ConstantVDrudeLangevinIntegrator                        *
 * -------------------------------------------------------------------------- */

#include "openmm/ConstantVDrudeLangevinIntegrator.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"
#include <cmath>
#include <algorithm>

using namespace OpenMM;
using std::vector;
using std::string;

// ═══════════════════════════════════════════════════════════════════════════
// Physical Constants (from professor's code)
// ═══════════════════════════════════════════════════════════════════════════

static const double CONVERSION_NM_TO_BOHR = 18.8973;
static const double CONVERSION_KJMOL_NM_TO_AU = CONVERSION_NM_TO_BOHR / 2625.5;
static const double CONVERSION_EV_TO_KJMOL = 96.487;

// ═══════════════════════════════════════════════════════════════════════════
// Constructor
// ═══════════════════════════════════════════════════════════════════════════

ConstantVDrudeLangevinIntegrator::ConstantVDrudeLangevinIntegrator(
    double temperature,
    double frictionCoeff,
    double drudeTemperature,
    double drudeFrictionCoeff,
    double stepSize,
    double voltage,
    double Lgap,
    double Lcell,
    int scfIterations
) : DrudeLangevinIntegrator(temperature, frictionCoeff, drudeTemperature,
                             drudeFrictionCoeff, stepSize),
    voltage(voltage),
    Lgap(Lgap),
    Lcell(Lcell),
    totalArea(0.0),
    z_cathode(0.0),
    z_anode(0.0),
    scfIterations(scfIterations),
    scfFrequency(1),  // Default: update every step
    electrodesInitialized(false)
{
    if (scfIterations < 1)
        throw OpenMMException("Number of SCF iterations must be at least 1");

    if (Lgap <= 0.0 || Lcell <= 0.0)
        throw OpenMMException("Lgap and Lcell must be positive");
}

ConstantVDrudeLangevinIntegrator::~ConstantVDrudeLangevinIntegrator() {
}

// ═══════════════════════════════════════════════════════════════════════════
// Electrode Configuration Methods
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::addCathodeAtom(int particle, double area) {
    cathodeIndices.push_back(particle);
    cathodeAreas.push_back(area);
}

void ConstantVDrudeLangevinIntegrator::addAnodeAtom(int particle, double area) {
    anodeIndices.push_back(particle);
    anodeAreas.push_back(area);
}

void ConstantVDrudeLangevinIntegrator::addElectrolyteAtom(int particle, double charge) {
    electrolyteIndices.push_back(particle);
    electrolyteCharges.push_back(charge);
}

void ConstantVDrudeLangevinIntegrator::addBuckyballConductor(
    const vector<int>& virtualIndices,
    const vector<int>& realIndices,
    const string& electrodeType,
    double voltage
) {
    if (electrodeType != "cathode" && electrodeType != "anode")
        throw OpenMMException("electrodeType must be 'cathode' or 'anode'");

    ConductorData conductor;
    conductor.virtualIndices = virtualIndices;
    conductor.realIndices = realIndices;
    conductor.electrodeType = electrodeType;
    conductor.voltage = voltage;
    conductor.axis = Vec3(0, 0, 0);  // Not used for Buckyball

    // Zip-sort virtual and real indices together (CRITICAL for cache coherency)
    vector<std::pair<int, int>> pairs;
    pairs.reserve(virtualIndices.size());
    for (size_t i = 0; i < virtualIndices.size(); i++)
        pairs.push_back({virtualIndices[i], realIndices[i]});

    std::sort(pairs.begin(), pairs.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.first < b.first;
        });

    for (size_t i = 0; i < pairs.size(); i++) {
        conductor.virtualIndices[i] = pairs[i].first;
        conductor.realIndices[i] = pairs[i].second;
    }

    buckyballs.push_back(conductor);
}

void ConstantVDrudeLangevinIntegrator::addNanotubeConductor(
    const vector<int>& virtualIndices,
    const vector<int>& realIndices,
    const string& electrodeType,
    double voltage,
    const Vec3& axis
) {
    if (electrodeType != "cathode" && electrodeType != "anode")
        throw OpenMMException("electrodeType must be 'cathode' or 'anode'");

    // Validate axis is normalized
    double norm = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
    if (std::abs(norm - 1.0) > 0.01)
        throw OpenMMException("Nanotube axis must be normalized (magnitude = 1.0)");

    ConductorData conductor;
    conductor.virtualIndices = virtualIndices;
    conductor.realIndices = realIndices;
    conductor.electrodeType = electrodeType;
    conductor.voltage = voltage;
    conductor.axis = axis;

    // Zip-sort
    vector<std::pair<int, int>> pairs;
    pairs.reserve(virtualIndices.size());
    for (size_t i = 0; i < virtualIndices.size(); i++)
        pairs.push_back({virtualIndices[i], realIndices[i]});

    std::sort(pairs.begin(), pairs.end(),
        [](const std::pair<int, int>& a, const std::pair<int, int>& b) {
            return a.first < b.first;
        });

    for (size_t i = 0; i < pairs.size(); i++) {
        conductor.virtualIndices[i] = pairs[i].first;
        conductor.realIndices[i] = pairs[i].second;
    }

    nanotubes.push_back(conductor);
}

// ═══════════════════════════════════════════════════════════════════════════
// Query Methods
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::getElectrodeCharges(
    vector<double>& cathodeCharges,
    vector<double>& anodeCharges
) const {
    // This method must be called AFTER Context is created
    // Query NonbondedForce parameters directly

    // Note: In native integration, we need ContextImpl access
    // This is a placeholder - actual implementation would query kernel state
    cathodeCharges.resize(cathodeIndices.size());
    anodeCharges.resize(anodeIndices.size());

    // TODO: Implement via kernel->getCharges() interface
}

double ConstantVDrudeLangevinIntegrator::getTotalCathodeCharge() const {
    vector<double> charges;
    vector<double> dummy;
    getElectrodeCharges(charges, dummy);

    double total = 0.0;
    for (double q : charges)
        total += q;
    return total;
}

double ConstantVDrudeLangevinIntegrator::getTotalAnodeCharge() const {
    vector<double> dummy;
    vector<double> charges;
    getElectrodeCharges(dummy, charges);

    double total = 0.0;
    for (double q : charges)
        total += q;
    return total;
}

// ═══════════════════════════════════════════════════════════════════════════
// Integration Step (The Core)
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::step(int steps) {
    if (!electrodesInitialized)
        throw OpenMMException("Electrodes not initialized. Call Context creation first.");

    // For each integration step:
    // 1. Run SCF charge update (via custom kernel)
    // 2. Call parent DrudeLangevinIntegrator::step()

    for (int i = 0; i < steps; i++) {
        // Step 1: SCF Charge Update
        // This would call a custom kernel: IntegrateConstantVDrudeLangevinStepKernel
        // The kernel performs:
        //   - Compute forces (NonbondedForce)
        //   - Update electrode charges (SCF loop)
        //   - Apply Green's Reciprocity scaling

        // TODO: Implement via custom kernel interface
        // kernel.updateElectrodeCharges(scfIterations);

        // Step 2: Integrate dynamics
        DrudeLangevinIntegrator::step(1);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Initialization (Called by ContextImpl)
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::initialize(ContextImpl& context) {
    // Call parent initialization first
    DrudeLangevinIntegrator::initialize(context);

    // Validate electrode configuration
    if (cathodeIndices.empty() || anodeIndices.empty())
        throw OpenMMException("Must add cathode and anode atoms before creating Context");

    if (totalArea <= 0.0)
        throw OpenMMException("Must set total electrode area before creating Context");

    // Initialize platform-specific kernel
    // This would create the custom CUDA/Reference kernel that handles SCF

    // TODO: Register custom kernel with ContextImpl
    // context.getPlatform().registerKernel(IntegrateConstantVDrudeLangevinStepKernel);

    electrodesInitialized = true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel Interface Definition (for platforms to implement)
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Custom kernel interface for ConstantV integration.
 *
 * Platform-specific implementations (CUDA, Reference) must implement:
 *   - initializeElectrodes(): Upload electrode data to GPU
 *   - updateElectrodeCharges(): Run SCF loop
 *   - integrate(): Perform Langevin integration step
 *
 * This kernel combines charge update + dynamics integration in a single
 * kernel launch to minimize memory transfers.
 */
class IntegrateConstantVDrudeLangevinStepKernel : public KernelImpl {
public:
    static std::string Name() {
        return "IntegrateConstantVDrudeLangevinStep";
    }

    /**
     * Initialize electrode data structures (called once).
     */
    virtual void initializeElectrodes(
        const std::vector<int>& cathodeIndices,
        const std::vector<double>& cathodeAreas,
        const std::vector<int>& anodeIndices,
        const std::vector<double>& anodeAreas,
        const std::vector<int>& electrolyteIndices,
        double voltage,
        double Lgap,
        double Lcell,
        double totalArea,
        double z_cathode,
        double z_anode
    ) = 0;

    /**
     * Update electrode charges (SCF loop).
     */
    virtual void updateElectrodeCharges(int scfIterations) = 0;

    /**
     * Perform integration step with updated charges.
     */
    virtual void execute(
        ContextImpl& context,
        const ConstantVDrudeLangevinIntegrator& integrator
    ) = 0;
};
