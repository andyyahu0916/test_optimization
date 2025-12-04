/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * See https://openmm.org/development.                                        *
 *                                                                            *
 * Portions copyright (c) 2025 Stanford University and the Authors.           *
 * Authors: Peter Eastman                                                     *
 * Contributors:                                                              *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

#include "openmm/ConstantVDrudeLangevinIntegrator.h"
#include "openmm/Context.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/kernels.h"
#include "openmm/Kernel.h"
#ifdef OPENMM_BUILD_DRUDE_PLUGIN
#include "openmm/DrudeLangevinIntegrator.h"
#include "openmm/DrudeKernels.h"
#include "openmm/internal/DrudeHelpers.h"
#endif
#include <cmath>
#include <algorithm>

using namespace OpenMM;
using std::vector;
using std::string;

// Kernel interface is now defined in olla/include/openmm/kernels.h

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
    electrodesInitialized(false),
    stepCount(0)  // Initialize step counter
{
    if (scfIterations < 1)
        throw OpenMMException("Number of SCF iterations must be at least 1");

    if (Lgap <= 0.0 || Lcell <= 0.0)
        throw OpenMMException("Lgap and Lcell must be positive");
}

ConstantVDrudeLangevinIntegrator::~ConstantVDrudeLangevinIntegrator() {
}

// ═══════════════════════════════════════════════════════════════════════════
// Kernel Names and Cleanup
// ═══════════════════════════════════════════════════════════════════════════

vector<string> ConstantVDrudeLangevinIntegrator::getKernelNames() {
    vector<string> names;
    // Get parent kernel names (DrudeLangevinIntegrator)
    vector<string> parentNames = DrudeLangevinIntegrator::getKernelNames();
    names.insert(names.end(), parentNames.begin(), parentNames.end());
    
    // Add our custom kernel
    names.push_back("IntegrateConstantVDrudeLangevinStep");
    return names;
}

void ConstantVDrudeLangevinIntegrator::cleanup() {
    stepKernel = Kernel();  // Release kernel
    DrudeLangevinIntegrator::cleanup();  // Call parent cleanup
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

void ConstantVDrudeLangevinIntegrator::getCathodeAtomParameters(int index, int& particle, double& area) const {
    if (index < 0 || index >= (int)cathodeIndices.size())
        throw OpenMMException("Cathode atom index out of range");
    particle = cathodeIndices[index];
    area = cathodeAreas[index];
}

void ConstantVDrudeLangevinIntegrator::getAnodeAtomParameters(int index, int& particle, double& area) const {
    if (index < 0 || index >= (int)anodeIndices.size())
        throw OpenMMException("Anode atom index out of range");
    particle = anodeIndices[index];
    area = anodeAreas[index];
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
// Conductor Parameter Getters (FIX P2-3)
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVDrudeLangevinIntegrator::getBuckyballConductorParameters(
    int index,
    vector<int>& virtualIndices,
    vector<int>& realIndices,
    string& electrodeType,
    double& voltage) const
{
    if (index < 0 || index >= (int)buckyballs.size())
        throw OpenMMException("Buckyball conductor index out of range");

    const ConductorData& conductor = buckyballs[index];
    virtualIndices = conductor.virtualIndices;
    realIndices = conductor.realIndices;
    electrodeType = conductor.electrodeType;
    voltage = conductor.voltage;
}

void ConstantVDrudeLangevinIntegrator::getNanotubeConductorParameters(
    int index,
    vector<int>& virtualIndices,
    vector<int>& realIndices,
    string& electrodeType,
    double& voltage,
    Vec3& axis) const
{
    if (index < 0 || index >= (int)nanotubes.size())
        throw OpenMMException("Nanotube conductor index out of range");

    const ConductorData& conductor = nanotubes[index];
    virtualIndices = conductor.virtualIndices;
    realIndices = conductor.realIndices;
    electrodeType = conductor.electrodeType;
    voltage = conductor.voltage;
    axis = conductor.axis;
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
    if (context == NULL)
        throw OpenMMException("This Integrator is not bound to a context!");
    
    if (!electrodesInitialized)
        throw OpenMMException("Electrodes not initialized. Call Context creation first.");

    // FIX: Use our custom kernel that combines SCF + MD
    // This 100% aligns with original Python: Poisson_solver_fixed_voltage() + simmd.step()
    // Get platform-specific kernel implementation
    IntegrateConstantVDrudeLangevinStepKernel& kernelImpl = 
        stepKernel.getAs<IntegrateConstantVDrudeLangevinStepKernel>();

    for (int i = 0; i < steps; i++) {
        // Check if we need to update charges this step
        if ((stepCount % scfFrequency) == 0) {
            // Execute SCF + MD in single kernel call
            // This matches: Poisson_solver_fixed_voltage(Niterations) + simmd.step()
            kernelImpl.execute(*context, *this);
        } else {
            // Skip SCF, just do MD step (use parent integrator)
            DrudeLangevinIntegrator::step(1);
        }
        stepCount++;
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

    // FIX: Create platform-specific kernel
    stepKernel = context.getPlatform().createKernel("IntegrateConstantVDrudeLangevinStep", context);
    
    // Initialize the kernel with electrode data
    IntegrateConstantVDrudeLangevinStepKernel& kernelImpl = 
        dynamic_cast<IntegrateConstantVDrudeLangevinStepKernel&>(stepKernel.getImpl());
    
    kernelImpl.initialize(
        cathodeIndices,
        cathodeAreas,
        anodeIndices,
        anodeAreas,
        electrolyteIndices,
        electrolyteCharges,
        voltage * CONVERSION_EV_TO_KJMOL,  // Convert V to kJ/mol
        Lgap,
        Lcell,
        totalArea,
        z_cathode,
        z_anode,
        scfIterations
    );

    electrodesInitialized = true;
    stepCount = 0;  // Reset step counter
}

// Kernel interface is now defined at the top of the file (before use)
