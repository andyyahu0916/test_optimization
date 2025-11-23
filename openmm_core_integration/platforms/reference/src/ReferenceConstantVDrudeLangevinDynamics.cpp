/* -------------------------------------------------------------------------- *
 *                          OpenMM - Native ConstantV                        *
 * -------------------------------------------------------------------------- *
 * Reference Platform Implementation                                         *
 * -------------------------------------------------------------------------- */

#include "ReferenceConstantVDrudeLangevinDynamics.h"
#include "openmm/OpenMMException.h"
#include <cmath>
#include <algorithm>

using namespace OpenMM;
using std::vector;

// Physical constants (from professor's code)
static const double CONVERSION_NM_TO_BOHR = 18.8973;
static const double CONVERSION_KJMOL_NM_TO_AU = CONVERSION_NM_TO_BOHR / 2625.5;
static const double SMALL_THRESHOLD = 1e-6;
static const double FOUR_PI = 4.0 * M_PI;

ReferenceConstantVDrudeLangevinDynamics::ReferenceConstantVDrudeLangevinDynamics(
    int numParticles,
    double temperature,
    double friction,
    double drudeTemperature,
    double drudeFriction,
    double stepSize
) : ReferenceDrudeLangevinDynamics(numParticles, temperature, friction,
                                    drudeTemperature, drudeFriction, stepSize),
    voltage(0.0),
    Lgap(0.0),
    Lcell(0.0),
    totalArea(0.0),
    z_cathode(0.0),
    z_anode(0.0),
    scfIterations(4)
{
}

void ReferenceConstantVDrudeLangevinDynamics::updateElectrodeCharges(
    vector<Vec3>& positions,
    vector<Vec3>& forces,
    vector<double>& charges
) {
    // ═══════════════════════════════════════════════════════════════════════
    // SCF Loop (Professor's Algorithm)
    // ═══════════════════════════════════════════════════════════════════════

    for (int iter = 0; iter < scfIterations; iter++) {
        // Step 1: Compute analytic charges (Green's Reciprocity)
        double Q_analytic_cathode = computeAnalyticCharge(
            cathodeIndices, positions, charges, +1.0, z_anode
        );
        double Q_analytic_anode = computeAnalyticCharge(
            anodeIndices, positions, charges, -1.0, z_cathode
        );

        // Step 2: Update flat electrode charges
        updateFlatElectrodeCharges(
            cathodeIndices, cathodeAreas, forces, charges, +2.0
        );
        updateFlatElectrodeCharges(
            anodeIndices, anodeAreas, forces, charges, -2.0
        );

        // Step 3: Update conductor charges (if present)
        // ... (implementation omitted for brevity)

        // Step 4: Scale charges (Green's Reciprocity)
        scaleCharges(cathodeIndices, charges, Q_analytic_cathode);
        scaleCharges(anodeIndices, charges, Q_analytic_anode);
    }
}

double ReferenceConstantVDrudeLangevinDynamics::computeAnalyticCharge(
    const vector<int>& electrodeIndices,
    const vector<Vec3>& positions,
    const vector<double>& charges,
    double sign,
    double z_opposite
) {
    // Geometric contribution
    double Q_analytic = sign / FOUR_PI * totalArea *
                        (voltage / Lgap + voltage / Lcell) *
                        CONVERSION_KJMOL_NM_TO_AU;

    // Image charge contribution (electrolyte)
    for (int index : electrolyteIndices) {
        double q_i = charges[index];
        double z_atom = positions[index][2];
        double z_distance = std::abs(z_atom - z_opposite);
        Q_analytic += (z_distance / Lcell) * (-q_i);
    }

    return Q_analytic;
}

void ReferenceConstantVDrudeLangevinDynamics::updateFlatElectrodeCharges(
    const vector<int>& electrodeIndices,
    const vector<double>& areas,
    const vector<Vec3>& forces,
    vector<double>& charges,
    double sign
) {
    for (size_t i = 0; i < electrodeIndices.size(); i++) {
        int atomIdx = electrodeIndices[i];
        double area = areas[i];
        double q_old = charges[atomIdx];
        double F_z = forces[atomIdx][2];

        // Compute external field
        double Ez_external = 0.0;
        if (std::abs(q_old) > (0.9 * SMALL_THRESHOLD)) {
            Ez_external = F_z / q_old;
        }

        // Update charge
        double factor = sign / FOUR_PI * CONVERSION_KJMOL_NM_TO_AU;
        double v_over_lgap = voltage / Lgap;
        double q_new = factor * area * (v_over_lgap + Ez_external);

        // Low-charge protection
        if (std::abs(q_new) < SMALL_THRESHOLD) {
            q_new = sign / 2.0 * SMALL_THRESHOLD;
        }

        charges[atomIdx] = q_new;
    }
}

void ReferenceConstantVDrudeLangevinDynamics::scaleCharges(
    const vector<int>& electrodeIndices,
    vector<double>& charges,
    double Q_analytic
) {
    // Compute numeric charge
    double Q_numeric = 0.0;
    for (int idx : electrodeIndices) {
        Q_numeric += charges[idx];
    }

    // Scale factor
    double scale_factor = -1.0;
    if (std::abs(Q_numeric) > SMALL_THRESHOLD) {
        scale_factor = Q_analytic / Q_numeric;
    }

    // Apply scaling
    if (scale_factor > 0.0) {
        for (int idx : electrodeIndices) {
            charges[idx] *= scale_factor;
        }
    }
}
