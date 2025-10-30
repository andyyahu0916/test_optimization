#ifndef ELECTRODE_CHARGE_FORCE_H_
#define ELECTRODE_CHARGE_FORCE_H_

#include "openmm/Force.h"
#include <string>
#include <vector>

namespace ElectrodeChargePlugin {

/**
 * ElectrodeChargeForce stores all metadata required to perform the constant-voltage
 * Poisson solve.  It delegates the heavy lifting to platform specific kernels.
 */
class ElectrodeChargeForce : public OpenMM::Force {
public:
    struct Parameters {
        // Electrode data
        std::vector<int> cathodeIndices;
        std::vector<int> anodeIndices;
        double cathodeVoltage = 0.0;
        double anodeVoltage = 0.0;

        // Solver settings
        int numIterations = 4;
        double smallThreshold = 1.0e-6;

        // Cell geometry
        double lGap = 1.0;
        double lCell = 1.0;
        double sheetArea = 0.0; // Added for convenience
        double cathodeZ = 0.0;
        double anodeZ = 0.0;

        // Conductor metadata (optional)
        std::vector<int> conductorIndices;
        std::vector<double> conductorNormals;
        std::vector<double> conductorAreas;
        std::vector<int> conductorContactIndices;
        std::vector<double> conductorContactNormals;
        std::vector<double> conductorGeometries;
        std::vector<int> conductorAtomCondIds;
        std::vector<int> conductorAtomCounts;
    };

    ElectrodeChargeForce();

    void setCathode(const std::vector<int>& indices, double voltage);
    void setAnode(const std::vector<int>& indices, double voltage);
    void setNumIterations(int iterations);
    void setSmallThreshold(double value);
    void setCellGap(double gap);
    void setCellLength(double length);
    void setSheetArea(double area);
    void setCathodeZ(double z);
    void setAnodeZ(double z);

    void setConductorData(const std::vector<int>& indices,
                          const std::vector<double>& normals,
                          const std::vector<double>& areas,
                          const std::vector<int>& contactIndices,
                          const std::vector<double>& contactNormals,
                          const std::vector<double>& geometries,
                          const std::vector<int>& atomCondIds,
                          const std::vector<int>& atomCountsPerConductor);

    const Parameters& getParameters() const { return params; }

    OpenMM::ForceImpl* createImpl() const override;
    bool usesPeriodicBoundaryConditions() const override { return true; }

private:
    Parameters params;

};

} // namespace ElectrodeChargePlugin

#endif // ELECTRODE_CHARGE_FORCE_H_
