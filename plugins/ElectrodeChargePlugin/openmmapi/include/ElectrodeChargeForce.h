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
    struct ElectrodeRegion {
        std::vector<int> atomIndices;
        double voltage = 0.0;
    };

    ElectrodeChargeForce();

    /**
     * Set the cathode metadata (indices, potential).
     */
    void setCathode(const std::vector<int>& indices, double voltage);

    /**
     * Set the anode metadata (indices, potential).
     */
    void setAnode(const std::vector<int>& indices, double voltage);

    /**
     * Add a conductor region by specifying its atom indices.
     * This method can be called multiple times for multiple conductors.
     */
    void addConductor(const std::vector<int>& indices);

    /**
     * Accessors used by the ForceImpl/kernels.
     */
    const ElectrodeRegion& getCathode() const { return cathode; }
    const ElectrodeRegion& getAnode() const { return anode; }
    const std::vector<std::vector<int>>& getConductors() const { return conductors; }

    /**
     * Solver configuration.
     */
    void setNumIterations(int iterations) { numIterations = iterations; }
    int getNumIterations() const { return numIterations; }

    void setSmallThreshold(double value) { smallThreshold = value; }
    double getSmallThreshold() const { return smallThreshold; }

    void setCellGap(double gap) { lGap = gap; }
    double getCellGap() const { return lGap; }

    void setCellLength(double length) { lCell = length; }
    double getCellLength() const { return lCell; }

    /**
     * OpenMM book-keeping overrides.
     */
    OpenMM::ForceImpl* createImpl() const override;
    bool usesPeriodicBoundaryConditions() const override { return true; }

private:
    ElectrodeRegion cathode;
    ElectrodeRegion anode;
    std::vector<std::vector<int>> conductors;
    int numIterations = 4;
    double smallThreshold = 1.0e-6;
    double lGap = 1.0;
    double lCell = 1.0;
};

} // namespace ElectrodeChargePlugin

#endif // ELECTRODE_CHARGE_FORCE_H_
