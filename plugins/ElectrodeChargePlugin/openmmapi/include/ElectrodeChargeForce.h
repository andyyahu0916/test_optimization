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
     * Accessors used by the ForceImpl/kernels.
     */
    const ElectrodeRegion& getCathode() const { return cathode; }
    const ElectrodeRegion& getAnode() const { return anode; }

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
     * Conductor metadata setters/getters (optional; enable conductor two-stage method).
     * All conductor arrays are parallel and refer to virtual conductor atoms.
     * - indices: atom indices of conductor virtual atoms (flattened across all conductors)
     * - normals: per-atom surface normals (flattened triplets nx,ny,nz; size = 3*indices.size())
     * - areas: per-atom surface area (same length as indices)
     * - contactIndices: per-conductor contact atom index (length = numConductors)
     * - contactNormals: per-conductor contact normal (flattened triplets; size = 3*numConductors)
     * - geometries: per-conductor geometry scalar (buckyball: r_contact^2; nanotube: r_contact*L/2)
     * - types: per-conductor type id (0=buckyball, 1=nanotube)
     */
    void setConductorData(const std::vector<int>& indices,
                          const std::vector<double>& normals,
                          const std::vector<double>& areas,
                          const std::vector<int>& contactIndices,
                          const std::vector<double>& contactNormals,
                          const std::vector<double>& geometries,
                          const std::vector<int>& types,
                          const std::vector<int>& atomCondIds,
                          const std::vector<int>& atomCountsPerConductor) {
        conductorIndices = indices;
        conductorNormals = normals;
        conductorAreas = areas;
        conductorContactIndices = contactIndices;
        conductorContactNormals = contactNormals;
        conductorGeometries = geometries;
        conductorTypes = types;
        conductorAtomCondIds = atomCondIds;
        conductorAtomCounts = atomCountsPerConductor;
    }

    const std::vector<int>& getConductorIndices() const { return conductorIndices; }
    const std::vector<double>& getConductorNormals() const { return conductorNormals; }
    const std::vector<double>& getConductorAreas() const { return conductorAreas; }
    const std::vector<int>& getConductorContactIndices() const { return conductorContactIndices; }
    const std::vector<double>& getConductorContactNormals() const { return conductorContactNormals; }
    const std::vector<double>& getConductorGeometries() const { return conductorGeometries; }
    const std::vector<int>& getConductorTypes() const { return conductorTypes; }
    const std::vector<int>& getConductorAtomCondIds() const { return conductorAtomCondIds; }
    const std::vector<int>& getConductorAtomCounts() const { return conductorAtomCounts; }

    /**
     * OpenMM book-keeping overrides.
     */
    OpenMM::ForceImpl* createImpl() const override;
    bool usesPeriodicBoundaryConditions() const override { return true; }

private:
    ElectrodeRegion cathode;
    ElectrodeRegion anode;
    int numIterations = 4;
    double smallThreshold = 1.0e-6;
    double lGap = 1.0;
    double lCell = 1.0;

    // Conductor metadata (optional)
    std::vector<int> conductorIndices;
    std::vector<double> conductorNormals;
    std::vector<double> conductorAreas;
    std::vector<int> conductorContactIndices;
    std::vector<double> conductorContactNormals;
    std::vector<double> conductorGeometries;
    std::vector<int> conductorTypes;
    std::vector<int> conductorAtomCondIds;       // length = numConductorAtoms, maps each atom to conductor id
    std::vector<int> conductorAtomCounts;        // length = numConductors, number of atoms per conductor
};

} // namespace ElectrodeChargePlugin

#endif // ELECTRODE_CHARGE_FORCE_H_
