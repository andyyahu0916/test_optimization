#ifndef OPENMM_ELECTRODECHARGEFORCE_H_
#define OPENMM_ELECTRODECHARGEFORCE_H_

#include "openmm/Force.h"
#include <vector>
#include <string>

namespace ElectrodeChargePlugin {

/**
 * ElectrodeChargeForce computes electrode charges using a pre-calculated inverse capacitance matrix.
 *
 * Algorithm (single-pass, no iterations):
 *   q_e = C_inv * (V - E_f)
 *
 * Where:
 *   q_e: electrode charges (output, size N)
 *   C_inv: inverse capacitance matrix (input, size N*N, row-major)
 *   V: target potentials (input, size N)
 *   E_f: electric field from electrolyte (computed from positions, size N)
 *
 * This Force does NOT calculate forces or energy. It only updates NonbondedForce charges.
 */
class ElectrodeChargeForce : public OpenMM::Force {
public:
    ElectrodeChargeForce();

    /**
     * Add electrode atoms with their target potentials.
     *
     * @param atomIndices  Global indices of electrode atoms (size N)
     * @param potentials   Target potentials in kJ/mol (size N)
     */
    void addElectrodeAtoms(const std::vector<int>& atomIndices, const std::vector<double>& potentials);

    /**
     * Add electrolyte atoms with their fixed charges.
     *
     * @param atomIndices  Global indices of electrolyte atoms (size M)
     * @param fixedCharges Fixed charges in elementary charge units (size M)
     */
    void addElectrolyteAtoms(const std::vector<int>& atomIndices, const std::vector<double>& fixedCharges);

    /**
     * Set the inverse capacitance matrix C_inv.
     *
     * @param numElectrodes Number of electrode atoms (N)
     * @param flattenedMatrix Flattened N*N matrix in row-major order
     */
    void setInverseCapacitanceMatrix(int numElectrodes, const std::vector<double>& flattenedMatrix);

    // Getters
    const std::vector<int>& getElectrodeAtomIndices() const { return electrodeAtomIndices; }
    const std::vector<double>& getTargetPotentials() const { return targetPotentials; }
    const std::vector<int>& getElectrolyteAtomIndices() const { return electrolyteAtomIndices; }
    const std::vector<double>& getFixedCharges() const { return fixedCharges; }
    const std::vector<double>& getInverseCapacitanceMatrix() const { return invCapMatrix; }
    int getNumElectrodeAtoms() const { return numElectrodeAtoms; }

    // OpenMM Force interface
    bool usesPeriodicBoundaryConditions() const override { return true; }

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    int numElectrodeAtoms;
    std::vector<int> electrodeAtomIndices;      // size N
    std::vector<double> targetPotentials;       // size N (kJ/mol)
    std::vector<int> electrolyteAtomIndices;    // size M
    std::vector<double> fixedCharges;           // size M (elementary charge)
    std::vector<double> invCapMatrix;           // size N*N (row-major)
};

} // namespace ElectrodeChargePlugin

#endif // OPENMM_ELECTRODECHARGEFORCE_H_
