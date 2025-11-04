#ifndef OPENMM_CONSTANTVFORCE_H_
#define OPENMM_CONSTANTVFORCE_H_

#include "openmm/Force.h"
#include <vector>

namespace ConstantVPlugin {

/**
 * ConstantVForce updates electrode charges using a pre-calculated inverse capacitance matrix.
 *
 * Algorithm (single-pass):
 *   q_e = C_inv * (V - E_f)
 *
 * This Force does NOT compute forces or energy. It only updates NonbondedForce charges.
 */
class ConstantVForce : public OpenMM::Force {
public:
    ConstantVForce();

    /**
     * Get the number of electrode atoms.
     */
    int getNumElectrodeAtoms() const {
        return electrodeAtomIndices.size();
    }

    /**
     * Get the number of electrolyte atoms.
     */
    int getNumElectrolyteAtoms() const {
        return electrolyteAtomIndices.size();
    }

    /**
     * Add electrode atoms with their target potentials.
     *
     * @param particle      global index of the electrode atom
     * @param potential     target potential in kJ/mol
     * @return the index of the electrode atom that was added
     */
    int addElectrodeAtom(int particle, double potential);

    /**
     * Get parameters for an electrode atom.
     */
    void getElectrodeAtomParameters(int index, int& particle, double& potential) const;

    /**
     * Set parameters for an electrode atom.
     */
    void setElectrodeAtomParameters(int index, int particle, double potential);

    /**
     * Add an electrolyte atom with its fixed charge.
     *
     * @param particle      global index of the electrolyte atom
     * @param charge        fixed charge in elementary charge units
     * @return the index of the electrolyte atom that was added
     */
    int addElectrolyteAtom(int particle, double charge);

    /**
     * Get parameters for an electrolyte atom.
     */
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;

    /**
     * Set the inverse capacitance matrix C_inv.
     *
     * @param flattenedMatrix Flattened N*N matrix in row-major order
     */
    void setInverseCapacitanceMatrix(const std::vector<double>& flattenedMatrix);

    /**
     * Get the inverse capacitance matrix.
     */
    const std::vector<double>& getInverseCapacitanceMatrix() const {
        return invCapMatrix;
    }

    bool usesPeriodicBoundaryConditions() const override {
        return true;
    }

protected:
    OpenMM::ForceImpl* createImpl() const override;

private:
    class ElectrodeAtomInfo;
    class ElectrolyteAtomInfo;
    std::vector<ElectrodeAtomInfo> electrodeAtoms;
    std::vector<int> electrodeAtomIndices;
    std::vector<double> targetPotentials;
    std::vector<ElectrolyteAtomInfo> electrolyteAtoms;
    std::vector<int> electrolyteAtomIndices;
    std::vector<double> fixedCharges;
    std::vector<double> invCapMatrix;
};

/**
 * Information about an electrode atom.
 */
class ConstantVForce::ElectrodeAtomInfo {
public:
    int particle;
    double potential;
    ElectrodeAtomInfo() : particle(-1), potential(0.0) {}
    ElectrodeAtomInfo(int particle, double potential) : particle(particle), potential(potential) {}
};

/**
 * Information about an electrolyte atom.
 */
class ConstantVForce::ElectrolyteAtomInfo {
public:
    int particle;
    double charge;
    ElectrolyteAtomInfo() : particle(-1), charge(0.0) {}
    ElectrolyteAtomInfo(int particle, double charge) : particle(particle), charge(charge) {}
};

} // namespace ConstantVPlugin

#endif // OPENMM_CONSTANTVFORCE_H_
