#ifndef REFERENCE_CONSTANTV_DRUDE_LANGEVIN_DYNAMICS_H_
#define REFERENCE_CONSTANTV_DRUDE_LANGEVIN_DYNAMICS_H_

/* -------------------------------------------------------------------------- *
 *                          OpenMM - Native ConstantV                        *
 * -------------------------------------------------------------------------- *
 * Reference (CPU) ConstantV dynamics helper for the Drude integrator        *
 * -------------------------------------------------------------------------- */

#include "openmm/reference/ReferenceDrudeLangevinDynamics.h"
#include "openmm/Vec3.h"
#include <vector>

namespace OpenMM {

/**
 * ReferenceConstantVDrudeLangevinDynamics stores the metadata required to
 * perform the professor's ConstantV SCF loop on the Reference platform while
 * reusing the existing ReferenceDrudeLangevinDynamics implementation for the
 * actual time integration step.
 */
class ReferenceConstantVDrudeLangevinDynamics : public ReferenceDrudeLangevinDynamics {
public:
    ReferenceConstantVDrudeLangevinDynamics(int numParticles,
                                            double temperature,
                                            double friction,
                                            double drudeTemperature,
                                            double drudeFriction,
                                            double stepSize);

    ~ReferenceConstantVDrudeLangevinDynamics() override = default;

    // Electrode/electrolyte registration -------------------------------------------------
    void addCathodeAtom(int particle, double area);
    void addAnodeAtom(int particle, double area);
    void addElectrolyteAtom(int particle, double charge);

    int getNumCathodeAtoms() const { return static_cast<int>(cathodeIndices.size()); }
    int getNumAnodeAtoms() const { return static_cast<int>(anodeIndices.size()); }
    int getNumElectrolyteAtoms() const { return static_cast<int>(electrolyteIndices.size()); }

    // Geometry / control parameter setters ----------------------------------------------
    void setVoltage(double v) { voltage = v; }
    void setLgap(double gap) { Lgap = gap; }
    void setLcell(double cell) { Lcell = cell; }
    void setTotalArea(double area) { totalArea = area; }
    void setZCathode(double z) { z_cathode = z; }
    void setZAnode(double z) { z_anode = z; }
    void setNumSCFIterations(int n) { scfIterations = n; }

    double getVoltage() const { return voltage; }
    double getLgap() const { return Lgap; }
    double getLcell() const { return Lcell; }
    double getTotalArea() const { return totalArea; }
    double getZCathode() const { return z_cathode; }
    double getZAnode() const { return z_anode; }
    int getNumSCFIterations() const { return scfIterations; }

    // SCF update ------------------------------------------------------------------------
    void updateElectrodeCharges(std::vector<Vec3>& positions,
                                std::vector<Vec3>& forces,
                                std::vector<double>& charges);

private:
    double computeAnalyticCharge(const std::vector<int>& electrodeIndices,
                                 const std::vector<Vec3>& positions,
                                 const std::vector<double>& charges,
                                 double sign,
                                 double z_opposite);

    void updateFlatElectrodeCharges(const std::vector<int>& electrodeIndices,
                                    const std::vector<double>& areas,
                                    const std::vector<Vec3>& forces,
                                    std::vector<double>& charges,
                                    double sign);

    void scaleCharges(const std::vector<int>& electrodeIndices,
                      std::vector<double>& charges,
                      double Q_analytic);

    // Stored metadata -------------------------------------------------------------------
    double voltage;
    double Lgap;
    double Lcell;
    double totalArea;
    double z_cathode;
    double z_anode;
    int scfIterations;

    std::vector<int> cathodeIndices;
    std::vector<double> cathodeAreas;
    std::vector<int> anodeIndices;
    std::vector<double> anodeAreas;
    std::vector<int> electrolyteIndices;
    std::vector<double> electrolyteCharges;
};

} // namespace OpenMM

#endif // REFERENCE_CONSTANTV_DRUDE_LANGEVIN_DYNAMICS_H_
