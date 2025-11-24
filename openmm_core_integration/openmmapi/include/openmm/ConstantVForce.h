#ifndef OPENMM_CONSTANTVFORCE_H_
#define OPENMM_CONSTANTVFORCE_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * ConstantVForce: Native implementation of constant voltage boundary         *
 * conditions for electrochemical simulations.                                *
 *                                                                            *
 * This is NOT a plugin. It is a first-class citizen of OpenMM Core.         *
 *                                                                            *
 * Copyright (c) 2025 - Present                                               *
 * Authors: Based on Professor's MM_classes.py algorithm                      *
 * -------------------------------------------------------------------------- */

#include "openmm/Force.h"
#include "openmm/Vec3.h"
#include <vector>
#include <string>

namespace OpenMM {

/**
 * ConstantVForce implements self-consistent field (SCF) electrode charge
 * updates to enforce constant voltage boundary conditions.
 *
 * Algorithm (Professor's Method):
 * --------------------------------
 * for iter in range(Niterations):
 *     1. Get forces from OpenMM (includes all interactions)
 *     2. Compute Ez = F_z / q_old
 *     3. Update charge: q = 2/(4π) * area * (V/Lgap + Ez)
 *     4. Apply Green's Reciprocity correction
 *     5. Update OpenMM context
 *
 * Translated from: MM_classes.py::Poisson_solver_fixed_voltage
 *
 * Physical Rationale:
 * -------------------
 * - Electrodes must maintain constant voltage (Maxwell boundary condition)
 * - Electrode charges depend on nearby electrolyte positions (Ez_external)
 * - Self-consistent solution via iterative refinement (SCF)
 * - Green's Reciprocity ensures global charge neutrality
 *
 * Supported Electrode Geometries:
 * --------------------------------
 * 1. Flat Electrodes: Planar surfaces (cathode/anode)
 * 2. Buckyball Conductors: Spherical surfaces (C60, nanoparticles)
 * 3. Nanotube Conductors: Cylindrical surfaces (CNT, nanowires)
 *
 * Design Philosophy (Native Integration):
 * ----------------------------------------
 * - Embedded directly in OpenMM Core (NOT a loadable plugin)
 * - Zero Force Group overhead (direct kernel invocation)
 * - Reuses OpenMM's platform infrastructure (CUDA/Reference/OpenCL)
 * - Full API compatibility with ForceImpl/Kernel architecture
 *
 * Example Usage:
 * --------------
 * \code
 * ConstantVForce* force = new ConstantVForce();
 * force->setVoltage(2.0);  // 2.0 V
 * force->setLgap(3.5);     // 3.5 nm
 * force->setLcell(5.0);    // 5.0 nm
 * force->setNumIterations(4);
 *
 * // Add flat electrodes
 * for (int i : cathode_atoms)
 *     force->addCathodeAtom(i, area_per_atom);
 * for (int i : anode_atoms)
 *     force->addAnodeAtom(i, area_per_atom);
 *
 * // Optional: Add conductor geometries
 * force->addBuckyballConductor(virtual_atoms, real_atoms, "cathode", 1.0);
 * force->addNanotubeConductor(virtual_atoms, real_atoms, "anode", -1.0, axis);
 *
 * system.addForce(force);
 * \endcode
 */
class OPENMM_EXPORT ConstantVForce : public Force {
public:
    /**
     * Create a ConstantVForce.
     */
    ConstantVForce();

    /**
     * Destructor.
     */
    ~ConstantVForce();

    // ═══════════════════════════════════════════════════════════════════════
    // Flat Electrode Management
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Add a cathode (negative) electrode atom.
     *
     * @param particle   global atom index
     * @param area       surface area for this atom (nm²)
     * @return index in cathode atom list
     */
    int addCathodeAtom(int particle, double area);

    /**
     * Add an anode (positive) electrode atom.
     *
     * @param particle   global atom index
     * @param area       surface area for this atom (nm²)
     * @return index in anode atom list
     */
    int addAnodeAtom(int particle, double area);

    /**
     * Get number of cathode atoms.
     */
    int getNumCathodeAtoms() const {
        return cathodeAtoms.size();
    }

    /**
     * Get number of anode atoms.
     */
    int getNumAnodeAtoms() const {
        return anodeAtoms.size();
    }

    /**
     * Get cathode atom parameters.
     *
     * @param index      index in cathode list
     * @param particle   [out] atom index
     * @param area       [out] surface area (nm²)
     */
    void getCathodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * Set cathode atom parameters.
     */
    void setCathodeAtomParameters(int index, int particle, double area);

    /**
     * Get anode atom parameters.
     */
    void getAnodeAtomParameters(int index, int& particle, double& area) const;

    /**
     * Set anode atom parameters.
     */
    void setAnodeAtomParameters(int index, int particle, double area);

    // ═══════════════════════════════════════════════════════════════════════
    // Electrolyte Atom Management
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Add an electrolyte atom (fixed charge for Green's Reciprocity).
     *
     * @param particle   global atom index
     * @param charge     fixed charge (elementary charge units)
     * @return index in electrolyte atom list
     */
    int addElectrolyteAtom(int particle, double charge);

    /**
     * Get number of electrolyte atoms.
     */
    int getNumElectrolyteAtoms() const {
        return electrolyteAtoms.size();
    }

    /**
     * Get electrolyte atom parameters.
     */
    void getElectrolyteAtomParameters(int index, int& particle, double& charge) const;

    /**
     * Set electrolyte atom parameters.
     */
    void setElectrolyteAtomParameters(int index, int particle, double charge);

    // ═══════════════════════════════════════════════════════════════════════
    // Buckyball Conductor Management
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Add a Buckyball (spherical) conductor.
     *
     * Physics:
     * --------
     * - Virtual layer: Image charges for Maxwell boundary conditions
     * - Real layer: VDW/steric repulsion (prevents ion penetration)
     * - Geometry: Sphere center, radius, surface area per atom
     * - Normal vectors: Radial direction (atom_pos - center) / r
     * - Charge update: q = factor * area * (V/r + E_n_external)
     *
     * Corresponds to: Buckyball_Virtual class (Fixed_Voltage_routines.py:391-473)
     *
     * @param virtualAtoms    virtual layer atom indices (for electrostatics)
     * @param realAtoms       real layer atom indices (for VDW/steric)
     * @param electrodeType   "cathode" or "anode"
     * @param voltage         applied voltage (V)
     * @return index in Buckyball conductor list
     */
    int addBuckyballConductor(const std::vector<int>& virtualAtoms,
                              const std::vector<int>& realAtoms,
                              const std::string& electrodeType,
                              double voltage);

    /**
     * Get number of Buckyball conductors.
     */
    int getNumBuckyballConductors() const {
        return buckyballConductors.size();
    }

    /**
     * Get Buckyball conductor parameters.
     */
    void getBuckyballConductorParameters(int index,
                                         std::vector<int>& virtualAtoms,
                                         std::vector<int>& realAtoms,
                                         std::string& electrodeType,
                                         double& voltage) const;

    /**
     * Set Buckyball conductor parameters.
     */
    void setBuckyballConductorParameters(int index,
                                         const std::vector<int>& virtualAtoms,
                                         const std::vector<int>& realAtoms,
                                         const std::string& electrodeType,
                                         double voltage);

    // ═══════════════════════════════════════════════════════════════════════
    // Nanotube Conductor Management
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Add a Nanotube (cylindrical) conductor.
     *
     * Physics:
     * --------
     * - Cylindrical geometry: axis direction, radius, length
     * - Normal vectors: Radial direction (perpendicular to axis)
     * - Area per atom: 2πrL / N (cylindrical surface area)
     * - Charge update: q = factor * area * (V/r + E_radial)
     *
     * Corresponds to: Nanotube_Virtual class (Fixed_Voltage_routines.py:482-589)
     *
     * @param virtualAtoms    virtual layer atom indices
     * @param realAtoms       real layer atom indices
     * @param electrodeType   "cathode" or "anode"
     * @param voltage         applied voltage (V)
     * @param axis            nanotube axis direction [ax, ay, az] (will be normalized)
     * @return index in Nanotube conductor list
     */
    int addNanotubeConductor(const std::vector<int>& virtualAtoms,
                             const std::vector<int>& realAtoms,
                             const std::string& electrodeType,
                             double voltage,
                             const std::vector<double>& axis);

    /**
     * Get number of Nanotube conductors.
     */
    int getNumNanotubeConductors() const {
        return nanotubeConductors.size();
    }

    /**
     * Get Nanotube conductor parameters.
     */
    void getNanotubeConductorParameters(int index,
                                        std::vector<int>& virtualAtoms,
                                        std::vector<int>& realAtoms,
                                        std::string& electrodeType,
                                        double& voltage,
                                        std::vector<double>& axis) const;

    /**
     * Set Nanotube conductor parameters.
     */
    void setNanotubeConductorParameters(int index,
                                        const std::vector<int>& virtualAtoms,
                                        const std::vector<int>& realAtoms,
                                        const std::string& electrodeType,
                                        double voltage,
                                        const std::vector<double>& axis);

    // ═══════════════════════════════════════════════════════════════════════
    // System Geometry Parameters
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Set the applied voltage (V).
     *
     * Note: Internally converted to kJ/mol by multiplying with 96.487
     * (conversion_eV_to_kJmol).
     */
    void setVoltage(double voltage);

    /**
     * Get the applied voltage (V).
     */
    double getVoltage() const {
        return voltageVolts;
    }

    /**
     * Set the vacuum gap distance (nm).
     *
     * Corresponds to: MMsys.Lgap
     */
    void setLgap(double gap);

    /**
     * Get the vacuum gap distance (nm).
     */
    double getLgap() const {
        return Lgap;
    }

    /**
     * Set the electrode spacing (nm).
     *
     * Corresponds to: MMsys.Lcell
     */
    void setLcell(double cell);

    /**
     * Get the electrode spacing (nm).
     */
    double getLcell() const {
        return Lcell;
    }

    /**
     * Set the total electrode area (nm²).
     *
     * Corresponds to: self.Cathode.sheet_area
     */
    void setTotalArea(double area);

    /**
     * Get the total electrode area (nm²).
     */
    double getTotalArea() const {
        return totalArea;
    }

    /**
     * Set the cathode Z position (nm).
     */
    void setZCathode(double z);

    /**
     * Get the cathode Z position (nm).
     */
    double getZCathode() const {
        return z_cathode;
    }

    /**
     * Set the anode Z position (nm).
     */
    void setZAnode(double z);

    /**
     * Get the anode Z position (nm).
     */
    double getZAnode() const {
        return z_anode;
    }

    // ═══════════════════════════════════════════════════════════════════════
    // SCF Parameters
    // ═══════════════════════════════════════════════════════════════════════

    /**
     * Set the number of SCF iterations per force evaluation.
     *
     * Default: 4 (Professor's recommendation)
     */
    void setNumIterations(int n);

    /**
     * Get the number of SCF iterations.
     */
    int getNumIterations() const {
        return nIterations;
    }

    /**
     * Get the names of all Force objects that this Force depends on.
     *
     * This Force modifies NonbondedForce charges, so it should be evaluated
     * AFTER NonbondedForce computes forces.
     */
    bool usesPeriodicBoundaryConditions() const override {
        return true;
    }

protected:
    ForceImpl* createImpl() const override;

private:
    // Flat electrode data
    class CathodeAtomInfo;
    class AnodeAtomInfo;
    class ElectrolyteAtomInfo;

    std::vector<CathodeAtomInfo> cathodeAtoms;
    std::vector<AnodeAtomInfo> anodeAtoms;
    std::vector<ElectrolyteAtomInfo> electrolyteAtoms;

    // Conductor data
    class BuckyballConductorInfo;
    class NanotubeConductorInfo;

    std::vector<BuckyballConductorInfo> buckyballConductors;
    std::vector<NanotubeConductorInfo> nanotubeConductors;

    // System parameters
    double voltageVolts;    // Input voltage (V)
    double voltageKjMol;    // Internal use (kJ/mol)
    double Lgap;            // Vacuum gap (nm)
    double Lcell;           // Electrode spacing (nm)
    double totalArea;       // Electrode area (nm²)
    double z_cathode;       // Cathode Z position (nm)
    double z_anode;         // Anode Z position (nm)

    // SCF parameters
    int nIterations;        // Number of iterations (default 4)
};

/**
 * Internal data structures (hidden from public API)
 */

class ConstantVForce::CathodeAtomInfo {
public:
    int particle;
    double area;
    CathodeAtomInfo() : particle(-1), area(0.0) {}
    CathodeAtomInfo(int p, double a) : particle(p), area(a) {}
};

class ConstantVForce::AnodeAtomInfo {
public:
    int particle;
    double area;
    AnodeAtomInfo() : particle(-1), area(0.0) {}
    AnodeAtomInfo(int p, double a) : particle(p), area(a) {}
};

class ConstantVForce::ElectrolyteAtomInfo {
public:
    int particle;
    double charge;
    ElectrolyteAtomInfo() : particle(-1), charge(0.0) {}
    ElectrolyteAtomInfo(int p, double c) : particle(p), charge(c) {}
};

class ConstantVForce::BuckyballConductorInfo {
public:
    std::vector<int> virtualAtomIndices;
    std::vector<int> realAtomIndices;
    std::string electrodeType;
    double voltageVolts;
    double voltageKjMol;

    // Geometry parameters (computed during initialization)
    Vec3 center;            // Sphere center (nm)
    double radius;          // Sphere radius (nm)
    double area_atom;       // Area per atom (nm²)

    // Surface normal vectors (radial direction)
    std::vector<Vec3> normalVectors;

    // Contact neighbor information
    int contactAtomIndex;
    double dr_center_contact;
    bool closeToElectrode;
    double closeThreshold;

    BuckyballConductorInfo() :
        electrodeType(""),
        voltageVolts(0.0),
        voltageKjMol(0.0),
        radius(0.0),
        area_atom(0.0),
        contactAtomIndex(-1),
        dr_center_contact(0.0),
        closeToElectrode(true),
        closeThreshold(1.5) {}

    BuckyballConductorInfo(const std::vector<int>& vAtoms,
                           const std::vector<int>& rAtoms,
                           const std::string& type,
                           double voltage) :
        virtualAtomIndices(vAtoms),
        realAtomIndices(rAtoms),
        electrodeType(type),
        voltageVolts(voltage),
        voltageKjMol(voltage * 96.487),
        radius(0.0),
        area_atom(0.0),
        contactAtomIndex(-1),
        dr_center_contact(0.0),
        closeToElectrode(true),
        closeThreshold(1.5) {}
};

class ConstantVForce::NanotubeConductorInfo {
public:
    std::vector<int> virtualAtomIndices;
    std::vector<int> realAtomIndices;
    std::string electrodeType;
    double voltageVolts;
    double voltageKjMol;

    // Geometry parameters (computed during initialization)
    Vec3 axis;              // Nanotube axis (normalized)
    Vec3 center;            // Center position (nm)
    double radius;          // Radius (nm)
    double length;          // Length (nm)
    double area_atom;       // Area per atom (nm²)

    // Radial normal vectors (perpendicular to axis)
    std::vector<Vec3> normalVectors;

    // Contact neighbor information
    int contactAtomIndex;
    double dr_center_contact;
    bool closeToElectrode;
    double closeThreshold;

    NanotubeConductorInfo() :
        electrodeType(""),
        voltageVolts(0.0),
        voltageKjMol(0.0),
        radius(0.0),
        length(0.0),
        area_atom(0.0),
        contactAtomIndex(-1),
        dr_center_contact(0.0),
        closeToElectrode(true),
        closeThreshold(1.5) {}

    NanotubeConductorInfo(const std::vector<int>& vAtoms,
                          const std::vector<int>& rAtoms,
                          const std::string& type,
                          double voltage,
                          const std::vector<double>& axisVec) :
        virtualAtomIndices(vAtoms),
        realAtomIndices(rAtoms),
        electrodeType(type),
        voltageVolts(voltage),
        voltageKjMol(voltage * 96.487),
        radius(0.0),
        length(0.0),
        area_atom(0.0),
        contactAtomIndex(-1),
        dr_center_contact(0.0),
        closeToElectrode(true),
        closeThreshold(1.5)
    {
        if (axisVec.size() == 3) {
            axis = Vec3(axisVec[0], axisVec[1], axisVec[2]);
            // Normalize
            double norm = std::sqrt(axis[0]*axis[0] + axis[1]*axis[1] + axis[2]*axis[2]);
            if (norm > 1e-10) {
                axis *= (1.0 / norm);
            } else {
                axis = Vec3(1, 0, 0);  // Default to x-axis
            }
        } else {
            axis = Vec3(1, 0, 0);  // Default to x-axis
        }
    }
};

} // namespace OpenMM

#endif // OPENMM_CONSTANTVFORCE_H_
