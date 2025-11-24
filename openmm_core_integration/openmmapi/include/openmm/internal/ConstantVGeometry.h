#ifndef OPENMM_CONSTANTVGEOMETRY_H_
#define OPENMM_CONSTANTVGEOMETRY_H_

/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * Geometry calculations for Buckyball and Nanotube conductors                *
 *                                                                            *
 * Translated from: Fixed_Voltage_routines.py                                 *
 * - Buckyball_Virtual class (Lines 391-473)                                 *
 * - Nanotube_Virtual class (Lines 482-589)                                  *
 * -------------------------------------------------------------------------- */

#include "openmm/Vec3.h"
#include <vector>
#include <cmath>

namespace OpenMM {

/**
 * Compute the center of a sphere from atom positions.
 *
 * Algorithm: Average of all atom positions
 * Corresponds to: Fixed_Voltage_routines.py Lines 428-436
 */
inline Vec3 computeSphereCenter(const std::vector<Vec3>& positions) {
    Vec3 center(0, 0, 0);
    for (const Vec3& pos : positions) {
        center += pos;
    }
    center *= (1.0 / positions.size());
    return center;
}

/**
 * Compute the radius of a sphere from atom positions and center.
 *
 * Algorithm: Average distance from center to atoms
 * Corresponds to: Fixed_Voltage_routines.py Lines 440-446
 */
inline double computeSphereRadius(const std::vector<Vec3>& positions, const Vec3& center) {
    double radius = 0.0;
    for (const Vec3& pos : positions) {
        Vec3 diff = pos - center;
        radius += std::sqrt(diff.dot(diff));
    }
    radius /= positions.size();
    return radius;
}

/**
 * Compute surface normal vectors for a sphere.
 *
 * Algorithm: Normal = (atom_pos - center) / |atom_pos - center|
 * Corresponds to: Fixed_Voltage_routines.py Lines 451-456
 */
inline std::vector<Vec3> computeSphereNormals(const std::vector<Vec3>& positions, const Vec3& center) {
    std::vector<Vec3> normals;
    normals.reserve(positions.size());

    for (const Vec3& pos : positions) {
        Vec3 diff = pos - center;
        double r = std::sqrt(diff.dot(diff));
        if (r > 1e-10) {
            normals.push_back(diff * (1.0 / r));
        } else {
            // Degenerate case: atom at center (should never happen)
            normals.push_back(Vec3(0, 0, 1));  // Arbitrary direction
        }
    }

    return normals;
}

/**
 * Compute the center of a nanotube from atom positions.
 *
 * Algorithm: Average of all atom positions
 * Corresponds to: Fixed_Voltage_routines.py Lines 521-529
 */
inline Vec3 computeNanotubeCenter(const std::vector<Vec3>& positions) {
    return computeSphereCenter(positions);  // Same algorithm
}

/**
 * Project a vector onto the plane perpendicular to an axis.
 *
 * Algorithm: v_perp = v - axis * dot(v, axis)
 * Corresponds to: Fixed_Voltage_routines.py::project_orthogonal_to_axis
 */
inline Vec3 projectOrthogonalToAxis(const Vec3& vec, const Vec3& axis) {
    double dotProduct = vec.dot(axis);
    return vec - axis * dotProduct;
}

/**
 * Compute the radius of a nanotube.
 *
 * Algorithm: Average radial distance from axis
 * Corresponds to: Fixed_Voltage_routines.py Lines 541-556
 *
 * @param positions   atom positions
 * @param center      nanotube center
 * @param axis        nanotube axis (normalized)
 */
inline double computeNanotubeRadius(const std::vector<Vec3>& positions,
                                   const Vec3& center,
                                   const Vec3& axis) {
    double radius = 0.0;

    for (const Vec3& pos : positions) {
        // Vector from center to atom
        Vec3 diff = pos - center;

        // Project onto plane perpendicular to axis
        Vec3 radial = projectOrthogonalToAxis(diff, axis);

        // Accumulate radial distance
        radius += std::sqrt(radial.dot(radial));
    }

    radius /= positions.size();
    return radius;
}

/**
 * Compute radial normal vectors for a nanotube.
 *
 * Algorithm: Normal = (atom_pos - center - axis_component) normalized
 * Corresponds to: Fixed_Voltage_routines.py Line 558
 *
 * @param positions   atom positions
 * @param center      nanotube center
 * @param axis        nanotube axis (normalized)
 */
inline std::vector<Vec3> computeNanotubeNormals(const std::vector<Vec3>& positions,
                                                const Vec3& center,
                                                const Vec3& axis) {
    std::vector<Vec3> normals;
    normals.reserve(positions.size());

    for (const Vec3& pos : positions) {
        // Vector from center to atom
        Vec3 diff = pos - center;

        // Project onto plane perpendicular to axis (radial direction)
        Vec3 radial = projectOrthogonalToAxis(diff, axis);

        // Normalize
        double r = std::sqrt(radial.dot(radial));
        if (r > 1e-10) {
            normals.push_back(radial * (1.0 / r));
        } else {
            // Degenerate case: atom on axis (should never happen for nanotube)
            // Create arbitrary perpendicular vector
            Vec3 perp;
            if (std::abs(axis[0]) < 0.9) {
                perp = Vec3(1, 0, 0) - axis * axis[0];
            } else {
                perp = Vec3(0, 1, 0) - axis * axis[1];
            }
            double norm = std::sqrt(perp.dot(perp));
            normals.push_back(perp * (1.0 / norm));
        }
    }

    return normals;
}

/**
 * Compute nanotube length from box vectors.
 *
 * Algorithm: Length = norm(box_vector_a)
 * Corresponds to: Fixed_Voltage_routines.py Lines 532-536
 *
 * For periodic systems, the nanotube length is typically the box dimension
 * along the nanotube axis.
 */
inline double computeNanotubeLength(const Vec3& boxVectorA,
                                   const Vec3& boxVectorB,
                                   const Vec3& boxVectorC,
                                   const Vec3& axis) {
    // Find which box vector is most aligned with nanotube axis
    double dotA = std::abs(boxVectorA.dot(axis));
    double dotB = std::abs(boxVectorB.dot(axis));
    double dotC = std::abs(boxVectorC.dot(axis));

    Vec3 alignedVector;
    if (dotA >= dotB && dotA >= dotC) {
        alignedVector = boxVectorA;
    } else if (dotB >= dotA && dotB >= dotC) {
        alignedVector = boxVectorB;
    } else {
        alignedVector = boxVectorC;
    }

    return std::sqrt(alignedVector.dot(alignedVector));
}

/**
 * Find the closest electrode atom to a conductor center.
 *
 * Algorithm: Find atom with minimum distance to center
 * Corresponds to: Fixed_Voltage_routines.py::find_contact_neighbor_conductor (Line 459, 564)
 *
 * @param center              conductor center
 * @param electrodePositions  positions of electrode atoms
 * @param contactIndex        [out] index of closest electrode atom
 * @param contactDistance     [out] distance to closest electrode atom
 */
inline void findContactNeighbor(const Vec3& center,
                               const std::vector<Vec3>& electrodePositions,
                               int& contactIndex,
                               double& contactDistance) {
    contactIndex = -1;
    contactDistance = std::numeric_limits<double>::max();

    for (int i = 0; i < (int)electrodePositions.size(); i++) {
        Vec3 diff = electrodePositions[i] - center;
        double dist = std::sqrt(diff.dot(diff));

        if (dist < contactDistance) {
            contactDistance = dist;
            contactIndex = i;
        }
    }
}

/**
 * Compute area per atom for a spherical surface.
 *
 * Algorithm: area_per_atom = 4 * π * r² / N
 * Corresponds to: Fixed_Voltage_routines.py Line 447
 */
inline double computeSphereAreaPerAtom(double radius, int numAtoms) {
    const double FOUR_PI = 4.0 * M_PI;
    return FOUR_PI * radius * radius / numAtoms;
}

/**
 * Compute area per atom for a cylindrical surface.
 *
 * Algorithm: area_per_atom = 2 * π * r * L / N
 * Corresponds to: Fixed_Voltage_routines.py Line 561
 */
inline double computeCylinderAreaPerAtom(double radius, double length, int numAtoms) {
    const double TWO_PI = 2.0 * M_PI;
    return TWO_PI * radius * length / numAtoms;
}

} // namespace OpenMM

#endif // OPENMM_CONSTANTVGEOMETRY_H_
