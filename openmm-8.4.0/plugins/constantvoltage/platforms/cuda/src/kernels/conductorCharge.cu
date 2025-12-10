/* -------------------------------------------------------------------------- *
 * Conductor Charge Update Kernels (Buckyball / Nanotube)
 * -------------------------------------------------------------------------- *
 * These kernels implement the Numerical_charge_Conductor algorithm from:
 * - MM_classes.py: Numerical_charge_Conductor()
 * - Fixed_Voltage_routines.py: Buckyball_Virtual, Nanotube_Virtual
 *
 * Algorithm:
 * 1. Image charges: Project E-field to surface normal, solve for image charge
 * 2. Charge transfer: Distribute charge to satisfy boundary conditions
 *
 * Reference formulas:
 * - Image charge:   q_i = (2 / 4π) * area_atom * En_external * K
 * - Buckyball dQ:   dQ = -dE * dr_center_contact²
 * - Nanotube dQ:    dQ = -dE * dr_center_contact * length / 2
 * -------------------------------------------------------------------------- */

// Conversion constants
#define CONVERSION_KJMOL_NM_AU 0.00719475f
#define FOUR_PI 12.566370614359172f
#define SMALL_THRESHOLD 1e-6f

// Conductor types
#define CONDUCTOR_BUCKYBALL 0
#define CONDUCTOR_NANOTUBE  1


/**
 * Compute image charges on conductor surface atoms.
 * 
 * Step 1 of Numerical_charge_Conductor:
 * For each atom, project electric field onto surface normal,
 * then solve for image charge that cancels normal field inside.
 *
 * @param numAtoms            Number of atoms in this conductor
 * @param conductorIndices    Particle indices in the conductor
 * @param conductorNormals    Surface normal (nx, ny, nz) for each atom
 * @param areaPerAtom         Area associated with each conductor atom
 * @param posq                Particle positions and charges
 * @param forceBuffer         Force buffer (fixed-point, needs scaling)
 * @param paddedNumAtoms      Padded atom count for force buffer indexing
 * @param smallThreshold      Minimum charge threshold
 */
extern "C" __global__ void computeConductorImageCharges(
    int numAtoms,
    const int* __restrict__ conductorIndices,
    const float4* __restrict__ conductorNormals,  // (nx, ny, nz, unused)
    float areaPerAtom,
    float4* __restrict__ posq,
    const long long* __restrict__ forceBuffer,
    int paddedNumAtoms,
    float smallThreshold
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;
    
    int particleIdx = conductorIndices[idx];
    
    // Get current charge
    float4 pos = posq[particleIdx];
    float q_current = pos.w;
    
    // Skip if charge too small (can't compute E from F)
    if (fabsf(q_current) < 0.9f * smallThreshold) {
        // Set to small positive value
        posq[particleIdx].w = smallThreshold;
        return;
    }
    
    // Read forces (fixed-point to float)
    const float forceScale = 1.0f / 0x100000000;
    float Fx = forceBuffer[particleIdx] * forceScale;
    float Fy = forceBuffer[particleIdx + paddedNumAtoms] * forceScale;
    float Fz = forceBuffer[particleIdx + 2 * paddedNumAtoms] * forceScale;
    
    // E = F / q
    float Ex = Fx / q_current;
    float Ey = Fy / q_current;
    float Ez = Fz / q_current;
    
    // Project onto surface normal
    float4 normal = conductorNormals[idx];
    float En = Ex * normal.x + Ey * normal.y + Ez * normal.z;
    
    // Image charge formula:
    // q_i = (2 / 4π) * area * En * conversion
    float q_new = (2.0f / FOUR_PI) * areaPerAtom * En * CONVERSION_KJMOL_NM_AU;
    
    // Update charge in posq
    posq[particleIdx].w = q_new;
}


/**
 * Compute charge transfer for conductor boundary condition.
 *
 * Step 2 of Numerical_charge_Conductor:
 * Based on contact with electrode or another conductor,
 * compute the charge transfer to satisfy equipotential.
 *
 * @param numAtoms            Number of atoms in this conductor
 * @param conductorIndices    Particle indices in the conductor
 * @param contactAtomIdx      Index of the contact atom (for field reading)
 * @param contactNormal       Normal vector at contact atom
 * @param conductorType       0 = Buckyball, 1 = Nanotube
 * @param drCenterContact     Distance from conductor center to contact point
 * @param conductorLength     Length (for Nanotube only)
 * @param isCloseToElectrode  True if contact is with primary electrode
 * @param voltage             Applied voltage (kJ/mol/e)
 * @param Lgap                Electrode gap (nm)
 * @param posq                Particle positions and charges
 * @param forceBuffer         Force buffer
 * @param paddedNumAtoms      Padded atom count
 */
extern "C" __global__ void computeConductorChargeTransfer(
    int numAtoms,
    const int* __restrict__ conductorIndices,
    int contactAtomIdx,
    float3 contactNormal,
    int conductorType,
    float drCenterContact,
    float conductorLength,
    int isCloseToElectrode,
    float voltage,
    float Lgap,
    float4* __restrict__ posq,
    const long long* __restrict__ forceBuffer,
    int paddedNumAtoms
) {
    // Only thread 0 computes the charge transfer, then broadcasts
    __shared__ float dq_per_atom;
    
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Read force on contact atom
        const float forceScale = 1.0f / 0x100000000;
        float Fx = forceBuffer[contactAtomIdx] * forceScale;
        float Fy = forceBuffer[contactAtomIdx + paddedNumAtoms] * forceScale;
        float Fz = forceBuffer[contactAtomIdx + 2 * paddedNumAtoms] * forceScale;
        
        float q_contact = posq[contactAtomIdx].w;
        float En_external = 0.0f;
        
        if (fabsf(q_contact) > SMALL_THRESHOLD) {
            float Ex = Fx / q_contact;
            float Ey = Fy / q_contact;
            float Ez = Fz / q_contact;
            En_external = Ex * contactNormal.x + Ey * contactNormal.y + Ez * contactNormal.z;
        }
        
        // Compute dE based on contact type
        float dE_conductor;
        if (isCloseToElectrode) {
            // Contact with primary electrode
            // dE = -(En + V/Lgap/2) * K
            dE_conductor = -(En_external + voltage / Lgap / 2.0f) * CONVERSION_KJMOL_NM_AU;
        } else {
            // Contact with another conductor
            // dE = -En * K
            dE_conductor = -En_external * CONVERSION_KJMOL_NM_AU;
        }
        
        // Geometry-dependent charge transfer
        float dQ_conductor;
        float sign = -1.0f;  // positive z displacement from cathode → negative field for positive charge
        
        if (conductorType == CONDUCTOR_BUCKYBALL) {
            // Spherical: Q = E * A / 4π,  A = 4π * r²  →  Q = E * r²
            dQ_conductor = sign * dE_conductor * drCenterContact * drCenterContact;
        } else {  // CONDUCTOR_NANOTUBE
            // Cylindrical: Q = E * A / 4π,  A = 2π * r * L  →  Q = E * r * L / 2
            dQ_conductor = sign * dE_conductor * drCenterContact * conductorLength / 2.0f;
        }
        
        // Per-atom charge
        dq_per_atom = dQ_conductor / (float)numAtoms;
    }
    
    __syncthreads();
    
    // All threads add the charge to their atoms
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;
    
    int particleIdx = conductorIndices[idx];
    posq[particleIdx].w += dq_per_atom;
}


/**
 * Scale all electrode and conductor charges to analytic normalization.
 *
 * When conductors are present:
 * - Anode scaled normally (independent)
 * - Q_analytic = -Anode.Q_analytic
 * - Q_numeric_total = Cathode_total + sum(Conductor_total)
 * - scale_factor = Q_analytic / Q_numeric_total
 * - Apply scale_factor to cathode + all conductors
 *
 * @param numCathodeAtoms     Number of cathode atoms
 * @param cathodeIndices      Cathode atom indices
 * @param numConductorAtoms   Total conductor atoms (all conductors combined)
 * @param conductorIndices    Conductor atom indices (flattened)
 * @param qAnalyticAnode      Analytic charge on anode (already computed)
 * @param posq                Particle positions and charges
 * @param smallThreshold      Minimum threshold to avoid division by zero
 */
extern "C" __global__ void scaleElectrodeChargesWithConductors(
    int numCathodeAtoms,
    const int* __restrict__ cathodeIndices,
    int numConductorAtoms,
    const int* __restrict__ conductorIndices,
    float qAnalyticAnode,
    float4* __restrict__ posq,
    float smallThreshold
) {
    // Step 1: Compute total numeric charge on cathode + conductors
    // Use reduction pattern
    __shared__ float sdata[256];
    
    float localSum = 0.0f;
    
    // Sum cathode charges
    for (int i = threadIdx.x; i < numCathodeAtoms; i += blockDim.x) {
        int pIdx = cathodeIndices[i];
        localSum += posq[pIdx].w;
    }
    
    // Sum conductor charges
    for (int i = threadIdx.x; i < numConductorAtoms; i += blockDim.x) {
        int pIdx = conductorIndices[i];
        localSum += posq[pIdx].w;
    }
    
    sdata[threadIdx.x] = localSum;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    __shared__ float scaleFactor;
    if (threadIdx.x == 0) {
        float qNumericTotal = sdata[0];
        float qAnalytic = -qAnalyticAnode;  // opposite sign
        
        if (fabsf(qNumericTotal) > smallThreshold) {
            scaleFactor = qAnalytic / qNumericTotal;
        } else {
            scaleFactor = 1.0f;  // no scaling
        }
    }
    __syncthreads();
    
    // Apply scale factor if positive
    if (scaleFactor > 0.0f) {
        // Scale cathode
        for (int i = threadIdx.x; i < numCathodeAtoms; i += blockDim.x) {
            int pIdx = cathodeIndices[i];
            posq[pIdx].w *= scaleFactor;
        }
        
        // Scale conductors
        for (int i = threadIdx.x; i < numConductorAtoms; i += blockDim.x) {
            int pIdx = conductorIndices[i];
            posq[pIdx].w *= scaleFactor;
        }
    }
}


/**
 * Initialize conductor geometry (center, radius, normals).
 * Called once during setup, not per-step.
 *
 * @param numAtoms          Number of atoms in conductor
 * @param conductorIndices  Particle indices
 * @param conductorType     0 = Buckyball, 1 = Nanotube
 * @param axis              Nanotube axis (ignored for Buckyball)
 * @param posq              Particle positions
 * @param normals           Output: surface normals
 * @param center            Output: conductor center
 * @param radius            Output: conductor radius
 */
extern "C" __global__ void initConductorGeometry(
    int numAtoms,
    const int* __restrict__ conductorIndices,
    int conductorType,
    float3 axis,
    const float4* __restrict__ posq,
    float4* __restrict__ normals,
    float3* center,
    float* radius
) {
    // First, compute center (parallel reduction)
    __shared__ float3 sdata3[256];
    
    float3 localSum = make_float3(0.0f, 0.0f, 0.0f);
    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
        int pIdx = conductorIndices[i];
        float4 pos = posq[pIdx];
        localSum.x += pos.x;
        localSum.y += pos.y;
        localSum.z += pos.z;
    }
    
    sdata3[threadIdx.x] = localSum;
    __syncthreads();
    
    // Reduction for center
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sdata3[threadIdx.x].x += sdata3[threadIdx.x + s].x;
            sdata3[threadIdx.x].y += sdata3[threadIdx.x + s].y;
            sdata3[threadIdx.x].z += sdata3[threadIdx.x + s].z;
        }
        __syncthreads();
    }
    
    __shared__ float3 sharedCenter;
    __shared__ float sharedRadius;
    
    if (threadIdx.x == 0) {
        sharedCenter.x = sdata3[0].x / numAtoms;
        sharedCenter.y = sdata3[0].y / numAtoms;
        sharedCenter.z = sdata3[0].z / numAtoms;
        *center = sharedCenter;
        sharedRadius = 0.0f;
    }
    __syncthreads();
    
    // Compute radius and normals
    for (int i = threadIdx.x; i < numAtoms; i += blockDim.x) {
        int pIdx = conductorIndices[i];
        float4 pos = posq[pIdx];
        
        float dx = pos.x - sharedCenter.x;
        float dy = pos.y - sharedCenter.y;
        float dz = pos.z - sharedCenter.z;
        
        float3 radialVec;
        
        if (conductorType == CONDUCTOR_NANOTUBE) {
            // Project out component along axis
            float axisProj = dx * axis.x + dy * axis.y + dz * axis.z;
            radialVec.x = dx - axisProj * axis.x;
            radialVec.y = dy - axisProj * axis.y;
            radialVec.z = dz - axisProj * axis.z;
        } else {  // BUCKYBALL
            radialVec.x = dx;
            radialVec.y = dy;
            radialVec.z = dz;
        }
        
        float r = sqrtf(radialVec.x * radialVec.x + radialVec.y * radialVec.y + radialVec.z * radialVec.z);
        
        // Store normal
        if (r > 1e-8f) {
            normals[i] = make_float4(radialVec.x / r, radialVec.y / r, radialVec.z / r, 0.0f);
        } else {
            normals[i] = make_float4(0.0f, 0.0f, 1.0f, 0.0f);
        }
        
        // First thread with valid radius sets it
        if (i == 0) {
            sharedRadius = r;
            *radius = r;
        }
    }
}
