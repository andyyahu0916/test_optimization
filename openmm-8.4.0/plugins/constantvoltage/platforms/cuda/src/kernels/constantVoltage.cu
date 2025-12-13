/**
 * ConstantVoltage CUDA Kernels
 * 
 * GPU-native implementation of SCF electrode charge updates.
 * Based on Professor McDaniel's Python algorithm.
 * 
 * CRITICAL OPENMM CONVENTIONS:
 * 1. posq = float4(x, y, z, charge)
 * 2. LongForceBuffer = int64 fixed-point, scale = 0x100000000
 * 3. Force layout: Fx at [i], Fy at [i+paddedNumAtoms], Fz at [i+2*paddedNumAtoms]
 */

// Physical constants (from Fixed_Voltage_routines.py)
#ifndef CONVERSION_KJMOL_NM_AU
#define CONVERSION_KJMOL_NM_AU 0.00719760046f  // 18.8973/2625.5 (precise)
#endif

#ifndef FOUR_PI
#define FOUR_PI 12.566370614359172f
#endif

#ifndef SMALL_THRESHOLD
#define SMALL_THRESHOLD 1e-6f
#endif

// Fixed-point force scale (from OpenMM convention)
#define FORCE_SCALE (1.0f / 4294967296.0f)  // 1.0 / 0x100000000

/**
 * Update cathode electrode charges based on SCF iteration.
 * 
 * Python Original (MM_classes.py:323-335):
 *   Ez_external = forces[index][2] / q_i_old  if |q_i_old| > 0.9*small_threshold else 0
 *   q_i = 2.0 / (4.0 * pi) * area * (V/Lgap + Ez) * conversion
 *   if |q_i| < threshold: q_i = threshold (positive for cathode)
 */
extern "C" __global__ void updateCathodeCharges(
    float4* __restrict__ posq,
    const long long* __restrict__ forceBuffer,
    const int* __restrict__ cathodeIndices,
    const float* __restrict__ cathodeAreas,
    int numCathodes,
    int paddedNumAtoms,
    float voltage_kjmol,
    float Lgap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numCathodes) return;
    
    int atomIndex = cathodeIndices[idx];
    float area = cathodeAreas[idx];
    
    // Get current charge
    float q_old = posq[atomIndex].w;
    
    // Get Ez from force buffer (z-component)
    // Force layout: Fz at [atomIndex + 2*paddedNumAtoms]
    long long fz_fixed = forceBuffer[atomIndex + 2 * paddedNumAtoms];
    float fz = ((float)fz_fixed) * FORCE_SCALE;
    
    // Compute external field: Ez = Fz / q
    float Ez_external = 0.0f;
    if (fabsf(q_old) > 0.9f * SMALL_THRESHOLD) {
        Ez_external = fz / q_old;
    }
    
    // New charge from boundary condition: q = (2/4π) * area * (V/Lgap + Ez) * K
    // For CATHODE: positive voltage, positive charge
    float q_new = (2.0f / FOUR_PI) * area * (voltage_kjmol / Lgap + Ez_external) * CONVERSION_KJMOL_NM_AU;
    
    // Prevent charge from getting too small (numerical stability)
    if (fabsf(q_new) < SMALL_THRESHOLD) {
        q_new = SMALL_THRESHOLD;  // Cathode: positive
    }
    
    // Update charge in posq
    posq[atomIndex].w = q_new;
}

/**
 * Update anode electrode charges based on SCF iteration.
 * 
 * Python Original (MM_classes.py:338-350):
 *   Same as cathode but with negative sign
 *   q_i = -2.0 / (4.0 * pi) * area * (V/Lgap + Ez) * conversion
 *   if |q_i| < threshold: q_i = -threshold (negative for anode)
 */
extern "C" __global__ void updateAnodeCharges(
    float4* __restrict__ posq,
    const long long* __restrict__ forceBuffer,
    const int* __restrict__ anodeIndices,
    const float* __restrict__ anodeAreas,
    int numAnodes,
    int paddedNumAtoms,
    float voltage_kjmol,
    float Lgap)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAnodes) return;
    
    int atomIndex = anodeIndices[idx];
    float area = anodeAreas[idx];
    
    // Get current charge
    float q_old = posq[atomIndex].w;
    
    // Get Ez from force buffer
    long long fz_fixed = forceBuffer[atomIndex + 2 * paddedNumAtoms];
    float fz = ((float)fz_fixed) * FORCE_SCALE;
    
    // Compute external field
    float Ez_external = 0.0f;
    if (fabsf(q_old) > 0.9f * SMALL_THRESHOLD) {
        Ez_external = fz / q_old;
    }
    
    // New charge: ANODE = negative sign
    float q_new = -(2.0f / FOUR_PI) * area * (voltage_kjmol / Lgap + Ez_external) * CONVERSION_KJMOL_NM_AU;
    
    // Prevent charge from getting too small (numerical stability)
    if (fabsf(q_new) < SMALL_THRESHOLD) {
        q_new = -SMALL_THRESHOLD;  // Anode: negative
    }
    
    // Update charge in posq
    posq[atomIndex].w = q_new;
}

/**
 * Compute total charge on electrode (parallel reduction).
 * 
 * Uses atomic add for simple implementation.
 * For production, should use proper parallel reduction.
 */
extern "C" __global__ void computeElectrodeCharge(
    const float4* __restrict__ posq,
    const int* __restrict__ electrodeIndices,
    int numElectrode,
    float* __restrict__ totalCharge)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElectrode) return;
    
    int atomIndex = electrodeIndices[idx];
    float charge = posq[atomIndex].w;
    
    atomicAdd(totalCharge, charge);
}

/**
 * Compute analytic charge based on Green's Reciprocity.
 * 
 * Python Original (Fixed_Voltage_routines.py:318-344):
 *   Q_analytic = sign / (4*pi) * sheet_area * (V/Lgap + V/Lcell) * K
 *   
 *   For each electrolyte atom:
 *     z_distance = |z_atom - z_opposite|
 *     Q_analytic += (z_distance / Lcell) * (-q_i)
 * 
 * This kernel computes the image charge contribution.
 */
extern "C" __global__ void computeAnalyticCharge(
    const float4* __restrict__ posq,
    const int* __restrict__ electrolyteIndices,
    int numElectrolytes,
    float z_opposite,
    float Lcell,
    float* __restrict__ analyticCharge)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElectrolytes) return;
    
    int atomIndex = electrolyteIndices[idx];
    float z_atom = posq[atomIndex].z;
    float q_i = posq[atomIndex].w;
    
    // Image charge contribution: (|z - z_opposite| / Lcell) * (-q)
    float z_distance = fabsf(z_atom - z_opposite);
    float contribution = (z_distance / Lcell) * (-q_i);
    
    atomicAdd(analyticCharge, contribution);
}

/**
 * Scale electrode charges to match analytic normalization.
 * 
 * Python Original (MM_classes.py:536-545):
 *   scale_factor = Q_analytic / Q_numeric_total
 *   for atom in electrode_atoms:
 *       atom.charge = atom.charge * scale_factor
 */
extern "C" __global__ void scaleElectrodeCharges(
    float4* __restrict__ posq,
    const int* __restrict__ electrodeIndices,
    int numElectrode,
    float scaleFactor)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numElectrode) return;
    
    int atomIndex = electrodeIndices[idx];
    posq[atomIndex].w *= scaleFactor;
}
