/**
 * ConstantVoltage Drude Langevin CUDA Kernels
 * 
 * GPU-native implementation of Drude Langevin dynamics with dual-temperature thermostat.
 * Directly copied from OpenMM's drudeLangevin.cc with CUDA-specific modifications.
 * 
 * This file is compiled via JIT and should use CUDA-compatible syntax.
 */

// ═══════════════════════════════════════════════════════════════════════════
// Drude Langevin Integration Part 1: Velocity Update with Dual-Temperature
// ═══════════════════════════════════════════════════════════════════════════

/**
 * Perform the first step of Langevin integration.
 * 
 * Updates velocities using:
 * - vscale, fscale, noisescale for normal particles (system temperature)
 * - vscaleDrude, fscaleDrude, noisescaleDrude for Drude pairs (Drude temperature)
 */
extern "C" __global__ void integrateDrudeLangevinPart1(
    float4* __restrict__ velm,
    const long long* __restrict__ forceBuffer,
    float4* __restrict__ posDelta,
    const int* __restrict__ normalParticles,
    const int2* __restrict__ pairParticles,
    const float2* __restrict__ stepSizeBuffer,
    float vscale,
    float fscale,
    float noisescale,
    float vscaleDrude,
    float fscaleDrude,
    float noisescaleDrude,
    const float4* __restrict__ random,
    unsigned int randomIndex,
    int numNormalParticles,
    int numPairs,
    int paddedNumAtoms)
{
    float stepSize = stepSizeBuffer[0].y;
    
    // Fixed-point force scale
    const float FORCE_SCALE = 1.0f / 4294967296.0f;  // 1/0x100000000
    
    // Update normal particles
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < numNormalParticles) {
        int index = normalParticles[idx];
        float4 velocity = velm[index];
        if (velocity.w != 0.0f) {
            float sqrtInvMass = sqrtf(velocity.w);
            float4 rand = random[randomIndex + index];
            
            // Read forces and convert from fixed-point
            float fx = ((float)forceBuffer[index]) * FORCE_SCALE;
            float fy = ((float)forceBuffer[index + paddedNumAtoms]) * FORCE_SCALE;
            float fz = ((float)forceBuffer[index + 2*paddedNumAtoms]) * FORCE_SCALE;
            
            velocity.x = vscale*velocity.x + fscale*velocity.w*fx + noisescale*sqrtInvMass*rand.x;
            velocity.y = vscale*velocity.y + fscale*velocity.w*fy + noisescale*sqrtInvMass*rand.y;
            velocity.z = vscale*velocity.z + fscale*velocity.w*fz + noisescale*sqrtInvMass*rand.z;
            
            velm[index] = velocity;
            posDelta[index] = make_float4(stepSize*velocity.x, stepSize*velocity.y, stepSize*velocity.z, 0.0f);
        }
    }
}

/**
 * Update Drude particle pairs with dual-temperature thermostat.
 * 
 * For Drude pairs:
 * - Center of mass uses system temperature
 * - Relative motion uses Drude (cold) temperature
 */
extern "C" __global__ void integrateDrudePairs(
    float4* __restrict__ velm,
    const long long* __restrict__ forceBuffer,
    float4* __restrict__ posDelta,
    const int2* __restrict__ pairParticles,
    const float2* __restrict__ stepSizeBuffer,
    float vscale,
    float fscale,
    float noisescale,
    float vscaleDrude,
    float fscaleDrude,
    float noisescaleDrude,
    const float4* __restrict__ random,
    unsigned int randomIndex,
    int numPairs,
    int paddedNumAtoms)
{
    float stepSize = stepSizeBuffer[0].y;
    const float FORCE_SCALE = 1.0f / 4294967296.0f;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPairs) return;
    
    int2 particles = pairParticles[idx];
    float4 velocity1 = velm[particles.x];
    float4 velocity2 = velm[particles.y];
    
    float mass1 = 1.0f / velocity1.w;
    float mass2 = 1.0f / velocity2.w;
    float invTotalMass = 1.0f / (mass1 + mass2);
    float invReducedMass = (mass1 + mass2) * velocity1.w * velocity2.w;
    float mass1fract = invTotalMass * mass1;
    float mass2fract = invTotalMass * mass2;
    float sqrtInvTotalMass = sqrtf(invTotalMass);
    float sqrtInvReducedMass = sqrtf(invReducedMass);
    
    // Center of mass velocity
    float4 cmVel = make_float4(
        velocity1.x*mass1fract + velocity2.x*mass2fract,
        velocity1.y*mass1fract + velocity2.y*mass2fract,
        velocity1.z*mass1fract + velocity2.z*mass2fract,
        0.0f
    );
    
    // Relative velocity
    float4 relVel = make_float4(
        velocity2.x - velocity1.x,
        velocity2.y - velocity1.y,
        velocity2.z - velocity1.z,
        0.0f
    );
    
    // Read forces
    float3 force1 = make_float3(
        ((float)forceBuffer[particles.x]) * FORCE_SCALE,
        ((float)forceBuffer[particles.x + paddedNumAtoms]) * FORCE_SCALE,
        ((float)forceBuffer[particles.x + 2*paddedNumAtoms]) * FORCE_SCALE
    );
    float3 force2 = make_float3(
        ((float)forceBuffer[particles.y]) * FORCE_SCALE,
        ((float)forceBuffer[particles.y + paddedNumAtoms]) * FORCE_SCALE,
        ((float)forceBuffer[particles.y + 2*paddedNumAtoms]) * FORCE_SCALE
    );
    
    float3 cmForce = make_float3(force1.x + force2.x, force1.y + force2.y, force1.z + force2.z);
    float3 relForce = make_float3(
        force2.x*mass1fract - force1.x*mass2fract,
        force2.y*mass1fract - force1.y*mass2fract,
        force2.z*mass1fract - force1.z*mass2fract
    );
    
    // Random numbers for thermal noise
    float4 rand1 = random[randomIndex + 2*idx];
    float4 rand2 = random[randomIndex + 2*idx + 1];
    
    // Update center of mass velocity (system temperature)
    cmVel.x = vscale*cmVel.x + fscale*invTotalMass*cmForce.x + noisescale*sqrtInvTotalMass*rand1.x;
    cmVel.y = vscale*cmVel.y + fscale*invTotalMass*cmForce.y + noisescale*sqrtInvTotalMass*rand1.y;
    cmVel.z = vscale*cmVel.z + fscale*invTotalMass*cmForce.z + noisescale*sqrtInvTotalMass*rand1.z;
    
    // Update relative velocity (Drude temperature - cold)
    relVel.x = vscaleDrude*relVel.x + fscaleDrude*invReducedMass*relForce.x + noisescaleDrude*sqrtInvReducedMass*rand2.x;
    relVel.y = vscaleDrude*relVel.y + fscaleDrude*invReducedMass*relForce.y + noisescaleDrude*sqrtInvReducedMass*rand2.y;
    relVel.z = vscaleDrude*relVel.z + fscaleDrude*invReducedMass*relForce.z + noisescaleDrude*sqrtInvReducedMass*rand2.z;
    
    // Reconstruct individual velocities
    velocity1.x = cmVel.x - relVel.x*mass2fract;
    velocity1.y = cmVel.y - relVel.y*mass2fract;
    velocity1.z = cmVel.z - relVel.z*mass2fract;
    velocity2.x = cmVel.x + relVel.x*mass1fract;
    velocity2.y = cmVel.y + relVel.y*mass1fract;
    velocity2.z = cmVel.z + relVel.z*mass1fract;
    
    velm[particles.x] = velocity1;
    velm[particles.y] = velocity2;
    posDelta[particles.x] = make_float4(stepSize*velocity1.x, stepSize*velocity1.y, stepSize*velocity1.z, 0.0f);
    posDelta[particles.y] = make_float4(stepSize*velocity2.x, stepSize*velocity2.y, stepSize*velocity2.z, 0.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// Drude Langevin Integration Part 2: Position Update
// ═══════════════════════════════════════════════════════════════════════════

extern "C" __global__ void integrateDrudeLangevinPart2(
    float4* __restrict__ posq,
    const float4* __restrict__ posDelta,
    float4* __restrict__ velm,
    const float2* __restrict__ stepSizeBuffer,
    int numAtoms)
{
    float invStepSize = 1.0f / stepSizeBuffer[0].y;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numAtoms) return;
    
    float4 vel = velm[idx];
    if (vel.w != 0.0f) {
        float4 pos = posq[idx];
        float4 delta = posDelta[idx];
        
        pos.x += delta.x;
        pos.y += delta.y;
        pos.z += delta.z;
        
        vel.x = invStepSize * delta.x;
        vel.y = invStepSize * delta.y;
        vel.z = invStepSize * delta.z;
        
        posq[idx] = pos;
        velm[idx] = vel;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Hard Wall Constraints: Prevent Drude-Parent Distance Exceeding Maximum
// ═══════════════════════════════════════════════════════════════════════════

extern "C" __global__ void applyHardWallConstraints(
    float4* __restrict__ posq,
    float4* __restrict__ velm,
    const int2* __restrict__ pairParticles,
    const float2* __restrict__ stepSizeBuffer,
    float maxDrudeDistance,
    float hardwallscaleDrude,
    int numPairs)
{
    float stepSize = stepSizeBuffer[0].y;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numPairs) return;
    
    int2 particles = pairParticles[idx];
    float4 pos1 = posq[particles.x];
    float4 pos2 = posq[particles.y];
    
    float dx = pos1.x - pos2.x;
    float dy = pos1.y - pos2.y;
    float dz = pos1.z - pos2.z;
    float r = sqrtf(dx*dx + dy*dy + dz*dz);
    float rInv = 1.0f / r;
    
    if (rInv * maxDrudeDistance < 1.0f) {
        // Constraint violated - "bounce" off hard wall
        float3 bondDir = make_float3(dx*rInv, dy*rInv, dz*rInv);
        float4 vel1 = velm[particles.x];
        float4 vel2 = velm[particles.y];
        float mass1 = 1.0f / vel1.w;
        float mass2 = 1.0f / vel2.w;
        float deltaR = r - maxDrudeDistance;
        float deltaT = stepSize;
        
        float dotvr1 = vel1.x*bondDir.x + vel1.y*bondDir.y + vel1.z*bondDir.z;
        float3 vb1 = make_float3(bondDir.x*dotvr1, bondDir.y*dotvr1, bondDir.z*dotvr1);
        float3 vp1 = make_float3(vel1.x - vb1.x, vel1.y - vb1.y, vel1.z - vb1.z);
        
        if (vel2.w == 0.0f) {
            // Parent is massless - move only Drude particle
            if (dotvr1 != 0.0f)
                deltaT = deltaR / fabsf(dotvr1);
            if (deltaT > stepSize)
                deltaT = stepSize;
            dotvr1 = -dotvr1 * hardwallscaleDrude / (fabsf(dotvr1) * sqrtf(mass1));
            float dr = -deltaR + deltaT * dotvr1;
            pos1.x += bondDir.x * dr;
            pos1.y += bondDir.y * dr;
            pos1.z += bondDir.z * dr;
            posq[particles.x] = pos1;
            vel1.x = vp1.x + bondDir.x * dotvr1;
            vel1.y = vp1.y + bondDir.y * dotvr1;
            vel1.z = vp1.z + bondDir.z * dotvr1;
            velm[particles.x] = vel1;
        }
        else {
            // Both particles have mass - move both
            float invTotalMass = 1.0f / (mass1 + mass2);
            float dotvr2 = vel2.x*bondDir.x + vel2.y*bondDir.y + vel2.z*bondDir.z;
            float3 vb2 = make_float3(bondDir.x*dotvr2, bondDir.y*dotvr2, bondDir.z*dotvr2);
            float3 vp2 = make_float3(vel2.x - vb2.x, vel2.y - vb2.y, vel2.z - vb2.z);
            float vbCMass = (mass1*dotvr1 + mass2*dotvr2) * invTotalMass;
            dotvr1 -= vbCMass;
            dotvr2 -= vbCMass;
            if (dotvr1 != dotvr2)
                deltaT = deltaR / fabsf(dotvr1 - dotvr2);
            if (deltaT > stepSize)
                deltaT = stepSize;
            float vBond = hardwallscaleDrude / sqrtf(mass1);
            dotvr1 = -dotvr1 * vBond * mass2 * invTotalMass / fabsf(dotvr1);
            dotvr2 = -dotvr2 * vBond * mass1 * invTotalMass / fabsf(dotvr2);
            float dr1 = -deltaR * mass2 * invTotalMass + deltaT * dotvr1;
            float dr2 = deltaR * mass1 * invTotalMass + deltaT * dotvr2;
            dotvr1 += vbCMass;
            dotvr2 += vbCMass;
            pos1.x += bondDir.x * dr1;
            pos1.y += bondDir.y * dr1;
            pos1.z += bondDir.z * dr1;
            pos2.x += bondDir.x * dr2;
            pos2.y += bondDir.y * dr2;
            pos2.z += bondDir.z * dr2;
            posq[particles.x] = pos1;
            posq[particles.y] = pos2;
            vel1.x = vp1.x + bondDir.x * dotvr1;
            vel1.y = vp1.y + bondDir.y * dotvr1;
            vel1.z = vp1.z + bondDir.z * dotvr1;
            vel2.x = vp2.x + bondDir.x * dotvr2;
            vel2.y = vp2.y + bondDir.y * dotvr2;
            vel2.z = vp2.z + bondDir.z * dotvr2;
            velm[particles.x] = vel1;
            velm[particles.y] = vel2;
        }
    }
}
