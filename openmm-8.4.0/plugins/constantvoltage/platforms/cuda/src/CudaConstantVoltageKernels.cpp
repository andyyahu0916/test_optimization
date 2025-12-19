/* -------------------------------------------------------------------------- *
 *                                   OpenMM                                   *
 * -------------------------------------------------------------------------- *
 * This is part of the OpenMM molecular simulation toolkit.                   *
 * https://openmm.org                                                         *
 *                                                                            *
 * Copyright (c) 2024 Stanford University and the Authors.                    *
 * Authors: Andy (Constant Voltage Integration)                               *
 * Contributors: Prof. McDaniel (Original Algorithm)                          *
 *                                                                            *
 * Permission is hereby granted, free of charge, to any person obtaining a    *
 * copy of this software and associated documentation files (the "Software"), *
 * to deal in the Software without restriction, including without limitation  *
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,   *
 * and/or sell copies of the Software, and to permit persons to whom the      *
 * Software is furnished to do so, subject to the following conditions:       *
 *                                                                            *
 * The above copyright notice and this permission notice shall be included in *
 * all copies or substantial portions of the Software.                        *
 *                                                                            *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,   *
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL    *
 * THE AUTHORS, CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,    *
 * DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR      *
 * OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE  *
 * USE OR OTHER DEALINGS IN THE SOFTWARE.                                     *
 * -------------------------------------------------------------------------- */

/**
 * CUDA Implementation of ConstantVoltage Kernels
 *
 * Implements GPU-native SCF charge updates + Drude Langevin dynamics.
 * Uses Common platform's ComputeKernel API for portable CUDA/OpenCL code.
 */

#include "CudaConstantVoltageKernels.h"
#include "ConstantVoltageKernelSources.h"
#include "openmm/ConstantVoltageForce.h"
#include "openmm/ConstantVDrudeLangevinIntegrator.h"
#include "openmm/DrudeForce.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaContext.h"
#include "openmm/cuda/CudaArray.h"
#include "openmm/cuda/CudaIntegrationUtilities.h"
#include "openmm/common/ContextSelector.h"
#include "SimTKOpenMMRealType.h"
#include <set>
#include <cmath>

using namespace OpenMM;
using namespace std;

// Physical constants (matching Professor's Python code)
static const double CONVERSION_KJMOL_NM_AU = 18.8973 / 2625.5;  // 0.00719760046
static const double FOUR_PI = 4.0 * 3.14159265358979323846;
static const double SMALL_THRESHOLD = 1e-6;
static const double VOLTAGE_TO_KJMOL = 96.487;  // 1V * 1e = 96.487 kJ/mol (matching Python conversion_eV_Kjmol)
static const double BOLTZMANN_CONSTANTV = 1.380649e-23 * 6.022140857e23 / 1000.0;  // kJ/(mol*K)

// ═══════════════════════════════════════════════════════════════════════════
// CudaCalcConstantVoltageForceKernel
// ═══════════════════════════════════════════════════════════════════════════

CudaCalcConstantVoltageForceKernel::~CudaCalcConstantVoltageForceKernel() {
}

void CudaCalcConstantVoltageForceKernel::initialize(const System& system, const ConstantVoltageForce& f) {
    force = &f;
    hasInitialized = true;
    ContextSelector selector(cu);

    // Copy electrode data
    numCathodes = f.getNumCathodeAtoms();
    numAnodes = f.getNumAnodeAtoms();
    numElectrolytes = f.getNumElectrolyteAtoms();

    // Upload cathode data
    if (numCathodes > 0) {
        vector<int> cathodeIdx(numCathodes);
        vector<float> cathodeAr(numCathodes);
        for (int i = 0; i < numCathodes; i++) {
            int particle;
            double area;
            f.getCathodeAtomParameters(i, particle, area);
            cathodeIdx[i] = particle;
            cathodeAr[i] = (float)area;
        }
        cathodeIndices.initialize(cu, numCathodes, sizeof(int), "cathodeIndices");
        cathodeAreas.initialize(cu, numCathodes, sizeof(float), "cathodeAreas");
        cathodeIndices.upload(cathodeIdx);
        cathodeAreas.upload(cathodeAr);
    }

    // Upload anode data
    if (numAnodes > 0) {
        vector<int> anodeIdx(numAnodes);
        vector<float> anodeAr(numAnodes);
        for (int i = 0; i < numAnodes; i++) {
            int particle;
            double area;
            f.getAnodeAtomParameters(i, particle, area);
            anodeIdx[i] = particle;
            anodeAr[i] = (float)area;
        }
        anodeIndices.initialize(cu, numAnodes, sizeof(int), "anodeIndices");
        anodeAreas.initialize(cu, numAnodes, sizeof(float), "anodeAreas");
        anodeIndices.upload(anodeIdx);
        anodeAreas.upload(anodeAr);
    }

    // Upload electrolyte data (for Green's reciprocity calculation)
    if (numElectrolytes > 0) {
        vector<int> electrolyteIdx(numElectrolytes);
        for (int i = 0; i < numElectrolytes; i++) {
            electrolyteIdx[i] = f.getElectrolyteAtomParticle(i);
        }
        electrolyteIndices.initialize(cu, numElectrolytes, sizeof(int), "electrolyteIndices");
        electrolyteIndices.upload(electrolyteIdx);
    }

    // Store parameters (convert voltage from V to kJ/mol)
    voltage_kjmol = f.getVoltage() * VOLTAGE_TO_KJMOL;
    Lgap = f.getLgap();
    Lcell = f.getLcell();
    totalArea = f.getTotalArea();
    zCathode = f.getZCathode();
    zAnode = f.getZAnode();
    smallThreshold = f.getSmallThreshold();
    numSCFIterations = f.getNumSCFIterations();

    // ═══════════════════════════════════════════════════════════════════════
    // Initialize Conductor Support (Buckyball / Nanotube)
    // ═══════════════════════════════════════════════════════════════════════
    
    numBuckyballs = f.getNumBuckyballConductors();
    numNanotubes = f.getNumNanotubeConductors();
    totalConductorAtoms = 0;
    
    // Initialize Buckyballs
    for (int b = 0; b < numBuckyballs; b++) {
        vector<int> virtualParticles, realParticles;
        string electrodeType;
        f.getBuckyballConductorParameters(b, virtualParticles, realParticles, electrodeType);
        
        // Create GPU array for this conductor's particles
        CudaArray indices;
        indices.initialize<int>(cu, virtualParticles.size(), "buckyballIndices_" + to_string(b));
        indices.upload(virtualParticles);
        conductorIndices.push_back(std::move(indices));
        
        // Normals will be computed in geometry initialization kernel
        CudaArray normals;
        normals.initialize<float4>(cu, virtualParticles.size(), "buckyballNormals_" + to_string(b));
        conductorNormals.push_back(std::move(normals));
        
        conductorTypes.push_back(0);  // CONDUCTOR_BUCKYBALL
        conductorLengths.push_back(0.0f);  // Not used for Buckyball
        totalConductorAtoms += virtualParticles.size();
    }
    
    // FIX D: Get box a vector length for nanotube length calculation
    // Reference: Fixed_Voltage_routines.py uses boxVecs[0][0] for nanotube length
    Vec3 boxVecs[3];
    system.getDefaultPeriodicBoxVectors(boxVecs[0], boxVecs[1], boxVecs[2]);
    double boxA_length = sqrt(boxVecs[0][0]*boxVecs[0][0] + 
                               boxVecs[0][1]*boxVecs[0][1] + 
                               boxVecs[0][2]*boxVecs[0][2]);
    
    // Initialize Nanotubes
    for (int n = 0; n < numNanotubes; n++) {
        vector<int> virtualParticles, realParticles;
        string electrodeType;
        Vec3 axis;
        f.getNanotubeConductorParameters(n, virtualParticles, realParticles, electrodeType, axis);
        
        // Create GPU array for this conductor's particles
        CudaArray indices;
        indices.initialize<int>(cu, virtualParticles.size(), "nanotubeIndices_" + to_string(n));
        indices.upload(virtualParticles);
        conductorIndices.push_back(std::move(indices));
        
        // Normals will be computed in geometry initialization kernel
        CudaArray normals;
        normals.initialize<float4>(cu, virtualParticles.size(), "nanotubeNormals_" + to_string(n));
        conductorNormals.push_back(std::move(normals));
        
        conductorTypes.push_back(1);  // CONDUCTOR_NANOTUBE
        // FIX D: Nanotube length from box a vector (matching Python's boxVecs[0][0])
        conductorLengths.push_back((float)boxA_length);
        totalConductorAtoms += virtualParticles.size();
    }
    
    // Create flattened conductor indices for combined scaling
    if (totalConductorAtoms > 0) {
        vector<int> allIdx;
        for (int c = 0; c < numBuckyballs + numNanotubes; c++) {
            // Get virtualParticles for this conductor
            vector<int> virtualParticles, realParticles;
            string electrodeType;
            if (c < numBuckyballs) {
                f.getBuckyballConductorParameters(c, virtualParticles, realParticles, electrodeType);
            } else {
                Vec3 axis;
                f.getNanotubeConductorParameters(c - numBuckyballs, virtualParticles, realParticles, electrodeType, axis);
            }
            for (int p : virtualParticles) {
                allIdx.push_back(p);
            }
        }
        allConductorIndices.initialize<int>(cu, max((int)allIdx.size(), 1), "allConductorIndices");
        allConductorIndices.upload(allIdx);
    }

    // Allocate reduction buffers
    totalCathodeCharge.initialize<float>(cu, 1, "totalCathodeCharge");
    totalAnodeCharge.initialize<float>(cu, 1, "totalAnodeCharge");
    analyticChargeBuffer.initialize<float>(cu, 1, "analyticChargeBuffer");

    // Compile CUDA kernels from JIT source
    map<string, string> defines;
    defines["NUM_CATHODES"] = cu.intToString(numCathodes);
    defines["NUM_ANODES"] = cu.intToString(numAnodes);
    defines["NUM_ELECTROLYTES"] = cu.intToString(numElectrolytes);
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    // FIX B: Add 'f' suffix to ensure float precision in CUDA kernels
    defines["SMALL_THRESHOLD"] = cu.doubleToString(SMALL_THRESHOLD) + "f";
    defines["CONVERSION_KJMOL_NM_AU"] = cu.doubleToString(CONVERSION_KJMOL_NM_AU) + "f";
    defines["FOUR_PI"] = cu.doubleToString(FOUR_PI) + "f";

    CUmodule module = cu.createModule(ConstantVoltageKernelSources::constantVoltage, defines);
    updateCathodeChargesKernel = cu.getKernel(module, "updateCathodeCharges");
    updateAnodeChargesKernel = cu.getKernel(module, "updateAnodeCharges");
    computeElectrodeChargeKernel = cu.getKernel(module, "computeElectrodeCharge");
    computeAnalyticChargeKernel = cu.getKernel(module, "computeAnalyticCharge");
    scaleElectrodeChargesKernel = cu.getKernel(module, "scaleElectrodeCharges");
    
    // Compile conductor kernels if needed
    if (numBuckyballs + numNanotubes > 0) {
        CUmodule conductorModule = cu.createModule(ConstantVoltageKernelSources::conductorCharge, defines);
        computeConductorImageChargesKernel = cu.getKernel(conductorModule, "computeConductorImageCharges");
        // FIX A: Split charge transfer into two kernels for multi-block safety
        computeConductorDqPerAtomKernel = cu.getKernel(conductorModule, "computeConductorDqPerAtom");
        applyConductorChargeTransferKernel = cu.getKernel(conductorModule, "applyConductorChargeTransfer");
        scaleElectrodeChargesWithConductorsKernel = cu.getKernel(conductorModule, "scaleElectrodeChargesWithConductors");
        initConductorGeometryKernel = cu.getKernel(conductorModule, "initConductorGeometry");
        
        // Initialize vectors for contact information (to be filled during first geometry init)
        conductorAreas.resize(numBuckyballs + numNanotubes, 0.0f);
        conductorDrContact.resize(numBuckyballs + numNanotubes, 0.0f);
        conductorContactAtoms.resize(numBuckyballs + numNanotubes, 0);
        conductorContactNormals.resize(numBuckyballs + numNanotubes, make_float3(0.0f, 0.0f, 1.0f));
        conductorIsCloseToElectrode.resize(numBuckyballs + numNanotubes, true);
        
        // FIX A: Allocate global buffer for charge transfer broadcast (multi-block safe)
        dqPerAtomBuffer.initialize<float>(cu, 1, "dqPerAtomBuffer");
    }
}

double CudaCalcConstantVoltageForceKernel::execute(ContextImpl& context, bool includeForces, bool includeEnergy) {
    // ConstantVoltageForce doesn't compute forces directly
    // SCF charge updates are triggered by the Integrator
    
    // Initialize conductor geometry on first call (when positions are available)
    static bool geometryInitialized = false;
    if (!geometryInitialized && (numBuckyballs + numNanotubes > 0)) {
        initializeConductorGeometry(context);
        geometryInitialized = true;
    }
    
    return 0.0;
}

void CudaCalcConstantVoltageForceKernel::initializeConductorGeometry(ContextImpl& context) {
    /**
     * Initialize conductor geometry (center, radius, surface normals) and
     * find contact atom/distance for each conductor.
     * 
     * Reference: Fixed_Voltage_routines.py
     * - Buckyball_Virtual.__init__ (line 424-459)
     * - Nanotube_Virtual.__init__ (line 517-572)
     * - find_contact_neighbor_conductor (line 170-210)
     */
    
    ContextSelector selector(cu);
    
    // Get positions from context
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    
    // Allocate temporary buffers for geometry output (per-conductor)
    CudaArray centerBuffer, radiusBuffer;
    centerBuffer.initialize<float3>(cu, 1, "conductorCenter");
    radiusBuffer.initialize<float>(cu, 1, "conductorRadius");
    
    for (int c = 0; c < numBuckyballs + numNanotubes; c++) {
        int numAtoms = (int)conductorIndices[c].getSize() / sizeof(int);
        int conductorType = conductorTypes[c];
        float length = conductorLengths[c];
        
        // Get axis for nanotube (default z for buckyball)
        float3 axis = make_float3(0.0f, 0.0f, 1.0f);
        if (conductorType == 1) {  // Nanotube
            // Get axis from force parameters
            vector<int> vp, rp;
            string et;
            Vec3 axisVec;
            force->getNanotubeConductorParameters(c - numBuckyballs, vp, rp, et, axisVec);
            axis = make_float3((float)axisVec[0], (float)axisVec[1], (float)axisVec[2]);
        }
        
        CUdeviceptr conductorIdxPtr = conductorIndices[c].getDevicePointer();
        CUdeviceptr normalsPtr = conductorNormals[c].getDevicePointer();
        CUdeviceptr centerPtr = centerBuffer.getDevicePointer();
        CUdeviceptr radiusPtr = radiusBuffer.getDevicePointer();
        
        // Call geometry init kernel
        void* geoArgs[] = {
            &numAtoms,
            &conductorIdxPtr,
            &conductorType,
            &axis,
            &posqPtr,
            &normalsPtr,
            &centerPtr,
            &radiusPtr
        };
        cu.executeKernel(initConductorGeometryKernel, geoArgs, numAtoms);
        
        // Download center and radius
        float3 center;
        float radius;
        centerBuffer.download(&center);
        radiusBuffer.download(&radius);
        
        // Compute area per atom based on conductor type
        // Reference: Buckyball: area = 4*pi*r^2/N, Nanotube: area = 2*pi*r*L/N
        if (conductorType == 0) {  // Buckyball
            conductorAreas[c] = (float)(4.0 * M_PI * radius * radius / numAtoms);
        } else {  // Nanotube
            conductorAreas[c] = (float)(2.0 * M_PI * radius * length / numAtoms);
        }
        
        // ==================================================================
        // FIX E: Find contact atom (matching find_contact_neighbor_conductor)
        // Reference: Fixed_Voltage_routines.py:177-227
        // 1. First check distance to primary electrode (cathode or anode)
        // 2. If too far (> threshold), search other conductors
        // ==================================================================
        
        // Download positions to CPU for contact detection
        vector<float4> positions(cu.getPaddedNumAtoms());
        cu.getPosq().download(positions);
        
        // Get electrode type for this conductor
        vector<int> vp, rp;
        string electrodeType;
        if (c < numBuckyballs) {
            force->getBuckyballConductorParameters(c, vp, rp, electrodeType);
        } else {
            Vec3 axisVec;
            force->getNanotubeConductorParameters(c - numBuckyballs, vp, rp, electrodeType, axisVec);
        }
        
        // Threshold for "close" contact (default 0.5 nm)
        const float closeThreshold = 0.5f;
        float minDist = 1e30f;
        int closestAtom = 0;
        bool isCloseToElectrode = false;
        bool isCathodeContact = false;
        
        // Step 1: Search primary electrode based on electrode_type
        if (electrodeType == "cathode" && numCathodes > 0) {
            // Search cathode
            vector<int> cathodeIdx(numCathodes);
            cathodeIndices.download(cathodeIdx);
            
            for (int i = 0; i < numCathodes; i++) {
                int idx = cathodeIdx[i];
                float dx = positions[idx].x - center.x;
                float dy = positions[idx].y - center.y;
                float dz = positions[idx].z - center.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < minDist) {
                    minDist = dist;
                    closestAtom = idx;
                    isCathodeContact = true;
                }
            }
            if (minDist < closeThreshold) {
                isCloseToElectrode = true;
            }
        } else if (electrodeType == "anode" && numAnodes > 0) {
            // Search anode
            vector<int> anodeIdx(numAnodes);
            anodeIndices.download(anodeIdx);
            
            for (int i = 0; i < numAnodes; i++) {
                int idx = anodeIdx[i];
                float dx = positions[idx].x - center.x;
                float dy = positions[idx].y - center.y;
                float dz = positions[idx].z - center.z;
                float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                if (dist < minDist) {
                    minDist = dist;
                    closestAtom = idx;
                    isCathodeContact = false;
                }
            }
            if (minDist < closeThreshold) {
                isCloseToElectrode = true;
            }
        }
        
        // Step 2: If not close to primary electrode, search other conductors
        if (!isCloseToElectrode) {
            // Loop over all other conductors
            for (int other = 0; other < numBuckyballs + numNanotubes; other++) {
                if (other == c) continue;  // Skip self
                
                // Get atoms of this conductor
                vector<int> otherIdx(conductorIndices[other].getSize() / sizeof(int));
                conductorIndices[other].download(otherIdx);
                
                // Get atoms of current conductor
                vector<int> currentIdx(conductorIndices[c].getSize() / sizeof(int));
                conductorIndices[c].download(currentIdx);
                
                // Double loop to find closest pair (as in Python)
                for (int atom1 : currentIdx) {
                    for (int atom2 : otherIdx) {
                        float dx = positions[atom1].x - positions[atom2].x;
                        float dy = positions[atom1].y - positions[atom2].y;
                        float dz = positions[atom1].z - positions[atom2].z;
                        float dist = sqrtf(dx*dx + dy*dy + dz*dz);
                        if (dist < minDist) {
                            minDist = dist;
                            closestAtom = atom2;
                        }
                    }
                }
                
                if (minDist < closeThreshold) {
                    isCloseToElectrode = false;  // Contact with conductor, not electrode
                    break;
                }
            }
        }
        
        // Store contact info
        conductorContactAtoms[c] = closestAtom;
        conductorDrContact[c] = minDist;
        conductorIsCloseToElectrode[c] = isCloseToElectrode;
        
        // Set contact normal based on contact type
        // Reference: Fixed_Voltage_routines.py:265-266
        if (isCloseToElectrode) {
            // Contact with flat electrode: use electrode surface normal
            // Cathode: normal points +z, Anode: normal points -z
            if (isCathodeContact) {
                conductorContactNormals[c] = make_float3(0.0f, 0.0f, 1.0f);
            } else {
                conductorContactNormals[c] = make_float3(0.0f, 0.0f, -1.0f);
            }
        } else {
            // Contact with another conductor: use geometric direction from contact to center
            float3 drVec;
            drVec.x = center.x - positions[closestAtom].x;
            drVec.y = center.y - positions[closestAtom].y;
            drVec.z = center.z - positions[closestAtom].z;
            float drMag = sqrtf(drVec.x*drVec.x + drVec.y*drVec.y + drVec.z*drVec.z);
            if (drMag > 1e-8f) {
                conductorContactNormals[c] = make_float3(drVec.x/drMag, drVec.y/drMag, drVec.z/drMag);
            } else {
                conductorContactNormals[c] = make_float3(0.0f, 0.0f, 1.0f);
            }
            
            // For Nanotube, need to project out axis component
            // Reference: Fixed_Voltage_routines.py:568-570
            if (conductorType == 1) {  // Nanotube
                Vec3 axisVec;
                force->getNanotubeConductorParameters(c - numBuckyballs, vp, rp, electrodeType, axisVec);
                float3 axis = make_float3((float)axisVec[0], (float)axisVec[1], (float)axisVec[2]);
                
                // Project dr to get radial component
                float axisProj = drVec.x * axis.x + drVec.y * axis.y + drVec.z * axis.z;
                float3 radialVec;
                radialVec.x = drVec.x - axisProj * axis.x;
                radialVec.y = drVec.y - axisProj * axis.y;
                radialVec.z = drVec.z - axisProj * axis.z;
                
                float radialMag = sqrtf(radialVec.x*radialVec.x + radialVec.y*radialVec.y + radialVec.z*radialVec.z);
                conductorDrContact[c] = radialMag;  // Use radial distance for nanotube
            }
        }
    }
}

void CudaCalcConstantVoltageForceKernel::updateElectrodeCharges(ContextImpl& context) {
    if (!hasInitialized || (numCathodes == 0 && numAnodes == 0))
        return;
    
    ContextSelector selector(cu);
    
    // Get buffer pointers for kernel arguments
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    
    // ═══════════════════════════════════════════════════════════════════════
    // STEP 1: Update flat electrode charges (cathode/anode)
    // Reference: MM_classes.py:321-350
    // ═══════════════════════════════════════════════════════════════════════
    
    // Update cathode charges
    if (numCathodes > 0) {
        float v_kjmol = (float)voltage_kjmol;
        float lgap = (float)Lgap;
        float smallThresholdF = (float)smallThreshold;  // FIX C: Runtime parameter
        CUdeviceptr cathodeIdxPtr = cathodeIndices.getDevicePointer();
        CUdeviceptr cathodeAreaPtr = cathodeAreas.getDevicePointer();
        
        void* cathodeArgs[] = {
            &posqPtr,
            &forcePtr,
            &cathodeIdxPtr,
            &cathodeAreaPtr,
            &numCathodes,
            &paddedNumAtoms,
            &v_kjmol,
            &lgap,
            &smallThresholdF  // FIX C: Pass runtime threshold
        };
        cu.executeKernel(updateCathodeChargesKernel, cathodeArgs, numCathodes);
    }
    
    // Update anode charges
    if (numAnodes > 0) {
        float v_kjmol = (float)voltage_kjmol;
        float lgap = (float)Lgap;
        float smallThresholdF = (float)smallThreshold;  // FIX C: Runtime parameter
        CUdeviceptr anodeIdxPtr = anodeIndices.getDevicePointer();
        CUdeviceptr anodeAreaPtr = anodeAreas.getDevicePointer();
        
        void* anodeArgs[] = {
            &posqPtr,
            &forcePtr,
            &anodeIdxPtr,
            &anodeAreaPtr,
            &numAnodes,
            &paddedNumAtoms,
            &v_kjmol,
            &lgap,
            &smallThresholdF  // FIX C: Pass runtime threshold
        };
        cu.executeKernel(updateAnodeChargesKernel, anodeArgs, numAnodes);
    }
    
    // ═══════════════════════════════════════════════════════════════════════
    // STEP 2: Update conductor charges (Buckyball / Nanotube)
    // Reference: MM_classes.py:352-381 (Numerical_charge_Conductor)
    // ═══════════════════════════════════════════════════════════════════════
    
    int numTotalConductors = numBuckyballs + numNanotubes;
    float smallThresholdF = (float)smallThreshold;
    
    if (numTotalConductors > 0) {
        // Step 2a: Image charges on each conductor
        // Reference: MM_classes.py:396-424 (Step 1 of Numerical_charge_Conductor)
        for (int c = 0; c < numTotalConductors; c++) {
            int numAtoms = (int)conductorIndices[c].getSize() / sizeof(int);
            float areaPerAtom = conductorAreas[c];
            
            CUdeviceptr conductorIdxPtr = conductorIndices[c].getDevicePointer();
            CUdeviceptr normalsPtr = conductorNormals[c].getDevicePointer();
            
            void* imageArgs[] = {
                &numAtoms,
                &conductorIdxPtr,
                &normalsPtr,
                &areaPerAtom,
                &posqPtr,
                &forcePtr,
                &paddedNumAtoms,
                &smallThresholdF
            };
            cu.executeKernel(computeConductorImageChargesKernel, imageArgs, numAtoms);
        }
        
        // Step 2b: CRITICAL - Recompute forces after image charges
        // Reference: MM_classes.py:424-426
        // The image charges affect the field at the contact atom, so we must
        // update the context and recalculate forces before charge transfer.
        cu.getPosq().copyTo(cu.getPosqCorrection());
        context.calcForcesAndEnergy(true, false);
        forcePtr = cu.getLongForceBuffer().getDevicePointer();  // Refresh pointer
        
        // Step 2c: Charge transfer for each conductor
        // Reference: MM_classes.py:429-495 (Step 2 of Numerical_charge_Conductor)
        for (int c = 0; c < numTotalConductors; c++) {
            int numAtoms = (int)conductorIndices[c].getSize() / sizeof(int);
            int conductorType = conductorTypes[c];
            float drContact = conductorDrContact[c];
            float length = conductorLengths[c];
            int contactAtomIdx = conductorContactAtoms[c];
            float3 contactNormal = conductorContactNormals[c];
            int isCloseToElectrode = conductorIsCloseToElectrode[c] ? 1 : 0;
            float v_kjmol = (float)voltage_kjmol;
            float lgap = (float)Lgap;
            
            CUdeviceptr conductorIdxPtr = conductorIndices[c].getDevicePointer();
            CUdeviceptr dqPerAtomPtr = dqPerAtomBuffer.getDevicePointer();  // FIX A
            
            // FIX A: Split into two kernel calls for multi-block safety
            // Step 1: Compute dq_per_atom (single thread)
            void* dqArgs[] = {
                &numAtoms,
                &contactAtomIdx,
                &contactNormal,
                &conductorType,
                &drContact,
                &length,
                &isCloseToElectrode,
                &v_kjmol,
                &lgap,
                &posqPtr,
                &forcePtr,
                &paddedNumAtoms,
                &dqPerAtomPtr
            };
            cu.executeKernel(computeConductorDqPerAtomKernel, dqArgs, 1);
            
            // Step 2: Apply charge transfer to all atoms
            void* applyArgs[] = {
                &numAtoms,
                &conductorIdxPtr,
                &posqPtr,
                &dqPerAtomPtr
            };
            cu.executeKernel(applyConductorChargeTransferKernel, applyArgs, numAtoms);
        }
        
        // ═══════════════════════════════════════════════════════════════════
        // FIX 2 & 5: Scale with proper analytic charge calculation
        // Reference: MM_classes.py:509-545 (Scale_charges_analytic_general)
        // ═══════════════════════════════════════════════════════════════════
        
        // Step 2d: First, scale anode INDEPENDENTLY
        // Reference: MM_classes.py:514-515
        // "assume anode is scaled normally..."
        double qAnalyticAnode = computeAnalyticChargeWithElectrolytePlusConductors(context, zCathode, false);
        double qNumericAnode = getTotalAnodeCharge(context);
        
        if (fabs(qNumericAnode) > smallThreshold) {
            double scaleAnode = qAnalyticAnode / qNumericAnode;
            if (scaleAnode > 0.0) {
                float scaleAnodeF = (float)scaleAnode;
                CUdeviceptr anodeIdxPtr = anodeIndices.getDevicePointer();
                
                void* scaleAnodeArgs[] = {
                    &posqPtr,
                    &anodeIdxPtr,
                    &numAnodes,
                    &scaleAnodeF
                };
                cu.executeKernel(scaleElectrodeChargesKernel, scaleAnodeArgs, numAnodes);
            }
        }
        
        // Step 2e: THEN scale cathode + all conductors using -qAnalyticAnode
        // Reference: MM_classes.py:517 "Q_analytic = -1.0 * self.Anode.Q_analytic"
        // Re-read anode analytic charge after scaling (should be same value)
        double qAnalyticCathodePlusConductors = -qAnalyticAnode;
        
        // Get numeric total of cathode + conductors
        double qNumericCathode = getTotalCathodeCharge(context);
        double qNumericConductors = 0.0;
        
        // Sum conductor charges
        for (int c = 0; c < numTotalConductors; c++) {
            int numAtoms = (int)conductorIndices[c].getSize() / sizeof(int);
            vector<int> condIdx(numAtoms);
            conductorIndices[c].download(condIdx);
            
            vector<float4> positions(cu.getPaddedNumAtoms());
            cu.getPosq().download(positions);
            
            for (int i = 0; i < numAtoms; i++) {
                qNumericConductors += positions[condIdx[i]].w;
            }
        }
        
        double qNumericTotal = qNumericCathode + qNumericConductors;
        
        if (fabs(qNumericTotal) > smallThreshold) {
            double scaleCathodeConductors = qAnalyticCathodePlusConductors / qNumericTotal;
            if (scaleCathodeConductors > 0.0) {
                float scaleF = (float)scaleCathodeConductors;
                
                // Scale cathode
                CUdeviceptr cathodeIdxPtr = cathodeIndices.getDevicePointer();
                void* scaleCathodeArgs[] = {
                    &posqPtr,
                    &cathodeIdxPtr,
                    &numCathodes,
                    &scaleF
                };
                cu.executeKernel(scaleElectrodeChargesKernel, scaleCathodeArgs, numCathodes);
                
                // Scale all conductors
                for (int c = 0; c < numTotalConductors; c++) {
                    int numAtoms = (int)conductorIndices[c].getSize() / sizeof(int);
                    CUdeviceptr conductorIdxPtr = conductorIndices[c].getDevicePointer();
                    
                    void* scaleConductorArgs[] = {
                        &posqPtr,
                        &conductorIdxPtr,
                        &numAtoms,
                        &scaleF
                    };
                    cu.executeKernel(scaleElectrodeChargesKernel, scaleConductorArgs, numAtoms);
                }
            }
        }
    } else {
        // ═══════════════════════════════════════════════════════════════════
        // FIX 1: No conductors - scale cathode and anode independently
        // Reference: MM_classes.py:547-550
        // "no extra conductors, scale each electrode to individual Analytic normalization"
        // ═══════════════════════════════════════════════════════════════════
        
        // Scale cathode
        if (numCathodes > 0) {
            double qAnalyticCathode = computeAnalyticChargeWithElectrolyte(context, zAnode, true);
            double qNumericCathode = getTotalCathodeCharge(context);
            
            if (fabs(qNumericCathode) > smallThreshold) {
                double scaleCathode = qAnalyticCathode / qNumericCathode;
                if (scaleCathode > 0.0) {
                    float scaleCathodeF = (float)scaleCathode;
                    CUdeviceptr cathodeIdxPtr = cathodeIndices.getDevicePointer();
                    
                    void* scaleCathodeArgs[] = {
                        &posqPtr,
                        &cathodeIdxPtr,
                        &numCathodes,
                        &scaleCathodeF
                    };
                    cu.executeKernel(scaleElectrodeChargesKernel, scaleCathodeArgs, numCathodes);
                }
            }
        }
        
        // Scale anode
        if (numAnodes > 0) {
            double qAnalyticAnode = computeAnalyticChargeWithElectrolyte(context, zCathode, false);
            double qNumericAnode = getTotalAnodeCharge(context);
            
            if (fabs(qNumericAnode) > smallThreshold) {
                double scaleAnode = qAnalyticAnode / qNumericAnode;
                if (scaleAnode > 0.0) {
                    float scaleAnodeF = (float)scaleAnode;
                    CUdeviceptr anodeIdxPtr = anodeIndices.getDevicePointer();
                    
                    void* scaleAnodeArgs[] = {
                        &posqPtr,
                        &anodeIdxPtr,
                        &numAnodes,
                        &scaleAnodeF
                    };
                    cu.executeKernel(scaleElectrodeChargesKernel, scaleAnodeArgs, numAnodes);
                }
            }
        }
    }
}

double CudaCalcConstantVoltageForceKernel::getTotalCathodeCharge(ContextImpl& context) {
    if (numCathodes == 0)
        return 0.0;
    
    ContextSelector selector(cu);
    
    // Reset accumulator
    float zero = 0.0f;
    totalCathodeCharge.upload(&zero);
    
    // Compute charge sum
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr cathodeIdxPtr = cathodeIndices.getDevicePointer();
    CUdeviceptr chargePtr = totalCathodeCharge.getDevicePointer();
    
    void* args[] = {
        &posqPtr,
        &cathodeIdxPtr,
        &numCathodes,
        &chargePtr
    };
    cu.executeKernel(computeElectrodeChargeKernel, args, numCathodes);
    
    // Download result
    float result;
    totalCathodeCharge.download(&result);
    return (double)result;
}

double CudaCalcConstantVoltageForceKernel::getTotalAnodeCharge(ContextImpl& context) {
    if (numAnodes == 0)
        return 0.0;
    
    ContextSelector selector(cu);
    
    // Reset accumulator
    float zero = 0.0f;
    totalAnodeCharge.upload(&zero);
    
    // Compute charge sum
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    CUdeviceptr anodeIdxPtr = anodeIndices.getDevicePointer();
    CUdeviceptr chargePtr = totalAnodeCharge.getDevicePointer();
    
    void* args[] = {
        &posqPtr,
        &anodeIdxPtr,
        &numAnodes,
        &chargePtr
    };
    cu.executeKernel(computeElectrodeChargeKernel, args, numAnodes);
    
    // Download result
    float result;
    totalAnodeCharge.download(&result);
    return (double)result;
}

double CudaCalcConstantVoltageForceKernel::computeAnalyticChargeWithElectrolyte(
    ContextImpl& context, double z_opposite, bool isCathode)
{
    /**
     * Compute analytic charge including electrolyte image charge contribution.
     * Reference: Fixed_Voltage_routines.py:318-344
     * 
     * Q_analytic = sign / (4π) * area * (V/Lgap + V/Lcell) * K
     *            + Σ_electrolyte (|z - z_opposite| / Lcell) * (-q_i)
     */
    
    double sign = isCathode ? 1.0 : -1.0;
    
    // Geometric contribution
    double qGeometric = sign * totalArea * (voltage_kjmol / Lgap + voltage_kjmol / Lcell) 
                        * CONVERSION_KJMOL_NM_AU / FOUR_PI;
    
    // Electrolyte image charge contribution
    if (numElectrolytes > 0) {
        ContextSelector selector(cu);
        
        // Reset accumulator
        float zero = 0.0f;
        analyticChargeBuffer.upload(&zero);
        
        // Call computeAnalyticCharge kernel
        CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
        CUdeviceptr electrolyteIdxPtr = electrolyteIndices.getDevicePointer();
        CUdeviceptr contribPtr = analyticChargeBuffer.getDevicePointer();
        float z_opp = (float)z_opposite;
        float lcell = (float)Lcell;
        
        void* args[] = {
            &posqPtr,
            &electrolyteIdxPtr,
            &numElectrolytes,
            &z_opp,
            &lcell,
            &contribPtr
        };
        cu.executeKernel(computeAnalyticChargeKernel, args, numElectrolytes);
        
        float imageContrib;
        analyticChargeBuffer.download(&imageContrib);
        qGeometric += (double)imageContrib;
    }
    
    return qGeometric;
}

double CudaCalcConstantVoltageForceKernel::computeAnalyticChargeWithElectrolytePlusConductors(
    ContextImpl& context, double z_opposite, bool isCathode)
{
    /**
     * Compute analytic charge including both electrolyte and conductor contributions.
     * Reference: Fixed_Voltage_routines.py:336-344
     * 
     * Conductors are effectively part of the electrolyte as far as the analytic
     * charge formula is concerned, so we add their image charge contribution.
     */
    
    double result = computeAnalyticChargeWithElectrolyte(context, z_opposite, isCathode);
    
    // Add conductor atoms contribution
    if (totalConductorAtoms > 0) {
        ContextSelector selector(cu);
        
        // Reset accumulator
        float zero = 0.0f;
        analyticChargeBuffer.upload(&zero);
        
        CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
        CUdeviceptr condIdxPtr = allConductorIndices.getDevicePointer();
        CUdeviceptr contribPtr = analyticChargeBuffer.getDevicePointer();
        float z_opp = (float)z_opposite;
        float lcell = (float)Lcell;
        
        void* args[] = {
            &posqPtr,
            &condIdxPtr,
            &totalConductorAtoms,
            &z_opp,
            &lcell,
            &contribPtr
        };
        cu.executeKernel(computeAnalyticChargeKernel, args, totalConductorAtoms);
        
        float condContrib;
        analyticChargeBuffer.download(&condContrib);
        result += (double)condContrib;
    }
    
    return result;
}

// ═══════════════════════════════════════════════════════════════════════════
// CudaIntegrateConstantVDrudeLangevinStepKernel
// ═══════════════════════════════════════════════════════════════════════════

CudaIntegrateConstantVDrudeLangevinStepKernel::~CudaIntegrateConstantVDrudeLangevinStepKernel() {
}

void CudaIntegrateConstantVDrudeLangevinStepKernel::initialize(
    const System& system,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    hasInitialized = true;
    stepCount = 0;
    maxDrudeDistance = integrator.getMaxDrudeDistance();
    prevStepSize = -1.0;
    ContextSelector selector(cu);
    
    // Initialize random number generator
    cu.getIntegrationUtilities().initRandomNumberGenerator((unsigned int)integrator.getRandomNumberSeed());

    // Find ConstantVoltageForce to get SCF parameters
    for (int i = 0; i < system.getNumForces(); i++) {
        const ConstantVoltageForce* cvForce = dynamic_cast<const ConstantVoltageForce*>(&system.getForce(i));
        if (cvForce != nullptr) {
            scfFrequency = cvForce->getSCFFrequency();
            numSCFIterations = cvForce->getNumSCFIterations();
            break;
        }
    }

    // Find DrudeForce and identify Drude particle pairs
    const DrudeForce* drudeForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        drudeForce = dynamic_cast<const DrudeForce*>(&system.getForce(i));
        if (drudeForce != nullptr)
            break;
    }

    // Identify particle pairs and ordinary particles
    set<int> particles;
    vector<int> normalParticleVec;
    vector<int2> pairParticleVec;
    
    for (int i = 0; i < system.getNumParticles(); i++)
        particles.insert(i);
    
    if (drudeForce != nullptr) {
        for (int i = 0; i < drudeForce->getNumParticles(); i++) {
            int p, p1, p2, p3, p4;
            double charge, polarizability, aniso12, aniso34;
            drudeForce->getParticleParameters(i, p, p1, p2, p3, p4, charge, polarizability, aniso12, aniso34);
            particles.erase(p);   // Drude particle
            particles.erase(p1);  // Parent particle
            int2 pair;
            pair.x = p;
            pair.y = p1;
            pairParticleVec.push_back(pair);
        }
    }
    
    normalParticleVec.insert(normalParticleVec.begin(), particles.begin(), particles.end());
    
    // Upload particle arrays to GPU
    normalParticles.initialize<int>(cu, max((int)normalParticleVec.size(), 1), "constantVNormalParticles");
    pairParticles.initialize<int2>(cu, max((int)pairParticleVec.size(), 1), "constantVPairParticles");
    
    if (normalParticleVec.size() > 0)
        normalParticles.upload(normalParticleVec);
    if (pairParticleVec.size() > 0)
        pairParticles.upload(pairParticleVec);
    
    numNormalParticles = normalParticleVec.size();
    numDrudePairs = pairParticleVec.size();

    // Compile Drude Langevin integration kernels
    map<string, string> defines;
    defines["NUM_ATOMS"] = cu.intToString(cu.getNumAtoms());
    defines["PADDED_NUM_ATOMS"] = cu.intToString(cu.getPaddedNumAtoms());
    defines["NUM_NORMAL_PARTICLES"] = cu.intToString(numNormalParticles);
    defines["NUM_PAIRS"] = cu.intToString(numDrudePairs);

    CUmodule module = cu.createModule(ConstantVoltageKernelSources::drudeLangevin, defines);
    kernel1 = cu.getKernel(module, "integrateDrudeLangevinPart1");
    kernelPairs = cu.getKernel(module, "integrateDrudePairs");
    kernel2 = cu.getKernel(module, "integrateDrudeLangevinPart2");
    hardwallKernel = cu.getKernel(module, "applyHardWallConstraints");
}

void CudaIntegrateConstantVDrudeLangevinStepKernel::execute(
    ContextImpl& context,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    ContextSelector selector(cu);
    CudaIntegrationUtilities& integration = cu.getIntegrationUtilities();
    int numAtoms = cu.getNumAtoms();
    
    // Check if SCF update is needed
    if (stepCount % scfFrequency == 0 && forceKernel != nullptr) {
        for (int i = 0; i < numSCFIterations; i++) {
            // Recalculate forces
            context.calcForcesAndEnergy(true, false, context.getIntegrator().getIntegrationForceGroups());
            // Update electrode charges
            forceKernel->updateElectrodeCharges(context);
        }
    }

    // Compute integrator coefficients
    double stepSize = integrator.getStepSize();
    double temperature = integrator.getTemperature();
    double friction = integrator.getFriction();
    double drudeTemperature = integrator.getDrudeTemperature();
    double drudeFriction = integrator.getDrudeFriction();
    
    double vscale = exp(-stepSize * friction);
    double fscale = (1 - vscale) / friction / (double)0x100000000;
    double noisescale = sqrt(2 * BOLTZMANN_CONSTANTV * temperature * friction) * sqrt(0.5 * (1 - vscale*vscale) / friction);
    
    double vscaleDrude = exp(-stepSize * drudeFriction);
    double fscaleDrude = (1 - vscaleDrude) / drudeFriction / (double)0x100000000;
    double noisescaleDrude = sqrt(2 * BOLTZMANN_CONSTANTV * drudeTemperature * drudeFriction) * sqrt(0.5 * (1 - vscaleDrude*vscaleDrude) / drudeFriction);
    
    double hardwallscaleDrude = sqrt(BOLTZMANN_CONSTANTV * drudeTemperature);
    
    // Update step size buffer if needed
    if (stepSize != prevStepSize) {
        float2 ss;
        ss.x = 0.0f;
        ss.y = (float)stepSize;
        integration.getStepSize().upload(&ss);
        prevStepSize = stepSize;
    }

    // Get buffer pointers
    CUdeviceptr velmPtr = cu.getVelm().getDevicePointer();
    CUdeviceptr forcePtr = cu.getLongForceBuffer().getDevicePointer();
    CUdeviceptr posDeltaPtr = integration.getPosDelta().getDevicePointer();
    CUdeviceptr normalPtr = normalParticles.getDevicePointer();
    CUdeviceptr pairPtr = pairParticles.getDevicePointer();
    CUdeviceptr stepSizePtr = integration.getStepSize().getDevicePointer();
    int paddedNumAtoms = cu.getPaddedNumAtoms();
    
    // Get random numbers
    unsigned int randomIndex = integration.prepareRandomNumbers(numNormalParticles + 2*numDrudePairs);
    CUdeviceptr randomPtr = integration.getRandom().getDevicePointer();
    
    // Convert to float for kernel args
    float vscaleF = (float)vscale;
    float fscaleF = (float)fscale;
    float noisescaleF = (float)noisescale;
    float vscaleDrudeF = (float)vscaleDrude;
    float fscaleDrudeF = (float)fscaleDrude;
    float noisescaleDrudeF = (float)noisescaleDrude;
    float maxDrudeDistF = (float)maxDrudeDistance;
    float hardwallscaleF = (float)hardwallscaleDrude;

    // Execute Part 1: Update normal particles
    if (numNormalParticles > 0) {
        void* args1[] = {
            &velmPtr,
            &forcePtr,
            &posDeltaPtr,
            &normalPtr,
            &pairPtr,
            &stepSizePtr,
            &vscaleF,
            &fscaleF,
            &noisescaleF,
            &vscaleDrudeF,
            &fscaleDrudeF,
            &noisescaleDrudeF,
            &randomPtr,
            &randomIndex,
            &numNormalParticles,
            &numDrudePairs,
            &paddedNumAtoms
        };
        cu.executeKernel(kernel1, args1, numNormalParticles);
    }
    
    // Execute: Update Drude pairs
    if (numDrudePairs > 0) {
        unsigned int pairRandomIndex = randomIndex + numNormalParticles;
        void* argsPairs[] = {
            &velmPtr,
            &forcePtr,
            &posDeltaPtr,
            &pairPtr,
            &stepSizePtr,
            &vscaleF,
            &fscaleF,
            &noisescaleF,
            &vscaleDrudeF,
            &fscaleDrudeF,
            &noisescaleDrudeF,
            &randomPtr,
            &pairRandomIndex,
            &numDrudePairs,
            &paddedNumAtoms
        };
        cu.executeKernel(kernelPairs, argsPairs, numDrudePairs);
    }
    
    // Apply constraints
    integration.applyConstraints(integrator.getConstraintTolerance());
    
    // Execute Part 2: Update positions
    CUdeviceptr posqPtr = cu.getPosq().getDevicePointer();
    void* args2[] = {
        &posqPtr,
        &posDeltaPtr,
        &velmPtr,
        &stepSizePtr,
        &numAtoms
    };
    cu.executeKernel(kernel2, args2, numAtoms);
    
    // Apply hard wall constraints
    if (maxDrudeDistance > 0 && numDrudePairs > 0) {
        void* argsHW[] = {
            &posqPtr,
            &velmPtr,
            &pairPtr,
            &stepSizePtr,
            &maxDrudeDistF,
            &hardwallscaleF,
            &numDrudePairs
        };
        cu.executeKernel(hardwallKernel, argsHW, numDrudePairs);
    }
    
    // Compute virtual sites
    integration.computeVirtualSites();

    // Update time and step count
    stepCount++;
    cu.setTime(cu.getTime() + stepSize);
    cu.setStepCount(cu.getStepCount() + 1);
    cu.reorderAtoms();
}

double CudaIntegrateConstantVDrudeLangevinStepKernel::computeKineticEnergy(
    ContextImpl& context,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    return cu.getIntegrationUtilities().computeKineticEnergy(0.5 * integrator.getStepSize());
}
