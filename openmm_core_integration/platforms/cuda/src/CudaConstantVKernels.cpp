/* -------------------------------------------------------------------------- *
 *                    CUDA ConstantV Kernel Implementation                    *
 * -------------------------------------------------------------------------- */

#include "CudaConstantVKernels.h"
#include "openmm/Context.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/DrudeForce.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/cuda/CudaBondedUtilities.h"
#include "openmm/cuda/CudaIntegrationUtilities.h"
#include <cuda.h>
#include <cuda_runtime.h>

using namespace OpenMM;
using namespace std;

// ═══════════════════════════════════════════════════════════════════════════
// BUG FIX #3: CUDA Error Checking Macro
// ═══════════════════════════════════════════════════════════════════════════

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw OpenMMException(string("CUDA error: ") + cudaGetErrorString(err) + \
                                  string(" at ") + __FILE__ + string(":") + to_string(__LINE__)); \
        } \
    } while (0)

// ═══════════════════════════════════════════════════════════════════════════
// Mirror Struct Definitions from .cu file (MUST MATCH EXACTLY)
// ═══════════════════════════════════════════════════════════════════════════

struct BuckyballData {
    int numAtoms;
    int* virtualIndices;      // Device pointer
    int* realIndices;         // Device pointer
    double* normals;          // Device pointer (3 * numAtoms doubles)
    double area_atom;
    double radius;
    double r_center[3];
    int contactAtomIndex;
    double dr_center_contact;
    double voltage_kjmol;
    char electrodeType;       // 'c' or 'a'
};

struct NanotubeData {
    int numAtoms;
    int* virtualIndices;      // Device pointer
    int* realIndices;         // Device pointer
    double* normals;          // Device pointer (3 * numAtoms doubles)
    double area_atom;
    double axis[3];           // Normalized axis vector
    double r_center[3];
    int contactAtomIndex;
    double dr_axis_contact;
    double voltage_kjmol;
    char electrodeType;       // 'c' or 'a'
};

struct ElectrodeData {
    // Flat electrodes
    int numCathodes;
    int numAnodes;
    int* cathodeIndices;      // Device pointer
    double* cathodeAreas;     // Device pointer
    int* anodeIndices;        // Device pointer
    double* anodeAreas;       // Device pointer

    // Electrolyte
    int numElectrolytes;
    int* electrolyteIndices;  // Device pointer

    // Conductors
    int numBuckyballs;
    BuckyballData* buckyballs;  // Device pointer to array of structs
    int numNanotubes;
    NanotubeData* nanotubes;    // Device pointer to array of structs

    // System parameters
    double voltage_kjmol;
    double Lgap;
    double Lcell;
    double totalArea;
    double z_cathode;
    double z_anode;
};

struct DrudeParticleData {
    int numPairs;
    int numNormalParticles;
    int2* pairParticles;      // Device pointer: (parent, drude)
    int* normalParticles;     // Device pointer
};

// Forward declaration of CUDA kernel (defined in .cu file)
extern "C" void executeConstantVDrudeLangevinStep(
    // System data
    int numAtoms,
    int paddedNumAtoms,
    float4* d_posq,
    float4* d_velm,
    long long* d_force,
    float4* d_posDelta,
    float4* d_random,
    unsigned int randomIndex,

    // Electrode data
    ElectrodeData* d_electrodeData,

    // Drude particle data
    DrudeParticleData* d_drudeData,

    // Integration parameters
    float stepSize,
    float temperature,
    float friction,
    float drudeTemperature,
    float drudeFriction,
    float maxDrudeDistance,
    int scfIterations,

    // Host-side counts (Optimization A: eliminate PCIe roundtrip)
    int numCathodes,
    int numAnodes,
    int numElectrolytes,
    int numBuckyballs,
    int numNanotubes,
    int numDrudePairs,
    int numNormalParticles
);

// ═══════════════════════════════════════════════════════════════════════════
// CudaCalcConstantVKernel
// ═══════════════════════════════════════════════════════════════════════════

CudaCalcConstantVKernel::CudaCalcConstantVKernel(string name, const Platform& platform, CudaContext& cu) :
    CalcConstantVKernel(name, platform),
    cu(cu),
    hasInitialized(false),
    cathodeIndicesGPU(nullptr),
    cathodeAreasGPU(nullptr),
    anodeIndicesGPU(nullptr),
    anodeAreasGPU(nullptr),
    electrolyteIndicesGPU(nullptr),
    electrodeDataGPU(nullptr),
    buckyballDataArrayGPU(nullptr),
    nanotubeDataArrayGPU(nullptr),
    numBuckyballs(0),
    numNanotubes(0),
    numCathodeAtoms(0),
    numAnodeAtoms(0),
    numElectrolyteAtoms(0)
{
}

CudaCalcConstantVKernel::~CudaCalcConstantVKernel() {
    if (cathodeIndicesGPU) delete cathodeIndicesGPU;
    if (cathodeAreasGPU) delete cathodeAreasGPU;
    if (anodeIndicesGPU) delete anodeIndicesGPU;
    if (anodeAreasGPU) delete anodeAreasGPU;
    if (electrolyteIndicesGPU) delete electrolyteIndicesGPU;
    if (electrodeDataGPU) delete electrodeDataGPU;

    // Clean up conductor arrays
    for (CudaArray* arr : conductorArrays)
        delete arr;
    if (buckyballDataArrayGPU) delete buckyballDataArrayGPU;
    if (nanotubeDataArrayGPU) delete nanotubeDataArrayGPU;

    // Clean up host-side structs
    for (void* ptr : buckyballStructsHost)
        delete (BuckyballData*)ptr;
    for (void* ptr : nanotubeStructsHost)
        delete (NanotubeData*)ptr;
}

void CudaCalcConstantVKernel::initialize(
    const System& system,
    const vector<int>& cathodeAtomIndices,
    const vector<double>& cathodeAreas,
    const vector<int>& anodeAtomIndices,
    const vector<double>& anodeAreas,
    const vector<int>& electrolyteAtomIndices,
    const vector<double>& electrolyteCharges,
    double voltage,
    double Lgap,
    double Lcell,
    double totalArea,
    double z_cathode,
    double z_anode,
    int nIterations)
{
    // Store parameters
    this->numCathodeAtoms = cathodeAtomIndices.size();
    this->numAnodeAtoms = anodeAtomIndices.size();
    this->numElectrolyteAtoms = electrolyteAtomIndices.size();
    this->voltage = voltage * 96.487;  // Convert V to kJ/mol
    this->Lgap = Lgap;
    this->Lcell = Lcell;
    this->totalArea = totalArea;
    this->z_cathode = z_cathode;
    this->z_anode = z_anode;
    this->nIterations = nIterations;

    // Allocate GPU arrays for cathode
    if (numCathodeAtoms > 0) {
        cathodeIndicesGPU = new CudaArray(cu, numCathodeAtoms, sizeof(int), "cathodeIndices");
        cathodeAreasGPU = new CudaArray(cu, numCathodeAtoms, sizeof(double), "cathodeAreas");
        cathodeIndicesGPU->upload(cathodeAtomIndices);
        cathodeAreasGPU->upload(cathodeAreas);
    }

    // Allocate GPU arrays for anode
    if (numAnodeAtoms > 0) {
        anodeIndicesGPU = new CudaArray(cu, numAnodeAtoms, sizeof(int), "anodeIndices");
        anodeAreasGPU = new CudaArray(cu, numAnodeAtoms, sizeof(double), "anodeAreas");
        anodeIndicesGPU->upload(anodeAtomIndices);
        anodeAreasGPU->upload(anodeAreas);
    }

    // Allocate GPU arrays for electrolyte
    if (numElectrolyteAtoms > 0) {
        electrolyteIndicesGPU = new CudaArray(cu, numElectrolyteAtoms, sizeof(int), "electrolyteIndices");
        electrolyteIndicesGPU->upload(electrolyteAtomIndices);
    }

    // Create ElectrodeData struct on HOST, populate with DEVICE pointers
    ElectrodeData hostElectrodeData;
    hostElectrodeData.numCathodes = numCathodeAtoms;
    hostElectrodeData.numAnodes = numAnodeAtoms;
    hostElectrodeData.cathodeIndices = (numCathodeAtoms > 0) ?
        (int*)cathodeIndicesGPU->getDevicePointer() : nullptr;
    hostElectrodeData.cathodeAreas = (numCathodeAtoms > 0) ?
        (double*)cathodeAreasGPU->getDevicePointer() : nullptr;
    hostElectrodeData.anodeIndices = (numAnodeAtoms > 0) ?
        (int*)anodeIndicesGPU->getDevicePointer() : nullptr;
    hostElectrodeData.anodeAreas = (numAnodeAtoms > 0) ?
        (double*)anodeAreasGPU->getDevicePointer() : nullptr;
    hostElectrodeData.numElectrolytes = numElectrolyteAtoms;
    hostElectrodeData.electrolyteIndices = (numElectrolyteAtoms > 0) ?
        (int*)electrolyteIndicesGPU->getDevicePointer() : nullptr;

    // Conductors will be added via addBuckyballConductor/addNanotubeConductor
    hostElectrodeData.numBuckyballs = 0;
    hostElectrodeData.buckyballs = nullptr;
    hostElectrodeData.numNanotubes = 0;
    hostElectrodeData.nanotubes = nullptr;

    hostElectrodeData.voltage_kjmol = this->voltage;
    hostElectrodeData.Lgap = Lgap;
    hostElectrodeData.Lcell = Lcell;
    hostElectrodeData.totalArea = totalArea;
    hostElectrodeData.z_cathode = z_cathode;
    hostElectrodeData.z_anode = z_anode;

    // Allocate ElectrodeData struct on DEVICE and upload
    electrodeDataGPU = new CudaArray(cu, 1, sizeof(ElectrodeData), "electrodeData");
    electrodeDataGPU->upload(&hostElectrodeData, 1);

    hasInitialized = true;
}

void CudaCalcConstantVKernel::addBuckyballConductor(
    const vector<int>& virtualAtomIndices,
    const vector<int>& realAtomIndices,
    const string& electrodeType,
    double voltage,
    const Vec3& center,
    double radius,
    const vector<Vec3>& normalVectors,
    double areaPerAtom,
    int contactAtomIndex,
    double contactDistance)
{
    int numAtoms = virtualAtomIndices.size();

    // ═══════════════════════════════════════════════════════════════════════
    // Step 1-3: Allocate CudaArrays for virtualIndices, realIndices, normals
    // ═══════════════════════════════════════════════════════════════════════

    CudaArray* virtualIndicesGPU = new CudaArray(cu, numAtoms, sizeof(int), "buckyball_virtualIndices");
    CudaArray* realIndicesGPU = new CudaArray(cu, numAtoms, sizeof(int), "buckyball_realIndices");
    CudaArray* normalsGPU = new CudaArray(cu, numAtoms * 3, sizeof(double), "buckyball_normals");

    // Upload data
    virtualIndicesGPU->upload(virtualAtomIndices);
    realIndicesGPU->upload(realAtomIndices);

    // Convert normalVectors (Vec3) to flat double array
    vector<double> normalsFlat(numAtoms * 3);
    for (int i = 0; i < numAtoms; i++) {
        normalsFlat[i * 3 + 0] = normalVectors[i][0];
        normalsFlat[i * 3 + 1] = normalVectors[i][1];
        normalsFlat[i * 3 + 2] = normalVectors[i][2];
    }
    normalsGPU->upload(normalsFlat);

    // Store for cleanup
    conductorArrays.push_back(virtualIndicesGPU);
    conductorArrays.push_back(realIndicesGPU);
    conductorArrays.push_back(normalsGPU);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 4: Create BuckyballData struct on HOST with DEVICE pointers
    // ═══════════════════════════════════════════════════════════════════════

    BuckyballData* hostStruct = new BuckyballData();
    hostStruct->numAtoms = numAtoms;
    hostStruct->virtualIndices = (int*)virtualIndicesGPU->getDevicePointer();  // DEVICE POINTER!
    hostStruct->realIndices = (int*)realIndicesGPU->getDevicePointer();        // DEVICE POINTER!
    hostStruct->normals = (double*)normalsGPU->getDevicePointer();              // DEVICE POINTER!
    hostStruct->area_atom = areaPerAtom;
    hostStruct->radius = radius;
    hostStruct->r_center[0] = center[0];
    hostStruct->r_center[1] = center[1];
    hostStruct->r_center[2] = center[2];
    hostStruct->contactAtomIndex = contactAtomIndex;
    hostStruct->dr_center_contact = contactDistance;
    hostStruct->voltage_kjmol = voltage * 96.487;  // V to kJ/mol
    hostStruct->electrodeType = (electrodeType == "cathode") ? 'c' : 'a';

    // ═══════════════════════════════════════════════════════════════════════
    // Step 5: Store pointer to struct in host-side vector
    // ═══════════════════════════════════════════════════════════════════════

    buckyballStructsHost.push_back((void*)hostStruct);
    numBuckyballs++;
}

void CudaCalcConstantVKernel::addNanotubeConductor(
    const vector<int>& virtualAtomIndices,
    const vector<int>& realAtomIndices,
    const string& electrodeType,
    double voltage,
    const Vec3& center,
    const Vec3& axis,
    double radius,
    double length,
    const vector<Vec3>& normalVectors,
    double areaPerAtom,
    int contactAtomIndex,
    double contactDistance)
{
    int numAtoms = virtualAtomIndices.size();

    // ═══════════════════════════════════════════════════════════════════════
    // Step 1-3: Allocate CudaArrays for virtualIndices, realIndices, normals
    // ═══════════════════════════════════════════════════════════════════════

    CudaArray* virtualIndicesGPU = new CudaArray(cu, numAtoms, sizeof(int), "nanotube_virtualIndices");
    CudaArray* realIndicesGPU = new CudaArray(cu, numAtoms, sizeof(int), "nanotube_realIndices");
    CudaArray* normalsGPU = new CudaArray(cu, numAtoms * 3, sizeof(double), "nanotube_normals");

    // Upload data
    virtualIndicesGPU->upload(virtualAtomIndices);
    realIndicesGPU->upload(realAtomIndices);

    // Convert normalVectors (Vec3) to flat double array
    vector<double> normalsFlat(numAtoms * 3);
    for (int i = 0; i < numAtoms; i++) {
        normalsFlat[i * 3 + 0] = normalVectors[i][0];
        normalsFlat[i * 3 + 1] = normalVectors[i][1];
        normalsFlat[i * 3 + 2] = normalVectors[i][2];
    }
    normalsGPU->upload(normalsFlat);

    // Store for cleanup
    conductorArrays.push_back(virtualIndicesGPU);
    conductorArrays.push_back(realIndicesGPU);
    conductorArrays.push_back(normalsGPU);

    // ═══════════════════════════════════════════════════════════════════════
    // Step 4: Create NanotubeData struct on HOST with DEVICE pointers
    // ═══════════════════════════════════════════════════════════════════════

    NanotubeData* hostStruct = new NanotubeData();
    hostStruct->numAtoms = numAtoms;
    hostStruct->virtualIndices = (int*)virtualIndicesGPU->getDevicePointer();  // DEVICE POINTER!
    hostStruct->realIndices = (int*)realIndicesGPU->getDevicePointer();        // DEVICE POINTER!
    hostStruct->normals = (double*)normalsGPU->getDevicePointer();              // DEVICE POINTER!
    hostStruct->area_atom = areaPerAtom;
    hostStruct->axis[0] = axis[0];
    hostStruct->axis[1] = axis[1];
    hostStruct->axis[2] = axis[2];
    hostStruct->r_center[0] = center[0];
    hostStruct->r_center[1] = center[1];
    hostStruct->r_center[2] = center[2];
    hostStruct->contactAtomIndex = contactAtomIndex;
    hostStruct->dr_axis_contact = contactDistance;
    hostStruct->voltage_kjmol = voltage * 96.487;  // V to kJ/mol
    hostStruct->electrodeType = (electrodeType == "cathode") ? 'c' : 'a';

    // ═══════════════════════════════════════════════════════════════════════
    // Step 5: Store pointer to struct in host-side vector
    // ═══════════════════════════════════════════════════════════════════════

    nanotubeStructsHost.push_back((void*)hostStruct);
    numNanotubes++;
}

/**
 * BUG FIX #2: Helper method to upload ElectrodeData to GPU
 *
 * This method ensures that conductor data added via addBuckyballConductor()
 * or addNanotubeConductor() is properly uploaded to GPU memory.
 *
 * Call this:
 *   - During initialize() (for initial upload)
 *   - In execute() (if conductors were added after initialize())
 *   - In updateParameters() (when parameters change)
 */
void CudaCalcConstantVKernel::uploadElectrodeDataToGPU() {
    // Upload Buckyball array of structs
    if (numBuckyballs > 0 && buckyballDataArrayGPU == nullptr) {
        // Convert void* pointers to BuckyballData* and create vector
        vector<BuckyballData> buckyballsVec;
        buckyballsVec.reserve(numBuckyballs);
        for (void* ptr : buckyballStructsHost) {
            buckyballsVec.push_back(*((BuckyballData*)ptr));
        }

        // Allocate GPU array for BuckyballData structs
        buckyballDataArrayGPU = new CudaArray(cu, numBuckyballs, sizeof(BuckyballData), "buckyballDataArray");
        buckyballDataArrayGPU->upload(buckyballsVec);
    }

    // Upload Nanotube array of structs
    if (numNanotubes > 0 && nanotubeDataArrayGPU == nullptr) {
        // Convert void* pointers to NanotubeData* and create vector
        vector<NanotubeData> nanotubesVec;
        nanotubesVec.reserve(numNanotubes);
        for (void* ptr : nanotubeStructsHost) {
            nanotubesVec.push_back(*((NanotubeData*)ptr));
        }

        // Allocate GPU array for NanotubeData structs
        nanotubeDataArrayGPU = new CudaArray(cu, numNanotubes, sizeof(NanotubeData), "nanotubeDataArray");
        nanotubeDataArrayGPU->upload(nanotubesVec);
    }

    // Update ElectrodeData struct on GPU (with conductor pointers)
    ElectrodeData hostElectrodeData;
    hostElectrodeData.numCathodes = numCathodeAtoms;
    hostElectrodeData.numAnodes = numAnodeAtoms;
    hostElectrodeData.cathodeIndices = (numCathodeAtoms > 0) ?
        (int*)cathodeIndicesGPU->getDevicePointer() : nullptr;
    hostElectrodeData.cathodeAreas = (numCathodeAtoms > 0) ?
        (double*)cathodeAreasGPU->getDevicePointer() : nullptr;
    hostElectrodeData.anodeIndices = (numAnodeAtoms > 0) ?
        (int*)anodeIndicesGPU->getDevicePointer() : nullptr;
    hostElectrodeData.anodeAreas = (numAnodeAtoms > 0) ?
        (double*)anodeAreasGPU->getDevicePointer() : nullptr;
    hostElectrodeData.numElectrolytes = numElectrolyteAtoms;
    hostElectrodeData.electrolyteIndices = (numElectrolyteAtoms > 0) ?
        (int*)electrolyteIndicesGPU->getDevicePointer() : nullptr;

    // Point to conductor arrays on GPU
    hostElectrodeData.numBuckyballs = numBuckyballs;
    hostElectrodeData.buckyballs = (numBuckyballs > 0) ?
        (BuckyballData*)buckyballDataArrayGPU->getDevicePointer() : nullptr;
    hostElectrodeData.numNanotubes = numNanotubes;
    hostElectrodeData.nanotubes = (numNanotubes > 0) ?
        (NanotubeData*)nanotubeDataArrayGPU->getDevicePointer() : nullptr;

    // System parameters
    hostElectrodeData.voltage_kjmol = voltage;
    hostElectrodeData.Lgap = Lgap;
    hostElectrodeData.Lcell = Lcell;
    hostElectrodeData.totalArea = totalArea;
    hostElectrodeData.z_cathode = z_cathode;
    hostElectrodeData.z_anode = z_anode;

    // Upload to GPU
    electrodeDataGPU->upload(&hostElectrodeData, 1);
}

double CudaCalcConstantVKernel::execute(ContextImpl& context, bool includeForces,
                                        bool includeEnergy, int groups)
{
    if (!hasInitialized)
        throw OpenMMException("CudaCalcConstantVKernel::execute() called before initialize()");

    // BUG FIX #2: Check if conductors were added but not uploaded
    if ((numBuckyballs > 0 && buckyballDataArrayGPU == nullptr) ||
        (numNanotubes > 0 && nanotubeDataArrayGPU == nullptr)) {
        uploadElectrodeDataToGPU();
    }

    // For Force-based API, we don't execute the full integration kernel
    // This would require implementing a separate SCF-only kernel
    // For now, return 0.0 (the integration kernel handles SCF)
    return 0.0;
}

void CudaCalcConstantVKernel::updateParameters(ContextImpl& context, const ConstantVForce& force)
{
    voltage = force.getVoltage() * 96.487;
    Lgap = force.getLgap();
    Lcell = force.getLcell();
    totalArea = force.getTotalArea();
    z_cathode = force.getZCathode();
    z_anode = force.getZAnode();
    nIterations = force.getNumIterations();

    // BUG FIX #2: Use helper method to upload electrode data
    uploadElectrodeDataToGPU();
}

// ═══════════════════════════════════════════════════════════════════════════
// CudaIntegrateConstantVDrudeLangevinStepKernel
// ═══════════════════════════════════════════════════════════════════════════

CudaIntegrateConstantVDrudeLangevinStepKernel::CudaIntegrateConstantVDrudeLangevinStepKernel(
    string name, const Platform& platform, CudaContext& cu) :
    KernelImpl(name, platform),
    cu(cu),
    hasInitialized(false),
    cathodeIndicesGPU(nullptr),
    cathodeAreasGPU(nullptr),
    anodeIndicesGPU(nullptr),
    anodeAreasGPU(nullptr),
    electrolyteIndicesGPU(nullptr),
    electrodeDataGPU(nullptr),
    pairParticlesGPU(nullptr),
    normalParticlesGPU(nullptr),
    drudeDataGPU(nullptr),
    posDeltaGPU(nullptr),
    stepCount(0)
{
}

CudaIntegrateConstantVDrudeLangevinStepKernel::~CudaIntegrateConstantVDrudeLangevinStepKernel() {
    if (cathodeIndicesGPU) delete cathodeIndicesGPU;
    if (cathodeAreasGPU) delete cathodeAreasGPU;
    if (anodeIndicesGPU) delete anodeIndicesGPU;
    if (anodeAreasGPU) delete anodeAreasGPU;
    if (electrolyteIndicesGPU) delete electrolyteIndicesGPU;
    if (electrodeDataGPU) delete electrodeDataGPU;
    if (pairParticlesGPU) delete pairParticlesGPU;
    if (normalParticlesGPU) delete normalParticlesGPU;
    if (drudeDataGPU) delete drudeDataGPU;
    if (posDeltaGPU) delete posDeltaGPU;
}

void CudaIntegrateConstantVDrudeLangevinStepKernel::initialize(
    const System& system,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    // Get electrode data from integrator
    numCathodeAtoms = integrator.getNumCathodeAtoms();
    numAnodeAtoms = integrator.getNumAnodeAtoms();
    numElectrolyteAtoms = integrator.getNumElectrolyteAtoms();

    voltage = integrator.getVoltage() * 96.487;  // V to kJ/mol
    Lgap = integrator.getLgap();
    Lcell = integrator.getLcell();
    totalArea = integrator.getTotalArea();
    z_cathode = integrator.getZCathode();
    z_anode = integrator.getZAnode();
    scfIterations = integrator.getNumSCFIterations();
    scfFrequency = integrator.getSCFFrequency();
    maxDrudeDistance = integrator.getMaxDrudeDistance();

    // Allocate GPU memory for cathode atoms
    if (numCathodeAtoms > 0) {
        vector<int> cathodeIndices;
        vector<double> cathodeAreas;
        cathodeIndices.reserve(numCathodeAtoms);
        cathodeAreas.reserve(numCathodeAtoms);

        for (int i = 0; i < numCathodeAtoms; i++) {
            int particle;
            double area;
            integrator.getCathodeAtomParameters(i, particle, area);
            cathodeIndices.push_back(particle);
            cathodeAreas.push_back(area);
        }

        cathodeIndicesGPU = new CudaArray(cu, numCathodeAtoms, sizeof(int), "cathodeIndices");
        cathodeAreasGPU = new CudaArray(cu, numCathodeAtoms, sizeof(double), "cathodeAreas");
        cathodeIndicesGPU->upload(cathodeIndices);
        cathodeAreasGPU->upload(cathodeAreas);
    }

    // Allocate GPU memory for anode atoms
    if (numAnodeAtoms > 0) {
        vector<int> anodeIndices;
        vector<double> anodeAreas;
        anodeIndices.reserve(numAnodeAtoms);
        anodeAreas.reserve(numAnodeAtoms);

        for (int i = 0; i < numAnodeAtoms; i++) {
            int particle;
            double area;
            integrator.getAnodeAtomParameters(i, particle, area);
            anodeIndices.push_back(particle);
            anodeAreas.push_back(area);
        }

        anodeIndicesGPU = new CudaArray(cu, numAnodeAtoms, sizeof(int), "anodeIndices");
        anodeAreasGPU = new CudaArray(cu, numAnodeAtoms, sizeof(double), "anodeAreas");
        anodeIndicesGPU->upload(anodeIndices);
        anodeAreasGPU->upload(anodeAreas);
    }

    // Allocate GPU memory for electrolyte atoms
    if (numElectrolyteAtoms > 0) {
        vector<int> electrolyteIndices;
        electrolyteIndices.reserve(numElectrolyteAtoms);

        for (int i = 0; i < numElectrolyteAtoms; i++) {
            int particle;
            double charge;
            integrator.getElectrolyteAtomParameters(i, particle, charge);
            electrolyteIndices.push_back(particle);
        }

        electrolyteIndicesGPU = new CudaArray(cu, numElectrolyteAtoms, sizeof(int), "electrolyteIndices");
        electrolyteIndicesGPU->upload(electrolyteIndices);
    }

    // Create ElectrodeData struct on HOST, populate with DEVICE pointers
    ElectrodeData hostElectrodeData;
    hostElectrodeData.numCathodes = numCathodeAtoms;
    hostElectrodeData.numAnodes = numAnodeAtoms;
    hostElectrodeData.cathodeIndices = (numCathodeAtoms > 0) ?
        (int*)cathodeIndicesGPU->getDevicePointer() : nullptr;
    hostElectrodeData.cathodeAreas = (numCathodeAtoms > 0) ?
        (double*)cathodeAreasGPU->getDevicePointer() : nullptr;
    hostElectrodeData.anodeIndices = (numAnodeAtoms > 0) ?
        (int*)anodeIndicesGPU->getDevicePointer() : nullptr;
    hostElectrodeData.anodeAreas = (numAnodeAtoms > 0) ?
        (double*)anodeAreasGPU->getDevicePointer() : nullptr;
    hostElectrodeData.numElectrolytes = numElectrolyteAtoms;
    hostElectrodeData.electrolyteIndices = (numElectrolyteAtoms > 0) ?
        (int*)electrolyteIndicesGPU->getDevicePointer() : nullptr;
    hostElectrodeData.numBuckyballs = 0;
    hostElectrodeData.buckyballs = nullptr;
    hostElectrodeData.numNanotubes = 0;
    hostElectrodeData.nanotubes = nullptr;
    hostElectrodeData.voltage_kjmol = voltage;
    hostElectrodeData.Lgap = Lgap;
    hostElectrodeData.Lcell = Lcell;
    hostElectrodeData.totalArea = totalArea;
    hostElectrodeData.z_cathode = z_cathode;
    hostElectrodeData.z_anode = z_anode;

    // Allocate ElectrodeData struct on DEVICE and upload
    electrodeDataGPU = new CudaArray(cu, 1, sizeof(ElectrodeData), "electrodeData");
    electrodeDataGPU->upload(&hostElectrodeData, 1);

    // Extract Drude particle information from System
    const DrudeForce* drudeForce = nullptr;
    for (int i = 0; i < system.getNumForces(); i++) {
        if (dynamic_cast<const DrudeForce*>(&system.getForce(i)) != nullptr) {
            drudeForce = dynamic_cast<const DrudeForce*>(&system.getForce(i));
            break;
        }
    }

    if (drudeForce != nullptr) {
        // Extract Drude pairs
        numDrudePairs = drudeForce->getNumParticles();
        vector<int2> pairParticles;
        pairParticles.reserve(numDrudePairs);

        for (int i = 0; i < numDrudePairs; i++) {
            int parent, drude, dummy1, dummy2, dummy3;
            double dummy4, dummy5, dummy6;
            drudeForce->getParticleParameters(i, parent, drude, dummy1, dummy2, dummy3, dummy4, dummy5, dummy6);
            int2 pair;
            pair.x = parent;
            pair.y = drude;
            pairParticles.push_back(pair);
        }

        // Identify normal particles (not parent, not drude)
        vector<bool> isDrudeParticle(system.getNumParticles(), false);
        for (const auto& pair : pairParticles) {
            isDrudeParticle[pair.x] = true;  // parent
            isDrudeParticle[pair.y] = true;  // drude
        }

        vector<int> normalParticles;
        for (int i = 0; i < system.getNumParticles(); i++) {
            if (!isDrudeParticle[i])
                normalParticles.push_back(i);
        }
        numNormalParticles = normalParticles.size();

        // Allocate and upload Drude data
        if (numDrudePairs > 0) {
            pairParticlesGPU = new CudaArray(cu, numDrudePairs, sizeof(int2), "pairParticles");
            pairParticlesGPU->upload(pairParticles);
        }

        if (numNormalParticles > 0) {
            normalParticlesGPU = new CudaArray(cu, numNormalParticles, sizeof(int), "normalParticles");
            normalParticlesGPU->upload(normalParticles);
        }

        // Create DrudeParticleData struct on HOST
        DrudeParticleData hostDrudeData;
        hostDrudeData.numPairs = numDrudePairs;
        hostDrudeData.numNormalParticles = numNormalParticles;
        hostDrudeData.pairParticles = (numDrudePairs > 0) ?
            (int2*)pairParticlesGPU->getDevicePointer() : nullptr;
        hostDrudeData.normalParticles = (numNormalParticles > 0) ?
            (int*)normalParticlesGPU->getDevicePointer() : nullptr;

        // Allocate DrudeParticleData struct on DEVICE and upload
        drudeDataGPU = new CudaArray(cu, 1, sizeof(DrudeParticleData), "drudeData");
        drudeDataGPU->upload(&hostDrudeData, 1);
    } else {
        // No Drude particles - create empty struct
        numDrudePairs = 0;
        numNormalParticles = system.getNumParticles();

        vector<int> normalParticles;
        for (int i = 0; i < numNormalParticles; i++)
            normalParticles.push_back(i);

        normalParticlesGPU = new CudaArray(cu, numNormalParticles, sizeof(int), "normalParticles");
        normalParticlesGPU->upload(normalParticles);

        DrudeParticleData hostDrudeData;
        hostDrudeData.numPairs = 0;
        hostDrudeData.numNormalParticles = numNormalParticles;
        hostDrudeData.pairParticles = nullptr;
        hostDrudeData.normalParticles = (int*)normalParticlesGPU->getDevicePointer();

        drudeDataGPU = new CudaArray(cu, 1, sizeof(DrudeParticleData), "drudeData");
        drudeDataGPU->upload(&hostDrudeData, 1);
    }

    // Allocate posDelta array for integration
    posDeltaGPU = new CudaArray(cu, cu.getPaddedNumAtoms(), 4*sizeof(float), "posDelta");

    hasInitialized = true;
}

void CudaIntegrateConstantVDrudeLangevinStepKernel::execute(
    ContextImpl& context,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    if (!hasInitialized)
        throw OpenMMException("CudaIntegrateConstantVDrudeLangevinStepKernel::execute() called before initialize()");

    // Get pointers to GPU arrays from CudaContext
    float4* d_posq = (float4*)cu.getPosq().getDevicePointer();
    float4* d_velm = (float4*)cu.getVelm().getDevicePointer();
    long long* d_force = (long long*)cu.getForce().getDevicePointer();
    float4* d_posDelta = (float4*)posDeltaGPU->getDevicePointer();
    float4* d_random = (float4*)cu.getIntegrationUtilities().getRandom().getDevicePointer();
    unsigned int randomIndex = cu.getIntegrationUtilities().prepareRandomNumbers(cu.getPaddedNumAtoms());

    ElectrodeData* d_electrodeData = (ElectrodeData*)electrodeDataGPU->getDevicePointer();
    DrudeParticleData* d_drudeData = (DrudeParticleData*)drudeDataGPU->getDevicePointer();

    // Call CUDA kernel (Optimization A: pass counts from host to eliminate PCIe roundtrip)
    executeConstantVDrudeLangevinStep(
        cu.getNumAtoms(),
        cu.getPaddedNumAtoms(),
        d_posq,
        d_velm,
        d_force,
        d_posDelta,
        d_random,
        randomIndex,
        d_electrodeData,
        d_drudeData,
        (float)integrator.getStepSize(),
        (float)integrator.getTemperature(),
        (float)integrator.getFriction(),
        (float)integrator.getDrudeTemperature(),
        (float)integrator.getDrudeFriction(),
        (float)maxDrudeDistance,
        scfIterations,
        // Host-side counts (eliminates cudaMemcpy)
        numCathodeAtoms,
        numAnodeAtoms,
        numElectrolyteAtoms,
        0,  // numBuckyballs (TODO: conductor support not implemented yet)
        0,  // numNanotubes (TODO: conductor support not implemented yet)
        numDrudePairs,
        numNormalParticles
    );

    // BUG FIX #3: Comprehensive error checking
    // Check for asynchronous errors (kernel launch failures)
    CUDA_CHECK(cudaGetLastError());

    // Check for synchronous errors (kernel execution failures)
    CUDA_CHECK(cudaDeviceSynchronize());

    stepCount++;
}
