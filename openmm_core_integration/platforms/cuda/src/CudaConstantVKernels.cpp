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
// Mirror Struct Definitions from .cu file (MUST MATCH EXACTLY)
// ═══════════════════════════════════════════════════════════════════════════

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

    // Conductors (not yet implemented)
    int numBuckyballs;
    void* buckyballs;         // Device pointer (placeholder)
    int numNanotubes;
    void* nanotubes;          // Device pointer (placeholder)

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
    int scfIterations
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
    throw OpenMMException("Buckyball conductors not yet implemented in CUDA platform");
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
    throw OpenMMException("Nanotube conductors not yet implemented in CUDA platform");
}

double CudaCalcConstantVKernel::execute(ContextImpl& context, bool includeForces,
                                        bool includeEnergy, int groups)
{
    if (!hasInitialized)
        throw OpenMMException("CudaCalcConstantVKernel::execute() called before initialize()");

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

    // Update ElectrodeData struct on GPU
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

    electrodeDataGPU->upload(&hostElectrodeData, 1);
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

    // Call CUDA kernel
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
        scfIterations
    );

    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw OpenMMException("CUDA error in ConstantVDrudeLangevinStep: " +
                            string(cudaGetErrorString(err)));
    }

    stepCount++;
}
