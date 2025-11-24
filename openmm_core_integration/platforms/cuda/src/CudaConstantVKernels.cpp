/* -------------------------------------------------------------------------- *
 *                    CUDA ConstantV Kernel Implementation                    *
 * -------------------------------------------------------------------------- */

#include "CudaConstantVKernels.h"
#include "openmm/Context.h"
#include "openmm/internal/ContextImpl.h"
#include "openmm/cuda/CudaForceInfo.h"
#include "openmm/cuda/CudaBondedUtilities.h"

using namespace OpenMM;
using namespace std;

// Forward declaration of CUDA kernel (defined in .cu file)
extern "C" void launchConstantVSCFKernel(
    int numAtoms,
    int numCathodes,
    int numAnodes,
    int numElectrolytes,
    const int* cathodeIndices,
    const double* cathodeAreas,
    const int* anodeIndices,
    const double* anodeAreas,
    const int* electrolyteIndices,
    const double* electrolyteCharges,
    double* cathodeCharges,
    double* anodeCharges,
    const float4* posq,
    double voltage_kjmol,
    double Lgap,
    double Lcell,
    double totalArea,
    double z_cathode,
    double z_anode,
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
    electrolyteChargesGPU(nullptr),
    cathodeChargesGPU(nullptr),
    anodeChargesGPU(nullptr),
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
    if (electrolyteChargesGPU) delete electrolyteChargesGPU;
    if (cathodeChargesGPU) delete cathodeChargesGPU;
    if (anodeChargesGPU) delete anodeChargesGPU;
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
        cathodeChargesGPU = new CudaArray(cu, numCathodeAtoms, sizeof(double), "cathodeCharges");

        cathodeIndicesGPU->upload(cathodeAtomIndices);
        cathodeAreasGPU->upload(cathodeAreas);

        // Initialize charges to zero
        vector<double> zeroCharges(numCathodeAtoms, 0.0);
        cathodeChargesGPU->upload(zeroCharges);
    }

    // Allocate GPU arrays for anode
    if (numAnodeAtoms > 0) {
        anodeIndicesGPU = new CudaArray(cu, numAnodeAtoms, sizeof(int), "anodeIndices");
        anodeAreasGPU = new CudaArray(cu, numAnodeAtoms, sizeof(double), "anodeAreas");
        anodeChargesGPU = new CudaArray(cu, numAnodeAtoms, sizeof(double), "anodeCharges");

        anodeIndicesGPU->upload(anodeAtomIndices);
        anodeAreasGPU->upload(anodeAreas);

        vector<double> zeroCharges(numAnodeAtoms, 0.0);
        anodeChargesGPU->upload(zeroCharges);
    }

    // Allocate GPU arrays for electrolyte
    if (numElectrolyteAtoms > 0) {
        electrolyteIndicesGPU = new CudaArray(cu, numElectrolyteAtoms, sizeof(int), "electrolyteIndices");
        electrolyteChargesGPU = new CudaArray(cu, numElectrolyteAtoms, sizeof(double), "electrolyteCharges");

        electrolyteIndicesGPU->upload(electrolyteAtomIndices);
        electrolyteChargesGPU->upload(electrolyteCharges);
    }

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
    // TODO: Implement Buckyball conductor support in CUDA
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
    // TODO: Implement Nanotube conductor support in CUDA
    throw OpenMMException("Nanotube conductors not yet implemented in CUDA platform");
}

double CudaCalcConstantVKernel::execute(ContextImpl& context, bool includeForces,
                                        bool includeEnergy, int groups)
{
    if (!hasInitialized)
        throw OpenMMException("CudaCalcConstantVKernel::execute() called before initialize()");

    // Get position array from context
    const CudaArray& posq = cu.getPosq();

    // Launch SCF kernel (this is a simplified version - full implementation would be in .cu file)
    // NOTE: This requires implementing launchConstantVSCFKernel in the .cu file
    if (numCathodeAtoms > 0 || numAnodeAtoms > 0) {
        // For now, just return 0.0 since the full kernel integration requires more work
        // The actual implementation would call the CUDA kernel here
    }

    return 0.0;  // Return electrostatic energy
}

void CudaCalcConstantVKernel::updateParameters(ContextImpl& context, const ConstantVForce& force)
{
    // Re-upload parameters if they changed
    voltage = force.getVoltage() * 96.487;
    Lgap = force.getLgap();
    Lcell = force.getLcell();
    totalArea = force.getTotalArea();
    z_cathode = force.getZCathode();
    z_anode = force.getZAnode();
    nIterations = force.getNumIterations();
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
    electrolyteChargesGPU(nullptr),
    cathodeChargesGPU(nullptr),
    anodeChargesGPU(nullptr),
    stepCount(0)
{
}

CudaIntegrateConstantVDrudeLangevinStepKernel::~CudaIntegrateConstantVDrudeLangevinStepKernel() {
    if (cathodeIndicesGPU) delete cathodeIndicesGPU;
    if (cathodeAreasGPU) delete cathodeAreasGPU;
    if (anodeIndicesGPU) delete anodeIndicesGPU;
    if (anodeAreasGPU) delete anodeAreasGPU;
    if (electrolyteIndicesGPU) delete electrolyteIndicesGPU;
    if (electrolyteChargesGPU) delete electrolyteChargesGPU;
    if (cathodeChargesGPU) delete cathodeChargesGPU;
    if (anodeChargesGPU) delete anodeChargesGPU;
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
        cathodeChargesGPU = new CudaArray(cu, numCathodeAtoms, sizeof(double), "cathodeCharges");

        cathodeIndicesGPU->upload(cathodeIndices);
        cathodeAreasGPU->upload(cathodeAreas);

        vector<double> zeroCharges(numCathodeAtoms, 0.0);
        cathodeChargesGPU->upload(zeroCharges);
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
        anodeChargesGPU = new CudaArray(cu, numAnodeAtoms, sizeof(double), "anodeCharges");

        anodeIndicesGPU->upload(anodeIndices);
        anodeAreasGPU->upload(anodeAreas);

        vector<double> zeroCharges(numAnodeAtoms, 0.0);
        anodeChargesGPU->upload(zeroCharges);
    }

    // Allocate GPU memory for electrolyte atoms
    if (numElectrolyteAtoms > 0) {
        vector<int> electrolyteIndices;
        vector<double> electrolyteCharges;
        electrolyteIndices.reserve(numElectrolyteAtoms);
        electrolyteCharges.reserve(numElectrolyteAtoms);

        for (int i = 0; i < numElectrolyteAtoms; i++) {
            int particle;
            double charge;
            integrator.getElectrolyteAtomParameters(i, particle, charge);
            electrolyteIndices.push_back(particle);
            electrolyteCharges.push_back(charge);
        }

        electrolyteIndicesGPU = new CudaArray(cu, numElectrolyteAtoms, sizeof(int), "electrolyteIndices");
        electrolyteChargesGPU = new CudaArray(cu, numElectrolyteAtoms, sizeof(double), "electrolyteCharges");

        electrolyteIndicesGPU->upload(electrolyteIndices);
        electrolyteChargesGPU->upload(electrolyteCharges);
    }

    hasInitialized = true;
}

void CudaIntegrateConstantVDrudeLangevinStepKernel::execute(
    ContextImpl& context,
    const ConstantVDrudeLangevinIntegrator& integrator)
{
    if (!hasInitialized)
        throw OpenMMException("CudaIntegrateConstantVDrudeLangevinStepKernel::execute() called before initialize()");

    // NOTE: This is a placeholder implementation
    // The full implementation would:
    // 1. Check if stepCount % scfFrequency == 0
    // 2. If yes, launch SCF kernel to update electrode charges
    // 3. Call parent DrudeLangevinIntegrator's CUDA kernel for integration
    // 4. Increment stepCount

    // For now, just call the parent integrator's step
    // (This requires proper kernel registration which is done in the factory)

    stepCount++;
}
