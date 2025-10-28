#include "ElectrodeChargeKernels.h"
#include "openmm/Platform.h"
#include "openmm/internal/windowsExport.h"
#include <exception>

#include "ReferenceElectrodeChargeKernelFactory.h"
#include "CudaElectrodeChargeKernelFactory.h"

using namespace ElectrodeChargePlugin;
using namespace OpenMM;

extern "C" OPENMM_EXPORT void registerElectrodeChargePlugin() {
    try {
        Platform& reference = Platform::getPlatformByName("Reference");
        reference.registerKernelFactory(CalcElectrodeChargeKernel::Name(), new ReferenceElectrodeChargeKernelFactory());
    }
    catch (std::exception&) {
        // Reference platform might not be available.
    }

    try {
        Platform& cuda = Platform::getPlatformByName("CUDA");
        cuda.registerKernelFactory(CalcElectrodeChargeKernel::Name(), new CudaElectrodeChargeKernelFactory());
    }
    catch (std::exception&) {
        // CUDA platform might not be available.
    }
}
