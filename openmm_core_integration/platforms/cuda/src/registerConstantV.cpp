/* -------------------------------------------------------------------------- *
 *                         OpenMM Native Core                                 *
 * -------------------------------------------------------------------------- *
 * Plugin Registration Entry Point                                            *
 * -------------------------------------------------------------------------- *
 * This file provides the entry point for OpenMM to discover and load the    *
 * ConstantV integration at runtime.                                          *
 * -------------------------------------------------------------------------- */

#include "openmm/internal/ConstantVKernelFactory.h"
#include "openmm/ConstantVKernels.h"
#include "openmm/Platform.h"
#include "openmm/OpenMMException.h"
#include "openmm/internal/windowsExport.h"

using namespace OpenMM;

/**
 * Entry point for plugin registration.
 *
 * This function is called by OpenMM's PluginInitializer when the library
 * is loaded. It registers the ConstantVKernelFactory with all available
 * platforms.
 *
 * IMPORTANT: This must be declared as extern "C" to prevent name mangling.
 */
extern "C" OPENMM_EXPORT void registerConstantVPlugin() {
    // Create the kernel factory
    ConstantVKernelFactory* factory = new ConstantVKernelFactory();

    // Register with all available platforms
    // The factory will check platform names in createKernelImpl()

    try {
        // Register with CUDA platform if available
        Platform& cudaPlatform = Platform::getPlatformByName("CUDA");
        cudaPlatform.registerKernelFactory(CalcConstantVKernel::Name(), factory);
        cudaPlatform.registerKernelFactory("IntegrateConstantVDrudeLangevinStep", factory);
    } catch (const std::exception&) {
        // CUDA platform not available, skip
    }

    try {
        // Register with Reference platform if available
        Platform& refPlatform = Platform::getPlatformByName("Reference");
        refPlatform.registerKernelFactory(CalcConstantVKernel::Name(), factory);
        refPlatform.registerKernelFactory("IntegrateConstantVDrudeLangevinStep", factory);
    } catch (const std::exception&) {
        // Reference platform not available, skip
    }

    try {
        // Register with OpenCL platform if available (not yet implemented)
        Platform& clPlatform = Platform::getPlatformByName("OpenCL");
        (void)clPlatform;  // Suppress unused variable warning
        // TODO: Add OpenCL kernels
    } catch (const std::exception&) {
        // OpenCL platform not available, skip
    }
}

/**
 * Alternative initialization function for static linking.
 *
 * If the library is statically linked rather than loaded as a plugin,
 * this function should be called explicitly during initialization.
 */
extern "C" OPENMM_EXPORT void registerConstantVKernelFactories() {
    registerConstantVPlugin();
}

/**
 * Plugin initialization marker.
 *
 * OpenMM's PluginInitializer looks for this symbol to identify plugin libraries.
 * The actual initialization is performed by registerConstantVPlugin().
 */
extern "C" OPENMM_EXPORT void registerPlatforms() {
    // This function is called by OpenMM's PluginInitializer
    // Delegate to our main registration function
    registerConstantVPlugin();
}
