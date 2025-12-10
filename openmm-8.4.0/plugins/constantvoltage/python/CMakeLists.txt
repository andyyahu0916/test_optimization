# Python SWIG wrapper for ConstantVoltage plugin
#
# This creates the Python module: openmm.constantvoltage

# Find required packages
find_package(SWIG REQUIRED)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
include(${SWIG_USE_FILE})

# Include paths
include_directories(
    ${CMAKE_SOURCE_DIR}/openmmapi/include
    ${CMAKE_SOURCE_DIR}/plugins/constantvoltage/openmmapi/include
    ${Python3_INCLUDE_DIRS}
)

# Configure SWIG
set(CMAKE_SWIG_FLAGS "")
set_source_files_properties(constantvoltage.i PROPERTIES CPLUSPLUS ON)

# Generate Python module
swig_add_library(constantvoltagepython
    TYPE MODULE
    LANGUAGE python
    SOURCES constantvoltage.i
)

# Link libraries
target_link_libraries(constantvoltagepython PRIVATE
    OpenMM
    OpenMMConstantVoltage
    ${Python3_LIBRARIES}
)

# Set output name
set_target_properties(constantvoltagepython PROPERTIES
    OUTPUT_NAME "_constantvoltage"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
)

# Install Python module
install(TARGETS constantvoltagepython
    LIBRARY DESTINATION ${Python3_SITEARCH}/openmm
)

install(FILES ${CMAKE_CURRENT_BINARY_DIR}/constantvoltage.py
    DESTINATION ${Python3_SITEARCH}/openmm
)
