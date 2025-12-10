# Reference platform for ConstantVoltage plugin

SET(REFERENCE_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ReferenceConstantVoltageKernels.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/ReferenceConstantVoltageKernelFactory.cpp
)

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/include)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/../common/include)

# Build as part of shared library
SET(SOURCE_CONSTANTV_FILES ${SOURCE_CONSTANTV_FILES} ${REFERENCE_SOURCE_FILES} PARENT_SCOPE)
