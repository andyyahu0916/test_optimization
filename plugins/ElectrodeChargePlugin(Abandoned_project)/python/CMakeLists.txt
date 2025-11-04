find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Linus-style: We already know where OpenMM is. No bullshit checks.

execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))"
    OUTPUT_VARIABLE PYTHON_EXT_SUFFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

set(MODULE_NAME electrodecharge)
set(WRAP_FILE "${CMAKE_CURRENT_BINARY_DIR}/ElectrodeChargePluginWrapper.cpp")
set(PY_MODULE_FILE "${CMAKE_CURRENT_BINARY_DIR}/${MODULE_NAME}.py")
set(PY_EXTENSION "${CMAKE_CURRENT_BINARY_DIR}/_${MODULE_NAME}${PYTHON_EXT_SUFFIX}")

add_custom_command(
    OUTPUT "${WRAP_FILE}" "${PY_MODULE_FILE}"
    COMMAND "${SWIG_EXECUTABLE}"
        -python -c++
        -DSWIG_TYPE_TABLE=OpenMM
        -o "${WRAP_FILE}"
        "-I${OPENMM_DIR}/include"
        "-I${OPENMM_DIR}/include/swig"
        "-I${CMAKE_SOURCE_DIR}/openmmapi/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/electrodecharge.i"
    DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/electrodecharge.i"
    WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)

include_directories(
    "${CMAKE_CURRENT_BINARY_DIR}"
    "${OPENMM_DIR}/include"
    "${OPENMM_DIR}/include/openmm"
    "${CMAKE_SOURCE_DIR}/openmmapi/include"
    "${Python3_INCLUDE_DIRS}"
)

add_library(_${MODULE_NAME} MODULE "${WRAP_FILE}")
target_link_libraries(_${MODULE_NAME}
    ${OPENMM_LIBRARIES}
    ElectrodeChargePlugin
    ${CMAKE_BINARY_DIR}/platforms/reference/libElectrodeChargePluginReference.so
    ${CMAKE_BINARY_DIR}/platforms/cuda/libElectrodeChargePluginCUDA.so
    Python3::Module
)
set_target_properties(_${MODULE_NAME} PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
    PREFIX "_"
    OUTPUT_NAME "${MODULE_NAME}"
    SUFFIX "${PYTHON_EXT_SUFFIX}"
)

add_custom_target(ElectrodeChargePython ALL DEPENDS _${MODULE_NAME} "${PY_MODULE_FILE}")
add_dependencies(ElectrodeChargePython ElectrodeChargePlugin)
