cmake_minimum_required(VERSION 3.17)

project(OpenMMElectrodeChargePlugin)

set(OPENMM_DIR "/usr/local/openmm" CACHE PATH "Where OpenMM is installed")

if(NOT OpenMM_DIR AND DEFINED OPENMM_DIR)
    set(OpenMM_DIR "${OPENMM_DIR}")
endif()

find_package(OpenMM REQUIRED)

include_directories(${OPENMM_INCLUDE_DIRS})

set(CMAKE_CXX_STANDARD 11)

if(APPLE)
    set(CMAKE_INSTALL_NAME_DIR "@rpath")
    set(EXTRA_COMPILE_FLAGS "-msse2 -stdlib=libc++")
endif()

if(${CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT})
    if(WIN32)
        set(CMAKE_INSTALL_PREFIX "$ENV{ProgramFiles}/OpenMM" CACHE PATH "Where to install the plugin" FORCE)
    else()
        set(CMAKE_INSTALL_PREFIX "/usr/local/openmm" CACHE PATH "Where to install the plugin" FORCE)
    endif()
endif()

set(PLUGIN_SOURCE_SUBDIRS openmmapi)
set(PLUGIN_LIBRARY_NAME ElectrodeChargePlugin)
set(SHARED_PLUGIN_TARGET ${PLUGIN_LIBRARY_NAME})

set(API_INCLUDE_DIRS "openmmapi/include" "openmmapi/include/internal")
set(API_INCLUDE_FILES)
foreach(dir ${API_INCLUDE_DIRS})
    file(GLOB fullpaths ${dir}/*.h)
    set(API_INCLUDE_FILES ${API_INCLUDE_FILES} ${fullpaths})
    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/${dir})
endforeach()

    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/platforms/reference/include)
    include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/platforms/cuda/include)

set(SOURCE_FILES)
set(SOURCE_INCLUDE_FILES)
foreach(subdir ${PLUGIN_SOURCE_SUBDIRS})
    file(GLOB src_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.cpp)
    file(GLOB internal_src_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/internal/*.cpp)
    file(GLOB incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/*.h)
    file(GLOB internal_incl_files ${CMAKE_CURRENT_SOURCE_DIR}/${subdir}/src/internal/*.h)
    set(SOURCE_FILES ${SOURCE_FILES} ${src_files} ${internal_src_files})
    set(SOURCE_INCLUDE_FILES ${SOURCE_INCLUDE_FILES} ${incl_files} ${internal_incl_files})
endforeach()

add_library(${SHARED_PLUGIN_TARGET} SHARED ${SOURCE_FILES} ${SOURCE_INCLUDE_FILES} ${API_INCLUDE_FILES})
set_target_properties(${SHARED_PLUGIN_TARGET}
    PROPERTIES COMPILE_FLAGS "-DELECTRODE_CHARGE_BUILDING_SHARED_LIBRARY ${EXTRA_COMPILE_FLAGS}"
               LINK_FLAGS "${EXTRA_COMPILE_FLAGS}")
target_link_libraries(${SHARED_PLUGIN_TARGET} ${OPENMM_LIBRARIES})
install(TARGETS ${SHARED_PLUGIN_TARGET} DESTINATION lib)

file(GLOB API_ONLY_INCLUDE_FILES "openmmapi/include/*.h")
install(FILES ${API_ONLY_INCLUDE_FILES} DESTINATION include)
file(GLOB API_ONLY_INCLUDE_FILES_INTERNAL "openmmapi/include/internal/*.h")
install(FILES ${API_ONLY_INCLUDE_FILES_INTERNAL} DESTINATION include/internal)

add_subdirectory(platforms/reference)
find_package(CUDA QUIET)
if(CUDA_FOUND)
    add_subdirectory(platforms/cuda)
endif()

find_program(PYTHON_EXECUTABLE python)
find_program(SWIG_EXECUTABLE swig)
if(PYTHON_EXECUTABLE AND SWIG_EXECUTABLE)
    add_subdirectory(python)
endif()
