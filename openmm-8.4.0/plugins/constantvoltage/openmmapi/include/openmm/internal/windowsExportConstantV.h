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

#ifndef OPENMM_WINDOWSEXPORT_CONSTANTV_H_
#define OPENMM_WINDOWSEXPORT_CONSTANTV_H_

/*
 * Shared libraries are messy in Visual Studio. We have to distinguish three
 * temporary cases:
 * (1) this header is being used to build the OpenMMConstantV shared library
 *     (dllexport)
 * (2) this header is being used by a *client* of the OpenMMConstantV shared
 *     library (dllimport)
 * (3) we are building the OpenMMConstantV static library, or the client is
 *     being compiled with the expectation of linking with the
 *     OpenMMConstantV static library (nothing special needed)
 */

#ifdef _MSC_VER
    #ifdef OPENMM_CONSTANTV_BUILDING_SHARED_LIBRARY
        #define OPENMM_EXPORT_CONSTANTV __declspec(dllexport)
    #elif defined(OPENMM_CONSTANTV_BUILDING_STATIC_LIBRARY) || defined(OPENMM_USE_STATIC_LIBRARIES)
        #define OPENMM_EXPORT_CONSTANTV
    #else
        #define OPENMM_EXPORT_CONSTANTV __declspec(dllimport)
    #endif
#else
    #define OPENMM_EXPORT_CONSTANTV
#endif

#endif // OPENMM_WINDOWSEXPORT_CONSTANTV_H_
