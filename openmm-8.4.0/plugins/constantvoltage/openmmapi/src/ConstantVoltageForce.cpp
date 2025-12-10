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

#include "openmm/ConstantVoltageForce.h"
#include "openmm/internal/ConstantVoltageForceImpl.h"
#include "openmm/OpenMMException.h"

using namespace OpenMM;

ConstantVoltageForce::ConstantVoltageForce()
    : voltage(0.0), Lgap(1.0), Lcell(1.0), totalArea(0.0),
      zCathode(0.0), zAnode(0.0),
      numSCFIterations(4), scfFrequency(200), smallThreshold(1e-6) {
}

// ═══════════════════════════════════════════════════════════════════════════
// Flat Electrode Atoms
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVoltageForce::addCathodeAtom(int particle, double area) {
    cathodeParticles.push_back(particle);
    cathodeAreas.push_back(area);
    return cathodeParticles.size() - 1;
}

void ConstantVoltageForce::getCathodeAtomParameters(int index, int& particle, double& area) const {
    if (index < 0 || index >= (int)cathodeParticles.size())
        throw OpenMMException("ConstantVoltageForce: cathode atom index out of range");
    particle = cathodeParticles[index];
    area = cathodeAreas[index];
}

void ConstantVoltageForce::setCathodeAtomParameters(int index, int particle, double area) {
    if (index < 0 || index >= (int)cathodeParticles.size())
        throw OpenMMException("ConstantVoltageForce: cathode atom index out of range");
    cathodeParticles[index] = particle;
    cathodeAreas[index] = area;
}

int ConstantVoltageForce::addAnodeAtom(int particle, double area) {
    anodeParticles.push_back(particle);
    anodeAreas.push_back(area);
    return anodeParticles.size() - 1;
}

void ConstantVoltageForce::getAnodeAtomParameters(int index, int& particle, double& area) const {
    if (index < 0 || index >= (int)anodeParticles.size())
        throw OpenMMException("ConstantVoltageForce: anode atom index out of range");
    particle = anodeParticles[index];
    area = anodeAreas[index];
}

void ConstantVoltageForce::setAnodeAtomParameters(int index, int particle, double area) {
    if (index < 0 || index >= (int)anodeParticles.size())
        throw OpenMMException("ConstantVoltageForce: anode atom index out of range");
    anodeParticles[index] = particle;
    anodeAreas[index] = area;
}

// ═══════════════════════════════════════════════════════════════════════════
// Electrolyte Atoms
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVoltageForce::addElectrolyteAtom(int particle) {
    electrolyteParticles.push_back(particle);
    return electrolyteParticles.size() - 1;
}

int ConstantVoltageForce::getElectrolyteAtomParticle(int index) const {
    if (index < 0 || index >= (int)electrolyteParticles.size())
        throw OpenMMException("ConstantVoltageForce: electrolyte atom index out of range");
    return electrolyteParticles[index];
}

// ═══════════════════════════════════════════════════════════════════════════
// Buckyball Conductors
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVoltageForce::addBuckyballConductor(
    const std::vector<int>& virtualParticles,
    const std::vector<int>& realParticles,
    const std::string& electrodeType)
{
    if (virtualParticles.size() != realParticles.size())
        throw OpenMMException("ConstantVoltageForce: virtualParticles and realParticles must have same size");
    if (electrodeType != "cathode" && electrodeType != "anode")
        throw OpenMMException("ConstantVoltageForce: electrodeType must be 'cathode' or 'anode'");

    buckyballVirtualParticles.push_back(virtualParticles);
    buckyballRealParticles.push_back(realParticles);
    buckyballElectrodeTypes.push_back(electrodeType);
    return buckyballVirtualParticles.size() - 1;
}

void ConstantVoltageForce::getBuckyballConductorParameters(
    int index,
    std::vector<int>& virtualParticles,
    std::vector<int>& realParticles,
    std::string& electrodeType) const
{
    if (index < 0 || index >= (int)buckyballVirtualParticles.size())
        throw OpenMMException("ConstantVoltageForce: buckyball conductor index out of range");
    virtualParticles = buckyballVirtualParticles[index];
    realParticles = buckyballRealParticles[index];
    electrodeType = buckyballElectrodeTypes[index];
}

// ═══════════════════════════════════════════════════════════════════════════
// Nanotube Conductors
// ═══════════════════════════════════════════════════════════════════════════

int ConstantVoltageForce::addNanotubeConductor(
    const std::vector<int>& virtualParticles,
    const std::vector<int>& realParticles,
    const std::string& electrodeType,
    const Vec3& axis)
{
    if (virtualParticles.size() != realParticles.size())
        throw OpenMMException("ConstantVoltageForce: virtualParticles and realParticles must have same size");
    if (electrodeType != "cathode" && electrodeType != "anode")
        throw OpenMMException("ConstantVoltageForce: electrodeType must be 'cathode' or 'anode'");

    nanotubeVirtualParticles.push_back(virtualParticles);
    nanotubeRealParticles.push_back(realParticles);
    nanotubeElectrodeTypes.push_back(electrodeType);
    nanotubeAxes.push_back(axis);
    return nanotubeVirtualParticles.size() - 1;
}

void ConstantVoltageForce::getNanotubeConductorParameters(
    int index,
    std::vector<int>& virtualParticles,
    std::vector<int>& realParticles,
    std::string& electrodeType,
    Vec3& axis) const
{
    if (index < 0 || index >= (int)nanotubeVirtualParticles.size())
        throw OpenMMException("ConstantVoltageForce: nanotube conductor index out of range");
    virtualParticles = nanotubeVirtualParticles[index];
    realParticles = nanotubeRealParticles[index];
    electrodeType = nanotubeElectrodeTypes[index];
    axis = nanotubeAxes[index];
}

// ═══════════════════════════════════════════════════════════════════════════
// System Parameters
// ═══════════════════════════════════════════════════════════════════════════

void ConstantVoltageForce::setVoltage(double v) {
    voltage = v;
}

void ConstantVoltageForce::setLgap(double lgap) {
    if (lgap <= 0)
        throw OpenMMException("ConstantVoltageForce: Lgap must be positive");
    Lgap = lgap;
}

void ConstantVoltageForce::setLcell(double lcell) {
    if (lcell <= 0)
        throw OpenMMException("ConstantVoltageForce: Lcell must be positive");
    Lcell = lcell;
}

void ConstantVoltageForce::setTotalArea(double area) {
    if (area < 0)
        throw OpenMMException("ConstantVoltageForce: totalArea must be non-negative");
    totalArea = area;
}

void ConstantVoltageForce::setElectrodeZPositions(double zc, double za) {
    zCathode = zc;
    zAnode = za;
}

// ═══════════════════════════════════════════════════════════════════════════
// Force Interface
// ═══════════════════════════════════════════════════════════════════════════

ForceImpl* ConstantVoltageForce::createImpl() const {
    return new ConstantVoltageForceImpl(*this);
}
