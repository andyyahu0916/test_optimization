#!/usr/bin/env python
"""最简单的Plugin测试"""

print("Step 1: Start")

import sys
print("Step 2: Importing openmm...")
from openmm.app import *
from openmm import *
from openmm.unit import *

print("Step 3: Importing constantvplugin...")
import constantvplugin

print("Step 4: Creating minimal system...")
system = System()
system.addParticle(12.0)

print("Step 4.5: Adding NonbondedForce...")
nonbonded = NonbondedForce()
nonbonded.setNonbondedMethod(NonbondedForce.PME)
nonbonded.setCutoffDistance(1.0*nanometer)
nonbonded.addParticle(0.0, 0.34, 0.1)
system.addForce(nonbonded)

print("Step 5: Creating ConstantVForce...")
force = constantvplugin.ConstantVForce()
force.setVoltage(1.0)
force.setLgap(2.0)
force.setLcell(3.0)
force.setTotalArea(4.0)
force.setZCathode(1.0)
force.setZAnode(4.0)
force.setNumIterations(1)  # 测试1次迭代

print("Step 6: Adding cathode atom...")
force.addCathodeAtom(0, 2.0)

print("Step 7: Adding force to system...")
system.addForce(force)

print("Step 8: Creating context...")
integrator = VerletIntegrator(0.001*picoseconds)
platform = Platform.getPlatformByName('Reference')
context = Context(system, integrator, platform)

print("Step 9: Setting positions...")
context.setPositions([Vec3(0, 0, 0)])

print("Step 10: Running one step (nIterations=0)...")
integrator.step(1)
print("✅ SUCCESS with nIterations=0!")
