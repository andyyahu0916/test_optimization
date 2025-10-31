#!/usr/bin/env python3
"""
Quick test: Can we create ElectrodeChargeForce and call setForceGroup?
"""
import sys
sys.path.insert(0, '/home/andy/test_optimization/plugins/ElectrodeChargePlugin/build/python')

print("=" * 60)
print("Quick Plugin Test")
print("=" * 60)

# Test 1: Import
try:
    import electrodecharge as ec
    print("\n✓ electrodecharge imported")
except Exception as e:
    print(f"\n✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create force
try:
    force = ec.ElectrodeChargeForce()
    print("✓ ElectrodeChargeForce created")
except Exception as e:
    print(f"✗ Force creation failed: {e}")
    sys.exit(1)

# Test 3: Call setForceGroup
try:
    force.setForceGroup(10)
    print("✓ setForceGroup(10) succeeded")
    print(f"  getForceGroup() = {force.getForceGroup()}")
except Exception as e:
    print(f"✗ setForceGroup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Test all setters
try:
    force.setCathode([0, 1, 2], 0.5)
    force.setAnode([3, 4, 5], 0.5)
    force.setNumIterations(4)
    force.setSmallThreshold(1e-6)
    force.setConductorData(
        [6, 7], [0, 0, 1, 0, 0, 1], [0.1, 0.1],
        [6], [0, 0, 1], [0.25],
        [0, 0], [2]
    )
    print("✓ All 8-parameter methods work")
except Exception as e:
    print(f"✗ Setter failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ All plugin tests PASSED!")
print("   Plugin is ready for use with run_openMM.py")
print("=" * 60)
