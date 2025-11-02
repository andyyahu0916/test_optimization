#!/bin/bash
# Complete OpenMM uninstall script (DELETE ONLY)
# This will remove ALL OpenMM installations from conda environment
# You will reinstall manually afterwards

set -e  # Exit on error

CONDA_ENV="/home/andy/miniforge3/envs/cuda"
BACKUP_DIR="/home/andy/test_optimization/openmm_complete_backup_$(date +%Y%m%d_%H%M%S)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "=========================================="
echo "OpenMM Complete Uninstall (DELETE ONLY)"
echo "=========================================="
echo ""
echo -e "${YELLOW}WARNING: This will remove ALL OpenMM files!${NC}"
echo ""
echo "Will DELETE from conda environment:"
echo "  ✗ All OpenMM Python packages"
echo "  ✗ All OpenMM shared libraries"
echo "  ✗ All OpenMM plugins (including your ElectrodeChargePlugin)"
echo "  ✗ All OpenMM headers"
echo ""
echo "Will KEEP:"
echo "  ✓ Source code in /home/andy/test_optimization/"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
read -p "Are you ABSOLUTELY SURE? (type yes to continue): " -r
echo
if [[ ! $REPLY == "yes" ]]; then
    echo "Aborted. No changes made."
    exit 1
fi

# Create backup directory
echo ""
echo -e "${GREEN}Step 1: Creating backup...${NC}"
mkdir -p "$BACKUP_DIR"

# Backup Python packages
echo "  Backing up Python packages..."
if [ -d "$CONDA_ENV/lib/python3.13/site-packages/openmm" ]; then
    cp -r "$CONDA_ENV/lib/python3.13/site-packages/openmm" "$BACKUP_DIR/" 2>/dev/null || true
    echo "    ✓ Backed up openmm package"
fi
if [ -d "$CONDA_ENV/lib/python3.13/site-packages/simtk" ]; then
    cp -r "$CONDA_ENV/lib/python3.13/site-packages/simtk" "$BACKUP_DIR/" 2>/dev/null || true
    echo "    ✓ Backed up simtk package"
fi

# Backup egg-info
for egg_info in "$CONDA_ENV/lib/python3.13/site-packages/OpenMM"*.egg-info; do
    if [ -d "$egg_info" ]; then
        cp -r "$egg_info" "$BACKUP_DIR/" 2>/dev/null || true
        echo "    ✓ Backed up $(basename $egg_info)"
    fi
done

# Backup libraries
echo "  Backing up shared libraries..."
mkdir -p "$BACKUP_DIR/lib"
cp "$CONDA_ENV/lib/libOpenMM"*.so* "$BACKUP_DIR/lib/" 2>/dev/null || true

# Backup plugins
echo "  Backing up plugins..."
mkdir -p "$BACKUP_DIR/plugins"
cp "$CONDA_ENV/lib/plugins/libOpenMM"*.so "$BACKUP_DIR/plugins/" 2>/dev/null || true
cp "$CONDA_ENV/lib/plugins/libElectrodeChargePlugin"*.so "$BACKUP_DIR/plugins/" 2>/dev/null || true

echo "  ✓ Backup complete"

# Uninstall via pip (this will update pip's database)
echo ""
echo -e "${GREEN}Step 2: Uninstalling OpenMM via pip...${NC}"
pip uninstall -y openmm 2>/dev/null && echo "  ✓ pip uninstall successful" || echo "  ℹ pip uninstall not needed (package not found)"

# Remove ALL OpenMM files from conda environment
echo ""
echo -e "${GREEN}Step 3: Removing all OpenMM files...${NC}"

# Remove Python packages
echo "  Removing Python packages..."
rm -rf "$CONDA_ENV/lib/python3.13/site-packages/openmm" 2>/dev/null && echo "    ✓ Removed openmm package" || true
rm -rf "$CONDA_ENV/lib/python3.13/site-packages/simtk" 2>/dev/null && echo "    ✓ Removed simtk package" || true
rm -rf "$CONDA_ENV/lib/python3.13/site-packages/OpenMM"*.egg-info 2>/dev/null && echo "    ✓ Removed egg-info" || true
rm -rf "$CONDA_ENV/lib/python3.13/site-packages/OpenMM"*.dist-info 2>/dev/null || true

# Remove shared libraries
echo "  Removing shared libraries..."
rm -f "$CONDA_ENV/lib/libOpenMM"*.so* 2>/dev/null && echo "    ✓ Removed libOpenMM*.so" || true
rm -f "$CONDA_ENV/lib64/libOpenMM"*.so* 2>/dev/null || true

# Remove ElectrodeCharge libraries
rm -f "$CONDA_ENV/lib/libElectrodeChargePlugin.so" 2>/dev/null && echo "    ✓ Removed ElectrodeChargePlugin lib" || true

# Remove plugins
echo "  Removing plugins..."
rm -f "$CONDA_ENV/lib/plugins/libOpenMM"*.so 2>/dev/null && echo "    ✓ Removed OpenMM plugins" || true
rm -f "$CONDA_ENV/lib/plugins/libElectrodeChargePlugin"*.so 2>/dev/null && echo "    ✓ Removed ElectrodeChargePlugin plugins" || true

# Remove headers
echo "  Removing headers..."
rm -rf "$CONDA_ENV/include/openmm" 2>/dev/null && echo "    ✓ Removed openmm headers" || true
rm -rf "$CONDA_ENV/include/OpenMM" 2>/dev/null || true
rm -f "$CONDA_ENV/include/ElectrodeCharge"*.h 2>/dev/null && echo "    ✓ Removed ElectrodeCharge headers" || true

# Remove any other OpenMM-related files (excluding openmmtools)
echo "  Scanning for remaining OpenMM files..."
removed_count=0
find "$CONDA_ENV" \( -name "*openmm*" -o -name "*OpenMM*" \) ! -name "*openmmtools*" 2>/dev/null | while read file; do
    if [ -e "$file" ]; then
        rm -rf "$file" 2>/dev/null && ((removed_count++)) || true
    fi
done

echo "  ✓ All OpenMM files removed"

# Verify removal
echo ""
echo -e "${GREEN}Step 4: Verifying removal...${NC}"
remaining=$(find "$CONDA_ENV" \( -name "*openmm*" -o -name "*OpenMM*" \) ! -name "*openmmtools*" 2>/dev/null | wc -l)
if [ "$remaining" -gt 0 ]; then
    echo -e "  ${YELLOW}Warning: $remaining OpenMM-related files still found${NC}"
    echo "  Listing remaining files:"
    find "$CONDA_ENV" \( -name "*openmm*" -o -name "*OpenMM*" \) ! -name "*openmmtools*" 2>/dev/null | head -10
else
    echo -e "  ${GREEN}✓ All OpenMM files successfully removed${NC}"
fi

# Test that OpenMM is really gone
echo ""
echo -e "${GREEN}Step 5: Testing removal...${NC}"
if python -c "import openmm" 2>&1 | grep -q "No module"; then
    echo -e "  ${GREEN}✓ OpenMM successfully removed (import fails as expected)${NC}"
else
    echo -e "  ${RED}✗ Warning: OpenMM can still be imported!${NC}"
    python -c "import openmm; print('  Found at:', openmm.__file__)"
fi

# Final summary
echo ""
echo "=========================================="
echo -e "${GREEN}Deletion Complete!${NC}"
echo "=========================================="
echo ""
echo "Backup location:"
echo "  $BACKUP_DIR"
echo ""
echo -e "${YELLOW}Next steps (do these MANUALLY):${NC}"
echo ""
echo "1. Install OpenMM 8.4.0:"
echo "   cd /home/andy/test_optimization/openmm-8.4.0/build"
echo "   make install"
echo ""
echo "2. Test OpenMM installation:"
echo "   python -m openmm.testInstallation"
echo ""
echo "3. Rebuild and install ElectrodeChargePlugin:"
echo "   cd /home/andy/test_optimization/plugins/ElectrodeChargePlugin/build"
echo "   cmake .."
echo "   make"
echo "   make install"
echo ""
echo "4. Test plugin:"
echo "   python -c 'from openmm import *; from electrodechargePlugin import *; print(\"OK\")'"
echo ""
echo "If something goes wrong, restore from backup:"
echo "  The backup is in: $BACKUP_DIR"
echo ""
