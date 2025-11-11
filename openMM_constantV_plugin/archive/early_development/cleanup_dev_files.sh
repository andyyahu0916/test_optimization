#!/bin/bash
#
# Cleanup development files - keep only production-ready files
#

echo "======================================================================"
echo "Cleaning up development files from plugin directory"
echo "======================================================================"
echo ""

# Create backup first
BACKUP_DIR="dev_files_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

echo "Creating backup in: $BACKUP_DIR"

# Files/directories to remove (backup first)
TO_REMOVE=(
    "test_numpy_reference.py"
    "test_plugin_vs_numpy.py"
    "test_cuda_vs_reference.py"
    "test_fv_md_with_real_data.py"
    "test_fv_md_quick.py"
    "test_debug_steps.py"
    "run_test.sh"
    "TEST_RESULTS.md"
    "lib/"
    "sapt_exclusions.py"
    "__pycache__/"
)

for item in "${TO_REMOVE[@]}"; do
    if [ -e "$item" ]; then
        echo "  Moving $item to backup..."
        mv "$item" "$BACKUP_DIR/"
    else
        echo "  Skipping $item (not found)"
    fi
done

echo ""
echo "======================================================================"
echo "Cleanup complete!"
echo "======================================================================"
echo ""
echo "Files backed up to: $BACKUP_DIR"
echo ""
echo "Production files remaining:"
ls -lh *.py *.sh *.md 2>/dev/null | grep -v "cleanup"
echo ""
echo "To restore, run: mv $BACKUP_DIR/* ."
