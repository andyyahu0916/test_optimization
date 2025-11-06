#!/bin/bash
#
# Quick Check Script for Exclusions Fix
#
# This script verifies that all necessary files are in place
# and the exclusions fix has been properly implemented.
#

echo "========================================================================"
echo "Checking Exclusions Fix Implementation"
echo "========================================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

ERRORS=0

# Check files
echo "1. Checking for required files..."
echo ""

FILES=(
    "fv_md_plugin/exclusions.py"
    "test_exclusions.py"
    "EXCLUSIONS_CRITICAL_FIX.md"
    "EXCLUSIONS_SUMMARY.md"
    "EXCLUSIONS_VISUAL_GUIDE.md"
    "EXCLUSIONS_IMPLEMENTATION_REPORT.md"
)

for file in "${FILES[@]}"; do
    if [ -f "$file" ]; then
        echo -e "  ${GREEN}✓${NC} $file"
    else
        echo -e "  ${RED}✗${NC} $file ${RED}MISSING${NC}"
        ((ERRORS++))
    fi
done

echo ""
echo "2. Checking if run_fv_md_production.py imports exclusions..."
echo ""

if grep -q "from exclusions import apply_all_exclusions" run_fv_md_production.py; then
    echo -e "  ${GREEN}✓${NC} Import statement found"
else
    echo -e "  ${RED}✗${NC} Import statement NOT FOUND"
    echo "     Please add: from exclusions import apply_all_exclusions"
    ((ERRORS++))
fi

echo ""
echo "3. Checking if run_fv_md_production.py calls apply_all_exclusions..."
echo ""

if grep -q "apply_all_exclusions" run_fv_md_production.py; then
    echo -e "  ${GREEN}✓${NC} Function call found"
    
    # Show the context
    echo ""
    echo "  Context:"
    grep -A 5 -B 2 "apply_all_exclusions" run_fv_md_production.py | sed 's/^/    /'
else
    echo -e "  ${RED}✗${NC} Function call NOT FOUND"
    echo "     Please add call to apply_all_exclusions() after electrode identification"
    ((ERRORS++))
fi

echo ""
echo "4. Checking exclusions.py implementation..."
echo ""

FUNCTIONS=(
    "apply_electrode_exclusions"
    "apply_sapt_exclusions"
    "apply_all_exclusions"
)

for func in "${FUNCTIONS[@]}"; do
    if grep -q "def $func" fv_md_plugin/exclusions.py 2>/dev/null; then
        echo -e "  ${GREEN}✓${NC} $func() defined"
    else
        echo -e "  ${RED}✗${NC} $func() NOT FOUND"
        ((ERRORS++))
    fi
done

echo ""
echo "5. Checking if test script is executable..."
echo ""

if [ -f "test_exclusions.py" ]; then
    if [ -x "test_exclusions.py" ]; then
        echo -e "  ${GREEN}✓${NC} test_exclusions.py is executable"
    else
        echo -e "  ${YELLOW}⚠${NC} test_exclusions.py is not executable"
        echo "     Running: chmod +x test_exclusions.py"
        chmod +x test_exclusions.py
    fi
else
    echo -e "  ${RED}✗${NC} test_exclusions.py not found"
    ((ERRORS++))
fi

echo ""
echo "========================================================================"

if [ $ERRORS -eq 0 ]; then
    echo -e "${GREEN}✓ ALL CHECKS PASSED${NC}"
    echo ""
    echo "Exclusions fix is properly implemented!"
    echo ""
    echo "Next steps:"
    echo "  1. Run test: python test_exclusions.py"
    echo "  2. If test passes, re-run your simulations"
    echo "  3. Compare results with previous runs (if any)"
    echo ""
else
    echo -e "${RED}✗ $ERRORS ERROR(S) FOUND${NC}"
    echo ""
    echo "Please fix the issues above before proceeding."
    echo ""
    exit 1
fi

echo "========================================================================"
