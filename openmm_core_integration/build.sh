#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
# Build Script for ConstantV Native Core Integration
# ═══════════════════════════════════════════════════════════════════════════

set -e  # Exit on error

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# OpenMM installation path (adjust if needed)
OPENMM_DIR="${OPENMM_DIR:-/usr/local/openmm}"

# Build type (Release or Debug)
BUILD_TYPE="${BUILD_TYPE:-Release}"

# Number of parallel jobs
JOBS="${JOBS:-$(nproc)}"

# CUDA architectures (adjust for your GPU)
# sm_70: V100, sm_75: T4, sm_80: A100, sm_86: RTX 30xx, sm_89: RTX 40xx, sm_90: H100
CUDA_ARCHS="${CUDA_ARCHS:-70;75;80;86}"

# Build directory
BUILD_DIR="build"

# ═══════════════════════════════════════════════════════════════════════════
# Colors for output
# ═══════════════════════════════════════════════════════════════════════════

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ═══════════════════════════════════════════════════════════════════════════
# Helper Functions
# ═══════════════════════════════════════════════════════════════════════════

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# ═══════════════════════════════════════════════════════════════════════════
# Pre-flight Checks
# ═══════════════════════════════════════════════════════════════════════════

log_info "Starting ConstantV Native Integration build..."
echo ""

# Check for CMake
if ! command -v cmake &> /dev/null; then
    log_error "CMake not found. Please install CMake 3.18 or later."
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -n1 | awk '{print $3}')
log_info "CMake version: $CMAKE_VERSION"

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    log_warn "CUDA compiler (nvcc) not found. CUDA library will be disabled."
    BUILD_CUDA=OFF
else
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | tr -d ',')
    log_info "CUDA version: $NVCC_VERSION"
    BUILD_CUDA=ON
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    log_error "Python 3 not found. Please install Python 3."
    exit 1
fi

PYTHON_VERSION=$(python3 --version | awk '{print $2}')
log_info "Python version: $PYTHON_VERSION"

# Check for SWIG
if ! command -v swig &> /dev/null; then
    log_warn "SWIG not found. Python bindings will be disabled."
    BUILD_PYTHON=OFF
else
    SWIG_VERSION=$(swig -version | grep "SWIG Version" | awk '{print $3}')
    log_info "SWIG version: $SWIG_VERSION"
    BUILD_PYTHON=ON
fi

# Check for OpenMM
if [ ! -d "$OPENMM_DIR" ]; then
    log_error "OpenMM not found at $OPENMM_DIR"
    log_error "Set OPENMM_DIR environment variable to your OpenMM installation path"
    exit 1
fi

log_info "OpenMM directory: $OPENMM_DIR"

echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Clean Build
# ═══════════════════════════════════════════════════════════════════════════

if [ -d "$BUILD_DIR" ]; then
    log_info "Removing old build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# ═══════════════════════════════════════════════════════════════════════════
# CMake Configuration
# ═══════════════════════════════════════════════════════════════════════════

log_info "Configuring CMake..."
echo ""

cmake .. \
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
    -DOpenMM_DIR="$OPENMM_DIR" \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DBUILD_CUDA_LIB="$BUILD_CUDA" \
    -DBUILD_PYTHON_WRAPPERS="$BUILD_PYTHON" \
    -DCMAKE_INSTALL_PREFIX="$OPENMM_DIR" \
    || {
        log_error "CMake configuration failed!"
        exit 1
    }

echo ""
log_success "CMake configuration complete"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Build
# ═══════════════════════════════════════════════════════════════════════════

log_info "Building with $JOBS parallel jobs..."
echo ""

make -j"$JOBS" VERBOSE=1 || {
    log_error "Build failed!"
    exit 1
}

echo ""
log_success "Build complete"
echo ""

# ═══════════════════════════════════════════════════════════════════════════
# Installation
# ═══════════════════════════════════════════════════════════════════════════

if [ "$1" == "install" ]; then
    log_info "Installing..."
    sudo make install || {
        log_error "Installation failed!"
        exit 1
    }
    log_success "Installation complete"
    echo ""
fi

# ═══════════════════════════════════════════════════════════════════════════
# Summary
# ═══════════════════════════════════════════════════════════════════════════

log_success "╔═══════════════════════════════════════════════════════════════════╗"
log_success "║         ConstantV Native Integration Build Complete! 🎉          ║"
log_success "╚═══════════════════════════════════════════════════════════════════╝"
echo ""

log_info "Build Summary:"
echo "  Build Type:         $BUILD_TYPE"
echo "  CUDA Support:       $BUILD_CUDA"
echo "  Python Bindings:    $BUILD_PYTHON"
echo "  CUDA Architectures: $CUDA_ARCHS"
echo ""

if [ "$BUILD_PYTHON" == "ON" ]; then
    log_info "To test Python bindings:"
    echo "  cd $BUILD_DIR"
    echo "  python3 -c 'import constantv; print(\"Import successful!\")'"
    echo ""
fi

if [ "$1" != "install" ]; then
    log_info "To install, run:"
    echo "  ./build.sh install"
    echo ""
fi

log_info "Enjoy your 6× speedup! 🚀"
