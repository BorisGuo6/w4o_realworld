#!/bin/bash

# World4Omni Main Installation Script
# This script provides easy access to all installation utilities

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALLATION_DIR="$SCRIPT_DIR/scripts/installation"

echo "🚀 World4Omni Installation Manager"
echo "================================="
echo ""
echo "Available installation scripts:"
echo "  1. PyTorch with CUDA 11.8 installation"
echo "  2. CUDA 11.8 configuration"
echo "  3. Install requirements.txt dependencies"
echo "  4. Grounding-SAM environment verification"
echo "  5. Complete installation (recommended)"
echo ""

# Check if installation directory exists
if [ ! -d "$INSTALLATION_DIR" ]; then
    echo "❌ Installation scripts directory not found: $INSTALLATION_DIR"
    exit 1
fi

# Function to run installation script
run_script() {
    local script_name="$1"
    local script_path="$INSTALLATION_DIR/$script_name"
    
    if [ -f "$script_path" ]; then
        echo "🔧 Running $script_name..."
        chmod +x "$script_path"
        "$script_path"
    else
        echo "❌ Script not found: $script_path"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-}" in
    "pytorch"|"1")
        echo "📦 Installing PyTorch with CUDA 11.8..."
        run_script "install_pytorch_cuda118.sh"
        ;;
    "cuda"|"2")
        echo "🔧 Configuring CUDA 11.8..."
        run_script "configure_cuda118.sh"
        ;;
    "requirements"|"req"|"3")
        echo "📦 Installing requirements.txt dependencies..."
        run_script "install_requirements.sh"
        ;;
    "verify"|"4")
        echo "🔍 Verifying Grounding-SAM environment..."
        run_script "verify_grounding_sam_env.sh"
        ;;
    "all"|"complete"|"5"|"")
        echo "🎯 Running complete installation..."
        echo ""
        echo "Step 1: Configuring CUDA 11.8..."
        run_script "configure_cuda118.sh"
        echo ""
        echo "Step 2: Installing PyTorch with CUDA 11.8..."
        run_script "install_pytorch_cuda118.sh"
        echo ""
        echo "Step 3: Installing requirements.txt dependencies..."
        run_script "install_requirements.sh"
        echo ""
        echo "Step 4: Verifying installation..."
        run_script "verify_grounding_sam_env.sh"
        ;;
    "help"|"-h"|"--help")
        echo "Usage: $0 [option]"
        echo ""
        echo "Options:"
        echo "  pytorch, 1    - Install PyTorch with CUDA 11.8"
        echo "  cuda, 2       - Configure CUDA 11.8 environment"
        echo "  requirements, 3 - Install requirements.txt dependencies"
        echo "  verify, 4     - Verify Grounding-SAM environment"
        echo "  all, 5        - Run complete installation (default)"
        echo "  help, -h      - Show this help message"
        echo ""
        echo "Examples:"
        echo "  $0            # Run complete installation"
        echo "  $0 pytorch    # Install PyTorch only"
        echo "  $0 requirements # Install requirements.txt only"
        echo "  $0 verify     # Verify environment only"
        ;;
    *)
        echo "❌ Unknown option: $1"
        echo "Run '$0 help' for usage information"
        exit 1
        ;;
esac

echo ""
echo "✅ Installation process completed!"
