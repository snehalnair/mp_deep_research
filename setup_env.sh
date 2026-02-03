#!/bin/bash
# ============================================================================
# Materials Project Deep Research - Environment Setup Script
# ============================================================================
#
# Usage:
#   chmod +x setup_env.sh
#   ./setup_env.sh
#
# ============================================================================

set -e  # Exit on error

# Configuration
ENV_NAME="${1:-mp_research}"
PYTHON_VERSION="3.11"

echo "=============================================="
echo "Materials Project Deep Research Setup"
echo "=============================================="
echo ""

# -----------------------------------------------------------------------------
# Step 1: Create Virtual Environment
# -----------------------------------------------------------------------------
echo "Step 1: Creating virtual environment '${ENV_NAME}'..."

# Check if conda is available
if command -v conda &> /dev/null; then
    echo "  Using conda..."
    conda create -n ${ENV_NAME} python=${PYTHON_VERSION} -y
    
    echo ""
    echo "  To activate: conda activate ${ENV_NAME}"
    echo ""
    
    # Activate for installation
    eval "$(conda shell.bash hook)"
    conda activate ${ENV_NAME}
    
elif command -v python3 &> /dev/null; then
    echo "  Using venv..."
    python3 -m venv ${ENV_NAME}
    
    echo ""
    echo "  To activate: source ${ENV_NAME}/bin/activate"
    echo ""
    
    # Activate for installation
    source ${ENV_NAME}/bin/activate
else
    echo "Error: Neither conda nor python3 found!"
    exit 1
fi

# -----------------------------------------------------------------------------
# Step 2: Upgrade pip
# -----------------------------------------------------------------------------
echo "Step 2: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# -----------------------------------------------------------------------------
# Step 3: Install Core Dependencies
# -----------------------------------------------------------------------------
echo ""
echo "Step 3: Installing core dependencies..."
pip install -r requirements.txt

# -----------------------------------------------------------------------------
# Step 4: Install atomate2 (Optional - requires VASP license)
# -----------------------------------------------------------------------------
echo ""
echo "Step 4: atomate2 installation (optional)..."
read -p "  Install atomate2 for DFT workflows? (y/n): " install_atomate

if [[ $install_atomate == "y" || $install_atomate == "Y" ]]; then
    pip install atomate2
    pip install fireworks
    pip install custodian
    echo "  atomate2, fireworks, custodian installed."
else
    echo "  Skipping atomate2 (can install later with: pip install atomate2)"
fi

# -----------------------------------------------------------------------------
# Step 5: Install ML packages (Optional)
# -----------------------------------------------------------------------------
echo ""
echo "Step 5: ML packages installation (optional)..."
read -p "  Install ML packages (matminer, scikit-learn)? (y/n): " install_ml

if [[ $install_ml == "y" || $install_ml == "Y" ]]; then
    pip install matminer scikit-learn
    echo "  ML packages installed."
else
    echo "  Skipping ML packages."
fi

# -----------------------------------------------------------------------------
# Step 6: Configure Environment Variables
# -----------------------------------------------------------------------------
echo ""
echo "Step 6: Environment configuration..."
echo ""

# Create .env template if it doesn't exist
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# ============================================================================
# Materials Project Deep Research - Environment Variables
# ============================================================================

# Materials Project API Key (required for MP queries)
# Get yours at: https://next-gen.materialsproject.org/api
MP_API_KEY=your_mp_api_key_here

# OpenAI API Key (required for LLM)
OPENAI_API_KEY=your_openai_api_key_here

# Anthropic API Key (optional, for Claude models)
# ANTHROPIC_API_KEY=your_anthropic_api_key_here

# LangSmith API Key (optional, for evaluation/tracing)
# LANGSMITH_API_KEY=your_langsmith_api_key_here
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_PROJECT=mp-deep-research

# LLM Model Configuration
LLM_MODEL=gpt-4o

# VASP Configuration (optional, for simulations)
# PMG_VASP_PSP_DIR=/path/to/vasp/potcars
# VASP_CMD=mpirun -np 16 vasp_std

# atomate2/FireWorks Configuration (optional)
# ATOMATE2_DB_FILE=/path/to/db.json
# FW_CONFIG_FILE=/path/to/my_launchpad.yaml
EOF
    echo "  Created .env template file."
    echo "  Please edit .env and add your API keys!"
else
    echo "  .env file already exists."
fi

# -----------------------------------------------------------------------------
# Step 7: Create Project Structure
# -----------------------------------------------------------------------------
echo ""
echo "Step 7: Creating project structure..."

mkdir -p src/mp_deep_research
mkdir -p notebooks
mkdir -p data
mkdir -p outputs
mkdir -p tests

# Create __init__.py files
touch src/__init__.py
touch src/mp_deep_research/__init__.py

echo "  Project structure created."

# -----------------------------------------------------------------------------
# Step 8: Verify Installation
# -----------------------------------------------------------------------------
echo ""
echo "Step 8: Verifying installation..."
echo ""

python << 'EOF'
import sys
print(f"Python: {sys.version}")
print()

# Check core packages
packages = [
    ("langchain", "langchain"),
    ("langgraph", "langgraph"),
    ("langsmith", "langsmith"),
    ("mp_api", "mp-api"),
    ("pymatgen", "pymatgen"),
    ("numpy", "numpy"),
    ("pandas", "pandas"),
    ("matplotlib", "matplotlib"),
    ("pydantic", "pydantic"),
]

print("Package Status:")
for import_name, display_name in packages:
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "installed")
        print(f"  ✓ {display_name}: {version}")
    except ImportError:
        print(f"  ✗ {display_name}: NOT INSTALLED")

# Check optional packages
print()
print("Optional Packages:")
optional = [
    ("atomate2", "atomate2"),
    ("fireworks", "fireworks"),
    ("matminer", "matminer"),
]
for import_name, display_name in optional:
    try:
        module = __import__(import_name)
        version = getattr(module, "__version__", "installed")
        print(f"  ✓ {display_name}: {version}")
    except ImportError:
        print(f"  - {display_name}: not installed (optional)")
EOF

# -----------------------------------------------------------------------------
# Done!
# -----------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "Setup Complete!"
echo "=============================================="
echo ""
echo "Next steps:"
echo "  1. Edit .env file with your API keys"
echo "  2. Activate environment:"
if command -v conda &> /dev/null; then
    echo "     conda activate ${ENV_NAME}"
else
    echo "     source ${ENV_NAME}/bin/activate"
fi
echo "  3. Run Jupyter:"
echo "     jupyter lab"
echo ""
echo "API Keys needed:"
echo "  - Materials Project: https://next-gen.materialsproject.org/api"
echo "  - OpenAI: https://platform.openai.com/api-keys"
echo "  - LangSmith (optional): https://smith.langchain.com/"
echo ""
