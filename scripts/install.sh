#!/bin/bash

# Trading AI Installation Script
# This script sets up the Trading AI system with all dependencies

set -e  # Exit on any error

echo "🤖 Trading AI Installation Script"
echo "=================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Error: Python 3.8 or higher is required. Found: $python_version"
    exit 1
fi
echo "✅ Python version $python_version is compatible"

# Check if we're in a virtual environment
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: Not in a virtual environment"
    echo "   It's recommended to use a virtual environment:"
    echo "   python3 -m venv trading-ai-env"
    echo "   source trading-ai-env/bin/activate"
    echo ""
    read -p "Continue anyway? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo "✅ Virtual environment detected: $VIRTUAL_ENV"
fi

# Update pip
echo "📦 Updating pip..."
python3 -m pip install --upgrade pip

# Install system dependencies for TA-Lib
echo "🔧 Installing system dependencies..."
OS="$(uname -s)"
case "${OS}" in
    Linux*)
        echo "Detected Linux system"
        if command -v apt-get &> /dev/null; then
            echo "Installing TA-Lib dependencies via apt..."
            sudo apt-get update
            sudo apt-get install -y build-essential wget
            
            # Install TA-Lib from source
            echo "Installing TA-Lib from source..."
            cd /tmp
            wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
            tar -xzf ta-lib-0.4.0-src.tar.gz
            cd ta-lib/
            ./configure --prefix=/usr
            make
            sudo make install
            cd ..
            rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
            
        elif command -v yum &> /dev/null; then
            echo "Installing TA-Lib dependencies via yum..."
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y wget
            
            # Install TA-Lib from source
            echo "Installing TA-Lib from source..."
            cd /tmp
            wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz
            tar -xzf ta-lib-0.4.0-src.tar.gz
            cd ta-lib/
            ./configure --prefix=/usr
            make
            sudo make install
            cd ..
            rm -rf ta-lib ta-lib-0.4.0-src.tar.gz
        else
            echo "⚠️  Could not detect package manager. You may need to install TA-Lib manually."
        fi
        ;;
    Darwin*)
        echo "Detected macOS system"
        if command -v brew &> /dev/null; then
            echo "Installing TA-Lib via Homebrew..."
            brew install ta-lib
        else
            echo "❌ Homebrew not found. Please install Homebrew first:"
            echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
            exit 1
        fi
        ;;
    *)
        echo "⚠️  Unsupported operating system: ${OS}"
        echo "   You may need to install TA-Lib manually"
        ;;
esac

# Go to the project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Install Python dependencies
echo "🐍 Installing Python dependencies..."
python3 -m pip install -r requirements.txt

# Install TA-Lib Python wrapper
echo "📊 Installing TA-Lib Python wrapper..."
python3 -m pip install TA-Lib

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p data/raw data/processed logs models/saved

# Copy environment file
if [ ! -f .env ]; then
    echo "📝 Creating environment configuration..."
    cp .env.example .env
    echo "✅ Created .env file. Please review and customize the settings."
else
    echo "✅ Environment file already exists"
fi

# Make scripts executable
echo "🔧 Setting script permissions..."
chmod +x scripts/*.sh
chmod +x main.py

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import sys
import importlib.util

required_packages = [
    'numpy', 'pandas', 'tensorflow', 'yfinance', 
    'streamlit', 'plotly', 'loguru', 'pydantic'
]

missing_packages = []
for package in required_packages:
    if importlib.util.find_spec(package) is None:
        missing_packages.append(package)

if missing_packages:
    print(f'❌ Missing packages: {missing_packages}')
    sys.exit(1)

# Test TA-Lib specifically
try:
    import talib
    print('✅ TA-Lib imported successfully')
except ImportError:
    print('⚠️  TA-Lib import failed - some indicators may not work')

print('✅ All core packages installed successfully')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "🎉 Installation completed successfully!"
    echo ""
    echo "📚 Next steps:"
    echo "1. Review and customize .env configuration"
    echo "2. Run: python main.py dashboard   (to launch web interface)"
    echo "3. Run: python main.py signals     (to generate signals)"
    echo "4. Run: python main.py train       (to train AI models)"
    echo ""
    echo "📖 See README.md for detailed usage instructions"
    echo ""
else
    echo "❌ Installation completed with warnings"
    echo "   Some features may not work properly"
    exit 1
fi