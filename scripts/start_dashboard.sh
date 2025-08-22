#!/bin/bash

# Trading AI Dashboard Launcher

echo "🚀 Starting Trading AI Dashboard..."
echo "==================================="

# Change to project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    echo "❌ Error: main.py not found in $PROJECT_DIR"
    exit 1
fi

# Check dependencies
echo "📋 Checking dependencies..."
python3 -c "
import sys
try:
    import streamlit
    print('✅ Streamlit found')
except ImportError:
    print('❌ Streamlit not installed. Run: pip install streamlit')
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Set environment variables
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH}"

echo "🌐 Starting dashboard on http://localhost:8501"
echo "📁 Project directory: $PROJECT_DIR"
echo ""
echo "💡 Dashboard Features:"
echo "   • Live trading signals"
echo "   • Technical analysis charts"
echo "   • Backtesting interface"
echo "   • Performance monitoring"
echo "   • Risk management tools"
echo ""
echo "🛑 Press Ctrl+C to stop the dashboard"
echo ""

# Start the dashboard
python3 main.py dashboard