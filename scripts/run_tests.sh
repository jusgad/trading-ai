#!/bin/bash

# Trading AI Test Runner Script

set -e

echo "🧪 Trading AI Test Suite"
echo "========================"

# Change to project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# Check if we're in the right directory
if [ ! -f "main.py" ]; then
    echo "❌ Error: Not in Trading AI project directory"
    exit 1
fi

# Check if dependencies are installed
echo "📋 Checking dependencies..."
python3 -c "
import sys
try:
    import numpy, pandas, tensorflow
    print('✅ Core dependencies found')
except ImportError as e:
    print(f'❌ Missing dependency: {e}')
    sys.exit(1)
"

# Run unit tests
echo "🔬 Running unit tests..."
echo "------------------------"

if [ -f "tests/test_signals.py" ]; then
    echo "Running signal generation tests..."
    python3 -m pytest tests/test_signals.py -v
    
    if [ $? -eq 0 ]; then
        echo "✅ Unit tests passed"
    else
        echo "❌ Unit tests failed"
        exit 1
    fi
else
    echo "⚠️  Unit test file not found, running basic import tests..."
    python3 -c "
import sys
sys.path.append('src')

try:
    from src.signals.signal_generator import SignalGenerator
    from src.risk.risk_manager import RiskManager
    from src.indicators.technical_indicators import TechnicalIndicators
    from src.data.data_fetcher import MarketDataFetcher
    print('✅ All modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
    "
fi

# Test signal generation with dummy data
echo ""
echo "🎯 Testing signal generation..."
echo "-------------------------------"

python3 -c "
import sys
sys.path.append('src')
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

try:
    from src.indicators.technical_indicators import TechnicalIndicators
    from src.risk.risk_manager import RiskManager
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
    np.random.seed(42)
    
    prices = [100]
    for _ in range(199):
        prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
    
    data = pd.DataFrame({
        'open': prices,
        'high': [p * 1.02 for p in prices],
        'low': [p * 0.98 for p in prices],
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, 200)
    }, index=dates)
    
    # Test technical indicators
    indicators = TechnicalIndicators.calculate_all_indicators(data)
    print('✅ Technical indicators calculated successfully')
    
    # Test risk management
    risk_manager = RiskManager()
    risk_metrics = risk_manager.generate_risk_assessment(
        'BUY', 100.0, indicators, 0.7, 100000
    )
    print('✅ Risk assessment completed successfully')
    
    print(f'   - Position size: {risk_metrics.position_size:.0f}')
    print(f'   - Stop loss: \${risk_metrics.stop_loss:.2f}')
    print(f'   - Take profit: \${risk_metrics.take_profit:.2f}')
    print(f'   - Risk/Reward: {risk_metrics.risk_reward_ratio:.1f}')
    
except Exception as e:
    print(f'❌ Signal generation test failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Signal generation test passed"
else
    echo "❌ Signal generation test failed"
    exit 1
fi

# Test data fetching (with internet connection)
echo ""
echo "📡 Testing data fetching..."
echo "---------------------------"

python3 -c "
import sys
sys.path.append('src')

try:
    from src.data.data_fetcher import MarketDataFetcher
    
    fetcher = MarketDataFetcher()
    
    # Try to fetch a small amount of data
    data = fetcher.fetch_data('AAPL', period='5d')
    
    if data is not None and len(data) > 0:
        print('✅ Data fetching test passed')
        print(f'   - Fetched {len(data)} data points for AAPL')
    else:
        print('⚠️  Data fetching returned empty result (network issue?)')
        
except Exception as e:
    print(f'⚠️  Data fetching test failed: {e}')
    print('   This might be due to network connectivity issues')
"

# Test dashboard components
echo ""
echo "🖥️  Testing dashboard components..."
echo "-----------------------------------"

python3 -c "
import sys
sys.path.append('src')

try:
    from src.ui.dashboard import TradingDashboard
    
    # Test dashboard initialization
    dashboard = TradingDashboard()
    print('✅ Dashboard components loaded successfully')
    
except Exception as e:
    print(f'❌ Dashboard test failed: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ Dashboard test passed"
else
    echo "❌ Dashboard test failed"
    exit 1
fi

# Performance summary
echo ""
echo "📊 Test Summary"
echo "==============="
echo "✅ Import tests: PASSED"
echo "✅ Signal generation: PASSED"
echo "✅ Risk management: PASSED"
echo "✅ Dashboard components: PASSED"
echo ""
echo "🎉 All tests completed successfully!"
echo ""
echo "🚀 Ready to use Trading AI system:"
echo "   python main.py dashboard    # Launch web interface"
echo "   python main.py signals      # Generate signals"
echo "   python main.py train        # Train AI models"
echo ""