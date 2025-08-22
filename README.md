# 🤖 Trading AI - Análisis de Mercado y Generación de Señales

Un sistema integral de trading impulsado por IA que analiza mercados financieros y genera señales de trading usando aprendizaje por refuerzo. El sistema proporciona recomendaciones para ejecución manual, asegurando una separación completa entre análisis y ejecución de operaciones.

## 🎯 Características

### Capacidades Principales
- **Generación de Señales con IA**: Utiliza Deep Q-Network (DQN) de aprendizaje por refuerzo para análisis inteligente de mercados
- **Soporte Multi-Activos**: Analiza acciones, ETFs, forex, criptomonedas y commodities
- **Análisis Técnico Integral**: Más de 15 indicadores técnicos incluyendo RSI, MACD, Bandas de Bollinger, etc.
- **Gestión Avanzada de Riesgo**: Dimensionado dinámico de posición, cálculos de stop-loss y take-profit
- **Datos de Mercado en Tiempo Real**: Integración con Yahoo Finance para datos de mercado en vivo
- **Framework de Backtesting**: Pruebas integrales de rendimiento histórico
- **Dashboard Interactivo**: Interfaz web para monitoreo y análisis de señales
- **Monitoreo de Rendimiento**: Seguimiento de precisión de señales y rendimiento de trading a lo largo del tiempo

### Características de Seguridad y Riesgo
- **Sin Trading Directo**: El sistema solo genera señales - no hay ejecución automática de órdenes
- **Evaluación de Riesgo**: Análisis integral de riesgo para cada señal
- **Dimensionado de Posición**: Dimensionado inteligente de posición basado en saldo de cuenta y tolerancia al riesgo
- **Análisis de Condiciones de Mercado**: Se adapta a condiciones cambiantes del mercado
- **Validación de Señales**: Validación multi-capa antes de la generación de señales

## 🏗️ Arquitectura del Sistema

```
trading-ai/
├── src/
│   ├── data/           # Market data fetching and processing
│   ├── indicators/     # Technical indicators calculation
│   ├── models/         # Reinforcement learning models
│   ├── risk/           # Risk management system
│   ├── signals/        # Signal generation engine
│   ├── backtesting/    # Backtesting framework
│   ├── ui/             # Web dashboard interface
│   └── monitoring/     # Performance tracking
├── config/             # Configuration files
├── data/               # Data storage
├── logs/               # System logs
└── tests/              # Unit tests
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd trading-ai

# Install dependencies
pip install -r requirements.txt

# Install TA-Lib (for technical indicators)
# On Ubuntu/Debian:
sudo apt-get install libta-lib0-dev
pip install TA-Lib

# On macOS:
brew install ta-lib
pip install TA-Lib
```

### 2. Configuration

Edit `config/config.py` to customize:
- Default symbols to analyze
- Risk management parameters
- Model training settings
- Data sources and intervals

### 3. Running the System

#### Launch Interactive Dashboard
```bash
python main.py dashboard
```

#### Generate Live Signals
```bash
# Generate signals for default symbols
python main.py signals

# Generate signals for specific symbols
python main.py signals --symbols AAPL GOOGL TSLA BTC-USD
```

#### Run Backtesting
```bash
# Backtest with default parameters
python main.py backtest

# Custom backtest
python main.py backtest --symbols AAPL MSFT --start-date 2023-01-01 --end-date 2024-01-01
```

#### Train AI Models
```bash
# Train models for all default symbols
python main.py train

# Train models for specific symbols
python main.py train --symbols AAPL GOOGL
```

#### Generate Performance Report
```bash
python main.py report
```

## 📊 Dashboard Features

The web dashboard provides:

### 🎯 Live Signals Tab
- Real-time signal generation for selected assets
- Signal confidence scores and reasoning
- Risk metrics and position sizing recommendations
- Technical analysis summaries
- Market condition assessments

### 📈 Analysis Tab
- Interactive price charts with technical indicators
- Bollinger Bands, moving averages, RSI, MACD
- Volume and momentum analysis
- Multi-timeframe analysis

### 🔄 Backtesting Tab
- Historical strategy performance testing
- Comprehensive performance metrics
- Portfolio value tracking
- Trade analysis and statistics
- Downloadable reports

### ⚙️ Settings Tab
- Risk management configuration
- Model training interface
- Parameter optimization
- System monitoring

## 🧠 AI Model Details

### Reinforcement Learning Architecture
- **Algorithm**: Deep Q-Network (DQN)
- **State Space**: Technical indicators + market conditions + portfolio info
- **Action Space**: BUY, SELL, HOLD
- **Reward Function**: Portfolio performance with risk adjustments
- **Training**: Experience replay with target network updates

### Technical Indicators Used
- Moving Averages (SMA, EMA)
- Relative Strength Index (RSI)
- Moving Average Convergence Divergence (MACD)
- Bollinger Bands
- Stochastic Oscillator
- Average True Range (ATR)
- Williams %R
- Commodity Channel Index (CCI)
- Volume analysis
- Price momentum and volatility

## 📈 Signal Generation Process

1. **Data Collection**: Fetch latest market data and calculate indicators
2. **AI Analysis**: RL model processes market state and generates raw signal
3. **Risk Assessment**: Comprehensive risk analysis and position sizing
4. **Signal Validation**: Multi-layer validation against risk rules
5. **Signal Output**: Final recommendation with reasoning and metrics

## 🛡️ Risk Management

### Position Sizing
- Maximum risk per trade (default: 2% of account)
- Confidence-adjusted position sizing
- Market volatility considerations
- Account balance protection

### Stop Loss & Take Profit
- Dynamic ATR-based stop losses
- Risk-reward ratio optimization (min 1.5:1)
- Market condition adjustments
- Volatility-aware positioning

### Portfolio Risk
- Concentration risk monitoring
- Correlation analysis
- Maximum drawdown protection
- Real-time risk assessment

## 📊 Performance Monitoring

### Tracking Metrics
- Signal accuracy and win rate
- Average profit/loss per signal
- Risk-adjusted returns (Sharpe ratio)
- Maximum drawdown
- Average trade duration

### Reporting
- Daily, weekly, and monthly reports
- Performance by symbol and confidence level
- Signal quality analysis
- Model performance tracking

## 🔧 Configuration Options

### Risk Management Settings
```python
MAX_RISK_PER_TRADE = 0.02  # 2% max risk per trade
STOP_LOSS_MULTIPLIER = 2.0  # ATR multiplier for stop loss
TAKE_PROFIT_MULTIPLIER = 3.0  # Risk-reward ratio
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for signals
```

### Model Parameters
```python
LEARNING_RATE = 0.001
BATCH_SIZE = 32
EPOCHS = 100
LOOKBACK_PERIOD = 100  # Historical data points
```

### Data Settings
```python
DEFAULT_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"]
DATA_INTERVAL = "1h"  # 1m, 5m, 15m, 30m, 1h, 1d
```

## 📝 Usage Examples

### Example 1: Generate Signals for Crypto
```bash
python main.py signals --symbols BTC-USD ETH-USD ADA-USD
```

### Example 2: Backtest Tech Stocks
```bash
python main.py backtest --symbols AAPL GOOGL MSFT AMZN --start-date 2023-01-01
```

### Example 3: Train Model for Forex
```bash
python main.py train --symbols EURUSD=X GBPUSD=X USDJPY=X
```

## 🔍 Interpreting Signals

### Signal Types
- **🟢 BUY**: AI recommends long position
- **🔴 SELL**: AI recommends short position  
- **⚪ HOLD**: No clear directional bias

### Confidence Levels
- **High (>75%)**: Strong signal with favorable conditions
- **Medium (50-75%)**: Moderate signal, proceed with caution
- **Low (<50%)**: Weak signal, may want to wait for better opportunities

### Risk Metrics
- **Position Size**: Recommended number of shares/units
- **Stop Loss**: Price level to exit if trade goes against you
- **Take Profit**: Target price for profit-taking
- **Max Loss**: Maximum potential loss in dollars
- **Risk/Reward**: Ratio of potential profit to potential loss

## ⚠️ Important Disclaimers

1. **Not Financial Advice**: This system is for educational and research purposes only
2. **No Guaranteed Returns**: Past performance does not predict future results
3. **Manual Execution**: All trading decisions and executions are manual
4. **Risk Warning**: Trading involves substantial risk of loss
5. **Backtesting Limitations**: Historical results may not reflect future performance

## 🛠️ Development

### Adding New Indicators
```python
# In src/indicators/technical_indicators.py
@staticmethod
def custom_indicator(data: pd.Series, window: int = 14) -> pd.Series:
    # Your indicator calculation
    return result
```

### Extending the RL Model
```python
# In src/models/rl_agent.py
# Modify state space, action space, or reward function
# Update network architecture or training parameters
```

### Custom Risk Rules
```python
# In src/risk/risk_manager.py
# Add custom validation rules or risk calculations
```

## 📚 Dependencies

- **numpy**: Numerical computations
- **pandas**: Data manipulation
- **tensorflow**: Deep learning framework
- **yfinance**: Market data retrieval
- **streamlit**: Web dashboard
- **plotly**: Interactive charts
- **ta-lib**: Technical analysis library
- **loguru**: Logging system
- **pydantic**: Configuration management

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## 📄 License

This project is for educational purposes. Please review the license file for details.

## 🆘 Support

- Check the logs in the `logs/` directory for debugging
- Review configuration in `config/config.py`
- Ensure all dependencies are properly installed
- Verify market data access (Yahoo Finance API)

---

**⚠️ Risk Warning**: Trading financial instruments involves substantial risk of loss. This system is designed for analysis and education only. Always conduct your own research and consider consulting with a financial advisor before making investment decisions.