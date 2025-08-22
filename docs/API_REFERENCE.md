# 🔧 Trading AI API Reference

Esta documentación describe las principales clases y métodos del sistema Trading AI.

## 📊 Core Classes

### SignalGenerator

La clase principal para generar señales de trading usando IA.

```python
from src.signals.signal_generator import SignalGenerator

# Inicializar
generator = SignalGenerator(account_balance=100000)

# Entrenar modelo para un símbolo
success = generator.train_model('AAPL', training_period='2y')

# Generar señal individual
signal = generator.generate_signal('AAPL')

# Generar múltiples señales
signals = generator.generate_multiple_signals(['AAPL', 'GOOGL', 'MSFT'])
```

#### Métodos

##### `train_model(symbol, training_period='2y')`
Entrena el modelo de RL para un símbolo específico.

**Parámetros:**
- `symbol` (str): Símbolo de trading (ej: 'AAPL', 'BTC-USD')
- `training_period` (str): Período de datos para entrenamiento

**Retorna:**
- `bool`: True si el entrenamiento fue exitoso

##### `generate_signal(symbol, use_realtime=True)`
Genera una señal de trading para un símbolo.

**Parámetros:**
- `symbol` (str): Símbolo de trading
- `use_realtime` (bool): Usar datos en tiempo real

**Retorna:**
- `TradingSignal`: Objeto con la señal generada

##### `generate_multiple_signals(symbols=None)`
Genera señales para múltiples símbolos.

**Parámetros:**
- `symbols` (List[str]): Lista de símbolos. Si None, usa símbolos por defecto

**Retorna:**
- `List[TradingSignal]`: Lista de señales ordenadas por confianza

---

### TradingSignal

Clase de datos que contiene información completa de una señal.

```python
@dataclass
class TradingSignal:
    symbol: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    current_price: float
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size: float
    risk_reward_ratio: float
    max_loss: float
    timestamp: datetime
    reasoning: str
    technical_analysis: Dict
    risk_assessment: Dict
    market_conditions: Dict
```

#### Propiedades

- `symbol`: Símbolo del activo
- `signal`: Tipo de señal ('BUY', 'SELL', 'HOLD')
- `confidence`: Confianza del modelo (0.0-1.0)
- `current_price`: Precio actual del activo
- `entry_price`: Precio de entrada recomendado
- `stop_loss`: Precio de stop loss
- `take_profit`: Precio objetivo
- `position_size`: Tamaño de posición recomendado
- `risk_reward_ratio`: Ratio riesgo-beneficio
- `max_loss`: Pérdida máxima potencial
- `reasoning`: Explicación de la señal
- `technical_analysis`: Análisis técnico detallado
- `risk_assessment`: Evaluación de riesgo
- `market_conditions`: Condiciones del mercado

---

### RiskManager

Gestión integral de riesgo para las operaciones.

```python
from src.risk.risk_manager import RiskManager

# Inicializar
risk_manager = RiskManager(initial_capital=100000)

# Calcular tamaño de posición
position_size = risk_manager.calculate_position_size(
    current_price=100.0,
    stop_loss_price=95.0,
    account_balance=100000.0,
    confidence=0.8
)

# Generar evaluación completa de riesgo
risk_metrics = risk_manager.generate_risk_assessment(
    signal='BUY',
    current_price=100.0,
    market_data=df,
    confidence=0.8,
    account_balance=100000
)
```

#### Métodos

##### `calculate_position_size(current_price, stop_loss_price, account_balance, confidence=1.0)`
Calcula el tamaño óptimo de posición.

**Parámetros:**
- `current_price` (float): Precio actual del activo
- `stop_loss_price` (float): Precio de stop loss
- `account_balance` (float): Balance disponible
- `confidence` (float): Confianza de la señal (0-1)

**Retorna:**
- `float`: Tamaño de posición recomendado

##### `calculate_stop_loss(entry_price, signal_direction, atr, volatility=None)`
Calcula stop loss dinámico basado en volatilidad.

**Parámetros:**
- `entry_price` (float): Precio de entrada
- `signal_direction` (str): 'BUY' o 'SELL'
- `atr` (float): Average True Range
- `volatility` (float): Medida adicional de volatilidad

**Retorna:**
- `float`: Precio de stop loss

##### `generate_risk_assessment(signal, current_price, market_data, confidence, account_balance)`
Genera evaluación completa de riesgo.

**Retorna:**
- `RiskMetrics`: Objeto con todas las métricas de riesgo

---

### TechnicalIndicators

Cálculo de indicadores técnicos.

```python
from src.indicators.technical_indicators import TechnicalIndicators

# Calcular indicador individual
rsi = TechnicalIndicators.rsi(data['close'], window=14)
macd = TechnicalIndicators.macd(data['close'])

# Calcular todos los indicadores
data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)
```

#### Métodos Estáticos

##### `sma(data, window=20)`
Simple Moving Average
- `data` (pd.Series): Serie de precios
- `window` (int): Período de cálculo

##### `ema(data, window=20)`
Exponential Moving Average

##### `rsi(data, window=14)`
Relative Strength Index

##### `macd(data, fast=12, slow=26, signal=9)`
Moving Average Convergence Divergence
- Retorna dict con 'macd', 'signal', 'histogram'

##### `bollinger_bands(data, window=20, std_dev=2)`
Bollinger Bands
- Retorna dict con 'upper', 'middle', 'lower'

##### `calculate_all_indicators(data)`
Calcula todos los indicadores disponibles.

**Parámetros:**
- `data` (pd.DataFrame): DataFrame con columnas OHLCV

**Retorna:**
- `pd.DataFrame`: Data original + todos los indicadores

---

### MarketDataFetcher

Obtención y procesamiento de datos de mercado.

```python
from src.data.data_fetcher import MarketDataFetcher

# Inicializar
fetcher = MarketDataFetcher()

# Obtener datos históricos
data = fetcher.fetch_data('AAPL', period='1y', interval='1h')

# Obtener múltiples símbolos
data_dict = fetcher.fetch_multiple_symbols(['AAPL', 'GOOGL'])

# Datos en tiempo real
realtime = fetcher.get_realtime_data('AAPL')
```

#### Métodos

##### `fetch_data(symbol, period='1y', interval=None)`
Obtiene datos históricos para un símbolo.

**Parámetros:**
- `symbol` (str): Símbolo de trading
- `period` (str): Período de datos ('1d', '1mo', '1y', etc.)
- `interval` (str): Intervalo de datos ('1m', '1h', '1d', etc.)

**Retorna:**
- `pd.DataFrame`: Datos OHLCV

##### `fetch_multiple_symbols(symbols=None, period='1y')`
Obtiene datos para múltiples símbolos.

**Retorna:**
- `dict`: {símbolo: DataFrame}

##### `preprocess_data(data)`
Preprocesa datos para análisis.

**Retorna:**
- `pd.DataFrame`: Datos procesados con retornos y volatilidad

---

### Backtester

Framework completo de backtesting.

```python
from src.backtesting.backtester import Backtester

# Inicializar
backtester = Backtester(initial_capital=100000, commission=0.001)

# Ejecutar backtest
results = backtester.run_backtest(
    symbols=['AAPL', 'GOOGL'],
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# Generar reporte
report = backtester.generate_report(results)

# Crear gráficos
backtester.plot_results(results, save_path='backtest_charts.png')
```

#### Métodos

##### `run_backtest(symbols, start_date, end_date, rebalance_frequency='daily')`
Ejecuta backtest completo.

**Parámetros:**
- `symbols` (List[str]): Símbolos a probar
- `start_date` (str): Fecha inicio 'YYYY-MM-DD'
- `end_date` (str): Fecha fin 'YYYY-MM-DD'
- `rebalance_frequency` (str): Frecuencia de rebalanceo

**Retorna:**
- `BacktestResults`: Resultados completos del backtest

##### `generate_report(results)`
Genera reporte textual de resultados.

##### `plot_results(results, save_path=None)`
Genera visualizaciones de resultados.

---

### PerformanceMonitor

Monitoreo y seguimiento de rendimiento de señales.

```python
from src.monitoring.performance_monitor import PerformanceMonitor

# Inicializar
monitor = PerformanceMonitor()

# Registrar señal
signal_id = monitor.log_signal(trading_signal)

# Actualizar precios actuales
monitor.update_active_signals({'AAPL': 175.50, 'GOOGL': 2800.00})

# Obtener datos del dashboard
dashboard_data = monitor.get_dashboard_data()

# Generar reportes
daily_report = monitor.generate_daily_report()
weekly_report = monitor.generate_weekly_report()
```

#### Métodos

##### `log_signal(signal)`
Registra una nueva señal para seguimiento.

**Parámetros:**
- `signal` (TradingSignal): Señal a registrar

**Retorna:**
- `str`: ID único de la señal

##### `update_active_signals(current_prices)`
Actualiza precios de señales activas.

**Parámetros:**
- `current_prices` (Dict[str, float]): {símbolo: precio_actual}

##### `get_dashboard_data()`
Obtiene datos para dashboard de rendimiento.

**Retorna:**
- `Dict`: Datos formateados para dashboard

---

## 📱 Dashboard API

### TradingDashboard

Interface web principal del sistema.

```python
from src.ui.dashboard import TradingDashboard

# Inicializar y ejecutar
dashboard = TradingDashboard()
dashboard.run()
```

#### Métodos Principales

##### `render_live_signals()`
Renderiza tab de señales en vivo.

##### `render_analysis()`
Renderiza tab de análisis técnico.

##### `render_backtesting()`
Renderiza tab de backtesting.

##### `render_settings()`
Renderiza tab de configuración.

---

## 🛠️ Configuration API

### TradingConfig

Configuración centralizada del sistema.

```python
from config.config import config

# Acceder a configuración
symbols = config.DEFAULT_SYMBOLS
risk_per_trade = config.MAX_RISK_PER_TRADE
learning_rate = config.LEARNING_RATE

# Modificar configuración
config.CONFIDENCE_THRESHOLD = 0.8
config.LOOKBACK_PERIOD = 150
```

#### Parámetros Principales

##### Risk Management
- `MAX_RISK_PER_TRADE`: Riesgo máximo por operación (default: 0.02)
- `STOP_LOSS_MULTIPLIER`: Multiplicador para stop loss (default: 2.0)
- `TAKE_PROFIT_MULTIPLIER`: Multiplicador para take profit (default: 3.0)

##### Model Parameters
- `LEARNING_RATE`: Tasa de aprendizaje (default: 0.001)
- `BATCH_SIZE`: Tamaño del batch (default: 32)
- `EPOCHS`: Épocas de entrenamiento (default: 100)
- `LOOKBACK_PERIOD`: Períodos históricos (default: 100)

##### Data Settings
- `DEFAULT_SYMBOLS`: Símbolos por defecto
- `DATA_INTERVAL`: Intervalo de datos (default: '1h')
- `CONFIDENCE_THRESHOLD`: Umbral de confianza (default: 0.7)

---

## 🔄 Workflow Examples

### Ejemplo 1: Flujo Básico de Señales

```python
# 1. Inicializar componentes
from src.signals.signal_generator import SignalGenerator
from src.monitoring.performance_monitor import PerformanceMonitor

generator = SignalGenerator(account_balance=100000)
monitor = PerformanceMonitor()

# 2. Entrenar modelo si es necesario
if not generator.trained_models.get('AAPL'):
    generator.train_model('AAPL')

# 3. Generar señal
signal = generator.generate_signal('AAPL')

if signal and signal.signal != 'HOLD':
    # 4. Registrar para seguimiento
    signal_id = monitor.log_signal(signal)
    
    # 5. Mostrar señal
    print(f"Señal: {signal.signal}")
    print(f"Confianza: {signal.confidence:.1%}")
    print(f"Precio objetivo: ${signal.take_profit:.2f}")
    print(f"Stop loss: ${signal.stop_loss:.2f}")
```

### Ejemplo 2: Backtesting Personalizado

```python
from src.backtesting.backtester import Backtester

# 1. Configurar backtester
backtester = Backtester(
    initial_capital=50000,
    commission=0.001  # 0.1%
)

# 2. Ejecutar backtest para portfolio tech
results = backtester.run_backtest(
    symbols=['AAPL', 'GOOGL', 'MSFT', 'AMZN'],
    start_date='2023-01-01',
    end_date='2024-01-01'
)

# 3. Analizar resultados
print(f"Retorno total: {results.total_return_percentage:.2f}%")
print(f"Sharpe ratio: {results.sharpe_ratio:.3f}")
print(f"Max drawdown: {results.max_drawdown_percentage:.2f}%")
print(f"Win rate: {results.win_rate:.1%}")

# 4. Generar reporte detallado
report = backtester.generate_report(results)
with open('backtest_tech_portfolio.txt', 'w') as f:
    f.write(report)
```

### Ejemplo 3: Análisis de Riesgo Avanzado

```python
from src.risk.risk_manager import RiskManager
from src.data.data_fetcher import MarketDataFetcher
from src.indicators.technical_indicators import TechnicalIndicators

# 1. Obtener datos y calcular indicadores
fetcher = MarketDataFetcher()
data = fetcher.fetch_data('AAPL', period='3mo')
data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)

# 2. Inicializar risk manager
risk_manager = RiskManager(initial_capital=100000)

# 3. Evaluar condiciones de mercado
market_risk = risk_manager.assess_market_risk(data_with_indicators)
print(f"Riesgo general del mercado: {market_risk['overall_risk']:.1%}")

# 4. Calcular métricas para una señal hipotética
current_price = data_with_indicators['close'].iloc[-1]
risk_metrics = risk_manager.generate_risk_assessment(
    signal='BUY',
    current_price=current_price,
    market_data=data_with_indicators,
    confidence=0.75,
    account_balance=100000
)

# 5. Validar señal
validation = risk_manager.validate_signal(
    'BUY', risk_metrics, data_with_indicators
)

if validation['is_valid']:
    print("✅ Señal válida para ejecución")
    print(f"Tamaño de posición: {risk_metrics.position_size:.0f}")
    print(f"Riesgo máximo: ${risk_metrics.max_loss:.2f}")
else:
    print("❌ Señal rechazada por riesgo alto")
    print(f"Advertencias: {validation['warnings']}")
```

---

## 🚨 Error Handling

### Excepciones Comunes

```python
try:
    signal = generator.generate_signal('INVALID_SYMBOL')
except Exception as e:
    print(f"Error generando señal: {e}")

try:
    data = fetcher.fetch_data('AAPL', period='invalid_period')
except Exception as e:
    print(f"Error obteniendo datos: {e}")
```

### Logging

```python
from loguru import logger

# El sistema usa loguru para logging
# Los logs se guardan en logs/trading_ai.log

# Configurar nivel de log
import sys
sys.path.append('src')
from config.config import config

config.LOG_LEVEL = "DEBUG"  # DEBUG, INFO, WARNING, ERROR
```

---

## 🔗 Integration Examples

### Ejemplo: Integración con Telegram

```python
# Extensión para enviar señales por Telegram
import requests

def send_telegram_signal(signal: TradingSignal, bot_token: str, chat_id: str):
    message = f"""
🤖 TRADING AI SIGNAL
    
Symbol: {signal.symbol}
Signal: {signal.signal}
Confidence: {signal.confidence:.1%}
Price: ${signal.current_price:.2f}
Stop Loss: ${signal.stop_loss:.2f}
Take Profit: ${signal.take_profit:.2f}

Reasoning: {signal.reasoning}
    """
    
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    requests.post(url, data={
        'chat_id': chat_id,
        'text': message
    })
```

### Ejemplo: Exportar a CSV

```python
import csv
from datetime import datetime

def export_signals_to_csv(signals: List[TradingSignal], filename: str):
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = [
            'timestamp', 'symbol', 'signal', 'confidence', 
            'current_price', 'stop_loss', 'take_profit', 
            'position_size', 'max_loss', 'reasoning'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for signal in signals:
            writer.writerow({
                'timestamp': signal.timestamp.isoformat(),
                'symbol': signal.symbol,
                'signal': signal.signal,
                'confidence': signal.confidence,
                'current_price': signal.current_price,
                'stop_loss': signal.stop_loss,
                'take_profit': signal.take_profit,
                'position_size': signal.position_size,
                'max_loss': signal.max_loss,
                'reasoning': signal.reasoning
            })
```

---

Esta API reference te permite integrar y extender el sistema Trading AI según tus necesidades específicas. Cada componente está diseñado para ser modular y fácilmente personalizable.