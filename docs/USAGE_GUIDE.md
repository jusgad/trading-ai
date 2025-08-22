# 📖 Trading AI Usage Guide

Esta guía te ayudará a usar el sistema Trading AI paso a paso, desde la instalación hasta la generación de señales avanzadas.

## 🚀 Instalación Rápida

### Método 1: Script Automático (Recomendado)
```bash
# Clonar el repositorio
git clone <repository-url>
cd trading-ai

# Ejecutar script de instalación
./scripts/install.sh
```

### Método 2: Instalación Manual
```bash
# Crear entorno virtual
python3 -m venv trading-ai-env
source trading-ai-env/bin/activate

# Instalar dependencias
pip install -r requirements.txt

# Instalar TA-Lib (Linux/Ubuntu)
sudo apt-get install libta-lib0-dev
pip install TA-Lib

# Instalar TA-Lib (macOS)
brew install ta-lib
pip install TA-Lib
```

## 🎯 Primeros Pasos

### 1. Configuración Inicial
```bash
# Copiar archivo de configuración
cp .env.example .env

# Editar configuración (opcional)
nano .env
```

### 2. Verificar Instalación
```bash
# Ejecutar tests
./scripts/run_tests.sh
```

### 3. Generar Primeras Señales
```bash
# Generar señales para acciones populares
python main.py signals --symbols AAPL GOOGL MSFT

# Generar señales para criptomonedas
python main.py signals --symbols BTC-USD ETH-USD ADA-USD
```

## 🖥️ Usando el Dashboard Web

### Lanzar Dashboard
```bash
# Método 1: Script directo
./scripts/start_dashboard.sh

# Método 2: Comando directo
python main.py dashboard
```

### Navegación del Dashboard

#### 🎯 Tab "Live Signals"
- **Función**: Genera y muestra señales de trading en tiempo real
- **Cómo usar**:
  1. Selecciona símbolos en la barra lateral
  2. Ajusta configuración de riesgo
  3. Haz clic en "Refresh Signals"
  4. Revisa las señales generadas

#### 📊 Tab "Analysis"
- **Función**: Análisis técnico detallado con gráficos interactivos
- **Cómo usar**:
  1. Selecciona un símbolo del dropdown
  2. Revisa el gráfico de precios con indicadores
  3. Analiza RSI, MACD y otros indicadores
  4. Usa zoom y herramientas interactivas

#### 🔄 Tab "Backtesting"
- **Función**: Prueba estrategias con datos históricos
- **Cómo usar**:
  1. Configura fechas de inicio y fin
  2. Selecciona símbolos para probar
  3. Ajusta capital inicial
  4. Haz clic en "Run Backtest"
  5. Revisa métricas de rendimiento

#### ⚙️ Tab "Settings"
- **Función**: Configuración avanzada del sistema
- **Cómo usar**:
  1. Ajusta parámetros de riesgo
  2. Modifica configuración del modelo AI
  3. Entrena modelos para nuevos símbolos
  4. Guarda configuraciones

## 💡 Comandos Principales

### Generación de Señales
```bash
# Señales básicas
python main.py signals

# Señales específicas
python main.py signals --symbols AAPL GOOGL TSLA

# Señales crypto
python main.py signals --symbols BTC-USD ETH-USD DOT-USD

# Señales forex (formato especial)
python main.py signals --symbols EURUSD=X GBPUSD=X USDJPY=X
```

### Backtesting Avanzado
```bash
# Backtest básico
python main.py backtest

# Backtest personalizado
python main.py backtest \
  --symbols AAPL MSFT GOOGL \
  --start-date 2023-01-01 \
  --end-date 2024-01-01

# Backtest de portfolio diversificado
python main.py backtest \
  --symbols AAPL BTC-USD EURUSD=X GLD \
  --start-date 2022-01-01
```

### Entrenamiento de Modelos
```bash
# Entrenar todos los modelos por defecto
python main.py train

# Entrenar modelos específicos
python main.py train --symbols AAPL GOOGL

# Entrenar con logging detallado
python main.py train --log-level DEBUG
```

### Reportes de Rendimiento
```bash
# Reporte semanal
python main.py report

# Ver dashboard de rendimiento
python main.py dashboard
# Ir a Settings > Performance Monitoring
```

## 🎛️ Configuración Avanzada

### Parámetros de Riesgo
```python
# En .env o config/config.py
MAX_RISK_PER_TRADE = 0.02    # 2% máximo por operación
STOP_LOSS_MULTIPLIER = 2.0   # Multiplicador ATR para stop loss
TAKE_PROFIT_MULTIPLIER = 3.0 # Ratio riesgo-beneficio
CONFIDENCE_THRESHOLD = 0.7   # Confianza mínima para señales
```

### Parámetros del Modelo AI
```python
LEARNING_RATE = 0.001       # Tasa de aprendizaje
BATCH_SIZE = 32            # Tamaño del batch
EPOCHS = 100               # Épocas de entrenamiento
LOOKBACK_PERIOD = 100      # Períodos históricos a considerar
```

### Símbolos y Datos
```python
DEFAULT_SYMBOLS = [
    "AAPL", "GOOGL", "MSFT",     # Acciones
    "BTC-USD", "ETH-USD",        # Crypto
    "EURUSD=X", "GBPUSD=X",      # Forex
    "GLD", "SLV"                 # Commodities
]
DATA_INTERVAL = "1h"            # Intervalo de datos
```

## 📊 Interpretando las Señales

### Tipos de Señal
- **🟢 BUY**: Recomendación de compra (posición larga)
- **🔴 SELL**: Recomendación de venta (posición corta)
- **⚪ HOLD**: Mantener posición actual

### Métricas Clave
- **Confidence**: Confianza del AI (0-100%)
- **Position Size**: Tamaño recomendado de posición
- **Stop Loss**: Precio de stop loss sugerido
- **Take Profit**: Precio objetivo de beneficio
- **Risk/Reward**: Ratio riesgo-beneficio
- **Max Loss**: Pérdida máxima potencial en $

### Ejemplo de Interpretación
```
Symbol: AAPL
Signal: 🟢 BUY (85% confidence)
Current Price: $175.50
Stop Loss: $170.00
Take Profit: $185.00
Position Size: 57 shares
Risk/Reward: 1.7
Max Loss: $313.50

Reasoning: AI model recommends BUY with 85% confidence. 
Market trend: UPTREND. Favorable risk/reward ratio of 1.7. 
RSI indicates NEUTRAL conditions.
```

**Interpretación**: 
- Señal fuerte de compra (85% confianza)
- Riesgo controlado ($313.50 máximo)
- Buen ratio riesgo-beneficio (1.7:1)
- Condiciones de mercado favorables

## 🔍 Análisis Técnico Detallado

### Indicadores Disponibles
1. **RSI**: Sobrecompra (>70) / Sobreventa (<30)
2. **MACD**: Momentum y cambios de tendencia
3. **Bollinger Bands**: Volatilidad y niveles de precio
4. **Moving Averages**: Tendencia general
5. **Stochastic**: Momentum de precio
6. **ATR**: Volatilidad y rangos de precio
7. **Volume**: Confirmación de movimientos

### Condiciones de Mercado
- **Trend**: UPTREND, DOWNTREND, SIDEWAYS
- **Volatility**: HIGH, NORMAL, LOW
- **Volume**: HIGH, NORMAL, LOW
- **Momentum**: BULLISH, BEARISH, NEUTRAL

## 🧪 Backtesting Detallado

### Métricas de Rendimiento
- **Total Return**: Retorno total en %
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Max Drawdown**: Máxima pérdida desde pico
- **Win Rate**: Porcentaje de operaciones ganadoras
- **Profit Factor**: Ratio beneficio/pérdida brutos
- **Average Trade Duration**: Duración promedio

### Ejemplo de Resultados
```
=== BACKTEST REPORT ===
Initial Capital: $100,000.00
Final Capital: $127,350.00
Total Return: $27,350.00 (27.35%)
Benchmark Return: 7.00%
Outperformance: 20.35%

Max Drawdown: $8,420.00 (8.42%)
Sharpe Ratio: 1.834

Total Trades: 45
Winning Trades: 28 (62.2%)
Losing Trades: 17
Profit Factor: 2.1
```

## 🎯 Casos de Uso Específicos

### Caso 1: Trading Diario de Acciones
```bash
# 1. Generar señales matutinas
python main.py signals --symbols AAPL GOOGL MSFT AMZN

# 2. Revisar análisis técnico
python main.py dashboard
# Ir a Analysis tab

# 3. Ejecutar manualmente las operaciones recomendadas
```

### Caso 2: Inversión en Criptomonedas
```bash
# 1. Entrenar modelos crypto
python main.py train --symbols BTC-USD ETH-USD ADA-USD

# 2. Generar señales crypto
python main.py signals --symbols BTC-USD ETH-USD ADA-USD DOT-USD

# 3. Backtesting crypto
python main.py backtest --symbols BTC-USD ETH-USD --start-date 2023-01-01
```

### Caso 3: Portfolio Diversificado
```bash
# Señales para portfolio mixto
python main.py signals --symbols \
  AAPL GOOGL \      # Tech stocks
  BTC-USD ETH-USD \ # Crypto
  GLD SLV \         # Precious metals
  EURUSD=X          # Forex
```

### Caso 4: Análisis de Forex
```bash
# Señales principales de forex
python main.py signals --symbols \
  EURUSD=X GBPUSD=X USDJPY=X \
  AUDUSD=X USDCAD=X USDCHF=X
```

## 🛡️ Mejores Prácticas de Riesgo

### 1. Gestión de Capital
- Nunca arriesgues más del 2% por operación
- Diversifica entre diferentes activos
- Mantén reserva de efectivo (20-30%)

### 2. Seguimiento de Señales
- Revisa siempre el reasoning del AI
- Verifica condiciones de mercado
- Confirma con análisis técnico

### 3. Ejecución Manual
- Usa órdenes stop-loss siempre
- Considera deslizamientos de precio
- Revisa spread y comisiones

### 4. Monitoreo Continuo
- Genera reportes de rendimiento semanales
- Ajusta parámetros según resultados
- Mantén registro de operaciones

## 🔧 Solución de Problemas

### Error: "No data found"
```bash
# Verificar conexión a internet
ping google.com

# Verificar símbolo válido
python main.py signals --symbols AAPL  # Debería funcionar

# Reintentar con período más corto
python main.py signals --symbols YOUR_SYMBOL
```

### Error: "TA-Lib not found"
```bash
# Ubuntu/Debian
sudo apt-get install libta-lib0-dev
pip install TA-Lib

# macOS
brew install ta-lib
pip install TA-Lib

# Verificar instalación
python -c "import talib; print('TA-Lib OK')"
```

### Dashboard no carga
```bash
# Verificar Streamlit
pip install streamlit

# Verificar puerto
netstat -an | grep 8501

# Reiniciar dashboard
./scripts/start_dashboard.sh
```

### Modelos no entrenan
```bash
# Verificar TensorFlow
python -c "import tensorflow as tf; print(tf.__version__)"

# Entrenar con logging detallado
python main.py train --log-level DEBUG

# Verificar datos suficientes
python main.py signals --symbols AAPL  # Probar con símbolo conocido
```

## 📈 Optimización de Rendimiento

### 1. Ajuste de Parámetros
- Aumenta `CONFIDENCE_THRESHOLD` para señales más selectivas
- Ajusta `LOOKBACK_PERIOD` según volatilidad del mercado
- Modifica ratios de riesgo según tu tolerancia

### 2. Selección de Activos
- Enfócate en activos con buena liquidez
- Evita penny stocks o criptos muy volátiles
- Considera correlaciones entre activos

### 3. Timing del Mercado
- Genera señales en horarios de mayor actividad
- Considera eventos económicos importantes
- Evita períodos de baja liquidez

## 🎓 Recursos Adicionales

### Documentación Técnica
- `README.md`: Guía general del sistema
- `config/config.py`: Todos los parámetros configurables
- `src/`: Código fuente detallado

### Logs y Monitoreo
- `logs/trading_ai.log`: Log principal del sistema
- Dashboard > Settings: Monitoreo de rendimiento
- `python main.py report`: Reportes detallados

### Comunidad y Soporte
- Issues: Reportar problemas en GitHub
- Discusiones: Compartir estrategias y consejos
- Actualizaciones: Seguir releases para nuevas funciones

---

**⚠️ Disclaimer**: Este sistema es para análisis y educación. Siempre revisa las señales y ejecuta operaciones manualmente. El trading conlleva riesgos significativos.