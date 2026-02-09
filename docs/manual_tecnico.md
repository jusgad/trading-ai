# Manual Técnico - Trading AI

## Visión General del Sistema
Este sistema es una plataforma de trading algorítmico avanzada que utiliza Deep Learning (Transformers) para predecir movimientos de mercado. Se ha diseñado modularmente para facilitar la experimentación, el mantenimiento y la escalabilidad.

## Arquitectura del Sistema

### 1. Ingesta y Procesamiento de Datos (`src/data`)
El sistema comienza obteniendo datos históricos (OHLCV) a través de `yfinance`.
- **`data_fetcher.py`**: Gestiona la descarga de datos.
- **`features.py`**: Motor de Ingeniería de Características.
    - **Indicadores Técnicos (`pandas-ta`)**:
        - **Momento**: RSI (Índice de Fuerza Relativa), MACD (Convergencia/Divergencia de Medias Móviles).
        - **Volatilidad**: Bandas de Bollinger, ATR (Rango Verdadero Promedio).
        - **Volumen**: OBV (On-Balance Volume).
    - **Normalización**: Se utiliza `RobustScaler` de Scikit-Learn. Este escalador es vital para manejar "outliers" (valores atípicos) comunes en finanzas. Se ajusta (`fit`) SOLO con datos de entrenamiento para evitar *Data Leakage* (fuga de información del futuro).
    - **Etiquetado (Target Generation)**: Implementación del **Triple Barrier Method**.
        - En lugar de predecir el precio exacto, clasificamos el futuro en 3 estados:
            - **Clase 1 (Compra)**: El precio toca la barrera superior (Take Profit) primero.
            - **Clase 2 (Venta)**: El precio toca la barrera inferior (Stop Loss) primero.
            - **Clase 0 (Hold/Lateral)**: El precio no toca ninguna barrera en el tiempo límite.

### 2. Modelo de Inteligencia Artificial (`src/models`)
El núcleo predictivo es un **Transformer Encoder** (`transformer_model.py`).
- **Por qué Transformer?**: A diferencia de las LSTMs (Redes recurrentes) que procesan paso a paso, los Transformers utilizan mecanismos de **Auto-Atención (Self-Attention)**. Esto permite al modelo ponderar la importancia de diferentes momentos en el pasado simultáneamente, capturando patrones complejos y dependencias a largo plazo de manera más eficiente.
- **Estructura**:
    - **Input Embedding**: Secuencia de vectores de características normalizadas.
    - **Bloques Transformer**: Capas de Multi-Head Attention + Feed Forward Networks con conexiones residuales y normalización.
    - **Global Average Pooling**: Reduce la secuencia temporal a un vector de contexto.
    - **MLP Head**: Capas densas finales que producen la probabilidad de cada una de las 3 clases (Buy/Sell/Hold).

### 3. Entrenamiento (`src/training`)
- **`trainer.py`**: Gestiona el ciclo de vida del entrenamiento.
- **Validación Cruzada**: Se usa `TimeSeriesSplit`. Esto es crucial en series temporales. No podemos mezclar datos aleatoriamente (k-fold estándar) porque romperíamos la causalidad temporal. `TimeSeriesSplit` entrena en el pasado y valida en el futuro inmediato, moviendo la ventana hacia adelante.
- **Función de Pérdida**: `Weighted Categorical Crossentropy`. Dado que el mercado suele estar "lateral" (Hold) la mayor parte del tiempo, las clases están desbalanceadas. Esta función penaliza más los errores en las clases minoritarias (Buy/Sell) para forzar al modelo a aprender patrones de entrada.

### 4. Generación de Señales (`src/signals`)
- **`signal_generator.py`**: Orquestador que une datos, modelo y gestión de riesgo.
- Descarga datos recientes -> Calcula indicadores -> Normaliza usando el escalador guardado -> Infiere con el modelo -> Filtra la señal con `RiskManager`.

### 5. Gestión de Riesgo (`src/risk`)
- **`risk_manager.py`**: Valida si una señal es ejecutable.
    - **Stop Loss Dinámico**: Basado en ATR.
    - **Tamaño de Posición**: Ajustado por la confianza del modelo y la volatilidad del mercado (Criterio de Kelly parcial).
    - **Filtros**: Rechaza operaciones si el ratio Riesgo/Beneficio es bajo o si hay alta volatilidad peligrosa.

### 6. Backtesting (`src/backtesting`)
- **`backtester.py`**: Simulador de mercado.
- **Slippage (Deslizamiento)**: Simula la diferencia entre el precio teórico y el precio real de ejecución (vital para realismo).
- **Comisiones**: Simula costos de transacción.
- **Métricas**: Calcula Ratio de Sharpe, Max Drawdown y Profit Factor para evaluar la estrategia.

## Flujo de Datos
1.  **Raw Data** (OHLCV) -> `FeatureEngineer` -> **Features & Labels**
2.  **Features** (Train Split) -> `RobustScaler.fit` -> **Scaled Features**
3.  **Scaled Features** -> `Transformer Model` -> **Probabilidades**
4.  **Probabilidades** -> `Signal Logic` -> **Señal (Buy/Sell/Hold)**
5.  **Señal** -> `RiskManager` -> **Orden Ejecutable**
