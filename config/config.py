from typing import Dict, List
from pydantic import BaseSettings

class TradingConfig(BaseSettings):
    """Configuración del sistema de Trading AI"""
    
    # Fuentes de datos
    DEFAULT_SYMBOLS: List[str] = ["AAPL", "GOOGL", "MSFT", "TSLA", "BTC-USD", "ETH-USD"]
    DATA_INTERVAL: str = "1h"  # 1m, 5m, 15m, 30m, 1h, 1d
    LOOKBACK_PERIOD: int = 100  # Número de períodos hacia atrás
    
    # Parámetros del modelo
    LEARNING_RATE: float = 0.001
    BATCH_SIZE: int = 32
    EPOCHS: int = 100
    
    # Gestión de riesgo
    MAX_RISK_PER_TRADE: float = 0.02  # 2% máximo riesgo por operación
    STOP_LOSS_MULTIPLIER: float = 2.0
    TAKE_PROFIT_MULTIPLIER: float = 3.0
    
    # Generación de señales
    CONFIDENCE_THRESHOLD: float = 0.7  # Confianza mínima para generar señales
    
    # Backtesting
    INITIAL_CAPITAL: float = 100000.0
    COMMISSION: float = 0.001  # 0.1% comisión
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/trading_ai.log"
    
    class Config:
        env_file = ".env"

config = TradingConfig()