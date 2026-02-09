import pandas as pd
import numpy as np
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from loguru import logger
import tensorflow as tf

from src.data.data_fetcher import MarketDataFetcher
from src.data.features import FeatureEngineer
from src.models.transformer_model import TransformerBlock
from src.training.trainer import ModelTrainer
from src.risk.risk_manager import RiskManager, RiskMetrics
from config.config import config

@dataclass
class TradingSignal:
    """Container for trading signal information"""
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

class SignalGenerator:
    """
    Signal Generator using Transformer-based model and Advanced Risk Management.
    """
    
    def __init__(self, account_balance: float = None):
        self.data_fetcher = MarketDataFetcher()
        self.risk_manager = RiskManager(account_balance)
        self.account_balance = account_balance or config.INITIAL_CAPITAL
        self.models_dir = os.path.join(os.path.dirname(__file__), '..', 'models', 'saved_models')
        os.makedirs(self.models_dir, exist_ok=True)
        
    def _get_model_path(self, symbol: str) -> str:
        return os.path.join(self.models_dir, f"{symbol}_transformer.h5")
        
    def _get_scaler_path(self, symbol: str) -> str:
        return os.path.join(self.models_dir, f"{symbol}_scaler.pkl")

    def train_model(self, symbol: str, training_period: str = "2y") -> bool:
        """
        Train Transformer model for a specific symbol.
        """
        try:
            logger.info(f"Training model for {symbol}...")
            
            # Fetch Data
            data = self.data_fetcher.fetch_data(symbol, period=training_period)
            if data is None or len(data) < 200:
                logger.error(f"Insufficient data for training {symbol}")
                return False
                
            # Feature Engineering
            fe = FeatureEngineer()
            df_features = fe.create_features(data)
            
            # Train
            trainer = ModelTrainer(epochs=config.EPOCHS if hasattr(config, 'EPOCHS') else 50)
            model = trainer.train(df_features, fe)
            
            # Save Model & Scaler
            model.save(self._get_model_path(symbol))
            fe.save_scaler(self._get_scaler_path(symbol))
            
            logger.info(f"Model and scaler saved for {symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            return False

    def generate_signal(self, symbol: str, use_realtime: bool = True) -> Optional[TradingSignal]:
        """
        Generate trading signal using the trained model.
        """
        try:
            model_path = self._get_model_path(symbol)
            scaler_path = self._get_scaler_path(symbol)
            
            # Check if model exists
            if not os.path.exists(model_path) or not os.path.exists(scaler_path):
                logger.warning(f"No trained model found for {symbol}. Training now...")
                if not self.train_model(symbol):
                    return None
            
            # Load Model
            model = tf.keras.models.load_model(model_path, custom_objects={'TransformerBlock': TransformerBlock})
            
            # Load Scaler (Feature Engineer)
            fe = FeatureEngineer()
            fe.load_scaler(scaler_path)
            
            # Fetch recent data
            # Need enough data for lag features + window size
            lookback = 200 # Safe buffer
            data = self.data_fetcher.fetch_data(symbol, period="6mo") # Fetch plenty
            if data is None:
                return None
                
            # Create Features
            df_features = fe.create_features(data)
            
            # Check if we have enough data after feature creation
            window_size = 60 # Match trainer window size
            if len(df_features) < window_size:
                logger.error(f"Not enough data for inference on {symbol}")
                return None
                
            # Prepare Input Sequence
            # Take the last window_size records
            current_sequence_raw = df_features.iloc[-window_size:]
            current_sequence_scaled = fe.transform(current_sequence_raw)
            
            # Predict
            X_pred = np.expand_dims(current_sequence_scaled[fe.feature_columns].values, axis=0) # (1, window, features)
            preds = model.predict(X_pred, verbose=0)[0] # (3,)
            
            # Interpret Prediction
            # 0: HOLD, 1: BUY, 2: SELL
            class_idx = np.argmax(preds)
            confidence = preds[class_idx]
            
            signal_map = {0: 'HOLD', 1: 'BUY', 2: 'SELL'}
            signal_str = signal_map[class_idx]
            
            # Current Price
            current_price = data['close'].iloc[-1]
            
            # Risk Management
            # We construct a dummy 'market_data' for RiskManager from the features + raw data
            # risk_manager expects a dataframe with 'atr', 'volatility', etc.
            # df_features has them.
            
            risk_metrics = self.risk_manager.generate_risk_assessment(
                signal=signal_str,
                current_price=current_price,
                market_data=df_features, # It has ATR, volatility, RSI as columns
                confidence=float(confidence),
                account_balance=self.account_balance
            )
            
            # Validation
            validation = self.risk_manager.validate_signal(
                signal_str,
                risk_metrics,
                df_features
            )
            
            # Override to HOLD if invalid
            final_signal = signal_str
            if not validation['is_valid']:
                final_signal = 'HOLD'
                logger.warning(f"Signal invalidated for {symbol}: {validation['warnings']}")

            # Reasoning
            reasoning = (f"Model predicted {signal_str} with {confidence:.1%} confidence. "
                         f"Risk Score: {validation.get('risk_score', 0):.2f}. "
                         f"{'; '.join(validation['warnings'])}")
                         
            # Technical Analysis Summary
            ta_summary = {
                'rsi': df_features['rsi'].iloc[-1] if 'rsi' in df_features else 0,
                'atr': df_features['atr'].iloc[-1] if 'atr' in df_features else 0,
                'volatility': df_features.get('volatility', pd.Series([0])).iloc[-1]
            }

            return TradingSignal(
                symbol=symbol,
                signal=final_signal,
                confidence=float(confidence),
                entry_price=current_price,
                current_price=current_price,
                stop_loss=risk_metrics.stop_loss,
                take_profit=risk_metrics.take_profit,
                position_size=risk_metrics.confidence_adjusted_size,
                risk_reward_ratio=risk_metrics.risk_reward_ratio,
                max_loss=risk_metrics.max_loss,
                timestamp=datetime.now(),
                reasoning=reasoning,
                technical_analysis=ta_summary,
                risk_assessment=asdict(risk_metrics),
                market_conditions={'valid': validation['is_valid']}
            )

        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def generate_multiple_signals(self, symbols: List[str] = None) -> List[TradingSignal]:
        if symbols is None:
            symbols = config.DEFAULT_SYMBOLS
        
        signals = []
        for symbol in symbols:
            signal = self.generate_signal(symbol)
            if signal:
                signals.append(signal)
                
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals