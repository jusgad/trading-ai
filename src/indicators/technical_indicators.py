import pandas as pd
import numpy as np
from typing import Dict, Optional
import talib
from loguru import logger

class TechnicalIndicators:
    """Calculate various technical indicators for market analysis"""
    
    @staticmethod
    def sma(data: pd.Series, window: int = 20) -> pd.Series:
        """Simple Moving Average"""
        return data.rolling(window=window).mean()
    
    @staticmethod
    def ema(data: pd.Series, window: int = 20) -> pd.Series:
        """Exponential Moving Average"""
        return data.ewm(span=window).mean()
    
    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        try:
            return pd.Series(talib.RSI(data.values, timeperiod=window), index=data.index)
        except:
            # Fallback implementation
            delta = data.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
    
    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Moving Average Convergence Divergence"""
        try:
            macd_line, macd_signal, macd_histogram = talib.MACD(data.values, 
                                                               fastperiod=fast, 
                                                               slowperiod=slow, 
                                                               signalperiod=signal)
            return {
                'macd': pd.Series(macd_line, index=data.index),
                'signal': pd.Series(macd_signal, index=data.index),
                'histogram': pd.Series(macd_histogram, index=data.index)
            }
        except:
            # Fallback implementation
            ema_fast = data.ewm(span=fast).mean()
            ema_slow = data.ewm(span=slow).mean()
            macd_line = ema_fast - ema_slow
            macd_signal = macd_line.ewm(span=signal).mean()
            macd_histogram = macd_line - macd_signal
            
            return {
                'macd': macd_line,
                'signal': macd_signal,
                'histogram': macd_histogram
            }
    
    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, std_dev: float = 2) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()
        
        return {
            'middle': sma,
            'upper': sma + (std * std_dev),
            'lower': sma - (std * std_dev)
        }
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                   k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        try:
            slowk, slowd = talib.STOCH(high.values, low.values, close.values,
                                      fastk_period=k_window, slowk_period=d_window,
                                      slowd_period=d_window)
            return {
                'k': pd.Series(slowk, index=close.index),
                'd': pd.Series(slowd, index=close.index)
            }
        except:
            # Fallback implementation
            lowest_low = low.rolling(window=k_window).min()
            highest_high = high.rolling(window=k_window).max()
            k_percent = ((close - lowest_low) / (highest_high - lowest_low)) * 100
            d_percent = k_percent.rolling(window=d_window).mean()
            
            return {
                'k': k_percent,
                'd': d_percent
            }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        try:
            return pd.Series(talib.ATR(high.values, low.values, close.values, timeperiod=window), 
                           index=close.index)
        except:
            # Fallback implementation
            high_low = high - low
            high_close = np.abs(high - close.shift())
            low_close = np.abs(low - close.shift())
            
            tr = np.maximum(high_low, np.maximum(high_close, low_close))
            return pd.Series(tr).rolling(window=window).mean()
    
    @staticmethod
    def williams_r(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Williams %R"""
        try:
            return pd.Series(talib.WILLR(high.values, low.values, close.values, timeperiod=window),
                           index=close.index)
        except:
            # Fallback implementation
            highest_high = high.rolling(window=window).max()
            lowest_low = low.rolling(window=window).min()
            return -100 * (highest_high - close) / (highest_high - lowest_low)
    
    @staticmethod
    def cci(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20) -> pd.Series:
        """Commodity Channel Index"""
        try:
            return pd.Series(talib.CCI(high.values, low.values, close.values, timeperiod=window),
                           index=close.index)
        except:
            # Fallback implementation
            tp = (high + low + close) / 3
            sma_tp = tp.rolling(window=window).mean()
            mad = tp.rolling(window=window).apply(lambda x: np.mean(np.abs(x - np.mean(x))))
            return (tp - sma_tp) / (0.015 * mad)
    
    @staticmethod
    def calculate_all_indicators(data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a given dataset
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical indicators added
        """
        try:
            df = data.copy()
            
            # Moving averages
            df['sma_20'] = TechnicalIndicators.sma(df['close'], 20)
            df['sma_50'] = TechnicalIndicators.sma(df['close'], 50)
            df['ema_12'] = TechnicalIndicators.ema(df['close'], 12)
            df['ema_26'] = TechnicalIndicators.ema(df['close'], 26)
            
            # RSI
            df['rsi'] = TechnicalIndicators.rsi(df['close'])
            
            # MACD
            macd_data = TechnicalIndicators.macd(df['close'])
            df['macd'] = macd_data['macd']
            df['macd_signal'] = macd_data['signal']
            df['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = TechnicalIndicators.bollinger_bands(df['close'])
            df['bb_upper'] = bb_data['upper']
            df['bb_middle'] = bb_data['middle']
            df['bb_lower'] = bb_data['lower']
            df['bb_width'] = (bb_data['upper'] - bb_data['lower']) / bb_data['middle']
            
            # Stochastic
            stoch_data = TechnicalIndicators.stochastic(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_data['k']
            df['stoch_d'] = stoch_data['d']
            
            # ATR
            df['atr'] = TechnicalIndicators.atr(df['high'], df['low'], df['close'])
            
            # Williams %R
            df['williams_r'] = TechnicalIndicators.williams_r(df['high'], df['low'], df['close'])
            
            # CCI
            df['cci'] = TechnicalIndicators.cci(df['high'], df['low'], df['close'])
            
            # Volume indicators
            df['volume_sma'] = TechnicalIndicators.sma(df['volume'], 20)
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            
            # Price momentum
            df['momentum'] = df['close'] / df['close'].shift(10) - 1
            df['rate_of_change'] = df['close'].pct_change(10)
            
            # Volatility
            df['volatility'] = df['close'].rolling(20).std() / df['close'].rolling(20).mean()
            
            logger.info(f"Calculated technical indicators for {len(df)} data points")
            return df.dropna()
            
        except Exception as e:
            logger.error(f"Error calculating technical indicators: {str(e)}")
            return data