import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import RobustScaler
import pickle
import joblib 
from loguru import logger

class FeatureEngineer:
    """
    Advanced Feature Engineering Pipeline using pandas-ta.
    Handles technical indicators, Triple Barrier Method labeling, and robust normalization.
    """
    
    def __init__(self, use_scale: bool = True):
        self.scaler = RobustScaler()
        self.use_scale = use_scale
        self.feature_columns = []
        self.is_fitted = False

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates technical indicators using pandas-ta.
        """
        # Ensure we don't modify original
        data = df.copy()
        
        # Ensure lowercase columns if not already
        data.columns = [c.lower() for c in data.columns]
        
        # 1. Momentum Indicators
        # RSI
        # pandas_ta appends columns to the dataframe
        data.ta.rsi(length=14, append=True)
        
        # MACD
        data.ta.macd(fast=12, slow=26, signal=9, append=True)
        # Rename standard output columns to cleaner names
        # MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
        # We'll map them dynamically or just use the generated names
        
        # 2. Volatility Indicators
        # Bollinger Bands
        data.ta.bbands(length=20, std=2, append=True)
        # BBL_20_2.0, BBM_20_2.0, BBU_20_2.0
        
        # ATR
        data.ta.atr(length=14, append=True)
        # ATR_14

        # 3. Volume Indicators
        # OBV
        data.ta.obv(append=True)
        # OBV
        
        # 4. Returns & Log Returns
        # We need to fill NaNs for log returns or drop them later
        data['log_return'] = np.log(data['close'] / data['close'].shift(1))
        
        # Clean up NaNs created by lags/indicators
        data.dropna(inplace=True)
        
        # Define feature columns based on what was added
        # We'll dynamically identify them or hardcode expected names
        # For robustness, let's identify them
        all_cols = data.columns.tolist()
        exclude = ['open', 'high', 'low', 'close', 'volume', 'dividends', 'stock splits', 'date', 'symbol']
        self.feature_columns = [c for c in all_cols if c not in exclude]

        return data

    def fit(self, data: pd.DataFrame):
        """Fit the scaler on training data."""
        if self.use_scale:
            # Check if columns exist
            missing = [c for c in self.feature_columns if c not in data.columns]
            if missing:
                raise ValueError(f"Missing columns for fitting: {missing}")
                
            self.scaler.fit(data[self.feature_columns])
            self.is_fitted = True
            
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data using fitted scaler."""
        if not self.is_fitted and self.use_scale:
            raise ValueError("Scaler not fitted. Call fit() first.")
        
        df_scaled = data.copy()
        if self.use_scale:
             df_scaled[self.feature_columns] = self.scaler.transform(data[self.feature_columns])
        
        return df_scaled

    def triple_barrier_labels(self, data: pd.DataFrame, t: int = 20, ub: float = 0.02, lb: float = 0.02) -> pd.Series:
        """
        Triple Barrier Method for labeling.
        0: Hold (Hit vertical barrier / time limit)
        1: Buy (Hit upper barrier)
        2: Sell (Hit lower barrier)
        """
        labels = []
        prices = data['close'].values
        
        # We need to look forward 't' periods
        n = len(prices)
        
        for i in range(n):
            # End of window
            end = min(i + t, n)
            window = prices[i+1 : end+1] # Look ahead
            
            if len(window) == 0:
                labels.append(0)
                continue
                
            current_price = prices[i]
            
            # Barriers
            upper_barrier = current_price * (1 + ub)
            lower_barrier = current_price * (1 - lb)
            
            # Check hits
            # First index where price >= upper
            hit_up = np.where(window >= upper_barrier)[0]
            # First index where price <= lower
            hit_down = np.where(window <= lower_barrier)[0]
            
            first_up = hit_up[0] if len(hit_up) > 0 else t + 1
            first_down = hit_down[0] if len(hit_down) > 0 else t + 1
            
            if first_up < first_down and first_up < t:
                labels.append(1) # Buy
            elif first_down < first_up and first_down < t:
                labels.append(2) # Sell
            else:
                labels.append(0) # Hold (Vertical barrier)
                
        return pd.Series(labels, index=data.index)

    def save_scaler(self, path: str):
        import joblib
        joblib.dump(self.scaler, path)
            
    def load_scaler(self, path: str):
        import joblib
        self.scaler = joblib.load(path)
        self.is_fitted = True
