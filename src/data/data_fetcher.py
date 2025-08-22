import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List
from loguru import logger
from config.config import config

class MarketDataFetcher:
    """Fetches and processes market data from various sources"""
    
    def __init__(self):
        self.symbols = config.DEFAULT_SYMBOLS
        
    def fetch_data(self, 
                   symbol: str, 
                   period: str = "1y", 
                   interval: str = None) -> Optional[pd.DataFrame]:
        """
        Fetch market data for a given symbol
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTC-USD')
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            interval: Data interval (1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            if interval is None:
                interval = config.DATA_INTERVAL
                
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period, interval=interval)
            
            if data.empty:
                logger.warning(f"No data found for symbol: {symbol}")
                return None
                
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Add symbol column
            data['symbol'] = symbol
            
            logger.info(f"Fetched {len(data)} records for {symbol}")
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return None
    
    def fetch_multiple_symbols(self, 
                             symbols: List[str] = None, 
                             period: str = "1y") -> dict:
        """
        Fetch data for multiple symbols
        
        Args:
            symbols: List of symbols to fetch
            period: Time period for data
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        if symbols is None:
            symbols = self.symbols
            
        data_dict = {}
        for symbol in symbols:
            data = self.fetch_data(symbol, period)
            if data is not None:
                data_dict[symbol] = data
                
        return data_dict
    
    def get_realtime_data(self, symbol: str) -> Optional[dict]:
        """
        Get real-time data for a symbol
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with current price info
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'current_price': info.get('currentPrice', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('open', 0),
                'day_high': info.get('dayHigh', 0),
                'day_low': info.get('dayLow', 0),
                'volume': info.get('volume', 0),
                'market_cap': info.get('marketCap', 0),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error fetching real-time data for {symbol}: {str(e)}")
            return None
    
    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess market data for analysis
        
        Args:
            data: Raw market data
            
        Returns:
            Preprocessed DataFrame
        """
        # Remove any NaN values
        data = data.dropna()
        
        # Calculate returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Calculate volatility (rolling standard deviation)
        data['volatility'] = data['returns'].rolling(window=20).std()
        
        # Calculate price ranges
        data['high_low_pct'] = (data['high'] - data['low']) / data['close'] * 100
        data['open_close_pct'] = (data['close'] - data['open']) / data['open'] * 100
        
        return data.dropna()

import numpy as np