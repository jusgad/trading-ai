import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add src to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.signals.signal_generator import SignalGenerator
from src.indicators.technical_indicators import TechnicalIndicators
from src.risk.risk_manager import RiskManager
from src.data.data_fetcher import MarketDataFetcher

class TestSignalGeneration(unittest.TestCase):
    """Test signal generation functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.signal_generator = SignalGenerator(account_balance=100000)
        self.risk_manager = RiskManager()
        
        # Create sample data
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.001, 0.02, len(dates))
        prices = [100]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        self.sample_data = pd.DataFrame({
            'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
            'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, len(dates))
        }, index=dates)
    
    def test_technical_indicators_calculation(self):
        """Test technical indicators are calculated correctly"""
        indicators = TechnicalIndicators.calculate_all_indicators(self.sample_data)
        
        # Check that indicators are added
        expected_indicators = ['sma_20', 'sma_50', 'rsi', 'macd', 'bb_upper', 'bb_lower']
        for indicator in expected_indicators:
            self.assertIn(indicator, indicators.columns)
        
        # Check RSI is within valid range
        rsi_values = indicators['rsi'].dropna()
        self.assertTrue(all(0 <= val <= 100 for val in rsi_values))
        
        # Check moving averages are calculated
        self.assertFalse(indicators['sma_20'].isna().all())
        self.assertFalse(indicators['sma_50'].isna().all())
    
    def test_risk_management_calculations(self):
        """Test risk management calculations"""
        current_price = 100.0
        stop_loss_price = 95.0
        account_balance = 100000.0
        confidence = 0.8
        
        position_size = self.risk_manager.calculate_position_size(
            current_price, stop_loss_price, account_balance, confidence
        )
        
        # Position size should be positive
        self.assertGreater(position_size, 0)
        
        # Should not exceed account balance
        total_cost = position_size * current_price
        self.assertLessEqual(total_cost, account_balance)
        
        # Test stop loss calculation
        atr = 2.0
        stop_loss = self.risk_manager.calculate_stop_loss(
            current_price, 'BUY', atr
        )
        
        self.assertLess(stop_loss, current_price)  # Stop loss should be below entry for BUY
        self.assertGreater(stop_loss, 0)  # Should be positive
    
    def test_signal_validation(self):
        """Test signal validation logic"""
        # Add indicators to sample data
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(self.sample_data)
        
        # Test with valid data
        if len(data_with_indicators) > 100:  # Ensure we have enough data
            latest_price = data_with_indicators['close'].iloc[-1]
            atr = data_with_indicators.get('atr', pd.Series([2.0])).iloc[-1] or 2.0
            
            stop_loss = self.risk_manager.calculate_stop_loss(latest_price, 'BUY', atr)
            risk_metrics = self.risk_manager.generate_risk_assessment(
                'BUY', latest_price, data_with_indicators, 0.7, 100000
            )
            
            # Risk metrics should be properly calculated
            self.assertIsNotNone(risk_metrics.position_size)
            self.assertIsNotNone(risk_metrics.stop_loss)
            self.assertIsNotNone(risk_metrics.take_profit)
            self.assertGreater(risk_metrics.risk_reward_ratio, 0)
    
    def test_market_data_fetcher(self):
        """Test market data fetching (mock test)"""
        fetcher = MarketDataFetcher()
        
        # Test symbol validation
        self.assertIn('AAPL', fetcher.symbols)
        
        # Test data preprocessing
        preprocessed = fetcher.preprocess_data(self.sample_data)
        
        # Should have additional columns
        self.assertIn('returns', preprocessed.columns)
        self.assertIn('volatility', preprocessed.columns)
    
    def test_signal_generation_pipeline(self):
        """Test the complete signal generation pipeline"""
        # This is a simplified test since we can't easily test the full RL model
        data_with_indicators = TechnicalIndicators.calculate_all_indicators(self.sample_data)
        
        if len(data_with_indicators) > 100:
            # Test market condition assessment
            conditions = self.signal_generator._assess_market_conditions(data_with_indicators)
            
            # Should return valid conditions
            self.assertIn('trend', conditions)
            self.assertIn('volatility', conditions)
            self.assertIn('momentum', conditions)
            
            # Test technical analysis extraction
            tech_analysis = self.signal_generator._get_technical_analysis(data_with_indicators)
            
            self.assertIn('rsi', tech_analysis)
            self.assertIn('moving_averages', tech_analysis)

class TestDataProcessing(unittest.TestCase):
    """Test data processing functionality"""
    
    def test_data_cleaning(self):
        """Test data cleaning and preprocessing"""
        # Create sample data with some issues
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'open': np.random.randn(100) * 10 + 100,
            'high': np.random.randn(100) * 10 + 105,
            'low': np.random.randn(100) * 10 + 95,
            'close': np.random.randn(100) * 10 + 100,
            'volume': np.random.randint(1000, 10000, 100)
        }, index=dates)
        
        # Add some NaN values
        data.iloc[10:15] = np.nan
        
        fetcher = MarketDataFetcher()
        cleaned_data = fetcher.preprocess_data(data)
        
        # Should have no NaN values in the result
        self.assertFalse(cleaned_data.isnull().any().any())
        
        # Should have calculated returns
        self.assertIn('returns', cleaned_data.columns)

class TestRiskManagement(unittest.TestCase):
    """Test risk management functionality"""
    
    def setUp(self):
        self.risk_manager = RiskManager(initial_capital=100000)
    
    def test_position_sizing_limits(self):
        """Test position sizing respects limits"""
        current_price = 100.0
        stop_loss_price = 90.0  # 10% stop loss
        account_balance = 10000.0
        confidence = 1.0
        
        position_size = self.risk_manager.calculate_position_size(
            current_price, stop_loss_price, account_balance, confidence
        )
        
        # Total risk should not exceed max risk per trade
        risk_per_share = current_price - stop_loss_price
        total_risk = position_size * risk_per_share
        max_risk = account_balance * self.risk_manager.max_risk_per_trade
        
        self.assertLessEqual(total_risk, max_risk * 1.01)  # Small tolerance for rounding
    
    def test_risk_reward_calculation(self):
        """Test risk-reward ratio calculation"""
        entry_price = 100.0
        stop_loss = 95.0
        take_profit = self.risk_manager.calculate_take_profit(
            entry_price, stop_loss, 'BUY'
        )
        
        risk = entry_price - stop_loss
        reward = take_profit - entry_price
        risk_reward_ratio = reward / risk
        
        self.assertGreaterEqual(risk_reward_ratio, 1.0)  # Should be at least 1:1

if __name__ == '__main__':
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_suite.addTest(unittest.makeSuite(TestSignalGeneration))
    test_suite.addTest(unittest.makeSuite(TestDataProcessing))
    test_suite.addTest(unittest.makeSuite(TestRiskManagement))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Exit with error code if tests failed
    sys.exit(0 if result.wasSuccessful() else 1)