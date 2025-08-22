import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
from loguru import logger

from src.data.data_fetcher import MarketDataFetcher
from src.indicators.technical_indicators import TechnicalIndicators
from src.models.rl_agent import RLTrainer
from src.risk.risk_manager import RiskManager, RiskMetrics
from config.config import config

@dataclass
class TradingSignal:
    """Contenedor para información de señales de trading"""
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
    """Motor principal de generación de señales combinando modelo RL y gestión de riesgo"""
    
    def __init__(self, account_balance: float = None):
        self.data_fetcher = MarketDataFetcher()
        self.risk_manager = RiskManager(account_balance)
        self.trained_models = {}  # Store trained models for different symbols
        self.account_balance = account_balance or config.INITIAL_CAPITAL
        
    def train_model(self, symbol: str, training_period: str = "2y") -> bool:
        """
        Train RL model for a specific symbol
        
        Args:
            symbol: Trading symbol
            training_period: Period for training data
            
        Returns:
            True if training successful
        """
        try:
            logger.info(f"Training model for {symbol}")
            
            # Fetch training data
            data = self.data_fetcher.fetch_data(symbol, period=training_period)
            if data is None or len(data) < 100:
                logger.error(f"Insufficient data for training {symbol}")
                return False
            
            # Calculate technical indicators
            data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)
            
            # Train RL model
            trainer = RLTrainer(data_with_indicators)
            training_results = trainer.train(episodes=config.EPOCHS)
            
            # Store trained model
            self.trained_models[symbol] = trainer
            
            logger.info(f"Model training completed for {symbol}")
            logger.info(f"Final portfolio value: ${training_results['portfolio_values'][-1]:.2f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training model for {symbol}: {str(e)}")
            return False
    
    def generate_signal(self, symbol: str, use_realtime: bool = True) -> Optional[TradingSignal]:
        """
        Generate trading signal for a symbol
        
        Args:
            symbol: Trading symbol
            use_realtime: Whether to use real-time data
            
        Returns:
            TradingSignal object or None
        """
        try:
            # Check if model is trained for this symbol
            if symbol not in self.trained_models:
                logger.warning(f"No trained model for {symbol}. Training now...")
                if not self.train_model(symbol):
                    return None
            
            # Fetch recent data
            data = self.data_fetcher.fetch_data(symbol, period="3mo")
            if data is None or len(data) < config.LOOKBACK_PERIOD:
                logger.error(f"Insufficient data for {symbol}")
                return None
            
            # Calculate technical indicators
            data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)
            
            # Get RL model signal
            rl_signal = self.trained_models[symbol].get_signal(data_with_indicators)
            
            # Get current price
            current_price = data_with_indicators['close'].iloc[-1]
            
            # Assess market conditions
            market_conditions = self._assess_market_conditions(data_with_indicators)
            
            # Generate risk assessment
            risk_metrics = self.risk_manager.generate_risk_assessment(
                signal=rl_signal['signal'],
                current_price=current_price,
                market_data=data_with_indicators,
                confidence=rl_signal['confidence'],
                account_balance=self.account_balance
            )
            
            # Validate signal
            validation = self.risk_manager.validate_signal(
                rl_signal['signal'],
                risk_metrics,
                data_with_indicators
            )
            
            # Adjust signal based on validation
            final_signal = rl_signal['signal']
            if not validation['is_valid']:
                final_signal = 'HOLD'
                logger.warning(f"Signal invalidated for {symbol}: {validation['warnings']}")
            
            # Generate reasoning
            reasoning = self._generate_reasoning(
                rl_signal, risk_metrics, validation, market_conditions
            )
            
            # Create trading signal
            trading_signal = TradingSignal(
                symbol=symbol,
                signal=final_signal,
                confidence=rl_signal['confidence'],
                current_price=current_price,
                entry_price=current_price,
                stop_loss=risk_metrics.stop_loss,
                take_profit=risk_metrics.take_profit,
                position_size=risk_metrics.confidence_adjusted_size,
                risk_reward_ratio=risk_metrics.risk_reward_ratio,
                max_loss=risk_metrics.max_loss,
                timestamp=datetime.now(),
                reasoning=reasoning,
                technical_analysis=self._get_technical_analysis(data_with_indicators),
                risk_assessment=asdict(risk_metrics),
                market_conditions=market_conditions
            )
            
            logger.info(f"Generated signal for {symbol}: {final_signal} "
                       f"(Confidence: {rl_signal['confidence']:.2f})")
            
            return trading_signal
            
        except Exception as e:
            logger.error(f"Error generating signal for {symbol}: {str(e)}")
            return None
    
    def generate_multiple_signals(self, symbols: List[str] = None) -> List[TradingSignal]:
        """
        Generate signals for multiple symbols
        
        Args:
            symbols: List of symbols to analyze
            
        Returns:
            List of TradingSignal objects
        """
        if symbols is None:
            symbols = config.DEFAULT_SYMBOLS
        
        signals = []
        for symbol in symbols:
            signal = self.generate_signal(symbol)
            if signal:
                signals.append(signal)
        
        # Sort by confidence descending
        signals.sort(key=lambda x: x.confidence, reverse=True)
        
        return signals
    
    def _assess_market_conditions(self, data: pd.DataFrame) -> Dict:
        """Assess current market conditions"""
        latest = data.iloc[-1]
        
        conditions = {
            'trend': self._determine_trend(data),
            'volatility': 'HIGH' if latest.get('volatility', 0) > 0.03 else 'NORMAL',
            'volume': 'HIGH' if latest.get('volume_ratio', 1) > 1.5 else 'NORMAL',
            'momentum': 'BULLISH' if latest.get('momentum', 0) > 0.05 else 'BEARISH' if latest.get('momentum', 0) < -0.05 else 'NEUTRAL',
            'rsi_level': 'OVERBOUGHT' if latest.get('rsi', 50) > 70 else 'OVERSOLD' if latest.get('rsi', 50) < 30 else 'NEUTRAL',
            'bb_position': self._get_bollinger_position(latest)
        }
        
        return conditions
    
    def _determine_trend(self, data: pd.DataFrame) -> str:
        """Determine market trend"""
        latest = data.iloc[-1]
        sma_20 = latest.get('sma_20', latest['close'])
        sma_50 = latest.get('sma_50', latest['close'])
        
        if latest['close'] > sma_20 > sma_50:
            return 'UPTREND'
        elif latest['close'] < sma_20 < sma_50:
            return 'DOWNTREND'
        else:
            return 'SIDEWAYS'
    
    def _get_bollinger_position(self, latest_data) -> str:
        """Determine position relative to Bollinger Bands"""
        close = latest_data['close']
        bb_upper = latest_data.get('bb_upper', close)
        bb_lower = latest_data.get('bb_lower', close)
        bb_middle = latest_data.get('bb_middle', close)
        
        if close > bb_upper:
            return 'ABOVE_UPPER'
        elif close < bb_lower:
            return 'BELOW_LOWER'
        elif close > bb_middle:
            return 'UPPER_HALF'
        else:
            return 'LOWER_HALF'
    
    def _generate_reasoning(self, rl_signal: Dict, risk_metrics: RiskMetrics, 
                          validation: Dict, market_conditions: Dict) -> str:
        """Generate human-readable reasoning for the signal"""
        reasoning_parts = []
        
        # RL model confidence
        reasoning_parts.append(f"AI model recommends {rl_signal['signal']} with {rl_signal['confidence']:.1%} confidence")
        
        # Market conditions
        trend = market_conditions.get('trend', 'UNKNOWN')
        reasoning_parts.append(f"Market trend: {trend}")
        
        # Risk assessment
        if risk_metrics.risk_reward_ratio > 2:
            reasoning_parts.append(f"Favorable risk/reward ratio of {risk_metrics.risk_reward_ratio:.1f}")
        elif risk_metrics.risk_reward_ratio < 1.5:
            reasoning_parts.append(f"Unfavorable risk/reward ratio of {risk_metrics.risk_reward_ratio:.1f}")
        
        # Validation warnings
        if validation['warnings']:
            reasoning_parts.append(f"Risk warnings: {', '.join(validation['warnings'])}")
        
        # Technical indicators
        rsi_level = market_conditions.get('rsi_level', 'NEUTRAL')
        if rsi_level != 'NEUTRAL':
            reasoning_parts.append(f"RSI indicates {rsi_level} conditions")
        
        return '. '.join(reasoning_parts)
    
    def _get_technical_analysis(self, data: pd.DataFrame) -> Dict:
        """Extract key technical analysis points"""
        latest = data.iloc[-1]
        
        return {
            'rsi': latest.get('rsi', 50),
            'macd_signal': 'BULLISH' if latest.get('macd', 0) > latest.get('macd_signal', 0) else 'BEARISH',
            'moving_averages': {
                'price_vs_sma20': 'ABOVE' if latest['close'] > latest.get('sma_20', latest['close']) else 'BELOW',
                'sma20_vs_sma50': 'ABOVE' if latest.get('sma_20', 0) > latest.get('sma_50', 0) else 'BELOW'
            },
            'volatility': latest.get('volatility', 0),
            'volume_analysis': 'HIGH' if latest.get('volume_ratio', 1) > 1.2 else 'NORMAL'
        }
    
    def get_portfolio_recommendations(self, current_positions: List[Dict] = None) -> Dict:
        """
        Get portfolio-level recommendations
        
        Args:
            current_positions: List of current portfolio positions
            
        Returns:
            Portfolio recommendations
        """
        if current_positions is None:
            current_positions = []
        
        # Generate signals for all default symbols
        signals = self.generate_multiple_signals()
        
        # Portfolio risk assessment
        portfolio_risk = self.risk_manager.calculate_portfolio_risk(current_positions)
        
        # Filter signals based on portfolio risk
        recommended_signals = []
        for signal in signals:
            if signal.signal != 'HOLD' and portfolio_risk['concentration_risk'] < 0.3:
                recommended_signals.append(signal)
        
        return {
            'signals': recommended_signals,
            'portfolio_risk': portfolio_risk,
            'recommendations': self._generate_portfolio_recommendations(
                recommended_signals, portfolio_risk, current_positions
            )
        }
    
    def _generate_portfolio_recommendations(self, signals: List[TradingSignal], 
                                          portfolio_risk: Dict, 
                                          current_positions: List[Dict]) -> List[str]:
        """Generate portfolio-level recommendations"""
        recommendations = []
        
        if portfolio_risk['concentration_risk'] > 0.5:
            recommendations.append("Portfolio is too concentrated - consider diversifying")
        
        if len(signals) == 0:
            recommendations.append("No high-quality signals found - consider staying in cash")
        
        buy_signals = [s for s in signals if s.signal == 'BUY']
        sell_signals = [s for s in signals if s.signal == 'SELL']
        
        if len(buy_signals) > len(sell_signals):
            recommendations.append("Market sentiment appears bullish")
        elif len(sell_signals) > len(buy_signals):
            recommendations.append("Market sentiment appears bearish")
        
        return recommendations