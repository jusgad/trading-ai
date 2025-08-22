import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger
from config.config import config

@dataclass
class RiskMetrics:
    """Container for risk assessment metrics"""
    position_size: float
    stop_loss: float
    take_profit: float
    risk_reward_ratio: float
    max_loss: float
    volatility_risk: float
    confidence_adjusted_size: float

class RiskManager:
    """Comprehensive risk management system for trading signals"""
    
    def __init__(self, initial_capital: float = None):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.max_risk_per_trade = config.MAX_RISK_PER_TRADE
        self.stop_loss_multiplier = config.STOP_LOSS_MULTIPLIER
        self.take_profit_multiplier = config.TAKE_PROFIT_MULTIPLIER
        
    def calculate_position_size(self, 
                              current_price: float,
                              stop_loss_price: float,
                              account_balance: float,
                              confidence: float = 1.0) -> float:
        """
        Calculate optimal position size based on risk management rules
        
        Args:
            current_price: Current asset price
            stop_loss_price: Stop loss price level
            account_balance: Available account balance
            confidence: Signal confidence (0-1)
            
        Returns:
            Recommended position size
        """
        # Calculate risk per share
        risk_per_share = abs(current_price - stop_loss_price)
        
        if risk_per_share == 0:
            return 0
        
        # Maximum risk amount
        max_risk_amount = account_balance * self.max_risk_per_trade
        
        # Base position size
        base_position_size = max_risk_amount / risk_per_share
        
        # Adjust for confidence
        confidence_adjusted_size = base_position_size * confidence
        
        # Ensure we don't exceed available balance
        max_affordable_shares = account_balance / current_price
        
        return min(confidence_adjusted_size, max_affordable_shares)
    
    def calculate_stop_loss(self, 
                          entry_price: float,
                          signal_direction: str,
                          atr: float,
                          volatility: float = None) -> float:
        """
        Calculate dynamic stop loss based on volatility and market conditions
        
        Args:
            entry_price: Entry price for the position
            signal_direction: 'BUY' or 'SELL'
            atr: Average True Range
            volatility: Additional volatility measure
            
        Returns:
            Stop loss price level
        """
        # Base stop loss using ATR
        atr_multiplier = self.stop_loss_multiplier
        
        # Adjust multiplier based on volatility
        if volatility and volatility > 0.02:  # High volatility
            atr_multiplier *= 1.5
        elif volatility and volatility < 0.01:  # Low volatility
            atr_multiplier *= 0.8
        
        if signal_direction.upper() == 'BUY':
            stop_loss = entry_price - (atr * atr_multiplier)
        else:  # SELL
            stop_loss = entry_price + (atr * atr_multiplier)
        
        return max(stop_loss, 0)  # Ensure positive price
    
    def calculate_take_profit(self, 
                            entry_price: float,
                            stop_loss_price: float,
                            signal_direction: str,
                            risk_reward_ratio: float = None) -> float:
        """
        Calculate take profit level based on risk-reward ratio
        
        Args:
            entry_price: Entry price for the position
            stop_loss_price: Stop loss price level
            signal_direction: 'BUY' or 'SELL'
            risk_reward_ratio: Desired risk-reward ratio
            
        Returns:
            Take profit price level
        """
        if risk_reward_ratio is None:
            risk_reward_ratio = self.take_profit_multiplier
        
        risk_amount = abs(entry_price - stop_loss_price)
        reward_amount = risk_amount * risk_reward_ratio
        
        if signal_direction.upper() == 'BUY':
            take_profit = entry_price + reward_amount
        else:  # SELL
            take_profit = entry_price - reward_amount
        
        return max(take_profit, 0)  # Ensure positive price
    
    def assess_market_risk(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        Assess overall market risk conditions
        
        Args:
            data: Market data with technical indicators
            
        Returns:
            Dictionary with risk assessment metrics
        """
        if len(data) < 20:
            return {'overall_risk': 0.5, 'volatility_risk': 0.5, 'trend_risk': 0.5}
        
        recent_data = data.tail(20)
        
        # Volatility risk
        price_volatility = recent_data['close'].std() / recent_data['close'].mean()
        volume_volatility = recent_data['volume'].std() / recent_data['volume'].mean()
        
        # Trend consistency risk
        sma_20 = recent_data['close'].rolling(20).mean().iloc[-1]
        current_price = recent_data['close'].iloc[-1]
        trend_strength = abs(current_price - sma_20) / sma_20
        
        # RSI extreme levels
        rsi = recent_data.get('rsi', pd.Series([50])).iloc[-1]
        rsi_risk = 1.0 if rsi > 80 or rsi < 20 else 0.0
        
        # Overall risk score (0-1, where 1 is highest risk)
        volatility_risk = min(price_volatility * 10, 1.0)
        trend_risk = min(trend_strength * 5, 1.0)
        overall_risk = (volatility_risk + trend_risk + rsi_risk) / 3
        
        return {
            'overall_risk': overall_risk,
            'volatility_risk': volatility_risk,
            'trend_risk': trend_risk,
            'rsi_risk': rsi_risk,
            'price_volatility': price_volatility,
            'volume_volatility': volume_volatility
        }
    
    def calculate_portfolio_risk(self, positions: list, correlations: pd.DataFrame = None) -> Dict[str, float]:
        """
        Calculate portfolio-level risk metrics
        
        Args:
            positions: List of current positions
            correlations: Correlation matrix between assets
            
        Returns:
            Portfolio risk metrics
        """
        if not positions:
            return {'portfolio_risk': 0.0, 'concentration_risk': 0.0, 'correlation_risk': 0.0}
        
        # Calculate concentration risk
        total_exposure = sum(pos.get('value', 0) for pos in positions)
        max_position = max(pos.get('value', 0) for pos in positions)
        concentration_risk = max_position / total_exposure if total_exposure > 0 else 0
        
        # Correlation risk (simplified)
        correlation_risk = 0.5 if len(positions) > 1 else 0.0
        
        # Overall portfolio risk
        portfolio_risk = (concentration_risk + correlation_risk) / 2
        
        return {
            'portfolio_risk': portfolio_risk,
            'concentration_risk': concentration_risk,
            'correlation_risk': correlation_risk,
            'total_exposure': total_exposure
        }
    
    def generate_risk_assessment(self, 
                               signal: str,
                               current_price: float,
                               market_data: pd.DataFrame,
                               confidence: float,
                               account_balance: float) -> RiskMetrics:
        """
        Generate comprehensive risk assessment for a trading signal
        
        Args:
            signal: Trading signal ('BUY', 'SELL', 'HOLD')
            current_price: Current asset price
            market_data: Historical market data
            confidence: Signal confidence score
            account_balance: Available account balance
            
        Returns:
            RiskMetrics object with all risk calculations
        """
        if signal == 'HOLD':
            return RiskMetrics(
                position_size=0,
                stop_loss=current_price,
                take_profit=current_price,
                risk_reward_ratio=0,
                max_loss=0,
                volatility_risk=0,
                confidence_adjusted_size=0
            )
        
        # Get latest market indicators
        latest_data = market_data.iloc[-1]
        atr = latest_data.get('atr', current_price * 0.02)  # Default to 2% if ATR not available
        volatility = latest_data.get('volatility', 0.02)
        
        # Calculate stop loss and take profit
        stop_loss = self.calculate_stop_loss(current_price, signal, atr, volatility)
        take_profit = self.calculate_take_profit(current_price, stop_loss, signal)
        
        # Calculate position size
        position_size = self.calculate_position_size(
            current_price, stop_loss, account_balance, confidence
        )
        
        # Risk-reward ratio
        risk_amount = abs(current_price - stop_loss)
        reward_amount = abs(take_profit - current_price)
        risk_reward_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        # Maximum potential loss
        max_loss = position_size * risk_amount
        
        # Market risk assessment
        market_risk = self.assess_market_risk(market_data)
        volatility_risk = market_risk['volatility_risk']
        
        # Adjust position size for high market risk
        if market_risk['overall_risk'] > 0.7:
            position_size *= 0.5  # Reduce position size in high-risk conditions
        
        confidence_adjusted_size = position_size * confidence
        
        return RiskMetrics(
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            risk_reward_ratio=risk_reward_ratio,
            max_loss=max_loss,
            volatility_risk=volatility_risk,
            confidence_adjusted_size=confidence_adjusted_size
        )
    
    def validate_signal(self, 
                       signal: str,
                       risk_metrics: RiskMetrics,
                       market_data: pd.DataFrame) -> Dict[str, any]:
        """
        Validate trading signal against risk management rules
        
        Args:
            signal: Trading signal
            risk_metrics: Calculated risk metrics
            market_data: Market data for validation
            
        Returns:
            Validation results with recommendations
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'recommendations': [],
            'risk_score': 0.0
        }
        
        # Check risk-reward ratio
        if risk_metrics.risk_reward_ratio < 1.5:
            validation_results['warnings'].append("Low risk-reward ratio")
            validation_results['risk_score'] += 0.2
        
        # Check maximum loss
        max_acceptable_loss = self.initial_capital * self.max_risk_per_trade
        if risk_metrics.max_loss > max_acceptable_loss:
            validation_results['warnings'].append("Exceeds maximum acceptable loss")
            validation_results['risk_score'] += 0.3
            validation_results['is_valid'] = False
        
        # Check volatility risk
        if risk_metrics.volatility_risk > 0.8:
            validation_results['warnings'].append("High volatility detected")
            validation_results['recommendations'].append("Consider reducing position size")
            validation_results['risk_score'] += 0.2
        
        # Check market conditions
        market_risk = self.assess_market_risk(market_data)
        if market_risk['overall_risk'] > 0.8:
            validation_results['warnings'].append("High overall market risk")
            validation_results['recommendations'].append("Consider waiting for better conditions")
            validation_results['risk_score'] += 0.3
        
        # Final risk score (0-1)
        validation_results['risk_score'] = min(validation_results['risk_score'], 1.0)
        
        # Invalidate signal if risk score is too high
        if validation_results['risk_score'] > 0.7:
            validation_results['is_valid'] = False
        
        return validation_results