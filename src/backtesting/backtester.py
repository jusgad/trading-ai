import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, asdict
from loguru import logger

from src.signals.signal_generator import SignalGenerator, TradingSignal
from config.config import config

@dataclass
class Trade:
    """Individual trade record"""
    symbol: str
    entry_date: datetime
    exit_date: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    signal_type: str  # 'BUY' or 'SELL'
    position_size: float
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    stop_loss: float
    take_profit: float
    exit_reason: str  # 'STOP_LOSS', 'TAKE_PROFIT', 'SIGNAL_CHANGE', 'END_OF_PERIOD'
    duration_days: Optional[int]

@dataclass
class BacktestResults:
    """Comprehensive backtesting results"""
    initial_capital: float
    final_capital: float
    total_return: float
    total_return_percentage: float
    max_drawdown: float
    max_drawdown_percentage: float
    sharpe_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    avg_trade_duration: float
    trades: List[Trade]
    daily_returns: pd.Series
    portfolio_values: pd.Series
    benchmark_return: float

class Backtester:
    """Comprehensive backtesting framework for trading strategies"""
    
    def __init__(self, initial_capital: float = None, commission: float = None, slippage: float = 0.001):
        self.initial_capital = initial_capital or config.INITIAL_CAPITAL
        self.commission = commission or config.COMMISSION
        self.slippage = slippage
        self.signal_generator = SignalGenerator(self.initial_capital)
        
    def run_backtest(self, 
                    symbols: List[str], 
                    start_date: str, 
                    end_date: str,
                    rebalance_frequency: str = 'daily') -> BacktestResults:
        """
        Run comprehensive backtest
        
        Args:
            symbols: List of symbols to test
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            rebalance_frequency: How often to generate new signals
            
        Returns:
            BacktestResults object
        """
        logger.info(f"Starting backtest for {symbols} from {start_date} to {end_date}")
        
        # Initialize tracking variables
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        positions = {}  # {symbol: {'shares': int, 'entry_price': float, 'entry_date': datetime}}
        trades = []
        daily_values = []
        daily_dates = []
        
        # Generate date range
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        # Train models for all symbols first
        for symbol in symbols:
            logger.info(f"Training model for {symbol}")
            self.signal_generator.train_model(symbol, training_period="5y")
        
        current_date = start_dt
        while current_date <= end_dt:
            try:
                # Get signals for all symbols
                signals = {}
                for symbol in symbols:
                    signal = self._get_historical_signal(symbol, current_date)
                    if signal:
                        signals[symbol] = signal
                
                # Process signals and execute trades
                cash, positions, new_trades = self._process_signals(
                    signals, cash, positions, current_date
                )
                trades.extend(new_trades)
                
                # Calculate portfolio value
                portfolio_value = cash
                for symbol, position in positions.items():
                    current_price = self._get_price_on_date(symbol, current_date)
                    if current_price:
                        portfolio_value += position['shares'] * current_price
                
                daily_values.append(portfolio_value)
                daily_dates.append(current_date)
                
                # Move to next trading day
                current_date += timedelta(days=1)
                
            except Exception as e:
                logger.error(f"Error on {current_date}: {str(e)}")
                current_date += timedelta(days=1)
                continue
        
        # Close any remaining positions
        final_trades = self._close_all_positions(positions, end_dt)
        trades.extend(final_trades)
        
        # Calculate final portfolio value
        final_portfolio_value = cash
        for symbol, position in positions.items():
            final_price = self._get_price_on_date(symbol, end_dt)
            if final_price:
                final_portfolio_value += position['shares'] * final_price
        
        # Create results
        results = self._calculate_results(
            trades, daily_values, daily_dates, final_portfolio_value
        )
        
        logger.info(f"Backtest completed. Final return: {results.total_return_percentage:.2f}%")
        return results
    
    def _get_historical_signal(self, symbol: str, date: datetime) -> Optional[TradingSignal]:
        """Get trading signal for a specific date (simulating historical analysis)"""
        try:
            # Fetch data up to the current date
            end_date = date.strftime('%Y-%m-%d')
            start_date = (date - timedelta(days=365)).strftime('%Y-%m-%d')
            
            # In a real backtest, we'd use only data available up to this date
            # For simulation, we'll generate a signal based on historical data
            return self.signal_generator.generate_signal(symbol, use_realtime=False)
            
        except Exception as e:
            logger.error(f"Error getting signal for {symbol} on {date}: {str(e)}")
            return None
    
    def _get_price_on_date(self, symbol: str, date: datetime) -> Optional[float]:
        """Get price for a symbol on a specific date"""
        try:
            # This is a simplified implementation
            # In practice, you'd need historical price data aligned with dates
            data = self.signal_generator.data_fetcher.fetch_data(symbol, period="1y")
            if data is not None and len(data) > 0:
                return data['close'].iloc[-1]  # Use latest available price
            return None
        except:
            return None
    
    def _process_signals(self, signals: Dict[str, TradingSignal], 
                        cash: float, positions: dict, 
                        current_date: datetime) -> Tuple[float, dict, List[Trade]]:
        """Process trading signals and execute trades"""
        new_trades = []
        
        for symbol, signal in signals.items():
            if signal.signal == 'BUY' and symbol not in positions:
                # Enter long position
                if cash >= signal.position_size * signal.current_price:
                    shares = int(signal.position_size)
                    # Apply slippage to entry price
                    entry_price_with_slippage = signal.current_price * (1 + self.slippage)
                    
                    cost = shares * entry_price_with_slippage
                    commission_cost = cost * self.commission
                    total_cost = cost + commission_cost
                    
                    if cash >= total_cost:
                        cash -= total_cost
                        positions[symbol] = {
                            'shares': shares,
                            'entry_price': entry_price_with_slippage,
                            'entry_date': current_date,
                            'stop_loss': signal.stop_loss,
                            'take_profit': signal.take_profit
                        }
                        
                        # Create trade record (entry)
                        trade = Trade(
                            symbol=symbol,
                            entry_date=current_date,
                            exit_date=None,
                            entry_price=signal.current_price,
                            exit_price=None,
                            signal_type='BUY',
                            position_size=shares,
                            pnl=None,
                            pnl_percentage=None,
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            exit_reason='',
                            duration_days=None
                        )
                        new_trades.append(trade)
            
            elif signal.signal == 'SELL' and symbol in positions:
                # Exit long position
                position = positions[symbol]
                shares = position['shares']
                
                # Apply slippage to exit price
                exit_price_with_slippage = signal.current_price * (1 - self.slippage)
                
                revenue = shares * exit_price_with_slippage
                commission_cost = revenue * self.commission
                net_revenue = revenue - commission_cost
                
                cash += net_revenue
                
                # Calculate P&L
                pnl = net_revenue - (shares * position['entry_price'])
                pnl_percentage = (pnl / (shares * position['entry_price'])) * 100
                
                # Update trade record
                duration = (current_date - position['entry_date']).days
                
                # Find the corresponding entry trade and update it
                for trade in reversed(new_trades):
                    if (trade.symbol == symbol and 
                        trade.entry_date == position['entry_date'] and 
                        trade.exit_date is None):
                        trade.exit_date = current_date
                        trade.exit_price = signal.current_price
                        trade.pnl = pnl
                        trade.pnl_percentage = pnl_percentage
                        trade.exit_reason = 'SIGNAL_CHANGE'
                        trade.duration_days = duration
                        break
                
                del positions[symbol]
        
        # Check for stop loss and take profit triggers
        for symbol in list(positions.keys()):
            current_price = self._get_price_on_date(symbol, current_date)
            if current_price:
                position = positions[symbol]
                
                # Check stop loss
                if current_price <= position['stop_loss']:
                    new_trades.extend(self._exit_position(
                        symbol, position, current_price, current_date, 'STOP_LOSS'
                    ))
                    cash += position['shares'] * current_price * (1 - self.commission)
                    del positions[symbol]
                
                # Check take profit
                elif current_price >= position['take_profit']:
                    new_trades.extend(self._exit_position(
                        symbol, position, current_price, current_date, 'TAKE_PROFIT'
                    ))
                    cash += position['shares'] * current_price * (1 - self.commission)
                    del positions[symbol]
        
        return cash, positions, new_trades
    
    def _exit_position(self, symbol: str, position: dict, 
                      exit_price: float, exit_date: datetime, 
                      exit_reason: str) -> List[Trade]:
        """Create exit trade record"""
        shares = position['shares']
        pnl = shares * (exit_price - position['entry_price'])
        pnl_percentage = (pnl / (shares * position['entry_price'])) * 100
        duration = (exit_date - position['entry_date']).days
        
        trade = Trade(
            symbol=symbol,
            entry_date=position['entry_date'],
            exit_date=exit_date,
            entry_price=position['entry_price'],
            exit_price=exit_price,
            signal_type='SELL',
            position_size=shares,
            pnl=pnl,
            pnl_percentage=pnl_percentage,
            stop_loss=position['stop_loss'],
            take_profit=position['take_profit'],
            exit_reason=exit_reason,
            duration_days=duration
        )
        
        return [trade]
    
    def _close_all_positions(self, positions: dict, end_date: datetime) -> List[Trade]:
        """Close all remaining positions at the end of backtest"""
        final_trades = []
        
        for symbol, position in positions.items():
            final_price = self._get_price_on_date(symbol, end_date)
            if final_price:
                final_trades.extend(self._exit_position(
                    symbol, position, final_price, end_date, 'END_OF_PERIOD'
                ))
        
        return final_trades
    
    def _calculate_results(self, trades: List[Trade], daily_values: List[float], 
                          daily_dates: List[datetime], final_value: float) -> BacktestResults:
        """Calculate comprehensive backtest results"""
        
        # Filter completed trades
        completed_trades = [t for t in trades if t.exit_date is not None]
        
        # Basic metrics
        total_return = final_value - self.initial_capital
        total_return_percentage = (total_return / self.initial_capital) * 100
        
        # Trade statistics
        winning_trades = [t for t in completed_trades if t.pnl > 0]
        losing_trades = [t for t in completed_trades if t.pnl <= 0]
        
        win_rate = len(winning_trades) / len(completed_trades) if completed_trades else 0
        
        avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
        
        largest_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
        largest_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0
        
        # Profit factor
        gross_profit = sum([t.pnl for t in winning_trades])
        gross_loss = abs(sum([t.pnl for t in losing_trades]))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade duration
        avg_trade_duration = np.mean([t.duration_days for t in completed_trades if t.duration_days]) if completed_trades else 0
        
        # Convert to pandas series for advanced calculations
        portfolio_series = pd.Series(daily_values, index=daily_dates)
        daily_returns = portfolio_series.pct_change().dropna()
        
        # Sharpe ratio (assuming 252 trading days per year)
        if len(daily_returns) > 0 and daily_returns.std() > 0:
            sharpe_ratio = (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252))
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        peak = portfolio_series.expanding().max()
        drawdown = (portfolio_series - peak) / peak
        max_drawdown = drawdown.min()
        max_drawdown_percentage = max_drawdown * 100
        
        # Benchmark return (assuming 7% annual return)
        benchmark_return = 0.07 * len(daily_values) / 252
        
        return BacktestResults(
            initial_capital=self.initial_capital,
            final_capital=final_value,
            total_return=total_return,
            total_return_percentage=total_return_percentage,
            max_drawdown=abs(max_drawdown * self.initial_capital),
            max_drawdown_percentage=abs(max_drawdown_percentage),
            sharpe_ratio=sharpe_ratio,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(completed_trades),
            winning_trades=len(winning_trades),
            losing_trades=len(losing_trades),
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            avg_trade_duration=avg_trade_duration,
            trades=completed_trades,
            daily_returns=daily_returns,
            portfolio_values=portfolio_series,
            benchmark_return=benchmark_return * 100
        )
    
    def generate_report(self, results: BacktestResults, save_path: str = None) -> str:
        """Generate comprehensive backtest report"""
        
        report = f"""
=== TRADING AI BACKTEST REPORT ===

SUMMARY STATISTICS:
Initial Capital: ${results.initial_capital:,.2f}
Final Capital: ${results.final_capital:,.2f}
Total Return: ${results.total_return:,.2f} ({results.total_return_percentage:.2f}%)
Benchmark Return: {results.benchmark_return:.2f}%
Outperformance: {results.total_return_percentage - results.benchmark_return:.2f}%

RISK METRICS:
Maximum Drawdown: ${results.max_drawdown:,.2f} ({results.max_drawdown_percentage:.2f}%)
Sharpe Ratio: {results.sharpe_ratio:.3f}

TRADE STATISTICS:
Total Trades: {results.total_trades}
Winning Trades: {results.winning_trades} ({results.win_rate:.1%})
Losing Trades: {results.losing_trades}
Profit Factor: {results.profit_factor:.2f}

TRADE PERFORMANCE:
Average Win: ${results.avg_win:.2f}
Average Loss: ${results.avg_loss:.2f}
Largest Win: ${results.largest_win:.2f}
Largest Loss: ${results.largest_loss:.2f}
Average Trade Duration: {results.avg_trade_duration:.1f} days

=== END REPORT ===
        """
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Report saved to {save_path}")
        
        return report
    
    def plot_results(self, results: BacktestResults, save_path: str = None):
        """Generate visualization plots for backtest results"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Portfolio value over time
        axes[0, 0].plot(results.portfolio_values.index, results.portfolio_values.values)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].grid(True)
        
        # Daily returns distribution
        axes[0, 1].hist(results.daily_returns, bins=50, alpha=0.7)
        axes[0, 1].set_title('Daily Returns Distribution')
        axes[0, 1].set_xlabel('Daily Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True)
        
        # Trade P&L distribution
        trade_pnl = [t.pnl for t in results.trades if t.pnl is not None]
        if trade_pnl:
            axes[1, 0].hist(trade_pnl, bins=30, alpha=0.7)
            axes[1, 0].set_title('Trade P&L Distribution')
            axes[1, 0].set_xlabel('P&L ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].grid(True)
        
        # Drawdown chart
        peak = results.portfolio_values.expanding().max()
        drawdown = (results.portfolio_values - peak) / peak * 100
        axes[1, 1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[1, 1].set_title('Drawdown Over Time')
        axes[1, 1].set_ylabel('Drawdown (%)')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Charts saved to {save_path}")
        
        plt.show()
        
        return fig