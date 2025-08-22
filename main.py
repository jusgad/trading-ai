#!/usr/bin/env python3
"""
Trading AI Main Application
Entry point for the market analysis and signal generation system
"""

import sys
import os
import argparse
from datetime import datetime
from loguru import logger

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.signals.signal_generator import SignalGenerator
from src.backtesting.backtester import Backtester
from src.monitoring.performance_monitor import PerformanceMonitor
from config.config import config

def setup_logging():
    """Setup logging configuration"""
    logger.remove()  # Remove default handler
    
    # Console logging
    logger.add(
        sys.stdout,
        level=config.LOG_LEVEL,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
    )
    
    # File logging
    logger.add(
        config.LOG_FILE,
        level=config.LOG_LEVEL,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="1 day",
        retention="30 days"
    )
    
    logger.info("Trading AI system initialized")

def run_dashboard():
    """Run the Streamlit dashboard"""
    import subprocess
    logger.info("Starting Streamlit dashboard...")
    
    dashboard_path = os.path.join(os.path.dirname(__file__), 'src', 'ui', 'dashboard.py')
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', dashboard_path])

def generate_signals(symbols=None):
    """Generate trading signals for specified symbols"""
    logger.info("Generating trading signals...")
    
    signal_generator = SignalGenerator()
    monitor = PerformanceMonitor()
    
    if symbols is None:
        symbols = config.DEFAULT_SYMBOLS
    
    signals = signal_generator.generate_multiple_signals(symbols)
    
    if not signals:
        logger.warning("No signals generated")
        return
    
    print("\n" + "="*80)
    print("🤖 TRADING AI SIGNALS")
    print("="*80)
    print(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Symbols analyzed: {', '.join(symbols)}")
    print("-"*80)
    
    for signal in signals:
        # Log signal to monitoring system
        monitor.log_signal(signal)
        
        # Display signal
        print(f"\n📊 {signal.symbol}")
        print(f"Signal: {signal.signal} ({'🟢' if signal.signal == 'BUY' else '🔴' if signal.signal == 'SELL' else '⚪'})")
        print(f"Confidence: {signal.confidence:.1%}")
        print(f"Current Price: ${signal.current_price:.2f}")
        print(f"Entry Price: ${signal.entry_price:.2f}")
        print(f"Stop Loss: ${signal.stop_loss:.2f}")
        print(f"Take Profit: ${signal.take_profit:.2f}")
        print(f"Risk/Reward: {signal.risk_reward_ratio:.1f}")
        print(f"Position Size: {signal.position_size:.0f}")
        print(f"Max Loss: ${signal.max_loss:.2f}")
        print(f"Reasoning: {signal.reasoning}")
        print("-"*40)
    
    print(f"\n✅ Generated {len(signals)} signals")
    print("="*80)

def run_backtest(symbols=None, start_date=None, end_date=None):
    """Run backtesting for specified parameters"""
    logger.info("Starting backtest...")
    
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    if start_date is None:
        start_date = '2023-01-01'
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    backtester = Backtester()
    results = backtester.run_backtest(symbols, start_date, end_date)
    
    # Generate and display report
    report = backtester.generate_report(results)
    print(report)
    
    # Save report to file
    report_filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    logger.info(f"Backtest report saved to {report_filename}")

def train_models(symbols=None):
    """Train AI models for specified symbols"""
    logger.info("Training AI models...")
    
    if symbols is None:
        symbols = config.DEFAULT_SYMBOLS
    
    signal_generator = SignalGenerator()
    
    for symbol in symbols:
        logger.info(f"Training model for {symbol}")
        success = signal_generator.train_model(symbol)
        
        if success:
            logger.info(f"✅ Successfully trained model for {symbol}")
        else:
            logger.error(f"❌ Failed to train model for {symbol}")
    
    logger.info("Model training completed")

def generate_performance_report():
    """Generate performance report"""
    logger.info("Generating performance report...")
    
    monitor = PerformanceMonitor()
    report = monitor.generate_weekly_report()
    
    print("\n" + "="*60)
    print("📈 PERFORMANCE REPORT")
    print("="*60)
    print(report)
    
    # Save report
    report_filename = f"performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    logger.info(f"Performance report saved to {report_filename}")

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description="Trading AI System")
    parser.add_argument('command', choices=['dashboard', 'signals', 'backtest', 'train', 'report'], 
                       help='Command to execute')
    
    # Optional arguments
    parser.add_argument('--symbols', nargs='+', help='Symbols to analyze')
    parser.add_argument('--start-date', help='Start date for backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date for backtest (YYYY-MM-DD)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Setup logging
    config.LOG_LEVEL = args.log_level
    setup_logging()
    
    try:
        if args.command == 'dashboard':
            run_dashboard()
        
        elif args.command == 'signals':
            generate_signals(args.symbols)
        
        elif args.command == 'backtest':
            run_backtest(args.symbols, args.start_date, args.end_date)
        
        elif args.command == 'train':
            train_models(args.symbols)
        
        elif args.command == 'report':
            generate_performance_report()
    
    except KeyboardInterrupt:
        logger.info("Application interrupted by user")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()