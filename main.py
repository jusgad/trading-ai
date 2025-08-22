#!/usr/bin/env python3
"""
Aplicación Principal de Trading AI
Punto de entrada para el sistema de análisis de mercado y generación de señales
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
    """Configurar sistema de logging"""
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
    
    logger.info("Sistema Trading AI inicializado")

def run_dashboard():
    """Ejecutar el dashboard de Streamlit"""
    import subprocess
    logger.info("Iniciando dashboard de Streamlit...")
    
    dashboard_path = os.path.join(os.path.dirname(__file__), 'src', 'ui', 'dashboard.py')
    subprocess.run([sys.executable, '-m', 'streamlit', 'run', dashboard_path])

def generate_signals(symbols=None):
    """Generar señales de trading para símbolos específicos"""
    logger.info("Generando señales de trading...")
    
    signal_generator = SignalGenerator()
    monitor = PerformanceMonitor()
    
    if symbols is None:
        symbols = config.DEFAULT_SYMBOLS
    
    signals = signal_generator.generate_multiple_signals(symbols)
    
    if not signals:
        logger.warning("No se generaron señales")
        return
    
    print("\n" + "="*80)
    print("🤖 SEÑALES DE TRADING AI")
    print("="*80)
    print(f"Generado el: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Símbolos analizados: {', '.join(symbols)}")
    print("-"*80)
    
    for signal in signals:
        # Log signal to monitoring system
        monitor.log_signal(signal)
        
        # Display signal
        print(f"\n📊 {signal.symbol}")
        print(f"Señal: {signal.signal} ({'🟢' if signal.signal == 'BUY' else '🔴' if signal.signal == 'SELL' else '⚪'})")
        print(f"Confianza: {signal.confidence:.1%}")
        print(f"Precio Actual: ${signal.current_price:.2f}")
        print(f"Precio de Entrada: ${signal.entry_price:.2f}")
        print(f"Stop Loss: ${signal.stop_loss:.2f}")
        print(f"Take Profit: ${signal.take_profit:.2f}")
        print(f"Riesgo/Recompensa: {signal.risk_reward_ratio:.1f}")
        print(f"Tamaño de Posición: {signal.position_size:.0f}")
        print(f"Pérdida Máxima: ${signal.max_loss:.2f}")
        print(f"Razonamiento: {signal.reasoning}")
        print("-"*40)
    
    print(f"\n✅ Se generaron {len(signals)} señales")
    print("="*80)

def run_backtest(symbols=None, start_date=None, end_date=None):
    """Ejecutar backtesting con parámetros especificados"""
    logger.info("Iniciando backtest...")
    
    if symbols is None:
        symbols = ['AAPL', 'GOOGL', 'MSFT']
    
    if start_date is None:
        start_date = '2023-01-01'
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    backtester = Backtester()
    results = backtester.run_backtest(symbols, start_date, end_date)
    
    # Generar y mostrar reporte
    report = backtester.generate_report(results)
    print(report)
    
    # Guardar reporte en archivo
    report_filename = f"reporte_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    logger.info(f"Reporte de backtest guardado en {report_filename}")

def train_models(symbols=None):
    """Entrenar modelos de IA para símbolos especificados"""
    logger.info("Entrenando modelos de IA...")
    
    if symbols is None:
        symbols = config.DEFAULT_SYMBOLS
    
    signal_generator = SignalGenerator()
    
    for symbol in symbols:
        logger.info(f"Entrenando modelo para {symbol}")
        success = signal_generator.train_model(symbol)
        
        if success:
            logger.info(f"✅ Modelo entrenado exitosamente para {symbol}")
        else:
            logger.error(f"❌ Fallo al entrenar modelo para {symbol}")
    
    logger.info("Entrenamiento de modelos completado")

def generate_performance_report():
    """Generar reporte de rendimiento"""
    logger.info("Generando reporte de rendimiento...")
    
    monitor = PerformanceMonitor()
    report = monitor.generate_weekly_report()
    
    print("\n" + "="*60)
    print("📈 REPORTE DE RENDIMIENTO")
    print("="*60)
    print(report)
    
    # Guardar reporte
    report_filename = f"reporte_rendimiento_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_filename, 'w') as f:
        f.write(report)
    
    logger.info(f"Reporte de rendimiento guardado en {report_filename}")

def main():
    """Punto de entrada principal de la aplicación"""
    parser = argparse.ArgumentParser(description="Sistema Trading AI")
    parser.add_argument('command', choices=['dashboard', 'signals', 'backtest', 'train', 'report'], 
                       help='Comando a ejecutar')
    
    # Argumentos opcionales
    parser.add_argument('--symbols', nargs='+', help='Símbolos a analizar')
    parser.add_argument('--start-date', help='Fecha de inicio para backtest (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='Fecha de fin para backtest (YYYY-MM-DD)')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'])
    
    args = parser.parse_args()
    
    # Configurar logging
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
        logger.info("Aplicación interrumpida por el usuario")
        sys.exit(0)
    
    except Exception as e:
        logger.error(f"Error en aplicación: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()