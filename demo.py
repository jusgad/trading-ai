#!/usr/bin/env python3
"""
Trading AI Demo Script
Demonstrates the key capabilities of the Trading AI system
"""

import sys
import os
import time
from datetime import datetime, timedelta

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def print_header(title):
    """Print formatted header"""
    print("\n" + "="*60)
    print(f"🤖 {title}")
    print("="*60)

def print_section(title):
    """Print formatted section"""
    print(f"\n📊 {title}")
    print("-"*40)

def demo_signal_generation():
    """Demonstrate signal generation"""
    print_header("DEMO: Generación de Señales de Trading")
    
    try:
        from src.signals.signal_generator import SignalGenerator
        from src.monitoring.performance_monitor import PerformanceMonitor
        
        print("✅ Inicializando sistema...")
        generator = SignalGenerator(account_balance=100000)
        monitor = PerformanceMonitor()
        
        # Demo symbols
        demo_symbols = ['AAPL', 'GOOGL', 'MSFT']
        
        print(f"📈 Generando señales para: {', '.join(demo_symbols)}")
        print("⏳ Esto puede tomar unos segundos...")
        
        signals = []
        for symbol in demo_symbols:
            try:
                print(f"   Analizando {symbol}...")
                signal = generator.generate_signal(symbol)
                if signal:
                    signals.append(signal)
                    # Log signal to monitoring system
                    monitor.log_signal(signal)
            except Exception as e:
                print(f"   ⚠️  Error con {symbol}: {str(e)}")
        
        if signals:
            print_section("Señales Generadas")
            for i, signal in enumerate(signals, 1):
                print(f"\n{i}. {signal.symbol}")
                print(f"   Señal: {_format_signal(signal.signal)} ({signal.confidence:.1%} confianza)")
                print(f"   Precio actual: ${signal.current_price:.2f}")
                print(f"   Stop Loss: ${signal.stop_loss:.2f}")
                print(f"   Take Profit: ${signal.take_profit:.2f}")
                print(f"   Tamaño posición: {signal.position_size:.0f}")
                print(f"   Riesgo máximo: ${signal.max_loss:.2f}")
                print(f"   Ratio R/R: {signal.risk_reward_ratio:.1f}")
                print(f"   Razón: {signal.reasoning[:100]}...")
        else:
            print("⚠️  No se generaron señales válidas")
            
    except ImportError as e:
        print(f"❌ Error de importación: {e}")
        print("💡 Ejecuta: pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_technical_analysis():
    """Demonstrate technical analysis"""
    print_header("DEMO: Análisis Técnico")
    
    try:
        from src.data.data_fetcher import MarketDataFetcher
        from src.indicators.technical_indicators import TechnicalIndicators
        
        print("📊 Obteniendo datos de mercado...")
        fetcher = MarketDataFetcher()
        
        # Get sample data
        symbol = 'AAPL'
        data = fetcher.fetch_data(symbol, period='1mo')
        
        if data is not None and len(data) > 0:
            print(f"✅ Datos obtenidos: {len(data)} puntos para {symbol}")
            
            print("🔧 Calculando indicadores técnicos...")
            indicators = TechnicalIndicators.calculate_all_indicators(data)
            
            latest = indicators.iloc[-1]
            
            print_section("Análisis Técnico Actual")
            print(f"Símbolo: {symbol}")
            print(f"Precio actual: ${latest['close']:.2f}")
            print(f"Volumen: {latest['volume']:,}")
            
            if 'rsi' in indicators.columns:
                rsi = latest['rsi']
                rsi_status = "Sobrecomprado" if rsi > 70 else "Sobrevendido" if rsi < 30 else "Neutral"
                print(f"RSI: {rsi:.1f} ({rsi_status})")
            
            if 'sma_20' in indicators.columns and 'sma_50' in indicators.columns:
                trend = "Alcista" if latest['sma_20'] > latest['sma_50'] else "Bajista"
                print(f"Tendencia (SMA): {trend}")
            
            if 'macd' in indicators.columns and 'macd_signal' in indicators.columns:
                macd_trend = "Alcista" if latest['macd'] > latest['macd_signal'] else "Bajista"
                print(f"MACD: {macd_trend}")
            
            if 'bb_upper' in indicators.columns and 'bb_lower' in indicators.columns:
                bb_pos = "Banda superior" if latest['close'] > latest['bb_upper'] else \
                        "Banda inferior" if latest['close'] < latest['bb_lower'] else \
                        "Rango normal"
                print(f"Bollinger Bands: {bb_pos}")
            
            if 'volatility' in indicators.columns:
                vol_level = "Alta" if latest['volatility'] > 0.03 else "Normal"
                print(f"Volatilidad: {vol_level} ({latest['volatility']:.1%})")
            
        else:
            print("❌ No se pudieron obtener datos de mercado")
            print("💡 Verifica tu conexión a internet")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_risk_management():
    """Demonstrate risk management"""
    print_header("DEMO: Gestión de Riesgo")
    
    try:
        from src.risk.risk_manager import RiskManager
        import pandas as pd
        import numpy as np
        
        print("🛡️  Inicializando sistema de gestión de riesgo...")
        risk_manager = RiskManager(initial_capital=100000)
        
        # Create sample market data
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        np.random.seed(42)
        
        prices = [100]
        for _ in range(99):
            prices.append(prices[-1] * (1 + np.random.normal(0, 0.02)))
        
        sample_data = pd.DataFrame({
            'open': prices,
            'high': [p * 1.02 for p in prices],
            'low': [p * 0.98 for p in prices],
            'close': prices,
            'volume': np.random.randint(1000000, 10000000, 100),
            'atr': [2.0] * 100,
            'volatility': [0.02] * 100
        }, index=dates)
        
        print_section("Ejemplo de Cálculo de Riesgo")
        
        current_price = 100.0
        confidence = 0.75
        account_balance = 100000.0
        
        print(f"Precio actual: ${current_price:.2f}")
        print(f"Confianza de señal: {confidence:.1%}")
        print(f"Balance de cuenta: ${account_balance:,.0f}")
        
        # Calculate risk metrics
        risk_metrics = risk_manager.generate_risk_assessment(
            signal='BUY',
            current_price=current_price,
            market_data=sample_data,
            confidence=confidence,
            account_balance=account_balance
        )
        
        print(f"\n📋 Métricas de Riesgo Calculadas:")
        print(f"   Tamaño de posición: {risk_metrics.position_size:.0f} acciones")
        print(f"   Stop Loss: ${risk_metrics.stop_loss:.2f}")
        print(f"   Take Profit: ${risk_metrics.take_profit:.2f}")
        print(f"   Ratio Riesgo/Beneficio: {risk_metrics.risk_reward_ratio:.1f}")
        print(f"   Pérdida máxima: ${risk_metrics.max_loss:.2f}")
        print(f"   Riesgo de volatilidad: {risk_metrics.volatility_risk:.1%}")
        
        # Validate signal
        validation = risk_manager.validate_signal('BUY', risk_metrics, sample_data)
        
        print(f"\n✅ Validación de Señal:")
        print(f"   Válida: {'Sí' if validation['is_valid'] else 'No'}")
        print(f"   Score de riesgo: {validation['risk_score']:.1%}")
        if validation['warnings']:
            print(f"   Advertencias: {', '.join(validation['warnings'])}")
        if validation['recommendations']:
            print(f"   Recomendaciones: {', '.join(validation['recommendations'])}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_backtesting():
    """Demonstrate backtesting (simplified)"""
    print_header("DEMO: Backtesting")
    
    try:
        print("📈 El backtesting completo requiere datos históricos extensos")
        print("🔧 Mostrando capacidades del framework...")
        
        from src.backtesting.backtester import Backtester
        
        backtester = Backtester(initial_capital=100000, commission=0.001)
        
        print_section("Configuración de Backtesting")
        print(f"Capital inicial: ${backtester.initial_capital:,.0f}")
        print(f"Comisión: {backtester.commission:.1%}")
        print("Símbolos de ejemplo: AAPL, GOOGL, MSFT")
        print("Período de ejemplo: 2023-01-01 a 2024-01-01")
        
        print(f"\n💡 Para ejecutar un backtest completo:")
        print(f"   python main.py backtest --symbols AAPL GOOGL --start-date 2023-01-01")
        
        print(f"\n📊 El backtesting incluye:")
        print(f"   • Retorno total y porcentual")
        print(f"   • Ratio de Sharpe")
        print(f"   • Máximo drawdown")
        print(f"   • Tasa de éxito")
        print(f"   • Factor de beneficio")
        print(f"   • Duración promedio de operaciones")
        print(f"   • Análisis de trades individuales")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def demo_monitoring():
    """Demonstrate performance monitoring"""
    print_header("DEMO: Monitoreo de Rendimiento")
    
    try:
        from src.monitoring.performance_monitor import PerformanceMonitor
        from src.signals.signal_generator import TradingSignal
        from datetime import datetime
        
        print("📊 Inicializando sistema de monitoreo...")
        monitor = PerformanceMonitor()
        
        print_section("Capacidades de Monitoreo")
        print("✅ Base de datos SQLite para almacenamiento")
        print("✅ Seguimiento de señales activas")
        print("✅ Cálculo automático de P&L")
        print("✅ Reportes de rendimiento")
        print("✅ Análisis por símbolo y confianza")
        
        # Get dashboard data
        dashboard_data = monitor.get_dashboard_data()
        
        overall_perf = dashboard_data['overall_performance']
        
        print_section("Estado Actual del Sistema")
        print(f"Señales totales: {overall_perf['total_signals']}")
        print(f"Señales cerradas: {overall_perf['closed_signals']}")
        print(f"Señales activas: {overall_perf['active_signals']}")
        print(f"Tasa de éxito: {overall_perf['win_rate']:.1%}")
        print(f"P&L promedio: ${overall_perf['avg_pnl']:.2f}")
        
        print(f"\n💡 Para ver reportes detallados:")
        print(f"   python main.py report")
        print(f"   python main.py dashboard  # Tab 'Settings' > Performance")
        
    except Exception as e:
        print(f"❌ Error: {e}")

def _format_signal(signal):
    """Format signal with emoji"""
    if signal == 'BUY':
        return '🟢 COMPRAR'
    elif signal == 'SELL':
        return '🔴 VENDER'
    else:
        return '⚪ MANTENER'

def demo_dashboard_info():
    """Show dashboard information"""
    print_header("DEMO: Dashboard Web")
    
    print("🌐 El Trading AI incluye un dashboard web completo")
    
    print_section("Características del Dashboard")
    print("🎯 Live Signals:")
    print("   • Generación de señales en tiempo real")
    print("   • Análisis de confianza y riesgo")
    print("   • Recomendaciones de posición")
    
    print("\n📊 Analysis:")
    print("   • Gráficos interactivos de precios")
    print("   • Indicadores técnicos superpuestos")
    print("   • Análisis de múltiples timeframes")
    
    print("\n🔄 Backtesting:")
    print("   • Interface para backtesting histórico")
    print("   • Métricas de rendimiento detalladas")
    print("   • Gráficos de drawdown y retornos")
    
    print("\n⚙️ Settings:")
    print("   • Configuración de parámetros de riesgo")
    print("   • Entrenamiento de modelos AI")
    print("   • Monitoreo de rendimiento")
    
    print(f"\n🚀 Para iniciar el dashboard:")
    print(f"   ./scripts/start_dashboard.sh")
    print(f"   # o")
    print(f"   python main.py dashboard")
    print(f"\n🌐 Luego visita: http://localhost:8501")

def main():
    """Run all demos"""
    print("🤖 TRADING AI - SISTEMA DE DEMOSTRACIÓN")
    print("=" * 60)
    print("Este demo muestra las principales capacidades del sistema")
    print("⚠️  Los datos generados son para demostración únicamente")
    print("")
    
    demos = [
        ("Información del Dashboard", demo_dashboard_info),
        ("Análisis Técnico", demo_technical_analysis),
        ("Gestión de Riesgo", demo_risk_management),
        ("Generación de Señales", demo_signal_generation),
        ("Backtesting", demo_backtesting),
        ("Monitoreo de Rendimiento", demo_monitoring),
    ]
    
    try:
        for i, (name, demo_func) in enumerate(demos, 1):
            print(f"\n{'='*20} DEMO {i}/{len(demos)} {'='*20}")
            print(f"▶️  {name}")
            
            try:
                demo_func()
            except KeyboardInterrupt:
                print("\n\n⏹️  Demo interrumpido por el usuario")
                break
            except Exception as e:
                print(f"\n❌ Error en demo: {e}")
            
            if i < len(demos):
                print(f"\n⏳ Continuando al siguiente demo en 3 segundos...")
                time.sleep(3)
        
        print_header("DEMO COMPLETADO")
        print("🎉 Has visto las principales capacidades del Trading AI!")
        print("")
        print("🚀 Próximos pasos:")
        print("1. ./scripts/install.sh           # Instalar dependencias")
        print("2. ./scripts/run_tests.sh         # Verificar funcionamiento")
        print("3. python main.py dashboard       # Lanzar interfaz web")
        print("4. python main.py signals         # Generar señales reales")
        print("5. Ver docs/USAGE_GUIDE.md        # Guía detallada")
        print("")
        print("📚 Documentación disponible:")
        print("   • README.md - Información general")
        print("   • docs/USAGE_GUIDE.md - Guía de uso detallada")
        print("   • docs/API_REFERENCE.md - Referencia de API")
        print("")
        print("⚠️  IMPORTANTE: Este sistema es para análisis y educación.")
        print("   Siempre ejecuta operaciones manualmente tras revisar las señales.")
        
    except KeyboardInterrupt:
        print("\n\n👋 Demo cancelado. ¡Gracias por probar Trading AI!")

if __name__ == "__main__":
    main()