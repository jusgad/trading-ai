import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import sqlite3
from loguru import logger
from dataclasses import dataclass, asdict
import os

from src.signals.signal_generator import TradingSignal
from config.config import config

@dataclass
class SignalPerformance:
    """Performance metrics for a trading signal"""
    signal_id: str
    symbol: str
    signal_type: str
    generated_at: datetime
    confidence: float
    entry_price: float
    current_price: Optional[float]
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    status: str  # 'ACTIVE', 'HIT_TP', 'HIT_SL', 'CANCELLED', 'EXPIRED'
    pnl: Optional[float]
    pnl_percentage: Optional[float]
    max_favorable: Optional[float]
    max_adverse: Optional[float]
    duration_hours: Optional[float]
    accuracy_score: Optional[float]

class SignalDatabase:
    """Database manager for signal storage and retrieval"""
    
    def __init__(self, db_path: str = "data/signals.db"):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()
    
    def init_database(self):
        """Initialize the signal database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS signals (
                    signal_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    generated_at TIMESTAMP NOT NULL,
                    confidence REAL NOT NULL,
                    entry_price REAL NOT NULL,
                    current_price REAL,
                    exit_price REAL,
                    stop_loss REAL NOT NULL,
                    take_profit REAL NOT NULL,
                    status TEXT DEFAULT 'ACTIVE',
                    pnl REAL,
                    pnl_percentage REAL,
                    max_favorable REAL,
                    max_adverse REAL,
                    duration_hours REAL,
                    accuracy_score REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS price_updates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    price REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (signal_id) REFERENCES signals (signal_id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_signals_symbol ON signals(symbol);
                CREATE INDEX IF NOT EXISTS idx_signals_status ON signals(status);
                CREATE INDEX IF NOT EXISTS idx_signals_generated_at ON signals(generated_at);
            """)
    
    def save_signal(self, signal: TradingSignal) -> str:
        """Save a trading signal to database"""
        signal_id = f"{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        signal_data = SignalPerformance(
            signal_id=signal_id,
            symbol=signal.symbol,
            signal_type=signal.signal,
            generated_at=signal.timestamp,
            confidence=signal.confidence,
            entry_price=signal.entry_price,
            current_price=signal.current_price,
            exit_price=None,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status='ACTIVE',
            pnl=None,
            pnl_percentage=None,
            max_favorable=None,
            max_adverse=None,
            duration_hours=None,
            accuracy_score=None
        )
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO signals 
                (signal_id, symbol, signal_type, generated_at, confidence, entry_price, 
                 current_price, stop_loss, take_profit, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                signal_data.signal_id, signal_data.symbol, signal_data.signal_type,
                signal_data.generated_at, signal_data.confidence, signal_data.entry_price,
                signal_data.current_price, signal_data.stop_loss, signal_data.take_profit,
                signal_data.status
            ))
        
        logger.info(f"Saved signal {signal_id} to database")
        return signal_id
    
    def update_signal_price(self, signal_id: str, current_price: float):
        """Update current price for a signal"""
        with sqlite3.connect(self.db_path) as conn:
            # Update signal table
            conn.execute("""
                UPDATE signals 
                SET current_price = ? 
                WHERE signal_id = ?
            """, (current_price, signal_id))
            
            # Log price update
            conn.execute("""
                INSERT INTO price_updates (signal_id, symbol, price)
                SELECT signal_id, symbol, ? FROM signals WHERE signal_id = ?
            """, (current_price, signal_id))
    
    def close_signal(self, signal_id: str, exit_price: float, status: str):
        """Close a signal with final results"""
        with sqlite3.connect(self.db_path) as conn:
            # Get signal details
            signal_data = conn.execute("""
                SELECT * FROM signals WHERE signal_id = ?
            """, (signal_id,)).fetchone()
            
            if signal_data:
                entry_price = signal_data[5]  # entry_price column
                signal_type = signal_data[2]  # signal_type column
                generated_at = datetime.fromisoformat(signal_data[3])
                
                # Calculate P&L
                if signal_type == 'BUY':
                    pnl = exit_price - entry_price
                else:  # SELL
                    pnl = entry_price - exit_price
                
                pnl_percentage = (pnl / entry_price) * 100
                duration_hours = (datetime.now() - generated_at).total_seconds() / 3600
                
                # Update signal
                conn.execute("""
                    UPDATE signals 
                    SET exit_price = ?, status = ?, pnl = ?, pnl_percentage = ?, duration_hours = ?
                    WHERE signal_id = ?
                """, (exit_price, status, pnl, pnl_percentage, duration_hours, signal_id))
                
                logger.info(f"Closed signal {signal_id} with {status}, P&L: {pnl:.2f}")
    
    def get_active_signals(self) -> List[SignalPerformance]:
        """Get all active signals"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM signals WHERE status = 'ACTIVE' ORDER BY generated_at DESC
            """)
            
            signals = []
            for row in cursor.fetchall():
                signal = SignalPerformance(
                    signal_id=row[0],
                    symbol=row[1],
                    signal_type=row[2],
                    generated_at=datetime.fromisoformat(row[3]),
                    confidence=row[4],
                    entry_price=row[5],
                    current_price=row[6],
                    exit_price=row[7],
                    stop_loss=row[8],
                    take_profit=row[9],
                    status=row[10],
                    pnl=row[11],
                    pnl_percentage=row[12],
                    max_favorable=row[13],
                    max_adverse=row[14],
                    duration_hours=row[15],
                    accuracy_score=row[16]
                )
                signals.append(signal)
            
            return signals
    
    def get_signal_history(self, symbol: str = None, days: int = 30) -> List[SignalPerformance]:
        """Get signal history"""
        start_date = datetime.now() - timedelta(days=days)
        
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT * FROM signals 
                WHERE generated_at >= ?
            """
            params = [start_date.isoformat()]
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            query += " ORDER BY generated_at DESC"
            
            cursor = conn.execute(query, params)
            
            signals = []
            for row in cursor.fetchall():
                signal = SignalPerformance(
                    signal_id=row[0],
                    symbol=row[1],
                    signal_type=row[2],
                    generated_at=datetime.fromisoformat(row[3]),
                    confidence=row[4],
                    entry_price=row[5],
                    current_price=row[6],
                    exit_price=row[7],
                    stop_loss=row[8],
                    take_profit=row[9],
                    status=row[10],
                    pnl=row[11],
                    pnl_percentage=row[12],
                    max_favorable=row[13],
                    max_adverse=row[14],
                    duration_hours=row[15],
                    accuracy_score=row[16]
                )
                signals.append(signal)
            
            return signals

class PerformanceAnalyzer:
    """Analyze trading signal performance and generate insights"""
    
    def __init__(self, db: SignalDatabase):
        self.db = db
    
    def calculate_overall_performance(self, days: int = 30) -> Dict:
        """Calculate overall performance metrics"""
        signals = self.db.get_signal_history(days=days)
        closed_signals = [s for s in signals if s.status in ['HIT_TP', 'HIT_SL', 'CANCELLED']]
        
        if not closed_signals:
            return {
                'total_signals': len(signals),
                'closed_signals': 0,
                'win_rate': 0,
                'avg_pnl': 0,
                'total_pnl': 0,
                'avg_duration': 0,
                'best_signal': None,
                'worst_signal': None
            }
        
        # Calculate metrics
        winning_signals = [s for s in closed_signals if s.pnl and s.pnl > 0]
        losing_signals = [s for s in closed_signals if s.pnl and s.pnl <= 0]
        
        win_rate = len(winning_signals) / len(closed_signals) if closed_signals else 0
        
        total_pnl = sum(s.pnl for s in closed_signals if s.pnl)
        avg_pnl = total_pnl / len(closed_signals) if closed_signals else 0
        
        avg_duration = np.mean([s.duration_hours for s in closed_signals if s.duration_hours])
        
        best_signal = max(closed_signals, key=lambda x: x.pnl or 0)
        worst_signal = min(closed_signals, key=lambda x: x.pnl or 0)
        
        return {
            'total_signals': len(signals),
            'closed_signals': len(closed_signals),
            'active_signals': len(signals) - len(closed_signals),
            'win_rate': win_rate,
            'winning_signals': len(winning_signals),
            'losing_signals': len(losing_signals),
            'avg_pnl': avg_pnl,
            'total_pnl': total_pnl,
            'avg_duration': avg_duration,
            'best_signal': asdict(best_signal),
            'worst_signal': asdict(worst_signal)
        }
    
    def analyze_by_symbol(self, days: int = 30) -> Dict[str, Dict]:
        """Analyze performance by symbol"""
        symbols = set()
        all_signals = self.db.get_signal_history(days=days)
        
        for signal in all_signals:
            symbols.add(signal.symbol)
        
        symbol_performance = {}
        for symbol in symbols:
            symbol_signals = [s for s in all_signals if s.symbol == symbol]
            closed_signals = [s for s in symbol_signals if s.status in ['HIT_TP', 'HIT_SL', 'CANCELLED']]
            
            if closed_signals:
                winning_signals = [s for s in closed_signals if s.pnl and s.pnl > 0]
                win_rate = len(winning_signals) / len(closed_signals)
                total_pnl = sum(s.pnl for s in closed_signals if s.pnl)
                avg_pnl = total_pnl / len(closed_signals)
            else:
                win_rate = 0
                total_pnl = 0
                avg_pnl = 0
            
            symbol_performance[symbol] = {
                'total_signals': len(symbol_signals),
                'closed_signals': len(closed_signals),
                'win_rate': win_rate,
                'total_pnl': total_pnl,
                'avg_pnl': avg_pnl
            }
        
        return symbol_performance
    
    def analyze_by_confidence(self, days: int = 30) -> Dict:
        """Analyze performance by confidence levels"""
        signals = self.db.get_signal_history(days=days)
        closed_signals = [s for s in signals if s.status in ['HIT_TP', 'HIT_SL', 'CANCELLED']]
        
        # Group by confidence ranges
        confidence_ranges = {
            'low': (0.0, 0.5),
            'medium': (0.5, 0.75),
            'high': (0.75, 1.0)
        }
        
        confidence_performance = {}
        for range_name, (min_conf, max_conf) in confidence_ranges.items():
            range_signals = [
                s for s in closed_signals 
                if min_conf <= s.confidence < max_conf
            ]
            
            if range_signals:
                winning_signals = [s for s in range_signals if s.pnl and s.pnl > 0]
                win_rate = len(winning_signals) / len(range_signals)
                avg_pnl = np.mean([s.pnl for s in range_signals if s.pnl])
            else:
                win_rate = 0
                avg_pnl = 0
            
            confidence_performance[range_name] = {
                'count': len(range_signals),
                'win_rate': win_rate,
                'avg_pnl': avg_pnl,
                'confidence_range': f"{min_conf:.1f}-{max_conf:.1f}"
            }
        
        return confidence_performance
    
    def generate_performance_report(self, days: int = 30) -> str:
        """Generate comprehensive performance report"""
        overall = self.calculate_overall_performance(days)
        by_symbol = self.analyze_by_symbol(days)
        by_confidence = self.analyze_by_confidence(days)
        
        report = f"""
=== TRADING AI PERFORMANCE REPORT ===
Period: Last {days} days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

OVERALL PERFORMANCE:
- Total Signals Generated: {overall['total_signals']}
- Closed Signals: {overall['closed_signals']}
- Active Signals: {overall['active_signals']}
- Win Rate: {overall['win_rate']:.1%}
- Total P&L: ${overall['total_pnl']:.2f}
- Average P&L per Trade: ${overall['avg_pnl']:.2f}
- Average Duration: {overall['avg_duration']:.1f} hours

PERFORMANCE BY SYMBOL:
"""
        
        for symbol, data in by_symbol.items():
            report += f"""
{symbol}:
  - Signals: {data['total_signals']} (Closed: {data['closed_signals']})
  - Win Rate: {data['win_rate']:.1%}
  - Total P&L: ${data['total_pnl']:.2f}
  - Avg P&L: ${data['avg_pnl']:.2f}
"""
        
        report += "\nPERFORMANCE BY CONFIDENCE:\n"
        for conf_range, data in by_confidence.items():
            report += f"""
{conf_range.capitalize()} Confidence ({data['confidence_range']}):
  - Count: {data['count']}
  - Win Rate: {data['win_rate']:.1%}
  - Avg P&L: ${data['avg_pnl']:.2f}
"""
        
        report += "\n=== END REPORT ===\n"
        return report

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self):
        self.db = SignalDatabase()
        self.analyzer = PerformanceAnalyzer(self.db)
    
    def log_signal(self, signal: TradingSignal) -> str:
        """Log a new trading signal"""
        return self.db.save_signal(signal)
    
    def update_active_signals(self, current_prices: Dict[str, float]):
        """Update prices for active signals and check for exit conditions"""
        active_signals = self.db.get_active_signals()
        
        for signal in active_signals:
            if signal.symbol in current_prices:
                current_price = current_prices[signal.symbol]
                self.db.update_signal_price(signal.signal_id, current_price)
                
                # Check exit conditions
                self._check_exit_conditions(signal, current_price)
    
    def _check_exit_conditions(self, signal: SignalPerformance, current_price: float):
        """Check if signal should be closed based on stop loss or take profit"""
        if signal.signal_type == 'BUY':
            if current_price <= signal.stop_loss:
                self.db.close_signal(signal.signal_id, current_price, 'HIT_SL')
                logger.info(f"Signal {signal.signal_id} hit stop loss at {current_price}")
            elif current_price >= signal.take_profit:
                self.db.close_signal(signal.signal_id, current_price, 'HIT_TP')
                logger.info(f"Signal {signal.signal_id} hit take profit at {current_price}")
        
        elif signal.signal_type == 'SELL':
            if current_price >= signal.stop_loss:
                self.db.close_signal(signal.signal_id, current_price, 'HIT_SL')
                logger.info(f"Signal {signal.signal_id} hit stop loss at {current_price}")
            elif current_price <= signal.take_profit:
                self.db.close_signal(signal.signal_id, current_price, 'HIT_TP')
                logger.info(f"Signal {signal.signal_id} hit take profit at {current_price}")
    
    def get_dashboard_data(self) -> Dict:
        """Get data for performance dashboard"""
        return {
            'overall_performance': self.analyzer.calculate_overall_performance(),
            'symbol_performance': self.analyzer.analyze_by_symbol(),
            'confidence_analysis': self.analyzer.analyze_by_confidence(),
            'active_signals': [asdict(s) for s in self.db.get_active_signals()],
            'recent_signals': [asdict(s) for s in self.db.get_signal_history(days=7)]
        }
    
    def generate_daily_report(self) -> str:
        """Generate daily performance report"""
        return self.analyzer.generate_performance_report(days=1)
    
    def generate_weekly_report(self) -> str:
        """Generate weekly performance report"""
        return self.analyzer.generate_performance_report(days=7)
    
    def generate_monthly_report(self) -> str:
        """Generate monthly performance report"""
        return self.analyzer.generate_performance_report(days=30)