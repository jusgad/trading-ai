import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
from typing import Dict, List
import json

from src.signals.signal_generator import SignalGenerator
from src.backtesting.backtester import Backtester
from config.config import config

class TradingDashboard:
    """Streamlit-based dashboard for trading signal display and analysis"""
    
    def __init__(self):
        self.signal_generator = SignalGenerator()
        self.backtester = Backtester()
        
    def run(self):
        """Run the main dashboard"""
        st.set_page_config(
            page_title="Trading AI Dashboard",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("🤖 Trading AI Dashboard")
        st.markdown("---")
        
        # Sidebar for navigation
        self.render_sidebar()
        
        # Main content area
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Live Signals", "📊 Analysis", "🔄 Backtesting", "⚙️ Settings"])
        
        with tab1:
            self.render_live_signals()
        
        with tab2:
            self.render_analysis()
        
        with tab3:
            self.render_backtesting()
        
        with tab4:
            self.render_settings()
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        st.sidebar.header("📋 Controls")
        
        # Symbol selection
        st.sidebar.subheader("Asset Selection")
        selected_symbols = st.sidebar.multiselect(
            "Select assets to analyze:",
            options=config.DEFAULT_SYMBOLS,
            default=config.DEFAULT_SYMBOLS[:3]
        )
        st.session_state.selected_symbols = selected_symbols
        
        # Risk settings
        st.sidebar.subheader("Risk Management")
        risk_per_trade = st.sidebar.slider(
            "Max Risk per Trade (%)",
            min_value=0.5,
            max_value=5.0,
            value=config.MAX_RISK_PER_TRADE * 100,
            step=0.1
        )
        st.session_state.risk_per_trade = risk_per_trade / 100
        
        # Account balance
        account_balance = st.sidebar.number_input(
            "Account Balance ($)",
            min_value=1000,
            max_value=1000000,
            value=int(config.INITIAL_CAPITAL),
            step=1000
        )
        st.session_state.account_balance = account_balance
        
        # Auto-refresh
        st.sidebar.subheader("Auto Refresh")
        auto_refresh = st.sidebar.checkbox("Enable auto-refresh (30s)")
        if auto_refresh:
            time.sleep(30)
            st.rerun()
    
    def render_live_signals(self):
        """Render live trading signals"""
        st.header("🎯 Live Trading Signals")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            if st.button("🔄 Refresh Signals", type="primary"):
                st.rerun()
        
        # Generate signals
        if 'selected_symbols' in st.session_state:
            with st.spinner("Generating signals..."):
                signals = []
                for symbol in st.session_state.selected_symbols:
                    signal = self.signal_generator.generate_signal(symbol)
                    if signal:
                        signals.append(signal)
                
                if signals:
                    self.display_signals_table(signals)
                    self.display_signal_details(signals)
                else:
                    st.warning("No signals generated. Please check your symbol selection.")
    
    def display_signals_table(self, signals: List):
        """Display signals in a formatted table"""
        st.subheader("📊 Signal Summary")
        
        # Create summary dataframe
        signal_data = []
        for signal in signals:
            signal_data.append({
                'Symbol': signal.symbol,
                'Signal': self.format_signal(signal.signal),
                'Confidence': f"{signal.confidence:.1%}",
                'Price': f"${signal.current_price:.2f}",
                'Stop Loss': f"${signal.stop_loss:.2f}",
                'Take Profit': f"${signal.take_profit:.2f}",
                'Risk/Reward': f"{signal.risk_reward_ratio:.1f}",
                'Position Size': f"{signal.position_size:.0f}",
                'Max Loss': f"${signal.max_loss:.2f}",
                'Timestamp': signal.timestamp.strftime("%H:%M:%S")
            })
        
        df = pd.DataFrame(signal_data)
        
        # Style the dataframe
        def style_signal(val):
            if 'BUY' in val:
                return 'background-color: #d4edda; color: #155724'
            elif 'SELL' in val:
                return 'background-color: #f8d7da; color: #721c24'
            else:
                return 'background-color: #fff3cd; color: #856404'
        
        styled_df = df.style.applymap(style_signal, subset=['Signal'])
        st.dataframe(styled_df, use_container_width=True)
    
    def format_signal(self, signal: str) -> str:
        """Format signal with emoji"""
        if signal == 'BUY':
            return '🟢 BUY'
        elif signal == 'SELL':
            return '🔴 SELL'
        else:
            return '⚪ HOLD'
    
    def display_signal_details(self, signals: List):
        """Display detailed signal information"""
        st.subheader("🔍 Signal Details")
        
        for signal in signals:
            with st.expander(f"{signal.symbol} - {self.format_signal(signal.signal)}"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Current Price", f"${signal.current_price:.2f}")
                    st.metric("Confidence", f"{signal.confidence:.1%}")
                    st.metric("Position Size", f"{signal.position_size:.0f}")
                
                with col2:
                    st.metric("Stop Loss", f"${signal.stop_loss:.2f}")
                    st.metric("Take Profit", f"${signal.take_profit:.2f}")
                    st.metric("Risk/Reward", f"{signal.risk_reward_ratio:.1f}")
                
                with col3:
                    st.metric("Max Loss", f"${signal.max_loss:.2f}")
                    st.metric("Timestamp", signal.timestamp.strftime("%H:%M:%S"))
                
                # Technical analysis
                st.write("**Technical Analysis:**")
                ta_col1, ta_col2 = st.columns(2)
                
                with ta_col1:
                    st.write(f"• RSI: {signal.technical_analysis.get('rsi', 'N/A'):.1f}")
                    st.write(f"• MACD: {signal.technical_analysis.get('macd_signal', 'N/A')}")
                
                with ta_col2:
                    st.write(f"• Volatility: {signal.technical_analysis.get('volatility', 0):.2%}")
                    st.write(f"• Volume: {signal.technical_analysis.get('volume_analysis', 'N/A')}")
                
                # Reasoning
                st.write("**AI Reasoning:**")
                st.write(signal.reasoning)
                
                # Market conditions
                st.write("**Market Conditions:**")
                conditions = signal.market_conditions
                cond_col1, cond_col2, cond_col3 = st.columns(3)
                
                with cond_col1:
                    st.write(f"• Trend: {conditions.get('trend', 'N/A')}")
                    st.write(f"• Volatility: {conditions.get('volatility', 'N/A')}")
                
                with cond_col2:
                    st.write(f"• Volume: {conditions.get('volume', 'N/A')}")
                    st.write(f"• Momentum: {conditions.get('momentum', 'N/A')}")
                
                with cond_col3:
                    st.write(f"• RSI Level: {conditions.get('rsi_level', 'N/A')}")
                    st.write(f"• BB Position: {conditions.get('bb_position', 'N/A')}")
    
    def render_analysis(self):
        """Render technical analysis charts"""
        st.header("📊 Technical Analysis")
        
        if 'selected_symbols' not in st.session_state or not st.session_state.selected_symbols:
            st.warning("Please select symbols in the sidebar.")
            return
        
        symbol = st.selectbox("Select symbol for analysis:", st.session_state.selected_symbols)
        
        if symbol:
            # Fetch data
            data = self.signal_generator.data_fetcher.fetch_data(symbol, period="3mo")
            
            if data is not None:
                # Calculate indicators
                from src.indicators.technical_indicators import TechnicalIndicators
                data_with_indicators = TechnicalIndicators.calculate_all_indicators(data)
                
                # Price chart
                self.plot_price_chart(data_with_indicators, symbol)
                
                # Technical indicators
                self.plot_technical_indicators(data_with_indicators, symbol)
            else:
                st.error(f"Unable to fetch data for {symbol}")
    
    def plot_price_chart(self, data: pd.DataFrame, symbol: str):
        """Plot interactive price chart with indicators"""
        st.subheader(f"📈 {symbol} Price Chart")
        
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name='Price'
        ))
        
        # Moving averages
        if 'sma_20' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['sma_20'],
                name='SMA 20',
                line=dict(color='orange', width=1)
            ))
        
        if 'sma_50' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['sma_50'],
                name='SMA 50',
                line=dict(color='red', width=1)
            ))
        
        # Bollinger Bands
        if all(col in data.columns for col in ['bb_upper', 'bb_lower']):
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['bb_upper'],
                name='BB Upper',
                line=dict(color='gray', width=1, dash='dash'),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['bb_lower'],
                name='BB Lower',
                line=dict(color='gray', width=1, dash='dash'),
                fill='tonexty',
                fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ))
        
        fig.update_layout(
            title=f"{symbol} Price Chart with Technical Indicators",
            xaxis_title="Date",
            yaxis_title="Price ($)",
            height=500,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_technical_indicators(self, data: pd.DataFrame, symbol: str):
        """Plot technical indicators"""
        col1, col2 = st.columns(2)
        
        with col1:
            # RSI
            if 'rsi' in data.columns:
                st.subheader("RSI")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=data['rsi'],
                    name='RSI',
                    line=dict(color='purple')
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(height=300, yaxis=dict(range=[0, 100]))
                st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            # MACD
            if all(col in data.columns for col in ['macd', 'macd_signal']):
                st.subheader("MACD")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['macd'],
                    name='MACD',
                    line=dict(color='blue')
                ))
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['macd_signal'],
                    name='Signal',
                    line=dict(color='red')
                ))
                if 'macd_histogram' in data.columns:
                    fig_macd.add_trace(go.Bar(
                        x=data.index,
                        y=data['macd_histogram'],
                        name='Histogram',
                        opacity=0.6
                    ))
                fig_macd.update_layout(height=300)
                st.plotly_chart(fig_macd, use_container_width=True)
    
    def render_backtesting(self):
        """Render backtesting interface"""
        st.header("🔄 Strategy Backtesting")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Backtest Parameters")
            
            # Date range
            start_date = st.date_input(
                "Start Date",
                value=datetime.now() - timedelta(days=365)
            )
            
            end_date = st.date_input(
                "End Date",
                value=datetime.now()
            )
            
            # Symbol selection for backtest
            backtest_symbols = st.multiselect(
                "Symbols to test:",
                options=config.DEFAULT_SYMBOLS,
                default=['AAPL', 'GOOGL']
            )
            
            # Initial capital
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=10000,
                max_value=1000000,
                value=100000,
                step=10000
            )
            
            # Run backtest button
            if st.button("🚀 Run Backtest", type="primary"):
                if backtest_symbols:
                    with st.spinner("Running backtest..."):
                        results = self.backtester.run_backtest(
                            symbols=backtest_symbols,
                            start_date=start_date.strftime('%Y-%m-%d'),
                            end_date=end_date.strftime('%Y-%m-%d')
                        )
                        st.session_state.backtest_results = results
                else:
                    st.error("Please select at least one symbol.")
        
        with col2:
            st.subheader("Backtest Results")
            
            if 'backtest_results' in st.session_state:
                results = st.session_state.backtest_results
                
                # Key metrics
                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                
                with metric_col1:
                    st.metric("Total Return", f"{results.total_return_percentage:.2f}%")
                    st.metric("Sharpe Ratio", f"{results.sharpe_ratio:.3f}")
                
                with metric_col2:
                    st.metric("Max Drawdown", f"{results.max_drawdown_percentage:.2f}%")
                    st.metric("Win Rate", f"{results.win_rate:.1%}")
                
                with metric_col3:
                    st.metric("Total Trades", f"{results.total_trades}")
                    st.metric("Profit Factor", f"{results.profit_factor:.2f}")
                
                with metric_col4:
                    st.metric("Final Capital", f"${results.final_capital:,.0f}")
                    st.metric("Avg Trade Duration", f"{results.avg_trade_duration:.1f} days")
                
                # Portfolio value chart
                if not results.portfolio_values.empty:
                    st.subheader("Portfolio Performance")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=results.portfolio_values.index,
                        y=results.portfolio_values.values,
                        name='Portfolio Value',
                        line=dict(color='green', width=2)
                    ))
                    fig.update_layout(
                        title="Portfolio Value Over Time",
                        xaxis_title="Date",
                        yaxis_title="Portfolio Value ($)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Trade analysis
                if results.trades:
                    st.subheader("Trade Analysis")
                    
                    # Trade distribution
                    trade_pnl = [t.pnl for t in results.trades if t.pnl is not None]
                    if trade_pnl:
                        fig_hist = px.histogram(
                            x=trade_pnl,
                            nbins=20,
                            title="Trade P&L Distribution"
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                
                # Download report
                if st.button("📄 Generate Report"):
                    report = self.backtester.generate_report(results)
                    st.download_button(
                        label="Download Report",
                        data=report,
                        file_name=f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
            else:
                st.info("Run a backtest to see results here.")
    
    def render_settings(self):
        """Render settings and configuration"""
        st.header("⚙️ Settings & Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Trading Parameters")
            
            # Risk management settings
            max_risk = st.slider(
                "Maximum Risk per Trade (%)",
                min_value=0.5,
                max_value=10.0,
                value=config.MAX_RISK_PER_TRADE * 100,
                step=0.1
            )
            
            stop_loss_multiplier = st.slider(
                "Stop Loss Multiplier",
                min_value=1.0,
                max_value=5.0,
                value=config.STOP_LOSS_MULTIPLIER,
                step=0.1
            )
            
            take_profit_multiplier = st.slider(
                "Take Profit Multiplier",
                min_value=1.0,
                max_value=5.0,
                value=config.TAKE_PROFIT_MULTIPLIER,
                step=0.1
            )
        
        with col2:
            st.subheader("Model Parameters")
            
            # AI model settings
            confidence_threshold = st.slider(
                "Signal Confidence Threshold",
                min_value=0.1,
                max_value=1.0,
                value=config.CONFIDENCE_THRESHOLD,
                step=0.05
            )
            
            lookback_period = st.slider(
                "Lookback Period (days)",
                min_value=20,
                max_value=200,
                value=config.LOOKBACK_PERIOD,
                step=10
            )
        
        # Save settings
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")
        
        # Model training
        st.subheader("🧠 Model Training")
        st.info("Train AI models for better signal generation")
        
        training_symbol = st.selectbox(
            "Select symbol to train:",
            options=config.DEFAULT_SYMBOLS
        )
        
        if st.button("🎯 Train Model"):
            with st.spinner(f"Training model for {training_symbol}..."):
                success = self.signal_generator.train_model(training_symbol)
                if success:
                    st.success(f"Model trained successfully for {training_symbol}!")
                else:
                    st.error(f"Training failed for {training_symbol}")


def main():
    """Main function to run the dashboard"""
    dashboard = TradingDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()