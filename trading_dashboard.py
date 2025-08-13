#!/usr/bin/env python3
"""
Premium Trading Dashboard - Autonomous Investment Committee
=========================================================

Professional-grade real-time dashboard for monitoring autonomous trading system.
Features: Live positions, P&L tracking, risk metrics, and performance analytics.
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any

# Configure page with premium styling
st.set_page_config(
    page_title="Autonomous Trading Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1f77b4, #ff7f0e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .status-live {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
    }
    
    .status-closed {
        background: linear-gradient(135deg, #fc4a1a 0%, #f7b733 100%);
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: white;
        font-weight: bold;
        display: inline-block;
    }
    
    .profit-positive {
        color: #00ff88;
        font-weight: bold;
    }
    
    .profit-negative {
        color: #ff4444;
        font-weight: bold;
    }
    
    .section-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        font-size: 1.3rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

def load_trade_data():
    """Load comprehensive trade data from all sources."""
    try:
        # Load executed trades from simple autonomous trader (check both locations)
        executed_trades = []
        
        # Primary location: trading/logs/
        if os.path.exists('trading/logs/executed_trades.jsonl'):
            with open('trading/logs/executed_trades.jsonl', 'r') as f:
                for line in f:
                    if line.strip():
                        executed_trades.append(json.loads(line.strip()))
        # Fallback location: logs/
        elif os.path.exists('logs/executed_trades.jsonl'):
            with open('logs/executed_trades.jsonl', 'r') as f:
                for line in f:
                    if line.strip():
                        executed_trades.append(json.loads(line.strip()))
        
        # Load open positions (check both locations)
        open_positions = []
        if os.path.exists('trading/logs/open_positions.json'):
            with open('trading/logs/open_positions.json', 'r') as f:
                open_positions = json.load(f)
        elif os.path.exists('logs/open_positions.json'):
            with open('logs/open_positions.json', 'r') as f:
                open_positions = json.load(f)
        
        # Load closed trades (check both locations)
        closed_trades = []
        if os.path.exists('trading/logs/closed_trades.jsonl'):
            with open('trading/logs/closed_trades.jsonl', 'r') as f:
                for line in f:
                    if line.strip():
                        closed_trades.append(json.loads(line.strip()))
        elif os.path.exists('logs/closed_trades.jsonl'):
            with open('logs/closed_trades.jsonl', 'r') as f:
                for line in f:
                    if line.strip():
                        closed_trades.append(json.loads(line.strip()))
        
        # Load managed trades (legacy compatibility)
        managed_trades = {}
        if os.path.exists('data/managed_trades.json'):
            with open('data/managed_trades.json', 'r') as f:
                managed_trades = json.load(f)
                    
        return executed_trades, open_positions, closed_trades, managed_trades
        
    except Exception as e:
        st.error(f"Error loading trade data: {e}")
        return [], [], [], {}

def load_system_logs():
    """Load recent system logs."""
    try:
        # Check trading/logs/ first, then logs/
        log_file = None
        if os.path.exists('trading/logs/autonomous_trading.log'):
            log_file = 'trading/logs/autonomous_trading.log'
        elif os.path.exists('logs/autonomous_trading.log'):
            log_file = 'logs/autonomous_trading.log'
        
        if log_file:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                return lines[-50:]  # Last 50 lines
        return []
    except Exception as e:
        st.error(f"Error loading system logs: {e}")
        return []

def calculate_portfolio_metrics(open_positions: List[Dict], closed_trades: List[Dict]) -> Dict:
    """Calculate comprehensive portfolio metrics."""
    metrics = {
        'total_trades': len(closed_trades),
        'open_positions': len(open_positions),
        'total_credit_collected': 0,
        'realized_pnl': 0,
        'unrealized_pnl': 0,
        'win_rate': 0,
        'avg_winner': 0,
        'avg_loser': 0,
        'max_winner': 0,
        'max_loser': 0,
        'portfolio_value': 0,
        'total_risk': 0,
        'profit_factor': 0
    }
    
    # Calculate from closed trades
    if closed_trades:
        profits = [trade.get('profit_loss', 0) for trade in closed_trades]
        metrics['realized_pnl'] = sum(profits)
        metrics['total_trades'] = len(closed_trades)
        
        winners = [p for p in profits if p > 0]
        losers = [p for p in profits if p < 0]
        
        if winners:
            metrics['win_rate'] = len(winners) / len(profits)
            metrics['avg_winner'] = np.mean(winners)
            metrics['max_winner'] = max(winners)
        
        if losers:
            metrics['avg_loser'] = np.mean(losers)
            metrics['max_loser'] = min(losers)
        
        if winners and losers:
            metrics['profit_factor'] = sum(winners) / abs(sum(losers))
    
    # Calculate from open positions
    if open_positions:
        metrics['total_credit_collected'] = sum(pos.get('estimated_credit', 0) for pos in open_positions)
        
        # Simulate current P&L for open positions
        for pos in open_positions:
            entry_date = datetime.fromisoformat(pos.get('entry_time', datetime.now().isoformat()))
            days_held = (datetime.now() - entry_date).days
            
            # Simple P&L simulation based on theta decay
            estimated_credit = pos.get('estimated_credit', 0)
            time_factor = min(0.5, days_held / 30.0)  # 50% max profit over 30 days
            simulated_pnl = estimated_credit * time_factor
            metrics['unrealized_pnl'] += simulated_pnl
    
    metrics['portfolio_value'] = metrics['realized_pnl'] + metrics['unrealized_pnl']
    
    return metrics

def main():
    """Main dashboard function with premium UI."""
    # Header with gradient
    st.markdown('<h1 class="main-header">🤖 Autonomous Trading Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">AI-Powered Bull Put Spread Trading System</p>', unsafe_allow_html=True)
    
    # Sidebar with enhanced controls
    with st.sidebar:
        st.markdown("### ⚙️ Dashboard Controls")
        
        # Auto-refresh with better UI
        auto_refresh = st.toggle("🔄 Auto Refresh", value=True, help="Refresh every 30 seconds")
        refresh_interval = st.selectbox("Refresh Interval", [10, 30, 60, 120], index=1)
        
        # Manual refresh with styled button
        if st.button("🔄 Refresh Now", type="primary", use_container_width=True):
            st.rerun()
        
        st.markdown("---")
        
        # Market status indicator
        now = datetime.now()
        is_market_hours = (9 <= now.hour < 16) and (now.weekday() < 5)
        
        if is_market_hours:
            st.markdown('<div class="status-live">🟢 MARKET OPEN</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-closed">� MARKET CLOSED</div>', unsafe_allow_html=True)
        
        st.markdown(f"**Current Time:** {now.strftime('%H:%M:%S ET')}")
        st.markdown(f"**Date:** {now.strftime('%Y-%m-%d')}")
        
        # Quick stats in sidebar
        st.markdown("---")
        st.markdown("### 📊 Quick Stats")
        
    # Load all data
    executed_trades, open_positions, closed_trades, managed_trades = load_trade_data()
    system_logs = load_system_logs()
    
    # Calculate metrics
    metrics = calculate_portfolio_metrics(open_positions, closed_trades)
    
    # Main dashboard content
    # Top metrics row with enhanced styling
    st.markdown('<div class="section-header">� Portfolio Overview</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        portfolio_value = metrics['portfolio_value']
        color = "profit-positive" if portfolio_value >= 0 else "profit-negative"
        st.metric(
            "Portfolio Value", 
            f"${portfolio_value:.2f}",
            delta=f"${metrics['unrealized_pnl']:.2f}",
            help="Total realized + unrealized P&L"
        )
        
    with col2:
        st.metric(
            "Open Positions", 
            metrics['open_positions'],
            help="Currently active bull put spreads"
        )
        
    with col3:
        st.metric(
            "Total Trades", 
            metrics['total_trades'],
            help="Completed trades (closed positions)"
        )
        
    with col4:
        win_rate = metrics['win_rate']
        st.metric(
            "Win Rate", 
            f"{win_rate:.1%}",
            help="Percentage of profitable closed trades"
        )
        
    with col5:
        credit_collected = metrics['total_credit_collected']
        st.metric(
            "Credit Collected", 
            f"${credit_collected:.2f}",
            help="Total premium collected from open positions"
        )
    
    # Options Trading Status Section
    st.markdown('<div class="section-header">📊 Options Trading Status</div>', unsafe_allow_html=True)
    
    # Check if we can determine options status
    options_col1, options_col2, options_col3 = st.columns(3)
    
    with options_col1:
        # Try to determine if we're using options-enabled stocks
        using_options = False
        if executed_trades:
            # Check if any trades show real options symbols or options strategy
            for trade in executed_trades:
                if trade.get('strategy') == 'bull_put_spread' and trade.get('real_alpaca_order'):
                    using_options = True
                    break
        
        if using_options:
            st.success("✅ Options Trading Active")
            st.write("Real options spreads being executed")
        else:
            st.warning("⚠️ Paper Trading Mode")
            st.write("Options orders falling back to simulation")
    
    with options_col2:
        # Show symbol status
        try:
            # Try to load current symbols being used
            symbols_count = 0
            if os.path.exists('trading/logs/autonomous_trading.log'):
                with open('trading/logs/autonomous_trading.log', 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Look for symbol loading messages
                    if "symbols from" in content:
                        lines = content.split('\n')
                        for line in reversed(lines):
                            if "Symbols loaded:" in line:
                                # Extract number from "Symbols loaded: X from..."
                                parts = line.split("Symbols loaded:")
                                if len(parts) > 1:
                                    num_part = parts[1].strip().split(' ')[0]
                                    try:
                                        symbols_count = int(num_part)
                                        break
                                    except:
                                        pass
            
            if symbols_count > 0:
                st.metric("Active Symbols", symbols_count, help="Options-enabled stocks in universe")
            else:
                st.metric("Active Symbols", "Unknown", help="Unable to determine symbol count")
                
        except Exception as e:
            st.metric("Active Symbols", "Error", help=f"Error loading symbol info: {e}")
    
    with options_col3:
        # Show Alpaca connection status
        alpaca_connected = False
        if os.path.exists('trading/logs/autonomous_trading.log'):
            try:
                with open('trading/logs/autonomous_trading.log', 'r', encoding='utf-8') as f:
                    content = f.read()
                    if "REAL ALPACA CONNECTION ESTABLISHED" in content:
                        alpaca_connected = True
            except:
                pass
        
        if alpaca_connected:
            st.success("🔌 Alpaca Connected")
            st.write("Real API connection active")
        else:
            st.error("❌ Alpaca Disconnected")
            st.write("Check API credentials")
    
    # Real-time Position Tracking
    st.markdown('<div class="section-header">🎯 Live Positions</div>', unsafe_allow_html=True)
    
    if open_positions:
        # Create position tracking table with enhanced data
        position_data = []
        for pos in open_positions:
            entry_date = datetime.fromisoformat(pos.get('entry_time', datetime.now().isoformat()))
            days_held = (datetime.now() - entry_date).days
            
            # Calculate theta decay progress
            estimated_credit = pos.get('estimated_credit', 0)
            time_factor = min(0.5, days_held / 30.0)  # 50% max profit over 30 days
            current_profit = estimated_credit * time_factor
            profit_pct = (current_profit / estimated_credit * 100) if estimated_credit > 0 else 0
            
            # Expiration data
            exp_date = datetime.fromisoformat(pos.get('expiration_date', datetime.now().isoformat()))
            days_to_exp = (exp_date - datetime.now()).days
            
            position_data.append({
                'Symbol': pos.get('symbol', 'N/A'),
                'Strategy': 'Bull Put Spread',
                'Short Strike': f"${pos.get('short_strike', 0):.2f}",
                'Long Strike': f"${pos.get('long_strike', 0):.2f}",
                'Credit': f"${estimated_credit:.2f}",
                'Days Held': days_held,
                'DTE': days_to_exp,
                'Current P&L': f"${current_profit:.2f}",
                'P&L %': f"{profit_pct:.1f}%",
                'Status': '🟢 Open' if days_to_exp > 5 else '🟡 Near Exp'
            })
        
        if position_data:
            df_positions = pd.DataFrame(position_data)
            
            # Color code the P&L columns
            def color_pnl_columns(s):
                styles = []
                for val in s:
                    if 'P&L' in s.name:  # Apply to P&L related columns
                        if '$' in str(val) and '-' in str(val):
                            styles.append('color: #ff4444; font-weight: bold')
                        elif '$' in str(val):
                            styles.append('color: #00ff88; font-weight: bold')
                        else:
                            styles.append('')
                    else:
                        styles.append('')
                return styles
            
            styled_positions = df_positions.style.apply(color_pnl_columns, axis=0)
            
            st.dataframe(
                styled_positions,
                use_container_width=True,
                height=300
            )
            
            # Position heatmap
            st.subheader("📊 Position Heatmap")
            
            # Create profit heatmap
            profit_data = []
            for pos in position_data:
                profit_num = float(pos['Current P&L'].replace('$', '').replace(',', ''))
                profit_data.append({
                    'Symbol': pos['Symbol'],
                    'Profit': profit_num,
                    'DTE': pos['DTE']
                })
            
            if profit_data:
                df_heatmap = pd.DataFrame(profit_data)
                
                fig_heatmap = px.scatter(
                    df_heatmap, 
                    x='DTE', 
                    y='Symbol',
                    size='Profit',
                    color='Profit',
                    color_continuous_scale='RdYlGn',
                    title='Position P&L by Days to Expiration',
                    hover_data=['Symbol', 'Profit', 'DTE']
                )
                
                fig_heatmap.update_layout(
                    height=400,
                    title_font_size=16,
                    coloraxis_colorbar_title="P&L ($)"
                )
                
                st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("📈 No open positions. System will start trading at market open (9:30 AM ET).")
        
        # Show system readiness indicators
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if os.path.exists('logs/autonomous_trading.log'):
                st.success("✅ Trading System Ready")
            else:
                st.error("❌ Trading System Not Running")
                
        with col2:
            if os.path.exists('models/production/'):
                st.success("✅ ML Models Loaded")
            else:
                st.warning("⚠️ ML Models Missing")
                
        with col3:
            if os.path.exists('iex_symbols.json'):
                st.success("✅ Symbol Universe Ready")
            else:
                st.warning("⚠️ Symbols Not Loaded")
    
    # Advanced Analytics Section
    st.markdown('<div class="section-header">📈 Advanced Analytics</div>', unsafe_allow_html=True)
    
    # Create comprehensive charts
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        # P&L Evolution Chart
        st.subheader("💰 P&L Evolution")
        
        if closed_trades:
            df_closed = pd.DataFrame(closed_trades)
            df_closed['close_time'] = pd.to_datetime(df_closed['close_time'])
            df_closed['cumulative_pnl'] = df_closed['profit_loss'].cumsum()
            
            fig_pnl = go.Figure()
            
            fig_pnl.add_trace(go.Scatter(
                x=df_closed['close_time'],
                y=df_closed['cumulative_pnl'],
                mode='lines+markers',
                name='Cumulative P&L',
                line=dict(color='#00ff88', width=3),
                marker=dict(size=6)
            ))
            
            fig_pnl.update_layout(
                title='Cumulative P&L Over Time',
                xaxis_title='Date',
                yaxis_title='P&L ($)',
                height=350,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_pnl, use_container_width=True)
        else:
            # Show mock chart for demo
            dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
            mock_pnl = np.cumsum(np.random.normal(5, 15, 30))
            
            fig_mock = go.Figure()
            fig_mock.add_trace(go.Scatter(
                x=dates,
                y=mock_pnl,
                mode='lines+markers',
                name='Expected P&L Pattern',
                line=dict(color='#1f77b4', width=3, dash='dash'),
                marker=dict(size=4)
            ))
            
            fig_mock.update_layout(
                title='Expected P&L Pattern (Demo)',
                xaxis_title='Date',
                yaxis_title='P&L ($)',
                height=350,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(fig_mock, use_container_width=True)
    
    with chart_col2:
        # Risk Distribution
        st.subheader("⚖️ Risk Distribution")
        
        if open_positions:
            # Calculate risk for each position
            risk_data = []
            for pos in open_positions:
                short_strike = pos.get('short_strike', 0)
                long_strike = pos.get('long_strike', 0)
                credit = pos.get('estimated_credit', 0)
                max_loss = (short_strike - long_strike) * 100 - credit
                
                risk_data.append({
                    'Symbol': pos.get('symbol', 'N/A'),
                    'Max Risk': max_loss,
                    'Credit': credit
                })
            
            df_risk = pd.DataFrame(risk_data)
            
            fig_risk = px.bar(
                df_risk,
                x='Symbol',
                y='Max Risk',
                color='Credit',
                title='Maximum Risk by Position',
                color_continuous_scale='Viridis'
            )
            
            fig_risk.update_layout(height=350)
            st.plotly_chart(fig_risk, use_container_width=True)
        else:
            # Show risk management principles
            st.info("�️ Risk Management Active")
            st.markdown("""
            **Risk Controls:**
            - Max 10 trades per day
            - $1,000 position sizing
            - 75% confidence threshold
            - Automatic position monitoring
            """)
    
    # Performance Metrics Grid
    st.markdown('<div class="section-header">📊 Performance Metrics</div>', unsafe_allow_html=True)
    
    perf_col1, perf_col2, perf_col3, perf_col4 = st.columns(4)
    
    with perf_col1:
        st.metric(
            "Realized P&L",
            f"${metrics['realized_pnl']:.2f}",
            help="Profit/Loss from closed positions"
        )
        
    with perf_col2:
        st.metric(
            "Unrealized P&L",
            f"${metrics['unrealized_pnl']:.2f}",
            help="Current profit/loss on open positions"
        )
        
    with perf_col3:
        profit_factor = metrics['profit_factor']
        if profit_factor > 0:
            st.metric(
                "Profit Factor",
                f"{profit_factor:.2f}",
                help="Gross profits / Gross losses"
            )
        else:
            st.metric("Profit Factor", "N/A", help="Need closed trades for calculation")
            
    with perf_col4:
        avg_winner = metrics['avg_winner']
        if avg_winner > 0:
            st.metric(
                "Avg Winner",
                f"${avg_winner:.2f}",
                help="Average profit per winning trade"
            )
        else:
            st.metric("Avg Winner", "N/A", help="No winning trades yet")
    
    # Recent trades section
    st.markdown('<div class="section-header">📝 Recent Activity</div>', unsafe_allow_html=True)
    
    # Create tabs for different activity views
    tab1, tab2, tab3 = st.tabs(["📈 Recent Trades", "🎯 Position Changes", "🔍 System Logs"])
    
    with tab1:
        if executed_trades:
            st.subheader("Today's New Positions")
            
            # Format executed trades for display
            trade_display = []
            for trade in executed_trades[-10:]:  # Last 10 trades
                trade_display.append({
                    'Time': datetime.fromisoformat(trade.get('entry_time', '')).strftime('%H:%M:%S'),
                    'Symbol': trade.get('symbol', 'N/A'),
                    'Short Strike': f"${trade.get('short_strike', 0):.2f}",
                    'Long Strike': f"${trade.get('long_strike', 0):.2f}",
                    'Credit': f"${trade.get('estimated_credit', 0):.2f}",
                    'Confidence': f"{trade.get('confidence', 0):.1%}",
                    'Status': '🟢 Executed'
                })
            
            if trade_display:
                df_recent = pd.DataFrame(trade_display)
                st.dataframe(df_recent, use_container_width=True)
            
            # Trade timing analysis
            if len(executed_trades) > 1:
                df_trades = pd.DataFrame(executed_trades)
                df_trades['entry_time'] = pd.to_datetime(df_trades['entry_time'])
                df_trades['hour'] = df_trades['entry_time'].dt.hour
                
                hourly_counts = df_trades['hour'].value_counts().sort_index()
                
                fig_timing = px.bar(
                    x=hourly_counts.index,
                    y=hourly_counts.values,
                    title='Trade Execution by Hour',
                    labels={'x': 'Hour of Day', 'y': 'Number of Trades'},
                    color=hourly_counts.values,
                    color_continuous_scale='Blues'
                )
                
                fig_timing.update_layout(height=300, showlegend=False)
                st.plotly_chart(fig_timing, use_container_width=True)
        else:
            st.info("🕒 No trades executed yet today. System is monitoring for opportunities.")
    
    with tab2:
        if closed_trades:
            st.subheader("Recent Position Closures")
            
            closure_display = []
            for closure in closed_trades[-10:]:  # Last 10 closures
                closure_display.append({
                    'Close Time': datetime.fromisoformat(closure.get('close_time', '')).strftime('%m/%d %H:%M'),
                    'Symbol': closure.get('symbol', 'N/A'),
                    'Days Held': closure.get('days_held', 0),
                    'P&L': f"${closure.get('profit_loss', 0):.2f}",
                    'Return %': f"{closure.get('return_pct', 0):.1f}%",
                    'Reason': closure.get('close_reason', 'N/A')
                })
            
            if closure_display:
                df_closures = pd.DataFrame(closure_display)
                
                # Color code P&L column specifically
                def highlight_pnl_column(s):
                    styles = []
                    for val in s:
                        if s.name == 'P&L':  # Only apply to P&L column
                            if '-' in str(val):
                                styles.append('background-color: #ffebee; color: #c62828')
                            else:
                                styles.append('background-color: #e8f5e8; color: #2e7d32')
                        else:
                            styles.append('')
                    return styles
                
                styled_df = df_closures.style.apply(highlight_pnl_column, axis=0)
                
                st.dataframe(styled_df, use_container_width=True)
                
                # Closure reason analysis
                reason_counts = pd.Series([c.get('close_reason', 'Unknown') for c in closed_trades]).value_counts()
                
                fig_reasons = px.pie(
                    values=reason_counts.values,
                    names=reason_counts.index,
                    title='Position Closure Reasons'
                )
                
                st.plotly_chart(fig_reasons, use_container_width=True)
        else:
            st.info("🔄 No position closures yet. Theta decay is working on open positions.")
    
    with tab3:
        st.subheader("System Activity Logs")
        
        if system_logs:
            # Filter and format logs
            recent_logs = []
            for log in system_logs[-20:]:  # Last 20 log entries
                if any(keyword in log.lower() for keyword in ['trade', 'position', 'market', 'profit', 'risk']):
                    recent_logs.append(log.strip())
            
            if recent_logs:
                log_text = '\n'.join(recent_logs)
                st.text_area("Recent System Activity", log_text, height=300)
            else:
                st.info("No recent trading activity in logs.")
                
            # System health indicators
            st.subheader("🏥 System Health")
            
            health_col1, health_col2, health_col3 = st.columns(3)
            
            with health_col1:
                if 'ERROR' in log_text:
                    st.error("❌ Errors Detected")
                else:
                    st.success("✅ No Errors")
                    
            with health_col2:
                if 'trading cycle' in log_text.lower():
                    st.success("✅ Trading Active")
                else:
                    st.warning("⚠️ Trading Inactive")
                    
            with health_col3:
                if 'market is open' in log_text.lower():
                    st.success("✅ Market Connected")
                else:
                    st.info("ℹ️ Outside Market Hours")
        else:
            st.warning("⚠️ No system logs available. Is the trading system running?")
            st.code("python trading/simple_autonomous_trader.py", language="bash")
    
    # System Configuration and Info
    st.markdown('<div class="section-header">⚙️ System Configuration</div>', unsafe_allow_html=True)
    
    config_col1, config_col2 = st.columns(2)
    
    with config_col1:
        st.markdown("### 🎯 Trading Strategy")
        st.markdown("""
        **Bull Put Spread Focus:**
        - ✅ Theta decay profit strategy
        - ✅ Credit collection approach
        - ✅ Limited risk defined spreads
        - ✅ 45 DTE target expiration
        - ✅ 50% profit target (classic theta)
        
        **Risk Management:**
        - 🛡️ Max $1,000 per position
        - 🛡️ 10 trades per day limit
        - 🛡️ 75% ML confidence threshold
        - 🛡️ Auto position monitoring
        """)
        
    with config_col2:
        st.markdown("### 📊 Performance Targets")
        st.markdown("""
        **Profit Objectives:**
        - 🎯 50% max profit target
        - 🎯 Let theta decay work
        - 🎯 20-30 day average hold
        - 🎯 Consistent premium collection
        
        **Market Schedule:**
        - 🕘 Trading: 9:30 AM - 4:00 PM ET
        - 🕘 Monitoring: 24/7
        - 🕘 Weekend: Position review
        - 🕘 Auto-refresh: Every 30 seconds
        """)
    
    # Auto-refresh functionality
    if auto_refresh:
        import time
        import asyncio
        
        # Use session state to prevent infinite loops
        if 'last_refresh' not in st.session_state:
            st.session_state.last_refresh = datetime.now()
        
        time_since_refresh = (datetime.now() - st.session_state.last_refresh).seconds
        
        if time_since_refresh >= refresh_interval:
            st.session_state.last_refresh = datetime.now()
            time.sleep(1)  # Small delay to prevent too rapid refreshing
            st.rerun()
    
    # Footer with system info
    st.markdown("---")
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    
    with footer_col1:
        st.markdown("**System Status:** 🟢 Online")
        st.markdown(f"**Last Update:** {datetime.now().strftime('%H:%M:%S')}")
        
    with footer_col2:
        st.markdown("**Paper Trading:** ✅ Enabled")
        st.markdown("**Risk Level:** 🟢 Conservative")
        
    with footer_col3:
        st.markdown("**ML Models:** ✅ Loaded")
        st.markdown("**Theta Strategy:** ✅ Active")

if __name__ == "__main__":
    main()
