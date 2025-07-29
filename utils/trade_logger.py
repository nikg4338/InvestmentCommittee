# Trade logging system for Investment Committee
# Supports CSV and SQLite logging for trade decisions and executions

import logging
import sqlite3
import csv
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class TradeLog:
    """Trade log entry."""
    timestamp: str
    trade_id: str
    symbol: str
    signal: str
    confidence: float
    reasoning: str
    execution_status: str
    order_details: str
    meta_decision: str
    performance_metrics: Optional[str] = None
    llm_analysis: Optional[str] = None


class TradeLogger:
    """
    Comprehensive trade logging system.
    Supports both CSV and SQLite database logging.
    """
    
    def __init__(self, log_dir: str = "logs", use_database: bool = True, use_csv: bool = True):
        """
        Initialize trade logger.
        
        Args:
            log_dir (str): Directory for log files
            use_database (bool): Whether to use SQLite database
            use_csv (bool): Whether to use CSV logging
        """
        self.log_dir = log_dir
        self.use_database = use_database
        self.use_csv = use_csv
        
        # Create log directory
        os.makedirs(log_dir, exist_ok=True)
        
        # Database setup
        if use_database:
            self.db_path = os.path.join(log_dir, "trades.db")
            self._setup_database()
        
        # CSV setup
        if use_csv:
            self.csv_path = os.path.join(log_dir, "trades.csv")
            self._setup_csv()
        
        logger.info(f"Trade logger initialized (db={use_database}, csv={use_csv})")
    
    def _setup_database(self):
        """Setup SQLite database schema."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create trades table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    trade_id TEXT UNIQUE NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reasoning TEXT,
                    execution_status TEXT NOT NULL,
                    order_details TEXT,
                    meta_decision TEXT,
                    performance_metrics TEXT,
                    llm_analysis TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create performance table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS performance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    entry_time TEXT,
                    exit_time TEXT,
                    entry_price REAL,
                    exit_price REAL,
                    pnl REAL,
                    pnl_percent REAL,
                    max_drawdown REAL,
                    hold_days INTEGER,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades (trade_id)
                )
            ''')
            
            # Create model predictions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS model_predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trade_id TEXT NOT NULL,
                    model_name TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    metadata TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (trade_id) REFERENCES trades (trade_id)
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info("Database schema created successfully")
            
        except Exception as e:
            logger.error(f"Error setting up database: {e}")
    
    def _setup_csv(self):
        """Setup CSV file headers."""
        try:
            # Check if CSV file exists
            if not os.path.exists(self.csv_path):
                headers = [
                    'timestamp', 'trade_id', 'symbol', 'signal', 'confidence',
                    'reasoning', 'execution_status', 'order_details', 'meta_decision',
                    'performance_metrics', 'llm_analysis'
                ]
                
                with open(self.csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(headers)
                
                logger.info("CSV file created with headers")
            
        except Exception as e:
            logger.error(f"Error setting up CSV: {e}")
    
    def log_trade(self, trade_execution: Any):
        """
        Log a trade execution.
        
        Args:
            trade_execution: Trade execution object with required fields
        """
        try:
            # Convert to TradeLog format
            trade_log = TradeLog(
                timestamp=datetime.now().isoformat(),
                trade_id=trade_execution.trade_id,
                symbol=trade_execution.symbol,
                signal=trade_execution.signal,
                confidence=trade_execution.confidence,
                reasoning=json.dumps(trade_execution.reasoning),
                execution_status=trade_execution.execution_status,
                order_details=json.dumps(trade_execution.order_details),
                meta_decision=json.dumps(trade_execution.meta_decision),
                llm_analysis=json.dumps(trade_execution.llm_analysis) if hasattr(trade_execution, 'llm_analysis') and trade_execution.llm_analysis else None
            )
            
            # Log to database
            if self.use_database:
                self._log_to_database(trade_log)
            
            # Log to CSV
            if self.use_csv:
                self._log_to_csv(trade_log)
            
            logger.info(f"Trade logged: {trade_log.trade_id}")
            
        except Exception as e:
            logger.error(f"Error logging trade: {e}")
    
    def _log_to_database(self, trade_log: TradeLog):
        """Log trade to SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO trades (
                    timestamp, trade_id, symbol, signal, confidence, reasoning,
                    execution_status, order_details, meta_decision, performance_metrics, llm_analysis
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                trade_log.timestamp,
                trade_log.trade_id,
                trade_log.symbol,
                trade_log.signal,
                trade_log.confidence,
                trade_log.reasoning,
                trade_log.execution_status,
                trade_log.order_details,
                trade_log.meta_decision,
                trade_log.performance_metrics,
                trade_log.llm_analysis
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging to database: {e}")
    
    def _log_to_csv(self, trade_log: TradeLog):
        """Log trade to CSV file."""
        try:
            with open(self.csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    trade_log.timestamp,
                    trade_log.trade_id,
                    trade_log.symbol,
                    trade_log.signal,
                    trade_log.confidence,
                    trade_log.reasoning,
                    trade_log.execution_status,
                    trade_log.order_details,
                    trade_log.meta_decision,
                    trade_log.performance_metrics,
                    trade_log.llm_analysis
                ])
            
        except Exception as e:
            logger.error(f"Error logging to CSV: {e}")
    
    def log_model_prediction(self, trade_id: str, model_name: str, direction: str, 
                           confidence: float, metadata: Dict[str, Any]):
        """
        Log individual model prediction.
        
        Args:
            trade_id (str): Trade ID
            model_name (str): Model name
            direction (str): Prediction direction
            confidence (float): Confidence score
            metadata (Dict[str, Any]): Model metadata
        """
        if not self.use_database:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO model_predictions (
                    trade_id, model_name, direction, confidence, metadata
                ) VALUES (?, ?, ?, ?, ?)
            ''', (
                trade_id,
                model_name,
                direction,
                confidence,
                json.dumps(metadata)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Error logging model prediction: {e}")
    
    def get_trades(self, symbol: Optional[str] = None, 
                  start_date: Optional[str] = None, 
                  end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get trade history.
        
        Args:
            symbol (str, optional): Filter by symbol
            start_date (str, optional): Start date (YYYY-MM-DD)
            end_date (str, optional): End date (YYYY-MM-DD)
            
        Returns:
            List[Dict[str, Any]]: Trade history
        """
        if self.use_database:
            return self._get_trades_from_database(symbol, start_date, end_date)
        elif self.use_csv:
            return self._get_trades_from_csv(symbol, start_date, end_date)
        else:
            return []
    
    def _get_trades_from_database(self, symbol: Optional[str] = None,
                                start_date: Optional[str] = None,
                                end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trades from SQLite database."""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if symbol:
                query += " AND symbol = ?"
                params.append(symbol)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(start_date)
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(end_date)
            
            query += " ORDER BY timestamp DESC"
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            columns = [description[0] for description in cursor.description]
            rows = cursor.fetchall()
            
            trades = [dict(zip(columns, row)) for row in rows]
            
            conn.close()
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trades from database: {e}")
            return []
    
    def _get_trades_from_csv(self, symbol: Optional[str] = None,
                           start_date: Optional[str] = None,
                           end_date: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get trades from CSV file."""
        try:
            if not os.path.exists(self.csv_path):
                return []
            
            # Read CSV manually to avoid pandas type checking issues
            trades = []
            with open(self.csv_path, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    # Apply filters
                    if symbol and row.get('symbol') != symbol:
                        continue
                    if start_date and row.get('timestamp', '') < start_date:
                        continue
                    if end_date and row.get('timestamp', '') > end_date:
                        continue
                    trades.append(row)
            
            return trades
            
        except Exception as e:
            logger.error(f"Error getting trades from CSV: {e}")
            return []
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics.
        
        Returns:
            Dict[str, Any]: Performance statistics
        """
        try:
            trades = self.get_trades()
            
            if not trades:
                return {'total_trades': 0}
            
            total_trades = len(trades)
            buy_signals = len([t for t in trades if t['signal'] == 'BUY'])
            successful_trades = len([t for t in trades if t['execution_status'] == 'FILLED'])
            
            avg_confidence = sum(float(t['confidence']) for t in trades) / total_trades
            
            # Symbol distribution
            symbols = {}
            for trade in trades:
                symbol = trade['symbol']
                symbols[symbol] = symbols.get(symbol, 0) + 1
            
            return {
                'total_trades': total_trades,
                'buy_signals': buy_signals,
                'pass_signals': total_trades - buy_signals,
                'successful_trades': successful_trades,
                'success_rate': successful_trades / total_trades if total_trades > 0 else 0,
                'average_confidence': avg_confidence,
                'symbol_distribution': symbols,
                'most_traded_symbol': max(symbols.items(), key=lambda x: x[1])[0] if symbols else None
            }
            
        except Exception as e:
            logger.error(f"Error getting performance stats: {e}")
            return {'error': str(e)}
    
    def export_to_excel(self, filename: str):
        """
        Export trades to Excel file.
        
        Args:
            filename (str): Output filename
        """
        try:
            trades = self.get_trades()
            
            if not trades:
                logger.warning("No trades to export")
                return
            
            df = pd.DataFrame(trades)
            
            # Parse JSON columns for better readability
            for col in ['reasoning', 'order_details', 'meta_decision']:
                if col in df.columns:
                    df[col] = df[col].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
            
            # Export to Excel
            excel_path = os.path.join(self.log_dir, filename)
            df.to_excel(excel_path, index=False)
            
            logger.info(f"Trades exported to {excel_path}")
            
        except Exception as e:
            logger.error(f"Error exporting to Excel: {e}")


# Global trade logger instance
trade_logger = TradeLogger()


def log_trade(trade_execution: Any):
    """
    Convenience function to log a trade.
    
    Args:
        trade_execution: Trade execution object
    """
    trade_logger.log_trade(trade_execution)


def get_trade_performance() -> Dict[str, Any]:
    """
    Get trade performance statistics.
    
    Returns:
        Dict[str, Any]: Performance statistics
    """
    return trade_logger.get_performance_stats()


def test_trade_logger():
    """Test the trade logger."""
    print("Testing Trade Logger...")
    
    # Create mock trade execution
    from dataclasses import dataclass
    from typing import List
    
    @dataclass
    class MockTradeExecution:
        trade_id: str
        symbol: str
        signal: str
        confidence: float
        reasoning: List[str]
        execution_status: str
        order_details: Dict[str, Any]
        meta_decision: Dict[str, Any]
    
    # Create sample trade
    sample_trade = MockTradeExecution(
        trade_id="TEST_AAPL_20240120_143000",
        symbol="AAPL",
        signal="BUY",
        confidence=0.75,
        reasoning=["Strong bullish consensus", "High confidence"],
        execution_status="FILLED",
        order_details={"contracts": 5, "credit_received": 250},
        meta_decision={"weighted_score": 0.8, "agreement": 0.9}
    )
    
    # Test logging
    test_logger = TradeLogger(log_dir="test_logs")
    test_logger.log_trade(sample_trade)
    
    # Test retrieval
    trades = test_logger.get_trades()
    print(f"Retrieved {len(trades)} trades")
    
    # Test performance stats
    stats = test_logger.get_performance_stats()
    print(f"Performance stats: {stats}")
    
    print("Trade logger test completed!")


if __name__ == "__main__":
    test_trade_logger() 