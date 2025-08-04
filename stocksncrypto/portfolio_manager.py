import sqlite3
import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass

@dataclass
class Position:
    """Position tracking"""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    asset_type: str
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price
    
    @property
    def cost_basis(self) -> float:
        return self.quantity * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_percent(self) -> float:
        if self.cost_basis == 0:
            return 0
        return (self.unrealized_pnl / self.cost_basis) * 100

@dataclass
class PortfolioMetrics:
    """Portfolio metrics"""
    total_value: float
    total_cost: float
    total_pnl: float
    total_pnl_percent: float
    daily_return: float
    positions: List[Position]

class PortfolioTracker:
    """Portfolio tracking"""
    
    def __init__(self, db_path: str = "portfolio.db"):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        self._init_database()
    
    def _init_database(self):
        """Initialize database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    symbol TEXT NOT NULL,
                    asset_type TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity REAL NOT NULL,
                    price REAL NOT NULL,
                    fees REAL DEFAULT 0
                )
            ''')
            conn.commit()
    
    def add_transaction(self, symbol: str, asset_type: str, action: str, 
                       quantity: float, price: float, fees: float = 0):
        """Add transaction"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO transactions (symbol, asset_type, action, quantity, price, fees)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (symbol, asset_type, action.lower(), quantity, price, fees))
            conn.commit()
        self.logger.info(f"Transaction added: {action} {quantity} {symbol} at ${price}")
    
    def get_current_positions(self, current_prices: Dict[str, float]) -> List[Position]:
        """Get current positions"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT symbol, asset_type, action, quantity, price, fees
                FROM transactions
                ORDER BY timestamp
            ''', conn)
        
        if df.empty:
            return []
        
        positions = {}
        
        for _, row in df.iterrows():
            symbol = row['symbol']
            if symbol not in positions:
                positions[symbol] = {
                    'quantity': 0,
                    'total_cost': 0,
                    'asset_type': row['asset_type']
                }
            
            if row['action'] == 'buy':
                positions[symbol]['quantity'] += row['quantity']
                positions[symbol]['total_cost'] += (row['quantity'] * row['price']) + row['fees']
            elif row['action'] == 'sell':
                if positions[symbol]['quantity'] > 0:
                    avg_cost = positions[symbol]['total_cost'] / positions[symbol]['quantity']
                    sold_cost = row['quantity'] * avg_cost
                    positions[symbol]['quantity'] -= row['quantity']
                    positions[symbol]['total_cost'] -= sold_cost
        
        result = []
        for symbol, data in positions.items():
            if data['quantity'] > 0:
                avg_cost = data['total_cost'] / data['quantity']
                current_price = current_prices.get(symbol, 0)
                
                position = Position(
                    symbol=symbol,
                    quantity=data['quantity'],
                    avg_cost=avg_cost,
                    current_price=current_price,
                    asset_type=data['asset_type']
                )
                result.append(position)
        
        return result
    
    def calculate_portfolio_metrics(self, positions: List[Position]) -> PortfolioMetrics:
        """Calculate portfolio metrics"""
        if not positions:
            return PortfolioMetrics(0, 0, 0, 0, 0, [])
        
        total_value = sum(pos.market_value for pos in positions)
        total_cost = sum(pos.cost_basis for pos in positions)
        total_pnl = total_value - total_cost
        total_pnl_percent = (total_pnl / total_cost * 100) if total_cost > 0 else 0
        
        return PortfolioMetrics(
            total_value=total_value,
            total_cost=total_cost,
            total_pnl=total_pnl,
            total_pnl_percent=total_pnl_percent,
            daily_return=0,  # Simplified for now
            positions=positions
        )
    
    def export_to_csv(self, filename: str = None):
        """Export portfolio data to CSV"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"portfolio_export_{timestamp}.csv"
        
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query('''
                SELECT * FROM transactions
                ORDER BY timestamp DESC
            ''', conn)
        
        df.to_csv(filename, index=False)
        self.logger.info(f"Portfolio exported to {filename}")
