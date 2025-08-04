#!/usr/bin/env python3
"""
Database Setup Script for Financial Analysis Bot
Creates and initializes the database with sample data
"""

import os
import asyncio
import asyncpg
import logging
from datetime import datetime, timedelta
from decimal import Decimal
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
DATABASE_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME', 'financial_analysis'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', 'password')
}

async def create_database_connection():
    """Create database connection"""
    try:
        conn = await asyncpg.connect(**DATABASE_CONFIG)
        logger.info("Database connection established")
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

async def setup_sample_data(conn):
    """Insert sample data for testing"""
    logger.info("Setting up sample data...")
    
    # Sample assets
    assets_data = [
        ('AAPL', 'Apple Inc.', 'stock', 'NASDAQ', 'Technology'),
        ('MSFT', 'Microsoft Corporation', 'stock', 'NASDAQ', 'Technology'),
        ('GOOGL', 'Alphabet Inc.', 'stock', 'NASDAQ', 'Technology'),
        ('TSLA', 'Tesla Inc.', 'stock', 'NASDAQ', 'Automotive'),
        ('BTC', 'Bitcoin', 'crypto', 'Binance', 'Cryptocurrency'),
        ('ETH', 'Ethereum', 'crypto', 'Binance', 'Cryptocurrency'),
        ('ADA', 'Cardano', 'crypto', 'Binance', 'Cryptocurrency'),
    ]
    
    asset_ids = {}
    for symbol, name, asset_type, exchange, sector in assets_data:
        asset_id = str(uuid.uuid4())
        await conn.execute("""
            INSERT INTO assets (id, symbol, name, asset_type, exchange, sector, is_active)
            VALUES ($1, $2, $3, $4, $5, $6, true)
            ON CONFLICT (symbol, asset_type, exchange) DO NOTHING
        """, asset_id, symbol, name, asset_type, exchange, sector)
        asset_ids[symbol] = asset_id
    
    logger.info(f"Inserted {len(assets_data)} sample assets")
    
    # Sample user
    user_id = str(uuid.uuid4())
    await conn.execute("""
        INSERT INTO users (id, email, password_hash, first_name, last_name, role)
        VALUES ($1, 'demo@example.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewdBPj3bp.Gm.F5e', 'Demo', 'User', 'user')
        ON CONFLICT (email) DO NOTHING
    """, user_id, )
    
    # Sample portfolio
    portfolio_id = str(uuid.uuid4())
    await conn.execute("""
        INSERT INTO portfolios (id, user_id, name, description, is_default)
        VALUES ($1, $2, 'Demo Portfolio', 'Sample portfolio for testing', true)
    """, portfolio_id, user_id)
    
    # Sample market data
    base_time = datetime.now() - timedelta(days=30)
    for i, (symbol, _, _, _, _) in enumerate(assets_data):
        asset_id = asset_ids[symbol]
        
        # Generate sample price data
        base_price = 100 + (i * 50)  # Different base prices
        for day in range(30):
            timestamp = base_time + timedelta(days=day)
            
            # Simulate price movement
            price_change = (day % 7 - 3) * 0.02  # Simple oscillation
            current_price = base_price * (1 + price_change)
            
            await conn.execute("""
                INSERT INTO market_data (time, asset_id, open_price, high_price, low_price, close_price, volume, timeframe, source)
                VALUES ($1, $2, $3, $4, $5, $6, $7, '1d', 'sample')
            """, timestamp, asset_id, 
                Decimal(str(current_price * 0.99)), 
                Decimal(str(current_price * 1.02)), 
                Decimal(str(current_price * 0.98)), 
                Decimal(str(current_price)), 
                Decimal('1000000'))
    
    logger.info("Sample market data inserted")
    
    # Sample ML model
    model_id = str(uuid.uuid4())
    await conn.execute("""
        INSERT INTO ml_models (id, name, model_type, version, asset_type, target_variable, features, performance_metrics, is_active)
        VALUES ($1, 'LSTM Price Predictor', 'lstm', '1.0', 'stock', 'price', $2, $3, true)
    """, model_id, 
        json.dumps(['close_price', 'volume', 'rsi', 'macd']),
        json.dumps({'accuracy': 0.85, 'mse': 0.02}))
    
    logger.info("Sample data setup completed")

async def main():
    """Main setup function"""
    logger.info("Starting database setup...")
    
    conn = await create_database_connection()
    
    try:
        # Read and execute schema
        schema_path = os.path.join(os.path.dirname(__file__), '..', 'app', 'database', 'schema.sql')
        if os.path.exists(schema_path):
            with open(schema_path, 'r') as f:
                schema_sql = f.read()
            
            # Execute schema in parts (TimescaleDB functions need to be run separately)
            schema_parts = schema_sql.split('SELECT create_hypertable')
            
            # Execute main schema
            await conn.execute(schema_parts[0])
            logger.info("Base schema created")
            
            # Execute hypertable creations
            for i, part in enumerate(schema_parts[1:], 1):
                hypertable_sql = 'SELECT create_hypertable' + part.split(';')[0] + ';'
                try:
                    await conn.execute(hypertable_sql)
                    logger.info(f"Hypertable {i} created")
                except Exception as e:
                    logger.warning(f"Hypertable creation {i} failed: {e}")
        
        # Setup sample data
        await setup_sample_data(conn)
        
        logger.info("Database setup completed successfully!")
        
    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(main())
