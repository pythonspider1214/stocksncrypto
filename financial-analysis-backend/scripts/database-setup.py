#!/usr/bin/env python3
"""
Database Setup Script for Financial Analysis Bot
Creates and initializes the database with sample data
"""

import os
import asyncio
import asyncpg
import logging
from typing import Dict, Any
from datetime import datetime
from decimal import Decimal
import json
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSetup:
    """Database setup and initialization"""
    
    def __init__(self):
        self.db_config = {
            'host': os.getenv('DATABASE_HOST', 'localhost'),
            'port': int(os.getenv('DATABASE_PORT', 5432)),
            'database': os.getenv('DATABASE_NAME', 'financial_analysis'),
            'user': os.getenv('DATABASE_USER', 'postgres'),
            'password': os.getenv('DATABASE_PASSWORD', 'password')
        }
    
    async def create_connection(self) -> asyncpg.Connection:
        """Create database connection"""
        try:
            conn = await asyncpg.connect(**self.db_config)
            logger.info("Database connection established")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def setup_database(self):
        """Setup complete database schema"""
        conn = await self.create_connection()
        
        try:
            # Enable TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
            logger.info("TimescaleDB extension enabled")
            
            # Create core tables
            await self.create_core_tables(conn)
            
            # Create time-series tables
            await self.create_timeseries_tables(conn)
            
            # Create portfolio tables
            await self.create_portfolio_tables(conn)
            
            # Create ML and analytics tables
            await self.create_ml_tables(conn)
            
            # Create news and sentiment tables
            await self.create_news_tables(conn)
            
            # Create monitoring tables
            await self.create_monitoring_tables(conn)
            
            # Create indexes
            await self.create_indexes(conn)
            
            # Create views
            await self.create_views(conn)
            
            # Setup retention policies
            await self.setup_retention_policies(conn)
            
            # Insert sample data
            await self.insert_sample_data(conn)
            
            logger.info("Database setup completed successfully")
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            raise
        finally:
            await conn.close()
    
    async def create_core_tables(self, conn: asyncpg.Connection):
        """Create core application tables"""
        
        # Users table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                email VARCHAR(255) UNIQUE NOT NULL,
                password_hash VARCHAR(255) NOT NULL,
                first_name VARCHAR(100),
                last_name VARCHAR(100),
                role VARCHAR(50) DEFAULT 'user',
                is_active BOOLEAN DEFAULT true,
                email_verified BOOLEAN DEFAULT false,
                phone VARCHAR(20),
                timezone VARCHAR(50) DEFAULT 'UTC',
                preferences JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                last_login TIMESTAMP WITH TIME ZONE
            );
        """)
        
        # API Keys table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_keys (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID REFERENCES users(id) ON DELETE CASCADE,
                provider VARCHAR(100) NOT NULL,
                key_name VARCHAR(100) NOT NULL,
                encrypted_key TEXT NOT NULL,
                is_active BOOLEAN DEFAULT true,
                rate_limit_per_minute INTEGER DEFAULT 60,
                usage_count INTEGER DEFAULT 0,
                last_used TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE
            );
        """)
        
        # Assets table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS assets (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                symbol VARCHAR(20) NOT NULL,
                name VARCHAR(255) NOT NULL,
                asset_type VARCHAR(50) NOT NULL,
                exchange VARCHAR(100),
                sector VARCHAR(100),
                industry VARCHAR(100),
                market_cap BIGINT,
                description TEXT,
                website VARCHAR(255),
                logo_url VARCHAR(255),
                is_active BOOLEAN DEFAULT true,
                metadata JSONB DEFAULT '{}',
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(symbol, asset_type, exchange)
            );
        """)
        
        logger.info("Core tables created")
    
    async def create_timeseries_tables(self, conn: asyncpg.Connection):
        """Create time-series tables with TimescaleDB"""
        
        # Market Data table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                asset_id UUID NOT NULL REFERENCES assets(id),
                open_price DECIMAL(20,8) NOT NULL,
                high_price DECIMAL(20,8) NOT NULL,
                low_price DECIMAL(20,8) NOT NULL,
                close_price DECIMAL(20,8) NOT NULL,
                volume DECIMAL(20,8) NOT NULL,
                volume_usd DECIMAL(20,2),
                timeframe VARCHAR(10) NOT NULL,
                source VARCHAR(50) NOT NULL,
                quality_score DECIMAL(3,2) DEFAULT 1.0,
                PRIMARY KEY (time, asset_id, timeframe)
            );
        """)
        
        # Create hypertable
        try:
            await conn.execute("""
                SELECT create_hypertable('market_data', 'time', 
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE);
            """)
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # Real-time Price Updates
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS price_updates (
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                asset_id UUID NOT NULL REFERENCES assets(id),
                price DECIMAL(20,8) NOT NULL,
                volume_24h DECIMAL(20,8),
                price_change_24h DECIMAL(10,4),
                price_change_percent_24h DECIMAL(8,4),
                market_cap BIGINT,
                circulating_supply DECIMAL(20,8),
                total_supply DECIMAL(20,8),
                source VARCHAR(50) NOT NULL,
                PRIMARY KEY (time, asset_id)
            );
        """)
        
        try:
            await conn.execute("""
                SELECT create_hypertable('price_updates', 'time',
                chunk_time_interval => INTERVAL '1 hour',
                if_not_exists => TRUE);
            """)
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # Technical Indicators
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS technical_indicators (
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                asset_id UUID NOT NULL REFERENCES assets(id),
                indicator_name VARCHAR(50) NOT NULL,
                timeframe VARCHAR(10) NOT NULL,
                value DECIMAL(20,8) NOT NULL,
                signal VARCHAR(20),
                metadata JSONB DEFAULT '{}',
                PRIMARY KEY (time, asset_id, indicator_name, timeframe)
            );
        """)
        
        try:
            await conn.execute("""
                SELECT create_hypertable('technical_indicators', 'time',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE);
            """)
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        logger.info("Time-series tables created")
    
    async def create_portfolio_tables(self, conn: asyncpg.Connection):
        """Create portfolio management tables"""
        
        # Portfolios
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS portfolios (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                is_default BOOLEAN DEFAULT false,
                is_public BOOLEAN DEFAULT false,
                total_value DECIMAL(20,2) DEFAULT 0,
                total_cost DECIMAL(20,2) DEFAULT 0,
                total_pnl DECIMAL(20,2) DEFAULT 0,
                total_pnl_percent DECIMAL(8,4) DEFAULT 0,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Positions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
                asset_id UUID NOT NULL REFERENCES assets(id),
                quantity DECIMAL(20,8) NOT NULL,
                average_cost DECIMAL(20,8) NOT NULL,
                current_price DECIMAL(20,8),
                market_value DECIMAL(20,2),
                unrealized_pnl DECIMAL(20,2),
                unrealized_pnl_percent DECIMAL(8,4),
                allocation_percent DECIMAL(5,2),
                last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                UNIQUE(portfolio_id, asset_id)
            );
        """)
        
        # Transactions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
                asset_id UUID NOT NULL REFERENCES assets(id),
                transaction_type VARCHAR(20) NOT NULL,
                quantity DECIMAL(20,8) NOT NULL,
                price DECIMAL(20,8) NOT NULL,
                fees DECIMAL(20,8) DEFAULT 0,
                total_amount DECIMAL(20,2) NOT NULL,
                exchange VARCHAR(100),
                order_id VARCHAR(255),
                notes TEXT,
                executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                metadata JSONB DEFAULT '{}'
            );
        """)
        
        # Watchlists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlists (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                is_public BOOLEAN DEFAULT false,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Watchlist Items
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS watchlist_items (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                watchlist_id UUID NOT NULL REFERENCES watchlists(id) ON DELETE CASCADE,
                asset_id UUID NOT NULL REFERENCES assets(id),
                added_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                notes TEXT,
                UNIQUE(watchlist_id, asset_id)
            );
        """)
        
        logger.info("Portfolio tables created")
    
    async def create_ml_tables(self, conn: asyncpg.Connection):
        """Create ML and analytics tables"""
        
        # ML Models
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS ml_models (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                name VARCHAR(255) NOT NULL,
                model_type VARCHAR(100) NOT NULL,
                version VARCHAR(50) NOT NULL,
                asset_type VARCHAR(50),
                target_variable VARCHAR(100),
                features JSONB NOT NULL,
                hyperparameters JSONB DEFAULT '{}',
                performance_metrics JSONB DEFAULT '{}',
                model_path TEXT,
                model_size_bytes BIGINT,
                training_data_size INTEGER,
                is_active BOOLEAN DEFAULT true,
                trained_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_by UUID REFERENCES users(id)
            );
        """)
        
        # ML Predictions
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                model_id UUID NOT NULL REFERENCES ml_models(id),
                asset_id UUID NOT NULL REFERENCES assets(id),
                prediction_type VARCHAR(50) NOT NULL,
                predicted_value DECIMAL(20,8),
                confidence_score DECIMAL(5,4),
                actual_value DECIMAL(20,8),
                horizon_minutes INTEGER NOT NULL,
                features_used JSONB DEFAULT '{}',
                error_value DECIMAL(20,8),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                PRIMARY KEY (time, model_id, asset_id, prediction_type)
            );
        """)
        
        try:
            await conn.execute("""
                SELECT create_hypertable('predictions', 'time',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE);
            """)
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # Risk Metrics
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_metrics (
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                portfolio_id UUID REFERENCES portfolios(id),
                asset_id UUID REFERENCES assets(id),
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(20,8) NOT NULL,
                timeframe VARCHAR(20) NOT NULL,
                confidence_level DECIMAL(5,4),
                benchmark_value DECIMAL(20,8),
                percentile_rank DECIMAL(5,2),
                metadata JSONB DEFAULT '{}',
                PRIMARY KEY (time, COALESCE(portfolio_id, asset_id), metric_name, timeframe)
            );
        """)
        
        try:
            await conn.execute("""
                SELECT create_hypertable('risk_metrics', 'time',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE);
            """)
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        logger.info("ML and analytics tables created")
    
    async def create_news_tables(self, conn: asyncpg.Connection):
        """Create news and sentiment analysis tables"""
        
        # News Articles
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS news_articles (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                title TEXT NOT NULL,
                content TEXT,
                summary TEXT,
                url TEXT UNIQUE,
                source VARCHAR(255) NOT NULL,
                author VARCHAR(255),
                published_at TIMESTAMP WITH TIME ZONE NOT NULL,
                sentiment_score DECIMAL(5,4),
                sentiment_label VARCHAR(20),
                relevance_score DECIMAL(5,4),
                impact_score DECIMAL(5,4),
                language VARCHAR(10) DEFAULT 'en',
                category VARCHAR(100),
                tags TEXT[],
                image_url VARCHAR(500),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Asset-News Relationships
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS asset_news (
                asset_id UUID NOT NULL REFERENCES assets(id),
                news_id UUID NOT NULL REFERENCES news_articles(id),
                relevance_score DECIMAL(5,4) NOT NULL,
                mentioned_count INTEGER DEFAULT 1,
                sentiment_impact DECIMAL(5,4),
                PRIMARY KEY (asset_id, news_id)
            );
        """)
        
        # Social Media Sentiment
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS social_sentiment (
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                asset_id UUID NOT NULL REFERENCES assets(id),
                platform VARCHAR(50) NOT NULL,
                sentiment_score DECIMAL(5,4) NOT NULL,
                mention_count INTEGER NOT NULL,
                engagement_score DECIMAL(10,2),
                trending_score DECIMAL(10,2),
                volume_change_percent DECIMAL(8,4),
                PRIMARY KEY (time, asset_id, platform)
            );
        """)
        
        try:
            await conn.execute("""
                SELECT create_hypertable('social_sentiment', 'time',
                chunk_time_interval => INTERVAL '1 hour',
                if_not_exists => TRUE);
            """)
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        logger.info("News and sentiment tables created")
    
    async def create_monitoring_tables(self, conn: asyncpg.Connection):
        """Create system monitoring tables"""
        
        # API Usage Tracking
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS api_usage (
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                user_id UUID REFERENCES users(id),
                endpoint VARCHAR(255) NOT NULL,
                method VARCHAR(10) NOT NULL,
                status_code INTEGER NOT NULL,
                response_time_ms INTEGER,
                request_size_bytes INTEGER,
                response_size_bytes INTEGER,
                ip_address INET,
                user_agent TEXT,
                error_message TEXT,
                PRIMARY KEY (time, user_id, endpoint)
            );
        """)
        
        try:
            await conn.execute("""
                SELECT create_hypertable('api_usage', 'time',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE);
            """)
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # System Health Metrics
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS system_metrics (
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                service_name VARCHAR(100) NOT NULL,
                metric_name VARCHAR(100) NOT NULL,
                metric_value DECIMAL(20,8) NOT NULL,
                unit VARCHAR(20),
                hostname VARCHAR(100),
                tags JSONB DEFAULT '{}',
                PRIMARY KEY (time, service_name, metric_name)
            );
        """)
        
        try:
            await conn.execute("""
                SELECT create_hypertable('system_metrics', 'time',
                chunk_time_interval => INTERVAL '1 hour',
                if_not_exists => TRUE);
            """)
        except Exception as e:
            logger.warning(f"Hypertable creation warning: {e}")
        
        # Error Logs
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS error_logs (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                time TIMESTAMP WITH TIME ZONE NOT NULL,
                service_name VARCHAR(100) NOT NULL,
                error_level VARCHAR(20) NOT NULL,
                error_message TEXT NOT NULL,
                stack_trace TEXT,
                user_id UUID REFERENCES users(id),
                request_id UUID,
                session_id VARCHAR(255),
                metadata JSONB DEFAULT '{}',
                resolved BOOLEAN DEFAULT false,
                resolved_at TIMESTAMP WITH TIME ZONE,
                resolved_by UUID REFERENCES users(id)
            );
        """)
        
        # Alerts and Notifications
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS alerts (
                id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                user_id UUID NOT NULL REFERENCES users(id),
                alert_type VARCHAR(50) NOT NULL,
                title VARCHAR(255) NOT NULL,
                message TEXT NOT NULL,
                severity VARCHAR(20) DEFAULT 'info',
                asset_id UUID REFERENCES assets(id),
                portfolio_id UUID REFERENCES portfolios(id),
                trigger_condition JSONB,
                trigger_value DECIMAL(20,8),
                current_value DECIMAL(20,8),
                is_read BOOLEAN DEFAULT false,
                is_sent BOOLEAN DEFAULT false,
                sent_at TIMESTAMP WITH TIME ZONE,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                expires_at TIMESTAMP WITH TIME ZONE
            );
        """)
        
        logger.info("Monitoring tables created")
    
    async def create_indexes(self, conn: asyncpg.Connection):
        """Create database indexes for performance"""
        
        indexes = [
            # User indexes
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
            "CREATE INDEX IF NOT EXISTS idx_users_active ON users(is_active) WHERE is_active = true;",
            
            # Asset indexes
            "CREATE INDEX IF NOT EXISTS idx_assets_symbol ON assets(symbol);",
            "CREATE INDEX IF NOT EXISTS idx_assets_type ON assets(asset_type);",
            "CREATE INDEX IF NOT EXISTS idx_assets_active ON assets(is_active) WHERE is_active = true;",
            
            # Market data indexes
            "CREATE INDEX IF NOT EXISTS idx_market_data_asset_time ON market_data(asset_id, time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_market_data_timeframe ON market_data(timeframe, time DESC);",
            
            # Price update indexes
            "CREATE INDEX IF NOT EXISTS idx_price_updates_asset ON price_updates(asset_id, time DESC);",
            
            # Technical indicator indexes
            "CREATE INDEX IF NOT EXISTS idx_technical_indicators_asset_name ON technical_indicators(asset_id, indicator_name, time DESC);",
            
            # Portfolio indexes
            "CREATE INDEX IF NOT EXISTS idx_portfolios_user ON portfolios(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON positions(portfolio_id);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_portfolio_time ON transactions(portfolio_id, executed_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_transactions_asset ON transactions(asset_id);",
            
            # Watchlist indexes
            "CREATE INDEX IF NOT EXISTS idx_watchlists_user ON watchlists(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_watchlist_items_watchlist ON watchlist_items(watchlist_id);",
            
            # Prediction indexes
            "CREATE INDEX IF NOT EXISTS idx_predictions_asset_model ON predictions(asset_id, model_id, time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_predictions_type_time ON predictions(prediction_type, time DESC);",
            
            # News indexes
            "CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_news_sentiment ON news_articles(sentiment_score, published_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_asset_news_relevance ON asset_news(asset_id, relevance_score DESC);",
            
            # API usage indexes
            "CREATE INDEX IF NOT EXISTS idx_api_usage_user_time ON api_usage(user_id, time DESC);",
            "CREATE INDEX IF NOT EXISTS idx_api_usage_endpoint ON api_usage(endpoint, time DESC);",
            
            # Alert indexes
            "CREATE INDEX IF NOT EXISTS idx_alerts_user_unread ON alerts(user_id, is_read) WHERE is_read = false;",
            "CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC);",
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
            except Exception as e:
                logger.warning(f"Index creation warning: {e}")
        
        logger.info("Database indexes created")
    
    async def create_views(self, conn: asyncpg.Connection):
        """Create database views for common queries"""
        
        # Latest prices view
        await conn.execute("""
            CREATE OR REPLACE VIEW latest_prices AS
            SELECT DISTINCT ON (asset_id)
                asset_id,
                price,
                volume_24h,
                price_change_24h,
                price_change_percent_24h,
                market_cap,
                time
            FROM price_updates
            ORDER BY asset_id, time DESC;
        """)
        
        # Portfolio summary view
        await conn.execute("""
            CREATE OR REPLACE VIEW portfolio_summary AS
            SELECT 
                p.id as portfolio_id,
                p.name as portfolio_name,
                p.user_id,
                COUNT(pos.id) as position_count,
                COALESCE(SUM(pos.market_value), 0) as total_value,
                COALESCE(SUM(pos.unrealized_pnl), 0) as total_unrealized_pnl,
                CASE 
                    WHEN COALESCE(SUM(pos.quantity * pos.average_cost), 0) > 0 
                    THEN (COALESCE(SUM(pos.unrealized_pnl), 0) / COALESCE(SUM(pos.quantity * pos.average_cost), 1)) * 100
                    ELSE 0 
                END as total_pnl_percent
            FROM portfolios p
            LEFT JOIN positions pos ON p.id = pos.portfolio_id
            GROUP BY p.id, p.name, p.user_id;
        """)
        
        # Asset performance view
        await conn.execute("""
            CREATE OR REPLACE VIEW asset_performance AS
            SELECT 
                a.id as asset_id,
                a.symbol,
                a.name,
                a.asset_type,
                lp.price as current_price,
                lp.price_change_24h,
                lp.price_change_percent_24h,
                lp.volume_24h,
                lp.market_cap,
                rm.metric_value as volatility_30d
            FROM assets a
            LEFT JOIN latest_prices lp ON a.id = lp.asset_id
            LEFT JOIN LATERAL (
                SELECT metric_value
                FROM risk_metrics
                WHERE asset_id = a.id 
                AND metric_name = 'volatility'
                AND timeframe = '30d'
                ORDER BY time DESC
                LIMIT 1
            ) rm ON true
            WHERE a.is_active = true;
        """)
        
        # Top movers view
        await conn.execute("""
            CREATE OR REPLACE VIEW top_movers AS
            SELECT 
                a.symbol,
                a.name,
                a.asset_type,
                lp.price,
                lp.price_change_percent_24h,
                lp.volume_24h,
                lp.market_cap
            FROM assets a
            JOIN latest_prices lp ON a.id = lp.asset_id
            WHERE a.is_active = true
            AND lp.price_change_percent_24h IS NOT NULL
            ORDER BY ABS(lp.price_change_percent_24h) DESC;
        """)
        
        logger.info("Database views created")
    
    async def setup_retention_policies(self, conn: asyncpg.Connection):
        """Setup data retention policies"""
        
        retention_policies = [
            ("market_data", "2 years"),
            ("price_updates", "1 year"),
            ("technical_indicators", "1 year"),
            ("predictions", "6 months"),
            ("social_sentiment", "3 months"),
            ("api_usage", "1 year"),
            ("system_metrics", "6 months"),
        ]
        
        compression_policies = [
            ("market_data", "7 days"),
            ("price_updates", "1 day"),
            ("technical_indicators", "7 days"),
        ]
        
        for table, interval in retention_policies:
            try:
                await conn.execute(f"""
                    SELECT add_retention_policy('{table}', INTERVAL '{interval}');
                """)
                logger.info(f"Retention policy set for {table}: {interval}")
            except Exception as e:
                logger.warning(f"Retention policy warning for {table}: {e}")
        
        for table, interval in compression_policies:
            try:
                await conn.execute(f"""
                    SELECT add_compression_policy('{table}', INTERVAL '{interval}');
                """)
                logger.info(f"Compression policy set for {table}: {interval}")
            except Exception as e:
                logger.warning(f"Compression policy warning for {table}: {e}")
        
        logger.info("Retention policies configured")
    
    async def insert_sample_data(self, conn: asyncpg.Connection):
        """Insert sample data for testing"""
        
        # Sample assets
        sample_assets = [
            ('AAPL', 'Apple Inc.', 'stock', 'NASDAQ', 'Technology', 'Consumer Electronics'),
            ('MSFT', 'Microsoft Corporation', 'stock', 'NASDAQ', 'Technology', 'Software'),
            ('GOOGL', 'Alphabet Inc.', 'stock', 'NASDAQ', 'Technology', 'Internet Services'),
            ('TSLA', 'Tesla Inc.', 'stock', 'NASDAQ', 'Consumer Cyclical', 'Auto Manufacturers'),
            ('bitcoin', 'Bitcoin', 'crypto', 'Crypto', 'Cryptocurrency', 'Digital Currency'),
            ('ethereum', 'Ethereum', 'crypto', 'Crypto', 'Cryptocurrency', 'Smart Contracts'),
            ('cardano', 'Cardano', 'crypto', 'Crypto', 'Cryptocurrency', 'Blockchain Platform'),
        ]
        
        for symbol, name, asset_type, exchange, sector, industry in sample_assets:
            try:
                await conn.execute("""
                    INSERT INTO assets (symbol, name, asset_type, exchange, sector, industry)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (symbol, asset_type, exchange) DO NOTHING;
                """, symbol, name, asset_type, exchange, sector, industry)
            except Exception as e:
                logger.warning(f"Sample data insertion warning: {e}")
        
        # Sample admin user
        try:
            await conn.execute("""
                INSERT INTO users (email, password_hash, first_name, last_name, role, is_active, email_verified)
                VALUES ($1, $2, $3, $4, $5, $6, $7)
                ON CONFLICT (email) DO NOTHING;
            """, 'admin@example.com', 'hashed_password', 'Admin', 'User', 'admin', True, True)
        except Exception as e:
            logger.warning(f"Sample user insertion warning: {e}")
        
        logger.info("Sample data inserted")

async def main():
    """Main setup function"""
    logger.info("Starting database setup...")
    
    setup = DatabaseSetup()
    await setup.setup_database()
    
    logger.info("Database setup completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
