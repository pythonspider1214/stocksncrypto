-- Financial Analysis Bot Database Schema
-- Optimized for time-series data and high-frequency trading

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- =============================================
-- CORE TABLES
-- =============================================

-- Users and Authentication
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    first_name VARCHAR(100),
    last_name VARCHAR(100),
    role VARCHAR(50) DEFAULT 'user',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- API Keys and External Integrations
CREATE TABLE api_keys (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(100) NOT NULL, -- 'alpha_vantage', 'coingecko', etc.
    key_name VARCHAR(100) NOT NULL,
    encrypted_key TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    rate_limit_per_minute INTEGER DEFAULT 60,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Assets (Stocks, Crypto, etc.)
CREATE TABLE assets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol VARCHAR(20) NOT NULL,
    name VARCHAR(255) NOT NULL,
    asset_type VARCHAR(50) NOT NULL, -- 'stock', 'crypto', 'forex', 'commodity'
    exchange VARCHAR(100),
    sector VARCHAR(100),
    market_cap BIGINT,
    is_active BOOLEAN DEFAULT true,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(symbol, asset_type, exchange)
);

-- =============================================
-- TIME-SERIES TABLES (TimescaleDB Hypertables)
-- =============================================

-- Market Data (OHLCV)
CREATE TABLE market_data (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    asset_id UUID NOT NULL REFERENCES assets(id),
    open_price DECIMAL(20,8) NOT NULL,
    high_price DECIMAL(20,8) NOT NULL,
    low_price DECIMAL(20,8) NOT NULL,
    close_price DECIMAL(20,8) NOT NULL,
    volume DECIMAL(20,8) NOT NULL,
    volume_usd DECIMAL(20,2),
    timeframe VARCHAR(10) NOT NULL, -- '1m', '5m', '1h', '1d'
    source VARCHAR(50) NOT NULL,
    PRIMARY KEY (time, asset_id, timeframe)
);

-- Convert to hypertable
SELECT create_hypertable('market_data', 'time', chunk_time_interval => INTERVAL '1 day');

-- Real-time Price Updates
CREATE TABLE price_updates (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    asset_id UUID NOT NULL REFERENCES assets(id),
    price DECIMAL(20,8) NOT NULL,
    volume_24h DECIMAL(20,8),
    price_change_24h DECIMAL(10,4),
    market_cap BIGINT,
    source VARCHAR(50) NOT NULL,
    PRIMARY KEY (time, asset_id)
);

SELECT create_hypertable('price_updates', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Technical Indicators
CREATE TABLE technical_indicators (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    asset_id UUID NOT NULL REFERENCES assets(id),
    indicator_name VARCHAR(50) NOT NULL,
    timeframe VARCHAR(10) NOT NULL,
    value DECIMAL(20,8) NOT NULL,
    metadata JSONB,
    PRIMARY KEY (time, asset_id, indicator_name, timeframe)
);

SELECT create_hypertable('technical_indicators', 'time', chunk_time_interval => INTERVAL '1 day');

-- =============================================
-- PORTFOLIO MANAGEMENT
-- =============================================

-- User Portfolios
CREATE TABLE portfolios (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    is_default BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio Positions
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    asset_id UUID NOT NULL REFERENCES assets(id),
    quantity DECIMAL(20,8) NOT NULL,
    average_cost DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,2),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(portfolio_id, asset_id)
);

-- Trading Transactions
CREATE TABLE transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    portfolio_id UUID NOT NULL REFERENCES portfolios(id) ON DELETE CASCADE,
    asset_id UUID NOT NULL REFERENCES assets(id),
    transaction_type VARCHAR(20) NOT NULL, -- 'buy', 'sell'
    quantity DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    fees DECIMAL(20,8) DEFAULT 0,
    total_amount DECIMAL(20,2) NOT NULL,
    executed_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB
);

-- =============================================
-- MACHINE LEARNING & ANALYTICS
-- =============================================

-- ML Models
CREATE TABLE ml_models (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    model_type VARCHAR(100) NOT NULL, -- 'lstm', 'random_forest', 'xgboost'
    version VARCHAR(50) NOT NULL,
    asset_type VARCHAR(50),
    target_variable VARCHAR(100), -- 'price', 'direction', 'volatility'
    features JSONB NOT NULL,
    hyperparameters JSONB,
    performance_metrics JSONB,
    model_path TEXT,
    is_active BOOLEAN DEFAULT true,
    trained_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ML Predictions
CREATE TABLE predictions (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    model_id UUID NOT NULL REFERENCES ml_models(id),
    asset_id UUID NOT NULL REFERENCES assets(id),
    prediction_type VARCHAR(50) NOT NULL, -- 'price', 'direction', 'volatility'
    predicted_value DECIMAL(20,8),
    confidence_score DECIMAL(5,4),
    actual_value DECIMAL(20,8),
    horizon_minutes INTEGER NOT NULL, -- prediction horizon
    features_used JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    PRIMARY KEY (time, model_id, asset_id, prediction_type)
);

SELECT create_hypertable('predictions', 'time', chunk_time_interval => INTERVAL '1 day');

-- Risk Metrics
CREATE TABLE risk_metrics (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    portfolio_id UUID REFERENCES portfolios(id),
    asset_id UUID REFERENCES assets(id),
    metric_name VARCHAR(100) NOT NULL, -- 'var', 'sharpe_ratio', 'max_drawdown'
    metric_value DECIMAL(20,8) NOT NULL,
    timeframe VARCHAR(20) NOT NULL, -- '1d', '7d', '30d', '1y'
    confidence_level DECIMAL(5,4), -- for VaR calculations
    metadata JSONB,
    PRIMARY KEY (time, COALESCE(portfolio_id, asset_id), metric_name, timeframe)
);

SELECT create_hypertable('risk_metrics', 'time', chunk_time_interval => INTERVAL '1 day');

-- =============================================
-- NEWS & SENTIMENT ANALYSIS
-- =============================================

-- News Articles
CREATE TABLE news_articles (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    title TEXT NOT NULL,
    content TEXT,
    url TEXT UNIQUE,
    source VARCHAR(255) NOT NULL,
    author VARCHAR(255),
    published_at TIMESTAMP WITH TIME ZONE NOT NULL,
    sentiment_score DECIMAL(5,4), -- -1 to 1
    sentiment_label VARCHAR(20), -- 'positive', 'negative', 'neutral'
    relevance_score DECIMAL(5,4),
    language VARCHAR(10) DEFAULT 'en',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Asset-News Relationships
CREATE TABLE asset_news (
    asset_id UUID NOT NULL REFERENCES assets(id),
    news_id UUID NOT NULL REFERENCES news_articles(id),
    relevance_score DECIMAL(5,4) NOT NULL,
    mentioned_count INTEGER DEFAULT 1,
    PRIMARY KEY (asset_id, news_id)
);

-- Social Media Sentiment
CREATE TABLE social_sentiment (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    asset_id UUID NOT NULL REFERENCES assets(id),
    platform VARCHAR(50) NOT NULL, -- 'twitter', 'reddit', 'telegram'
    sentiment_score DECIMAL(5,4) NOT NULL,
    mention_count INTEGER NOT NULL,
    engagement_score DECIMAL(10,2),
    trending_score DECIMAL(10,2),
    PRIMARY KEY (time, asset_id, platform)
);

SELECT create_hypertable('social_sentiment', 'time', chunk_time_interval => INTERVAL '1 hour');

-- =============================================
-- SYSTEM MONITORING & LOGS
-- =============================================

-- API Usage Tracking
CREATE TABLE api_usage (
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
    PRIMARY KEY (time, user_id, endpoint)
);

SELECT create_hypertable('api_usage', 'time', chunk_time_interval => INTERVAL '1 day');

-- System Health Metrics
CREATE TABLE system_metrics (
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20,8) NOT NULL,
    unit VARCHAR(20),
    tags JSONB,
    PRIMARY KEY (time, service_name, metric_name)
);

SELECT create_hypertable('system_metrics', 'time', chunk_time_interval => INTERVAL '1 hour');

-- Error Logs
CREATE TABLE error_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    time TIMESTAMP WITH TIME ZONE NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    error_level VARCHAR(20) NOT NULL, -- 'ERROR', 'CRITICAL', 'WARNING'
    error_message TEXT NOT NULL,
    stack_trace TEXT,
    user_id UUID REFERENCES users(id),
    request_id UUID,
    metadata JSONB,
    resolved BOOLEAN DEFAULT false,
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- =============================================
-- INDEXES FOR PERFORMANCE
-- =============================================

-- Market Data Indexes
CREATE INDEX idx_market_data_asset_time ON market_data (asset_id, time DESC);
CREATE INDEX idx_market_data_timeframe ON market_data (timeframe, time DESC);

-- Price Updates Indexes
CREATE INDEX idx_price_updates_asset ON price_updates (asset_id, time DESC);

-- Technical Indicators Indexes
CREATE INDEX idx_technical_indicators_asset_name ON technical_indicators (asset_id, indicator_name, time DESC);

-- Portfolio Indexes
CREATE INDEX idx_positions_portfolio ON positions (portfolio_id);
CREATE INDEX idx_transactions_portfolio_time ON transactions (portfolio_id, executed_at DESC);

-- Predictions Indexes
CREATE INDEX idx_predictions_asset_model ON predictions (asset_id, model_id, time DESC);
CREATE INDEX idx_predictions_type_time ON predictions (prediction_type, time DESC);

-- News Indexes
CREATE INDEX idx_news_published ON news_articles (published_at DESC);
CREATE INDEX idx_news_sentiment ON news_articles (sentiment_score, published_at DESC);
CREATE INDEX idx_asset_news_relevance ON asset_news (asset_id, relevance_score DESC);

-- API Usage Indexes
CREATE INDEX idx_api_usage_user_time ON api_usage (user_id, time DESC);
CREATE INDEX idx_api_usage_endpoint ON api_usage (endpoint, time DESC);

-- =============================================
-- DATA RETENTION POLICIES
-- =============================================

-- Retain raw market data for 2 years, then compress
SELECT add_retention_policy('market_data', INTERVAL '2 years');
SELECT add_compression_policy('market_data', INTERVAL '7 days');

-- Retain price updates for 1 year
SELECT add_retention_policy('price_updates', INTERVAL '1 year');
SELECT add_compression_policy('price_updates', INTERVAL '1 day');

-- Retain technical indicators for 1 year
SELECT add_retention_policy('technical_indicators', INTERVAL '1 year');
SELECT add_compression_policy('technical_indicators', INTERVAL '7 days');

-- Retain predictions for 6 months
SELECT add_retention_policy('predictions', INTERVAL '6 months');

-- Retain social sentiment for 3 months
SELECT add_retention_policy('social_sentiment', INTERVAL '3 months');

-- Retain API usage logs for 1 year
SELECT add_retention_policy('api_usage', INTERVAL '1 year');

-- Retain system metrics for 6 months
SELECT add_retention_policy('system_metrics', INTERVAL '6 months');

-- =============================================
-- FUNCTIONS AND TRIGGERS
-- =============================================

-- Update timestamp trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply to relevant tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_assets_updated_at BEFORE UPDATE ON assets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_portfolios_updated_at BEFORE UPDATE ON portfolios
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Calculate portfolio value function
CREATE OR REPLACE FUNCTION calculate_portfolio_value(portfolio_uuid UUID)
RETURNS DECIMAL(20,2) AS $$
DECLARE
    total_value DECIMAL(20,2) := 0;
BEGIN
    SELECT COALESCE(SUM(quantity * current_price), 0)
    INTO total_value
    FROM positions
    WHERE portfolio_id = portfolio_uuid;
    
    RETURN total_value;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- VIEWS FOR COMMON QUERIES
-- =============================================

-- Latest prices view
CREATE VIEW latest_prices AS
SELECT DISTINCT ON (asset_id)
    asset_id,
    price,
    volume_24h,
    price_change_24h,
    market_cap,
    time
FROM price_updates
ORDER BY asset_id, time DESC;

-- Portfolio summary view
CREATE VIEW portfolio_summary AS
SELECT 
    p.id as portfolio_id,
    p.name as portfolio_name,
    p.user_id,
    COUNT(pos.id) as position_count,
    SUM(pos.quantity * pos.current_price) as total_value,
    SUM(pos.unrealized_pnl) as total_unrealized_pnl
FROM portfolios p
LEFT JOIN positions pos ON p.id = pos.portfolio_id
GROUP BY p.id, p.name, p.user_id;

-- Asset performance view
CREATE VIEW asset_performance AS
SELECT 
    a.id as asset_id,
    a.symbol,
    a.name,
    lp.price as current_price,
    lp.price_change_24h,
    lp.volume_24h,
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
) rm ON true;
