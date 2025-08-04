#!/usr/bin/env python3
"""
Test Suite for API Integration Module
Comprehensive tests for data ingestion and API clients
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json
from decimal import Decimal

# Import modules to test
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.api_integration import (
    APIConfig, MarketData, RateLimiter, CircuitBreaker,
    APIClient, AlphaVantageClient, CoinGeckoClient,
    PolygonClient, NewsAPIClient, DataIngestionManager
)

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Test that rate limiter allows requests within the limit"""
        limiter = RateLimiter(rate_limit=60)  # 60 requests per minute
        
        # Should allow first request immediately
        result = await limiter.acquire()
        assert result is True
        
        # Should allow multiple requests up to limit
        for _ in range(5):
            result = await limiter.acquire()
            assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self):
        """Test that rate limiter blocks requests exceeding the limit"""
        limiter = RateLimiter(rate_limit=2)  # 2 requests per minute
        
        # Use up the tokens
        await limiter.acquire()
        await limiter.acquire()
        
        # Next request should be delayed
        start_time = asyncio.get_event_loop().time()
        await limiter.acquire()
        end_time = asyncio.get_event_loop().time()
        
        # Should have waited some time
        assert end_time - start_time > 0

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_initial_state(self):
        """Test circuit breaker initial state"""
        breaker = CircuitBreaker(failure_threshold=3, timeout=60)
        assert breaker.state == 'CLOSED'
        assert breaker.can_execute() is True
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test circuit breaker opens after threshold failures"""
        breaker = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Record failures
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.state == 'OPEN'
        assert breaker.can_execute() is False
    
    def test_circuit_breaker_resets_on_success(self):
        """Test circuit breaker resets on success"""
        breaker = CircuitBreaker(failure_threshold=3, timeout=60)
        
        # Record some failures
        breaker.record_failure()
        breaker.record_failure()
        
        # Record success
        breaker.record_success()
        
        assert breaker.failure_count == 0
        assert breaker.state == 'CLOSED'

class TestAPIClient:
    """Test base API client functionality"""
    
    @pytest.fixture
    def api_config(self):
        """Create test API configuration"""
        return APIConfig(
            name="Test API",
            base_url="https://api.test.com",
            api_key="test_key",
            rate_limit=60,
            timeout=30
        )
    
    @pytest.mark.asyncio
    async def test_api_client_successful_request(self, api_config):
        """Test successful API request"""
        client = APIClient(api_config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client:
                result = await client.make_request("https://api.test.com/data")
                
                assert result == {"data": "test"}
                assert client.circuit_breaker.state == 'CLOSED'
    
    @pytest.mark.asyncio
    async def test_api_client_handles_rate_limit(self, api_config):
        """Test API client handles rate limiting"""
        client = APIClient(api_config)
        
        # Mock rate limit response
        mock_response = Mock()
        mock_response.status = 429
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client:
                result = await client.make_request("https://api.test.com/data")
                
                assert result is None
    
    @pytest.mark.asyncio
    async def test_api_client_handles_errors(self, api_config):
        """Test API client handles errors properly"""
        client = APIClient(api_config)
        
        # Mock error response
        mock_response = Mock()
        mock_response.status = 500
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client:
                result = await client.make_request("https://api.test.com/data")
                
                assert result is None
                assert client.circuit_breaker.failure_count > 0

class TestAlphaVantageClient:
    """Test Alpha Vantage API client"""
    
    @pytest.fixture
    def alpha_vantage_config(self):
        """Create Alpha Vantage API configuration"""
        return APIConfig(
            name="Alpha Vantage",
            base_url="https://www.alphavantage.co",
            api_key="test_key",
            rate_limit=5
        )
    
    @pytest.fixture
    def sample_alpha_vantage_response(self):
        """Sample Alpha Vantage API response"""
        return {
            "Time Series (Daily)": {
                "2023-12-01": {
                    "1. open": "150.00",
                    "2. high": "155.00",
                    "3. low": "149.00",
                    "4. close": "154.00",
                    "5. volume": "1000000"
                },
                "2023-11-30": {
                    "1. open": "148.00",
                    "2. high": "152.00",
                    "3. low": "147.00",
                    "4. close": "151.00",
                    "5. volume": "950000"
                }
            }
        }
    
    @pytest.mark.asyncio
    async def test_alpha_vantage_get_daily_data(self, alpha_vantage_config, sample_alpha_vantage_response):
        """Test Alpha Vantage daily data retrieval"""
        client = AlphaVantageClient(alpha_vantage_config)
        
        with patch.object(client, 'make_request', return_value=sample_alpha_vantage_response):
            async with client:
                result = await client.get_daily_data("AAPL")
                
                assert result is not None
                assert len(result) == 2
                assert isinstance(result[0], MarketData)
                assert result[0].symbol == "AAPL"
                assert result[0].close_price == Decimal("154.00")
    
    @pytest.mark.asyncio
    async def test_alpha_vantage_handles_invalid_response(self, alpha_vantage_config):
        """Test Alpha Vantage handles invalid response"""
        client = AlphaVantageClient(alpha_vantage_config)
        
        with patch.object(client, 'make_request', return_value={"error": "Invalid API call"}):
            async with client:
                result = await client.get_daily_data("INVALID")
                
                assert result is None

class TestCoinGeckoClient:
    """Test CoinGecko API client"""
    
    @pytest.fixture
    def coingecko_config(self):
        """Create CoinGecko API configuration"""
        return APIConfig(
            name="CoinGecko",
            base_url="https://api.coingecko.com/api/v3",
            api_key="",
            rate_limit=50
        )
    
    @pytest.fixture
    def sample_coingecko_price_response(self):
        """Sample CoinGecko price response"""
        return {
            "bitcoin": {
                "usd": 45000.0,
                "usd_24h_change": 2.5,
                "usd_24h_vol": 25000000000,
                "usd_market_cap": 850000000000
            },
            "ethereum": {
                "usd": 3000.0,
                "usd_24h_change": -1.2,
                "usd_24h_vol": 15000000000,
                "usd_market_cap": 360000000000
            }
        }
    
    @pytest.fixture
    def sample_coingecko_historical_response(self):
        """Sample CoinGecko historical response"""
        return {
            "prices": [
                [1701388800000, 44000.0],  # timestamp, price
                [1701475200000, 45000.0],
                [1701561600000, 46000.0]
            ],
            "total_volumes": [
                [1701388800000, 24000000000],
                [1701475200000, 25000000000],
                [1701561600000, 26000000000]
            ]
        }
    
    @pytest.mark.asyncio
    async def test_coingecko_get_price_data(self, coingecko_config, sample_coingecko_price_response):
        """Test CoinGecko price data retrieval"""
        client = CoinGeckoClient(coingecko_config)
        
        with patch.object(client, 'make_request', return_value=sample_coingecko_price_response):
            async with client:
                result = await client.get_price_data(["bitcoin", "ethereum"])
                
                assert result is not None
                assert "bitcoin" in result
                assert "ethereum" in result
                assert result["bitcoin"]["usd"] == 45000.0
    
    @pytest.mark.asyncio
    async def test_coingecko_get_historical_data(self, coingecko_config, sample_coingecko_historical_response):
        """Test CoinGecko historical data retrieval"""
        client = CoinGeckoClient(coingecko_config)
        
        with patch.object(client, 'make_request', return_value=sample_coingecko_historical_response):
            async with client:
                result = await client.get_historical_data("bitcoin", days=30)
                
                assert result is not None
                assert len(result) == 3
                assert isinstance(result[0], MarketData)
                assert result[0].symbol == "bitcoin"
                assert result[0].source == "coingecko"

class TestDataIngestionManager:
    """Test data ingestion manager"""
    
    @pytest.fixture
    def manager(self):
        """Create data ingestion manager"""
        return DataIngestionManager()
    
    @pytest.mark.asyncio
    async def test_fetch_stock_data(self, manager):
        """Test stock data fetching"""
        # Mock Alpha Vantage client
        mock_client = Mock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_daily_data = AsyncMock(return_value=[
            MarketData(
                symbol="AAPL",
                timestamp=datetime.now(),
                open_price=Decimal("150.00"),
                high_price=Decimal("155.00"),
                low_price=Decimal("149.00"),
                close_price=Decimal("154.00"),
                volume=Decimal("1000000"),
                source="alpha_vantage"
            )
        ])
        
        manager.clients['alpha_vantage'] = mock_client
        
        result = await manager.fetch_stock_data(["AAPL"])
        
        assert "AAPL" in result
        assert len(result["AAPL"]) == 1
        assert result["AAPL"][0].symbol == "AAPL"
    
    @pytest.mark.asyncio
    async def test_fetch_crypto_data(self, manager):
        """Test crypto data fetching"""
        # Mock CoinGecko client
        mock_client = Mock()
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=None)
        mock_client.get_price_data = AsyncMock(return_value={
            "bitcoin": {"usd": 45000.0}
        })
        mock_client.get_historical_data = AsyncMock(return_value=[
            MarketData(
                symbol="bitcoin",
                timestamp=datetime.now(),
                open_price=Decimal("44000.00"),
                high_price=Decimal("45000.00"),
                low_price=Decimal("43000.00"),
                close_price=Decimal("45000.00"),
                volume=Decimal("25000000000"),
                source="coingecko"
            )
        ])
        
        manager.clients['coingecko'] = mock_client
        
        result = await manager.fetch_crypto_data(["bitcoin"])
        
        assert "prices" in result
        assert "historical" in result
        assert "bitcoin" in result["prices"]
        assert result["prices"]["bitcoin"]["usd"] == 45000.0

class TestIntegration:
    """Integration tests"""
    
    @pytest.mark.asyncio
    async def test_full_data_ingestion_workflow(self):
        """Test complete data ingestion workflow"""
        manager = DataIngestionManager()
        
        # Test with mock data
        with patch.dict(os.environ, {
            'ALPHA_VANTAGE_API_KEY': 'test_key',
            'NEWS_API_KEY': 'test_key'
        }):
            # Reinitialize to pick up environment variables
            manager.setup_clients()
            
            # Verify clients are set up
            assert 'alpha_vantage' in manager.clients
            assert 'coingecko' in manager.clients
            assert 'news_api' in manager.clients
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        config = APIConfig(
            name="Test API",
            base_url="https://api.test.com",
            api_key="test_key",
            rate_limit=60
        )
        
        client = APIClient(config)
        
        # Test circuit breaker functionality
        with patch('aiohttp.ClientSession.get') as mock_get:
            # Simulate multiple failures
            mock_response = Mock()
            mock_response.status = 500
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client:
                # Make multiple failed requests
                for _ in range(6):
                    await client.make_request("https://api.test.com/data")
                
                # Circuit breaker should be open
                assert client.circuit_breaker.state == 'OPEN'
                assert client.circuit_breaker.can_execute() is False

# Performance tests
class TestPerformance:
    """Performance and load tests"""
    
    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling of concurrent requests"""
        config = APIConfig(
            name="Test API",
            base_url="https://api.test.com",
            api_key="test_key",
            rate_limit=100
        )
        
        client = APIClient(config)
        
        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": "test"})
        
        with patch('aiohttp.ClientSession.get') as mock_get:
            mock_get.return_value.__aenter__.return_value = mock_response
            
            async with client:
                # Make concurrent requests
                tasks = []
                for i in range(10):
                    task = client.make_request(f"https://api.test.com/data/{i}")
                    tasks.append(task)
                
                results = await asyncio.gather(*tasks)
                
                # All requests should succeed
                assert len(results) == 10
                assert all(result == {"data": "test"} for result in results)
    
    @pytest.mark.asyncio
    async def test_rate_limiter_performance(self):
        """Test rate limiter performance under load"""
        limiter = RateLimiter(rate_limit=100)  # 100 requests per minute
        
        start_time = asyncio.get_event_loop().time()
        
        # Make requests up to the limit
        tasks = []
        for _ in range(50):
            task = limiter.acquire()
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        end_time = asyncio.get_event_loop().time()
        
        # Should complete relatively quickly
        assert end_time - start_time < 5.0  # Less than 5 seconds

# Fixtures for test data
@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    return MarketData(
        symbol="AAPL",
        timestamp=datetime.now(),
        open_price=Decimal("150.00"),
        high_price=Decimal("155.00"),
        low_price=Decimal("149.00"),
        close_price=Decimal("154.00"),
        volume=Decimal("1000000"),
        source="test"
    )

@pytest.fixture
def sample_news_articles():
    """Sample news articles for testing"""
    return [
        {
            "title": "Apple Reports Strong Q4 Earnings",
            "content": "Apple Inc. reported strong quarterly earnings...",
            "url": "https://example.com/news/1",
            "source": "Financial Times",
            "publishedAt": "2023-12-01T10:00:00Z"
        },
        {
            "title": "Tesla Stock Surges on New Model Announcement",
            "content": "Tesla stock jumped 5% after announcing...",
            "url": "https://example.com/news/2",
            "source": "Reuters",
            "publishedAt": "2023-12-01T11:00:00Z"
        }
    ]

# Test configuration
@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
