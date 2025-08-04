#!/usr/bin/env python3
"""
API Integration Module for Financial Analysis Bot
Handles data ingestion from multiple financial APIs with robust error handling
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
import os
from decimal import Decimal
import time
import hashlib
import hmac

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class APIConfig:
    """API configuration"""
    name: str
    base_url: str
    api_key: str
    rate_limit: int  # requests per minute
    timeout: int = 30

@dataclass
class MarketData:
    """Market data structure"""
    symbol: str
    timestamp: datetime
    open_price: Decimal
    high_price: Decimal
    low_price: Decimal
    close_price: Decimal
    volume: Decimal
    source: str

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token for API call"""
        async with self.lock:
            now = time.time()
            # Add tokens based on time passed
            time_passed = now - self.last_update
            self.tokens = min(self.rate_limit, self.tokens + time_passed * (self.rate_limit / 60))
            self.last_update = now
            
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            else:
                # Calculate wait time
                wait_time = (1 - self.tokens) * (60 / self.rate_limit)
                await asyncio.sleep(wait_time)
                self.tokens = 0
                return True

class CircuitBreaker:
    """Circuit breaker for API failures"""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
    
    def can_execute(self) -> bool:
        """Check if request can be executed"""
        if self.state == 'CLOSED':
            return True
        elif self.state == 'OPEN':
            if time.time() - self.last_failure_time > self.timeout:
                self.state = 'HALF_OPEN'
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def record_success(self):
        """Record successful request"""
        self.failure_count = 0
        self.state = 'CLOSED'
    
    def record_failure(self):
        """Record failed request"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'OPEN'

class APIClient:
    """Base API client with error handling and rate limiting"""
    
    def __init__(self, config: APIConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.rate_limit)
        self.circuit_breaker = CircuitBreaker()
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.timeout)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def make_request(self, url: str, params: Dict = None, headers: Dict = None) -> Optional[Dict]:
        """Make API request with error handling"""
        if not self.circuit_breaker.can_execute():
            logger.warning(f"Circuit breaker OPEN for {self.config.name}")
            return None
        
        await self.rate_limiter.acquire()
        
        try:
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    self.circuit_breaker.record_success()
                    return data
                elif response.status == 429:  # Rate limited
                    logger.warning(f"Rate limited by {self.config.name}")
                    await asyncio.sleep(60)  # Wait 1 minute
                    return None
                else:
                    logger.error(f"API error {response.status} from {self.config.name}")
                    self.circuit_breaker.record_failure()
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout error for {self.config.name}")
            self.circuit_breaker.record_failure()
            return None
        except Exception as e:
            logger.error(f"Request error for {self.config.name}: {e}")
            self.circuit_breaker.record_failure()
            return None

class AlphaVantageClient(APIClient):
    """Alpha Vantage API client"""
    
    async def get_daily_data(self, symbol: str) -> Optional[List[MarketData]]:
        """Get daily stock data"""
        url = f"{self.config.base_url}/query"
        params = {
            'function': 'TIME_SERIES_DAILY',
            'symbol': symbol,
            'apikey': self.config.api_key,
            'outputsize': 'compact'
        }
        
        data = await self.make_request(url, params)
        if not data or 'Time Series (Daily)' not in data:
            return None
        
        market_data = []
        for date_str, values in data['Time Series (Daily)'].items():
            try:
                market_data.append(MarketData(
                    symbol=symbol,
                    timestamp=datetime.strptime(date_str, '%Y-%m-%d'),
                    open_price=Decimal(values['1. open']),
                    high_price=Decimal(values['2. high']),
                    low_price=Decimal(values['3. low']),
                    close_price=Decimal(values['4. close']),
                    volume=Decimal(values['5. volume']),
                    source='alpha_vantage'
                ))
            except (KeyError, ValueError) as e:
                logger.error(f"Error parsing Alpha Vantage data: {e}")
                continue
        
        return market_data

class CoinGeckoClient(APIClient):
    """CoinGecko API client"""
    
    async def get_price_data(self, coin_ids: List[str]) -> Optional[Dict]:
        """Get current crypto prices"""
        url = f"{self.config.base_url}/simple/price"
        params = {
            'ids': ','.join(coin_ids),
            'vs_currencies': 'usd',
            'include_24hr_change': 'true',
            'include_24hr_vol': 'true',
            'include_market_cap': 'true'
        }
        
        return await self.make_request(url, params)
    
    async def get_historical_data(self, coin_id: str, days: int = 30) -> Optional[List[MarketData]]:
        """Get historical crypto data"""
        url = f"{self.config.base_url}/coins/{coin_id}/market_chart"
        params = {
            'vs_currency': 'usd',
            'days': days,
            'interval': 'daily'
        }
        
        data = await self.make_request(url, params)
        if not data or 'prices' not in data:
            return None
        
        market_data = []
        prices = data['prices']
        volumes = data.get('total_volumes', [])
        
        for i, (timestamp_ms, price) in enumerate(prices):
            try:
                volume = volumes[i][1] if i < len(volumes) else 0
                market_data.append(MarketData(
                    symbol=coin_id,
                    timestamp=datetime.fromtimestamp(timestamp_ms / 1000),
                    open_price=Decimal(str(price)),
                    high_price=Decimal(str(price)),
                    low_price=Decimal(str(price)),
                    close_price=Decimal(str(price)),
                    volume=Decimal(str(volume)),
                    source='coingecko'
                ))
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing CoinGecko data: {e}")
                continue
        
        return market_data

class PolygonClient(APIClient):
    """Polygon.io API client for real-time data"""
    
    async def get_real_time_quotes(self, symbols: List[str]) -> Optional[Dict]:
        """Get real-time stock quotes"""
        results = {}
        
        for symbol in symbols:
            url = f"{self.config.base_url}/v2/last/trade/{symbol}"
            params = {'apikey': self.config.api_key}
            
            data = await self.make_request(url, params)
            if data and 'results' in data:
                results[symbol] = data['results']
        
        return results

class NewsAPIClient(APIClient):
    """News API client for sentiment analysis"""
    
    async def get_financial_news(self, query: str, from_date: datetime = None) -> Optional[List[Dict]]:
        """Get financial news articles"""
        url = f"{self.config.base_url}/everything"
        params = {
            'q': query,
            'apiKey': self.config.api_key,
            'language': 'en',
            'sortBy': 'publishedAt',
            'pageSize': 100
        }
        
        if from_date:
            params['from'] = from_date.strftime('%Y-%m-%d')
        
        data = await self.make_request(url, params)
        return data.get('articles', []) if data else None

class DataIngestionManager:
    """Manages data ingestion from multiple sources"""
    
    def __init__(self):
        self.clients = {}
        self.setup_clients()
    
    def setup_clients(self):
        """Setup API clients"""
        # Alpha Vantage
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            self.clients['alpha_vantage'] = AlphaVantageClient(APIConfig(
                name='Alpha Vantage',
                base_url='https://www.alphavantage.co',
                api_key=os.getenv('ALPHA_VANTAGE_API_KEY'),
                rate_limit=5  # 5 requests per minute
            ))
        
        # CoinGecko
        self.clients['coingecko'] = CoinGeckoClient(APIConfig(
            name='CoinGecko',
            base_url='https://api.coingecko.com/api/v3',
            api_key='',  # CoinGecko free tier doesn't require API key
            rate_limit=50  # 50 requests per minute
        ))
        
        # Polygon.io
        if os.getenv('POLYGON_API_KEY'):
            self.clients['polygon'] = PolygonClient(APIConfig(
                name='Polygon',
                base_url='https://api.polygon.io',
                api_key=os.getenv('POLYGON_API_KEY'),
                rate_limit=5  # 5 requests per minute for free tier
            ))
        
        # News API
        if os.getenv('NEWS_API_KEY'):
            self.clients['news_api'] = NewsAPIClient(APIConfig(
                name='News API',
                base_url='https://newsapi.org/v2',
                api_key=os.getenv('NEWS_API_KEY'),
                rate_limit=1000  # 1000 requests per day
            ))
    
    async def fetch_stock_data(self, symbols: List[str]) -> Dict[str, List[MarketData]]:
        """Fetch stock data from multiple sources"""
        results = {}
        
        # Try Alpha Vantage first
        if 'alpha_vantage' in self.clients:
            async with self.clients['alpha_vantage'] as client:
                for symbol in symbols:
                    data = await client.get_daily_data(symbol)
                    if data:
                        results[symbol] = data
                        logger.info(f"Fetched {len(data)} records for {symbol} from Alpha Vantage")
        
        # Fallback to other sources if needed
        # Implementation for other stock data sources...
        
        return results
    
    async def fetch_crypto_data(self, coin_ids: List[str]) -> Dict[str, Any]:
        """Fetch crypto data from CoinGecko"""
        results = {}
        
        if 'coingecko' in self.clients:
            async with self.clients['coingecko'] as client:
                # Current prices
                price_data = await client.get_price_data(coin_ids)
                if price_data:
                    results['prices'] = price_data
                
                # Historical data
                historical_data = {}
                for coin_id in coin_ids:
                    data = await client.get_historical_data(coin_id)
                    if data:
                        historical_data[coin_id] = data
                        logger.info(f"Fetched {len(data)} historical records for {coin_id}")
                
                results['historical'] = historical_data
        
        return results
    
    async def fetch_news_data(self, queries: List[str]) -> Dict[str, List[Dict]]:
        """Fetch news data for sentiment analysis"""
        results = {}
        
        if 'news_api' in self.clients:
            async with self.clients['news_api'] as client:
                for query in queries:
                    articles = await client.get_financial_news(
                        query, 
                        from_date=datetime.now() - timedelta(days=7)
                    )
                    if articles:
                        results[query] = articles
                        logger.info(f"Fetched {len(articles)} articles for query: {query}")
        
        return results

async def main():
    """Test the API integration"""
    logger.info("Testing API integration...")
    
    manager = DataIngestionManager()
    
    # Test stock data
    stock_symbols = ['AAPL', 'MSFT', 'GOOGL']
    stock_data = await manager.fetch_stock_data(stock_symbols)
    logger.info(f"Stock data fetched for {len(stock_data)} symbols")
    
    # Test crypto data
    crypto_ids = ['bitcoin', 'ethereum', 'cardano']
    crypto_data = await manager.fetch_crypto_data(crypto_ids)
    logger.info(f"Crypto data fetched: {list(crypto_data.keys())}")
    
    # Test news data
    news_queries = ['Apple stock', 'Bitcoin price', 'Tesla earnings']
    news_data = await manager.fetch_news_data(news_queries)
    logger.info(f"News data fetched for {len(news_data)} queries")
    
    logger.info("API integration test completed")

if __name__ == "__main__":
    asyncio.run(main())
