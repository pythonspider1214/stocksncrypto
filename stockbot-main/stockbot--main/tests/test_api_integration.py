#!/usr/bin/env python3
"""
Test suite for API integration components
"""

import pytest
import asyncio
import aiohttp
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import json

# Import modules to test
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from api_integration import (
    APIClient, APIConfig, RateLimiter, CircuitBreaker,
    AlphaVantageClient, CoinGeckoClient, DataIngestionManager
)

class TestRateLimiter:
    """Test rate limiting functionality"""
    
    @pytest.mark.asyncio
    async def test_rate_limiter_allows_requests_within_limit(self):
        """Test that rate limiter allows requests within the limit"""
        limiter = RateLimiter(rate_limit=60)  # 60 requests per minute
        
        # Should allow immediate request
        result = await limiter.acquire()
        assert result is True
        
        # Should allow another request (we have 60 tokens)
        result = await limiter.acquire()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excessive_requests(self):
        """Test that rate limiter blocks excessive requests"""
        limiter = RateLimiter(rate_limit=1)  # 1 request per minute
        
        # First request should succeed
        result = await limiter.acquire()
        assert result is True
        
        # Second request should be delayed (but we won't wait for it in test)
        # This is a simplified test - in practice, acquire() would sleep
        assert limiter.tokens < 1

class TestCircuitBreaker:
    """Test circuit breaker functionality"""
    
    def test_circuit_breaker_starts_closed(self):
        """Test that circuit breaker starts in CLOSED state"""
        breaker = CircuitBreaker(failure_threshold=3)
        assert breaker.state == 'CLOSED'
        assert breaker.can_execute() is True
    
    def test_circuit_breaker_opens_after_failures(self):
        """Test that circuit breaker opens after threshold failures"""
        breaker = CircuitBreaker(failure_threshold=3)
        
        # Record failures
        for _ in range(3):
            breaker.record_failure()
        
        assert breaker.state ==
