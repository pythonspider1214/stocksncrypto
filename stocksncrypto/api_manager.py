import os
import logging
import aiohttp
import requests
from typing import Dict, Optional, Any
from functools import wraps
import time

class APIKeyManager:
    """Secure API key management"""
    
    def __init__(self):
        self.fallback_keys = {
            'finnhub': 'd18j33hr01qg5218ces0d18j33hr01qg5218cesg',
        }
        self.warned_keys = set()
        self.logger = logging.getLogger(__name__)
    
    def get_key(self, service: str, required: bool = False) -> Optional[str]:
        """Get API key with secure fallback"""
        env_var = f'{service.upper()}_API_KEY'
        api_key = os.getenv(env_var)
        
        if api_key and api_key != 'your_key_here':
            return api_key
        
        if service.lower() in self.fallback_keys:
            if service not in self.warned_keys:
                self.logger.warning(f"⚠️  Using fallback {service} API key. Set {env_var} for security!")
                self.warned_keys.add(service)
            return self.fallback_keys[service.lower()]
        
        if required:
            raise ValueError(f"No API key found for {service}. Set {env_var} environment variable.")
        return None

def api_retry(max_retries=3, delay=1, backoff=2):
    """Decorator for API calls with retry logic"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except requests.exceptions.RequestException as e:
                    if attempt < max_retries - 1:
                        wait_time = delay * (backoff ** attempt)
                        time.sleep(wait_time)
                    else:
                        logging.error(f"API call failed after {max_retries} attempts: {e}")
                except Exception as e:
                    logging.error(f"Unexpected error in API call: {e}")
                    break
            return None
        return wrapper
    return decorator

class AsyncAPIClient:
    """Async API client with connection pooling"""
    
    def __init__(self, timeout: int = 30, max_connections: int = 10):
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.connector = aiohttp.TCPConnector(
            limit=max_connections,
            limit_per_host=5,
            keepalive_timeout=30
        )
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=self.timeout
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def get(self, url: str, **kwargs) -> Dict[str, Any]:
        """Async GET request with error handling"""
        try:
            async with self.session.get(url, **kwargs) as response:
                response.raise_for_status()
                return await response.json()
        except Exception as e:
            logging.error(f"Async API call failed for {url}: {e}")
            return {}

# Global API manager
api_manager = APIKeyManager()
