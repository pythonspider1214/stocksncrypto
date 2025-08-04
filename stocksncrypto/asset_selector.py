import requests
import logging
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import time
from api_manager import api_manager, api_retry

class AssetSelector:
    """Intelligent asset selection for crypto and stocks"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.cache_duration = 24  # hours
        self.last_update = {}
        
    @api_retry(max_retries=3)
    def get_top_cryptos(self, count: int = 20, criteria: str = "market_cap") -> List[str]:
        """Get top cryptocurrencies with better quality filtering"""
        try:
            self.logger.info(f"Fetching top {count} cryptos by {criteria}...")
            
            url = "https://api.coingecko.com/api/v3/coins/markets"
            params = {
                'vs_currency': 'usd',
                'order': f'{criteria}_desc' if criteria != 'price_change' else 'price_change_percentage_24h_desc',
                'per_page': count * 3,  # Get more to filter better
                'page': 1,
                'sparkline': False,
                'price_change_percentage': '24h'
            }
            
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            # Enhanced filtering
            excluded_keywords = ['usd', 'usdt', 'usdc', 'busd', 'dai', 'tusd', 'pax', 'frax', 'lusd']
            low_quality_keywords = ['doge', 'shib', 'pepe', 'floki', 'baby', 'safe', 'moon', 'lambo', 'inu']
            
            filtered_cryptos = []
            for coin in data:
                coin_id = coin['id'].lower()
                coin_name = coin['name'].lower()
                coin_symbol = coin['symbol'].lower()
                current_price = coin.get('current_price', 0)
                market_cap = coin.get('market_cap', 0)
                volume = coin.get('total_volume', 0)
                
                # Quality filters
                is_stablecoin = any(keyword in coin_id or keyword in coin_name for keyword in excluded_keywords)
                is_low_quality = any(keyword in coin_id or keyword in coin_name or keyword in coin_symbol for keyword in low_quality_keywords)
                has_good_metrics = (
                    current_price > 0.001 and  # Not a micro-penny coin
                    market_cap > 100000000 and  # Min $100M market cap
                    volume > 5000000  # Min $5M daily volume
                )
                
                if not is_stablecoin and not is_low_quality and has_good_metrics:
                    filtered_cryptos.append(coin['id'])
                
                if len(filtered_cryptos) >= count:
                    break
            
            self.logger.info(f"Selected {len(filtered_cryptos)} quality cryptos")
            return filtered_cryptos[:count]
            
        except Exception as e:
            self.logger.error(f"Error fetching top cryptos: {e}")
            return self._get_fallback_cryptos(count)
    
    @api_retry(max_retries=3)
    def get_trending_cryptos(self, count: int = 10) -> List[str]:
        """Get trending cryptocurrencies"""
        try:
            self.logger.info(f"Fetching {count} trending cryptos...")
            
            url = "https://api.coingecko.com/api/v3/search/trending"
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            trending_ids = []
            for coin_data in data.get('coins', []):
                coin_id = coin_data.get('item', {}).get('id')
                if coin_id:
                    trending_ids.append(coin_id)
                
                if len(trending_ids) >= count:
                    break
            
            self.logger.info(f"Found {len(trending_ids)} trending cryptos")
            return trending_ids
            
        except Exception as e:
            self.logger.error(f"Error fetching trending cryptos: {e}")
            return []
    
    @api_retry(max_retries=3)
    def get_top_stocks_by_volume(self, count: int = 20) -> List[str]:
        """Get most actively traded stocks"""
        try:
            self.logger.info(f"Fetching top {count} stocks by volume...")
            
            # Use Yahoo Finance screener for most active stocks
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            params = {
                'formatted': 'true',
                'crumb': 'mock',
                'lang': 'en-US',
                'region': 'US',
                'scrIds': 'most_actives',
                'count': count * 2
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            stocks = []
            quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
            
            for quote in quotes:
                symbol = quote.get('symbol', '')
                market_cap = quote.get('marketCap', {}).get('raw', 0)
                
                # Filter for quality stocks
                if (symbol and 
                    len(symbol) <= 5 and  # Reasonable symbol length
                    not '.' in symbol and  # No OTC or foreign stocks
                    market_cap > 1000000000):  # Min $1B market cap
                    
                    stocks.append(symbol)
                
                if len(stocks) >= count:
                    break
            
            self.logger.info(f"Selected {len(stocks)} active stocks: {stocks[:5]}...")
            return stocks[:count]
            
        except Exception as e:
            self.logger.error(f"Error fetching active stocks: {e}")
            return self._get_fallback_stocks(count)
    
    @api_retry(max_retries=3)
    def get_top_stocks_by_performance(self, count: int = 10, timeframe: str = "1d") -> List[str]:
        """Get top performing stocks"""
        try:
            self.logger.info(f"Fetching top {count} performing stocks ({timeframe})...")
            
            # Use Yahoo Finance gainers
            url = "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
            params = {
                'formatted': 'true',
                'scrIds': 'day_gainers',
                'count': count * 2
            }
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(url, params=params, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            stocks = []
            quotes = data.get('finance', {}).get('result', [{}])[0].get('quotes', [])
            
            for quote in quotes:
                symbol = quote.get('symbol', '')
                change_percent = quote.get('regularMarketChangePercent', {}).get('raw', 0)
                
                # Filter for reasonable gainers (avoid penny stocks)
                if (symbol and 
                    len(symbol) <= 5 and
                    not '.' in symbol and
                    0 < change_percent < 50):  # Reasonable gains (not manipulation)
                    
                    stocks.append(symbol)
                
                if len(stocks) >= count:
                    break
            
            self.logger.info(f"Selected {len(stocks)} top performers")
            return stocks[:count]
            
        except Exception as e:
            self.logger.error(f"Error fetching top performers: {e}")
            return []
    
    def get_mixed_crypto_portfolio(self, total_count: int = 20) -> List[str]:
        """Create a mixed portfolio of cryptos with enhanced quality filtering"""
        try:
            self.logger.info(f"Creating mixed crypto portfolio of {total_count} assets...")
            
            # Enhanced allocation strategy
            top_market_cap = max(1, int(total_count * 0.7))      # 70% top market cap (more stable)
            trending = max(1, int(total_count * 0.2))            # 20% trending  
            high_volume = max(1, int(total_count * 0.1))         # 10% high volume
            
            portfolio = set()
            
            # Get top market cap cryptos (most reliable)
            top_coins = self.get_top_cryptos(top_market_cap, "market_cap")
            portfolio.update(top_coins)
            
            # Add trending cryptos (but filter quality)
            trending_coins = self.get_trending_cryptos(trending)
            # Filter trending coins for quality
            quality_trending = []
            for coin in trending_coins:
                if not any(bad_word in coin.lower() for bad_word in ['lambo', 'moon', 'baby', 'safe', 'doge', 'shib']):
                    quality_trending.append(coin)
            portfolio.update(quality_trending[:trending])
            
            # Add high volume cryptos
            volume_coins = self.get_top_cryptos(high_volume, "volume")
            portfolio.update(volume_coins)
            
            # Convert to list and pad if needed with more top market cap
            portfolio_list = list(portfolio)
            
            # If we need more, add from top 50 market cap (but exclude meme coins)
            if len(portfolio_list) < total_count:
                additional_needed = total_count - len(portfolio_list)
                additional_coins = self.get_top_cryptos(50, "market_cap")  # Get top 50
                
                for coin in additional_coins:
                    if (coin not in portfolio_list and 
                        not any(bad_word in coin.lower() for bad_word in ['lambo', 'moon', 'baby', 'safe', 'meme', 'pepe', 'floki'])):
                        portfolio_list.append(coin)
                        if len(portfolio_list) >= total_count:
                            break
            
            result = portfolio_list[:total_count]
            self.logger.info(f"Created crypto portfolio: {len(result)} quality assets")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating mixed crypto portfolio: {e}")
            return self._get_fallback_cryptos(total_count)
    
    def get_mixed_stock_portfolio(self, total_count: int = 20) -> List[str]:
        """Create a mixed portfolio of stocks with different strategies"""
        try:
            self.logger.info(f"Creating mixed stock portfolio of {total_count} assets...")
            
            # Allocation strategy
            most_active = max(1, int(total_count * 0.7))         # 70% most active
            top_performers = max(1, int(total_count * 0.3))      # 30% top performers
            
            portfolio = set()
            
            # Get most active stocks
            active_stocks = self.get_top_stocks_by_volume(most_active)
            portfolio.update(active_stocks)
            
            # Add top performers
            performer_stocks = self.get_top_stocks_by_performance(top_performers)
            portfolio.update(performer_stocks)
            
            # Convert to list
            portfolio_list = list(portfolio)
            
            # Pad if needed with more active stocks
            if len(portfolio_list) < total_count:
                additional_needed = total_count - len(portfolio_list)
                additional_stocks = self.get_top_stocks_by_volume(most_active + additional_needed)
                for stock in additional_stocks:
                    if stock not in portfolio_list:
                        portfolio_list.append(stock)
                        if len(portfolio_list) >= total_count:
                            break
            
            result = portfolio_list[:total_count]
            self.logger.info(f"Created stock portfolio: {len(result)} assets")
            return result
            
        except Exception as e:
            self.logger.error(f"Error creating mixed stock portfolio: {e}")
            return self._get_fallback_stocks(total_count)
    
    def _get_fallback_cryptos(self, count: int) -> List[str]:
        """Fallback crypto list if API fails"""
        fallback = [
            "bitcoin", "ethereum", "binancecoin", "solana", "cardano",
            "ripple", "polkadot", "dogecoin", "avalanche-2", "chainlink",
            "polygon", "litecoin", "stellar", "vechain", "algorand",
            "cosmos", "internet-computer", "filecoin", "hedera-hashgraph",
            "theta-token", "tron", "ethereum-classic", "monero", "near"
        ]
        return fallback[:count]
    
    def _get_fallback_stocks(self, count: int) -> List[str]:
        """Fallback stock list if API fails"""
        fallback = [
            "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX",
            "AMD", "INTC", "JPM", "V", "MA", "JNJ", "PG", "UNH", "HD", "DIS",
            "PYPL", "CRM", "BAC", "WMT", "KO", "PFE", "XOM", "VZ", "T", "CSCO"
        ]
        return fallback[:count]
    
    def should_refresh_assets(self, last_update_time: Optional[datetime] = None) -> bool:
        """Check if assets should be refreshed"""
        if last_update_time is None:
            return True
        
        time_diff = datetime.now() - last_update_time
        return time_diff > timedelta(hours=self.cache_duration)
    
    def get_auto_selected_assets(self, crypto_count: int = 20, stock_count: int = 20) -> Dict[str, List[str]]:
        """Main method to get automatically selected assets"""
        try:
            self.logger.info("🤖 Auto-selecting assets...")
            
            # Get mixed portfolios
            crypto_list = self.get_mixed_crypto_portfolio(crypto_count)
            stock_list = self.get_mixed_stock_portfolio(stock_count)
            
            result = {
                'crypto_list': crypto_list,
                'stock_list': stock_list,
                'last_updated': datetime.now().isoformat(),
                'selection_strategy': 'mixed_portfolio'
            }
            
            self.logger.info(f"✅ Auto-selected {len(crypto_list)} cryptos and {len(stock_list)} stocks")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in auto-selection: {e}")
            # Return fallback
            return {
                'crypto_list': self._get_fallback_cryptos(crypto_count),
                'stock_list': self._get_fallback_stocks(stock_count),
                'last_updated': datetime.now().isoformat(),
                'selection_strategy': 'fallback'
            }

# Global asset selector
asset_selector = AssetSelector()
