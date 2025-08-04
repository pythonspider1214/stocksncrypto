import os
import asyncio
import aiohttp
import requests
import time
import json
import logging
from datetime import datetime
from typing import Dict, List
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import our modules
from utils import setup_logging, clear_screen, format_currency, format_percentage
from config_manager import ConfigManager
from api_manager import api_manager, api_retry, AsyncAPIClient
from portfolio_manager import PortfolioTracker
from technical_analysis import TechnicalIndicators
from sentiment_analysis import SentimentAnalyzer
from ml_models import ml_manager

# --- Safe Logger Wrapper ---
def safe_log_info(logger, message):
    """Safe logging that handles Unicode issues"""
    clean_message = message.encode('ascii', errors='replace').decode('ascii')
    logger.info(clean_message)
    print(message)  # Original message to console

def safe_log_error(logger, message):
    """Safe error logging"""
    clean_message = message.encode('ascii', errors='replace').decode('ascii')
    logger.error(clean_message)
    print(f"❌ {message}")

def safe_log_warning(logger, message):
    """Safe warning logging"""
    clean_message = message.encode('ascii', errors='replace').decode('ascii')
    logger.warning(clean_message)
    print(f"⚠️ {message}")
# --- End Safe Logger Wrapper ---

# Setup logging
setup_logging(
    log_level=os.getenv('LOG_LEVEL', 'INFO'),
    log_file=os.getenv('LOG_FILE', 'crypto_tracker.log')
)

logger = logging.getLogger(__name__)

class CryptoTracker:
    """Main crypto tracker application"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.config
        self.portfolio = PortfolioTracker()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_indicators = TechnicalIndicators()
        self.logger = logger
        
        logger.info("Crypto Tracker initialized")
        logger.info(f"Tracking {len(self.config.crypto_list)} cryptos and {len(self.config.stock_list)} stocks")
    
    @api_retry(max_retries=3)
    def get_crypto_prices(self, crypto_ids: List[str]) -> Dict[str, any]:
        """Fetch cryptocurrency prices"""
        try:
            ids = ",".join(crypto_ids)
            vs_currencies = "usd,aud,eur"
            url = f"https://api.coingecko.com/api/v3/simple/price?ids={ids}&vs_currencies={vs_currencies}"
            
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            logger.info(f"Fetched crypto prices for {len(crypto_ids)} coins")
            return data
        except Exception as e:
            logger.error(f"Error fetching crypto prices: {e}")
            return {}
    
    @api_retry(max_retries=3)
    def get_stock_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Fetch stock prices using yfinance"""
        from utils import safe_yf_ticker_history
        
        stock_prices = {}
        for symbol in symbols:
            try:
                df = safe_yf_ticker_history(symbol, period="1d")
                if df is not None and 'Close' in df.columns and not df.empty:
                    stock_prices[symbol] = float(df['Close'].iloc[-1])
                    logger.debug(f"Fetched price for {symbol}: ${stock_prices[symbol]:.2f}")
                else:
                    logger.warning(f"No valid close price for {symbol}")
                    stock_prices[symbol] = None
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
                stock_prices[symbol] = None
        
        logger.info(f"Fetched stock prices for {len([s for s in stock_prices.values() if s is not None])} stocks")
        return stock_prices
    
    def get_stock_analysis(self, symbol: str) -> str:
        """Get comprehensive stock analysis"""
        try:
            from utils import safe_yf_ticker_history, get_stock_info_cached
            
            # Get price data
            df = safe_yf_ticker_history(symbol, period="6mo")
            if df is None or len(df) < 30:
                return f"{symbol}: Insufficient data"
            
            # Technical analysis
            df = self.technical_indicators.calculate_indicators(df)
            latest = df.iloc[-1]
            
            # ML signal
            ml_signal = ml_manager.generate_signal(df)
            ml_signals = {1: 'BUY', 0: 'HOLD', -1: 'SELL', None: 'N/A'}
            ml_str = ml_signals.get(ml_signal, 'N/A')
            
            # Get stock info
            try:
                info = get_stock_info_cached(symbol)
                stock_name = info.get('longName') or info.get('shortName') or symbol
            except:
                stock_name = symbol
            
            # Determine country
            country = 'AU' if symbol.endswith('.AX') else 'US'
            
            # Technical signals
            tech_analysis = self.technical_indicators.generate_signals(symbol)
            tech_recommendation = tech_analysis.get('recommendation', 'HOLD')
            
            # Sentiment analysis
            sentiment = self.sentiment_analyzer.get_news_sentiment(symbol)
            sentiment_str = sentiment.get('overall_sentiment', 'Neutral')
            
            return (f"[{country}] {symbol} ({stock_name}): "
                   f"${latest['Close']:.2f} | "
                   f"SMA20: ${latest.get('SMA20', 0):.2f} | "
                   f"RSI: {latest.get('RSI14', 0):.1f} | "
                   f"Tech: {tech_recommendation} | "
                   f"ML: {ml_str} | "
                   f"Sentiment: {sentiment_str}")
                   
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
            return f"{symbol}: Analysis error - {str(e)}"
    
    def save_crypto_prices_to_csv(self, filename: str, timestamp: str, crypto_prices: Dict[str, any]):
        """Save crypto prices to CSV"""
        try:
            import csv
            file_exists = os.path.isfile(filename)
            
            with open(filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(['Timestamp', 'Crypto', 'USD', 'AUD', 'EUR'])
                
                for crypto, prices in crypto_prices.items():
                    if isinstance(prices, dict):
                        writer.writerow([
                            timestamp,
                            crypto.capitalize(),
                            prices.get('usd', 'N/A'),
                            prices.get('aud', 'N/A'),
                            prices.get('eur', 'N/A')
                        ])
            
            logger.info(f"Saved crypto prices to {filename}")
        except Exception as e:
            logger.error(f"Error saving to CSV: {e}")
    
    def display_crypto_prices(self, crypto_data: Dict[str, any]):
        """Display cryptocurrency prices"""
        print("\n" + "="*50)
        print("💰 CRYPTOCURRENCY PRICES")
        print("="*50)
        
        for coin in self.config.crypto_list:
            data = crypto_data.get(coin)
            if data and isinstance(data, dict):
                usd = data.get('usd', 0)
                aud = data.get('aud', 0)
                eur = data.get('eur', 0)
                print(f"{coin.upper():>12}: USD ${usd:>8.2f} | AUD ${aud:>8.2f} | EUR €{eur:>8.2f}")
            else:
                print(f"{coin.upper():>12}: Data unavailable")
    
    def display_stock_analysis(self):
        """Display stock analysis for ALL configured stocks"""
        print("\n" + "="*80)
        print("📈 STOCK ANALYSIS & INDICATORS")
        print("="*80)
        
        total_stocks = len(self.config.stock_list)
        print(f"Analyzing all {total_stocks} configured stocks...\n")
        
        # Process ALL stocks, not just a subset
        for i, symbol in enumerate(self.config.stock_list, 1):
            try:
                analysis = self.get_stock_analysis(symbol)
                print(f"{i:2d}. {analysis}")
                
                # Add small delay every 5 stocks to avoid rate limiting
                if i % 5 == 0 and i < total_stocks:
                    print(f"    ... processed {i}/{total_stocks} stocks, continuing...")
                    time.sleep(0.5)
                    
            except Exception as e:
                print(f"{i:2d}. {symbol}: Analysis error - {str(e)}")
        
        print(f"\n📊 Completed analysis of all {total_stocks} stocks")
    
    def display_portfolio_summary(self, current_prices: Dict[str, float]):
        """Display portfolio summary"""
        try:
            positions = self.portfolio.get_current_positions(current_prices)
            if not positions:
                print("\n📊 PORTFOLIO: No positions")
                return
            
            metrics = self.portfolio.calculate_portfolio_metrics(positions)
            
            print("\n" + "="*50)
            print("📊 PORTFOLIO SUMMARY")
            print("="*50)
            print(f"Total Value: {format_currency(metrics.total_value)}")
            print(f"Total Cost:  {format_currency(metrics.total_cost)}")
            print(f"P&L:         {format_currency(metrics.total_pnl)} ({format_percentage(metrics.total_pnl_percent)})")
            print(f"Daily Return: {format_percentage(metrics.daily_return)}")
            
            if positions:
                print(f"\nTop Holdings:")
                sorted_positions = sorted(positions, key=lambda p: p.market_value, reverse=True)
                for pos in sorted_positions[:5]:
                    print(f"  {pos.symbol}: {format_currency(pos.market_value)} ({format_percentage(pos.unrealized_pnl_percent)})")
                    
        except Exception as e:
            logger.error(f"Error displaying portfolio: {e}")
            print(f"📊 PORTFOLIO: Error - {str(e)}")
    
    def interactive_menu(self):
        """Enhanced interactive menu"""
        print("\n" + "="*50)
        print("🎯 CRYPTO TRACKER MENU")
        print("="*50)
        print("Commands:")
        print("  'p' or 'portfolio' - Portfolio menu")
        print("  'r' or 'refresh'   - Auto-refresh assets")
        print("  'a' or 'auto'      - Manual asset selection")
        print("  't' or 'train'     - Train ML model")
        print("  's' or 'summary'   - Show tracking summary")
        print("  'd' or 'detail'    - Detailed asset analysis")
        print("  'q' or 'quality'   - Check asset quality")  # Updated wording
        print("  'c' or 'clear'     - Clear screen")
        print("  'quit'             - Quit application")
        print("  Enter              - Continue tracking")
        
        command = input("\nEnter command: ").strip().lower()
        
        if command in ['p', 'portfolio']:
            self.portfolio_menu()
        elif command in ['r', 'refresh']:
            self.auto_refresh_assets()
        elif command in ['a', 'auto']:
            self.manual_refresh_assets()
        elif command in ['t', 'train']:
            self.train_ml_model()
        elif command in ['s', 'summary']:
            self.show_tracking_summary()
        elif command in ['d', 'detail']:
            self.show_detailed_summary()
        elif command in ['q', 'quality']:
            self.check_asset_quality()
        elif command in ['c', 'clear']:
            clear_screen()
        elif command in ['quit']:
            return False
        
        return True
    
    def portfolio_menu(self):
        """Portfolio management menu"""
        while True:
            print("\n" + "="*40)
            print("📊 PORTFOLIO MENU")
            print("="*40)
            print("1. View Holdings")
            print("2. Add Transaction")
            print("3. Export to CSV")
            print("4. Back to Main Menu")
            
            choice = input("Choose option (1-4): ").strip()
            
            if choice == '1':
                # Get current prices for portfolio calculation
                crypto_prices = self.get_crypto_prices(self.config.crypto_list)
                stock_prices = self.get_stock_prices(self.config.stock_list)
                
                # Combine prices
                all_prices = {}
                for coin, data in crypto_prices.items():
                    if isinstance(data, dict):
                        all_prices[coin.upper()] = data.get('usd', 0)
                
                for symbol, price in stock_prices.items():
                    if price is not None:
                        all_prices[symbol] = price
                
                self.display_portfolio_summary(all_prices)
                
            elif choice == '2':
                self.add_transaction_menu()
                
            elif choice == '3':
                try:
                    self.portfolio.export_to_csv()
                    print("✅ Portfolio exported to CSV successfully!")
                except Exception as e:
                    print(f"❌ Export failed: {e}")
                    
            elif choice == '4':
                break
            else:
                print("Invalid choice. Please try again.")
    
    def add_transaction_menu(self):
        """Add transaction menu"""
        try:
            print("\n" + "="*40)
            print("💰 ADD TRANSACTION")
            print("="*40)
            
            symbol = input("Symbol (e.g., AAPL, bitcoin): ").strip().upper()
            asset_type = input("Asset type (stock/crypto): ").strip().lower()
            action = input("Action (buy/sell): ").strip().lower()
            quantity = float(input("Quantity: "))
            price = float(input("Price: "))
            fees = float(input("Fees (optional, default 0): ") or "0")
            
            self.portfolio.add_transaction(symbol, asset_type, action, quantity, price, fees)
            print("✅ Transaction added successfully!")
            
        except Exception as e:
            print(f"❌ Error adding transaction: {e}")
    
    def refresh_stock_lists(self):
        """Refresh active stock lists"""
        print("🔄 Refreshing stock lists...")
        try:
            # This is a simplified version - you can expand with actual API calls
            new_stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"]
            self.config.stock_list = new_stocks
            self.config_manager.save_config()
            print(f"✅ Updated stock list with {len(new_stocks)} stocks")
        except Exception as e:
            print(f"❌ Error refreshing stock lists: {e}")
    
    def train_ml_model(self):
        """Train ML model"""
        print("🤖 Training ML model...")
        try:
            success = ml_manager.train_model('AAPL')
            if success:
                print("✅ ML model trained successfully!")
            else:
                print("❌ ML model training failed!")
        except Exception as e:
            print(f"❌ Error training ML model: {e}")
    
    def auto_refresh_assets(self):
        """Automatically refresh assets if needed"""
        try:
            if self.config_manager.should_refresh_assets():
                safe_log_info(self.logger, "🤖 Auto-refreshing assets...")
                print("🤖 Auto-selecting new assets...")
                
                from asset_selector import asset_selector
                
                # Get auto-selected assets
                selected_assets = asset_selector.get_auto_selected_assets(
                    crypto_count=self.config.auto_selection.crypto_count,
                    stock_count=self.config.auto_selection.stock_count
                )
                
                # Update configuration
                self.config_manager.update_assets_from_auto_selection(selected_assets)
                self.config = self.config_manager.config
                
                print(f"✅ Auto-selected {len(self.config.crypto_list)} cryptos and {len(self.config.stock_list)} stocks")
                print(f"📊 Strategy: {selected_assets['selection_strategy']}")
                
                return True
            return False
        except Exception as e:
            safe_log_error(self.logger, f"Error in auto-refresh: {e}")
            print(f"❌ Auto-refresh failed: {e}")
            return False

    def manual_refresh_assets(self):
        """Manually refresh assets"""
        try:
            print("🤖 Manually refreshing assets...")
            from asset_selector import asset_selector
            # Get auto-selected assets
            selected_assets = asset_selector.get_auto_selected_assets(
                crypto_count=self.config.auto_selection.crypto_count,
                stock_count=self.config.auto_selection.stock_count
            )
            # Show preview
            print(f"\n📊 NEW ASSET SELECTION:")
            print(f"🪙 Cryptos ({len(selected_assets['crypto_list'])}): {', '.join(selected_assets['crypto_list'][:5])}...")
            print(f"📈 Stocks ({len(selected_assets['stock_list'])}): {', '.join(selected_assets['stock_list'][:5])}...")
            confirm = input("\n❓ Apply these changes? (y/n): ")

            if confirm == 'y':
                # Update configuration
                self.config_manager.update_assets_from_auto_selection(selected_assets)
                self.config = self.config_manager.config
                print("✅ Assets updated successfully!")
                return True
            else:
                print("❌ Asset refresh cancelled")
                return False
        except Exception as e:
            safe_log_error(self.logger, f"Error in manual refresh: {e}")
            print(f"❌ Manual refresh failed: {e}")
            return False
    
    def show_tracking_summary(self):
        """Show a summary of the current tracking state"""
        print("\n" + "="*50)
        print("📊 TRACKING SUMMARY")
        print("="*50)
        print(f"Cryptos tracked: {len(self.config.crypto_list)}")
        print(f"Stocks tracked:  {len(self.config.stock_list)}")
        print(f"Auto-selection:  {'✅ Enabled' if self.config.auto_selection.enabled else '❌ Disabled'}")
        last_update = getattr(self.config, 'last_asset_update', None)
        if last_update:
            print(f"Last asset update: {last_update}")
        print(f"Update interval:  {self.config.update_interval} seconds")
        # Portfolio value summary
        try:
            crypto_prices = self.get_crypto_prices(self.config.crypto_list)
            stock_prices = self.get_stock_prices(self.config.stock_list)
            all_prices = {}
            for coin, data in crypto_prices.items():
                if isinstance(data, dict):
                    all_prices[coin.upper()] = data.get('usd', 0)
            for symbol, price in stock_prices.items():
                if price is not None:
                    all_prices[symbol] = price
            positions = self.portfolio.get_current_positions(all_prices)
            metrics = self.portfolio.calculate_portfolio_metrics(positions)
            print(f"Portfolio Value: {format_currency(metrics.total_value)}")
        except Exception as e:
            safe_log_error(self.logger, f"Portfolio Value: N/A ({e})")
            print(f"Portfolio Value: N/A ({e})")
        print("="*50)
    
    def show_detailed_summary(self):
        """Show detailed asset information"""
        print("\n" + "="*60)
        print("🔍 DETAILED ASSET ANALYSIS")
        print("="*60)
        
        print(f"📋 CONFIGURED ASSETS:")
        print(f"  🪙 Crypto List ({len(self.config.crypto_list)}): {self.config.crypto_list}")
        print(f"  📈 Stock List ({len(self.config.stock_list)}): {self.config.stock_list}")
        
        print(f"\n📊 AUTO-SELECTION CONFIG:")
        print(f"  Enabled: {self.config.auto_selection.enabled}")
        print(f"  Crypto Count Target: {self.config.auto_selection.crypto_count}")
        print(f"  Stock Count Target: {self.config.auto_selection.stock_count}")
        print(f"  Last Update: {self.config.last_asset_update}")
        
        # Count actual valid prices
        try:
            crypto_data = self.get_crypto_prices(self.config.crypto_list)
            stock_data = self.get_stock_prices(self.config.stock_list)
            
            valid_cryptos = sum(1 for coin, data in crypto_data.items() 
                               if isinstance(data, dict) and data.get('usd', 0) > 0)
            valid_stocks = sum(1 for symbol, price in stock_data.items() 
                              if price is not None and price > 0)
            
            print(f"\n✅ VALID DATA:")
            print(f"  🪙 Valid Cryptos: {valid_cryptos}/{len(self.config.crypto_list)}")
            print(f"  📈 Valid Stocks: {valid_stocks}/{len(self.config.stock_list)}")
            
        except Exception as e:
            print(f"❌ Error checking data: {e}")
    
    def check_asset_quality(self):
        """Check and report on current asset quality"""
        print("\n" + "="*60)
        print("🔍 ASSET QUALITY ANALYSIS")
        print("="*60)
        
        # Check crypto quality
        print("\n🪙 CRYPTO QUALITY CHECK:")
        crypto_data = self.get_crypto_prices(self.config.crypto_list)
        
        low_quality_cryptos = []
        good_cryptos = []
        
        for coin, data in crypto_data.items():
            if isinstance(data, dict):
                price = data.get('usd', 0)
                if price < 0.01:  # Very low price (potential quality issue)
                    low_quality_cryptos.append(f"{coin}: ${price}")
                elif any(word in coin.lower() for word in ['lambo', 'moon', 'baby', 'safe', 'meme']):
                    low_quality_cryptos.append(f"{coin}: Meme/speculative coin")
                else:
                    good_cryptos.append(coin)
        
        print(f"  ✅ Good Quality: {len(good_cryptos)}/{len(self.config.crypto_list)}")
        print(f"  ⚠️ Questionable: {len(low_quality_cryptos)}/{len(self.config.crypto_list)}")
        
        if low_quality_cryptos:
            print("\n  🚨 Low Quality Assets:")
            for asset in low_quality_cryptos:
                print(f"    • {asset}")
        
        # Check stock quality
        print(f"\n📈 STOCK QUALITY CHECK:")
        stock_data = self.get_stock_prices(self.config.stock_list)
        
        penny_stocks = []
        good_stocks = []
        
        for symbol, price in stock_data.items():
            if price and price < 5:  # Penny stock territory
                penny_stocks.append(f"{symbol}: ${price:.2f}")
            elif price and price > 5:
                good_stocks.append(symbol)
        
        print(f"  ✅ Good Quality: {len(good_stocks)}/{len(self.config.stock_list)}")
        print(f"  ⚠️ Penny Stocks: {len(penny_stocks)}/{len(self.config.stock_list)}")
        
        if penny_stocks:
            print("\n  🚨 Penny Stocks:")
            for stock in penny_stocks:
                print(f"    • {stock}")
        
        # Recommendation
        if low_quality_cryptos or penny_stocks:
            print(f"\n💡 RECOMMENDATION:")
            print(f"  Consider refreshing assets to improve quality")
            print(f"  Type 'a' to manually refresh asset selection")
    
    def run(self):
        """Main application loop with auto-refresh"""
        logger.info("Starting Crypto Tracker...")
        print("🚀 CRYPTO TRACKER STARTED")

        # Auto-refresh assets on startup if needed
        self.auto_refresh_assets()

        print(f"Tracking {len(self.config.crypto_list)} cryptos and {len(self.config.stock_list)} stocks")
        print(f"Update interval: {self.config.update_interval} seconds")
        print(f"Auto-selection: {'✅ Enabled' if self.config.auto_selection.enabled else '❌ Disabled'}")

        try:
            while True:
                # Auto-refresh check
                if self.auto_refresh_assets():
                    print("🔄 Assets refreshed automatically!")

                # Get current timestamp
                now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"\n🕐 {now}")
                
                # Fetch crypto prices
                crypto_data = self.get_crypto_prices(self.config.crypto_list)
                self.display_crypto_prices(crypto_data)
                
                # Save crypto prices to CSV
                self.save_crypto_prices_to_csv("crypto_prices.csv", now, crypto_data)
                
                # Display stock analysis
                self.display_stock_analysis()
                
                # Get current prices for portfolio
                stock_prices = self.get_stock_prices(self.config.stock_list)
                all_prices = {}
                
                # Combine crypto and stock prices
                for coin, data in crypto_data.items():
                    if isinstance(data, dict):
                        all_prices[coin.upper()] = data.get('usd', 0)
                
                for symbol, price in stock_prices.items():
                    if price is not None:
                        all_prices[symbol] = price
                
                # Display portfolio summary
                self.display_portfolio_summary(all_prices)
                
                # Interactive menu
                if not self.interactive_menu():
                    break
                
                # Wait for next update
                print(f"\n⏳ Refreshing in {self.config.update_interval} seconds... (Press Ctrl+C to stop)")
                time.sleep(self.config.update_interval)
                
        except KeyboardInterrupt:
            print("\n\n👋 Crypto Tracker stopped by user")
            logger.info("Crypto Tracker stopped by user")
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            logger.error(f"Unexpected error: {e}")
        finally:
            logger.info("Crypto Tracker shutdown complete")

def main():
    """Entry point"""
    try:
        tracker = CryptoTracker()
        tracker.run()
    except Exception as e:
        print(f"❌ Failed to start Crypto Tracker: {e}")
        logger.error(f"Failed to start: {e}")

if __name__ == "__main__":
    main()
