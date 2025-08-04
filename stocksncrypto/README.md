# Crypto & Stock Tracker

A comprehensive Python application for tracking cryptocurrencies and stocks, performing technical and sentiment analysis, and managing your portfolio with intelligent asset selection and machine learning.

## Features
- **Crypto & Stock Price Tracking:** Real-time price fetching for your selected assets.
- **Portfolio Management:** Add, view, and export your holdings and transactions.
- **Technical Analysis:** Calculate indicators like SMA and RSI, and generate buy/sell/hold signals.
- **Sentiment Analysis:** Analyze news sentiment for stocks (mock/TextBlob-based).
- **Machine Learning:** Train and use ML models for asset signal prediction.
- **Intelligent Asset Selection:** Auto-select trending, top, or mixed assets for both crypto and stocks.
- **Auto & Manual Asset Refresh:** Automatically or manually update your tracked asset lists.
- **Safe Logging:** Unicode-safe logging to both file and console, with emoji support in the console.

## How It Works
- The main app is in `tracker.py`. It loads configuration, sets up logging, and starts the main loop.
- You interact with the app via a menu: view portfolio, refresh assets, train ML, see summary, etc.
- Asset selection can be automatic (based on config and time) or manual (on demand).
- All logs are written to `crypto_tracker.log` (with emoji-safe encoding).
- All configuration is managed via `config.json` and `.env` for secrets.

## Setup
1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Configure your `.env` and `config.json`** as needed.
3. **Run the tracker:**
   ```sh
   python tracker.py
   ```

## Logging
- Logging is set up via `logging_config.py` for both file and console output.
- Use the provided `safe_log_info`, `safe_log_error`, and `safe_log_warning` wrappers for Unicode-safe logging.

## Project Structure
- `tracker.py` - Main application logic and menu
- `config_manager.py` - Configuration management
- `portfolio_manager.py` - Portfolio and transaction logic
- `technical_analysis.py` - Technical indicators
- `sentiment_analysis.py` - Sentiment analysis
- `ml_models.py` - Machine learning models
- `asset_selector.py` - Intelligent asset selection
- `utils.py` - Utility functions
- `logging_config.py` - Logging setup
- `.env`, `config.json`, `requirements.txt` - Configuration and dependencies

---

**In summary:**
This project is a robust, modular, and extensible tracker for crypto and stocks, with advanced analytics, portfolio management, and safe logging for both file and console. All logging is handled in a way that avoids Unicode/emoji issues in log files, while still providing a rich user experience in the terminal.
