# StockSnCrypto: AI-Enhanced Trading Toolkit 📈🤖

A multi-project repository containing intelligent tools for automated trading in stocks and cryptocurrencies.

## 🧩 Projects in This Repo

### 📦 1. CryptoTracker/
A tool that:
- Fetches crypto prices (via CoinGecko, Yahoo Finance)
- Applies technical indicators (MACD, RSI, MA, Bollinger)
- Uses a trained ML model to generate buy/sell signals
- Stores data in a SQLite portfolio

### 📦 2. stockbot-/
An experimental project that focuses on:
- Backtesting strategies
- Modular trade logging
- (Planned) integration with live trading APIs like Binance or IBKR

### 🔄 tracker.py
A standalone script that:
- Ties everything together
- Runs daily scans and logs opportunities
- Can be used in cron or daemon mode

## 📊 Features

- ✅ Multi-timeframe analysis
- ✅ ML signal classification
- ✅ Auto-logging to SQLite
- ✅ Configurable risk management

## 🔧 Requirements

Install dependencies:

```bash
pip install -r requirements.txt
