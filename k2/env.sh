#!/usr/bin/env bash
# Copy to env.sh and edit values, then: source env.sh

export BINANCE_AI_KEY=""
export BINANCE_API_SECRET=""

export TELEGRAM_BOT_TOKEN="8035191346:AAFIICfxI_DYRUIGpX9DFUWdxRaGcG4Nc5Q"
export TELEGRAM_CHAT_ID="1876379484"

export SYMBOL="BNBUSDT"
export INTERVAL="15m"
export RSI_PERIOD="21"
export ENTRY_RSI="26"
export EXIT_RSI="80"
export STOP_PCT="0.08"
export FEE_RATE="0.001"

# Use either ALLOCATION_PCT or ALLOCATION_USDT
export ALLOCATION_PCT="0.95"
export ALLOCATION_USDT="0"

export USE_LIMIT_MAKER="0"
export PRICE_BUFFER_BPS="5"

export LIVE_TRADING="0"
# Default state path is ~/.k2_rsi_bot/bot_state.json if left empty
# export STATE_PATH="/absolute/path/to/bot_state.json"
export POLL_INTERVAL_SEC="60"
export KLINES_LIMIT="200"

# Optional custom base URL (e.g., testnet)
# export BINANCE_BASE_URL="https://testnet.binance.vision"
