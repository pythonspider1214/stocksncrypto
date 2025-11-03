# Copy to env.ps1 and edit values, then: .\env.ps1

$env:BINANCE_API_KEY = ""
$env:BINANCE_API_SECRET = ""

$env:TELEGRAM_BOT_TOKEN = ""
$env:TELEGRAM_CHAT_ID = ""

$env:SYMBOL = "BNBUSDT"
$env:INTERVAL = "15m"
$env:RSI_PERIOD = "21"
$env:ENTRY_RSI = "26"
$env:EXIT_RSI = "80"
$env:STOP_PCT = "0.08"
$env:FEE_RATE = "0.001"

# Use either ALLOCATION_PCT or ALLOCATION_USDT
$env:ALLOCATION_PCT = "0.95"
$env:ALLOCATION_USDT = "0"

$env:USE_LIMIT_MAKER = "0"
$env:PRICE_BUFFER_BPS = "5"

$env:LIVE_TRADING = "0"
# Default state path is %USERPROFILE%\.k2_rsi_bot\bot_state.json if left empty
# $env:STATE_PATH = "D:\path\to\bot_state.json"
$env:POLL_INTERVAL_SEC = "5"
$env:KLINES_LIMIT = "200"

# Optional custom base URL (e.g., testnet)
# $env:BINANCE_BASE_URL = "https://testnet.binance.vision"
