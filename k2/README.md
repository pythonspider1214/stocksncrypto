**Overview**
- Simple RSI backtest on BNB/USDT using Yahoo Finance data (`BNB-USD`), over ~60 days (2 months) at the finest interval Yahoo allows.
- Strategy: enter long when RSI <= 28, exit when RSI >= 55, 10% stop loss.
- Initial capital: $1500. Includes Binance-like fees (default 0.1%).

**Install**
- Python 3.10+
- Create a virtual environment (optional) and install deps:
- `pip install -r requirements.txt`

**Run**
- `python backtest_rsi_bnb.py`

**Options**
- `--period 60d` Range to download (default 60d).
- `--csv bnb_60d_15m.csv` Use a local Yahoo CSV instead of downloading.
- `--ticker BNB-USD` Yahoo symbol (default BNB-USD).
- `--interval 15m` Force a specific interval (optional). If not set, the script tries several from smallest upward.
- `--fee 0.001` Trading fee rate (default 0.001 = 0.1%).
- `--initial 1500` Initial USD capital.
- `--rsi_period 14` RSI lookback period.
- `--entry 28` RSI entry threshold.
- `--exit 55` RSI exit threshold.
- `--stop 0.10` Stop loss percent (0.10 = 10%).
- `--atr_period 14` ATR lookback for volatility filter.
- `--atr_thresh 0.0` ATR% threshold for entries; e.g. 0.005 = 0.5%.

**Optimization mode**
- Find best params quickly over ranges while reusing RSI per period:
  - `python backtest_rsi_bnb.py --csv bnb_60d_15m.csv --optimize --rsi_periods 10,14,21 --entry_range 20,35,2 --exit_range 50,80,5 --stops 0.05,0.08,0.10,0.12 --top_k 10 --results_csv results.csv`
- Flags:
  - `--optimize` Run grid search.
  - `--rsi_periods` Comma list of periods (default `10,14,21`).
  - `--entry_range` Range `start,end,step` inclusive (default `20,35,2`).
  - `--exit_range` Range `start,end,step` inclusive (default `50,80,5`).
  - `--stops` Comma list of stop-loss percents (default `0.05,0.08,0.10,0.12`).
  - `--top_k` How many top rows to display.
  - `--results_csv` Save full grid results to CSV.
  - `--objective` Choose `final`, `return`, or `trades` (with `--min_final`).
  - `--min_final` Minimum final equity when `--objective trades` is set (default `1500`).
  - `--atr_period` ATR period for volatility filter.
  - `--atr_thresh` Only allow entries when ATR/Close >= threshold.

Example: maximize trades on 15m subject to final >= 1500 with ATR filter
- `python backtest_rsi_bnb.py --csv bnb_60d_15m.csv --optimize --objective trades --min_final 1500 --atr_period 14 --atr_thresh 0.005 --rsi_periods 8,10 --entry_range 45,55,5 --exit_range 55,70,5 --stops 0.005,0.01 --top_k 10 --results_csv results_15m_atr_trades.csv`

**Notes**
- Uses Yahoo ticker `BNB-USD` as a proxy for BNB/USDT spot.
- Attempts the smallest allowed interval for a 60d range by trying `2m`, then `5m`, then `15m`.
- If an interval returns no data, it falls back to the next one automatically.
- At the end of the test window, any open position is closed to realize PnL.

**Troubleshooting**
- If you see "Failed to download data":
  - Try forcing a coarser interval, e.g. `--interval 15m` or `--interval 60m`.
  - Reduce the period, e.g. `--period 30d`.
  - Ensure internet access and `yfinance` is up-to-date: `pip install -U yfinance`.

**Live Trading Bot (Binance + Telegram)**
- File: `live_rsi_bot.py` — limit-only live runner for the top config (RSI 21, entry<=26, exit>=80, stop 8%).
- Reads config from environment variables and uses Telegram for notifications.

- Required env vars:
  - `BINANCE_API_KEY`, `BINANCE_API_SECRET` (Binance Spot keys)
  - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` (optional but recommended)

- Optional env vars (defaults in parentheses):
  - `SYMBOL=BNBUSDT`, `INTERVAL=15m`
  - `RSI_PERIOD=21`, `ENTRY_RSI=26`, `EXIT_RSI=80`, `STOP_PCT=0.08`
  - `FEE_RATE=0.001` approximate fee used for PnL receipts
  - `ALLOCATION_PCT=0.95` or `ALLOCATION_USDT=0` (fixed amount)
  - `PRICE_BUFFER_BPS=5` (price offset to improve fills), `USE_LIMIT_MAKER=0`
  - `LIVE_TRADING=0` (set to `1` to place real orders), `BINANCE_BASE_URL` (optional)
  - `STATE_PATH=bot_state.json`, `POLL_INTERVAL_SEC=5`, `KLINES_LIMIT=200`

- Dry-run first (no orders sent):
  - `set LIVE_TRADING=0`
  - `python live_rsi_bot.py`

- Go live (be careful):
  - `set LIVE_TRADING=1`
  - `python live_rsi_bot.py`

Notes:
- The bot acts on closed 15m candles to avoid intra-bar noise.
- Always uses limit (or LIMIT_MAKER) orders; stop-loss uses STOP_LOSS_LIMIT.
- Quantizes prices/quantities per exchange filters and checks min notional.
- Maintains simple JSON state and sends Telegram notifications on key events.

**Run Anywhere with Venv + Env Scripts**
- Copy an env template and edit values:
  - PowerShell: copy `env.example.ps1` to `env.ps1` and fill in keys
  - Bash: copy `env.example.sh` to `env.sh` and fill in keys
- Use the provided runners from any directory:
  - Windows: `powershell -ExecutionPolicy Bypass -File d:\MyPythonProjects\k2\run.ps1`
  - macOS/Linux: `bash /path/to/k2/run.sh`
- The runners:
  - Create `.venv` if missing
  - Install `requirements.txt`
  - Load `env.ps1` or `env.sh` if present (and `.env` is auto-loaded)
  - Launch `live_rsi_bot.py`
