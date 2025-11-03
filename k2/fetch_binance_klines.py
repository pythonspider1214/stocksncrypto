from binance.spot import Spot
import pandas as pd
import time

# --- Configuration ---
sym = "BNBUSDT"
interval = "5m"  # change to "5m" for 5-minute data
days = 60
# ---------------------

client = Spot()  # public; no keys needed

end = int(time.time() * 1000)
start = end - days * 24 * 60 * 60 * 1000
rows = []
st = start

# Loop to fetch data in chunks of 1000 candles
while st < end:
    # Indentation starts the 'while' loop block
    ks = client.klines(symbol=sym, interval=interval, startTime=st, endTime=end, limit=1000)
    
    # Indentation starts the 'if' block
    if not ks:
        break
    
    rows.extend(ks)
    # The [6] index is the closeTime; add 1 millisecond for the next starting point
    st = ks[-1][6] + 1 
    
# --- Data Processing ---
df = pd.DataFrame(
    rows,
    columns=["open_time", "open", "high", "low", "close", "volume", "close_time", "q", "n", "taker_base", "taker_quote", "ignore"],
)

# Rename columns
df = df.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

# Convert price and volume columns to numeric
for c in ["Open", "High", "Low", "Close", "Volume"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# Convert open_time (ms) to readable Datetime
df["Datetime"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)

# Select and reorder final columns
df = df[["Datetime", "Open", "High", "Low", "Close", "Volume"]]

# --- Save to File ---
out = f"bnb_binance_{days}d_{interval}.csv"
df.to_csv(out, index=False)

print(f"Saved {len(df)} rows to {out}")
