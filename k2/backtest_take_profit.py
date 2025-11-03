import sys
import argparse
import pandas as pd
from typing import Dict, Any

def load_csv(path: str) -> pd.DataFrame:
    """Loads a CSV file and prepares the data for backtesting."""
    df = pd.read_csv(path)
    
    # Handle either Datetime column or index-based timestamps
    if "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True, errors="coerce")
        df = df.dropna(subset=["Datetime"]).set_index("Datetime")
    else:
        df.index = pd.to_datetime(df.index, utc=True, errors="coerce")
        df = df.dropna()
    
    # Ensure required columns are present
    needed = {"Open", "High", "Low", "Close", "Volume"}
    have = set(df.columns)
    missing = needed - have
    if missing:
        raise RuntimeError(f"{path}: missing columns: {missing}")
    
    # Convert required columns to numeric
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
        
    return df.dropna()

# Note: The backtest logic is highly dependent on the correct indentation below.
def backtest_tp(df: pd.DataFrame, initial_usd=1500.0, tp_usd=2.0, fee=0.001, close_on_end=True) -> Dict[str, Any]:
    """Runs a backtest: Buy on close, sell intrabar when +$TP is hit."""
    
    equity = float(initial_usd)
    qty = 0.0
    in_pos = False
    entry_px = 0.0
    entry_cost = 0.0  # includes entry fee
    skip_next = False
    trades = []
    bars = df.shape[0]

    for i, (ts, row) in enumerate(df.iterrows()):
        high = float(row["High"])
        close = float(row["Close"])

        if not in_pos:
            if skip_next:
                # skip exactly one candle after exit
                skip_next = False
                continue
            
            # Buy now at close with all equity
            if equity > 0 and close > 0:
                qty = equity / (close * (1.0 + fee))
                entry_px = close
                entry_cost = entry_px * qty * (1.0 + fee)
                equity = 0.0
                in_pos = True
            continue

        # In position: check take-profit intrabar (use High)
        target_px = entry_px + tp_usd
        if high >= target_px:
            # Exit at target
            gross = target_px * qty
            exit_fee = gross * fee
            proceeds = gross - exit_fee
            pnl = proceeds - entry_cost
            equity += proceeds
            trades.append({"entry_px": entry_px, "exit_px": target_px, "pnl": pnl})
            
            # Reset
            qty = 0.0
            in_pos = False
            entry_px = 0.0
            entry_cost = 0.0
            skip_next = True  # wait for next candle before re-enter

    # 🚨 CRITICAL FIX: This block must NOT be indented inside the for-loop.
    # Close on end if still in position (at last close)
    if close_on_end and in_pos and qty > 0:
        last_close = float(df.iloc[-1]["Close"])
        gross = last_close * qty
        exit_fee = gross * fee
        proceeds = gross - exit_fee
        pnl = proceeds - entry_cost
        equity += proceeds
        trades.append({"entry_px": entry_px, "exit_px": last_close, "pnl": pnl})

    wins = sum(1 for t in trades if t["pnl"] > 0)
    ntr = len(trades)
    win_rate = (wins / ntr * 100.0) if ntr else 0.0
    ret_pct = (equity / initial_usd - 1.0) * 100.0
    
    return {"final": equity, "return_pct": ret_pct, "trades": ntr, "win_rate_pct": win_rate, "bars": bars}


def main():
    """Parses arguments, loads data, runs backtest, and prints results."""
    ap = argparse.ArgumentParser(description="Take-profit backtest: buy immediately, sell when +$TP, skip 1 candle, repeat.")
    ap.add_argument("files", nargs="+", help="CSV paths (e.g., bnb_binance_60d_15m.csv bnb_binance_60d_5m.csv)")
    ap.add_argument("--initial", type=float, default=1500.0, help="Initial USD")
    ap.add_argument("--tp", type=float, default=2.0, help="Take profit in USD per BNB")
    ap.add_argument("--fee", type=float, default=0.001, help="Fee rate (e.g., 0.001)")
    args = ap.parse_args()
    rows = []
    
    # Align on overlapping time window for fairness
    dfs = [(p, load_csv(p)) for p in args.files]
    starts = [d.index[0] for _, d in dfs]
    ends = [d.index[-1] for _, d in dfs]
    start = max(starts)
    end = min(ends)
    
    for path, df in dfs:
        dfx = df[(df.index >= start) & (df.index <= end)]
        res = backtest_tp(dfx, initial_usd=args.initial, tp_usd=args.tp, fee=args.fee)
        rows.append({
            "file": path,
            "bars": res["bars"],
            "trades": res["trades"],
            "win_rate_%": f"{res['win_rate_pct']:.1f}",
            "final_$": f"{res['final']:.2f}",
            "return_%": f"{res['return_pct']:.2f}",
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))

# 🚨 CRITICAL FIX: The special code block for running main() must use standard Python syntax.
if __name__ == "__main__":
    main()
