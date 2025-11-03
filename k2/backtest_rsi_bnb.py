import argparse
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
import yfinance as yf


def load_yf_csv(path: str) -> pd.DataFrame:
    """Load a Yahoo Finance CSV saved via yfinance.DataFrame.to_csv.

    Handles typical multi-row headers by:
    - Attempting MultiIndex header parse
    - Coercing index to datetime and dropping non-datetime rows
    - Collapsing to single ticker and columns [Open, High, Low, Close, Volume]
    Fallbacks to simple single-header CSV with explicit names if needed.
    """
    # Attempt MultiIndex header (common for yfinance when multiple tickers/fields)
    try:
        df = pd.read_csv(path, header=[0, 1], index_col=0)
        # Remove any stray header rows that ended up as data (e.g., a 'Datetime' row)
        idx = pd.to_datetime(df.index, errors="coerce")
        df = df[~idx.isna()].copy()
        df.index = pd.to_datetime(df.index)

        if isinstance(df.columns, pd.MultiIndex) and df.columns.nlevels == 2:
            # Level 0 might be [Open, High, Low, Close, Volume] (order may vary)
            lvl0 = set(df.columns.get_level_values(0))
            want = ["Open", "High", "Low", "Close", "Volume"]
            out = {}
            for name in want:
                if name in lvl0:
                    sub = df[name]
                    # If multiple tickers, pick the first one
                    if isinstance(sub, pd.DataFrame):
                        first_col = sub.columns[0]
                        out[name] = pd.to_numeric(sub[first_col], errors="coerce")
                    else:
                        out[name] = pd.to_numeric(sub, errors="coerce")
            out_df = pd.DataFrame(out).dropna(how="any")
            if {"Open", "High", "Low", "Close", "Volume"}.issubset(out_df.columns) and not out_df.empty:
                return out_df
    except Exception:
        pass

    # Fallback: try single header row where columns are already [Datetime, Close, High, Low, Open, Volume]
    try:
        df2 = pd.read_csv(path, header=0)
        # If there are extra header lines, drop non-numeric rows and coerce datetime
        if "Datetime" in df2.columns:
            # Ensure expected field names exist; if not, try to rename common variants
            rename_map = {"Date": "Datetime", "date": "Datetime"}
            df2 = df2.rename(columns=rename_map)
            df2["Datetime"] = pd.to_datetime(df2["Datetime"], errors="coerce")
            df2 = df2.dropna(subset=["Datetime"])  # remove any non-data header leftovers
            # Harmonize column names
            candidates = {
                "Open": ["Open", "open"],
                "High": ["High", "high"],
                "Low": ["Low", "low"],
                "Close": ["Close", "close", "Adj Close"],
                "Volume": ["Volume", "volume"],
            }
            cols = {}
            for key, opts in candidates.items():
                for o in opts:
                    if o in df2.columns:
                        cols[key] = o
                        break
            missing = {k for k in ["Open", "High", "Low", "Close", "Volume"] if k not in cols}
            if not missing:
                out_df = (
                    df2[["Datetime", cols["Open"], cols["High"], cols["Low"], cols["Close"], cols["Volume"]]]
                    .rename(columns={
                        cols["Open"]: "Open",
                        cols["High"]: "High",
                        cols["Low"]: "Low",
                        cols["Close"]: "Close",
                        cols["Volume"]: "Volume",
                    })
                    .set_index("Datetime")
                )
                # Ensure numeric types
                for c in ["Open", "High", "Low", "Close", "Volume"]:
                    out_df[c] = pd.to_numeric(out_df[c], errors="coerce")
                out_df = out_df.dropna(how="any")
                if not out_df.empty:
                    return out_df
    except Exception:
        pass

    raise RuntimeError(
        "Failed to parse CSV. Ensure it was produced by yfinance.to_csv and includes columns for Open, High, Low, Close, Volume."
    )

def fetch_bnb_data(
    period: str = "60d",
    ticker: str = "BNB-USD",
    force_interval: Optional[str] = None,
) -> pd.DataFrame:
    # Try smallest feasible intervals for multi-week crypto data.
    # Yahoo often allows: 1m (short windows), 2m (up to ~60d), 5m/15m etc.
    default_intervals = ["2m", "5m", "15m", "30m", "60m", "90m", "1h", "2h", "4h", "1d"]
    intervals = [force_interval] if force_interval else default_intervals
    last_error: Optional[Exception] = None
    attempted: List[str] = []
    for interval in intervals:
        if interval is None:
            continue
        attempted.append(interval)
        try:
            df = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            if isinstance(df, pd.DataFrame) and not df.empty and {"Open", "High", "Low", "Close", "Volume"}.issubset(df.columns):
                df = df.dropna().copy()
                df["Interval"] = interval
                return df
        except Exception as e:  # noqa: BLE001
            last_error = e
            continue
    msg = (
        f"Failed to download data for {ticker}. Tried intervals: {attempted}. "
        "Try specifying --interval 15m (or 30m/60m) or reducing --period."
    )
    if last_error:
        raise RuntimeError(msg) from last_error
    raise RuntimeError(msg)


def compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"].astype(float)
    low = df["Low"].astype(float)
    close = df["Close"].astype(float)
    prev_close = close.shift(1)
    tr1 = (high - low).abs()
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    return atr


@dataclass
class Trade:
    entry_time: pd.Timestamp
    entry_price: float
    qty: float
    entry_fee: float
    exit_time: pd.Timestamp
    exit_price: float
    exit_fee: float
    pnl: float


def backtest_rsi(
    df: pd.DataFrame,
    initial_usd: float = 1500.0,
    fee_rate: float = 0.001,
    rsi_period: int = 14,
    entry_rsi: float = 28.0,
    exit_rsi: float = 55.0,
    stop_loss_pct: float = 0.10,
    atr_period: int = 14,
    atr_thresh: float = 0.0,
    close_on_end: bool = True,
) -> dict:
    data = df.copy()
    data["RSI"] = compute_rsi(data["Close"], period=rsi_period)
    if atr_thresh and atr_thresh > 0:
        atr = compute_atr(data, period=atr_period)
        data["ATR_PCT"] = (atr / data["Close"].astype(float)).fillna(0.0)
    else:
        data["ATR_PCT"] = 0.0

    usd = float(initial_usd)
    qty = 0.0
    entry_price = 0.0
    entry_time: Optional[pd.Timestamp] = None
    trades: List[Trade] = []

    for ts, row in data.iterrows():
        price = float(row["Close"])
        rsi = float(row["RSI"])

        # Entry condition: RSI <= entry_rsi, ATR% filter passed, and not in position
        atr_ok = (atr_thresh <= 0.0) or (float(row.get("ATR_PCT", 0.0)) >= float(atr_thresh))
        if qty == 0.0 and rsi <= entry_rsi and atr_ok:
            if price > 0.0 and usd > 0.0:
                qty = usd / (price * (1.0 + fee_rate))
                entry_fee = price * qty * fee_rate
                usd = 0.0
                entry_price = price
                entry_time = ts
            continue

        # Exit logic if in position
        if qty > 0.0:
            stop_trigger = price <= entry_price * (1.0 - stop_loss_pct)
            exit_signal = rsi >= exit_rsi
            if stop_trigger or exit_signal:
                gross = price * qty
                exit_fee = gross * fee_rate
                proceeds = gross - exit_fee
                pnl = proceeds - (entry_price * qty + entry_price * qty * fee_rate)
                usd += proceeds
                trades.append(
                    Trade(
                        entry_time=entry_time if entry_time is not None else ts,
                        entry_price=entry_price,
                        qty=qty,
                        entry_fee=entry_price * qty * fee_rate,
                        exit_time=ts,
                        exit_price=price,
                        exit_fee=exit_fee,
                        pnl=pnl,
                    )
                )
                qty = 0.0
                entry_price = 0.0
                entry_time = None

    # Close at the end if requested
    if close_on_end and qty > 0.0:
        last_ts = data.index[-1]
        last_px = float(data.iloc[-1]["Close"])
        gross = last_px * qty
        exit_fee = gross * fee_rate
        proceeds = gross - exit_fee
        pnl = proceeds - (entry_price * qty + entry_price * qty * fee_rate)
        usd += proceeds
        trades.append(
            Trade(
                entry_time=entry_time if entry_time is not None else last_ts,
                entry_price=entry_price,
                qty=qty,
                entry_fee=entry_price * qty * fee_rate,
                exit_time=last_ts,
                exit_price=last_px,
                exit_fee=exit_fee,
                pnl=pnl,
            )
        )
        qty = 0.0

    final_equity = usd
    ret_pct = (final_equity / initial_usd - 1.0) * 100.0
    wins = sum(1 for t in trades if t.pnl > 0)
    losses = sum(1 for t in trades if t.pnl <= 0)

    return {
        "initial": initial_usd,
        "final": final_equity,
        "return_pct": ret_pct,
        "trades": trades,
        "win_rate_pct": (wins / len(trades) * 100.0) if trades else 0.0,
        "n_trades": len(trades),
        "interval": df.get("Interval").iloc[0] if "Interval" in df.columns else "",
    }


def backtest_with_precomputed_rsi(
    close: pd.Series,
    rsi: pd.Series,
    initial_usd: float,
    fee_rate: float,
    entry_rsi: float,
    exit_rsi: float,
    stop_loss_pct: float,
    atr_pct: Optional[pd.Series] = None,
    atr_thresh: float = 0.0,
    close_on_end: bool = True,
):
    usd = float(initial_usd)
    qty = 0.0
    entry_price = 0.0

    n_trades = 0
    wins = 0

    for ts in close.index:
        price = float(close.loc[ts])
        r = float(rsi.loc[ts])

        atr_ok = True
        if atr_pct is not None and atr_thresh and atr_thresh > 0:
            ap = float(atr_pct.loc[ts]) if ts in atr_pct.index else 0.0
            atr_ok = ap >= float(atr_thresh)

        if qty == 0.0 and r <= entry_rsi and atr_ok:
            if price > 0.0 and usd > 0.0:
                qty = usd / (price * (1.0 + fee_rate))
                usd = 0.0
                entry_price = price
            continue

        if qty > 0.0:
            stop_trigger = price <= entry_price * (1.0 - stop_loss_pct)
            exit_signal = r >= exit_rsi
            if stop_trigger or exit_signal:
                gross = price * qty
                exit_fee = gross * fee_rate
                proceeds = gross - exit_fee
                pnl = proceeds - (entry_price * qty + entry_price * qty * fee_rate)
                usd += proceeds
                n_trades += 1
                if pnl > 0:
                    wins += 1
                qty = 0.0
                entry_price = 0.0

    if close_on_end and qty > 0.0:
        last_px = float(close.iloc[-1])
        gross = last_px * qty
        exit_fee = gross * fee_rate
        proceeds = gross - exit_fee
        pnl = proceeds - (entry_price * qty + entry_price * qty * fee_rate)
        usd += proceeds
        n_trades += 1
        if pnl > 0:
            wins += 1

    final_equity = usd
    ret_pct = (final_equity / initial_usd - 1.0) * 100.0
    win_rate = (wins / n_trades * 100.0) if n_trades else 0.0
    return final_equity, ret_pct, n_trades, win_rate


def parse_int_range(spec: str) -> List[int]:
    # format: "start,end,step" inclusive
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError("Range must be 'start,end,step'")
    start, end, step = map(int, parts)
    if step <= 0:
        raise ValueError("step must be > 0")
    return list(range(start, end + 1, step))


def parse_float_list(spec: str) -> List[float]:
    # format: "0.05,0.08,0.10"
    return [float(x.strip()) for x in spec.split(",") if x.strip()]


def grid_search(
    df: pd.DataFrame,
    initial_usd: float,
    fee_rate: float,
    rsi_periods: List[int],
    entry_vals: List[int],
    exit_vals: List[int],
    stop_list: List[float],
    atr_period: int = 14,
    atr_thresh: float = 0.0,
    close_on_end: bool = True,
    top_k: int = 10,
):
    close = df["Close"].astype(float).copy()
    atr_series = None
    if atr_thresh and atr_thresh > 0:
        atr = compute_atr(df, period=atr_period)
        atr_series = (atr / close).fillna(0.0)

    results = []
    for p in rsi_periods:
        rsi = compute_rsi(close, period=p)
        for e in entry_vals:
            for x in exit_vals:
                if x <= e:  # require exit above entry
                    continue
                for s in stop_list:
                    final_equity, ret_pct, n_trades, win_rate = backtest_with_precomputed_rsi(
                        close, rsi, initial_usd, fee_rate, float(e), float(x), float(s), atr_series, atr_thresh, close_on_end
                    )
                    results.append(
                        {
                            "rsi_period": p,
                            "entry": e,
                            "exit": x,
                            "stop": s,
                            "final": final_equity,
                            "return_pct": ret_pct,
                            "trades": n_trades,
                            "win_rate": win_rate,
                        }
                    )

    res_df = pd.DataFrame(results)
    if res_df.empty:
        return res_df
    res_df = res_df.sort_values(by=["final", "return_pct"], ascending=[False, False]).reset_index(drop=True)
    return res_df.head(top_k), res_df


def main():
    parser = argparse.ArgumentParser(description="RSI backtest for BNB-USD (proxy for BNB/USDT)")
    parser.add_argument("--period", default="60d", help="Yahoo period to download, e.g. 60d")
    parser.add_argument("--csv", default=None, help="Path to a local Yahoo CSV (uses this instead of downloading)")
    parser.add_argument("--ticker", default="BNB-USD", help="Yahoo ticker symbol, default BNB-USD")
    parser.add_argument(
        "--interval",
        default=None,
        help="Force a specific interval (e.g. 15m, 30m, 60m). If not set, the script tries the finest allowed.",
    )
    parser.add_argument("--fee", type=float, default=0.001, help="Trading fee rate, e.g. 0.001 = 0.1%")
    parser.add_argument("--initial", type=float, default=1500.0, help="Initial USD capital")
    parser.add_argument("--rsi_period", type=int, default=14, help="RSI lookback period")
    parser.add_argument("--entry", type=float, default=28.0, help="RSI entry threshold")
    parser.add_argument("--exit", type=float, default=55.0, help="RSI exit threshold")
    parser.add_argument("--stop", type=float, default=0.10, help="Stop loss percent (0.10 = 10%)")
    parser.add_argument("--atr_period", type=int, default=14, help="ATR period for volatility filter")
    parser.add_argument("--atr_thresh", type=float, default=0.0, help="ATR% threshold for entries; e.g. 0.005 = 0.5%")
    parser.add_argument("--optimize", action="store_true", help="Run grid search to find best parameters")
    parser.add_argument(
        "--rsi_periods",
        default="10,14,21",
        help="Comma list of RSI periods for optimization (e.g. 10,14,21)",
    )
    parser.add_argument(
        "--entry_range",
        default="20,35,2",
        help="Entry RSI range start,end,step for optimization (inclusive)",
    )
    parser.add_argument(
        "--exit_range",
        default="50,80,5",
        help="Exit RSI range start,end,step for optimization (inclusive)",
    )
    parser.add_argument(
        "--stops",
        default="0.05,0.08,0.10,0.12",
        help="Comma list of stop loss percents to test (e.g. 0.05,0.08,0.10)",
    )
    parser.add_argument("--top_k", type=int, default=10, help="How many top results to display in optimize mode")
    parser.add_argument("--results_csv", default=None, help="Optional path to save all optimization results as CSV")
    parser.add_argument(
        "--objective",
        default="final",
        choices=["final", "return", "trades"],
        help="Optimization objective: final (equity), return (pct), or trades (with min_final)",
    )
    parser.add_argument(
        "--min_final",
        type=float,
        default=1500.0,
        help="Minimum final equity constraint when objective=trades",
    )
    args = parser.parse_args()

    if args.csv:
        df = load_yf_csv(args.csv)
    else:
        df = fetch_bnb_data(period=args.period, ticker=args.ticker, force_interval=args.interval)

    if args.optimize:
        rsi_periods = [int(x.strip()) for x in args.rsi_periods.split(",") if x.strip()]
        entry_vals = parse_int_range(args.entry_range)
        exit_vals = parse_int_range(args.exit_range)
        stop_list = parse_float_list(args.stops)
        top_df, full_df = grid_search(
            df,
            initial_usd=args.initial,
            fee_rate=args.fee,
            rsi_periods=rsi_periods,
            entry_vals=entry_vals,
            exit_vals=exit_vals,
            stop_list=stop_list,
            atr_period=args.atr_period,
            atr_thresh=args.atr_thresh,
            close_on_end=True,
            top_k=args.top_k,
        )
        if args.objective == "trades":
            feasible = full_df[full_df["final"] >= float(args.min_final)].copy()
            if feasible.empty:
                print(  # noqa: T201
                    f"No combinations met min_final=${args.min_final:.2f}. Showing best by trades (no constraint):"
                )
                use_df = full_df.sort_values(by=["trades", "final"], ascending=[False, False])
            else:
                use_df = feasible.sort_values(by=["trades", "final"], ascending=[False, False])
            show_df = use_df.head(args.top_k).reset_index(drop=True)
            print(f"Tried {len(full_df)} combinations. Top {args.top_k} by trades:")  # noqa: T201
            for i, row in show_df.iterrows():
                print(  # noqa: T201
                    f"#{i+1}: period={row['rsi_period']}, entry={row['entry']}, exit={row['exit']}, stop={row['stop']:.3f} | "
                    f"Final=${row['final']:.2f}, Return={row['return_pct']:.2f}%, Trades={int(row['trades'])}, WinRate={row['win_rate']:.1f}%"
                )
        else:
            sort_cols = ["final", "return_pct"] if args.objective == "final" else ["return_pct", "final"]
            use_df = full_df.sort_values(by=sort_cols, ascending=[False, False])
            show_df = use_df.head(args.top_k).reset_index(drop=True)
            print(f"Tried {len(full_df)} combinations. Top {args.top_k}:")  # noqa: T201
            for i, row in show_df.iterrows():
                print(  # noqa: T201
                    f"#{i+1}: period={row['rsi_period']}, entry={row['entry']}, exit={row['exit']}, stop={row['stop']:.3f} | "
                    f"Final=${row['final']:.2f}, Return={row['return_pct']:.2f}%, Trades={int(row['trades'])}, WinRate={row['win_rate']:.1f}%"
                )
        if args.results_csv:
            full_df.to_csv(args.results_csv, index=False)
            print(f"Saved full results to {args.results_csv}")  # noqa: T201
        return

    result = backtest_rsi(
        df,
        initial_usd=args.initial,
        fee_rate=args.fee,
        rsi_period=args.rsi_period,
        entry_rsi=args.entry,
        exit_rsi=args.exit,
        stop_loss_pct=args.stop,
        atr_period=args.atr_period,
        atr_thresh=args.atr_thresh,
        close_on_end=True,
    )

    print("Interval:", result["interval"])  # noqa: T201
    print(f"Initial: ${result['initial']:.2f}")  # noqa: T201
    print(f"Final:   ${result['final']:.2f}")  # noqa: T201
    print(f"Return:  {result['return_pct']:.2f}%")  # noqa: T201
    print(f"Trades:  {result['n_trades']}")  # noqa: T201
    print(f"WinRate: {result['win_rate_pct']:.2f}%")  # noqa: T201

    if result["trades"]:
        print("\nSample trades (up to 10):")  # noqa: T201
        for t in result["trades"][:10]:
            print(  # noqa: T201
                f"{t.entry_time} buy @ {t.entry_price:.2f} -> {t.exit_time} sell @ {t.exit_price:.2f} | PnL: {t.pnl:.2f}"
            )


if __name__ == "__main__":
    main()
