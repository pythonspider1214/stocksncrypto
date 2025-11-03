import os
import sys
import time
import json
import math
import signal
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

try:
    from dotenv import load_dotenv
except Exception:  # noqa: BLE001
    load_dotenv = None  # type: ignore

try:
    # Binance Spot SDK (binance-connector)
    from binance.spot import Spot
except Exception as e:  # noqa: BLE001
    Spot = None  # type: ignore

try:
    import requests
except Exception:  # noqa: BLE001
    requests = None  # type: ignore


def env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v is not None else default
    except Exception:  # noqa: BLE001
        return default


def env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v is not None else default
    except Exception:  # noqa: BLE001
        return default


def compute_rsi(close: pd.Series, period: int = 21) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0.0)
    losses = -delta.clip(upper=0.0)
    avg_gain = gains.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = losses.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / (avg_loss.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


@dataclass
class Filters:
    price_tick: float
    qty_step: float
    min_qty: float
    min_notional: float


def round_step(value: float, step: float) -> float:
    if step <= 0:
        return value
    precision = int(round(-math.log10(step))) if step < 1 else 0
    return float((math.floor(value / step) * step)) if precision == 0 else float(f"{value/step:.0f}") * step


def quantize(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.floor(value / step) * step


def quantize_up(value: float, step: float) -> float:
    if step <= 0:
        return value
    return math.ceil(value / step) * step


class Telegram:
    def __init__(self, token: Optional[str], chat_id: Optional[str]):
        self.token = token
        self.chat_id = chat_id

    def send(self, text: str) -> None:
        if not self.token or not self.chat_id or requests is None:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, json={"chat_id": self.chat_id, "text": text[:4000]})
        except Exception:  # noqa: BLE001
            pass


class BinanceBot:
    def __init__(self) -> None:
        # Load .env if present (cross-platform convenience)
        try:
            if load_dotenv is not None:
                # Load from CWD if present, then from user home
                load_dotenv(dotenv_path=Path.cwd() / ".env", override=False)
                load_dotenv(dotenv_path=Path.home() / ".env", override=False)
        except Exception:  # noqa: BLE001
            pass
        # Config
        self.symbol = os.getenv("SYMBOL", "BNBUSDT")
        self.interval = os.getenv("INTERVAL", "15m")
        self.rsi_period = env_int("RSI_PERIOD", 21)
        self.entry_rsi = env_float("ENTRY_RSI", 26.0)
        self.exit_rsi = env_float("EXIT_RSI", 80.0)
        self.stop_pct = env_float("STOP_PCT", 0.08)
        self.allocation_pct = env_float("ALLOCATION_PCT", 0.95)
        self.allocation_usdt = env_float("ALLOCATION_USDT", 0.0)
        self.fee_rate = env_float("FEE_RATE", 0.001)
        self.price_buffer_bps = env_float("PRICE_BUFFER_BPS", 5.0)  # 5 bps = 0.05%
        self.use_limit_maker = env_int("USE_LIMIT_MAKER", 0) == 1
        self.live = env_int("LIVE_TRADING", 0) == 1
        self.log_rsi = env_int("LOG_RSI", 0) == 1
        self.heartbeat_min = env_int("HEARTBEAT_MINUTES", 0)
        self.intrabar_entry = env_int("INTRABAR_ENTRY", 0) == 1
        self.entry_cross = env_int("ENTRY_CROSS", 0) == 1
        self.tp_pct = env_float("TP_PCT", 0.0)
        self.tp_trigger = os.getenv("TP_TRIGGER", "close").lower()
        # Default state path in user home for portability
        default_state = str(Path.home() / ".k2_rsi_bot" / "bot_state.json")
        self.state_path = os.getenv("STATE_PATH", default_state)
        state_dir = Path(self.state_path).parent
        state_dir.mkdir(parents=True, exist_ok=True)
        # Single-instance guard and status trigger paths
        self.lock_path = Path(os.getenv("LOCK_PATH", str(state_dir / "bot.lock")))
        self.status_request_path = Path(os.getenv("STATUS_REQUEST_PATH", str(state_dir / "status.request")))
        self.poll_sec = env_int("POLL_INTERVAL_SEC", 60)
        self.klines_limit = env_int("KLINES_LIMIT", 200)

        # API
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        base_url = os.getenv("BINANCE_BASE_URL")  # optionally override
        if Spot is None:
            raise RuntimeError("binance.spot SDK not available. Install binance-connector or binance-sdk-spot.")
        if base_url:
            self.client = Spot(api_key=api_key, api_secret=api_secret, base_url=base_url)
        else:
            self.client = Spot(api_key=api_key, api_secret=api_secret)

        # Telegram
        self.tg = Telegram(os.getenv("TELEGRAM_BOT_TOKEN"), os.getenv("TELEGRAM_CHAT_ID"))

        # Runtime
        self.in_position = False
        self.entry_price: Optional[float] = None
        self.position_qty: float = 0.0
        self.entry_value: float = 0.0  # includes fee on buy
        self.stop_order_id: Optional[int] = None
        self.filters: Optional[Filters] = None
        self.last_candle_open_time: Optional[int] = None
        self.last_heartbeat_ts: float = 0.0
        self.pid = os.getpid()

        # Logging
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        self.log = logging.getLogger("BNB-RSI-BOT")

    # --------------- Helpers ---------------
    def load_filters(self) -> None:
        info = self.client.exchange_info(symbol=self.symbol)
        fs = info["symbols"][0]["filters"]
        price_tick = 0.0
        qty_step = 0.0
        min_qty = 0.0
        min_notional = 10.0
        for f in fs:
            ft = f.get("filterType")
            if ft == "PRICE_FILTER":
                price_tick = float(f["tickSize"])
            elif ft == "LOT_SIZE":
                qty_step = float(f["stepSize"])
                min_qty = float(f["minQty"])
            elif ft == "MIN_NOTIONAL":
                min_notional = float(f.get("minNotional", 10.0))
        self.filters = Filters(price_tick=price_tick, qty_step=qty_step, min_qty=min_qty, min_notional=min_notional)
        self.log.info("Filters: %s", self.filters)

    def get_book_ticker(self) -> Dict[str, Any]:
        return self.client.book_ticker(symbol=self.symbol)

    def get_klines_df(self) -> pd.DataFrame:
        k = self.client.klines(symbol=self.symbol, interval=self.interval, limit=self.klines_limit)
        cols = ["open_time", "open", "high", "low", "close", "volume", "close_time", "q", "n", "taker_base", "taker_quote", "ignore"]
        df = pd.DataFrame(k, columns=cols)
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["open_time"] = df["open_time"].astype(np.int64)
        df["close_time"] = df["close_time"].astype(np.int64)
        return df[["open_time", "open", "high", "low", "close", "volume", "close_time"]]

    def price_with_buffer(self, mid: float, side: str) -> float:
        # Apply small buffer in basis points to improve fill probability
        bps = self.price_buffer_bps / 10000.0
        if side == "BUY":
            px = mid * (1 + bps) if not self.use_limit_maker else mid * (1 - bps)
        else:
            px = mid * (1 - bps) if not self.use_limit_maker else mid * (1 + bps)
        return px

    def quantize_price(self, price: float) -> float:
        if not self.filters:
            return price
        tick = self.filters.price_tick
        if tick <= 0:
            return price
        # round to nearest tick down for sell, up for buy handled by caller
        return round(price / tick) * tick

    def send_telegram(self, msg: str) -> None:
        self.log.info(msg)
        self.tg.send(msg)

    def save_state(self) -> None:
        try:
            # Ensure directory exists
            sp = Path(self.state_path)
            sp.parent.mkdir(parents=True, exist_ok=True)
            st = {
                "in_position": self.in_position,
                "entry_price": self.entry_price,
                "position_qty": self.position_qty,
                "entry_value": self.entry_value,
                "stop_order_id": self.stop_order_id,
                "last_candle_open_time": self.last_candle_open_time,
            }
            with open(sp, "w", encoding="utf-8") as f:
                json.dump(st, f)
        except Exception:  # noqa: BLE001
            pass

    def load_state(self) -> None:
        try:
            sp = Path(self.state_path)
            if not sp.exists():
                return
            with open(sp, "r", encoding="utf-8") as f:
                st = json.load(f)
            self.in_position = bool(st.get("in_position", False))
            self.entry_price = st.get("entry_price")
            self.position_qty = float(st.get("position_qty", 0.0))
            self.entry_value = float(st.get("entry_value", 0.0))
            self.stop_order_id = st.get("stop_order_id")
            self.last_candle_open_time = st.get("last_candle_open_time")
        except Exception:  # noqa: BLE001
            pass

    # --------------- Single-instance guard ---------------
    def acquire_lock(self) -> None:
        try:
            if self.lock_path.exists():
                try:
                    with open(self.lock_path, "r", encoding="utf-8") as f:
                        old_pid = int(f.read().strip())
                except Exception:
                    old_pid = -1
                # Check if process exists
                alive = False
                if old_pid > 0:
                    try:
                        os.kill(old_pid, 0)
                        alive = True
                    except Exception:
                        alive = False
                if alive and old_pid != self.pid:
                    raise SystemExit(f"Another instance is running with PID {old_pid}. Delete {self.lock_path} if stale.")
                # Stale lock
                try:
                    self.lock_path.unlink(missing_ok=True)  # type: ignore[call-arg]
                except TypeError:
                    # Python <3.8 fallback
                    try:
                        if self.lock_path.exists():
                            self.lock_path.unlink()
                    except Exception:
                        pass
            with open(self.lock_path, "w", encoding="utf-8") as f:
                f.write(str(self.pid))
        except SystemExit:
            raise
        except Exception:
            # If we can't create lock, proceed but warn
            self.log.warning("Could not acquire lock at %s", self.lock_path)

    def release_lock(self) -> None:
        try:
            if self.lock_path.exists():
                self.lock_path.unlink()
        except Exception:
            pass

    # --------------- Trading actions ---------------
    def place_order(self, side: str, qty: float, price: float, tif: str = "GTC") -> Optional[Dict[str, Any]]:
        if not self.filters:
            raise RuntimeError("Filters not loaded")
        qty = max(self.filters.min_qty, quantize(qty, self.filters.qty_step))
        if qty <= 0:
            return None
        price = quantize_up(price, self.filters.price_tick) if side == "BUY" else quantize(price, self.filters.price_tick)
        # Safety: min notional
        if price * qty < self.filters.min_notional:
            return None
        if not self.live:
            self.send_telegram(f"DRY-RUN: {side} {qty} {self.symbol} @ {price}")
            return {"orderId": -1, "status": "FILLED", "price": str(price), "origQty": str(qty)}
        params = {
            "symbol": self.symbol,
            "side": side,
            "type": "LIMIT_MAKER" if self.use_limit_maker else "LIMIT",
            "timeInForce": tif,
            "quantity": f"{qty:.8f}",
            "price": f"{price:.8f}",
            "newOrderRespType": "FULL",
        }
        od = self.client.new_order(**params)
        return od

    def place_stop_loss_limit(self, qty: float, stop_price: float) -> Optional[Dict[str, Any]]:
        # Place STOP_LOSS_LIMIT sell; set limit slightly below stopPrice
        if not self.filters:
            raise RuntimeError("Filters not loaded")
        qty = max(self.filters.min_qty, quantize(qty, self.filters.qty_step))
        stop_price_q = quantize(stop_price, self.filters.price_tick)
        limit_price = quantize(max(0.0, stop_price_q * (1 - 0.001)), self.filters.price_tick)
        if stop_price_q * qty < self.filters.min_notional:
            return None
        if not self.live:
            self.send_telegram(f"DRY-RUN: STOP_LOSS_LIMIT sell {qty} @ stop {stop_price_q}, limit {limit_price}")
            return {"orderId": -2, "status": "NEW", "stopPrice": str(stop_price_q), "price": str(limit_price)}
        od = self.client.new_order(
            symbol=self.symbol,
            side="SELL",
            type="STOP_LOSS_LIMIT",
            timeInForce="GTC",
            quantity=f"{qty:.8f}",
            price=f"{limit_price:.8f}",
            stopPrice=f"{stop_price_q:.8f}",
            newOrderRespType="FULL",
        )
        return od

    def cancel_order_safe(self, order_id: int) -> None:
        if not self.live or order_id in (-1, -2):  # dry-run pseudo ids
            return
        try:
            self.client.cancel_order(symbol=self.symbol, orderId=order_id)
        except Exception:  # noqa: BLE001
            pass

    def fetch_balance(self) -> Dict[str, float]:
        acc = self.client.account()
        bals = {b["asset"]: float(b["free"]) for b in acc["balances"]}
        return bals

    # --------------- Strategy loop ---------------
    def run(self) -> None:
        self.acquire_lock()
        self.load_filters()
        self.load_state()
        self.send_telegram(
            f"Bot start. Symbol={self.symbol}, Interval={self.interval}, RSI={self.rsi_period}, Entry<={self.entry_rsi}, Exit>={self.exit_rsi}, Stop={self.stop_pct*100:.1f}% | Live={self.live}"
        )

        stop = False

        def handle_sig(sig, frame):  # noqa: ANN001, ANN201
            nonlocal stop
            stop = True

        signal.signal(signal.SIGINT, handle_sig)
        signal.signal(signal.SIGTERM, handle_sig)

        while not stop:
            try:
                df = self.get_klines_df()
                if df.empty:
                    time.sleep(self.poll_sec)
                    continue
                last = df.iloc[-1]
                last_open_time = int(last["open_time"])
                now_ms = int(time.time() * 1000)
                is_open = now_ms < int(last["close_time"])
                if is_open:
                    row_closed = df.iloc[-2]
                    series_closed = df.iloc[:-1]["close"].copy()
                else:
                    row_closed = last
                    series_closed = df["close"].copy()

                series_live = df["close"].copy()
                row_live = last

                use_live_entry = self.intrabar_entry and is_open
                row = row_live if use_live_entry else row_closed
                close_series = series_live if use_live_entry else series_closed
                candle_open_time = int(row["open_time"])

                if (not use_live_entry) and self.last_candle_open_time == candle_open_time:
                    time.sleep(self.poll_sec)
                    continue

                rsi_series_closed = compute_rsi(series_closed, self.rsi_period)
                rsi_closed = float(rsi_series_closed.iloc[-1])
                if use_live_entry:
                    rsi_series_live = compute_rsi(series_live, self.rsi_period)
                    rsi_entry = float(rsi_series_live.iloc[-1])
                else:
                    rsi_entry = rsi_closed
                rsi_exit = rsi_closed

                if self.log_rsi:
                    tag = "live" if use_live_entry else "close"
                    self.log.info(
                        "Candle %s | close=%.4f RSI(%s)=%.2f (entry<=%.2f exit>=%.2f)",
                        time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(candle_open_time/1000)),
                        float(row["close"]),
                        tag,
                        rsi_entry,
                        self.entry_rsi,
                        self.exit_rsi,
                    )

                book = self.get_book_ticker()
                bid = float(book["bidPrice"]) if "bidPrice" in book else float(row["close"])
                ask = float(book["askPrice"]) if "askPrice" in book else float(row["close"])
                mid = (bid + ask) / 2

                # Position detection uses state; if desynced, fall back to balances
                if self.in_position and self.position_qty <= 0:
                    bals = self.fetch_balance()
                    self.position_qty = float(bals.get(self.symbol.replace("USDT", ""), 0.0))

                if not self.in_position:
                    # Entry signal (optionally require crossing below threshold)
                    prev_rsi = float(rsi_series_used.iloc[-2]) if 'rsi_series_used' in locals() and len(rsi_series_used) >= 2 else rsi_entry
                    entry_ok = (rsi_entry <= self.entry_rsi)
                    if self.entry_cross:
                        entry_ok = (prev_rsi > self.entry_rsi and rsi_entry <= self.entry_rsi)
                    if entry_ok:
                        usdt_to_spend: float
                        if self.allocation_usdt > 0:
                            usdt_to_spend = self.allocation_usdt
                        else:
                            bals = self.fetch_balance()
                            usdt_to_spend = float(bals.get("USDT", 0.0)) * self.allocation_pct
                        if usdt_to_spend <= 0:
                            self.send_telegram("No USDT available to allocate.")
                        else:
                            buy_price = self.price_with_buffer(mid, side="BUY")
                            qty = usdt_to_spend / buy_price
                            od = self.place_order("BUY", qty, buy_price)
                            if od is not None:
                                # Assume filled or soon-to-be filled; set state conservatively
                                self.entry_price = float(od.get("price", buy_price))
                                # Fetch actual fills if live
                                if self.live and od.get("status") != "FILLED":
                                    # Wait briefly and poll order status
                                    time.sleep(2)
                                    try:
                                        od = self.client.get_order(symbol=self.symbol, orderId=od["orderId"])  # type: ignore[index]
                                    except Exception:  # noqa: BLE001
                                        pass
                                executed_qty = float(od.get("executedQty", od.get("origQty", 0)))
                                self.position_qty = executed_qty if executed_qty > 0 else max(0.0, qty)
                                self.in_position = self.position_qty > 0
                                if self.in_position and self.entry_price:
                                    # Include taker fee approximation on buy side
                                    self.entry_value = self.entry_price * self.position_qty * (1 + self.fee_rate)
                                    stop_price = self.entry_price * (1 - self.stop_pct)
                                    sl = self.place_stop_loss_limit(self.position_qty, stop_price)
                                    self.stop_order_id = int(sl["orderId"]) if sl and "orderId" in sl else None
                                    self.send_telegram(
                                        f"ENTRY: qty={self.position_qty:.6f} @ {self.entry_price:.4f} | RSI={rsi_entry:.2f}; SL @ {stop_price:.4f}"
                                    )
                            self.save_state()
                else:
                    # Exit by RSI or TP; stop handled by stop order
                    tp_hit = False
                    if self.tp_pct > 0 and self.entry_price and self.position_qty > 0:
                        tp_target = self.entry_price * (1 + self.tp_pct)
                        try:
                            row_high = float(row["high"]) if "high" in row else float(row["close"])  # type: ignore[index]
                        except Exception:
                            row_high = float(row["close"])  # type: ignore[index]
                        tp_check = row_high if self.tp_trigger == "highlow" else float(row["close"])  # type: ignore[index]
                        if tp_check >= tp_target:
                            tp_hit = True
                    # If RSI exit condition or TP hit: cancel SL and place limit sell
                    if (tp_hit or (rsi_exit >= self.exit_rsi)) and self.position_qty > 0:
                        if self.stop_order_id:
                            self.cancel_order_safe(self.stop_order_id)
                            self.stop_order_id = None
                        sell_price = self.price_with_buffer(mid, side="SELL")
                        od = self.place_order("SELL", self.position_qty, sell_price)
                        if od is not None:
                            # Approximate realized PnL with fee on sell
                            exec_qty = float(od.get("executedQty", self.position_qty))
                            exec_price = float(od.get("price", sell_price))
                            proceeds = exec_qty * exec_price * (1 - self.fee_rate)
                            pnl = proceeds - self.entry_value
                            self.in_position = False
                            if tp_hit:
                                self.send_telegram(
                                    f"EXIT TP: sold {exec_qty:.6f} @ {exec_price:.4f} | Target>={tp_target:.4f} | PnL={pnl:.2f} USDT"
                                )
                            else:
                                self.send_telegram(
                                    f"EXIT: sold {exec_qty:.6f} @ {exec_price:.4f} | RSI={rsi_exit:.2f} | PnL={pnl:.2f} USDT"
                                )
                            self.position_qty = 0.0
                            self.entry_price = None
                            self.entry_value = 0.0
                            self.save_state()

                self.last_candle_open_time = candle_open_time
                self.save_state()
                # Status request trigger: if file exists, send a status snapshot
                try:
                    if self.status_request_path.exists():
                        self.send_status_snapshot(rsi_entry, float(row["close"]))
                        # remove trigger
                        try:
                            self.status_request_path.unlink()
                        except Exception:
                            pass
                except Exception:
                    pass
                # Optional heartbeat to Telegram
                if self.heartbeat_min and self.heartbeat_min > 0:
                    now_ts = time.time()
                    if now_ts - self.last_heartbeat_ts >= self.heartbeat_min * 60:
                        self.last_heartbeat_ts = now_ts
                        self.send_telegram(
                            f"Heartbeat: {self.symbol} {self.interval} running. Last RSI={rsi_entry:.2f}, price={float(row['close']):.4f}"
                        )
                time.sleep(self.poll_sec)
            except Exception as e:  # noqa: BLE001
                self.send_telegram(f"Loop error: {type(e).__name__}: {e}")
                time.sleep(5)
        # loop end
        self.release_lock()

    def send_status_snapshot(self, rsi_val: float, last_close: float) -> None:
        unreal = 0.0
        if self.in_position and self.entry_price and self.position_qty > 0:
            mark = last_close * self.position_qty * (1 - self.fee_rate)
            unreal = mark - self.entry_value
        self.send_telegram(
            (
                f"Status: {self.symbol} {self.interval} | Close={last_close:.4f} RSI={rsi_val:.2f}\n"
                f"Position: {'OPEN' if self.in_position else 'FLAT'}"
                + (
                    f" qty={self.position_qty:.6f} entry={self.entry_price:.4f} unrealPnL={unreal:.2f} USDT"
                    if self.in_position and self.entry_price
                    else ""
                )
            )
        )


def main() -> None:
    # Safety: default DRY RUN unless LIVE_TRADING=1
    bot = BinanceBot()
    try:
        bot.run()
    finally:
        bot.release_lock()


if __name__ == "__main__":
    main()
