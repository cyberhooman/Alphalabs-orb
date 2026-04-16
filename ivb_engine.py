"""
Fabio Valentini — IVB (Initial Balance Breakout) Strategy Engine
================================================================

Model summary
-------------
1. Initial Balance (IB): first `ib_minutes` of the NY cash session
   (default 60 min = 9:30–10:29 ET).

2. Volume Profile framing inside the IB:
   - POC  : price level with the most volume within the IB.
   - VAH/VAL: Value Area High/Low (70 % of IB volume around POC).
   These act as potential re-entry / confluence levels.

3. After the IB closes, wait for a breakout:
   - Price closes ABOVE IB_High  → potential Long
   - Price closes BELOW IB_Low   → potential Short

4. Confirmation — "Strong Delta":
   Delta approximation (no real order-book in synthetic data):
     delta_ratio = (2·close − high − low) / (high − low)
   Range: −1 (close at bar low, pure selling) to +1 (close at bar high).
   We require |delta_ratio| > delta_threshold (default 0.30) in the
   direction of the breakout to confirm aggressive order flow.

5. Entry  : at the close of the confirming bar.
   Stop   : at the OPPOSITE end of the IB range
            (IB_Low for Long; IB_High for Short). ← key spec requirement
   Target : entry ± stop_distance × rr_ratio  (default 1:1)
   Max 1 trade per session; exit at session close if still open.

6. Contract size: 1 NQ contract ($20 / point).

FundedNext overlay
------------------
Pass funded_params=None for the pure strategy (replicating the reel).
Pass funded_params=FUNDED_PARAMS to add halt rules.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Tuple

POINT_VALUE = 20.0   # NQ: $20 per index point

FUNDED_PARAMS = dict(
    account_size     = 100_000.0,
    daily_loss_limit =   2_000.0,
    max_trailing_dd  =   5_000.0,
    profit_target    =  10_000.0,
)

IVB_DEFAULTS = dict(
    ib_minutes       = 60,     # Initial Balance window (15, 30, or 60 min)
    bar_minutes      = 1,      # must match data_gen bar_minutes
    delta_threshold  = 0.40,   # strong-delta confirmation (0=any, 0.4=selective)
    rr_ratio         = 1.0,    # reward:risk (1.0 = 1:1)
)


# ── Volume Profile (simplified) ───────────────────────────────────────────────

def _volume_profile(highs: np.ndarray, lows: np.ndarray,
                    volume: np.ndarray,
                    n_levels: int = 50) -> Tuple[float, float, float]:
    """
    Approximate Point of Control and Value Area from bar data.

    Returns (POC_price, VAH_price, VAL_price).
    Uses a histogram of volume across the price range of the bars.
    """
    if len(highs) == 0:
        return (np.nan, np.nan, np.nan)

    lo = lows.min()
    hi = highs.max()
    if hi == lo:
        return (lo, lo, lo)

    edges  = np.linspace(lo, hi, n_levels + 1)
    bucket = np.zeros(n_levels)

    for i in range(len(highs)):
        # distribute bar volume proportionally across price levels it spans
        bar_lo, bar_hi, vol = lows[i], highs[i], volume[i]
        span = bar_hi - bar_lo if bar_hi > bar_lo else 1e-9
        for k in range(n_levels):
            level_lo = edges[k]
            level_hi = edges[k + 1]
            overlap  = max(0.0, min(bar_hi, level_hi) - max(bar_lo, level_lo))
            bucket[k] += vol * overlap / span

    poc_idx = int(np.argmax(bucket))
    poc     = (edges[poc_idx] + edges[poc_idx + 1]) / 2

    # Value Area: accumulate 70 % of total volume outward from POC
    total      = bucket.sum()
    target_vol = total * 0.70
    accumulated = bucket[poc_idx]
    lo_idx = hi_idx = poc_idx

    while accumulated < target_vol:
        can_expand_lo = lo_idx > 0
        can_expand_hi = hi_idx < n_levels - 1
        if not can_expand_lo and not can_expand_hi:
            break
        add_lo = bucket[lo_idx - 1] if can_expand_lo else -1
        add_hi = bucket[hi_idx + 1] if can_expand_hi else -1
        if add_lo >= add_hi:
            lo_idx    -= 1
            accumulated += bucket[lo_idx]
        else:
            hi_idx    += 1
            accumulated += bucket[hi_idx]

    vah = (edges[hi_idx] + edges[hi_idx + 1]) / 2
    val = (edges[lo_idx] + edges[lo_idx + 1]) / 2
    return poc, vah, val


# ── Delta ratio ───────────────────────────────────────────────────────────────

def _delta_ratio(high: np.ndarray, low: np.ndarray,
                 close: np.ndarray) -> np.ndarray:
    """
    Bar-level delta approximation.
    +1 → close at bar high (full buying aggression)
    −1 → close at bar low  (full selling aggression)
    0  → close at midpoint (neutral)
    """
    rng  = high - low
    mask = rng > 0
    d    = np.zeros(len(close))
    d[mask] = (2 * close[mask] - high[mask] - low[mask]) / rng[mask]
    return d


# ── Trade record ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time  : pd.Timestamp
    direction   : int           # +1 long, -1 short
    entry_price : float
    stop_price  : float         # IB_Low (long) or IB_High (short)
    target_price: float
    ib_high     : float
    ib_low      : float
    poc         : float
    vah         : float
    val         : float
    exit_time   : Optional[pd.Timestamp] = None
    exit_price  : Optional[float]        = None
    pnl         : float = 0.0
    status      : str   = "open"   # win | loss | eod


# ── FundedNext risk state ─────────────────────────────────────────────────────

class _FundedRisk:
    def __init__(self, p: dict):
        self._p      = p
        self.account = p["account_size"]
        self.peak    = p["account_size"]
        self.day_loss = 0.0
        self.halted  = False
        self.passed  = False

    def new_day(self):
        self.day_loss = 0.0

    def record(self, pnl: float):
        self.account += pnl
        if pnl < 0:
            self.day_loss += abs(pnl)
        if self.account > self.peak:
            self.peak = self.account
        if self.day_loss >= self._p["daily_loss_limit"]:
            self.day_loss = 0.0      # halt rest of day; reset counter
        if self.peak - self.account >= self._p["max_trailing_dd"]:
            self.halted = True
        if self.account - self._p["account_size"] >= self._p["profit_target"]:
            self.passed = True

    def can_trade(self) -> bool:
        return not (self.halted or self.passed)


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    df            : pd.DataFrame,
    params        : dict = IVB_DEFAULTS,
    funded_params : Optional[dict] = None,
) -> dict:
    """
    Execute IVB strategy on an OHLCV dataframe.

    Parameters
    ----------
    df            : output of data_gen.generate_nq_bars()
    params        : IVB strategy parameters (see IVB_DEFAULTS)
    funded_params : FundedNext halt rules; None = pure strategy (no halting)

    Returns
    -------
    dict: trades, equity_curve, metrics, [risk]
    """
    p = {**IVB_DEFAULTS, **params}

    closes     = df["close"].values.astype(float)
    highs      = df["high"].values.astype(float)
    lows       = df["low"].values.astype(float)
    volume_arr = df["volume"].values.astype(float)
    ts_index   = df.index
    n          = len(df)

    delta_arr = _delta_ratio(highs, lows, closes)

    ib_bars   = p["ib_minutes"] // p["bar_minutes"]
    rr        = p["rr_ratio"]
    delta_thr = p["delta_threshold"]

    risk      = _FundedRisk(funded_params) if funded_params else None
    start_eq  = (funded_params["account_size"]
                 if funded_params else 100_000.0)
    account   = start_eq
    equity    = np.full(n, start_eq)

    completed : List[Trade] = []
    open_trade: Optional[Trade] = None

    # Per-session accumulators
    ib_high = ib_low = np.nan
    ib_highs_list: List[float] = []
    ib_lows_list : List[float] = []
    ib_vols_list : List[float] = []
    poc = vah = val = np.nan
    day_bar     = 0
    trade_today = False
    cur_day     = ts_index[0].date()

    for i in range(n):
        ts  = ts_index[i]
        day = ts.date()

        # ── session boundary ──────────────────────────────────────────────────
        if day != cur_day:
            # Force-close at previous bar's close
            if open_trade is not None:
                ep  = closes[i - 1]
                pnl = (ep - open_trade.entry_price) * open_trade.direction * POINT_VALUE
                open_trade.exit_price = ep
                open_trade.exit_time  = ts_index[i - 1]
                open_trade.pnl        = pnl
                open_trade.status     = "eod"
                account += pnl
                if risk:
                    risk.record(pnl)
                completed.append(open_trade)
                open_trade = None

            if risk:
                risk.new_day()

            ib_high = ib_low = np.nan
            ib_highs_list.clear()
            ib_lows_list.clear()
            ib_vols_list.clear()
            poc = vah = val = np.nan
            day_bar     = 0
            trade_today = False
            cur_day     = day

        # ── Initial Balance accumulation ──────────────────────────────────────
        if day_bar < ib_bars:
            h, l, v = highs[i], lows[i], volume_arr[i]
            ib_high = h if np.isnan(ib_high) else max(ib_high, h)
            ib_low  = l if np.isnan(ib_low)  else min(ib_low,  l)
            ib_highs_list.append(h)
            ib_lows_list.append(l)
            ib_vols_list.append(v)
            day_bar += 1
            equity[i] = account
            continue

        # Compute volume profile once, on the bar right after IB closes
        if day_bar == ib_bars:
            poc, vah, val = _volume_profile(
                np.array(ib_highs_list),
                np.array(ib_lows_list),
                np.array(ib_vols_list),
            )

        day_bar += 1

        # ── Manage open trade ─────────────────────────────────────────────────
        if open_trade is not None:
            ct = open_trade
            if ct.direction == 1:         # long
                if lows[i] <= ct.stop_price:
                    pnl = (ct.stop_price - ct.entry_price) * POINT_VALUE
                    ct.exit_price = ct.stop_price
                    ct.exit_time  = ts
                    ct.pnl        = pnl
                    ct.status     = "loss"
                    account += pnl
                    if risk: risk.record(pnl)
                    completed.append(ct); open_trade = None
                elif highs[i] >= ct.target_price:
                    pnl = (ct.target_price - ct.entry_price) * POINT_VALUE
                    ct.exit_price = ct.target_price
                    ct.exit_time  = ts
                    ct.pnl        = pnl
                    ct.status     = "win"
                    account += pnl
                    if risk: risk.record(pnl)
                    completed.append(ct); open_trade = None
            else:                          # short
                if highs[i] >= ct.stop_price:
                    pnl = (ct.entry_price - ct.stop_price) * POINT_VALUE
                    ct.exit_price = ct.stop_price
                    ct.exit_time  = ts
                    ct.pnl        = pnl
                    ct.status     = "loss"
                    account += pnl
                    if risk: risk.record(pnl)
                    completed.append(ct); open_trade = None
                elif lows[i] <= ct.target_price:
                    pnl = (ct.entry_price - ct.target_price) * POINT_VALUE
                    ct.exit_price = ct.target_price
                    ct.exit_time  = ts
                    ct.pnl        = pnl
                    ct.status     = "win"
                    account += pnl
                    if risk: risk.record(pnl)
                    completed.append(ct); open_trade = None

        # ── IVB signal detection ──────────────────────────────────────────────
        if (open_trade is None
                and not trade_today
                and not np.isnan(ib_high)
                and (risk is None or risk.can_trade())):

            c  = closes[i]
            dr = delta_arr[i]       # delta ratio for this bar
            ib_range = ib_high - ib_low

            if ib_range > 0:
                if c > ib_high and dr > delta_thr:
                    # Long: stop at IB_Low (opposite end), target 1:n
                    stop   = ib_low
                    dist   = c - stop
                    target = c + rr * dist
                    open_trade = Trade(
                        ts, +1, c, stop, target,
                        ib_high, ib_low, poc, vah, val,
                    )
                    trade_today = True

                elif c < ib_low and dr < -delta_thr:
                    # Short: stop at IB_High (opposite end), target 1:n
                    stop   = ib_high
                    dist   = stop - c
                    target = c - rr * dist
                    open_trade = Trade(
                        ts, -1, c, stop, target,
                        ib_high, ib_low, poc, vah, val,
                    )
                    trade_today = True

        equity[i] = account

    trades_df    = _to_df(completed)
    metrics      = _metrics(trades_df, equity, start_eq)
    equity_curve = pd.Series(equity, index=ts_index, name="equity")

    out = dict(trades=trades_df, equity_curve=equity_curve, metrics=metrics)
    if risk:
        out["risk"] = risk
    return out


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price",
            "ib_high", "ib_low", "poc", "vah", "val",
            "pnl", "status",
        ])
    return pd.DataFrame([dict(
        entry_time  = t.entry_time,
        exit_time   = t.exit_time,
        direction   = "Long" if t.direction == 1 else "Short",
        entry_price = t.entry_price,
        exit_price  = t.exit_price,
        ib_high     = t.ib_high,
        ib_low      = t.ib_low,
        poc         = t.poc,
        vah         = t.vah,
        val         = t.val,
        pnl         = t.pnl,
        status      = t.status,
    ) for t in trades])


def _metrics(trades_df: pd.DataFrame,
             equity: np.ndarray, start_eq: float) -> dict:
    pnls   = trades_df["pnl"].values if not trades_df.empty else np.array([])
    wins   = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    total_pnl     = float(pnls.sum())
    win_rate      = len(wins) / len(pnls) if len(pnls) else 0.0
    avg_win       = float(wins.mean())    if len(wins)   else 0.0
    avg_loss      = float(losses.mean()) if len(losses)  else 0.0
    avg_trade_pnl = float(pnls.mean())   if len(pnls)    else 0.0
    profit_factor = (wins.sum() / abs(losses.sum())
                     if losses.sum() != 0 else float("inf"))

    eq_s     = pd.Series(equity)
    roll_max = eq_s.cummax()
    dd       = roll_max - eq_s
    max_dd   = float(dd.max())
    max_dd_pct = max_dd / float(roll_max.max()) * 100 if roll_max.max() > 0 else 0.0

    returns = eq_s.pct_change().dropna()
    bpy     = 252 * (390 // IVB_DEFAULTS["bar_minutes"])  # bars per year
    sharpe  = (float(returns.mean() / returns.std()) * np.sqrt(bpy)
               if returns.std() > 0 else 0.0)

    calmar = (((equity[-1] / start_eq) ** 0.2 - 1) / (max_dd / start_eq)
              if max_dd > 0 else 0.0)

    return dict(
        total_trades  = int(len(pnls)),
        win_rate      = win_rate,
        avg_win       = avg_win,
        avg_loss      = avg_loss,
        avg_trade_pnl = avg_trade_pnl,
        profit_factor = profit_factor,
        total_pnl     = total_pnl,
        max_drawdown  = max_dd,
        max_dd_pct    = max_dd_pct,
        sharpe_ratio  = sharpe,
        calmar_ratio  = calmar,
        final_equity  = float(equity[-1]),
    )
