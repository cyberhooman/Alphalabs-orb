"""
Fabio Valentini — IVB (Institutional Volume Breakout) Strategy Engine
FundedNext risk framework: daily loss limit, trailing drawdown, profit target.

Strategy logic
--------------
1. Volume spike (current bar volume > spike_mult × rolling average).
2. Close breaks above N-bar high  → Long entry.
   Close breaks below N-bar low   → Short entry.
3. Stop  = entry ± atr_stop_mult × ATR(14)
4. Target = entry ± reward_risk × stop_distance  (2 : 1 default)
5. End-of-day forced exit; max 3 trades per day.

FundedNext $100 K account rules
--------------------------------
  Daily loss limit     : $2,000   (halt trading for the day)
  Max trailing drawdown: $5,000   (account halted permanently)
  Phase-1 profit target: $10,000  (account passes)
  Point value (NQ)     : $20 / pt
  Max contracts        : 3
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional


# ── Constants ─────────────────────────────────────────────────────────────────

POINT_VALUE = 20.0   # NQ futures: $20 per index point

FUNDED_PARAMS = dict(
    account_size      = 100_000.0,
    daily_loss_limit  =   2_000.0,
    max_trailing_dd   =   5_000.0,
    profit_target     =  10_000.0,
    risk_per_trade    =   0.01,       # 1 % of current equity per trade
    max_contracts     =   3,
)

IVB_PARAMS = dict(
    vol_ma_period    = 20,
    vol_spike_mult   = 2.0,
    breakout_lookback = 10,
    atr_period       = 14,
    atr_stop_mult    = 1.5,
    reward_risk      = 2.0,
    max_trades_day   = 3,
)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_time  : pd.Timestamp
    direction   : int          # +1 long, -1 short
    entry_price : float
    stop_price  : float
    target_price: float
    contracts   : int
    exit_time   : Optional[pd.Timestamp] = None
    exit_price  : Optional[float]        = None
    pnl         : float                  = 0.0
    status      : str                    = "open"   # win | loss | eod


# ── Risk manager ──────────────────────────────────────────────────────────────

class FundedNextRisk:
    """Stateful FundedNext risk manager for one simulated account."""

    def __init__(self, params: dict = FUNDED_PARAMS):
        p = params
        self.account      = p["account_size"]
        self.peak         = p["account_size"]
        self.daily_loss   = 0.0
        self.total_pnl    = 0.0
        self.halted       = False        # permanent (trailing DD breach)
        self.day_halted   = False        # temporary (daily limit reached)
        self.phase_passed = False
        self._p           = p

    # ── daily reset ──────────────────────────────────────────────────────────
    def new_day(self):
        self.daily_loss = 0.0
        self.day_halted = False

    # ── record completed trade ────────────────────────────────────────────────
    def record_trade(self, pnl: float):
        self.account   += pnl
        self.total_pnl += pnl
        if pnl < 0:
            self.daily_loss += abs(pnl)
        if self.account > self.peak:
            self.peak = self.account
        self._check_limits()

    def _check_limits(self):
        if self.daily_loss >= self._p["daily_loss_limit"]:
            self.day_halted = True
        trailing_dd = self.peak - self.account
        if trailing_dd >= self._p["max_trailing_dd"]:
            self.halted = True
        if self.total_pnl >= self._p["profit_target"]:
            self.phase_passed = True

    # ── trading gate ──────────────────────────────────────────────────────────
    def can_trade(self) -> bool:
        if self.halted or self.phase_passed:
            return False
        if self.day_halted:
            return False
        return True

    # ── position sizing ───────────────────────────────────────────────────────
    def size_contracts(self, stop_distance: float) -> int:
        risk_dollars = self.account * self._p["risk_per_trade"]
        n = int(risk_dollars / (stop_distance * POINT_VALUE))
        return max(1, min(n, self._p["max_contracts"]))


# ── Indicators (vectorised) ───────────────────────────────────────────────────

def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> np.ndarray:
    prev_c = np.roll(closes, 1)
    prev_c[0] = closes[0]
    tr = np.maximum(highs - lows,
         np.maximum(np.abs(highs - prev_c), np.abs(lows - prev_c)))
    return pd.Series(tr).ewm(span=period, adjust=False).mean().values


def _rolling_max(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=1).max().values

def _rolling_min(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=1).min().values

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    return pd.Series(arr).rolling(window, min_periods=window).mean().values


# ── Main backtest ─────────────────────────────────────────────────────────────

def run_backtest(
    df          : pd.DataFrame,
    ivb_params  : dict = IVB_PARAMS,
    funded_params: dict = FUNDED_PARAMS,
) -> dict:
    """
    Execute IVB strategy on OHLCV dataframe.

    Returns
    -------
    dict with keys: trades, equity_curve, metrics, risk
    """
    ip   = ivb_params
    risk = FundedNextRisk(funded_params)

    closes    = df["close"].values.astype(float)
    highs     = df["high"].values.astype(float)
    lows      = df["low"].values.astype(float)
    volume    = df["volume"].values.astype(float)
    timestamps = df.index

    n = len(df)

    # Pre-compute indicators
    vol_ma    = _rolling_mean(volume, ip["vol_ma_period"])
    atr_vals  = _atr(highs, lows, closes, ip["atr_period"])
    lb        = ip["breakout_lookback"]
    roll_high = _rolling_max(highs, lb)
    roll_low  = _rolling_min(lows, lb)

    warmup = max(ip["vol_ma_period"], lb, ip["atr_period"])

    completed_trades: List[Trade] = []
    equity = np.full(n, risk.account)

    open_trade: Optional[Trade] = None
    day_trades  = 0
    current_day = timestamps[0].date()

    for i in range(warmup, n):
        ts  = timestamps[i]
        day = ts.date()

        # ── day change ────────────────────────────────────────────────────────
        if day != current_day:
            if open_trade is not None:
                # Force exit at open of new day (close[i] used as proxy)
                ct       = open_trade
                ep       = closes[i]
                pnl      = (ep - ct.entry_price) * ct.direction * ct.contracts * POINT_VALUE
                ct.exit_price = ep
                ct.exit_time  = ts
                ct.pnl        = pnl
                ct.status     = "eod"
                risk.record_trade(pnl)
                completed_trades.append(ct)
                open_trade = None

            risk.new_day()
            day_trades  = 0
            current_day = day

        # ── manage open trade ─────────────────────────────────────────────────
        if open_trade is not None:
            ct  = open_trade
            if ct.direction == 1:          # long
                if lows[i] <= ct.stop_price:
                    pnl = (ct.stop_price - ct.entry_price) * ct.contracts * POINT_VALUE
                    ct.exit_price = ct.stop_price
                    ct.exit_time  = ts
                    ct.pnl        = pnl
                    ct.status     = "loss"
                    risk.record_trade(pnl)
                    completed_trades.append(ct)
                    open_trade = None
                elif highs[i] >= ct.target_price:
                    pnl = (ct.target_price - ct.entry_price) * ct.contracts * POINT_VALUE
                    ct.exit_price = ct.target_price
                    ct.exit_time  = ts
                    ct.pnl        = pnl
                    ct.status     = "win"
                    risk.record_trade(pnl)
                    completed_trades.append(ct)
                    open_trade = None
            else:                          # short
                if highs[i] >= ct.stop_price:
                    pnl = (ct.entry_price - ct.stop_price) * ct.contracts * POINT_VALUE
                    ct.exit_price = ct.stop_price
                    ct.exit_time  = ts
                    ct.pnl        = pnl
                    ct.status     = "loss"
                    risk.record_trade(pnl)
                    completed_trades.append(ct)
                    open_trade = None
                elif lows[i] <= ct.target_price:
                    pnl = (ct.entry_price - ct.target_price) * ct.contracts * POINT_VALUE
                    ct.exit_price = ct.target_price
                    ct.exit_time  = ts
                    ct.pnl        = pnl
                    ct.status     = "win"
                    risk.record_trade(pnl)
                    completed_trades.append(ct)
                    open_trade = None

        # ── look for new signal ───────────────────────────────────────────────
        if (open_trade is None
                and risk.can_trade()
                and day_trades < ip["max_trades_day"]
                and not np.isnan(vol_ma[i])
                and volume[i] > ip["vol_spike_mult"] * vol_ma[i]):

            bar_atr = atr_vals[i]
            c       = closes[i]

            if bar_atr > 0:
                if c > roll_high[i - 1]:             # long breakout
                    stop  = c - ip["atr_stop_mult"] * bar_atr
                    dist  = c - stop
                    tgt   = c + ip["reward_risk"] * dist
                    cts   = risk.size_contracts(dist)
                    open_trade = Trade(ts, +1, c, stop, tgt, cts)
                    day_trades += 1

                elif c < roll_low[i - 1]:            # short breakout
                    stop  = c + ip["atr_stop_mult"] * bar_atr
                    dist  = stop - c
                    tgt   = c - ip["reward_risk"] * dist
                    cts   = risk.size_contracts(dist)
                    open_trade = Trade(ts, -1, c, stop, tgt, cts)
                    day_trades += 1

        equity[i] = risk.account

    # ── compile output ────────────────────────────────────────────────────────
    trades_df    = _to_df(completed_trades)
    metrics      = _metrics(trades_df, equity, timestamps, risk)
    equity_curve = pd.Series(equity, index=timestamps, name="equity")

    return dict(trades=trades_df, equity_curve=equity_curve,
                metrics=metrics, risk=risk)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_df(trades: List[Trade]) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=[
            "entry_time", "exit_time", "direction",
            "entry_price", "exit_price", "contracts", "pnl", "status"
        ])
    rows = [
        dict(
            entry_time  = t.entry_time,
            exit_time   = t.exit_time,
            direction   = "Long" if t.direction == 1 else "Short",
            entry_price = t.entry_price,
            exit_price  = t.exit_price,
            contracts   = t.contracts,
            pnl         = t.pnl,
            status      = t.status,
        )
        for t in trades
    ]
    return pd.DataFrame(rows)


def _metrics(trades_df: pd.DataFrame, equity: np.ndarray,
             timestamps, risk: FundedNextRisk) -> dict:
    pnls = trades_df["pnl"].values if not trades_df.empty else np.array([])
    wins = pnls[pnls > 0]
    losses = pnls[pnls < 0]

    win_rate       = len(wins) / len(pnls) if len(pnls) else 0.0
    avg_win        = float(wins.mean())  if len(wins)   else 0.0
    avg_loss       = float(losses.mean()) if len(losses) else 0.0
    profit_factor  = (wins.sum() / abs(losses.sum())
                      if losses.sum() != 0 else float("inf"))

    eq_s      = pd.Series(equity)
    roll_max  = eq_s.cummax()
    dd        = roll_max - eq_s
    max_dd    = float(dd.max())
    max_dd_pct = max_dd / float(roll_max.max()) * 100

    returns   = eq_s.pct_change().dropna()
    bpy       = 252 * 78           # bars per year for 5-min data
    sharpe    = (float(returns.mean() / returns.std()) * np.sqrt(bpy)
                 if returns.std() > 0 else 0.0)

    # Calmar
    ann_ret = (risk.account / FUNDED_PARAMS["account_size"]) ** (1 / 5) - 1
    calmar  = ann_ret / (max_dd / FUNDED_PARAMS["account_size"]) if max_dd > 0 else 0.0

    return dict(
        total_trades   = len(pnls),
        win_rate       = win_rate,
        avg_win        = avg_win,
        avg_loss       = avg_loss,
        profit_factor  = profit_factor,
        total_pnl      = float(pnls.sum()),
        max_drawdown   = max_dd,
        max_dd_pct     = max_dd_pct,
        sharpe_ratio   = sharpe,
        calmar_ratio   = calmar,
        final_equity   = risk.account,
        phase_passed   = risk.phase_passed,
        account_halted = risk.halted,
    )
