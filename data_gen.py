"""
Synthetic NQ Futures Data Generator — GBM-based, 5-minute bars.

THE BUG FIXED: The original additive-noise approach added a constant
sigma directly to price each bar, so variance grew as sigma^2 * n_bars
(not sigma^2 * T). Over 98,280 bars (5 yr × 252 days × 78 bars/day)
the standard deviation mushroomed to ~31,000 pts, letting prices go
negative or triple in a random walk explosion.

THE FIX: Geometric Brownian Motion — generate log-returns with
  σ_bar = σ_annual × sqrt(dt),  dt = 1/(252×78)
then compound:  S_t = S_0 × exp(Σ log_returns)
Volatility stays proportional to price level at all times.
"""

import numpy as np
import pandas as pd

# ── NQ futures calibration ────────────────────────────────────────────────────
S0 = 15_000.0          # realistic NQ starting price
ANNUAL_DRIFT = 0.10    # 10 % long-run drift
ANNUAL_VOL   = 0.18    # 18 % realised vol (slightly below 20 % for NQ)
TICK         = 0.25    # minimum price increment
BARS_PER_DAY = 78      # 9:30–16:00 ET in 5-min bars  (390 min / 5)
TRADING_DAYS = 252
BAR_MINUTES  = 5


def generate_nq_bars(
    n_years: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a DataFrame of synthetic NQ 5-min OHLCV bars.

    Parameters
    ----------
    n_years : trading years to simulate
    seed    : RNG seed for reproducibility

    Returns
    -------
    pd.DataFrame  columns: open, high, low, close, volume
                  index  : DatetimeIndex (NYSE trading hours)
    """
    rng = np.random.default_rng(seed)

    n_days = n_years * TRADING_DAYS
    n_bars = n_days * BARS_PER_DAY

    # ── 1. GBM log-returns (THE FIX) ─────────────────────────────────────────
    dt     = 1.0 / (TRADING_DAYS * BARS_PER_DAY)   # fraction of year per bar
    mu     = (ANNUAL_DRIFT - 0.5 * ANNUAL_VOL ** 2) * dt
    sigma  = ANNUAL_VOL * np.sqrt(dt)               # ≈ 0.00114 per bar

    log_ret = rng.normal(mu, sigma, n_bars)
    closes  = S0 * np.exp(np.cumsum(log_ret))       # always positive, no explosion

    # ── 2. Intrabar OHLC ─────────────────────────────────────────────────────
    intra_sigma = ANNUAL_VOL * np.sqrt(dt) * 0.45   # intrabar half-range

    # Open = previous close with small gap
    opens    = np.empty(n_bars)
    opens[0] = S0
    opens[1:] = closes[:-1] * np.exp(rng.normal(0, intra_sigma * 0.15, n_bars - 1))

    high_off = np.abs(rng.normal(0, intra_sigma, n_bars))
    low_off  = np.abs(rng.normal(0, intra_sigma, n_bars))

    highs = np.maximum(opens, closes) * (1 + high_off)
    lows  = np.minimum(opens, closes) * (1 - low_off)

    # Tick rounding
    for arr in (opens, highs, lows, closes):
        arr[:] = np.round(arr / TICK) * TICK

    # ── 3. Volume — log-normal baseline + institutional spikes ───────────────
    volume = rng.lognormal(mean=6.2, sigma=0.4, size=n_bars).astype(np.int64)
    n_spikes  = int(n_bars * 0.02)
    spike_idx = rng.choice(n_bars, size=n_spikes, replace=False)
    volume[spike_idx] = (volume[spike_idx] * rng.integers(4, 10, n_spikes)).astype(np.int64)

    # ── 4. Timestamps — NYSE calendar (bdate_range) ───────────────────────────
    start       = pd.Timestamp("2019-01-02")
    trading_days = pd.bdate_range(start, periods=n_days, freq="B")
    market_open  = pd.Timedelta(hours=9, minutes=30)

    timestamps = []
    for day in trading_days:
        base = pd.Timestamp(day.date()) + market_open
        for i in range(BARS_PER_DAY):
            timestamps.append(base + pd.Timedelta(minutes=i * BAR_MINUTES))

    idx = pd.DatetimeIndex(timestamps, name="timestamp")
    return pd.DataFrame(
        {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volume},
        index=idx,
    )
