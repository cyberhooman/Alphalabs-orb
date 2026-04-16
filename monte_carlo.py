"""
IVB Monte Carlo Stress Test — 20 000 bootstrap simulations.

Method: trade-P&L bootstrap resampling (Efron 1979).
  1. Base backtest produces a population of N trade P&Ls.
  2. Each simulation draws N samples (with replacement) and compounds
     them through the FundedNext risk engine.
  3. Records: total P&L, avg P&L per trade, max drawdown, Sharpe,
     pass/halt flags.

Key output (matching the reel):
  - Expected Value (EV) per trade ± 95 % confidence interval
    The reel reports: $194 EV/trade, CI [$65, $323]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _simulate_one(
    pnl_pool  : np.ndarray,
    n_trades  : int,
    account0  : float,
    daily_loss: float,
    max_trail : float,
    profit_tgt: float,
    rng       : np.random.Generator,
) -> dict:
    """Single bootstrap simulation through the risk engine."""
    draws   = rng.choice(pnl_pool, size=n_trades, replace=True)
    account = account0
    peak    = account0
    day_loss = 0.0
    total_pnl = 0.0
    halted  = False
    passed  = False
    max_dd  = 0.0
    trades_day = 0
    equity  = [account]

    for j, pnl in enumerate(draws):
        if halted or passed:
            break

        # Approximate 1 trade/session → new day every trade
        day_loss = 0.0
        trades_day = 0

        account   += pnl
        total_pnl += pnl
        if pnl < 0:
            day_loss += abs(pnl)
        if account > peak:
            peak = account
        dd = peak - account
        if dd > max_dd:
            max_dd = dd
        equity.append(account)

        if day_loss   >= daily_loss: pass        # day-halted; continue next day
        if dd         >= max_trail : halted = True
        if total_pnl  >= profit_tgt: passed = True

    eq   = np.array(equity)
    rets = np.diff(eq) / eq[:-1]
    sharpe = (rets.mean() / rets.std() * np.sqrt(252)
              if rets.std() > 0 else 0.0)

    n_completed = j + 1 if not (halted or passed) else j
    avg_trade   = total_pnl / max(n_completed, 1)

    return dict(
        total_pnl   = total_pnl,
        avg_trade   = avg_trade,
        final_equity= account,
        max_drawdown= max_dd,
        sharpe      = sharpe,
        passed      = passed,
        halted      = halted,
        n_trades    = n_completed,
    )


def run_monte_carlo(
    base_trades_df: pd.DataFrame,
    n_sims        : int   = 20_000,
    seed          : int   = 0,
    account0      : float = 100_000.0,
    daily_loss    : float =   2_000.0,
    max_trail     : float =   5_000.0,
    profit_tgt    : float =  10_000.0,
    verbose       : bool  = True,
) -> dict:
    """
    Run n_sims bootstrap Monte Carlo simulations.

    Parameters
    ----------
    base_trades_df : DataFrame from ivb_engine.run_backtest()['trades']
    n_sims         : number of simulations (default 20 000, as in reel)

    Returns
    -------
    dict with full summary statistics including EV per trade + 95 % CI.
    """
    if base_trades_df.empty:
        raise ValueError("No trades in base backtest — cannot bootstrap.")

    pnl_pool = base_trades_df["pnl"].values.astype(float)
    n_trades = len(pnl_pool)

    if verbose:
        print(f"[MC] Pool: {n_trades} trades  |  {n_sims:,} simulations …")

    rng      = np.random.default_rng(seed)
    rows     = []
    checkpoint = n_sims // 10

    for i in range(n_sims):
        r = _simulate_one(pnl_pool, n_trades, account0,
                          daily_loss, max_trail, profit_tgt, rng)
        rows.append(r)
        if verbose and (i + 1) % checkpoint == 0:
            print(f"  {i+1:>6,}/{n_sims:,}")

    df = pd.DataFrame(rows)

    # ── EV per trade + 95 % confidence interval ───────────────────────────────
    avg_trades   = df["avg_trade"].values
    ev_mean      = float(avg_trades.mean())
    ev_ci_lo     = float(np.percentile(avg_trades,  2.5))
    ev_ci_hi     = float(np.percentile(avg_trades, 97.5))

    summary = dict(
        n_sims        = n_sims,
        results_df    = df,

        # P&L distribution
        pnl_p5        = float(np.percentile(df["total_pnl"],  5)),
        pnl_p25       = float(np.percentile(df["total_pnl"], 25)),
        pnl_p50       = float(np.percentile(df["total_pnl"], 50)),
        pnl_p75       = float(np.percentile(df["total_pnl"], 75)),
        pnl_p95       = float(np.percentile(df["total_pnl"], 95)),

        # EV per trade (what the reel highlights)
        ev_mean       = ev_mean,
        ev_ci_lo      = ev_ci_lo,
        ev_ci_hi      = ev_ci_hi,

        # Drawdown
        max_dd_mean   = float(df["max_drawdown"].mean()),
        max_dd_p95    = float(np.percentile(df["max_drawdown"], 95)),

        # Sharpe
        sharpe_mean   = float(df["sharpe"].mean()),
        sharpe_std    = float(df["sharpe"].std()),

        # Pass/halt rates
        pct_profitable = float((df["total_pnl"] > 0).mean()),
        pct_passed     = float(df["passed"].mean()),
        pct_halted     = float(df["halted"].mean()),
    )

    if verbose:
        print()
        print("── Monte Carlo Summary ───────────────────────────────")
        print(f"  EV/trade (mean)  : ${ev_mean:>+,.0f}")
        print(f"  EV/trade 95% CI  : [${ev_ci_lo:>+,.0f}, ${ev_ci_hi:>+,.0f}]")
        print(f"  P&L p5/p50/p95   : "
              f"${summary['pnl_p5']:>+,.0f} / "
              f"${summary['pnl_p50']:>+,.0f} / "
              f"${summary['pnl_p95']:>+,.0f}")
        print(f"  Sharpe (mean±σ)  : "
              f"{summary['sharpe_mean']:.2f} ± {summary['sharpe_std']:.2f}")
        print(f"  Max DD p95       : ${summary['max_dd_p95']:,.0f}")
        print(f"  % Profitable     : {summary['pct_profitable']*100:.1f}%")
        print(f"  % Passed funded  : {summary['pct_passed']*100:.1f}%")
        print(f"  % Halted         : {summary['pct_halted']*100:.1f}%")
        print("─────────────────────────────────────────────────────\n")

    return summary


def equity_fan_curves(
    base_trades_df: pd.DataFrame,
    n_sims        : int   = 300,
    seed          : int   = 99,
    account0      : float = 100_000.0,
    daily_loss    : float =   2_000.0,
    max_trail     : float =   5_000.0,
    profit_tgt    : float =  10_000.0,
) -> np.ndarray:
    """
    Return (n_sims × max_steps) equity-curve array for fan-chart plotting.
    NaN-padded to equal length.
    """
    pnl_pool = base_trades_df["pnl"].values.astype(float)
    n_trades = len(pnl_pool)
    rng      = np.random.default_rng(seed)
    curves   = []

    for _ in range(n_sims):
        draws   = rng.choice(pnl_pool, size=n_trades, replace=True)
        account = account0
        peak    = account0
        eq      = [account]
        for pnl in draws:
            account += pnl
            if account > peak:
                peak = account
            eq.append(account)
            if peak - account >= max_trail: break
            if account - account0 >= profit_tgt: break
        curves.append(np.array(eq))

    max_len = max(len(c) for c in curves)
    out = np.full((n_sims, max_len), np.nan)
    for i, c in enumerate(curves):
        out[i, :len(c)] = c
    return out
