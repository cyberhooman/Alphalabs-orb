"""
IVB Backtest Runner — ties data_gen, ivb_engine, monte_carlo, charts together.
"""

import time
from data_gen    import generate_nq_bars
from ivb_engine  import run_backtest
from monte_carlo import run_monte_carlo, equity_fan_curves
from charts      import build_dashboard


def main():
    t0 = time.time()

    # ── 1. Generate synthetic NQ data (GBM — no price explosion) ─────────────
    print("Generating 5-year synthetic NQ data …")
    df = generate_nq_bars(n_years=5, seed=42)
    print(f"  {len(df):,} bars  |  price range "
          f"${df['close'].min():,.0f} – ${df['close'].max():,.0f}")

    # ── 2. Base backtest ──────────────────────────────────────────────────────
    print("\nRunning IVB backtest …")
    result = run_backtest(df)
    m = result["metrics"]
    r = result["risk"]

    print(f"  Trades       : {m['total_trades']:,}")
    print(f"  Win rate     : {m['win_rate']*100:.1f}%")
    print(f"  Profit factor: {m['profit_factor']:.2f}")
    print(f"  Total P&L    : ${m['total_pnl']:+,.0f}")
    print(f"  Max drawdown : ${m['max_drawdown']:,.0f}  ({m['max_dd_pct']:.1f}%)")
    print(f"  Sharpe       : {m['sharpe_ratio']:.2f}")
    print(f"  Calmar       : {m['calmar_ratio']:.2f}")
    print(f"  Final equity : ${r.account:,.0f}")
    print(f"  Phase passed : {r.phase_passed}")

    trades_df = result["trades"]
    if not trades_df.empty:
        trades_df.to_csv("ivb_trades.csv", index=False)
        print(f"  → Trades saved to ivb_trades.csv")

    # ── 3. Monte Carlo (1 000 bootstrap simulations) ─────────────────────────
    print()
    mc = run_monte_carlo(trades_df, n_sims=1_000, verbose=True)

    # Fan curves for chart (200 paths, lighter than 1 000)
    print("Building MC fan curves (200 paths) …")
    fan = equity_fan_curves(trades_df, n_sims=200, seed=7)

    # ── 4. Institutional dashboard ────────────────────────────────────────────
    print("Rendering dashboard …")
    build_dashboard(result, mc, fan, output_path="ivb_dashboard.png")

    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
