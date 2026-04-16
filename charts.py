"""
IVB Institutional Dashboard — dark-theme 6-panel matplotlib figure.

Panels
------
A (top, wide ×3)    : Equity curve
B (mid, wide ×3)    : Underwater drawdown
C (right col ×1, top 2 rows) : Metrics scorecard
D (bottom-left ×2)  : Monte Carlo equity fan + percentile bands
E (bottom-mid ×1)   : Trade P&L distribution
F (bottom-right ×1) : MC summary + EV/trade confidence interval
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from datetime import datetime
from typing import Optional

# ── Palette ───────────────────────────────────────────────────────────────────
BG      = "#0a0e1a"
PANEL   = "#111827"
BORDER  = "#1e2d40"
ACCENT  = "#00d4ff"
GREEN   = "#00e676"
RED     = "#ff1744"
GOLD    = "#ffd740"
TEXT    = "#cfd8dc"
SUBTEXT = "#78909c"
GRID    = "#1a2535"


def _style(ax, title="", xlabel="", ylabel=""):
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
    if title:
        ax.set_title(title, color=ACCENT, fontsize=10,
                     fontweight="bold", pad=6, fontfamily="monospace")
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)


def _year_ticks(ax, n_bars: int, bpy: int):
    ticks = np.arange(0, n_bars, bpy)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"Y{i+1}" for i in range(len(ticks))],
                        fontsize=8, color=SUBTEXT)


# ── Individual panels ─────────────────────────────────────────────────────────

def _panel_equity(ax, equity: pd.Series, metrics: dict, bpy: int):
    _style(ax, title="EQUITY CURVE  —  IVB Strategy  (1 NQ contract · 1:1 R:R)")
    x  = np.arange(len(equity))
    eq = equity.values

    ax.fill_between(x, eq/1000, eq[0]/1000,
                    where=eq >= eq[0], alpha=0.18, color=GREEN, interpolate=True)
    ax.fill_between(x, eq/1000, eq[0]/1000,
                    where=eq <  eq[0], alpha=0.25, color=RED,   interpolate=True)
    ax.plot(x, eq/1000, color=ACCENT, linewidth=1.1, label="Strategy equity")

    target_k = (eq[0] + 10_000) / 1000
    ax.axhline(target_k, color=GREEN, linewidth=0.8,
               linestyle=":", alpha=0.7, label=f"FN target ${target_k:.0f}K")

    # Annotate final P&L
    total_pnl = metrics["total_pnl"]
    color = GREEN if total_pnl >= 0 else RED
    ax.text(0.99, 0.96, f"Total P&L: ${total_pnl:+,.0f}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=10, fontweight="bold", color=color,
            fontfamily="monospace")

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:.0f}K"))
    ax.legend(fontsize=7, loc="upper left", framealpha=0.2,
              labelcolor=TEXT, facecolor=PANEL)
    _year_ticks(ax, len(equity), bpy)


def _panel_drawdown(ax, equity: pd.Series, bpy: int):
    _style(ax, title="UNDERWATER EQUITY DRAWDOWN")
    eq    = equity.values
    peak  = np.maximum.accumulate(eq)
    dd    = peak - eq
    x     = np.arange(len(dd))

    ax.fill_between(x, -dd, 0, color=RED, alpha=0.40)
    ax.plot(x, -dd, color=RED, linewidth=0.6)
    ax.axhline(-2_000, color=GOLD, linewidth=0.8, linestyle="--",
               alpha=0.7, label="Daily limit $2K")
    ax.axhline(-5_000, color=RED, linewidth=0.9, linestyle="--",
               alpha=0.9, label="Max trailing DD $5K")

    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda v, _: f"${abs(v):,.0f}"))
    ax.legend(fontsize=7, loc="lower left", framealpha=0.15,
              labelcolor=TEXT, facecolor=PANEL)
    _year_ticks(ax, len(dd), bpy)


def _metric_card(ax, rows, title, title_color=ACCENT):
    """Generic key-value card (no axes shown)."""
    ax.set_facecolor(PANEL)
    for sp in ax.spines.values():
        sp.set_edgecolor(BORDER)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, color=title_color, fontsize=10,
                 fontweight="bold", pad=6, fontfamily="monospace")

    n = len(rows)
    for i, (label, value, color) in enumerate(rows):
        y = 1 - (i + 1) / (n + 1)
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=9,
                color=SUBTEXT, va="center", fontfamily="monospace")
        ax.text(0.95, y, value, transform=ax.transAxes, fontsize=9,
                color=color, va="center", ha="right", fontweight="bold",
                fontfamily="monospace")
        # divider line
        ax.plot([0.02, 0.98], [y - 0.5/(n+1), y - 0.5/(n+1)],
                color=BORDER, linewidth=0.4, transform=ax.transAxes)


def _panel_scorecard(ax, m: dict):
    rows = [
        ("Total Trades",    f"{m['total_trades']:,}",                      TEXT),
        ("Win Rate",        f"{m['win_rate']*100:.1f}%",
         GREEN if m['win_rate'] > 0.5 else RED),
        ("Avg Win",         f"${m['avg_win']:,.0f}",                       GREEN),
        ("Avg Loss",        f"${m['avg_loss']:,.0f}",                      RED),
        ("Avg Trade P&L",   f"${m['avg_trade_pnl']:+,.0f}",
         GREEN if m['avg_trade_pnl'] > 0 else RED),
        ("Profit Factor",   f"{m['profit_factor']:.2f}",
         GREEN if m['profit_factor'] > 1.5 else GOLD),
        ("Total P&L",       f"${m['total_pnl']:+,.0f}",
         GREEN if m['total_pnl'] > 0 else RED),
        ("Max Drawdown",    f"${m['max_drawdown']:,.0f}  ({m['max_dd_pct']:.1f}%)", RED),
        ("Sharpe Ratio",    f"{m['sharpe_ratio']:.2f}",
         GREEN if m['sharpe_ratio'] > 1 else GOLD),
        ("Calmar Ratio",    f"{m['calmar_ratio']:.2f}",
         GREEN if m['calmar_ratio'] > 0.5 else GOLD),
        ("Final Equity",    f"${m['final_equity']:,.0f}",
         GREEN if m['final_equity'] > 100_000 else RED),
    ]
    _metric_card(ax, rows, "PERFORMANCE  SCORECARD")


def _panel_mc_fan(ax, fan: np.ndarray):
    _style(ax, title="MONTE CARLO  EQUITY FAN  (300 bootstrap paths)")

    x   = np.arange(fan.shape[1])
    p5  = np.nanpercentile(fan, 5,  axis=0)
    p25 = np.nanpercentile(fan, 25, axis=0)
    p50 = np.nanpercentile(fan, 50, axis=0)
    p75 = np.nanpercentile(fan, 75, axis=0)
    p95 = np.nanpercentile(fan, 95, axis=0)

    ax.fill_between(x, p5/1000,  p95/1000, alpha=0.10, color=ACCENT)
    ax.fill_between(x, p25/1000, p75/1000, alpha=0.20, color=ACCENT)
    ax.plot(x, p50/1000, color=ACCENT, linewidth=1.5, label="Median path")
    ax.plot(x, p95/1000, color=GREEN,  linewidth=0.8, linestyle="--", label="p95")
    ax.plot(x, p5/1000,  color=RED,    linewidth=0.8, linestyle="--", label="p5")

    step = max(1, fan.shape[0] // 40)
    for i in range(0, fan.shape[0], step):
        ax.plot(x, fan[i]/1000, color=ACCENT, linewidth=0.25, alpha=0.12)

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:.0f}K"))
    ax.legend(fontsize=7, loc="upper left", framealpha=0.15,
              labelcolor=TEXT, facecolor=PANEL)
    ax.set_xlabel("Trades (bootstrap)", fontsize=8)


def _panel_pnl_dist(ax, trades_df: pd.DataFrame, mc: dict):
    _style(ax, title="TRADE P&L DISTRIBUTION  +  EV/TRADE CI")

    pnls   = trades_df["pnl"].values
    wins   = pnls[pnls >= 0]
    losses = pnls[pnls <  0]

    ax.hist(wins,   bins=35, color=GREEN, alpha=0.70,
            label=f"Wins (n={len(wins)})")
    ax.hist(losses, bins=35, color=RED,   alpha=0.70,
            label=f"Losses (n={len(losses)})")

    ev   = mc["ev_mean"]
    ci_lo = mc["ev_ci_lo"]
    ci_hi = mc["ev_ci_hi"]
    ax.axvline(ev, color=GOLD, linewidth=1.5, linestyle="--",
               label=f"EV ${ev:+,.0f}")
    ax.axvspan(ci_lo, ci_hi, alpha=0.12, color=GOLD,
               label=f"95% CI [{ci_lo:+,.0f}, {ci_hi:+,.0f}]")

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=6.5, framealpha=0.15, labelcolor=TEXT, facecolor=PANEL)
    ax.set_xlabel("P&L per trade ($)", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)


def _panel_mc_stats(ax, mc: dict):
    rows = [
        ("Simulations",       f"{mc['n_sims']:,}",                          TEXT),
        ("EV / trade",        f"${mc['ev_mean']:>+,.0f}",
         GREEN if mc['ev_mean'] > 0 else RED),
        ("EV 95% CI low",     f"${mc['ev_ci_lo']:>+,.0f}",
         GREEN if mc['ev_ci_lo'] > 0 else RED),
        ("EV 95% CI high",    f"${mc['ev_ci_hi']:>+,.0f}",                  GREEN),
        ("P&L  p5",           f"${mc['pnl_p5']:>+,.0f}",
         RED if mc['pnl_p5'] < 0 else GREEN),
        ("P&L  p50",          f"${mc['pnl_p50']:>+,.0f}",                   GOLD),
        ("P&L  p95",          f"${mc['pnl_p95']:>+,.0f}",                   GREEN),
        ("Sharpe  (mean)",    f"{mc['sharpe_mean']:.2f}",
         GREEN if mc['sharpe_mean'] > 1 else GOLD),
        ("Max DD  p95",       f"${mc['max_dd_p95']:,.0f}",                  RED),
        ("% Profitable",      f"{mc['pct_profitable']*100:.1f}%",
         GREEN if mc['pct_profitable'] > 0.5 else GOLD),
        ("% Passed Funded",   f"{mc['pct_passed']*100:.1f}%",
         GREEN if mc['pct_passed'] > 0.5 else GOLD),
        ("% Halted",          f"{mc['pct_halted']*100:.1f}%",
         RED if mc['pct_halted'] > 0.1 else GREEN),
    ]
    _metric_card(ax, rows, "MONTE CARLO  STATISTICS", title_color=GOLD)


# ── Main entry ────────────────────────────────────────────────────────────────

def build_dashboard(
    backtest   : dict,
    mc_summary : dict,
    fan_curves : np.ndarray,
    output_path: str = "ivb_dashboard.png",
    bar_minutes: int = 1,
) -> str:
    """Save the 6-panel institutional IVB dashboard as a PNG."""
    bpy = 252 * (390 // bar_minutes)   # bars per year for x-axis ticks

    fig = plt.figure(figsize=(26, 17), facecolor=BG)
    gs  = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.46, wspace=0.36,
        left=0.04, right=0.97,
        top=0.91,  bottom=0.05,
    )

    # Header
    fig.text(0.5, 0.965,
             "IVB — INITIAL BALANCE BREAKOUT  ·  FABIO VALENTINI MODEL",
             ha="center", fontsize=20, fontweight="bold",
             color=ACCENT, fontfamily="monospace")
    fig.text(0.5, 0.945,
             f"FundedNext $100K Risk Framework  ·  NQ Futures  ·  "
             f"{bar_minutes}-Min Bars  ·  60-Min Initial Balance  ·  "
             f"5-Year Backtest  ·  {datetime.now().strftime('%Y-%m-%d')}",
             ha="center", fontsize=10, color=SUBTEXT)

    eq        = backtest["equity_curve"]
    trades_df = backtest["trades"]
    metrics   = backtest["metrics"]

    ax_eq   = fig.add_subplot(gs[0, :3])
    _panel_equity(ax_eq, eq, metrics, bpy)

    ax_dd   = fig.add_subplot(gs[1, :3])
    _panel_drawdown(ax_dd, eq, bpy)

    ax_sc   = fig.add_subplot(gs[0:2, 3])
    _panel_scorecard(ax_sc, metrics)

    ax_mc   = fig.add_subplot(gs[2, :2])
    _panel_mc_fan(ax_mc, fan_curves)

    ax_dist = fig.add_subplot(gs[2, 2])
    _panel_pnl_dist(ax_dist, trades_df, mc_summary)

    ax_mcs  = fig.add_subplot(gs[2, 3])
    _panel_mc_stats(ax_mcs, mc_summary)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[charts] Dashboard saved → {output_path}")
    return output_path
