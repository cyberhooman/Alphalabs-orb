"""
IVB Institutional Dashboard — dark-theme, 6-panel matplotlib figure.

Panels
------
 A (top-left, wide)  : Equity curve with benchmark NQ overlay
 B (mid-left, wide)  : Underwater drawdown chart
 C (right col, tall) : Metrics scorecard
 D (bottom-left)     : Monte Carlo equity fan + percentile bands
 E (bottom-mid)      : Trade P&L distribution histogram
 F (bottom-right)    : Monte Carlo summary stats card
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from matplotlib.patches import FancyBboxPatch
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


# ── Axes styling ──────────────────────────────────────────────────────────────

def _style_ax(ax, title: str = "", xlabel: str = "", ylabel: str = ""):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.tick_params(colors=SUBTEXT, labelsize=8)
    ax.xaxis.label.set_color(SUBTEXT)
    ax.yaxis.label.set_color(SUBTEXT)
    ax.grid(True, color=GRID, linewidth=0.5, alpha=0.8)
    if title:
        ax.set_title(title, color=ACCENT, fontsize=10, fontweight="bold",
                     pad=6, fontfamily="monospace")
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)


# ── Individual panels ─────────────────────────────────────────────────────────

def _panel_equity(ax, equity: pd.Series, nq_close: Optional[pd.Series] = None):
    _style_ax(ax, title="EQUITY CURVE  —  IVB Strategy vs NQ Benchmark")

    x = np.arange(len(equity))

    # Fill under equity curve
    ax.fill_between(x, equity.values / 1000, equity.values[0] / 1000,
                    where=equity.values >= equity.values[0],
                    alpha=0.18, color=GREEN, interpolate=True)
    ax.fill_between(x, equity.values / 1000, equity.values[0] / 1000,
                    where=equity.values < equity.values[0],
                    alpha=0.25, color=RED, interpolate=True)

    ax.plot(x, equity.values / 1000, color=ACCENT, linewidth=1.2,
            label="IVB Equity ($K)", zorder=3)

    # NQ benchmark (normalised to same starting equity)
    if nq_close is not None:
        nq_norm = nq_close.values / nq_close.values[0] * equity.values[0] / 1000
        # Downsample for visual clarity
        step = max(1, len(nq_norm) // len(equity))
        ax.plot(x, nq_norm[:len(x)], color=GOLD, linewidth=0.7,
                alpha=0.5, linestyle="--", label="NQ (normalised)", zorder=2)

    # Phase-target line
    target_k = (equity.values[0] + 10_000) / 1000
    ax.axhline(target_k, color=GREEN, linewidth=0.8, linestyle=":",
               alpha=0.7, label=f"FundedNext target ${target_k:.0f}K")

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:.0f}K"))
    ax.legend(fontsize=7, loc="upper left", framealpha=0.2,
              labelcolor=TEXT, facecolor=PANEL)

    # X-tick: show years
    n = len(equity)
    bpy = 252 * 78
    year_ticks = np.arange(0, n, bpy)
    labels = [f"Y{i+1}" for i in range(len(year_ticks))]
    ax.set_xticks(year_ticks)
    ax.set_xticklabels(labels, fontsize=8, color=SUBTEXT)


def _panel_drawdown(ax, equity: pd.Series):
    _style_ax(ax, title="UNDERWATER DRAWDOWN", ylabel="Drawdown ($)")

    eq    = equity.values
    peak  = np.maximum.accumulate(eq)
    dd    = peak - eq
    x     = np.arange(len(dd))

    ax.fill_between(x, -dd, 0, color=RED, alpha=0.45)
    ax.plot(x, -dd, color=RED, linewidth=0.7)

    # Daily / max trailing limit lines
    ax.axhline(-2_000, color=GOLD, linewidth=0.8, linestyle="--",
               alpha=0.7, label="Daily limit $2K")
    ax.axhline(-5_000, color=RED, linewidth=0.9, linestyle="--",
               alpha=0.9, label="Max trailing DD $5K")

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${abs(v):,.0f}"))
    ax.legend(fontsize=7, loc="lower left", framealpha=0.15,
              labelcolor=TEXT, facecolor=PANEL)

    bpy = 252 * 78
    n   = len(dd)
    ax.set_xticks(np.arange(0, n, bpy))
    ax.set_xticklabels([f"Y{i+1}" for i in range(len(np.arange(0, n, bpy)))],
                        fontsize=8, color=SUBTEXT)


def _panel_scorecard(ax, metrics: dict, risk):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("PERFORMANCE  SCORECARD", color=ACCENT, fontsize=10,
                 fontweight="bold", pad=6, fontfamily="monospace")

    m = metrics
    rows = [
        ("Total Trades",    f"{m['total_trades']:,}",              TEXT),
        ("Win Rate",        f"{m['win_rate']*100:.1f}%",           GREEN if m['win_rate'] > 0.5 else RED),
        ("Avg Win",         f"${m['avg_win']:,.0f}",               GREEN),
        ("Avg Loss",        f"${m['avg_loss']:,.0f}",              RED),
        ("Profit Factor",   f"{m['profit_factor']:.2f}",           GREEN if m['profit_factor'] > 1.5 else GOLD),
        ("Total P&L",       f"${m['total_pnl']:+,.0f}",            GREEN if m['total_pnl'] > 0 else RED),
        ("Max Drawdown",    f"${m['max_drawdown']:,.0f}  ({m['max_dd_pct']:.1f}%)", RED),
        ("Sharpe Ratio",    f"{m['sharpe_ratio']:.2f}",            GREEN if m['sharpe_ratio'] > 1 else GOLD),
        ("Calmar Ratio",    f"{m['calmar_ratio']:.2f}",            GREEN if m['calmar_ratio'] > 0.5 else GOLD),
        ("Final Equity",    f"${risk.account:,.0f}",               GREEN if risk.account > 100_000 else RED),
        ("Phase Passed",    "YES ✓" if risk.phase_passed else "NO", GREEN if risk.phase_passed else SUBTEXT),
        ("Account Halted",  "YES" if risk.halted else "NO",        RED if risk.halted else GREEN),
    ]

    n = len(rows)
    for i, (label, value, color) in enumerate(rows):
        y = 1 - (i + 1) / (n + 1)
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=9,
                color=SUBTEXT, va="center", fontfamily="monospace")
        ax.text(0.95, y, value, transform=ax.transAxes, fontsize=9,
                color=color, va="center", ha="right", fontweight="bold",
                fontfamily="monospace")
        # divider
        ax.plot([0.02, 0.98], [y - 0.5 / (n + 1), y - 0.5 / (n + 1)],
                color=BORDER, linewidth=0.4, transform=ax.transAxes)


def _panel_mc_fan(ax, fan_curves: np.ndarray, equity: pd.Series):
    _style_ax(ax, title="MONTE CARLO FAN  (200 paths, bootstrap)")

    n_steps = fan_curves.shape[1]
    x = np.arange(n_steps)

    # Percentile bands
    p5  = np.nanpercentile(fan_curves, 5,  axis=0)
    p25 = np.nanpercentile(fan_curves, 25, axis=0)
    p50 = np.nanpercentile(fan_curves, 50, axis=0)
    p75 = np.nanpercentile(fan_curves, 75, axis=0)
    p95 = np.nanpercentile(fan_curves, 95, axis=0)

    ax.fill_between(x, p5/1000,  p95/1000, alpha=0.12, color=ACCENT)
    ax.fill_between(x, p25/1000, p75/1000, alpha=0.22, color=ACCENT)
    ax.plot(x, p50/1000, color=ACCENT, linewidth=1.5, label="Median")
    ax.plot(x, p95/1000, color=GREEN, linewidth=0.8, linestyle="--", label="p95")
    ax.plot(x, p5/1000,  color=RED,   linewidth=0.8, linestyle="--", label="p5")

    # Overlay a few individual paths (semi-transparent)
    for i in range(0, min(30, fan_curves.shape[0])):
        ax.plot(x, fan_curves[i]/1000, color=ACCENT, linewidth=0.3, alpha=0.15)

    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:.0f}K"))
    ax.legend(fontsize=7, loc="upper left", framealpha=0.15,
              labelcolor=TEXT, facecolor=PANEL)
    ax.set_xlabel("Trades", fontsize=8)


def _panel_pnl_dist(ax, trades_df: pd.DataFrame):
    _style_ax(ax, title="TRADE P&L DISTRIBUTION")

    pnls = trades_df["pnl"].values
    wins   = pnls[pnls >= 0]
    losses = pnls[pnls <  0]

    bins = 40
    ax.hist(wins,   bins=bins, color=GREEN, alpha=0.7, label=f"Wins  (n={len(wins)})")
    ax.hist(losses, bins=bins, color=RED,   alpha=0.7, label=f"Losses (n={len(losses)})")
    ax.axvline(np.mean(pnls), color=GOLD, linewidth=1.2,
               linestyle="--", label=f"Mean ${np.mean(pnls):,.0f}")

    ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.legend(fontsize=7, framealpha=0.15, labelcolor=TEXT, facecolor=PANEL)
    ax.set_xlabel("P&L per trade", fontsize=8)
    ax.set_ylabel("Count", fontsize=8)


def _panel_mc_stats(ax, mc: dict):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("MONTE CARLO  STATISTICS", color=GOLD, fontsize=10,
                 fontweight="bold", pad=6, fontfamily="monospace")

    rows = [
        ("Simulations",    f"{mc['n_sims']:,}",                   TEXT),
        ("P&L  p5",        f"${mc['pnl_p5']:>+,.0f}",            RED if mc['pnl_p5'] < 0 else GREEN),
        ("P&L  p25",       f"${mc['pnl_p25']:>+,.0f}",           GOLD),
        ("P&L  p50",       f"${mc['pnl_p50']:>+,.0f}",           GOLD),
        ("P&L  p75",       f"${mc['pnl_p75']:>+,.0f}",           GREEN),
        ("P&L  p95",       f"${mc['pnl_p95']:>+,.0f}",           GREEN),
        ("Sharpe  mean",   f"{mc['sharpe_mean']:.2f}",            GREEN if mc['sharpe_mean'] > 1 else GOLD),
        ("Sharpe  σ",      f"{mc['sharpe_std']:.2f}",             TEXT),
        ("Max DD  mean",   f"${mc['max_dd_mean']:,.0f}",          GOLD),
        ("Max DD  p95",    f"${mc['max_dd_p95']:,.0f}",           RED),
        ("% Profitable",   f"{mc['pct_profitable']*100:.1f}%",   GREEN),
        ("% Passed Funded",f"{mc['pct_passed']*100:.1f}%",       GREEN if mc['pct_passed'] > 0.5 else GOLD),
        ("% Halted",       f"{mc['pct_halted']*100:.1f}%",       RED if mc['pct_halted'] > 0.1 else GREEN),
    ]

    n = len(rows)
    for i, (label, value, color) in enumerate(rows):
        y = 1 - (i + 1) / (n + 1)
        ax.text(0.05, y, label, transform=ax.transAxes, fontsize=8.5,
                color=SUBTEXT, va="center", fontfamily="monospace")
        ax.text(0.95, y, value, transform=ax.transAxes, fontsize=8.5,
                color=color, va="center", ha="right", fontweight="bold",
                fontfamily="monospace")
        ax.plot([0.02, 0.98], [y - 0.5 / (n + 1), y - 0.5 / (n + 1)],
                color=BORDER, linewidth=0.4, transform=ax.transAxes)


# ── Main entry point ──────────────────────────────────────────────────────────

def build_dashboard(
    backtest    : dict,
    mc_summary  : dict,
    fan_curves  : np.ndarray,
    output_path : str = "ivb_dashboard.png",
) -> str:
    """
    Compose and save the 6-panel institutional dashboard.

    Parameters
    ----------
    backtest    : return value of ivb_engine.run_backtest()
    mc_summary  : return value of monte_carlo.run_monte_carlo()
    fan_curves  : return value of monte_carlo.equity_fan_curves()
    output_path : file to save (PNG)
    """
    fig = plt.figure(figsize=(26, 17), facecolor=BG)

    gs = gridspec.GridSpec(
        3, 4, figure=fig,
        hspace=0.48, wspace=0.38,
        left=0.04, right=0.97,
        top=0.91, bottom=0.05,
    )

    # ── Header ───────────────────────────────────────────────────────────────
    fig.text(0.5, 0.964,
             "IVB — INSTITUTIONAL VOLUME BREAKOUT  |  FABIO VALENTINI MODEL",
             ha="center", fontsize=20, fontweight="bold", color=ACCENT,
             fontfamily="monospace")
    fig.text(0.5, 0.944,
             f"FundedNext $100K Risk Framework  •  NQ Futures  •  "
             f"5-Min Bars  •  5-Year Synthetic Backtest  •  "
             f"Generated {datetime.now().strftime('%Y-%m-%d  %H:%M')}",
             ha="center", fontsize=10, color=SUBTEXT)

    equity      = backtest["equity_curve"]
    trades_df   = backtest["trades"]
    metrics     = backtest["metrics"]
    risk        = backtest["risk"]

    # ── Panel A: Equity curve (spans 3 columns, 1 row) ────────────────────────
    ax_eq = fig.add_subplot(gs[0, :3])
    _panel_equity(ax_eq, equity)

    # ── Panel B: Drawdown (spans 3 columns, 1 row) ────────────────────────────
    ax_dd = fig.add_subplot(gs[1, :3])
    _panel_drawdown(ax_dd, equity)

    # ── Panel C: Scorecard (right column, top 2 rows) ─────────────────────────
    ax_sc = fig.add_subplot(gs[0:2, 3])
    _panel_scorecard(ax_sc, metrics, risk)

    # ── Panel D: MC fan (bottom-left, 2 cols) ────────────────────────────────
    ax_mc = fig.add_subplot(gs[2, :2])
    _panel_mc_fan(ax_mc, fan_curves, equity)

    # ── Panel E: P&L distribution ─────────────────────────────────────────────
    ax_dist = fig.add_subplot(gs[2, 2])
    _panel_pnl_dist(ax_dist, trades_df)

    # ── Panel F: MC statistics card ───────────────────────────────────────────
    ax_mc_stats = fig.add_subplot(gs[2, 3])
    _panel_mc_stats(ax_mc_stats, mc_summary)

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"[charts] Dashboard saved → {output_path}")
    return output_path
