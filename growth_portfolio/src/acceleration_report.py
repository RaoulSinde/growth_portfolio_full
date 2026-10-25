from __future__ import annotations

import base64
import html
import io
import json
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
DATA = ROOT / "data" / "processed"
OUTPUT = REPORTS / "acceleration_report.html"

STRATEGY_COLOR = "#0f766e"
BENCHMARK_COLOR = "#64748b"
ACCENT_COLOR = "#f59e0b"
POS_COLOR = "#15803d"
NEG_COLOR = "#b91c1c"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": "#cbd5e1",
        "axes.labelcolor": "#334155",
        "axes.titlecolor": "#0f172a",
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.color": "#e2e8f0",
        "grid.linewidth": 0.8,
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "legend.frameon": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    }
)

CSS = """
:root {
  --ink: #0f172a;
  --muted: #64748b;
  --line: #e2e8f0;
  --surface: #ffffff;
  --canvas: #eef2f7;
  --teal: #0f766e;
  --teal-dark: #115e59;
  --teal-soft: #ccfbf1;
  --amber: #f59e0b;
  --pos: #15803d;
  --neg: #b91c1c;
  --shadow: 0 10px 30px rgba(15, 23, 42, .07);
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--canvas);
  color: var(--ink);
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--teal-dark); text-decoration: none; }
a:hover { text-decoration: underline; }
.wrap { max-width: 1280px; margin: 0 auto; padding: 28px 22px 54px; }
.hero {
  position: relative;
  overflow: hidden;
  border-radius: 22px;
  padding: 38px 40px 34px;
  color: #f8fafc;
  background:
    radial-gradient(circle at 88% 18%, rgba(245, 158, 11, .24), transparent 26%),
    radial-gradient(circle at 12% 92%, rgba(20, 184, 166, .30), transparent 34%),
    linear-gradient(135deg, #0b2d2b 0%, #0f172a 55%, #1e293b 100%);
  box-shadow: var(--shadow);
}
.hero .eyebrow {
  text-transform: uppercase;
  letter-spacing: .16em;
  font-size: .72rem;
  font-weight: 700;
  color: #99f6e4;
  margin-bottom: 10px;
}
.hero h1 { margin: 0 0 10px; font-size: clamp(1.8rem, 3.4vw, 2.9rem); line-height: 1.12; letter-spacing: -.02em; }
.hero p.lead { margin: 0; max-width: 840px; color: #cbd5e1; font-size: 1.02rem; }
.hero .period { margin-top: 18px; color: #e2e8f0; font-size: .95rem; }
.badges { display: flex; flex-wrap: wrap; gap: 9px; margin-top: 20px; }
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255, 255, 255, .19);
  background: rgba(255, 255, 255, .09);
  color: #f8fafc;
  font-size: .78rem;
  font-weight: 600;
  letter-spacing: .01em;
}
.badge b { color: #fbbf24; font-weight: 800; }
.toc {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 18px 0 0;
  padding: 12px 14px;
  background: rgba(255, 255, 255, .92);
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: 0 4px 14px rgba(15, 23, 42, .04);
}
.toc a {
  padding: 5px 9px;
  border-radius: 8px;
  font-size: .8rem;
  font-weight: 600;
  color: #334155;
}
.toc a:hover { background: #f1f5f9; text-decoration: none; }
.section {
  margin-top: 24px;
  padding: 28px 30px 30px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 18px;
  box-shadow: var(--shadow);
}
.section-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 18px; margin-bottom: 18px; }
.kicker {
  display: inline-block;
  margin-bottom: 5px;
  color: var(--teal);
  font-size: .72rem;
  font-weight: 800;
  letter-spacing: .14em;
  text-transform: uppercase;
}
h2 { margin: 0; font-size: 1.45rem; letter-spacing: -.02em; line-height: 1.2; }
h3 { margin: 26px 0 12px; font-size: 1.06rem; letter-spacing: -.01em; }
h4 { margin: 18px 0 8px; font-size: .92rem; }
p { margin: 0 0 12px; }
p.note, .note {
  color: #475569;
  font-size: .82rem;
  line-height: 1.55;
}
.muted { color: var(--muted); }
.small { font-size: .78rem; }
.kpi-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(205px, 1fr));
  gap: 14px;
  margin-top: 18px;
}
.kpi {
  position: relative;
  overflow: hidden;
  padding: 17px 18px 16px;
  border: 1px solid var(--line);
  border-radius: 14px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}
.kpi::after {
  content: "";
  position: absolute;
  inset: auto 0 0 0;
  height: 3px;
  background: linear-gradient(90deg, var(--teal), #14b8a6);
}
.kpi.amber::after { background: linear-gradient(90deg, #d97706, #fbbf24); }
.kpi.slate::after { background: linear-gradient(90deg, #475569, #94a3b8); }
.kpi .label { color: var(--muted); font-size: .72rem; font-weight: 700; text-transform: uppercase; letter-spacing: .09em; }
.kpi .value { margin: 7px 0 2px; font-size: 1.65rem; font-weight: 800; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
.kpi .sub { color: #475569; font-size: .78rem; line-height: 1.4; }
.pos { color: var(--pos); }
.neg { color: var(--neg); }
.neutral { color: #475569; }
.chip {
  display: inline-block;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--teal-soft);
  color: var(--teal-dark);
  font-size: .72rem;
  font-weight: 700;
}
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid-3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; }
ul.clean { margin: 8px 0 0; padding-left: 20px; }
ul.clean li { margin: 9px 0; }
ul.clean li::marker { color: var(--teal); }
.callout {
  margin: 18px 0;
  padding: 15px 17px;
  border-left: 4px solid var(--amber);
  border-radius: 0 12px 12px 0;
  background: #fffbeb;
  color: #78350f;
  font-size: .88rem;
}
.callout strong { color: #92400e; }
.method-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; margin-top: 16px; }
.method-card { padding: 18px; border: 1px solid var(--line); border-radius: 14px; background: #fbfdff; }
.method-card h4 { margin-top: 0; color: var(--teal-dark); }
.method-card p:last-child { margin-bottom: 0; }
.formula {
  margin: 12px 0;
  padding: 13px 15px;
  border: 1px dashed #99f6e4;
  border-radius: 10px;
  background: #f0fdfa;
  font-family: "Cascadia Mono", "Segoe UI Mono", Consolas, monospace;
  font-size: .82rem;
  overflow-x: auto;
  color: #134e4a;
}
.table-wrap { width: 100%; overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; margin: 12px 0 6px; }
table.data { width: 100%; border-collapse: collapse; font-size: .82rem; background: #fff; }
table.data caption { caption-side: top; text-align: left; padding: 10px 12px; color: var(--muted); font-size: .78rem; }
table.data th {
  position: sticky;
  top: 0;
  z-index: 1;
  padding: 10px 11px;
  background: #f1f5f9;
  border-bottom: 1px solid #cbd5e1;
  color: #334155;
  font-size: .72rem;
  text-transform: uppercase;
  letter-spacing: .055em;
  white-space: nowrap;
  text-align: left;
}
table.data td { padding: 9px 11px; border-bottom: 1px solid #eef2f7; vertical-align: top; }
table.data tbody tr:nth-child(even) { background: #fbfdff; }
table.data tbody tr:hover { background: #f0fdfa; }
table.data tbody tr:last-child td { border-bottom: none; }
table.data tr.highlight { background: #fef3c7 !important; }
table.data tr.total { background: #f1f5f9; font-weight: 700; }
.num { text-align: right !important; white-space: nowrap; font-variant-numeric: tabular-nums; }
table.compact td, table.compact th { padding: 7px 9px; }
.figure { margin: 18px 0 6px; }
.figure img { display: block; width: 100%; height: auto; border: 1px solid var(--line); border-radius: 12px; background: #fff; }
.figure figcaption { margin-top: 8px; color: var(--muted); font-size: .78rem; }
.figure-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 18px; align-items: start; }
.stat-strip { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; margin: 14px 0; }
.stat { padding: 12px 13px; border: 1px solid var(--line); border-radius: 11px; background: #f8fafc; }
.stat .k { color: var(--muted); font-size: .68rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
.stat .v { margin-top: 4px; font-size: 1.08rem; font-weight: 800; font-variant-numeric: tabular-nums; }
.limits { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 12px; }
.limit {
  padding: 14px 15px;
  border: 1px solid #fecaca;
  border-radius: 12px;
  background: #fff7f7;
  color: #7f1d1d;
  font-size: .84rem;
}
.limit b { display: block; margin-bottom: 3px; color: #991b1b; }
.footer {
  margin-top: 24px;
  padding: 20px 8px 0;
  color: #64748b;
  font-size: .78rem;
  text-align: center;
}
.tag {
  display: inline-block;
  padding: 2px 7px;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  color: #475569;
  font-size: .7rem;
  font-weight: 600;
  white-space: nowrap;
}
.matrix-cell { min-width: 120px; }
@media (max-width: 980px) {
  .wrap { padding: 18px 12px 40px; }
  .hero { padding: 28px 22px; border-radius: 16px; }
  .section { padding: 22px 18px; border-radius: 14px; }
  .grid-2, .grid-3, .method-grid, .figure-grid, .limits { grid-template-columns: 1fr; }
  .section-head { flex-direction: column; gap: 4px; }
}
@media print {
  body { background: #fff; }
  .wrap { max-width: none; padding: 0; }
  .hero, .section { box-shadow: none; break-inside: avoid; }
  .toc { display: none; }
  .section { page-break-inside: avoid; }
}
"""


def esc(value) -> str:
    return html.escape("" if value is None else str(value))


def is_missing(value) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def fr(value: float, digits: int = 2, signed: bool = False) -> str:
    if is_missing(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return esc(value)
    if not math.isfinite(number):
        return "—"
    template = f"{{:+.{digits}f}}" if signed else f"{{:.{digits}f}}"
    return template.format(number).replace(".", ",")


def pct(value, digits: int = 1, signed: bool = False) -> str:
    if is_missing(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{fr(number * 100.0, digits, signed)} %"


def pp(value, digits: int = 1, signed: bool = True) -> str:
    if is_missing(value):
        return "—"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    return f"{fr(number * 100.0, digits, signed)} pp"


def sign_class(value) -> str:
    if is_missing(value):
        return "neutral"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "neutral"
    if number > 0:
        return "pos"
    if number < 0:
        return "neg"
    return "neutral"


def signed_pct_cell(value, digits: int = 1, cls: str = "num") -> str:
    return f'<td class="{cls} {sign_class(value)}">{pct(value, digits, signed=True)}</td>'


def table(headers, rows, num_cols=None, row_classes=None, table_class="data") -> str:
    num_cols = set(num_cols or [])
    parts = ['<div class="table-wrap"><table class="' + table_class + '">', "<thead><tr>"]
    for index, header in enumerate(headers):
        klass = "num" if index in num_cols else ""
        parts.append(f'<th class="{klass}">{header}</th>')
    parts.append("</tr></thead><tbody>")
    for row_index, row in enumerate(rows):
        extra = ""
        if row_classes and row_index < len(row_classes) and row_classes[row_index]:
            extra = f' class="{row_classes[row_index]}"'
        parts.append(f"<tr{extra}>")
        for cell in row:
            parts.append(str(cell))
        parts.append("</tr>")
    parts.append("</tbody></table></div>")
    return "".join(parts)


def figure_from_base64(encoded: str, alt: str, caption: str) -> str:
    return (
        '<figure class="figure">'
        f'<img src="data:image/png;base64,{encoded}" alt="{esc(alt)}">'
        f"<figcaption>{caption}</figcaption>"
        "</figure>"
    )


def fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=145, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pct_axis(axis, decimals: int = 0) -> None:
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.{decimals}f} %"))


def style_axis(axis) -> None:
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def load_data() -> dict:
    with open(REPORTS / "acceleration_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    with open(REPORTS / "benchmark_diagnostics.json", "r", encoding="utf-8") as handle:
        benchmark_diagnostics = json.load(handle)

    backtest = pd.read_csv(REPORTS / "acceleration_backtest.csv")
    backtest["date"] = pd.to_datetime(backtest["date"])

    annual = pd.read_csv(REPORTS / "acceleration_annual.csv", index_col=0)
    quarterly = pd.read_csv(REPORTS / "acceleration_quarterly.csv", index_col=0)

    sector = pd.read_csv(REPORTS / "acceleration_sector_exposure.csv")
    sector["date"] = pd.to_datetime(sector["date"])

    coverage = pd.read_csv(REPORTS / "acceleration_coverage.csv")
    coverage["date"] = pd.to_datetime(coverage["date"])

    portfolio = pd.read_csv(REPORTS / "acceleration_portfolio_summary.csv")
    portfolio["date"] = pd.to_datetime(portfolio["date"])

    holdings = pd.read_csv(REPORTS / "acceleration_top_holdings.csv")
    holdings["date"] = pd.to_datetime(holdings["date"])

    contributions = pd.read_csv(REPORTS / "acceleration_contributions.csv")
    sensitivity = pd.read_csv(REPORTS / "acceleration_sensitivity.csv")

    weights_daily = pd.read_parquet(REPORTS / "acceleration_weights_daily.parquet")

    return {
        "summary": summary,
        "benchmark_diagnostics": benchmark_diagnostics,
        "backtest": backtest,
        "annual": annual,
        "quarterly": quarterly,
        "sector": sector,
        "coverage": coverage,
        "portfolio": portfolio,
        "holdings": holdings,
        "contributions": contributions,
        "sensitivity": sensitivity,
        "weights_daily": weights_daily,
    }


def chart_nav_drawdown(data: dict) -> str:
    backtest = data["backtest"]
    dates = backtest["date"]
    nav_strategy = backtest["nav_strategy"]
    nav_benchmark = backtest["nav_benchmark"]
    drawdown_strategy = nav_strategy / nav_strategy.cummax() - 1.0
    drawdown_benchmark = nav_benchmark / nav_benchmark.cummax() - 1.0

    fig, (axis_nav, axis_dd) = plt.subplots(
        2,
        1,
        figsize=(12.4, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.08},
    )
    axis_nav.plot(dates, nav_strategy, color=STRATEGY_COLOR, linewidth=2.2, label="Stratégie Accélération")
    axis_nav.plot(dates, nav_benchmark, color=BENCHMARK_COLOR, linewidth=1.7, label="Benchmark MSCI USA Growth (prix)")
    axis_nav.set_ylabel("Indice base 100")
    axis_nav.set_title("Valeur liquidative et drawdown", loc="left")
    axis_nav.legend(loc="upper left", ncol=2)
    axis_nav.annotate(
        f"{nav_strategy.iloc[-1]:.1f}",
        xy=(dates.iloc[-1], nav_strategy.iloc[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        color=STRATEGY_COLOR,
        fontweight="bold",
        fontsize=10,
        va="center",
    )
    axis_nav.annotate(
        f"{nav_benchmark.iloc[-1]:.1f}",
        xy=(dates.iloc[-1], nav_benchmark.iloc[-1]),
        xytext=(8, 0),
        textcoords="offset points",
        color=BENCHMARK_COLOR,
        fontweight="bold",
        fontsize=10,
        va="center",
    )
    axis_nav.set_ylim(70, max(nav_strategy.max(), nav_benchmark.max()) * 1.08)
    style_axis(axis_nav)

    axis_dd.fill_between(dates, drawdown_strategy, 0, color=STRATEGY_COLOR, alpha=0.18)
    axis_dd.plot(dates, drawdown_strategy, color=STRATEGY_COLOR, linewidth=1.7, label="Stratégie")
    axis_dd.plot(dates, drawdown_benchmark, color=BENCHMARK_COLOR, linewidth=1.2, linestyle="--", label="Benchmark")
    axis_dd.axhline(0, color="#94a3b8", linewidth=0.8)
    axis_dd.set_ylabel("Drawdown")
    axis_dd.set_ylim(min(drawdown_strategy.min(), drawdown_benchmark.min()) * 1.12, 0.02)
    pct_axis(axis_dd, 0)
    axis_dd.legend(loc="lower left", ncol=2)
    axis_dd.xaxis.set_major_locator(mdates.YearLocator())
    axis_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_axis(axis_dd)

    fig.subplots_adjust(left=0.065, right=0.975, top=0.95, bottom=0.07, hspace=0.08)
    return fig_to_base64(fig)


def chart_annual_returns(data: dict) -> str:
    annual = data["annual"]
    years = [str(index) for index in annual.index]
    strategy = annual["Strategie"].astype(float).to_numpy()
    benchmark = annual["Benchmark"].astype(float).to_numpy()
    positions = np.arange(len(years))
    width = 0.36

    fig, axis = plt.subplots(figsize=(11.4, 5.1))
    bars_strategy = axis.bar(
        positions - width / 2,
        strategy,
        width,
        label="Stratégie",
        color=STRATEGY_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    bars_benchmark = axis.bar(
        positions + width / 2,
        benchmark,
        width,
        label="Benchmark",
        color=BENCHMARK_COLOR,
        edgecolor="white",
        linewidth=0.6,
    )
    axis.axhline(0, color="#94a3b8", linewidth=0.9)
    axis.set_xticks(positions, years)
    axis.set_ylabel("Rendement annuel")
    axis.set_title("Rendements calendaires : stratégie vs benchmark", loc="left")
    axis.legend(loc="upper left", ncol=2)
    pct_axis(axis, 0)
    axis.set_ylim(min(strategy.min(), benchmark.min()) * 1.25, max(strategy.max(), benchmark.max()) * 1.25)

    for bars in (bars_strategy, bars_benchmark):
        for bar in bars:
            height = bar.get_height()
            axis.annotate(
                f"{height * 100:.0f} %",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4 if height >= 0 else -12),
                textcoords="offset points",
                ha="center",
                va="bottom" if height >= 0 else "top",
                fontsize=8.5,
                color="#334155",
            )
    style_axis(axis)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_quarterly_heatmap(data: dict) -> str:
    quarterly = data["quarterly"].copy()
    labels = quarterly.index.astype(str)
    quarterly["year"] = labels.str.slice(0, 4).astype(int)
    quarterly["quarter"] = labels.str.slice(-1).astype(int)
    strategy_pivot = quarterly.pivot(index="year", columns="quarter", values="Strategie")
    benchmark_pivot = quarterly.pivot(index="year", columns="quarter", values="Benchmark")
    vmax = float(np.nanmax(np.abs(np.concatenate([strategy_pivot.to_numpy().ravel(), benchmark_pivot.to_numpy().ravel()]))))
    vmax = max(vmax, 0.01)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.9), sharey=True)
    panels = [
        (strategy_pivot, "Stratégie Accélération"),
        (benchmark_pivot, "Benchmark MSCI USA Growth"),
    ]
    last_image = None
    for axis, (panel, title) in zip(axes, panels):
        values = panel.to_numpy(dtype=float)
        last_image = axis.imshow(values, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
        axis.set_xticks(range(panel.shape[1]), [f"T{int(column)}" for column in panel.columns])
        axis.set_yticks(range(panel.shape[0]), [str(index) for index in panel.index])
        axis.set_title(title, loc="left", fontsize=11)
        axis.grid(False)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                color = "white" if abs(value) > vmax * 0.58 else "#1e293b"
                axis.text(
                    column,
                    row,
                    f"{value * 100:+.1f} %".replace(".", ","),
                    ha="center",
                    va="center",
                    fontsize=8.4,
                    color=color,
                    fontweight="bold" if abs(value) > vmax * 0.58 else "normal",
                )
    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes, shrink=0.86, pad=0.02, format=FuncFormatter(lambda value, _: f"{value * 100:.0f} %"))
        colorbar.ax.tick_params(labelsize=8)
    fig.suptitle("Rendements trimestriels", x=0.075, ha="left", fontsize=12, fontweight="bold")
    fig.subplots_adjust(top=0.84)
    return fig_to_base64(fig)


def chart_structure(data: dict) -> str:
    portfolio = data["portfolio"].sort_values("date")
    weights_daily = data["weights_daily"]
    squared = weights_daily.pow(2).sum(axis=1).replace(0, np.nan)
    effective_n = (1.0 / squared).replace([np.inf, -np.inf], np.nan)

    fig, axes = plt.subplots(3, 1, figsize=(12.2, 9.0), sharex=True)
    axes[0].plot(portfolio["date"], portfolio["max_weight"], color=STRATEGY_COLOR, marker="o", markersize=4.2, linewidth=1.8)
    axes[0].set_title("Poids maximal d'une ligne", loc="left")
    axes[0].set_ylabel("Poids")
    pct_axis(axes[0], 0)
    axes[0].annotate(
        pct(portfolio["max_weight"].iloc[-1]),
        xy=(portfolio["date"].iloc[-1], portfolio["max_weight"].iloc[-1]),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color=STRATEGY_COLOR,
        fontweight="bold",
    )

    axes[1].plot(portfolio["date"], portfolio["top10_weight"], color=ACCENT_COLOR, marker="o", markersize=4.2, linewidth=1.8)
    axes[1].set_title("Poids cumulé des 10 premières lignes", loc="left")
    axes[1].set_ylabel("Poids")
    pct_axis(axes[1], 0)
    axes[1].annotate(
        pct(portfolio["top10_weight"].iloc[-1]),
        xy=(portfolio["date"].iloc[-1], portfolio["top10_weight"].iloc[-1]),
        xytext=(7, 0),
        textcoords="offset points",
        va="center",
        fontsize=8.5,
        color="#b45309",
        fontweight="bold",
    )

    axes[2].plot(effective_n.index, effective_n.values, color="#475569", linewidth=1.0)
    mean_effective = float(effective_n.mean())
    axes[2].axhline(mean_effective, color=NEG_COLOR, linewidth=1.1, linestyle="--", label=f"Moyenne {mean_effective:.1f}")
    axes[2].set_title("Nombre effectif de lignes (1 / somme des poids²)", loc="left")
    axes[2].set_ylabel("Nombre effectif")
    axes[2].legend(loc="upper left")
    axes[2].yaxis.set_major_locator(MaxNLocator(nbins=6))
    axes[2].set_ylim(0, min(float(effective_n.max()) * 1.08, 32))

    axes[2].xaxis.set_major_locator(mdates.YearLocator())
    axes[2].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for axis in axes:
        style_axis(axis)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_turnover(data: dict) -> str:
    turnover = data["summary"].get("turnover", {})
    dates = pd.to_datetime(list(turnover.keys()))
    values = np.array(list(turnover.values()), dtype=float)
    mean_value = float(values.mean())

    fig, axis = plt.subplots(figsize=(12.2, 4.6))
    colors = [STRATEGY_COLOR if value <= mean_value * 1.3 else ACCENT_COLOR for value in values]
    axis.bar(dates, values, width=72, color=colors, edgecolor="white", linewidth=0.7)
    axis.axhline(mean_value, color=NEG_COLOR, linewidth=1.2, linestyle="--", label=f"Moyenne {mean_value * 100:.1f} %")
    axis.set_ylabel("Turnover du rebalancement")
    axis.set_title("Turnover trimestriel", loc="left")
    axis.legend(loc="upper left")
    pct_axis(axis, 0)
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set_ylim(0, max(values.max() * 1.18, 0.95))
    style_axis(axis)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_sector(data: dict) -> str:
    sector = data["sector"]
    latest_date = sector["date"].max()
    latest = sector[sector["date"] == latest_date].copy()
    latest = latest.sort_values("strategy", ascending=True)

    average_active = sector.groupby("sector")["active"].mean().sort_values(ascending=True)

    fig, (axis_latest, axis_average) = plt.subplots(1, 2, figsize=(14.0, 6.8))
    positions = np.arange(len(latest))
    height = 0.38
    axis_latest.barh(positions - height / 2, latest["strategy"], height, color=STRATEGY_COLOR, label="Stratégie")
    axis_latest.barh(positions + height / 2, latest["benchmark"], height, color=BENCHMARK_COLOR, label="Benchmark")
    axis_latest.set_yticks(positions, latest["sector"])
    axis_latest.set_xlabel("Poids")
    axis_latest.set_title(f"Exposition sectorielle au {latest_date:%d/%m/%Y}", loc="left", fontsize=11)
    axis_latest.legend(loc="lower right")
    axis_latest.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f} %"))
    axis_latest.set_xlim(0, max(float(latest["strategy"].max()), float(latest["benchmark"].max())) * 1.18)
    style_axis(axis_latest)

    average_positions = np.arange(len(average_active))
    colors = [POS_COLOR if value >= 0 else NEG_COLOR for value in average_active.to_numpy()]
    axis_average.barh(average_positions, average_active.to_numpy(), color=colors, alpha=0.85)
    axis_average.axvline(0, color="#475569", linewidth=0.9)
    axis_average.set_yticks(average_positions, average_active.index)
    axis_average.set_xlabel("Exposition active moyenne")
    axis_average.set_title("Biais sectoriel moyen sur la période", loc="left", fontsize=11)
    axis_average.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:+.0f} %"))
    style_axis(axis_average)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_coverage(data: dict) -> str:
    coverage = data["coverage"].sort_values("date")
    fig, axis = plt.subplots(figsize=(11.4, 4.6))
    axis.bar(coverage["date"], coverage["n_constituents"], width=72, color="#99f6e4", edgecolor=STRATEGY_COLOR, linewidth=0.5, label="Constituants")
    axis.plot(coverage["date"], coverage["n_eligible"], color=STRATEGY_COLOR, linewidth=1.6, marker="o", markersize=3.5, label="Éligibles")
    axis.set_ylabel("Nombre de titres")
    axis.set_title("Couverture de l'univers à chaque rebalancement", loc="left")
    axis.set_ylim(0, max(coverage["n_constituents"].max() * 1.18, 300))
    axis.xaxis.set_major_locator(mdates.YearLocator())
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    style_axis(axis)

    axis_cov = axis.twinx()
    axis_cov.plot(coverage["date"], coverage["covered_weight"], color=ACCENT_COLOR, linewidth=2.0, marker="D", markersize=3.5, label="Couverture pondérée")
    axis_cov.set_ylim(0.95, 1.005)
    axis_cov.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.1f} %"))
    axis_cov.spines["top"].set_visible(False)
    axis_cov.tick_params(axis="y", colors="#b45309")
    axis_cov.set_ylabel("Couverture pondérée", color="#b45309")

    handles_one, labels_one = axis.get_legend_handles_labels()
    handles_two, labels_two = axis_cov.get_legend_handles_labels()
    axis.legend(handles_one + handles_two, labels_one + labels_two, loc="lower right", ncol=3, fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_contributions(data: dict) -> str:
    overall = data["summary"].get("overall_contributions", {})
    series = pd.Series(overall, dtype=float).sort_values(ascending=False).head(15).sort_values(ascending=True)
    fig, axis = plt.subplots(figsize=(11.0, 6.0))
    colors = [ACCENT_COLOR if ticker == "NVDA" else STRATEGY_COLOR for ticker in series.index]
    positions = np.arange(len(series))
    axis.barh(positions, series.to_numpy(), color=colors, alpha=0.9)
    axis.set_yticks(positions, series.index)
    axis.set_xlabel("Contribution au rendement cumulé")
    axis.set_title("Top 15 des contributeurs sur la période", loc="left")
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value * 100:.0f} %"))
    for position, value in zip(positions, series.to_numpy()):
        axis.text(value + 0.008 if value >= 0 else value - 0.008, position, f"{value * 100:.1f} %", va="center", ha="left" if value >= 0 else "right", fontsize=8.2, color="#334155")
    axis.set_xlim(min(series.min() * 1.25, -0.02), series.max() * 1.18)
    axis.axvline(0, color="#475569", linewidth=0.9)
    style_axis(axis)
    fig.tight_layout()
    return fig_to_base64(fig)


def build_kpis(data: dict) -> str:
    summary = data["summary"]
    stats = summary["stats"]
    benchmark = summary["bench_stats"]
    benchmark_calendar_cagr = data["benchmark_diagnostics"]["headline"]["official_price"]["cagr_pct"] / 100.0
    cumulative_gap = float(stats["cum_return"]) - float(benchmark["cum_return"])
    cards = [
        ("Rendement cumulé", pct(stats["cum_return"], 1, True), f"Benchmark {pct(benchmark['cum_return'], 1, True)} · écart {pp(cumulative_gap, 1)}", "teal"),
        ("CAGR", pct(stats["ann_return"], 1), f"Benchmark officiel {pct(benchmark_calendar_cagr, 1)} · alpha {pp(stats['alpha_ann'], 1)}", "teal"),
        ("Volatilité annualisée", pct(stats["vol_ann"], 1), f"Benchmark {pct(benchmark['vol_ann'], 1)} · beta {fr(stats['beta'], 2)}", "amber"),
        ("Sharpe", fr(stats["sharpe"], 2), f"Benchmark {fr(benchmark['sharpe'], 2)} · Sortino {fr(stats['sortino'], 2)}", "teal"),
        ("Max drawdown", pct(stats["max_dd"], 1), f"Benchmark {pct(benchmark['max_dd'], 1)} · écart {pp(stats['max_dd'] - benchmark['max_dd'], 1)}", "slate"),
        ("Tracking error / IR", f"{pct(stats['te_ann'], 1)} / {fr(stats['info_ratio'], 2)}", f"Active share {pct(stats['active_share'], 1)} · top10 moyen {pct(stats['top10_weight'], 1)}", "amber"),
    ]
    blocks = []
    for label, value, sub, tone in cards:
        blocks.append(
            f'<div class="kpi {tone}"><div class="label">{esc(label)}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>'
        )
    return "".join(blocks)


def build_annual_table(data: dict) -> str:
    annual = data["annual"]
    rows = []
    for year, row in annual.iterrows():
        strategy = float(row["Strategie"])
        benchmark = float(row["Benchmark"])
        excess = strategy - benchmark
        rows.append(
            [
                f"<td>{esc(year)}</td>",
                signed_pct_cell(strategy),
                signed_pct_cell(benchmark),
                signed_pct_cell(excess),
                f'<td class="num">{pct(strategy, 1)} / {pct(benchmark, 1)}</td>',
            ]
        )
    return table(
        ["Année", "Stratégie", "Benchmark", "Écart", "Stratégie / benchmark"],
        rows,
        num_cols=[1, 2, 3, 4],
    )


def build_risk_table(summary: dict) -> str:
    stats = summary["stats"]
    benchmark = summary["bench_stats"]
    items = [
        ("Rendement cumulé", "cum_return", "pct", True),
        ("CAGR", "ann_return", "pct", True),
        ("Volatilité annualisée", "vol_ann", "pct", True),
        ("Sharpe", "sharpe", "num", True),
        ("Sortino", "sortino", "num", True),
        ("Alpha annualisé", "alpha_ann", "pct", True),
        ("Tracking error", "te_ann", "pct", True),
        ("Information ratio", "info_ratio", "num", True),
        ("Beta", "beta", "num", True),
        ("Max drawdown", "max_dd", "pct", True),
        ("Active share", "active_share", "pct", False),
        ("Poids top 10 moyen", "top10_weight", "pct", False),
        ("Nombre effectif moyen de lignes", "n_effective", "num", False),
        ("Turnover moyen", "turnover_avg", "pct", False),
        ("Capture haussière", "up_capture", "num", False),
        ("Capture baissière", "down_capture", "num", False),
    ]
    rows = []
    for label, key, kind, show_gap in items:
        strategy_value = stats.get(key)
        benchmark_value = benchmark.get(key)
        formatter = pct if kind == "pct" else (lambda value, digits=2, signed=False: fr(value, digits, signed))
        strategy_text = formatter(strategy_value, 1, True) if kind == "pct" else formatter(strategy_value, 2, False)
        benchmark_text = formatter(benchmark_value, 1, True) if kind == "pct" else formatter(benchmark_value, 2, False)
        if kind == "pct":
            strategy_text = formatter(strategy_value, 1, key in {"cum_return", "ann_return"})
            benchmark_text = formatter(benchmark_value, 1, key in {"cum_return", "ann_return"})
        gap_text = "—"
        gap_class = "neutral"
        if show_gap and not is_missing(strategy_value) and not is_missing(benchmark_value) and key not in {"te_ann", "info_ratio"}:
            gap = float(strategy_value) - float(benchmark_value)
            gap_text = pp(gap, 1) if kind == "pct" else fr(gap, 2, True)
            gap_class = sign_class(gap)
        rows.append(
            [
                f"<td>{esc(label)}</td>",
                f'<td class="num">{strategy_text}</td>',
                f'<td class="num">{benchmark_text}</td>',
                f'<td class="num {gap_class}">{gap_text}</td>',
            ]
        )
    return table(["Indicateur", "Stratégie", "Benchmark", "Écart"], rows, num_cols=[1, 2, 3])


def build_coverage_table(data: dict) -> str:
    coverage = data["coverage"].copy()
    portfolio = data["portfolio"][["date", "n_scored", "n_selected"]]
    merged = coverage.merge(portfolio, on="date", how="left").sort_values("date")
    rows = []
    for _, row in merged.iterrows():
        rows.append(
            [
                f"<td>{row['date']:%d/%m/%Y}</td>",
                f'<td class="num">{int(row["n_constituents"])}</td>',
                f'<td class="num">{int(row["n_eligible"])}</td>',
                f'<td class="num">{int(row["n_scored"]) if not is_missing(row["n_scored"]) else "—"}</td>',
                f'<td class="num">{int(row["n_selected"]) if not is_missing(row["n_selected"]) else "—"}</td>',
                f'<td class="num">{pct(row["covered_weight"], 2)}</td>',
            ]
        )
    return table(
        ["Date de rebalancement", "Constituants", "Éligibles", "Notés", "Sélectionnés", "Couverture pondérée"],
        rows,
        num_cols=[1, 2, 3, 4, 5],
    )


def build_sector_table(data: dict) -> str:
    sector = data["sector"]
    latest_date = sector["date"].max()
    latest = sector[sector["date"] == latest_date].sort_values("strategy", ascending=False)
    rows = []
    for _, row in latest.iterrows():
        rows.append(
            [
                f"<td>{esc(row['sector'])}</td>",
                f'<td class="num">{pct(row["strategy"], 1)}</td>',
                f'<td class="num">{pct(row["benchmark"], 1)}</td>',
                signed_pct_cell(row["active"], 1),
            ]
        )
    rows.append(
        [
            '<td class="muted">Total</td>',
            f'<td class="num">{pct(latest["strategy"].sum(), 1)}</td>',
            f'<td class="num">{pct(latest["benchmark"].sum(), 1)}</td>',
            signed_pct_cell(latest["strategy"].sum() - latest["benchmark"].sum(), 1),
        ]
    )
    table_html = table(["Secteur", "Stratégie", "Benchmark", "Actif"], rows, num_cols=[1, 2, 3], row_classes=[None] * (len(rows) - 1) + ["total"])
    return table_html


def build_holdings_table(data: dict) -> str:
    holdings = data["holdings"].copy()
    latest_date = holdings["date"].max()
    holdings = holdings[holdings["date"] == latest_date].sort_values("target_weight", ascending=False).head(20)
    rows = []
    for rank, (_, row) in enumerate(holdings.iterrows(), start=1):
        active = float(row["target_weight"]) - float(row["benchmark_weight"])
        ratio = "—"
        if float(row["benchmark_weight"]) > 0:
            ratio = f'{float(row["target_weight"]) / float(row["benchmark_weight"]):.1f}x'
        rows.append(
            [
                f'<td class="num">{rank}</td>',
                f"<td><b>{esc(row['ticker'])}</b></td>",
                f"<td>{esc(row['sector'])}</td>",
                f'<td class="num">{fr(row["z"], 2)}</td>',
                f'<td class="num">{pct(row["benchmark_weight"], 2)}</td>',
                f'<td class="num">{pct(row["target_weight"], 2)}</td>',
                signed_pct_cell(active, 2),
                f'<td class="num">{ratio}</td>',
            ]
        )
    return table(
        ["Rang", "Ticker", "Secteur", "Z", "Poids benchmark", "Poids cible", "Surpondération", "Ratio"],
        rows,
        num_cols=[0, 3, 4, 5, 6, 7],
    )


def build_contributions_matrix(data: dict) -> str:
    contributions = data["contributions"].copy()
    contributions["year"] = contributions["year"].astype(int)
    years = [year for year in sorted(contributions["year"].unique()) if year >= 2021]
    ranked = {}
    for year in years:
        subset = contributions[contributions["year"] == year].sort_values("contribution", ascending=False).head(10)
        ranked[year] = list(subset[["ticker", "contribution"]].itertuples(index=False, name=None))

    rows = []
    for rank in range(10):
        cells = [f'<td class="num">{rank + 1}</td>']
        for year in years:
            entries = ranked.get(year, [])
            if rank < len(entries):
                ticker, contribution = entries[rank]
                color = sign_class(contribution)
                cells.append(
                    f'<td class="matrix-cell"><b>{esc(ticker)}</b><br>'
                    f'<span class="{color} small">{pct(contribution, 1, True)}</span></td>'
                )
            else:
                cells.append('<td class="matrix-cell muted">—</td>')
        rows.append(cells)
    return table(["Rang"] + [str(year) for year in years], rows, num_cols=[0])


def build_overall_contributions_table(data: dict) -> str:
    overall = data["summary"].get("overall_contributions", {})
    series = pd.Series(overall, dtype=float).sort_values(ascending=False).head(15)
    rows = []
    for rank, (ticker, value) in enumerate(series.items(), start=1):
        rows.append(
            [
                f'<td class="num">{rank}</td>',
                f"<td><b>{esc(ticker)}</b></td>",
                signed_pct_cell(value, 1),
            ]
        )
    return table(["Rang", "Ticker", "Contribution cumulée"], rows, num_cols=[0, 2])


def build_sensitivity_table(data: dict) -> str:
    sensitivity = data["sensitivity"].copy()
    rows = []
    classes = []
    for _, row in sensitivity.iterrows():
        is_base = abs(float(row["lambda"]) - 0.5) < 1e-9
        rows.append(
            [
                f"<td><b>λ = {fr(row['lambda'], 2)}</b>{' <span class=\"chip\">base</span>' if is_base else ''}</td>",
                signed_pct_cell(row["cum_return"], 1),
                f'<td class="num">{pct(row["ann_return"], 1)}</td>',
                f'<td class="num">{pct(row["vol_ann"], 1)}</td>',
                f'<td class="num">{fr(row["sharpe"], 2)}</td>',
                signed_pct_cell(row["max_dd"], 1),
                f'<td class="num">{pct(row["te_ann"], 1)}</td>',
                f'<td class="num">{fr(row["info_ratio"], 2)}</td>',
                f'<td class="num">{pct(row["turnover_avg"], 1)}</td>',
            ]
        )
        classes.append("highlight" if is_base else None)
    return table(
        ["Lambda", "Cumulé", "CAGR", "Volatilité", "Sharpe", "Max DD", "TE", "IR", "Turnover"],
        rows,
        num_cols=[1, 2, 3, 4, 5, 6, 7, 8],
        row_classes=classes,
    )


def build_benchmark_table(data: dict) -> str:
    diagnostics = data["benchmark_diagnostics"]
    headline = diagnostics["headline"]
    entries = [
        ("Indice officiel — prix", "official_price"),
        ("Indice reconstruit — prix", "recon_price"),
        ("Indice reconstruit — total return", "recon_total_return"),
    ]
    rows = []
    for label, key in entries:
        block = headline[key]
        rows.append(
            [
                f"<td>{label}</td>",
                f'<td class="num">{pct(block["cumulative_return_pct"] / 100.0, 2, True)}</td>',
                f'<td class="num">{pct(block["cagr_pct"] / 100.0, 2)}</td>',
            ]
        )
    risk_official = diagnostics["risk_official_price"]
    risk_recon = diagnostics["risk_recon_price"]
    rows.append(
        [
            "<td>Écart officiel − reconstruit prix</td>",
            f'<td class="num">{pp(headline["official_minus_recon_price_gap_pp"] / 100.0, 2)}</td>',
            f'<td class="num muted">—</td>',
        ]
    )
    rows.append(
        [
            "<td>Écart officiel − reconstruit TR</td>",
            f'<td class="num">{pp(headline["official_minus_recon_total_return_gap_pp"] / 100.0, 2)}</td>',
            f'<td class="num muted">—</td>',
        ]
    )
    rows.append(
        [
            "<td>Effet dividendes (recon TR − recon prix)</td>",
            f'<td class="num">{pp(headline["dividend_effect_recon_pp"] / 100.0, 2)}</td>',
            f'<td class="num muted">—</td>',
        ]
    )
    table_html = table(
        ["Série", "Rendement cumulé", "CAGR"],
        rows,
        num_cols=[1, 2],
        row_classes=[None, None, None, "highlight", "highlight", "highlight"],
    )

    risk_rows = [
        [
            "<td>Indice officiel — prix</td>",
            f'<td class="num">{pct(risk_official["annualized_volatility_pct"] / 100.0, 2)}</td>',
            f'<td class="num">{pct(risk_official["max_drawdown_pct"] / 100.0, 2)}</td>',
            f'<td>{esc(risk_official["max_drawdown_peak"])} → {esc(risk_official["max_drawdown_trough"])}</td>',
        ],
        [
            "<td>Indice reconstruit — prix</td>",
            f'<td class="num">{pct(risk_recon["annualized_volatility_pct"] / 100.0, 2)}</td>',
            f'<td class="num">{pct(risk_recon["max_drawdown_pct"] / 100.0, 2)}</td>',
            f'<td>{esc(risk_recon["max_drawdown_peak"])} → {esc(risk_recon["max_drawdown_trough"])}</td>',
        ],
    ]
    risk_table = table(["Série", "Volatilité annualisée", "Max drawdown", "Pic → creux"], risk_rows, num_cols=[1, 2])
    return table_html + '<h4>Risque du benchmark</h4>' + risk_table


def build_data_quality(data: dict) -> str:
    summary = data["summary"]
    universe = summary["universe_summary"]
    diagnostics = data["benchmark_diagnostics"]
    coverage = diagnostics.get("coverage", {})
    anomalies = diagnostics.get("anomalies", {})

    stat_strip = (
        '<div class="stat-strip">'
        f'<div class="stat"><div class="k">Titres dans l\'univers</div><div class="v">{int(summary["config"]["n_tickers"])} / 494</div></div>'
        f'<div class="stat"><div class="k">Constituants par date</div><div class="v">{int(universe["n_constituents_min"])} – {int(universe["n_constituents_max"])}</div></div>'
        f'<div class="stat"><div class="k">Éligibles par date</div><div class="v">{int(universe["n_eligible_min"])} – {int(universe["n_eligible_max"])}</div></div>'
        f'<div class="stat"><div class="k">Couverture pondérée</div><div class="v">{pct(universe["coverage_weight_min"], 2)} – {pct(universe["coverage_weight_max"], 2)}</div></div>'
        f'<div class="stat"><div class="k">Secteurs PIT</div><div class="v">494 / 494</div></div>'
        f'<div class="stat"><div class="k">Secteurs manquants</div><div class="v">{int(universe["missing_sector_count"])}</div></div>'
        "</div>"
    )

    diagnostics_items = [
        f"Alignement des dates officielles et reconstruites : <b>{'oui' if diagnostics['alignment']['dates_equal'] else 'non'}</b> sur {diagnostics['alignment']['n_obs_period']} observations.",
        f"Couverture pondérée du benchmark reconstruit : moyenne {pct(coverage.get('weighted_coverage_mean'), 2)}, minimum {pct(coverage.get('weighted_coverage_min'), 2)} ; {int(coverage.get('missing_tickers_snapshots', 0))} observations titre-date manquantes.",
        f"Anomalies de niveau : {int(anomalies.get('non_positive_levels', {}).get('official', 0))} niveau nul ou négatif, {int(anomalies.get('duplicate_dates_in_inputs', {}).get('index_prices', 0))} date dupliquée, {int(anomalies.get('jumps_gt_20pct_count', 0))} saut quotidien supérieur à 20 %.",
        f"Divergence maximale de niveau entre officiel et reconstruit : {pct(anomalies.get('max_abs_level_divergence_pct', {}).get('value', np.nan) / 100.0, 2)} le {esc(anomalies.get('max_abs_level_divergence_pct', {}).get('date', '—'))}.",
        f"{int(anomalies.get('non_business_day_rows_count', 0))} lignes tombent un jour férié avec un rendement nul ; {int(anomalies.get('zero_return_rows_on_trading_days_count', 0))} jour de bourse présente un rendement nul.",
        "Écart officiel − reconstruit prix : <b>−2,54 pp</b> sur la période ; l'indice officiel est un indice prix, le reconstruit TR inclut les dividendes (+4,93 pp).",
    ]
    diagnostics_list = '<ul class="clean">' + "".join(f"<li>{item}</li>" for item in diagnostics_items) + "</ul>"

    excluded = (
        '<div class="callout"><strong>Périmètre et exclusions.</strong> L\'univers MSCI USA Growth contient 494 titres. '
        "Le backtest retient 450 titres disposant d'un historique de prix exploitable : 39 titres ont été écartés faute de prix "
        "et 5 historiques ont été jugés inutilisables. La couverture pondérée du portefeuille reste comprise entre "
        f"{pct(universe['coverage_weight_min'], 2)} et {pct(universe['coverage_weight_max'], 2)}. "
        "Le mapping sectoriel point-in-time couvre 494 CUSIP sur 494.</div>"
    )

    return (
        stat_strip
        + excluded
        + "<h3>Couverture à chaque rebalancement</h3>"
        + figure_from_base64(
            chart_coverage(data),
            "Couverture de l'univers",
            "Nombre de constituants et de titres éligibles (barres et ligne) et couverture pondérée (axe de droite).",
        )
        + build_coverage_table(data)
        + "<h3>Audit du benchmark officiel et reconstruit</h3>"
        + build_benchmark_table(data)
        + "<h4>Contrôles de qualité</h4>"
        + diagnostics_list
    )


def build_executive_summary(data: dict) -> str:
    summary = data["summary"]
    stats = summary["stats"]
    benchmark = summary["bench_stats"]
    universe = summary["universe_summary"]
    concentration = (
        f"La concentration est extrême : le top 10 pèse en moyenne {pct(stats['top10_weight'], 1)} "
        f"({pct(data['portfolio']['top10_weight'].iloc[-1], 1)} à la dernière date, jusqu'à "
        f"{pct(data['portfolio']['top10_weight'].max(), 1)} en cours de période), le nombre effectif de lignes "
        f"est d'environ {fr(stats['n_effective'], 1)} et NVDA a représenté jusqu'à ~53 % du portefeuille au "
        "rebalancement de 2024 (plus de 60 % en cours de trimestre après dérive). Cette concentration explique "
        "une part importante de la surperformance."
    )
    benchmark_calendar_cagr = data["benchmark_diagnostics"]["headline"]["official_price"]["cagr_pct"] / 100.0
    bullets = [
        f"<b>Surperformance marquée.</b> La stratégie réalise {pct(stats['cum_return'], 1, True)} contre "
        f"{pct(benchmark['cum_return'], 1, True)} pour le benchmark, soit un CAGR de {pct(stats['ann_return'], 1)} "
        f"contre {pct(benchmark_calendar_cagr, 1)} en base calendaire et un alpha annualisé de {pp(stats['alpha_ann'], 1)}.",
        f"<b>Rendement ajusté du risque supérieur.</b> Sharpe de {fr(stats['sharpe'], 2)} contre {fr(benchmark['sharpe'], 2)}, "
        f"information ratio de {fr(stats['info_ratio'], 2)} pour une tracking error de {pct(stats['te_ann'], 1)}. "
        f"Le beta est de {fr(stats['beta'], 2)}, cohérent avec une exposition croissance plus agressive.",
        f"<b>Risque de baisse comparable.</b> La volatilité atteint {pct(stats['vol_ann'], 1)} contre "
        f"{pct(benchmark['vol_ann'], 1)}, mais le max drawdown est quasi identique ({pct(stats['max_dd'], 1)} contre "
        f"{pct(benchmark['max_dd'], 1)}), grâce à des protections relatives lors de la correction 2022.",
        f"<b>Concentration non contrainte.</b> {concentration}",
        f"<b>Sélection robuste.</b> 50 titres sont retenus parmi 160 à 270 éligibles à chaque date, sur 21 rebalancements "
        f"trimestriels ; le turnover moyen s'établit à {pct(stats['turnover_avg'], 1)} et les coûts de transaction de 10 bps sont intégrés.",
        f"<b>Couverture quasi complète.</b> La couverture pondérée de l'univers oscille entre "
        f"{pct(universe['coverage_weight_min'], 2)} et {pct(universe['coverage_weight_max'], 2)}, avec 160 à 272 constituants "
        "et un mapping sectoriel PIT complet (494 / 494).",
        f"<b>Résultats stables au paramètre lambda.</b> Les variantes λ = 0,35 / 0,50 / 0,70 affichent des CAGR de "
        "28,7 % et des tracking errors de 12,9 % à 13,8 %, ce qui suggère que le résultat ne dépend pas finement du dosage "
        "de la surpondération.",
        "<b>Lecture prudente.</b> Une seule trajectoire historique, un univers réduit à 450 titres, un benchmark prix "
        "et non total return, et une performance fortement portée par quelques méga-caps : ces éléments limitent la "
        "portée extrapolable du backtest.",
    ]
    return '<ul class="clean">' + "".join(f"<li>{bullet}</li>" for bullet in bullets) + "</ul>"


def build_methodology(data: dict) -> str:
    summary = data["summary"]
    config = summary["config"]
    cards = [
        (
            "1. Univers MSCI USA Growth",
            f"<p>Univers de départ : {int(config['n_tickers'])} titres issus des snapshots trimestriels MSCI USA Growth, "
            f"sur {int(config['n_dates'])} dates de rebalancement entre le {pd.Timestamp(config['first_date']):%d/%m/%Y} "
            f"et le {pd.Timestamp(config['last_date']):%d/%m/%Y}.</p>"
            "<p>Les poids de benchmark sont point-in-time. Les titres sans historique de prix exploitable sont exclus ; "
            "les 450 titres retenus représentent l'essentiel de la capitalisation de l'indice.</p>",
        ),
        (
            "2. Score d'accélération fondamentale",
            "<p>Les fondamentaux proviennent des dépôts SEC, rattachés à la date de dépôt (<i>filed</i>) pour éviter "
            "toute anticipation. Trois axes sont calculés :</p>"
            "<ul class='clean'>"
            "<li><b>Accélération des ventes</b> : croissance N vs N−1 moins croissance N−1 vs N−2.</li>"
            "<li><b>Expansion de marge</b> : variation de la marge opérationnelle.</li>"
            "<li><b>Efficacité des actifs</b> : variation du chiffre d'affaires / total actifs ; "
            "pour les Financials, variation du ROA.</li>"
            "</ul>"
            "<p>Chaque axe est standardisé en Z-score par secteur, puis moyenné. Le score final Z_Acceleration est "
            "à nouveau standardisé par secteur afin de neutraliser les biais sectoriels.</p>",
        ),
        (
            "3. Sélection et construction des poids",
            "<p>À chaque date, les titres notés sont classés par Z_Acceleration et les <b>50 meilleurs</b> sont retenus. "
            "Le poids brut est proportionnel au poids benchmark multiplié par un facteur exponentiel :</p>"
            "<div class='formula'>w_i ∝ w_bench,i × exp(λ × z_i_clipped)</div>"
            "<p>Les poids sont ensuite normalisés à 100 % sur les 50 lignes sélectionnées. λ = 0,50 et le Z est clippé "
            "à ±3. Aucune contrainte de concentration, de secteur ou de liquidité n'est appliquée.</p>",
        ),
        (
            "4. Rebalancement, prix et coûts",
            "<p>Le portefeuille est reconstruit trimestriellement, à chaque date de snapshot. Les rendements quotidiens "
            "utilisent des prix ajustés Yahoo Finance. Les coûts de transaction sont de <b>10 bps</b> par unité de turnover. "
            "Le benchmark affiché est l'indice officiel en prix ; un indice reconstruit prix et total return est utilisé "
            "pour l'audit de qualité.</p>",
        ),
    ]
    return '<div class="method-grid">' + "".join(
        f'<div class="method-card"><h4>{title}</h4>{body}</div>' for title, body in cards
    ) + "</div>"


def build_limits(data: dict) -> str:
    limits = [
        (
            "Concentration non contrainte",
            "Le portefeuille peut investir une part très élevée dans une seule ligne (NVDA jusqu'à ~53 % en 2024), "
            "ce qui rend la performance et le risque dépendants de quelques méga-caps.",
        ),
        (
            "Résultat porté par NVDA et les méga-caps",
            "La contribution cumulée de NVDA ressort à environ 67 % du rendement de la stratégie. Une trajectoire "
            "différente de cette valeur aurait fortement modifié le résultat global.",
        ),
        (
            "Univers réduit à 450 titres",
            "39 titres sans prix et 5 historiques inutilisables sont exclus de l'univers initial de 494 titres. "
            "Cette exclusion peut introduire un biais de sélection, même si la couverture pondérée reste supérieure à 99,8 %.",
        ),
        (
            "Benchmark prix vs total return",
            "Le benchmark officiel est un indice prix. L'indice reconstruit TR affiche +110,06 % contre +105,13 % en prix, "
            "soit un avantage dividendes d'environ 4,93 pp sur la période. La comparaison à la stratégie ne rémunère donc "
            "pas les dividendes.",
        ),
        (
            "Une seule trajectoire historique",
            "Les résultats proviennent d'un unique chemin de backtest (2020-2025). Ils ne constituent pas une distribution "
            "de performances possibles et ne prouvent pas la robustesse hors échantillon.",
        ),
        (
            "Données SEC parfois manquantes",
            "Certains émetteurs ont des historiques fondamentaux incomplets ou des retraitements. Les scores sont calculés "
            "avec les données disponibles, ce qui peut réduire le nombre de titres notés à certaines dates.",
        ),
        (
            "Frictions et fiscalité non modélisées",
            "Les coûts de 10 bps couvrent le turnover, mais ni le spread, ni l'impact de marché, ni les frais de gestion, "
            "ni la fiscalité. Un portefeuille réel serait également soumis à des contraintes de capacité.",
        ),
        (
            "Benchmark reconstruit imparfait",
            "L'audit signale une couverture pondérée moyenne de 98,56 % et un écart de niveau maximal de 1,52 % entre "
            "l'indice officiel et l'indice reconstruit. Le benchmark officiel reste la référence de performance.",
        ),
    ]
    return '<div class="limits">' + "".join(
        f'<div class="limit"><b>{title}</b>{text}</div>' for title, text in limits
    ) + "</div>"


def build_appendix(data: dict) -> str:
    summary = data["summary"]
    config = summary["config"]
    config_labels = [
        ("start", "Date de début"),
        ("end", "Date de fin"),
        ("n_holdings", "Nombre de lignes"),
        ("lambda", "Lambda"),
        ("z_clip", "Clip du Z-score"),
        ("tx_cost_bps", "Coûts de transaction (bps)"),
        ("min_end", "Historique minimal"),
        ("n_tickers", "Titres dans l'univers"),
        ("n_dates", "Dates de rebalancement"),
        ("first_date", "Première date"),
        ("last_date", "Dernière date"),
    ]
    rows = []
    for key, label in config_labels:
        value = config.get(key)
        if key in {"start", "end", "first_date", "last_date", "min_end"} and value:
            value = pd.Timestamp(value).strftime("%d/%m/%Y")
        elif isinstance(value, float):
            value = fr(value, 2)
        rows.append([f"<td>{esc(label)}</td>", f"<td>{esc(value)}</td>"])
    config_table = table(["Paramètre", "Valeur"], rows)

    files = [
        ("reports/acceleration_report.html", "Rapport HTML autonome (ce fichier)"),
        ("reports/acceleration_summary.json", "Synthèse complète du backtest"),
        ("reports/acceleration_backtest.csv", "NAV et rendements quotidiens"),
        ("reports/acceleration_annual.csv", "Rendements calendaires"),
        ("reports/acceleration_quarterly.csv", "Rendements trimestriels"),
        ("reports/acceleration_sector_exposure.csv", "Expositions sectorielles"),
        ("reports/acceleration_coverage.csv", "Couverture de l'univers"),
        ("reports/acceleration_portfolio_summary.csv", "Structure du portefeuille"),
        ("reports/acceleration_top_holdings.csv", "Dernières participations"),
        ("reports/acceleration_contributions.csv", "Contributions annuelles"),
        ("reports/acceleration_sensitivity.csv", "Sensibilité à lambda"),
        ("reports/benchmark_diagnostics.json", "Audit du benchmark"),
        ("reports/acceleration_weights_daily.parquet", "Poids quotidiens"),
        ("data/processed/acceleration_scores.parquet", "Scores fondamentaux"),
        ("data/processed/acceleration_weights.parquet", "Poids de rebalancement"),
    ]
    file_rows = []
    for relative, role in files:
        path = ROOT / relative
        if path.exists():
            size = f"{path.stat().st_size / 1024:.1f} Ko"
        else:
            size = "absent"
        file_rows.append(
            [
                f"<td><span class='tag'>{esc(relative)}</span></td>",
                f"<td>{esc(role)}</td>",
                f'<td class="num">{size}</td>',
            ]
        )
    files_table = table(["Fichier", "Rôle", "Taille"], file_rows, num_cols=[2])
    generated = datetime.now().strftime("%d/%m/%Y à %H:%M")
    return (
        config_table
        + "<h3>Fichiers générés et sources</h3>"
        + files_table
        + f'<p class="note">Rapport généré le {generated}. La commande de reproduction est '
        "<code>python -m growth_portfolio.src.acceleration_report</code> depuis la racine du projet, "
        "ou <code>python -m src.acceleration_report</code> depuis le dossier <code>growth_portfolio</code>.</p>"
    )


def build_html(data: dict) -> str:
    summary = data["summary"]
    stats = summary["stats"]
    benchmark = summary["bench_stats"]
    config = summary["config"]
    start = pd.Timestamp(config["start"]).strftime("%d/%m/%Y")
    end = pd.Timestamp(config["end"]).strftime("%d/%m/%Y")
    generated = datetime.now().strftime("%d/%m/%Y à %H:%M")
    weights_daily = data["weights_daily"]
    squared_weights = weights_daily.pow(2).sum(axis=1).replace(0, np.nan)
    effective_n = (1.0 / squared_weights).replace([np.inf, -np.inf], np.nan)
    effective_n_last = float(effective_n.iloc[-1])

    nav_figure = figure_from_base64(
        chart_nav_drawdown(data),
        "Valeur liquidative et drawdown",
        "Indice base 100. La stratégie et le benchmark officiel en prix sont comparés sur la période de backtest. "
        "Le panneau inférieur montre le drawdown depuis le plus haut.",
    )
    annual_figure = figure_from_base64(
        chart_annual_returns(data),
        "Rendements annuels",
        "Rendements calendaires de la stratégie et du benchmark. Les barres positives et négatives sont annotées en pourcentage.",
    )
    quarterly_figure = figure_from_base64(
        chart_quarterly_heatmap(data),
        "Heatmap des rendements trimestriels",
        "Rendements trimestriels par année. Vert :performance positive ; rouge : performance négative. "
        "L'échelle est commune aux deux panneaux.",
    )
    structure_figure = figure_from_base64(
        chart_structure(data),
        "Structure et concentration du portefeuille",
        "Poids maximal, poids cumulé du top 10 et nombre effectif de lignes au fil du temps. "
        "Le nombre effectif est calculé quotidiennement comme 1 / somme des poids au carré.",
    )
    turnover_figure = figure_from_base64(
        chart_turnover(data),
        "Turnover trimestriel",
        "Turnover à chaque rebalancement. La ligne pointillée indique la moyenne sur les 21 dates.",
    )
    sector_figure = figure_from_base64(
        chart_sector(data),
        "Expositions sectorielles",
        "À gauche : exposition au dernier rebalancement. À droite : exposition active moyenne par secteur sur toute la période.",
    )
    contributions_figure = figure_from_base64(
        chart_contributions(data),
        "Contributeurs au rendement",
        "Top 15 des contributions cumulées au rendement de la stratégie. NVDA est mise en évidence en orange.",
    )

    header = f"""
    <header class="hero">
      <div class="eyebrow">Rapport de backtest · Actions américaines croissance</div>
      <h1>Accélération fondamentale<br>MSCI USA Growth</h1>
      <p class="lead">Stratégie long-only de sélection de 50 titres sur signaux d'accélération fondamentale,
      rebalancement trimestriel et pondération exponentielle du poids benchmark.</p>
      <div class="period">Période : <b>{start}</b> → <b>{end}</b> · {int(config['n_dates'])} rebalancements trimestriels · benchmark officiel MSCI USA Growth (prix)</div>
      <div class="badges">
        <span class="badge"><b>50</b> titres</span>
        <span class="badge"><b>λ = {fr(config['lambda'], 2)}</b></span>
        <span class="badge"><b>SEC PIT</b> fondamentaux</span>
        <span class="badge"><b>{fr(config['tx_cost_bps'], 0)} bps</b> coûts</span>
        <span class="badge"><b>21</b> rebalancements</span>
        <span class="badge"><b>{int(config['n_tickers'])}</b> titres dans l'univers</span>
      </div>
    </header>
    <nav class="toc" aria-label="Sommaire">
      <a href="#synthese">Synthèse</a>
      <a href="#executif">Résumé exécutif</a>
      <a href="#methodologie">Méthodologie</a>
      <a href="#donnees">Données et qualité</a>
      <a href="#performance">Performance</a>
      <a href="#risque">Risque et structure</a>
      <a href="#secteurs">Secteurs</a>
      <a href="#portefeuille">Portefeuille</a>
      <a href="#sensibilite">Sensibilité</a>
      <a href="#limites">Limites</a>
      <a href="#annexe">Annexe</a>
    </nav>
    """

    synthesis_section = f"""
    <section class="section" id="synthese">
      <div class="section-head">
        <div><div class="kicker">Synthèse</div><h2>Performance, risque et concentration</h2></div>
        <span class="chip">Backtest 2020 – 2025</span>
      </div>
      <div class="kpi-grid">{build_kpis(data)}</div>
      <p class="note" style="margin-top:14px;">Rendements nets de coûts de transaction estimés à 10 bps. Le benchmark est l'indice officiel en prix,
      sans dividendes. Les statistiques annualisées utilisent 252 jours de bourse. Le CAGR du benchmark affiché dans les cartes
      est le CAGR calendaire officiel de l'audit (15,2 %) ; le CAGR annualisé 252 jours du moteur de backtest ressort à 14,6 %.
      L'écart de convention est explicitement signalé dans la section Performance.</p>
    </section>
    """

    executive_section = f"""
    <section class="section" id="executif">
      <div class="section-head"><div><div class="kicker">Lecture rapide</div><h2>Résumé exécutif</h2></div></div>
      {build_executive_summary(data)}
    </section>
    """

    methodology_section = f"""
    <section class="section" id="methodologie">
      <div class="section-head"><div><div class="kicker">Protocole</div><h2>Méthodologie</h2></div></div>
      <p>La stratégie cherche à capter l'accélération des fondamentaux plutôt que le niveau absolu de croissance.
      Elle combine un univers croissance capitalisé, un signal fondamental point-in-time et une pondération qui
      amplifie modérément les meilleurs scores sans rompre complètement avec les poids du benchmark.</p>
      {build_methodology(data)}
      <div class="callout"><strong>Point clé.</strong> La pondération reste proportionnelle au poids benchmark :
      la stratégie ne choisit pas des poids égaux. Un titre déjà très présent dans l'indice et bien noté peut donc
      devenir très dominant après application du facteur exponentiel.</div>
    </section>
    """

    data_section = f"""
    <section class="section" id="donnees">
      <div class="section-head"><div><div class="kicker">Univers et fiabilité</div><h2>Données et qualité</h2></div></div>
      {build_data_quality(data)}
    </section>
    """

    performance_section = f"""
    <section class="section" id="performance">
      <div class="section-head">
        <div><div class="kicker">Performance</div><h2>Trajectoire, drawdown et rendements</h2></div>
        <span class="chip">CAGR {pct(stats['ann_return'], 1)} vs {pct(benchmark['ann_return'], 1)}</span>
      </div>
      {nav_figure}
      <p class="note">Convention d'annualisation : le moteur de backtest annualise les rendements quotidiens sur 252 jours,
      ce qui donne un CAGR de 14,6 % pour le benchmark. Le CAGR calendaire officiel publié par l'audit est de 15,2 %
      (102,59 % de rendement cumulé sur cinq ans). Les deux conventions sont conservées pour la traçabilité ; l'écart
      n'affecte pas les conclusions.</p>
      <div class="grid-2">
        <div>
          <h3>Rendements annuels</h3>
          {annual_figure}
          {build_annual_table(data)}
        </div>
        <div>
          <h3>Rendements trimestriels</h3>
          {quarterly_figure}
          <p class="note">La heatmap couvre les 20 trimestres complets de la période. Les rendements sont calculés à partir des NAV
          quotidiennes et incluent les coûts de transaction.</p>
        </div>
      </div>
    </section>
    """

    risk_section = f"""
    <section class="section" id="risque">
      <div class="section-head">
        <div><div class="kicker">Risque et structure</div><h2>Statistiques complètes et concentration</h2></div>
        <span class="chip">Beta {fr(stats['beta'], 2)} · TE {pct(stats['te_ann'], 1)}</span>
      </div>
      {build_risk_table(summary)}
      <p class="note">Le CAGR du benchmark de ce tableau (14,6 %) est annualisé sur 252 jours par le moteur ; le CAGR calendaire
      officiel de l'audit est de 15,2 %. Les ratios de capture sont calcules en composant les rendements mensuels, puis en comparant les seuls mois de hausse ou de baisse du benchmark ; ils ne sont pas annualises. L'écart est affiché en points de pourcentage
      pour les rendements et en points de ratio pour les autres mesures.</p>
      <h3>Concentration et nombre effectif de lignes</h3>
      {structure_figure}
      <div class="grid-2">
        <div>
          <h3>Turnover</h3>
          {turnover_figure}
          <p class="note">Le premier rebalancement correspond à la construction initiale du portefeuille. La moyenne affichée inclut ce point.</p>
        </div>
        <div>
          <h3>Lecture</h3>
          <div class="stat-strip">
            <div class="stat"><div class="k">Poids max final</div><div class="v">{pct(data['portfolio']['max_weight'].iloc[-1], 1)}</div></div>
            <div class="stat"><div class="k">Top 10 final</div><div class="v">{pct(data['portfolio']['top10_weight'].iloc[-1], 1)}</div></div>
            <div class="stat"><div class="k">N effectif final</div><div class="v">{fr(effective_n_last, 1)}</div></div>
            <div class="stat"><div class="k">N effectif moyen</div><div class="v">{fr(stats['n_effective'], 1)}</div></div>
          </div>
          <p class="note">Le nombre effectif est calculé exactement comme 1 / somme des poids au carré sur les poids quotidiens.
          La concentration est la conséquence directe de l'absence de plafond par ligne. Le poids maximal peut dériver au-delà
          de sa cible entre deux rebalancements.</p>
        </div>
      </div>
    </section>
    """

    sector_section = f"""
    <section class="section" id="secteurs">
      <div class="section-head"><div><div class="kicker">Allocation</div><h2>Exposition sectorielle</h2></div>
        <span class="chip">Dernière date : {data['sector']['date'].max():%d/%m/%Y}</span></div>
      {sector_figure}
      <h3>Dernière exposition par secteur</h3>
      {build_sector_table(data)}
      <p class="note">Les pondérations sont normalisées à 100 % à chaque date. L'exposition active est la différence entre
      la stratégie et le benchmark. Les secteurs absents de la stratégie sont affichés à 0 %.</p>
    </section>
    """

    portfolio_section = f"""
    <section class="section" id="portefeuille">
      <div class="section-head"><div><div class="kicker">Titres</div><h2>Portefeuille et contributions</h2></div>
        <span class="chip">Dernière date : {data['holdings']['date'].max():%d/%m/%Y}</span></div>
      <h3>Top 20 des lignes à la dernière date</h3>
      {build_holdings_table(data)}
      <p class="note">Le poids cible est issu de la normalisation des 50 lignes sélectionnées. La surpondération compare
      le poids cible au poids du titre dans le benchmark. Le ratio indique le multiple de pondération appliqué.</p>
      <h3>Top 10 des contributeurs par année</h3>
      {build_contributions_matrix(data)}
      <p class="note">Les contributions sont exprimées en pourcentage du rendement de la stratégie sur l'année.
      L'année 2020 n'est pas affichée : elle correspond à la construction initiale et ses contributions sont nulles.</p>
      <h3>Contributeurs globaux</h3>
      <div class="grid-2">
        <div>{contributions_figure}</div>
        <div>{build_overall_contributions_table(data)}</div>
      </div>
      <div class="callout"><strong>NVDA concentre le résultat.</strong> Sa contribution cumulée atteint environ
      {pct(summary['overall_contributions'].get('NVDA', np.nan), 1)} du rendement de la stratégie. Cette dépendance
      est la principale réserve à formuler sur la lisibilité du backtest.</div>
    </section>
    """

    sensitivity_section = f"""
    <section class="section" id="sensibilite">
      <div class="section-head"><div><div class="kicker">Robustesse</div><h2>Sensibilité au paramètre lambda</h2></div></div>
      <p>Le paramètre λ contrôle l'intensité de la surpondération des meilleurs scores. Trois valeurs sont testées
      avec les mêmes données, le même univers et les mêmes coûts. La stabilité des résultats est un signe de robustesse
      locale du signal, mais ne constitue pas une validation hors échantillon.</p>
      {build_sensitivity_table(data)}
      <p class="note">La ligne λ = 0,50 correspond à la configuration de référence du rapport. Les résultats varient
      très peu : le CAGR reste à environ 28,7 % et la tracking error entre 12,9 % et 13,8 %.</p>
    </section>
    """

    limits_section = f"""
    <section class="section" id="limites">
      <div class="section-head"><div><div class="kicker">Prudence</div><h2>Limites et avertissements</h2></div></div>
      {build_limits(data)}
      <div class="callout" style="margin-top:18px;"><strong>Avertissement.</strong> Ce document est un exercice de recherche
      et ne constitue pas un conseil en investissement. Les performances passées ne préjugent pas des performances futures.
      Les résultats du backtest sont sensibles aux choix de données, de paramètres et de période.</div>
    </section>
    """

    appendix_section = f"""
    <section class="section" id="annexe">
      <div class="section-head"><div><div class="kicker">Traçabilité</div><h2>Annexe</h2></div></div>
      <h3>Configuration du backtest</h3>
      {build_appendix(data)}
    </section>
    """

    footer = f"""
    <footer class="footer">
      Rapport généré automatiquement le {generated} · Sources : MSCI USA Growth, SEC EDGAR, Yahoo Finance ·
      Les figures sont embarquées en base64, le document ne dépend d'aucune ressource externe.
    </footer>
    """

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Rapport de backtest de la stratégie d'accélération fondamentale MSCI USA Growth">
<title>Accélération fondamentale — MSCI USA Growth</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{header}
{synthesis_section}
{executive_section}
{methodology_section}
{data_section}
{performance_section}
{risk_section}
{sector_section}
{portfolio_section}
{sensitivity_section}
{limits_section}
{appendix_section}
{footer}
</div>
</body>
</html>
"""


def main() -> None:
    for required in [
        REPORTS / "acceleration_summary.json",
        REPORTS / "acceleration_backtest.csv",
        REPORTS / "acceleration_annual.csv",
        REPORTS / "acceleration_quarterly.csv",
        REPORTS / "acceleration_sector_exposure.csv",
        REPORTS / "acceleration_coverage.csv",
        REPORTS / "acceleration_portfolio_summary.csv",
        REPORTS / "acceleration_top_holdings.csv",
        REPORTS / "acceleration_contributions.csv",
        REPORTS / "acceleration_sensitivity.csv",
        REPORTS / "benchmark_diagnostics.json",
        REPORTS / "acceleration_weights_daily.parquet",
    ]:
        if not required.exists():
            raise FileNotFoundError(f"Fichier source manquant : {required}")

    data = load_data()
    document = build_html(data)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(document, encoding="utf-8")

    size_bytes = OUTPUT.stat().st_size
    text = OUTPUT.read_text(encoding="utf-8")
    required_sections = [
        "Résumé exécutif",
        "Méthodologie",
        "Données et qualité",
        "Performance",
        "Risque et structure",
        "Exposition sectorielle",
        "Portefeuille et contributions",
        "Sensibilité au paramètre lambda",
        "Limites et avertissements",
        "Annexe",
    ]
    missing_sections = [section for section in required_sections if section not in text]
    image_count = text.count("data:image/png;base64,")

    if size_bytes <= 50_000:
        raise RuntimeError(f"Rapport trop petit : {size_bytes} octets (minimum 50 000).")
    if missing_sections:
        raise RuntimeError(f"Sections manquantes : {missing_sections}")
    if image_count == 0:
        raise RuntimeError("Aucune image base64 trouvée dans le rapport.")

    print(f"Rapport généré : {OUTPUT}")
    print(f"Taille finale : {size_bytes / 1024:.1f} Ko ({size_bytes} octets)")
    print(f"Images base64 : {image_count}")
    print(f"Sections vérifiées : {len(required_sections)} / {len(required_sections)}")


if __name__ == "__main__":
    main()
