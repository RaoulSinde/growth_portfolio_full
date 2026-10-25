"""Rapport HTML des variantes 60 lignes anchors + score picks."""
from __future__ import annotations

import base64
import html
import io
import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

try:
    from .acceleration_report import CSS
except ImportError:
    from acceleration_report import CSS

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"
OUTPUT = REPORTS / "anchor_report.html"

TEAL = "#0f766e"
AMBER = "#d97706"
GRAY = "#64748b"

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

VARIANT_STYLES = {
    "top10_plus_50": (TEAL, "-", None),
    "top20_plus_40": (AMBER, "-", None),
    "top10_plus_50_cap15": (TEAL, "--", "//"),
    "top20_plus_40_cap15": (AMBER, "--", "//"),
    "top10_plus_50_ex_nvda": (TEAL, ":", "\\\\"),
    "top20_plus_40_ex_nvda": (AMBER, ":", "\\\\"),
    "top10_plus_50_cap15_ex_nvda": (TEAL, "-.", "\\\\"),
    "top20_plus_40_cap15_ex_nvda": (AMBER, "-.", "\\\\"),
}


def load_data() -> dict:
    with open(REPORTS / "anchor_summary.json", "r", encoding="utf-8") as handle:
        summary = json.load(handle)
    comparison = pd.read_csv(REPORTS / "anchor_variant_comparison.csv")
    daily = pd.read_csv(REPORTS / "anchor_backtest.csv", index_col=0)
    daily.index = pd.to_datetime(daily.index)
    annual = pd.read_csv(REPORTS / "anchor_annual.csv", index_col=0)
    quarterly = pd.read_csv(REPORTS / "anchor_quarterly.csv", index_col=0)
    sectors = pd.read_csv(REPORTS / "anchor_sector_exposure.csv")
    sectors["date"] = pd.to_datetime(sectors["date"])
    coverage = pd.read_csv(REPORTS / "anchor_coverage.csv")
    coverage["date"] = pd.to_datetime(coverage["date"])
    portfolio = pd.read_csv(REPORTS / "anchor_portfolio_summary.csv")
    portfolio["date"] = pd.to_datetime(portfolio["date"])
    holdings = pd.read_csv(REPORTS / "anchor_holdings.csv")
    holdings["date"] = pd.to_datetime(holdings["date"])
    contributions = pd.read_csv(REPORTS / "anchor_contributions.csv")
    weights = {
        name: pd.read_parquet(REPORTS / f"anchor_{name}_weights_daily.parquet")
        for name in summary["variants"]
    }
    return {
        "summary": summary,
        "comparison": comparison,
        "daily": daily,
        "annual": annual,
        "quarterly": quarterly,
        "sectors": sectors,
        "coverage": coverage,
        "portfolio": portfolio,
        "holdings": holdings,
        "contributions": contributions,
        "weights": weights,
    }


def fig_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def pct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value * 100:.{digits}f} %"


def spct(value, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value * 100:+.{digits}f} pp"


def fr(value, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    return f"{value:.{digits}f}"


def esc(value) -> str:
    return html.escape(str(value))


def variant_names(data: dict) -> list[str]:
    return list(data["summary"]["variants"].keys())


def core_variant_names(data: dict) -> list[str]:
    return [name for name in variant_names(data) if not name.endswith("ex_nvda")]


def robust_variant_names(data: dict) -> list[str]:
    return [name for name in variant_names(data) if name.endswith("ex_nvda")]


def variant_label(data: dict, name: str) -> str:
    return data["summary"]["variants"][name]["label"]


def style_for(name: str):
    return VARIANT_STYLES.get(name, (TEAL, "-", None))


def chart_nav_drawdown(data: dict, names: list[str] | None = None) -> str:
    daily = data["daily"]
    names = names or core_variant_names(data)
    fig, (ax_nav, ax_dd) = plt.subplots(
        2,
        1,
        figsize=(12.4, 8.0),
        sharex=True,
        gridspec_kw={"height_ratios": [2.25, 1.0], "hspace": 0.08},
    )
    for name in names:
        color, linestyle, _ = style_for(name)
        nav = daily[f"nav_{name}"]
        ax_nav.plot(
            daily.index,
            nav,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=variant_label(data, name),
        )
        dd = nav / nav.cummax() - 1.0
        ax_dd.plot(
            daily.index,
            dd,
            color=color,
            linestyle=linestyle,
            linewidth=1.3,
            label=variant_label(data, name),
        )
    bench = daily["nav_benchmark"]
    ax_nav.plot(
        daily.index,
        bench,
        color=GRAY,
        linewidth=1.5,
        linestyle=":",
        label="MSCI USA Growth (prix)",
    )
    bench_dd = bench / bench.cummax() - 1.0
    ax_dd.plot(daily.index, bench_dd, color=GRAY, linewidth=1.1, linestyle=":", label="Benchmark")
    ax_nav.set_ylabel("Indice base 100")
    ax_nav.set_title("Valeur liquidative des quatre variantes", loc="left")
    ax_nav.legend(loc="upper left", ncol=2, fontsize=8)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f} %"))
    ax_dd.xaxis.set_major_locator(mdates.YearLocator())
    ax_dd.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_annual(data: dict, names: list[str] | None = None) -> str:
    annual = data["annual"]
    labels = list(annual.index)
    names = names or core_variant_names(data)
    x = np.arange(len(labels))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(12.4, 4.8))
    for i, name in enumerate(names):
        color, _, hatch = style_for(name)
        offset = (i - (len(names) - 1) / 2) * width
        ax.bar(
            x + offset,
            annual[variant_label(data, name)],
            width,
            color=color,
            alpha=0.9,
            hatch=hatch,
            edgecolor="white" if not hatch else "#475569",
            label=variant_label(data, name),
        )
    ax.plot(x, annual["Benchmark"], color=GRAY, marker="o", linewidth=1.3, label="Benchmark")
    ax.axhline(0.0, color="#94a3b8", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f} %"))
    ax.set_title("Rendements calendaires", loc="left")
    ax.legend(loc="upper left", ncol=3, fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_concentration(data: dict, names: list[str] | None = None) -> str:
    comp = data["comparison"].set_index("variant")
    names = names or core_variant_names(data)
    metrics = [
        ("max_weight_last", "Poids max"),
        ("top10_weight_last", "Top 10"),
        ("nvda_max_weight", "NVDA max"),
    ]
    x = np.arange(len(metrics))
    width = 0.8 / len(names)
    fig, ax = plt.subplots(figsize=(10.4, 4.6))
    for i, name in enumerate(names):
        color, _, hatch = style_for(name)
        vals = [comp.loc[name, key] for key, _ in metrics]
        offset = (i - (len(names) - 1) / 2) * width
        ax.bar(
            x + offset,
            vals,
            width,
            color=color,
            alpha=0.9,
            hatch=hatch,
            edgecolor="white" if not hatch else "#475569",
            label=variant_label(data, name),
        )
    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in metrics])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f} %"))
    ax.set_title("Concentration à la dernière date", loc="left")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_sector(data: dict, names: list[str] | None = None) -> str:
    sectors = data["sectors"]
    last = sectors["date"].max()
    frame = sectors[sectors["date"] == last]
    names = names or core_variant_names(data)
    pivot = frame.pivot_table(index="sector", columns="variant", values="strategy", aggfunc="sum").fillna(0.0)
    bench = frame.groupby("sector")["benchmark"].mean()
    order = (pivot.max(axis=1) + bench).sort_values(ascending=True).index
    pivot = pivot.reindex(order)
    bench = bench.reindex(order)
    fig, ax = plt.subplots(figsize=(10.8, 7.0))
    y = np.arange(len(order))
    height = 0.8 / (len(names) + 1)
    ax.barh(y - 0.4 + height / 2, bench.values, height, color=GRAY, alpha=0.8, label="Benchmark")
    for i, name in enumerate(names):
        color, _, hatch = style_for(name)
        ax.barh(
            y - 0.4 + height * (i + 1.5),
            pivot[name].values,
            height,
            color=color,
            alpha=0.9,
            hatch=hatch,
            edgecolor="white" if not hatch else "#475569",
            label=variant_label(data, name),
        )
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f} %"))
    ax.set_title(f"Exposition sectorielle au {last:%d/%m/%Y}", loc="left")
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_contributions(data: dict, names: list[str] | None = None) -> str:
    names = names or core_variant_names(data)
    rows = 2 if len(names) <= 4 else int(np.ceil(len(names) / 2))
    fig, axes = plt.subplots(rows, 2, figsize=(12.6, 4.5 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, name in zip(axes.flat, names):
        color, _, hatch = style_for(name)
        series = pd.Series(data["summary"]["variants"][name]["overall_contributions"]).astype(float)
        series = series.head(10).sort_values()
        ax.barh(series.index, series.values, color=color, alpha=0.9, hatch=hatch, edgecolor="white" if not hatch else "#475569")
        ax.axvline(0.0, color="#94a3b8", linewidth=0.9)
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v*100:.0f} %"))
        ax.set_title(variant_label(data, name), loc="left", fontsize=10)
    fig.tight_layout()
    return fig_to_base64(fig)


def chart_robustness(data: dict) -> str:
    daily = data["daily"]
    names = robust_variant_names(data)
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    for name in names:
        color, linestyle, _ = style_for(name)
        ax.plot(
            daily.index,
            daily[f"nav_{name}"],
            color=color,
            linestyle=linestyle,
            linewidth=1.8,
            label=variant_label(data, name),
        )
    ax.plot(
        daily.index,
        daily["nav_benchmark_ex_nvda"],
        color=GRAY,
        linestyle=":",
        linewidth=1.5,
        label="MSCI USA Growth ex-NVDA (reconstruit)",
    )
    ax.set_ylabel("Indice base 100")
    ax.set_title("Variantes sans NVDA face à l'indice sans NVDA", loc="left")
    ax.legend(loc="upper left", ncol=2, fontsize=8)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    fig.tight_layout()
    return fig_to_base64(fig)


def table(headers: list[str], rows: list[list[str]], num_cols: set[int] | None = None) -> str:
    num_cols = num_cols or set()
    head = "".join(
        f'<th class="{"num" if i in num_cols else ""}">{esc(h)}</th>'
        for i, h in enumerate(headers)
    )
    body = []
    for row in rows:
        cells = "".join(
            f'<td class="{"num" if i in num_cols else ""}">{value}</td>'
            for i, value in enumerate(row)
        )
        body.append(f"<tr>{cells}</tr>")
    return (
        '<div class="table-wrap"><table class="data">'
        f"<thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


def kpi(label: str, value: str, detail: str = "") -> str:
    return (
        f'<div class="kpi"><div class="k">{esc(label)}</div>'
        f'<div class="v">{esc(value)}</div><div class="d">{esc(detail)}</div></div>'
    )


def build_comparison_table(data: dict, names: list[str] | None = None) -> str:
    comp = data["comparison"].set_index("variant")
    names = names or core_variant_names(data)
    rows = [
        ("CAGR", "ann_return", pct),
        ("Volatilité annualisée", "vol_ann", pct),
        ("Sharpe", "sharpe", fr),
        ("Drawdown max", "max_dd", pct),
        ("Alpha annualisé", "alpha_ann", pct),
        ("Tracking error", "te_ann", pct),
        ("Information ratio", "info_ratio", fr),
        ("Bêta", "beta", fr),
        ("Turnover moyen", "turnover_avg", pct),
        ("Poids max final", "max_weight_last", pct),
        ("Top 10 final", "top10_weight_last", pct),
        ("NVDA poids max", "nvda_max_weight", pct),
        ("N effectif moyen", "n_effective", fr),
        ("Lignes en moyenne", "n_holdings_avg", fr),
        ("Cap 15 % touché / date", "cap_bindings_avg", fr),
        ("Lignes forcées au cap / date", "cap_forced_avg", fr),
        ("Bornes hautes touchées / date", "band_high_bindings_avg", fr),
        ("Active weight max moyen", "max_active_weight_avg", pct),
    ]
    headers = ["Mesure"] + [variant_label(data, name) for name in names]
    body = []
    for label, key, fmt in rows:
        body.append([esc(label)] + [fmt(comp.loc[name, key]) for name in names])
    return table(headers, body)


def build_robustness_table(data: dict) -> str:
    comp = data["comparison"].set_index("variant")
    pairs = {
        "top10_plus_50_ex_nvda": "top10_plus_50",
        "top10_plus_50_cap15_ex_nvda": "top10_plus_50_cap15",
        "top20_plus_40_ex_nvda": "top20_plus_40",
        "top20_plus_40_cap15_ex_nvda": "top20_plus_40_cap15",
    }
    headers = [
        "Variante sans NVDA",
        "Référence avec NVDA",
        "CAGR",
        "Écart CAGR vs réf.",
        "Alpha vs indice ex-NVDA",
        "TE vs indice ex-NVDA",
        "IR vs indice ex-NVDA",
        "Max DD",
        "Poids max final",
        "NVDA max",
    ]
    rows = []
    for name, reference in pairs.items():
        rows.append(
            [
                esc(variant_label(data, name)),
                esc(variant_label(data, reference)),
                pct(comp.loc[name, "ann_return"]),
                spct(comp.loc[name, "ann_return"] - comp.loc[reference, "ann_return"]),
                pct(comp.loc[name, "alpha_ann"]),
                pct(comp.loc[name, "te_ann"]),
                fr(comp.loc[name, "info_ratio"]),
                pct(comp.loc[name, "max_dd"]),
                pct(comp.loc[name, "max_weight_last"]),
                pct(comp.loc[name, "nvda_max_weight"]),
            ]
        )
    bench = data["summary"]["bench_ex_nvda_stats"]
    rows.append(
        [
            "Indice MSCI USA Growth ex-NVDA (reconstruit)",
            "—",
            pct(bench["ann_return"]),
            "—",
            "—",
            "—",
            "—",
            pct(bench["max_dd"]),
            "—",
            "0.0 %",
        ]
    )
    return table(headers, rows, num_cols=set(range(2, len(headers))))


def build_annual_table(data: dict, names: list[str] | None = None) -> str:
    annual = data["annual"]
    names = names or core_variant_names(data)
    columns = [variant_label(data, name) for name in names] + ["Benchmark"]
    rows = []
    for year, row in annual.iterrows():
        rows.append([esc(year)] + [pct(row[column]) for column in columns])
    return table(["Année"] + columns, rows, num_cols=set(range(1, len(columns) + 1)))


def build_sector_table(data: dict, names: list[str] | None = None) -> str:
    sectors = data["sectors"]
    last = sectors["date"].max()
    frame = sectors[sectors["date"] == last]
    names = names or core_variant_names(data)
    pivot = frame.pivot_table(index="sector", columns="variant", values="strategy", aggfunc="sum").fillna(0.0)
    active = frame.pivot_table(index="sector", columns="variant", values="active", aggfunc="sum").fillna(0.0)
    bench = frame.groupby("sector")["benchmark"].mean()
    order = (pivot.max(axis=1) + bench).sort_values(ascending=False).index
    headers = ["Secteur", "Benchmark"]
    for name in names:
        headers.extend([variant_label(data, name), "Actif"])
    rows = []
    for sector in order:
        row = [esc(sector), pct(bench.get(sector, 0.0))]
        for name in names:
            row.append(pct(pivot.loc[sector, name] if sector in pivot.index else 0.0))
            row.append(spct(active.loc[sector, name] if sector in active.index else 0.0))
        rows.append(row)
    return table(headers, rows, num_cols=set(range(1, len(headers))))


def build_holdings_table(data: dict, name: str, n: int = 20) -> str:
    holdings = data["holdings"]
    last = holdings["date"].max()
    frame = holdings[(holdings["variant"] == name) & (holdings["date"] == last)].sort_values("target_weight", ascending=False).head(n)
    headers = [
        "#",
        "Titre",
        "Secteur",
        "Score z",
        "Anchor",
        "Poids indice",
        "Poids rebasé",
        "Poids cible",
        "Actif",
        "Bande",
        "Cap",
    ]
    rows = []
    for rank, (_, row) in enumerate(frame.iterrows(), start=1):
        anchor = "Oui" if bool(row["is_anchor"]) else "Non"
        band = f"{pct(row['lower_weight'])} – {pct(row['upper_weight'])}"
        cap = pct(row["max_weight_cap"]) if pd.notna(row["max_weight_cap"]) else "—"
        rows.append(
            [
                str(rank),
                f"<strong>{esc(row['ticker'])}</strong>",
                esc(row["sector"]),
                fr(row["z"], 2),
                anchor,
                pct(row["benchmark_weight"]),
                pct(row["rebased_weight"]),
                pct(row["target_weight"]),
                spct(row["active_weight"]),
                band,
                cap,
            ]
        )
    return table(headers, rows, num_cols={0, 3, 5, 6, 7, 8})


def build_contribution_table(data: dict, name: str, n: int = 10) -> str:
    series = pd.Series(data["summary"]["variants"][name]["overall_contributions"]).astype(float).head(n)
    rows = [
        [str(rank), f"<strong>{esc(ticker)}</strong>", spct(value)]
        for rank, (ticker, value) in enumerate(series.items(), start=1)
    ]
    return table(["#", "Titre", "Contribution cumulée"], rows, num_cols={0, 2})


def build_config_table(data: dict) -> str:
    config = data["summary"]["config"]
    cap_value = pct(config.get("cap_max_weight", np.nan))
    rows = [
        ("Période", f"{pd.Timestamp(config['start']):%d/%m/%Y} → {pd.Timestamp(config['end']):%d/%m/%Y}"),
        ("Titres par variante", config["n_total_holdings"]),
        ("Anchors", ", ".join(f"{k} = {v}" for k, v in config["anchors"].items())),
        ("Sous-variantes capees", ", ".join(f"{k} = {v}" for k, v in config.get("capped_variants", {}).items())),
        ("Variantes sans NVDA", ", ".join(config.get("ex_nvda_variants", {}).keys())),
        ("Titres exclus", ", ".join(config.get("excluded_tickers", []))),
        ("Indice ex-NVDA", config.get("ex_nvda_benchmark", "—")),
        ("Cap absolu des sous-variantes", cap_value),
        ("λ", fr(config["lambda"], 2)),
        ("Clip des z", fr(config["z_clip"], 1)),
        ("Bande relative", pct(config["band_rel"], 0)),
        ("Bande absolue", pct(config["band_abs_cap"], 1)),
        ("Coûts de transaction", f"{fr(config['tx_cost_bps'], 1)} bps"),
        ("Univers", f"{config['n_tickers']} tickers, {config['n_dates']} dates"),
        ("Rebalancement", "Trimestriel"),
        ("Contrainte sectorielle", "Aucune"),
        ("Cap absolu sur les variantes de base", "Aucun"),
    ]
    return table(["Paramètre", "Valeur"], rows)


def build_html(data: dict) -> str:
    summary = data["summary"]
    comp = data["comparison"].set_index("variant")
    names = core_variant_names(data)
    robust_names = robust_variant_names(data)
    bench = summary["bench_stats"]
    universe = summary["universe_summary"]
    config = summary["config"]
    generated = datetime.now().strftime("%d/%m/%Y à %H:%M")

    base_top10 = "top10_plus_50"
    base_top20 = "top20_plus_40"
    cap_top10 = "top10_plus_50_cap15"
    cap_top20 = "top20_plus_40_cap15"

    nav_figure = chart_nav_drawdown(data)
    annual_figure = chart_annual(data)
    concentration_figure = chart_concentration(data)
    sector_figure = chart_sector(data)
    contributions_figure = chart_contributions(data)
    robustness_figure = chart_robustness(data)

    kpis = "".join(
        kpi(
            variant_label(data, name),
            pct(comp.loc[name, "ann_return"]),
            f"TE {pct(comp.loc[name, 'te_ann'])} · IR {fr(comp.loc[name, 'info_ratio'])} · cap {pct(comp.loc[name, 'max_weight_last'])}",
        )
        for name in names
    )
    kpis += kpi("Benchmark", pct(bench["ann_return"]), f"Vol {pct(bench['vol_ann'])} · DD {pct(bench['max_dd'])}")
    kpis += kpi("Univers", f"{universe['n_tickers']} titres", f"{universe['n_dates']} dates de rebalancement")

    hero = f"""
    <header class="hero">
      <div class="eyebrow">MSCI USA Growth · Anchors + accélération fondamentale</div>
      <h1>Huit variantes de 60 lignes, avec/sans cap et avec/sans NVDA</h1>
      <p class="lead">Deux constructions ancrées sur les plus gros poids de l'indice, chacune déclinée en version
      sans cap absolu, en sous-variante plafonnée à 15 % par ligne, et en version excluant NVDA pour tester la
      robustesse. Les poids sont rebasés, tiltés par le score d'accélération, puis projetés dans des bandes de
      50 % du poids rebasé, plafonnées à 4 points de pourcentage.</p>
      <div class="period">Période : {pd.Timestamp(config['start']):%d/%m/%Y} → {pd.Timestamp(config['end']):%d/%m/%Y} · Rebalancement trimestriel · Coûts 10 bps</div>
      <div class="badges">
        <span class="badge"><b>{config['n_total_holdings']}</b> lignes</span>
        <span class="badge"><b>10</b> anchors + 50 scores</span>
        <span class="badge"><b>20</b> anchors + 40 scores</span>
        <span class="badge">Cap <b>15 %</b> sur quatre sous-variantes</span>
        <span class="badge">Exclusion <b>NVDA</b> sur quatre variantes</span>
        <span class="badge">Bande <b>50 % / 4 pp</b></span>
        <span class="badge">λ <b>{fr(config['lambda'], 2)}</b></span>
      </div>
    </header>
    """

    toc = """
    <nav class="toc">
      <a href="#resume">Résumé</a>
      <a href="#methode">Méthodologie</a>
      <a href="#donnees">Données</a>
      <a href="#performance">Performance</a>
      <a href="#risque">Risque</a>
      <a href="#secteurs">Secteurs</a>
      <a href="#portefeuille">Portefeuille</a>
      <a href="#contributions">Contributions</a>
      <a href="#robustesse">Robustesse</a>
      <a href="#limites">Limites</a>
      <a href="#annexe">Annexe</a>
    </nav>
    """

    synthesis = f"""
    <section class="section" id="resume">
      <div class="section-head"><div><div class="kicker">Résumé exécutif</div><h2>Le cap réduit la concentration, l'exclusion de NVDA révèle la dépendance au titre</h2></div>
        <span class="chip">4 variantes principales · 4 variantes sans NVDA</span></div>
      <div class="kpi-grid">{kpis}</div>
      <p class="note" style="margin-top:16px;">Les variantes de base conservent les plus gros poids de l'indice et
      affichent une concentration élevée. Les sous-variantes cap 15 % plafonnent chaque ligne à 15 %, ce qui réduit
      mécaniquement le poids maximal et le top 10. Les versions sans NVDA, comparées à un indice ex-NVDA reconstruit,
      conservent un alpha positif mais plus faible : le CAGR de la variante 10 anchors passe de
      {pct(comp.loc[base_top10, 'ann_return'])} à {pct(comp.loc[robust_names[0], 'ann_return'])} et son alpha
      annualisé de {pct(comp.loc[base_top10, 'alpha_ann'])} à {pct(comp.loc[robust_names[0], 'alpha_ann'])}.</p>
      {build_comparison_table(data)}
    </section>
    """

    methodology = f"""
    <section class="section" id="methode">
      <div class="section-head"><div><div class="kicker">Construction</div><h2>Anchors, score picks, rebase, bandes et cap</h2></div></div>
      <div class="grid-2">
        <div>
          <h3>1. Sélection</h3>
          <p>À chaque date de rebalancement, les <strong>10</strong> ou <strong>20</strong> plus gros poids du benchmark
          sont retenus comme anchors, quel que soit leur score. Les <strong>50</strong> ou <strong>40</strong> meilleurs
          scores d'accélération hors anchors complètent la sélection, sans doublon.</p>
          <h3>2. Rebase et tilt</h3>
          <p>Les poids d'indice des 60 titres sont rebasés à 100 % : <code>w0 = w_bench / Σw_bench</code>. Le tilt applique
          <code>w0 × exp(λ × clip(z, -3, 3))</code>, puis normalise. Les anchors sans score utilisent <code>z = 0</code>.</p>
          <h3>3. Bandes</h3>
          <p>La bande est calculée après rebase : <code>b = min(50 % × w0, 4 pp)</code>. Les bornes sont
          <code>L = max(0, w0 - b)</code> et <code>U = w0 + b</code>. La projection finale impose <code>Σw = 1</code> et
          <code>L ≤ w ≤ U</code>.</p>
          <h3>4. Cap à 15 %</h3>
          <p>Dans les sous-variantes capees, la borne haute devient <code>U = min(w0 + b, 15 %)</code>. Si le poids rebasé
          dépasse 15 %, la ligne est forcée à 15 % : le cap prime alors sur la bande. La borne basse est ajustée à
          <code>L = min(max(0, w0 - b), U)</code> pour conserver un problème faisable.</p>
          <h3>5. Robustesse hors NVDA</h3>
          <p>Une version de chaque construction est recalculée en excluant NVDA de l'univers de sélection. Le titre est
          retiré avant le choix des anchors et des score picks ; le prochain poids éligible devient anchor. Cette
          variante mesure la part de performance attribuable à NVDA.</p>
        </div>
        <div>
          <h3>Paramètres</h3>
          {build_config_table(data)}
          <p class="note">Le portefeuille est long-only, sans contrainte sectorielle. Le rebalancement est trimestriel et
          les coûts de transaction sont de 10 bps par rotation. Aucune calibration de tracking error n'est réalisée.</p>
        </div>
      </div>
      <div class="callout"><strong>Point de vigilance.</strong> Le cap de 15 % plafonne le poids final, mais il ne
      supprime pas la concentration sectorielle ni la dépendance aux mégacapitalisations. Il rend simplement le
      portefeuille plus diversifié en poids individuels.</div>
    </section>
    """

    data_section = f"""
    <section class="section" id="donnees">
      <div class="section-head"><div><div class="kicker">Données et qualité</div><h2>Univers PIT et couverture</h2></div>
        <span class="chip">{universe['n_tickers']} tickers · {universe['n_dates']} dates</span></div>
      <div class="grid-2">
        <div>
          <h3>Univers</h3>
          <p>L'univers est l'union des constituants MSCI USA Growth sur la période. À chaque date, seuls les constituants
          réels et éligibles sont utilisés. La couverture pondérée minimale est de {pct(universe['coverage_weight_min'])}.</p>
          <div class="stat-strip">
            <div class="stat"><div class="k">Titres</div><div class="v">{universe['n_tickers']}</div></div>
            <div class="stat"><div class="k">Dates</div><div class="v">{universe['n_dates']}</div></div>
            <div class="stat"><div class="k">Éligibles min.</div><div class="v">{universe['n_eligible_min']}</div></div>
            <div class="stat"><div class="k">Couverture min.</div><div class="v">{pct(universe['coverage_weight_min'])}</div></div>
          </div>
        </div>
        <div>
          <h3>Qualité</h3>
          <p>Les fondamentaux proviennent de SEC EDGAR et sont utilisés avec leur date de publication. Les scores sont
          calculés sans données futures. Les prix sont ajustés et le benchmark est reconstruit à partir des composants.</p>
          <p class="note">La couverture est vérifiée à chaque date de rebalancement. Les titres sans prix exploitable ne
          peuvent pas être retenus comme anchors ; ils sont remplacés par le prochain poids éligible.</p>
        </div>
      </div>
    </section>
    """

    performance = f"""
    <section class="section" id="performance">
      <div class="section-head"><div><div class="kicker">Performance</div><h2>Trajectoire et rendements calendaires</h2></div>
        <span class="chip">CAGR {pct(comp.loc[base_top10, 'ann_return'])} / {pct(comp.loc[cap_top10, 'ann_return'])}</span></div>
      <img src="data:image/png;base64,{nav_figure}" alt="Valeur liquidative et drawdown">
      <img src="data:image/png;base64,{annual_figure}" alt="Rendements calendaires">
      {build_annual_table(data)}
      <p class="note">Les rendements sont calculés sur la valeur liquidative, nette des coûts de transaction. Le benchmark
      est l'indice MSCI USA Growth en prix, sans dividendes.</p>
    </section>
    """

    risk = f"""
    <section class="section" id="risque">
      <div class="section-head"><div><div class="kicker">Risque et structure</div><h2>Risque actif et concentration</h2></div>
        <span class="chip">TE {pct(comp.loc[base_top10, 'te_ann'])} / {pct(comp.loc[cap_top10, 'te_ann'])}</span></div>
      <img src="data:image/png;base64,{concentration_figure}" alt="Concentration">
      <div class="grid-2">
        <div>
          <h3>Effet du cap</h3>
          <p>La variante {esc(variant_label(data, base_top10))} affiche un poids maximal de
          {pct(comp.loc[base_top10, 'max_weight_last'])} et un top 10 de {pct(comp.loc[base_top10, 'top10_weight_last'])}.
          La version capee correspondante tombe à {pct(comp.loc[cap_top10, 'max_weight_last'])} de poids maximal et
          {pct(comp.loc[cap_top10, 'top10_weight_last'])} de top 10.</p>
          <p>Pour la paire 20 anchors, le poids maximal passe de {pct(comp.loc[base_top20, 'max_weight_last'])} à
          {pct(comp.loc[cap_top20, 'max_weight_last'])} et le top 10 de {pct(comp.loc[base_top20, 'top10_weight_last'])}
          à {pct(comp.loc[cap_top20, 'top10_weight_last'])}.</p>
        </div>
        <div>
          <h3>Coût du cap</h3>
          <p>Le cap 15 % réduit le CAGR de {spct(comp.loc[cap_top10, 'ann_return'] - comp.loc[base_top10, 'ann_return'])}
          pour la paire 10 anchors et de {spct(comp.loc[cap_top20, 'ann_return'] - comp.loc[base_top20, 'ann_return'])}
          pour la paire 20 anchors. La tracking error évolue de {pct(comp.loc[base_top10, 'te_ann'])} à
          {pct(comp.loc[cap_top10, 'te_ann'])} sur la première paire et de {pct(comp.loc[base_top20, 'te_ann'])} à
          {pct(comp.loc[cap_top20, 'te_ann'])} sur la seconde.</p>
          <p class="note">En moyenne, {fr(comp.loc[cap_top10, 'cap_bindings_avg'])} ligne par date touche le cap dans la
          sous-variante 10 anchors, contre {fr(comp.loc[cap_top20, 'cap_bindings_avg'])} dans la sous-variante 20 anchors.</p>
        </div>
      </div>
    </section>
    """

    sectors_section = f"""
    <section class="section" id="secteurs">
      <div class="section-head"><div><div class="kicker">Allocation</div><h2>Exposition sectorielle</h2></div>
        <span class="chip">Dernière date : {data['sectors']['date'].max():%d/%m/%Y}</span></div>
      <img src="data:image/png;base64,{sector_figure}" alt="Exposition sectorielle">
      {build_sector_table(data)}
      <p class="note">Les poids sont normalisés à 100 % à chaque date. L'exposition active est la différence entre la
      variante et le benchmark. Les secteurs absents sont affichés à 0 %.</p>
    </section>
    """

    portfolio_blocks = "".join(
        f"<h3>{esc(variant_label(data, name))}</h3>{build_holdings_table(data, name)}"
        for name in names
    )
    portfolio_section = f"""
    <section class="section" id="portefeuille">
      <div class="section-head"><div><div class="kicker">Titres</div><h2>Portefeuille à la dernière date</h2></div>
        <span class="chip">{data['holdings']['date'].max():%d/%m/%Y}</span></div>
      {portfolio_blocks}
      <p class="note">Le poids indice est le poids brut du titre dans le benchmark. Le poids rebasé est le poids d'indice
      des 60 titres normalisé à 100 %. La bande affiche les bornes inférieure et supérieure appliquées au poids final.
      La colonne Cap indique le plafond absolu de la sous-variante, le cas échéant.</p>
    </section>
    """

    contribution_blocks = "".join(
        f"<div><h3>{esc(variant_label(data, name))}</h3>{build_contribution_table(data, name)}</div>"
        for name in names
    )
    contributions_section = f"""
    <section class="section" id="contributions">
      <div class="section-head"><div><div class="kicker">Attribution</div><h2>Contributeurs principaux</h2></div></div>
      <img src="data:image/png;base64,{contributions_figure}" alt="Contributions">
      <div class="grid-2">{contribution_blocks}</div>
      <p class="note">Contribution approximative : poids de fin de journée précédent multiplié par le rendement du titre,
      agrégé sur toute la période. Les contributions sont exprimées en points de pourcentage.</p>
    </section>
    """

    robust_top10 = "top10_plus_50_ex_nvda"
    robust_top20 = "top20_plus_40_ex_nvda"
    robust_cap_top10 = "top10_plus_50_cap15_ex_nvda"
    robust_cap_top20 = "top20_plus_40_cap15_ex_nvda"
    robustness_section = f"""
    <section class="section" id="robustesse">
      <div class="section-head"><div><div class="kicker">Robustesse</div><h2>Que reste-t-il sans NVDA ?</h2></div>
        <span class="chip">CAGR {pct(comp.loc[robust_top10, 'ann_return'])} / {pct(comp.loc[robust_cap_top10, 'ann_return'])}</span></div>
      <p>Les quatre variantes sans NVDA excluent le titre avant la sélection des anchors et des score picks. Cette
      vérification isole la contribution de NVDA à la performance et au risque actif. Elle ne constitue pas un test
      hors échantillon, mais elle mesure la dépendance du résultat à un seul titre.</p>
      <img src="data:image/png;base64,{robustness_figure}" alt="Variantes sans NVDA">
      {build_robustness_table(data)}
      <div class="grid-2">
        <div>
          <h3>Alpha résiduel hors NVDA</h3>
          <p>La variante {esc(variant_label(data, robust_top10))} affiche un CAGR de
          {pct(comp.loc[robust_top10, 'ann_return'])} contre {pct(comp.loc[base_top10, 'ann_return'])} pour la version
          avec NVDA, et un alpha annualisé de {pct(comp.loc[robust_top10, 'alpha_ann'])} contre
          {pct(comp.loc[base_top10, 'alpha_ann'])}. Son information ratio passe de
          {fr(comp.loc[base_top10, 'info_ratio'])} à {fr(comp.loc[robust_top10, 'info_ratio'])}.</p>
          <p>La variante {esc(variant_label(data, robust_top20))} affiche un CAGR de
          {pct(comp.loc[robust_top20, 'ann_return'])} contre {pct(comp.loc[base_top20, 'ann_return'])} avec NVDA, et un
          alpha de {pct(comp.loc[robust_top20, 'alpha_ann'])} contre {pct(comp.loc[base_top20, 'alpha_ann'])}.</p>
        </div>
        <div>
          <h3>Lecture</h3>
          <p>Rapportées à l'indice ex-NVDA reconstruit, les stratégies sans NVDA conservent un alpha annualisé de
          {pct(comp.loc[robust_top10, 'alpha_ann'])} pour la paire 10 anchors et
          {pct(comp.loc[robust_top20, 'alpha_ann'])} pour la paire 20 anchors. NVDA explique donc une partie de la
          surperformance, mais la construction conserve un excédent hors NVDA. L'indice ex-NVDA affiche un CAGR de
          {pct(summary['bench_ex_nvda_stats']['ann_return'])} et un drawdown maximal de
          {pct(summary['bench_ex_nvda_stats']['max_dd'])}.</p>
          <p class="note">Le cap 15 % ne corrige pas la dépendance : il plafonne le poids de NVDA, mais le titre reste
          présent et continue de contribuer. Les versions capees sans NVDA affichent un alpha de
          {pct(comp.loc[robust_cap_top10, 'alpha_ann'])} et {pct(comp.loc[robust_cap_top20, 'alpha_ann'])} face à
          l'indice ex-NVDA, ce qui confirme la robustesse partielle du signal.</p>
        </div>
      </div>
    </section>
    """

    limits = f"""
    <section class="section" id="limites">
      <div class="section-head"><div><div class="kicker">Prudence</div><h2>Limites et avertissements</h2></div></div>
      <ul>
        <li><strong>Concentration résiduelle.</strong> Les variantes de base restent très concentrées : poids maximal de
        {pct(comp.loc[base_top10, 'max_weight_last'])} pour la variante 10 anchors et top 10 de
        {pct(comp.loc[base_top10, 'top10_weight_last'])}. Le cap 15 % réduit le poids individuel mais pas la dépendance
        aux grandes capitalisations.</li>
        <li><strong>Dépendance à NVDA.</strong> Le poids maximal de NVDA atteint {pct(comp.loc[base_top10, 'nvda_max_weight'])}
        dans la variante de base 10 anchors et {pct(comp.loc[cap_top10, 'nvda_max_weight'])} dans sa version capee.
        Sans NVDA, le CAGR tombe à {pct(comp.loc[robust_top10, 'ann_return'])} et l'alpha annualisé
        à {pct(comp.loc[robust_top10, 'alpha_ann'])} face à l'indice ex-NVDA, contre
        {pct(comp.loc[base_top10, 'alpha_ann'])} face à l'indice complet. La surperformance reste positive mais
        une partie de l'alpha est bien attribuable à NVDA.</li>
        <li><strong>Risque actif modéré.</strong> Les bandes de 50 % / 4 pp produisent une tracking error de
        {pct(comp.loc[base_top10, 'te_ann'])} à {pct(comp.loc[cap_top10, 'te_ann'])} sur la paire 10 anchors. Ces niveaux
        restent inférieurs à l'attente initiale de 6–9 % : la construction est plus proche d'un portefeuille core que d'une
        stratégie satellite très active.</li>
        <li><strong>Cap et bandes.</strong> Le cap 15 % peut dominer la bande et forcer une ligne à exactement 15 %. La
        contrainte devient alors plus contraignante que le mécanisme de tilt, ce qui explique une partie de l'écart de
        performance avec la variante de base.</li>
        <li><strong>Période courte et régime spécifique.</strong> Le backtest couvre 2021–2025, une période dominée par les
        grandes capitalisations technologiques. Les résultats ne constituent pas une validation hors échantillon.</li>
        <li><strong>Données et coûts.</strong> Les fondamentaux SEC peuvent contenir des retraitements et des décalages.
        Les coûts de transaction sont fixés à 10 bps et ne modélisent pas l'impact de marché.</li>
      </ul>
      <div class="callout" style="margin-top:18px;"><strong>Avertissement.</strong> Ce document est un exercice de recherche
      et ne constitue pas un conseil en investissement. Les performances passées ne préjugent pas des performances futures.</div>
    </section>
    """

    appendix = f"""
    <section class="section" id="annexe">
      <div class="section-head"><div><div class="kicker">Traçabilité</div><h2>Annexe</h2></div></div>
      <h3>Configuration</h3>
      {build_config_table(data)}
      <h3>Fichiers générés</h3>
      <p class="note">Les exports détaillés sont disponibles dans <code>reports/</code> :
      <code>anchor_summary.json</code>, <code>anchor_backtest.csv</code>, <code>anchor_annual.csv</code>,
      <code>anchor_quarterly.csv</code>, <code>anchor_sector_exposure.csv</code>, <code>anchor_coverage.csv</code>,
      <code>anchor_portfolio_summary.csv</code>, <code>anchor_holdings.csv</code>, <code>anchor_contributions.csv</code>,
      <code>anchor_variant_comparison.csv</code> et les poids quotidiens des huit variantes.</p>
    </section>
    """

    footer = f"""
    <footer class="footer">Rapport généré automatiquement le {generated} · Sources : MSCI USA Growth, SEC EDGAR,
    Yahoo Finance · Figures embarquées en base64.</footer>
    """

    return f"""<!DOCTYPE html>
<html lang="fr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Rapport des variantes 60 lignes anchors + accélération fondamentale">
<title>Anchors + accélération — MSCI USA Growth</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">
{hero}
{toc}
{synthesis}
{methodology}
{data_section}
{performance}
{risk}
{sectors_section}
{portfolio_section}
{contributions_section}
{robustness_section}
{limits}
{appendix}
{footer}
</div>
</body>
</html>
"""


def main() -> None:
    data = load_data()
    document = build_html(data)
    OUTPUT.write_text(document, encoding="utf-8")
    size = OUTPUT.stat().st_size
    text = OUTPUT.read_text(encoding="utf-8")
    if size < 50_000:
        raise RuntimeError(f"Rapport trop petit : {size} octets")
    if text.count("data:image/png;base64,") < 4:
        raise RuntimeError("Figures manquantes dans le rapport")
    print(f"Rapport généré : {OUTPUT}")
    print(f"Taille finale : {size / 1024:.1f} Ko ({size} octets)")


if __name__ == "__main__":
    main()
