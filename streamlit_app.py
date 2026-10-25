"""Dashboard Streamlit — Top 20 + 40 · variantes de cap (MSCI USA Growth).

Quatre variantes rebalancées trimestriellement : sans cap, cap 15 %, cap 12,5 %
et cap 10 %. Benchmark unique : MSCI USA Growth. Données issues de reports/.

Lancement :  streamlit run streamlit_app.py
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"

VARIANT_ORDER = [
    ("top20_plus_40", "Sans cap", "#8B949E", None),
    ("top20_plus_40_cap15", "Cap 15 %", "#2ECC71", 0.15),
    ("top20_plus_40_cap12_5", "Cap 12,5 %", "#F0B429", 0.125),
    ("top20_plus_40_cap10", "Cap 10 %", "#4C9AFF", 0.10),
]
VARIANT_NAMES = [name for name, _, _, _ in VARIANT_ORDER]
TAB_LABELS = {name: short for name, short, _, _ in VARIANT_ORDER}
COLORS = {name: color for name, _, color, _ in VARIANT_ORDER}
CAPS = {name: cap for name, _, _, cap in VARIANT_ORDER}

BENCH_COLOR = "#58A6FF"
BENCH_LABEL = "MSCI USA Growth"
SCORE_ONLY_KEY = "score_picks_40"
SCORE_ONLY_LABEL = "40 score picks · sans anchors"
SCORE_ONLY_COLOR = "#E879F9"
EX_TOP20_LABEL = "MSCI USA Growth ex-Top 20"
EX_TOP20_COLOR = "#22D3EE"
PLOTLY_TEMPLATE = "plotly_dark"
TRACE = "rgba(0,0,0,0)"
SECTOR_PALETTE = [
    "#4C9AFF", "#F0B429", "#2ECC71", "#E74C3C", "#B388FF", "#22D3EE",
    "#F97316", "#A3E635", "#F472B6", "#94A3B8", "#FACC15",
]
SIGNED = {"cum_return", "alpha_ann", "max_dd"}


def fmt(key: str, value) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    if key in {
        "cum_return", "ann_return", "vol_ann", "alpha_ann", "te_ann",
        "max_dd", "active_share", "top10_weight", "max_weight_last",
    }:
        return f"{v * 100:+.2f} %" if key in SIGNED else f"{v * 100:.2f} %"
    return f"{v:.2f}"


def pct(value, digits: int = 1, signed: bool = False) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not np.isfinite(v):
        return "—"
    return f"{v * 100:+.{digits}f} %" if signed else f"{v * 100:.{digits}f} %"


def layout(fig: go.Figure, title: str | None = None, height: int = 400) -> go.Figure:
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        title=dict(text=title, x=0, xanchor="left") if title else None,
        margin=dict(l=10, r=10, t=(64 if title else 30), b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0,
                    bgcolor=TRACE),
        hovermode="x unified",
    )
    return fig


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def relative(nav: pd.Series, bench: pd.Series) -> pd.Series:
    common = nav.index.intersection(bench.index)
    return nav.reindex(common) / bench.reindex(common) * 100.0


def section(title: str, caption: str | None = None) -> None:
    st.markdown(f"#### {title}")
    if caption:
        st.caption(caption)


@st.cache_data(show_spinner="Chargement des exports…")
def load_data() -> dict:
    with open(REPORTS / "anchor_summary.json", encoding="utf-8") as handle:
        summary = json.load(handle)

    daily = pd.read_csv(REPORTS / "anchor_backtest.csv", index_col=0)
    daily.index = pd.to_datetime(daily.index)
    annual = pd.read_csv(REPORTS / "anchor_annual.csv", index_col=0)
    quarterly = pd.read_csv(REPORTS / "anchor_quarterly.csv", index_col=0)
    comparison = pd.read_csv(REPORTS / "anchor_variant_comparison.csv")
    score_stats = pd.read_csv(
        REPORTS / "anchor_score_picks_40_stats.csv"
    ).iloc[0].to_dict()
    score_sectors = pd.read_csv(
        REPORTS / "anchor_score_picks_40_sector_exposure.csv"
    )
    score_sectors["date"] = pd.to_datetime(score_sectors["date"])
    score_contributions = pd.read_csv(
        REPORTS / "anchor_score_picks_40_contributions.csv"
    )
    score_overall = pd.read_csv(
        REPORTS / "anchor_score_picks_40_overall_contributions.csv", index_col=0
    )["contribution"]
    score_holdings = pd.read_csv(
        REPORTS / "anchor_score_picks_40_holdings.csv"
    )
    score_holdings["date"] = pd.to_datetime(score_holdings["date"])
    score_turnover = pd.read_csv(
        REPORTS / "anchor_score_picks_40_turnover.csv", index_col=0
    )["turnover"]
    score_weights = pd.read_parquet(
        REPORTS / "anchor_score_picks_40_weights_daily.parquet"
    )
    sectors = pd.read_csv(REPORTS / "anchor_sector_exposure.csv")
    sectors["date"] = pd.to_datetime(sectors["date"])
    holdings = pd.read_csv(REPORTS / "anchor_holdings.csv")
    holdings["date"] = pd.to_datetime(holdings["date"])
    contributions = pd.read_csv(REPORTS / "anchor_contributions.csv")
    coverage = pd.read_csv(REPORTS / "anchor_coverage.csv")
    coverage["date"] = pd.to_datetime(coverage["date"])
    portfolio = pd.read_csv(REPORTS / "anchor_portfolio_summary.csv")
    portfolio["date"] = pd.to_datetime(portfolio["date"])
    weights = {
        name: pd.read_parquet(REPORTS / f"anchor_{name}_weights_daily.parquet")
        for name in VARIANT_NAMES
    }
    sector_palette = {
        s: SECTOR_PALETTE[i % len(SECTOR_PALETTE)]
        for i, s in enumerate(sorted(sectors["sector"].dropna().unique()))
    }
    return {
        "summary": summary,
        "daily": daily,
        "annual": annual,
        "quarterly": quarterly,
        "comparison": comparison,
        "score_stats": score_stats,
        "score_sectors": score_sectors,
        "score_contributions": score_contributions,
        "score_overall": score_overall,
        "score_holdings": score_holdings,
        "score_turnover": score_turnover,
        "score_weights": score_weights,
        "sectors": sectors,
        "holdings": holdings,
        "contributions": contributions,
        "coverage": coverage,
        "portfolio": portfolio,
        "weights": weights,
        "sector_palette": sector_palette,
    }


def labels(data: dict) -> dict[str, str]:
    raw = data["summary"]["variants"]
    return {name: raw[name]["label"] for name in VARIANT_NAMES}


def stats_row(data: dict, name: str) -> dict:
    return data["comparison"].set_index("variant").loc[name].to_dict()


def rebalance_dates(data: dict) -> list[pd.Timestamp]:
    return sorted(pd.to_datetime(data["portfolio"]["date"].unique()))


def monthly_weights(weights: pd.DataFrame) -> pd.DataFrame:
    return weights.resample("ME").last()


def concentration(weights: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly_weights(weights)
    effective = 1.0 / monthly.pow(2).sum(axis=1).replace(0, np.nan)
    return pd.DataFrame({
        "max_weight": monthly.max(axis=1),
        "top10": monthly.apply(lambda row: row.nlargest(10).sum(), axis=1),
        "effective_n": effective,
    })


def holdings_heatmap_data(data: dict, name: str, n: int = 20):
    weights = data["weights"][name]
    dates = [d for d in rebalance_dates(data) if d in weights.index]
    top = weights.max().sort_values(ascending=False).head(n).index
    frame = weights.loc[dates, top].T
    return dates, list(top), frame


def nav_chart(data: dict, names: list[str], height: int = 480) -> go.Figure:
    daily = data["daily"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=daily["nav_benchmark"], name=BENCH_LABEL,
        line=dict(color=BENCH_COLOR, width=2.6, dash="dash"),
    ))
    for name in names:
        fig.add_trace(go.Scatter(
            x=daily.index, y=daily[f"nav_{name}"], name=labels(data)[name],
            line=dict(color=COLORS[name], width=2.0),
        ))
    fig.update_yaxes(title_text="Base 100")
    return layout(fig, "Valeur liquidative — Base 100", height)


def relative_chart(data: dict, names: list[str], height: int = 380) -> go.Figure:
    daily = data["daily"]
    bench = daily["nav_benchmark"]
    fig = go.Figure()
    for name in names:
        rel = relative(daily[f"nav_{name}"].dropna(), bench)
        fig.add_trace(go.Scatter(
            x=rel.index, y=rel.values, name=TAB_LABELS[name],
            line=dict(color=COLORS[name], width=1.8),
        ))
    fig.add_hline(y=100, line=dict(color="#8B949E", width=1, dash="dot"))
    fig.update_yaxes(title_text="Base 100 vs benchmark")
    return layout(fig, "Performance relative au benchmark", height)


def drawdown_chart(data: dict, names: list[str], height: int = 340) -> go.Figure:
    daily = data["daily"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily.index, y=drawdown(daily["nav_benchmark"]) * 100,
        name=BENCH_LABEL,
        line=dict(color=BENCH_COLOR, width=1.4, dash="dash"),
    ))
    for name in names:
        nav = daily[f"nav_{name}"].dropna()
        fig.add_trace(go.Scatter(
            x=nav.index, y=drawdown(nav) * 100, name=TAB_LABELS[name],
            line=dict(color=COLORS[name], width=1.6),
        ))
    fig.update_yaxes(title_text="Drawdown (%)")
    return layout(fig, "Drawdowns quotidiens", height)


def sector_area_chart(data: dict, name: str, height: int = 440) -> go.Figure:
    sectors = data["sectors"]
    frame = sectors[sectors["variant"] == name]
    pivot = frame.pivot_table(index="date", columns="sector", values="strategy",
                              aggfunc="sum").fillna(0.0)
    palette = data["sector_palette"]
    fig = go.Figure()
    for sector in pivot.columns:
        fig.add_trace(go.Scatter(
            x=pivot.index, y=pivot[sector] * 100, name=sector,
            mode="lines", stackgroup="one",
            line=dict(width=0.5, color=palette.get(sector)),
        ))
    fig.update_yaxes(title_text="Poids (%)")
    return layout(fig, "Évolution de la composition sectorielle", height)


def sector_active_heatmap(data: dict, name: str, height: int = 420) -> go.Figure:
    sectors = data["sectors"]
    frame = sectors[sectors["variant"] == name]
    pivot = frame.pivot_table(index="date", columns="sector", values="active",
                              aggfunc="sum").fillna(0.0) * 100
    pivot = pivot[sorted(pivot.columns)]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns),
        y=[d.strftime("%m/%Y") for d in pivot.index],
        colorscale="RdYlGn", zmid=0,
        text=np.round(pivot.values, 1),
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(title="Δ %"),
    ))
    fig.update_xaxes(tickangle=-35)
    return layout(fig, "Écarts sectoriels actifs vs benchmark (points de %)", height)


def anchors_picks_chart(data: dict, name: str, height: int = 360) -> go.Figure:
    frame = data["portfolio"]
    frame = frame[frame["variant"] == name].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame["date"], y=frame["anchor_target_weight"] * 100,
        name="Anchors", mode="lines", stackgroup="one",
        line=dict(color="#4C9AFF", width=0.8),
    ))
    fig.add_trace(go.Scatter(
        x=frame["date"], y=frame["score_target_weight"] * 100,
        name="Score picks", mode="lines", stackgroup="one",
        line=dict(color="#B388FF", width=0.8),
    ))
    fig.update_yaxes(title_text="Poids (%)")
    return layout(fig, "Répartition Anchors / Score picks", height)


def concentration_chart(data: dict, name: str, height: int = 400) -> go.Figure:
    frame = concentration(data["weights"][name])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["max_weight"] * 100, name="Poids max",
        line=dict(color=COLORS[name], width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["top10"] * 100, name="Top 10",
        line=dict(color="#F0B429", width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=frame.index, y=frame["effective_n"], name="N effectif",
        yaxis="y2", line=dict(color="#8B949E", width=1.4),
    ))
    cap = CAPS[name]
    if cap is not None:
        fig.add_hline(y=cap * 100, line=dict(color="#E74C3C", width=1.2, dash="dot"),
                      annotation_text=f"Cap {cap * 100:.1f} %".replace(".", ","),
                      annotation_position="top left")
    fig.update_layout(
        yaxis=dict(title="Poids (%)"),
        yaxis2=dict(title="N effectif", overlaying="y", side="right",
                    showgrid=False),
    )
    return layout(fig, "Concentration et diversification dans le temps", height)


def cap_binding_chart(data: dict, name: str, height: int = 320):
    frame = data["portfolio"]
    frame = frame[frame["variant"] == name].sort_values("date")
    if frame["cap_bindings"].notna().sum() == 0 or frame["cap_bindings"].max() == 0:
        return None
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=frame["date"], y=frame["cap_bindings"], name="Lignes au cap",
        marker_color="#F0B429",
    ))
    fig.add_trace(go.Bar(
        x=frame["date"], y=frame["cap_forced"], name="Lignes forcées au cap",
        marker_color="#E74C3C",
    ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Nombre de lignes")
    return layout(fig, "Lignes au cap à chaque rebalancement", height)


def holdings_heatmap(data: dict, name: str, height: int = 460) -> go.Figure:
    dates, tickers, frame = holdings_heatmap_data(data, name, n=20)
    fig = go.Figure(go.Heatmap(
        z=frame.values * 100, x=[d.strftime("%m/%Y") for d in dates],
        y=tickers, colorscale="Tealgrn",
        text=np.round(frame.values * 100, 1),
        texttemplate="%{text}", textfont=dict(size=7),
        colorbar=dict(title="Poids %"),
    ))
    fig.update_xaxes(tickangle=-40)
    return layout(fig, "Poids des 20 principaux titres par rebalancement", height)


def top10_area_chart(data: dict, name: str, height: int = 400) -> go.Figure:
    weights = data["weights"][name]
    dates = [d for d in rebalance_dates(data) if d in weights.index]
    top = weights.max().sort_values(ascending=False).head(12).index
    frame = weights.loc[dates, top].copy()
    frame["Autres"] = (1.0 - frame.sum(axis=1)).clip(lower=0.0)
    palette = data["sector_palette"]
    fig = go.Figure()
    for index, ticker in enumerate(list(top) + ["Autres"]):
        color = "#334155" if ticker == "Autres" else SECTOR_PALETTE[index % len(SECTOR_PALETTE)]
        fig.add_trace(go.Scatter(
            x=dates, y=frame[ticker] * 100, name=ticker, mode="lines",
            stackgroup="one", line=dict(width=0.5, color=color),
        ))
    fig.update_yaxes(title_text="Poids (%)")
    return layout(fig, "Évolution du Top 12 des lignes", height)


def treemap_chart(data: dict, name: str, height: int = 480) -> go.Figure:
    holdings = data["holdings"]
    holdings = holdings[holdings["variant"] == name]
    holdings = holdings[holdings["date"] == holdings["date"].max()]
    holdings = holdings.sort_values("target_weight", ascending=False)
    fig = go.Figure(go.Treemap(
        labels=holdings["ticker"],
        parents=holdings["sector"],
        values=holdings["target_weight"],
        branchvalues="total",
        text=[f"{pct(w, 2)} · actif {pct(a, 2, True)}"
              for w, a in zip(holdings["target_weight"], holdings["active_weight"])],
        hovertemplate="%{label}<br>%{text}<extra></extra>",
        marker=dict(
            colors=holdings["active_weight"],
            colorscale="RdYlGn", cmid=0,
            colorbar=dict(title="Actif"),
        ),
    ))
    return layout(fig, "Portefeuille à la dernière date", height)


def sector_radar_chart(data: dict, name: str, height: int = 440) -> go.Figure:
    sectors = data["sectors"]
    sectors = sectors[sectors["variant"] == name]
    latest = sectors["date"].max()
    frame = sectors[sectors["date"] == latest]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=frame["benchmark"] * 100, theta=frame["sector"],
        fill="toself", name=BENCH_LABEL,
        line=dict(color=BENCH_COLOR, width=1.8),
        opacity=0.5,
    ))
    fig.add_trace(go.Scatterpolar(
        r=frame["strategy"] * 100, theta=frame["sector"],
        fill="toself", name=TAB_LABELS[name],
        line=dict(color=COLORS[name], width=2),
        opacity=0.55,
    ))
    fig.update_layout(polar=dict(radialaxis=dict(visible=True, ticksuffix="%")),
                      showlegend=True)
    return layout(fig, f"Exposition sectorielle au {latest:%d/%m/%Y}", height)


def annual_bars_chart(data: dict, name: str, height: int = 340) -> go.Figure:
    annual = data["annual"]
    label = labels(data)[name]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[str(y) for y in annual.index], y=annual[label] * 100,
        name=TAB_LABELS[name], marker_color=COLORS[name],
    ))
    fig.add_trace(go.Scatter(
        x=[str(y) for y in annual.index], y=annual["Benchmark"] * 100,
        name=BENCH_LABEL, mode="lines+markers",
        line=dict(color=BENCH_COLOR, width=2),
    ))
    fig.update_yaxes(title_text="Rendement (%)")
    return layout(fig, "Rendements calendaires", height)


def quarterly_heatmap(data: dict, name: str, height: int = 340) -> go.Figure:
    quarterly = data["quarterly"][labels(data)[name]] * 100
    frame = pd.DataFrame({
        "year": [int(str(i)[:4]) for i in quarterly.index],
        "quarter": [str(i)[-1] for i in quarterly.index],
        "value": quarterly.values,
    })
    pivot = frame.pivot(index="year", columns="quarter", values="value")
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"T{c}" for c in pivot.columns],
        y=[str(i) for i in pivot.index], colorscale="RdYlGn", zmid=0,
        text=np.round(pivot.values, 1), texttemplate="%{text}",
        textfont=dict(size=9), colorbar=dict(title="%"),
    ))
    return layout(fig, "Rendements trimestriels (%)", height)


def contributions_bar(data: dict, name: str, height: int = 440) -> go.Figure:
    series = pd.Series(
        data["summary"]["variants"][name]["overall_contributions"], dtype=float
    ).sort_values(ascending=False).head(15).sort_values()
    fig = go.Figure(go.Bar(
        x=series.values * 100, y=list(series.index), orientation="h",
        marker_color=["#F0B429" if t == "NVDA" else COLORS[name]
                      for t in series.index],
        text=[f"{v * 100:.1f} %" for v in series.values],
        textposition="outside",
    ))
    fig.update_xaxes(title_text="Contribution cumulée (points de %)")
    return layout(fig, "Top 15 des contributeurs", height)


def contributions_heatmap(data: dict, name: str, height: int = 380) -> go.Figure:
    frame = data["contributions"]
    frame = frame[frame["variant"] == name]
    top = frame.groupby("ticker")["contribution"].sum().abs().nlargest(12).index
    pivot = frame[frame["ticker"].isin(top)].pivot_table(
        index="year", columns="ticker", values="contribution", aggfunc="sum"
    ).fillna(0.0) * 100
    pivot = pivot[sorted(pivot.columns)]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=list(pivot.columns),
        y=[str(y) for y in pivot.index], colorscale="RdYlGn", zmid=0,
        text=np.round(pivot.values, 1), texttemplate="%{text}",
        textfont=dict(size=8), colorbar=dict(title="%"),
    ))
    fig.update_xaxes(tickangle=-35)
    return layout(fig, "Contributions annuelles des principaux titres (%)", height)


def overview_tab(data: dict) -> None:
    names = VARIANT_NAMES
    comparison = data["comparison"].set_index("variant")
    bench = data["summary"]["bench_stats"]
    config = data["summary"]["config"]

    st.subheader("Vue d'ensemble — quatre niveaux de cap")
    st.caption(
        f"Univers MSCI USA Growth · {config['n_tickers']} titres · "
        f"{config['n_dates']} rebalancements trimestriels · "
        f"{pd.Timestamp(config['start']):%d/%m/%Y} → "
        f"{pd.Timestamp(config['end']):%d/%m/%Y} · 60 lignes · "
        f"benchmark {BENCH_LABEL}."
    )

    columns = st.columns(5)
    columns[0].metric("Benchmark CAGR", pct(bench["ann_return"]),
                      pct(bench["vol_ann"]) + " de vol.")
    for column, name in zip(columns[1:], names):
        row = comparison.loc[name]
        column.metric(TAB_LABELS[name], pct(row["ann_return"]),
                      pct(row["alpha_ann"], signed=True) + " d'alpha")

    st.plotly_chart(nav_chart(data, names, height=520), use_container_width=True)
    st.plotly_chart(relative_chart(data, names, height=380), use_container_width=True)

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        for name in names:
            row = comparison.loc[name]
            fig.add_trace(go.Scatter(
                x=[row["te_ann"] * 100], y=[row["ann_return"] * 100],
                mode="markers+text", name=TAB_LABELS[name],
                text=[TAB_LABELS[name]], textposition="bottom center",
                marker=dict(size=18, color=COLORS[name],
                            line=dict(color="#0f172a", width=1)),
            ))
        fig.update_xaxes(title_text="Tracking error (%)")
        fig.update_yaxes(title_text="CAGR (%)")
        st.plotly_chart(layout(fig, "CAGR vs risque actif", height=380),
                        use_container_width=True)
    with right:
        fig = go.Figure()
        for name in names:
            row = comparison.loc[name]
            fig.add_trace(go.Scatter(
                x=[row["max_dd"] * 100], y=[row["sharpe"]],
                mode="markers+text", name=TAB_LABELS[name],
                text=[TAB_LABELS[name]], textposition="bottom center",
                marker=dict(size=18, color=COLORS[name],
                            line=dict(color="#0f172a", width=1)),
            ))
        fig.update_xaxes(title_text="Max drawdown (%)")
        fig.update_yaxes(title_text="Sharpe")
        st.plotly_chart(layout(fig, "Sharpe vs drawdown maximal", height=380),
                        use_container_width=True)

    section("Concentration et effet du cap")
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[TAB_LABELS[n] for n in names],
            y=[comparison.loc[n, "max_weight_last"] * 100 for n in names],
            name="Poids max", marker_color=[COLORS[n] for n in names],
            text=[pct(comparison.loc[n, "max_weight_last"])
                  for n in names], textposition="outside",
        ))
        fig.update_yaxes(title_text="Poids (%)")
        st.plotly_chart(layout(fig, "Poids maximal d'une ligne (dernier rebalancement)",
                               height=340), use_container_width=True)
    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[TAB_LABELS[n] for n in names],
            y=[comparison.loc[n, "top10_weight_last"] * 100 for n in names],
            name="Top 10", marker_color=[COLORS[n] for n in names],
            text=[pct(comparison.loc[n, "top10_weight_last"])
                  for n in names], textposition="outside",
        ))
        fig.update_yaxes(title_text="Poids cumulé (%)")
        st.plotly_chart(layout(fig, "Poids cumulé du Top 10 (dernier rebalancement)",
                               height=340), use_container_width=True)

    fig = go.Figure()
    for name in names:
        frame = concentration(data["weights"][name])
        fig.add_trace(go.Scatter(
            x=frame.index, y=frame["max_weight"] * 100,
            name=TAB_LABELS[name], line=dict(color=COLORS[name], width=1.5),
        ))
        if CAPS[name] is not None:
            fig.add_hline(y=CAPS[name] * 100,
                          line=dict(color=COLORS[name], width=1, dash="dot"))
    fig.update_yaxes(title_text="Poids max (%)")
    st.plotly_chart(layout(fig, "Dérive du poids maximal entre deux rebalancements",
                           height=400), use_container_width=True)

    section("Rendements calendaires")
    annual = data["annual"]
    variant_labels = [labels(data)[n] for n in names]
    z = np.array([annual[label].values for label in variant_labels]) * 100
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(y) for y in annual.index],
        y=[TAB_LABELS[n] for n in names], colorscale="RdYlGn", zmid=0,
        text=np.round(z, 1), texttemplate="%{text}", textfont=dict(size=10),
        colorbar=dict(title="%"),
    ))
    st.plotly_chart(layout(fig, "Rendements annuels par variante (%)", height=340),
                    use_container_width=True)

    section("Biais sectoriels à la dernière date")
    sectors = data["sectors"]
    latest = sectors["date"].max()
    frame = sectors[sectors["date"] == latest]
    pivot = frame.pivot_table(index="sector", columns="variant",
                              values="active", aggfunc="sum").fillna(0.0) * 100
    pivot = pivot[names]
    fig = go.Figure(go.Heatmap(
        z=pivot.values, x=[TAB_LABELS[n] for n in names],
        y=list(pivot.index), colorscale="RdYlGn", zmid=0,
        text=np.round(pivot.values, 1), texttemplate="%{text}",
        textfont=dict(size=9), colorbar=dict(title="Δ %"),
    ))
    st.plotly_chart(layout(fig, f"Écarts actifs sectoriels au {latest:%d/%m/%Y} "
                               "vs benchmark (points de %)", height=460),
                    use_container_width=True)

    section("Tableau comparatif")
    rows = []
    for name in names:
        row = comparison.loc[name]
        rows.append({
            "Variante": TAB_LABELS[name],
            "CAGR": pct(row["ann_return"]),
            "Volatilité": pct(row["vol_ann"]),
            "Sharpe": f"{row['sharpe']:.2f}",
            "Max DD": pct(row["max_dd"]),
            "Alpha": pct(row["alpha_ann"], signed=True),
            "TE": pct(row["te_ann"]),
            "IR": f"{row['info_ratio']:.2f}",
            "Bêta": f"{row['beta']:.2f}",
            "Poids max": pct(row["max_weight_last"]),
            "Top 10": pct(row["top10_weight_last"]),
            "N effectif": f"{row['n_effective']:.1f}",
        })
    table = pd.DataFrame(rows)
    st.dataframe(table, hide_index=True, use_container_width=True)
    st.download_button(
        "⬇️ Exporter le comparatif (CSV)",
        data=table.to_csv(index=False).encode("utf-8"),
        file_name="comparatif_caps.csv",
        mime="text/csv",
    )


def variant_tab(data: dict, name: str) -> None:
    row = stats_row(data, name)
    daily = data["daily"]
    nav = daily[f"nav_{name}"].dropna()
    bench = daily["nav_benchmark"].dropna()

    st.subheader(labels(data)[name])
    if CAPS[name] is not None:
        st.caption(f"Cap de {CAPS[name] * 100:.1f} % par ligne appliqué à chaque "
                   "rebalancement trimestriel.".replace(".", ","))
    else:
        st.caption("Aucun cap individuel : les poids dérivent librement entre "
                   "deux rebalancements trimestriels.")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rendement cumulé", fmt("cum_return", row["cum_return"]))
    c2.metric("CAGR", fmt("ann_return", row["ann_return"]))
    c3.metric("Alpha", fmt("alpha_ann", row["alpha_ann"]))
    c4.metric("TE", fmt("te_ann", row["te_ann"]))
    c5.metric("IR", f"{row['info_ratio']:.2f}")
    c6.metric("Max DD", fmt("max_dd", row["max_dd"]))

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Volatilité", fmt("vol_ann", row["vol_ann"]))
    c2.metric("Sharpe", f"{row['sharpe']:.2f}")
    c3.metric("Sortino", f"{row['sortino']:.2f}")
    c4.metric("Bêta", f"{row['beta']:.2f}")
    c5.metric("Poids max final", fmt("max_weight_last", row["max_weight_last"]))
    c6.metric("Top 10 final", fmt("top10_weight_last", row["top10_weight_last"]))

    section("Performance")
    left, right = st.columns([1.35, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=bench.index, y=bench.values,
                                 name=BENCH_LABEL,
                                 line=dict(color=BENCH_COLOR, width=2.2, dash="dash")))
        fig.add_trace(go.Scatter(x=nav.index, y=nav.values,
                                 name=TAB_LABELS[name],
                                 line=dict(color=COLORS[name], width=2.4)))
        fig.update_yaxes(title_text="Base 100")
        st.plotly_chart(layout(fig, "Valeur liquidative", height=380),
                        use_container_width=True)
    with right:
        rel = relative(nav, bench)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rel.index, y=rel.values,
                                 name="Performance relative",
                                 line=dict(color=COLORS[name], width=2),
                                 fill="tozeroy"))
        fig.add_hline(y=100, line=dict(color="#8B949E", width=1, dash="dot"))
        fig.update_yaxes(title_text="Base 100")
        st.plotly_chart(layout(fig, "Alpha cumulé", height=380),
                        use_container_width=True)
    st.plotly_chart(drawdown_chart(data, [name], height=300),
                    use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(annual_bars_chart(data, name), use_container_width=True)
    with right:
        st.plotly_chart(quarterly_heatmap(data, name), use_container_width=True)

    section("Évolution de la composition")
    st.plotly_chart(sector_area_chart(data, name), use_container_width=True)
    st.plotly_chart(sector_active_heatmap(data, name), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(anchors_picks_chart(data, name), use_container_width=True)
    with right:
        st.plotly_chart(concentration_chart(data, name), use_container_width=True)

    binding = cap_binding_chart(data, name)
    if binding is not None:
        st.plotly_chart(binding, use_container_width=True)

    st.plotly_chart(holdings_heatmap(data, name), use_container_width=True)
    st.plotly_chart(top10_area_chart(data, name), use_container_width=True)

    left, right = st.columns([1.25, 1])
    with left:
        st.plotly_chart(treemap_chart(data, name), use_container_width=True)
    with right:
        st.plotly_chart(sector_radar_chart(data, name), use_container_width=True)

    section("Titres & attribution")
    holdings = data["holdings"]
    holdings = holdings[(holdings["variant"] == name)
                        & (holdings["date"] == holdings["date"].max())]
    holdings = holdings.sort_values("target_weight", ascending=False)
    display = holdings[[
        "ticker", "sector", "z", "is_anchor", "benchmark_weight",
        "target_weight", "active_weight", "cap_binding",
    ]].copy()
    display.columns = ["Titre", "Secteur", "Score Z", "Anchor",
                       "Poids indice", "Poids cible", "Actif", "Cap"]
    for col in ("Poids indice", "Poids cible"):
        display[col] = display[col].map(lambda v: pct(v, 2))
    display["Actif"] = display["Actif"].map(lambda v: pct(v, 2, signed=True))
    display["Score Z"] = display["Score Z"].map(lambda v: f"{v:.2f}")
    display["Anchor"] = display["Anchor"].map(lambda v: "Oui" if v else "Non")
    display["Cap"] = display["Cap"].map(lambda v: "Oui" if v else "")
    st.markdown(f"**Portefeuille au {holdings['date'].max():%d/%m/%Y}**")
    st.dataframe(display, hide_index=True, use_container_width=True, height=320)

    left, right = st.columns(2)
    with left:
        st.plotly_chart(contributions_bar(data, name), use_container_width=True)
    with right:
        st.plotly_chart(contributions_heatmap(data, name), use_container_width=True)

    stats = {
        "Rendement cumulé": fmt("cum_return", row["cum_return"]),
        "CAGR": fmt("ann_return", row["ann_return"]),
        "Volatilité": fmt("vol_ann", row["vol_ann"]),
        "Sharpe": f"{row['sharpe']:.2f}",
        "Sortino": f"{row['sortino']:.2f}",
        "Alpha annualisé": fmt("alpha_ann", row["alpha_ann"]),
        "Tracking error": fmt("te_ann", row["te_ann"]),
        "Information ratio": f"{row['info_ratio']:.2f}",
        "Bêta": f"{row['beta']:.2f}",
        "Max drawdown": fmt("max_dd", row["max_dd"]),
        "Active share": fmt("active_share", row["active_share"]),
        "Poids Top 10 moyen": fmt("top10_weight", row["top10_weight"]),
        "N effectif moyen": f"{row['n_effective']:.1f}",
        "Cap effectif": pct(CAPS[name]) if CAPS[name] is not None else "Aucun",
    }
    st.dataframe(pd.DataFrame({"Indicateur": list(stats.keys()),
                               "Valeur": list(stats.values())}),
                 hide_index=True, use_container_width=True)


def yearly_tab(data: dict) -> None:
    daily = data["daily"]
    years = sorted({int(y) for y in daily.index.year if int(y) >= 2021})

    st.subheader("Performance année par année")
    st.caption("Chaque graphique rebase les quatre variantes et le benchmark à "
               "100 au premier jour de bourse de l'année.")

    for year in years:
        frame = daily[daily.index.year == year]
        if frame.empty:
            continue
        fig = go.Figure()
        bench = frame["nav_benchmark"].dropna()
        fig.add_trace(go.Scatter(
            x=bench.index, y=bench / bench.iloc[0] * 100,
            name=BENCH_LABEL,
            line=dict(color=BENCH_COLOR, width=2.4, dash="dash"),
        ))
        for name in VARIANT_NAMES:
            nav = frame[f"nav_{name}"].dropna()
            if nav.empty:
                continue
            fig.add_trace(go.Scatter(
                x=nav.index, y=nav / nav.iloc[0] * 100,
                name=TAB_LABELS[name],
                line=dict(color=COLORS[name], width=2.0),
            ))
        fig.add_hline(y=100, line=dict(color="#8B949E", width=0.8, dash="dot"))
        fig.update_yaxes(title_text="Base 100 au 1er janvier")
        st.plotly_chart(layout(fig, f"Performance en {year}", height=380),
                        use_container_width=True)


def score_only_tab(data: dict) -> None:
    stats = data["score_stats"]
    bench = data["summary"]["score_picks_40"]["benchmark_stats"]
    daily = data["daily"]
    nav = daily[f"nav_{SCORE_ONLY_KEY}"].dropna()
    bench_nav = daily["nav_benchmark_ex_top20"].dropna()

    st.subheader(SCORE_ONLY_LABEL)
    st.caption(
        "Même sélection trimestrielle que Top 20 + 40, mais les 20 anchors sont "
        "retirés : seuls les 40 score picks sont conservés, puis renormalisés à "
        "100 % à chaque rebalancement. Aucun cap individuel. Benchmark : MSCI "
        "USA Growth privé de ses 20 anchors, renormalisé chaque trimestre."
    )

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("CAGR", pct(stats["ann_return"]),
              pct(stats["ann_return"] - bench["ann_return"], signed=True)
              + " vs benchmark")
    c2.metric("Alpha annualisé", pct(stats["alpha_ann"], signed=True))
    c3.metric("Tracking error", pct(stats["te_ann"]))
    c4.metric("Information ratio", f"{stats['info_ratio']:.2f}")
    c5.metric("Max drawdown", pct(stats["max_dd"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=bench_nav.index, y=bench_nav.values, name=EX_TOP20_LABEL,
        line=dict(color=EX_TOP20_COLOR, width=2.6, dash="dash"),
    ))
    fig.add_trace(go.Scatter(
        x=nav.index, y=nav.values, name=SCORE_ONLY_LABEL,
        line=dict(color=SCORE_ONLY_COLOR, width=2.2),
    ))
    fig.update_yaxes(title_text="Base 100")
    st.plotly_chart(layout(fig, "40 score picks vs benchmark ex-Top 20 — Base 100",
                           height=480), use_container_width=True)

    left, right = st.columns(2)
    with left:
        rel = relative(nav, bench_nav)
        fig = go.Figure(go.Scatter(
            x=rel.index, y=rel.values, name=SCORE_ONLY_LABEL,
            line=dict(color=SCORE_ONLY_COLOR, width=2.0),
            fill="tozeroy", fillcolor="rgba(232,121,249,0.10)",
        ))
        fig.add_hline(y=100, line=dict(color="#8B949E", width=1, dash="dot"))
        fig.update_yaxes(title_text="Base 100 vs benchmark")
        st.plotly_chart(layout(fig, "Performance relative", height=360),
                        use_container_width=True)
    with right:
        fig = go.Figure()
        for series, name, color, dash in [
            (bench_nav, EX_TOP20_LABEL, EX_TOP20_COLOR, "dash"),
            (nav, SCORE_ONLY_LABEL, SCORE_ONLY_COLOR, "solid"),
        ]:
            fig.add_trace(go.Scatter(
                x=series.index, y=drawdown(series) * 100, name=name,
                line=dict(color=color, width=1.8, dash=dash),
            ))
        fig.update_yaxes(title_text="Drawdown (%)")
        st.plotly_chart(layout(fig, "Drawdowns comparés", height=360),
                        use_container_width=True)

    annual = data["annual"]
    years = [str(y) for y in annual.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=annual[SCORE_ONLY_LABEL] * 100, name=SCORE_ONLY_LABEL,
        marker_color=SCORE_ONLY_COLOR,
        text=[pct(v) for v in annual[SCORE_ONLY_LABEL]], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=years, y=annual["Benchmark ex-Top 20"] * 100, name=EX_TOP20_LABEL,
        marker_color=EX_TOP20_COLOR,
        text=[pct(v) for v in annual["Benchmark ex-Top 20"]],
        textposition="outside",
    ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Rendement (%)")
    st.plotly_chart(layout(fig, "Rendements calendaires", height=380),
                    use_container_width=True)


def hac_alpha(active: pd.Series) -> dict:
    values = pd.Series(active).dropna().to_numpy(dtype=float)
    n = len(values)
    mean = float(values.mean())
    errors = values - mean
    lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    variance = float((errors * errors).mean())
    for shift in range(1, lag + 1):
        gamma = float((errors[shift:] * errors[:-shift]).mean())
        variance += 2 * (1 - shift / (lag + 1)) * gamma
    se = np.sqrt(max(variance, 0.0) / n)
    t_stat = mean / se if se > 0 else np.nan
    p_value = float(math.erfc(abs(t_stat) / np.sqrt(2))) if se > 0 else np.nan
    return {
        "alpha": mean * 252,
        "se": se * np.sqrt(252),
        "t_stat": t_stat,
        "p_value": p_value,
        "lag": lag,
        "n": n,
    }


def block_bootstrap_alpha(active: pd.Series, n_boot: int = 1000,
                          block: int = 21, seed: int = 7) -> dict:
    values = pd.Series(active).dropna().to_numpy(dtype=float)
    n = len(values)
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n
    indices = indices.reshape(n_boot, -1)[:, :n]
    means = values[indices].mean(axis=1) * 252
    return {
        "low": float(np.percentile(means, 2.5)),
        "high": float(np.percentile(means, 97.5)),
        "positive": float((means > 0).mean()),
        "mean": float(means.mean()),
        "distribution": means,
    }


def monthly_capture(ret: pd.Series, bench: pd.Series) -> tuple[float, float]:
    monthly_ret = (1 + ret.resample("ME").apply(lambda r: (1 + r).prod() - 1)) - 1
    monthly_bench = (1 + bench.resample("ME").apply(lambda r: (1 + r).prod() - 1)) - 1
    frame = pd.concat([monthly_ret.rename("p"), monthly_bench.rename("b")],
                      axis=1).dropna()
    up = frame["b"] > 0
    down = frame["b"] < 0
    up_capture = frame.loc[up, "p"].mean() / frame.loc[up, "b"].mean()
    down_capture = frame.loc[down, "p"].mean() / frame.loc[down, "b"].mean()
    return float(up_capture), float(down_capture)


def drawdown_episodes(nav: pd.Series, top: int = 5) -> pd.DataFrame:
    peak_date = nav.index[0]
    peak_value = nav.iloc[0]
    trough_date = None
    trough_value = 0.0
    in_drawdown = False
    episodes = []
    for date, value in nav.items():
        if value >= peak_value:
            if in_drawdown and trough_date is not None:
                episodes.append({
                    "peak": peak_date,
                    "trough": trough_date,
                    "recovery": date,
                    "depth": trough_value,
                    "days": (date - peak_date).days,
                })
            peak_date = date
            peak_value = value
            in_drawdown = False
            trough_date = None
            trough_value = 0.0
            continue
        depth = value / peak_value - 1.0
        if not in_drawdown or depth < trough_value:
            trough_date = date
            trough_value = depth
        in_drawdown = True
    if in_drawdown and trough_date is not None:
        episodes.append({
            "peak": peak_date,
            "trough": trough_date,
            "recovery": None,
            "depth": trough_value,
            "days": (nav.index[-1] - peak_date).days,
        })
    frame = pd.DataFrame(episodes).sort_values("depth").head(top)
    return frame


def analysis_tab(data: dict) -> None:
    stats = data["score_stats"]
    bench = data["summary"]["score_picks_40"]["benchmark_stats"]
    daily = data["daily"]
    ret = daily[f"ret_{SCORE_ONLY_KEY}"].dropna()
    bench_ret = daily["ret_benchmark_ex_top20"].reindex(ret.index)
    active = (ret - bench_ret).dropna()
    hac = hac_alpha(active)
    boot = block_bootstrap_alpha(active)
    up_capture, down_capture = monthly_capture(ret, bench_ret)

    st.subheader("Analyse complète et honnête — 40 score picks")
    st.caption(
        "Portefeuille : 40 score picks du Top 20 + 40, anchors retirés, poids "
        "renormalisés à chaque rebalancement. Benchmark : MSCI USA Growth "
        "ex-Top 20, reconstruit et renormalisé trimestriellement. Coûts de "
        "transaction inclus (10 bps par rotation)."
    )

    significant = "significatif" if hac["p_value"] < 0.05 else "non significatif"
    st.markdown(
        f"""
<div style="padding:16px; border:1px solid #334155; border-radius:12px; background:#111827; margin-bottom:14px;">
  <div style="font-weight:700; color:#E879F9; margin-bottom:6px;">Verdict quantitatif</div>
  <div>Sur la période, le portefeuille affiche un rendement annualisé de
  <b>{pct(stats['ann_return'])}</b> contre <b>{pct(bench['ann_return'])}</b> pour le
  benchmark ex-Top 20, soit un alpha annualisé de
  <b>{pct(hac['alpha'], signed=True)}</b> ({significant} au seuil de 5 %, t HAC =
  {hac['t_stat']:.2f}, p = {hac['p_value']:.3f}). Le risque est toutefois plus
  élevé : volatilité <b>{pct(stats['vol_ann'])}</b> contre
  <b>{pct(bench['vol_ann'])}</b>, bêta <b>{stats['beta']:.2f}</b> et drawdown
  maximal <b>{pct(stats['max_dd'])}</b> contre <b>{pct(bench['max_dd'])}</b>.
    Le Sharpe est de {stats['sharpe']:.2f} contre {bench['sharpe']:.2f}.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    columns = st.columns(6)
    columns[0].metric("CAGR", pct(stats["ann_return"]))
    columns[1].metric("Alpha moyen annualisé", pct(stats["alpha_ann"], signed=True))
    columns[2].metric("Tracking error", pct(stats["te_ann"]))
    columns[3].metric("Information ratio", f"{stats['info_ratio']:.2f}")
    columns[4].metric("Bêta", f"{stats['beta']:.2f}")
    columns[5].metric("Max drawdown", pct(stats["max_dd"]))

    st.markdown("#### Statistiques complètes")
    rows = [
        ("Rendement cumulé", stats["cum_return"], bench["cum_return"], "pct"),
        ("CAGR", stats["ann_return"], bench["ann_return"], "pct"),
        ("Volatilité annualisée", stats["vol_ann"], bench["vol_ann"], "pct"),
        ("Sharpe", stats["sharpe"], bench["sharpe"], "num"),
        ("Sortino", stats["sortino"], bench["sortino"], "num"),
        ("Max drawdown", stats["max_dd"], bench["max_dd"], "pct_signed"),
        ("Alpha moyen annualisé", stats["alpha_ann"], 0.0, "pct_signed"),
        ("Tracking error", stats["te_ann"], 0.0, "pct"),
        ("Information ratio", stats["info_ratio"], 0.0, "num"),
        ("Bêta", stats["beta"], 1.0, "num"),
        ("Active share vs ex-Top 20", stats["active_share"], 0.0, "pct"),
        ("Turnover moyen / rebalancement", stats["turnover_avg"], 0.0, "pct"),
        ("Nombre effectif moyen", stats["n_effective"], bench["n_effective"], "num"),
        ("Poids maximal final", stats["max_weight_last"], 0.0, "pct"),
        ("Top 10 final", stats["top10_weight_last"], 0.0, "pct"),
        ("Capture haussière mensuelle", up_capture, 1.0, "num"),
        ("Capture baissière mensuelle", down_capture, 1.0, "num"),
    ]
    table_rows = []
    for name, strategy, benchmark, kind in rows:
        if not np.isfinite(strategy):
            strategy_text = "—"
        elif kind == "num":
            strategy_text = f"{strategy:.2f}"
        elif kind == "pct_signed":
            strategy_text = pct(strategy, signed=True)
        else:
            strategy_text = pct(strategy)
        if not np.isfinite(benchmark):
            benchmark_text = "—"
        elif kind == "num":
            benchmark_text = f"{benchmark:.2f}"
        elif kind == "pct_signed":
            benchmark_text = pct(benchmark, signed=True)
        else:
            benchmark_text = pct(benchmark)
        table_rows.append({
            "Métrique": name,
            "Stratégie": strategy_text,
            "Benchmark": benchmark_text,
        })
    table = pd.DataFrame(table_rows)
    st.dataframe(table, hide_index=True, use_container_width=True)

    st.markdown("#### Significativité de l'alpha")
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"""
- Alpha annualisé : **{pct(hac['alpha'], signed=True)}**
- Erreur-type HAC : **{pct(hac['se'])}**
- t-statistique HAC (lag {hac['lag']}) : **{hac['t_stat']:.2f}**
- p-value bilatérale : **{hac['p_value']:.3f}**
- IC 95 % bootstrap par blocs : **[{pct(boot['low'], signed=True)} ; {pct(boot['high'], signed=True)}]**
- Probabilité bootstrap (alpha > 0) : **{boot['positive']:.1%}**
"""
        )
        st.caption(
            "Le bootstrap rééchantillonne des blocs de 21 jours pour préserver "
            "l'autocorrélation et les épisodes de marché. 1 000 tirages."
        )
    with right:
        fig = go.Figure(go.Histogram(
            x=boot["distribution"] * 100, nbinsx=40,
            marker_color=SCORE_ONLY_COLOR, opacity=0.75,
        ))
        fig.add_vline(x=0, line=dict(color="#8B949E", width=1, dash="dot"))
        fig.add_vline(x=hac["alpha"] * 100,
                      line=dict(color="#F0B429", width=2, dash="dash"))
        fig.update_xaxes(title_text="Alpha annualisé (%)")
        fig.update_yaxes(title_text="Tirages")
        st.plotly_chart(layout(fig, "Distribution bootstrap de l'alpha",
                               height=320), use_container_width=True,
                        key="analysis_bootstrap")

    section("Performance année par année")
    annual = data["annual"]
    years = [str(y) for y in annual.index]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=years, y=annual[SCORE_ONLY_LABEL] * 100, name=SCORE_ONLY_LABEL,
        marker_color=SCORE_ONLY_COLOR,
        text=[pct(v) for v in annual[SCORE_ONLY_LABEL]], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=years, y=annual["Benchmark ex-Top 20"] * 100, name=EX_TOP20_LABEL,
        marker_color=EX_TOP20_COLOR,
        text=[pct(v) for v in annual["Benchmark ex-Top 20"]],
        textposition="outside",
    ))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Rendement (%)")
    st.plotly_chart(layout(fig, "Rendements calendaires", height=380),
                    use_container_width=True, key="analysis_annual")

    quarterly = data["quarterly"]
    quarters = quarterly[[SCORE_ONLY_LABEL, "Benchmark ex-Top 20"]].copy()
    quarters["excess"] = quarters[SCORE_ONLY_LABEL] - quarters["Benchmark ex-Top 20"]
    periods = pd.PeriodIndex(quarters.index, freq="Q")
    periods_frame = pd.DataFrame({
        "year": [p.year for p in periods],
        "quarter": [p.quarter for p in periods],
        "strategy": quarters[SCORE_ONLY_LABEL].to_numpy(),
        "excess": quarters["excess"].to_numpy(),
    })
    left, right = st.columns(2)
    for column, value, title, colorscale in [
        (left, "strategy", "Rendements trimestriels de la stratégie", "RdYlGn"),
        (right, "excess", "Excès trimestriel vs benchmark ex-Top 20", "RdYlGn"),
    ]:
        pivot = periods_frame.pivot(index="year", columns="quarter", values=value)
        pivot = pivot.reindex(columns=[1, 2, 3, 4])
        with column:
            fig = go.Figure(go.Heatmap(
                z=pivot.values * 100, x=[f"T{q}" for q in pivot.columns],
                y=[str(y) for y in pivot.index], colorscale=colorscale, zmid=0,
                text=np.round(pivot.values * 100, 1), texttemplate="%{text}",
                textfont=dict(size=9), colorbar=dict(title="%"),
            ))
            st.plotly_chart(layout(fig, title, height=340),
                            use_container_width=True,
                            key=f"analysis_quarter_{value}")

    section("Risque et régularité")
    rolling = pd.DataFrame({
        "active": active.rolling(252).mean() * 252,
        "beta": ret.rolling(252).cov(bench_ret) / bench_ret.rolling(252).var(),
        "vol": ret.rolling(252).std() * np.sqrt(252),
        "bench_vol": bench_ret.rolling(252).std() * np.sqrt(252),
    }).dropna()
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Scatter(
            x=rolling.index, y=rolling["active"] * 100, name="Alpha glissant",
            line=dict(color=SCORE_ONLY_COLOR, width=2.0),
            fill="tozeroy", fillcolor="rgba(232,121,249,0.12)",
        ))
        fig.add_hline(y=0, line=dict(color="#8B949E", width=1, dash="dot"))
        fig.update_yaxes(title_text="Alpha annualisé glissant (%)")
        st.plotly_chart(layout(fig, "Alpha glissant 12 mois", height=340),
                        use_container_width=True, key="analysis_roll_alpha")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=rolling.index, y=rolling["beta"], name="Bêta glissant",
            line=dict(color=SCORE_ONLY_COLOR, width=2.0),
        ))
        fig.add_hline(y=1, line=dict(color="#8B949E", width=1, dash="dot"))
        fig.update_yaxes(title_text="Bêta")
        st.plotly_chart(layout(fig, "Bêta glissant 12 mois", height=340),
                        use_container_width=True, key="analysis_roll_beta")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling["vol"] * 100, name="Stratégie",
        line=dict(color=SCORE_ONLY_COLOR, width=1.8),
    ))
    fig.add_trace(go.Scatter(
        x=rolling.index, y=rolling["bench_vol"] * 100, name=EX_TOP20_LABEL,
        line=dict(color=EX_TOP20_COLOR, width=1.8, dash="dash"),
    ))
    fig.update_yaxes(title_text="Volatilité annualisée (%)")
    st.plotly_chart(layout(fig, "Volatilité glissante 12 mois", height=360),
                    use_container_width=True, key="analysis_roll_vol")

    section("Drawdowns")
    left, right = st.columns([1.25, 1])
    with left:
        nav = daily[f"nav_{SCORE_ONLY_KEY}"].dropna()
        bench_nav = daily["nav_benchmark_ex_top20"].dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=nav.index, y=drawdown(nav) * 100, name=SCORE_ONLY_LABEL,
            line=dict(color=SCORE_ONLY_COLOR, width=1.8),
            fill="tozeroy", fillcolor="rgba(232,121,249,0.10)",
        ))
        fig.add_trace(go.Scatter(
            x=bench_nav.index, y=drawdown(bench_nav) * 100, name=EX_TOP20_LABEL,
            line=dict(color=EX_TOP20_COLOR, width=1.5, dash="dash"),
        ))
        fig.update_yaxes(title_text="Drawdown (%)")
        st.plotly_chart(layout(fig, "Drawdowns comparés", height=360),
                        use_container_width=True, key="analysis_drawdown")
    with right:
        episodes = drawdown_episodes(nav)
        display = pd.DataFrame({
            "Creux": episodes["trough"].dt.strftime("%d/%m/%Y"),
            "Profondeur": episodes["depth"].map(lambda v: f"{v * 100:.1f} %"),
            "Durée (j)": episodes["days"],
            "Reprise": [
                date.strftime("%d/%m/%Y") if pd.notna(date) else "en cours"
                for date in episodes["recovery"]
            ],
        })
        st.markdown("**Cinq pires drawdowns**")
        st.dataframe(display, hide_index=True, use_container_width=True)

    section("Concentration, rotation et composition")
    concentration_frame = concentration(data["score_weights"])
    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=concentration_frame.index,
            y=concentration_frame["effective_n"], name="Nombre effectif",
            line=dict(color=SCORE_ONLY_COLOR, width=2.0),
        ))
        fig.add_hline(y=40, line=dict(color="#8B949E", width=1, dash="dot"))
        fig.update_yaxes(title_text="Titres équivalents")
        st.plotly_chart(layout(fig, "Nombre effectif de lignes", height=340),
                        use_container_width=True, key="analysis_effective_n")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=concentration_frame.index,
            y=concentration_frame["max_weight"] * 100, name="Poids max",
            line=dict(color="#F0B429", width=1.8),
        ))
        fig.add_trace(go.Scatter(
            x=concentration_frame.index,
            y=concentration_frame["top10"] * 100, name="Top 10",
            line=dict(color=SCORE_ONLY_COLOR, width=1.8),
        ))
        fig.update_yaxes(title_text="Poids (%)")
        st.plotly_chart(layout(fig, "Concentration", height=340),
                        use_container_width=True, key="analysis_concentration")

    turnover = data["score_turnover"]
    fig = go.Figure(go.Bar(
        x=turnover.index, y=turnover.values * 100, marker_color=SCORE_ONLY_COLOR,
        text=[f"{v * 100:.0f} %" for v in turnover.values],
        textposition="outside",
    ))
    fig.update_yaxes(title_text="Turnover (%)")
    st.plotly_chart(layout(fig, "Turnover à chaque rebalancement trimestriel",
                           height=340), use_container_width=True,
                    key="analysis_turnover")

    sector_active = (
        data["score_sectors"]
        .groupby("sector")["active"]
        .agg(["mean", "last"])
        .sort_values("mean")
    )
    left, right = st.columns([1, 1])
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=sector_active.index, x=sector_active["mean"] * 100,
            orientation="h", name="Moyenne", marker_color=SCORE_ONLY_COLOR,
        ))
        fig.add_trace(go.Bar(
            y=sector_active.index, x=sector_active["last"] * 100,
            orientation="h", name="Dernière date", marker_color="#F0B429",
        ))
        fig.update_layout(barmode="group")
        fig.update_xaxes(title_text="Écart actif moyen (%)")
        st.plotly_chart(layout(fig, "Biais sectoriels vs ex-Top 20", height=400),
                        use_container_width=True, key="analysis_sector_bar")
    with right:
        pivot = data["score_sectors"].pivot_table(
            index="date", columns="sector", values="active", aggfunc="sum"
        ).fillna(0.0) * 100
        fig = go.Figure(go.Heatmap(
            z=pivot.values, x=list(pivot.columns),
            y=[d.strftime("%d/%m/%Y") for d in pivot.index],
            colorscale="RdYlGn", zmid=0, colorbar=dict(title="Δ %"),
        ))
        fig.update_xaxes(tickangle=-35)
        st.plotly_chart(layout(fig, "Écarts sectoriels dans le temps", height=400),
                        use_container_width=True, key="analysis_sector_heat")

    section("Contribution des titres")
    overall = data["score_overall"].sort_values(ascending=False)
    selected = pd.concat([overall.head(10), overall.tail(5)]).sort_values()
    colors = ["#E74C3C" if value < 0 else "#2ECC71" for value in selected.values]
    fig = go.Figure(go.Bar(
        x=selected.values * 100, y=selected.index, orientation="h",
        marker_color=colors,
        text=[f"{v * 100:+.1f}" for v in selected.values], textposition="outside",
    ))
    fig.update_xaxes(title_text="Contribution cumulée (points de %)")
    st.plotly_chart(layout(fig, "Meilleurs et pires contributeurs", height=420),
                    use_container_width=True, key="analysis_contributors")

    with st.expander("Composition à la dernière date"):
        table = data["score_holdings"][
            ["ticker", "sector", "z", "score_rank", "benchmark_weight",
             "target_weight"]
        ].copy()
        table.columns = ["Titre", "Secteur", "Z-score", "Rang score",
                         "Poids benchmark", "Poids stratégie"]
        table["Z-score"] = table["Z-score"].map(lambda v: f"{v:+.2f}")
        table["Poids benchmark"] = table["Poids benchmark"].map(pct)
        table["Poids stratégie"] = table["Poids stratégie"].map(pct)
        st.dataframe(table, hide_index=True, use_container_width=True)

    section("Limites et lecture honnête")
    st.markdown(
        """
- **Un seul chemin historique.** Les résultats portent sur 2021-2025 ; ils ne
  constituent pas une preuve de robustesse hors échantillon.
- **Risque plus élevé.** Bêta supérieur à 1, volatilité et drawdown plus forts
  que le benchmark ex-Top 20 : la surperformance brute ne se transforme pas
  nécessairement en meilleur rendement ajusté du risque.
- **Alpha fragile.** La significativité dépend de la période, du modèle et de la
  correction HAC ; un p-value au seuil de 5 % ne garantit pas la persistance.
- **Coûts partiels.** Les 10 bps couvrent le turnover, mais ni le spread, ni
  l'impact de marché, ni les frais de gestion, ni la fiscalité.
- **Sharpe sans taux sans risque.** Le taux sans risque est fixé à 0 % ; le
  Sharpe et le Sortino sont donc des ratios bruts, non des excès sur le cash.
- **Benchmark reconstruit.** L'ex-Top 20 est obtenu en retirant les anchors puis
  en renormalisant les poids d'indice à chaque date ; il diffère d'un indice
  officiel et sa couverture peut être imparfaite.
- **Sélection conditionnelle.** Les 40 score picks dépendent du modèle
  d'accélération et de l'univers MSCI ; un changement de définition des anchors
  modifierait la composition et les résultats.
"""
    )


def methodology_tab(data: dict) -> None:
    summary = data["summary"]
    config = summary["config"]
    universe = summary["universe_summary"]

    st.subheader("Méthodologie & données")
    st.markdown(
        """
<div style="display:flex; gap:10px; flex-wrap:wrap; margin:8px 0 18px;">
  <div style="flex:1; min-width:180px; padding:14px; border:1px solid #334155; border-radius:12px; background:#111827;">
    <div style="color:#4C9AFF; font-weight:700;">1 · Anchors</div>
    <div>20 plus gros poids de l'indice, retenus quel que soit leur score.</div>
  </div>
  <div style="flex:1; min-width:180px; padding:14px; border:1px solid #334155; border-radius:12px; background:#111827;">
    <div style="color:#B388FF; font-weight:700;">2 · Score picks</div>
    <div>40 meilleurs scores d'accélération hors anchors, sans doublon.</div>
  </div>
  <div style="flex:1; min-width:180px; padding:14px; border:1px solid #334155; border-radius:12px; background:#111827;">
    <div style="color:#F0B429; font-weight:700;">3 · Rebase & tilt</div>
    <div>Poids d'indice rebasés à 100 %, puis tilt exponentiel du score Z.</div>
  </div>
  <div style="flex:1; min-width:180px; padding:14px; border:1px solid #334155; border-radius:12px; background:#111827;">
    <div style="color:#2ECC71; font-weight:700;">4 · Projection sous cap</div>
    <div>Bandes de 50 % / 4 pp et plafond individuel de 15 %, 12,5 % ou 10 %.</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
**Univers.** Union des constituants MSCI USA Growth : **{config['n_tickers']} titres**,
**{config['n_dates']} rebalancements trimestriels**, du
**{pd.Timestamp(config['start']):%d/%m/%Y}** au
**{pd.Timestamp(config['end']):%d/%m/%Y}**.

**Construction.** 20 anchors + 40 score picks = **{config['n_total_holdings']} lignes**.
Poids rebasés, tilt `w0 × exp(λ × clip(z, −{config['z_clip']:.0f}, +{config['z_clip']:.0f}))`
avec **λ = {config['lambda']:.2f}**, puis projection dans des bandes de
**{config['band_rel']:.0%}** du poids rebasé plafonnées à
**{config['band_abs_cap']:.0%}**.

**Caps.** Le plafond individuel est appliqué à **chaque rebalancement trimestriel**.
Entre deux rebalancements, les poids dérivent en buy & hold : le poids maximal
peut donc dépasser temporairement le cap. Aucun contrôle mensuel n'est appliqué.

**Coûts.** {config['tx_cost_bps']:.0f} bps par rotation.
"""
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Titres dans l'univers", universe["n_tickers"])
    c2.metric("Éligibles min.", universe["n_eligible_min"])
    c3.metric("Couverture min.", pct(universe["coverage_weight_min"], 2))
    c4.metric("Rebalancements", config["n_dates"])

    section("Couverture de l'univers")
    coverage = data["coverage"].sort_values("date")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=coverage["date"], y=coverage["n_constituents"],
                         name="Constituants", marker_color="#4C9AFF"))
    fig.add_trace(go.Scatter(x=coverage["date"], y=coverage["n_eligible"],
                             name="Éligibles", mode="lines+markers",
                             line=dict(color="#2ECC71", width=2)))
    fig.update_yaxes(title_text="Nombre de titres")
    st.plotly_chart(layout(fig, "Constituants et titres éligibles par rebalancement",
                           height=360), use_container_width=True)

    section("Paramètres des variantes")
    rows = []
    for name in VARIANT_NAMES:
        row = stats_row(data, name)
        rows.append({
            "Variante": TAB_LABELS[name],
            "Cap": pct(CAPS[name]) if CAPS[name] is not None else "Aucun",
            "Anchors": 20,
            "Score picks": 40,
            "CAGR": pct(row["ann_return"]),
            "TE": pct(row["te_ann"]),
            "Poids max final": pct(row["max_weight_last"]),
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.info(
        "Les métriques sont calculées contre le benchmark MSCI USA Growth. "
        "Aucune variante ex-NVDA ni aucun turnover ne sont affichés. "
        "Les performances passées ne préjugent pas des performances futures."
    )


def main() -> None:
    st.set_page_config(page_title="Top 20 + 40 — variantes de cap",
                       page_icon="📈", layout="wide")
    st.title("📈 MSCI USA Growth — Top 20 + 40")
    st.caption("Quatre variantes de cap, rebalancées trimestriellement. "
               "Benchmark unique : MSCI USA Growth.")

    data = load_data()
    labels_map = labels(data)
    config = data["summary"]["config"]

    with st.sidebar:
        st.header("⚙️ Paramètres")
        st.metric("Variantes", len(VARIANT_NAMES))
        st.metric("Période", f"{pd.Timestamp(config['start']):%Y}–"
                             f"{pd.Timestamp(config['end']):%Y}")
        st.metric("Rebalancements", config["n_dates"])
        st.divider()
        st.markdown("**Légende des caps**")
        for name, short, _, cap in VARIANT_ORDER:
            cap_txt = pct(cap) if cap is not None else "aucun"
            st.markdown(
                f"<span style='color:{COLORS[name]}'>■</span> "
                f"**{short}** — {cap_txt}",
                unsafe_allow_html=True,
            )
        st.divider()
        st.caption("Les poids dérivent en buy & hold entre deux rebalancements. "
                   "Aucun contrôle mensuel.")

    tabs = st.tabs(["Vue d'ensemble"] + [TAB_LABELS[n] for n in VARIANT_NAMES]
                   + ["Méthodologie", "Année par année", "40 score picks",
                      "Analyse 40 picks"])
    with tabs[0]:
        overview_tab(data)
    for index, name in enumerate(VARIANT_NAMES, start=1):
        with tabs[index]:
            variant_tab(data, name)
    with tabs[-4]:
        methodology_tab(data)
    with tabs[-3]:
        yearly_tab(data)
    with tabs[-2]:
        score_only_tab(data)
    with tabs[-1]:
        analysis_tab(data)


if __name__ == "__main__":
    main()
