"""Dashboard Streamlit independant du portefeuille two-pocket MSCI USA Growth.

Lancement :  streamlit run streamlit_two_pocket.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.two_pocket_backtest import (
    OUTPUT_DIR,
    QUALITY_CAP_LABEL,
    QUALITY_PURE_LABEL,
    build_portfolios,
    compute_results,
)
from src.two_pocket_data import build_universe_panel, load_market_prices

ROOT = Path(__file__).resolve().parent
EXPORTS = Path(OUTPUT_DIR)
TRADING_DAYS = 252

COLORS = {
    "Benchmark": "#58A6FF",
    QUALITY_PURE_LABEL: "#2ECC71",
    QUALITY_CAP_LABEL: "#F0B429",
}
LABELS = ["Benchmark", QUALITY_PURE_LABEL, QUALITY_CAP_LABEL]
VERSION_OPTIONS = ["Comparaison", QUALITY_PURE_LABEL, QUALITY_CAP_LABEL]
PLOT_TEMPLATE = "plotly_dark"
TRANSPARENT = "rgba(0,0,0,0)"

METRIC_LABELS = {
    "cum_return": "Rendement cumule",
    "ann_return": "Rendement annualise (CAGR)",
    "vol_ann": "Volatilite annualisee",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "calmar": "Calmar",
    "max_dd": "Max drawdown",
    "beta": "Beta",
    "alpha_ann": "Alpha annualise (CAPM)",
    "active_return_ann": "Rendement actif annualise",
    "te_ann": "Tracking error realisee",
    "info_ratio": "Information ratio",
    "up_capture": "Up capture (mensuel)",
    "down_capture": "Down capture (mensuel)",
    "active_share": "Active Share moyen",
    "active_share_latest": "Active Share dernier rebalancement",
    "turnover_avg": "Turnover moyen",
    "n_positions_latest": "Positions (dernier)",
    "n_positions_avg": "Positions (moyenne)",
    "mega_weight_latest": "Poche mega-caps (dernier)",
    "quant_weight_latest": "Poche quant (dernier)",
    "top10_weight": "Top 10 cumule",
    "n_effective": "Nombre effectif de titres",
    "sector_gap_max": "Ecart sectoriel max",
    "best_year": "Meilleure annee",
    "worst_year": "Pire annee",
    "positive_quarters": "Trimestres positifs",
    "skew": "Skewness",
    "kurtosis": "Kurtosis",
}

PERCENT_METRICS = {
    "cum_return",
    "ann_return",
    "vol_ann",
    "max_dd",
    "alpha_ann",
    "active_return_ann",
    "te_ann",
    "active_share",
    "active_share_latest",
    "turnover_avg",
    "mega_weight_latest",
    "quant_weight_latest",
    "top10_weight",
    "sector_gap_max",
    "best_year",
    "worst_year",
    "positive_quarters",
}
SIGNED_METRICS = {
    "cum_return",
    "max_dd",
    "alpha_ann",
    "active_return_ann",
    "sector_gap_max",
    "best_year",
    "worst_year",
}
INT_METRICS = {
    "n_positions_latest",
    "n_positions_avg",
}


def pct(value, digits: int = 2, signed: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number * 100:+.{digits}f} %" if signed else f"{number * 100:.{digits}f} %"


def num(value, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "-"
    if not np.isfinite(number):
        return "-"
    return f"{number:.{digits}f}"


def layout(fig: go.Figure, title: str | None = None, height: int = 420) -> go.Figure:
    fig.update_layout(
        template=PLOT_TEMPLATE,
        height=height,
        title=dict(text=title, x=0, xanchor="left") if title else None,
        margin=dict(l=10, r=10, t=(58 if title else 24), b=10),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.0,
            xanchor="left",
            x=0,
            bgcolor=TRANSPARENT,
        ),
        hovermode="x unified",
    )
    return fig


def drawdown(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1.0


def relative(nav: pd.Series, bench: pd.Series) -> pd.Series:
    common = nav.index.intersection(bench.index)
    return nav.reindex(common) / bench.reindex(common) * 100.0


@st.cache_data(show_spinner="Chargement du panel PIT...")
def get_panel() -> pd.DataFrame:
    path = EXPORTS / "universe_panel.parquet"
    if path.exists():
        panel = pd.read_parquet(path)
        panel["date"] = pd.to_datetime(panel["date"]).dt.normalize()
        if (
            "score_quality_pure" in panel.columns
            and "score_quality_garp" not in panel.columns
        ):
            return panel
    return build_universe_panel(force=True, save=True)


@st.cache_data(show_spinner="Chargement des prix et du benchmark...")
def get_market() -> dict:
    return load_market_prices()


@st.cache_data(show_spinner="Construction des portefeuilles...")
def get_portfolios() -> dict:
    return build_portfolios(get_panel(), get_market())


@st.cache_data(show_spinner="Backtest au niveau de cout demande...")
def get_results(tx_cost_bps: float) -> dict:
    return compute_results(
        get_panel(),
        get_market(),
        tx_cost_bps=float(tx_cost_bps),
        portfolios=get_portfolios(),
    )


def format_stats(stats: pd.DataFrame) -> pd.DataFrame:
    data = {}
    for strategy in stats.index:
        values = []
        for metric in stats.columns:
            value = stats.loc[strategy, metric]
            if metric in PERCENT_METRICS:
                values.append(pct(value, signed=metric in SIGNED_METRICS))
            elif metric in INT_METRICS:
                values.append("-" if pd.isna(value) else f"{int(round(float(value)))}")
            else:
                values.append(num(value))
        data[strategy] = values
    return pd.DataFrame(
        data,
        index=[METRIC_LABELS.get(metric, metric) for metric in stats.columns],
    )


def nav_chart(nav: pd.DataFrame, labels: list[str], height: int = 480) -> go.Figure:
    fig = go.Figure()
    for label in labels:
        fig.add_trace(
            go.Scatter(
                x=nav["date"],
                y=nav[label],
                name=label,
                line=dict(
                    color=COLORS[label],
                    width=2.7 if label != "Benchmark" else 2.4,
                    dash="dash" if label == "Benchmark" else "solid",
                ),
            )
        )
    fig.update_yaxes(title_text="Base 100")
    return layout(fig, "Evolution du portefeuille vs MSCI USA Growth", height)


def relative_chart(nav: pd.DataFrame, labels: list[str], height: int = 380) -> go.Figure:
    frame = nav.set_index("date")
    bench = frame["Benchmark"]
    fig = go.Figure()
    for label in labels:
        if label == "Benchmark":
            continue
        series = relative(frame[label], bench)
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=label,
                line=dict(color=COLORS[label], width=2.0),
            )
        )
    fig.add_hline(y=100, line=dict(color="#8B949E", width=1, dash="dot"))
    fig.update_yaxes(title_text="Base 100 vs benchmark")
    return layout(fig, "Performance relative", height)


def drawdown_chart(nav: pd.DataFrame, labels: list[str], height: int = 340) -> go.Figure:
    frame = nav.set_index("date")
    fig = go.Figure()
    for label in labels:
        series = drawdown(frame[label])
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series.values,
                name=label,
                line=dict(
                    color=COLORS[label],
                    width=2.0 if label != "Benchmark" else 1.8,
                    dash="dash" if label == "Benchmark" else "solid",
                ),
            )
        )
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%")
    return layout(fig, "Drawdowns", height)


def annual_chart(annual: pd.DataFrame, labels: list[str], height: int = 380) -> go.Figure:
    fig = go.Figure()
    for label in labels:
        fig.add_trace(
            go.Bar(
                x=annual.index,
                y=annual[label].values,
                name=label,
                marker_color=COLORS[label],
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return layout(fig, "Rendements annuels", height)


def quarterly_heatmap(quarterly: pd.DataFrame, label: str, height: int = 360) -> go.Figure:
    series = quarterly[label].dropna()
    rows = []
    for index, value in series.items():
        year = int(str(index)[:4])
        quarter = int(str(index)[-1])
        rows.append({"year": year, "quarter": quarter, "value": value})
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index="year", columns="quarter", values="value")
    text = [[pct(value, 1, signed=True) for value in row] for row in pivot.values]
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values * 100.0,
            x=[f"Q{quarter}" for quarter in pivot.columns],
            y=[str(year) for year in pivot.index],
            text=text,
            texttemplate="%{text}",
            colorscale="RdYlGn",
            zmid=0,
            colorbar=dict(title="%"),
        )
    )
    return layout(fig, f"Rendements trimestriels - {label}", height)


def top_bar(
    frame: pd.DataFrame,
    title: str,
    positive: bool,
    height: int = 380,
) -> go.Figure:
    data = frame.sort_values("active_weight", ascending=not positive).head(10)
    color = "#2ECC71" if positive else "#E74C3C"
    fig = go.Figure(
        go.Bar(
            x=data["active_weight"],
            y=data["ticker"],
            orientation="h",
            marker_color=color,
            text=[pct(value, 2, signed=True) for value in data["active_weight"]],
            textposition="outside",
        )
    )
    fig.update_xaxes(tickformat=".1%", title_text="Ecart actif")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return layout(fig, title, height)


def contribution_bar(
    frame: pd.DataFrame,
    value_col: str,
    name_col: str,
    title: str,
    positive: bool,
    height: int = 380,
) -> go.Figure:
    data = frame.sort_values(value_col, ascending=not positive).head(12)
    color = "#2ECC71" if positive else "#E74C3C"
    fig = go.Figure(
        go.Bar(
            x=data[value_col],
            y=data[name_col],
            orientation="h",
            marker_color=color,
            text=[pct(value, 2, signed=True) for value in data[value_col]],
            textposition="outside",
        )
    )
    fig.update_xaxes(tickformat=".1%", title_text="Contribution")
    fig.update_layout(yaxis=dict(autorange="reversed"))
    return layout(fig, title, height)


def rolling_chart(nav: pd.DataFrame, label: str, height: int = 380) -> go.Figure:
    frame = nav.set_index("date")
    returns = frame.pct_change().dropna()
    active = returns[label] - returns["Benchmark"]
    te = active.rolling(63).std() * np.sqrt(TRADING_DAYS)
    ir = (
        active.rolling(126).mean()
        / active.rolling(126).std()
        * np.sqrt(TRADING_DAYS)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=te.index,
            y=te.values,
            name="TE 63j annualisee",
            line=dict(color="#4C9AFF", width=2.0),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=ir.index,
            y=ir.values,
            name="IR 126j annualise",
            line=dict(color="#F0B429", width=2.0),
            yaxis="y2",
        )
    )
    fig.update_layout(
        yaxis=dict(title="Tracking error", tickformat=".1%"),
        yaxis2=dict(title="Information ratio", overlaying="y", side="right"),
    )
    return layout(fig, "Risque actif roulant", height)


def volatility_chart(nav: pd.DataFrame, label: str, height: int = 360) -> go.Figure:
    frame = nav.set_index("date")
    returns = frame.pct_change().dropna()
    fig = go.Figure()
    for window, color in ((21, "#F0B429"), (63, "#4C9AFF")):
        vol = returns[label].rolling(window).std() * np.sqrt(TRADING_DAYS)
        fig.add_trace(
            go.Scatter(
                x=vol.index,
                y=vol.values,
                name=f"{label} - vol {window}j",
                line=dict(color=color, width=2.0),
            )
        )
    bench_vol = returns["Benchmark"].rolling(63).std() * np.sqrt(TRADING_DAYS)
    fig.add_trace(
        go.Scatter(
            x=bench_vol.index,
            y=bench_vol.values,
            name="Benchmark - vol 63j",
            line=dict(color="#8B949E", width=1.6, dash="dot"),
        )
    )
    fig.update_yaxes(title_text="Volatilite annualisee", tickformat=".0%")
    return layout(fig, f"Volatilite roulante - {label}", height)


def mega_caps_chart(
    mega: pd.DataFrame,
    labels: list[str],
    height: int = 360,
) -> go.Figure:
    fig = go.Figure()
    benchmark = (
        mega[mega["version"] == labels[0]]
        .groupby("date")["company"]
        .nunique()
        .sort_index()
    )
    fig.add_trace(
        go.Scatter(
            x=benchmark.index,
            y=benchmark.values,
            name="Benchmark",
            line=dict(color=COLORS["Benchmark"], width=2.0, dash="dash"),
        )
    )
    for label in labels:
        data = (
            mega[(mega["version"] == label) & mega["held"].astype(bool)]
            .groupby("date")["company"]
            .nunique()
            .sort_index()
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                name=label,
                line=dict(color=COLORS[label], width=2.2),
                mode="lines+markers",
            )
        )
    fig.update_yaxes(title_text="Nombre de societes", rangemode="tozero")
    return layout(fig, "Mega-caps dynamiques detenues (> 3 % du benchmark)", height)


def mega_weight_chart(
    mega: pd.DataFrame,
    labels: list[str],
    height: int = 340,
) -> go.Figure:
    fig = go.Figure()
    for label in labels:
        data = (
            mega[mega["version"] == label]
            .groupby("date")["weight"]
            .sum()
            .sort_index()
        )
        fig.add_trace(
            go.Scatter(
                x=data.index,
                y=data.values,
                name=label,
                line=dict(color=COLORS[label], width=2.0),
            )
        )
    fig.update_yaxes(title_text="Poids mega-caps", tickformat=".0%")
    return layout(fig, "Poids de la poche mega-caps", height)


def company_view(
    frame: pd.DataFrame,
    weight_columns: list[str],
) -> pd.DataFrame:
    aggregations = {
        "tickers": ("ticker", lambda values: ", ".join(sorted(set(values)))),
        "sector": ("sector", "first"),
        **{column: (column, "sum") for column in weight_columns},
    }
    if "pocket" in frame.columns:
        aggregations["pocket"] = ("pocket", "first")
    if "score" in frame.columns:
        aggregations["score"] = ("score", "mean")
    grouped = frame.groupby(["date", "version", "company"], as_index=False).agg(
        **aggregations
    )
    grouped["ticker"] = grouped["company"]
    return grouped


def sector_chart(budget: pd.DataFrame, height: int = 420) -> go.Figure:
    frame = budget.sort_values("benchmark_weight", ascending=False)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=frame["sector"],
            y=frame["portfolio_weight"],
            name="Portefeuille",
            marker_color="#2ECC71",
        )
    )
    fig.add_trace(
        go.Bar(
            x=frame["sector"],
            y=frame["benchmark_weight"],
            name="Benchmark",
            marker_color="#58A6FF",
        )
    )
    fig.update_layout(barmode="group")
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(tickformat=".0%")
    return layout(fig, "Exposition sectorielle", height)


def pocket_donut(holdings: pd.DataFrame, height: int = 330) -> go.Figure:
    weights = holdings.groupby("pocket")["weight"].sum()
    fig = go.Figure(
        go.Pie(
            labels=weights.index,
            values=weights.values,
            hole=0.62,
            marker_colors=["#58A6FF", "#2ECC71"],
            textinfo="label+percent",
        )
    )
    return layout(fig, "Repartition mega-caps / quant", height)


def active_share_chart(validation: pd.DataFrame, labels: list[str], height: int = 330) -> go.Figure:
    fig = go.Figure()
    for label in labels:
        if label == "Benchmark":
            continue
        data = validation[validation["version"] == label].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["active_share"],
                name=label,
                line=dict(color=COLORS[label], width=2.0),
                mode="lines+markers",
            )
        )
    fig.update_yaxes(tickformat=".0%")
    return layout(fig, "Active Share par rebalancement", height)


def positions_chart(validation: pd.DataFrame, labels: list[str], height: int = 330) -> go.Figure:
    fig = go.Figure()
    for label in labels:
        if label == "Benchmark":
            continue
        data = validation[validation["version"] == label].sort_values("date")
        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["n_positions"],
                name=label,
                line=dict(color=COLORS[label], width=2.0),
                mode="lines+markers",
            )
        )
    fig.update_yaxes(title_text="Nombre de titres", rangemode="tozero")
    return layout(fig, "Evolution du nombre de titres", height)


def turnover_chart(turnover: pd.DataFrame, labels: list[str], height: int = 330) -> go.Figure:
    fig = go.Figure()
    for label in labels:
        if label == "Benchmark":
            continue
        data = turnover[turnover["version"] == label].sort_values("date")
        fig.add_trace(
            go.Bar(
                x=data["date"],
                y=data["turnover"],
                name=label,
                marker_color=COLORS[label],
            )
        )
    fig.update_layout(barmode="group")
    fig.update_yaxes(tickformat=".0%")
    return layout(fig, "Turnover par rebalancement", height)


def display_table(
    frame: pd.DataFrame,
    height: int | None = None,
    hide_index: bool = True,
) -> None:
    kwargs = {"use_container_width": True, "hide_index": hide_index}
    if height is not None:
        kwargs["height"] = height
    st.dataframe(frame, **kwargs)


def main() -> None:
    st.set_page_config(
        page_title="MSCI USA Growth - Two-Pocket",
        page_icon="📈",
        layout="wide",
    )

    panel = get_panel()
    dates = sorted(pd.to_datetime(panel["date"]).unique())

    with st.sidebar:
        st.header("Parametres")
        focus = st.radio("Version", VERSION_OPTIONS, index=0)
        detail_version = (
            st.selectbox(
                "Version detaillee", [QUALITY_PURE_LABEL, QUALITY_CAP_LABEL], index=0
            )
            if focus == "Comparaison"
            else focus
        )
        selected_date = st.selectbox(
            "Date de rebalancement",
            dates,
            index=len(dates) - 1,
            format_func=lambda value: pd.Timestamp(value).strftime("%d/%m/%Y"),
        )
        tx_cost = st.slider(
            "Couts de transaction (bps)",
            min_value=0,
            max_value=30,
            value=10,
            step=1,
        )
        st.caption("Backtest : 31/12/2020 -> 31/12/2025 | rebalancement trimestriel")
        if st.button("Recalculer les scores PIT"):
            with st.spinner("Recalcul du panel fondamental PIT..."):
                build_universe_panel(force=True, save=True)
            st.cache_data.clear()
            st.rerun()

    results = get_results(float(tx_cost))
    stats = results["stats"]
    nav = results["nav_daily"].copy()
    nav["date"] = pd.to_datetime(nav["date"])
    annual = results["annual_returns"]
    quarterly = results["quarterly_returns"]
    validation = results["validation"]
    holdings = results["holdings"]
    budget = results["sector_budget"]
    active = results["active_positions"]
    turnover = results["turnover"]
    sector_exposure = results["sector_exposure"]
    contributions = results["contributions_overall"]
    contributions_annual = results["contributions_annual"]
    sector_contributions = results["sector_contributions"]
    metadata = results["metadata"]
    mega_caps = results["mega_caps"]

    active_labels = (
        ["Benchmark", QUALITY_PURE_LABEL, QUALITY_CAP_LABEL]
        if focus == "Comparaison"
        else ["Benchmark", detail_version]
    )
    shown_versions = (
        [QUALITY_PURE_LABEL, QUALITY_CAP_LABEL]
        if focus == "Comparaison"
        else [detail_version]
    )

    st.title("MSCI USA Growth - Portefeuille Two-Pocket")
    st.caption(
        "Mega-caps dynamiques (poids societe > 3 %, Alphabet = GOOG + GOOGL) "
        "+ stock-picking Quant 40 sur score de qualite intra-sectoriel, "
        "neutralite sectorielle, deux variantes : quality pure et quality pure 3x."
    )

    latest_date = pd.to_datetime(validation["date"]).max()
    st.info(
        f"{metadata['n_rebalances']} rebalancements | "
        f"{metadata['start']} -> {metadata['end']} | "
        f"couts {tx_cost} bps | "
        f"seuil mega-cap {metadata['mega_weight_threshold']:.0%} | "
        f"dernier rebalancement : {latest_date.strftime('%d/%m/%Y')}"
    )

    kpi_columns = st.columns(len(active_labels))
    for column, label in zip(kpi_columns, active_labels):
        with column:
            st.markdown(f"#### {label}")
            row = stats.loc[label]
            st.metric("Rendement cumule", pct(row["cum_return"], signed=True))
            st.metric("CAGR", pct(row["ann_return"], signed=True))
            st.metric("Sharpe", num(row["sharpe"]))
            st.metric("Volatilite annualisee", pct(row["vol_ann"]))
            st.metric("Max drawdown", pct(row["max_dd"], signed=True))
            if label != "Benchmark":
                st.metric("Alpha annualise (CAPM)", pct(row["alpha_ann"], signed=True))
                st.metric("Active Share", pct(row["active_share_latest"]))

    tabs = st.tabs(
        [
            "Vue d'ensemble",
            "Performance",
            "Positionnement",
            "Mega-caps",
            "Secteurs & budgets",
            "Contributions",
            "Methodologie & validation",
        ]
    )

    with tabs[0]:
        left, right = st.columns([1.7, 1.0])
        with left:
            st.plotly_chart(nav_chart(nav, active_labels), use_container_width=True)
        with right:
            st.plotly_chart(relative_chart(nav, active_labels), use_container_width=True)
            st.plotly_chart(drawdown_chart(nav, active_labels), use_container_width=True)
        st.plotly_chart(annual_chart(annual, active_labels), use_container_width=True)
        st.markdown("#### Rendements annuels")
        display_table(
            annual.rename(index=str).map(lambda value: pct(value, 2, signed=True)),
            hide_index=False,
        )

    with tabs[1]:
        st.markdown(f"#### Indicateurs de performance - {detail_version}")
        display_table(format_stats(stats).rename_axis("Indicateur"), hide_index=False)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(rolling_chart(nav, detail_version), use_container_width=True)
        with right:
            st.plotly_chart(volatility_chart(nav, detail_version), use_container_width=True)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(quarterly_heatmap(quarterly, detail_version), use_container_width=True)
        with right:
            st.plotly_chart(quarterly_heatmap(quarterly, "Benchmark"), use_container_width=True)
        st.markdown("#### Rendements trimestriels")
        display_table(
            quarterly.map(lambda value: pct(value, 2, signed=True)).reset_index().rename(
                columns={"index": "Trimestre"}
            )
        )

    with tabs[2]:
        st.markdown(f"#### Positionnement au {pd.Timestamp(selected_date).strftime('%d/%m/%Y')} - {detail_version}")
        current_holdings_raw = holdings[
            (holdings["version"] == detail_version)
            & (pd.to_datetime(holdings["date"]) == pd.Timestamp(selected_date))
        ].copy()
        current_active_raw = active[
            (active["version"] == detail_version)
            & (pd.to_datetime(active["date"]) == pd.Timestamp(selected_date))
        ].copy()
        current_holdings = company_view(
            current_holdings_raw,
            ["benchmark_weight", "weight", "active_weight"],
        )
        current_active = company_view(
            current_active_raw,
            ["benchmark_weight", "portfolio_weight", "active_weight"],
        )

        over = current_active.sort_values("active_weight", ascending=False).head(10)
        under = current_active.sort_values("active_weight").head(10)
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                top_bar(over, "Top 10 surponderations", positive=True),
                use_container_width=True,
            )
            display_table(
                over[
                    [
                        "ticker",
                        "tickers",
                        "sector",
                        "benchmark_weight",
                        "portfolio_weight",
                        "active_weight",
                    ]
                ].assign(
                    benchmark_weight=lambda frame: frame["benchmark_weight"].map(pct),
                    portfolio_weight=lambda frame: frame["portfolio_weight"].map(pct),
                    active_weight=lambda frame: frame["active_weight"].map(lambda value: pct(value, signed=True)),
                )
            )
        with right:
            st.plotly_chart(
                top_bar(under, "Top 10 sous-ponderations", positive=False),
                use_container_width=True,
            )
            display_table(
                under[
                    [
                        "ticker",
                        "tickers",
                        "sector",
                        "benchmark_weight",
                        "portfolio_weight",
                        "active_weight",
                    ]
                ].assign(
                    benchmark_weight=lambda frame: frame["benchmark_weight"].map(pct),
                    portfolio_weight=lambda frame: frame["portfolio_weight"].map(pct),
                    active_weight=lambda frame: frame["active_weight"].map(lambda value: pct(value, signed=True)),
                )
            )

        left, right = st.columns([1.2, 1.0])
        with left:
            st.markdown("#### Holdings")
            display_table(
                current_holdings[
                    [
                        "company",
                        "tickers",
                        "sector",
                        "pocket",
                        "score",
                        "benchmark_weight",
                        "weight",
                        "active_weight",
                    ]
                ].assign(
                    score=lambda frame: frame["score"].map(num),
                    benchmark_weight=lambda frame: frame["benchmark_weight"].map(pct),
                    weight=lambda frame: frame["weight"].map(pct),
                    active_weight=lambda frame: frame["active_weight"].map(lambda value: pct(value, signed=True)),
                ),
                height=520,
            )
        with right:
            st.plotly_chart(pocket_donut(current_holdings), use_container_width=True)
            st.markdown(
                f"**Mega-caps** : {pct(current_holdings[current_holdings['pocket'] == 'Mega-cap']['weight'].sum())}  \n"
                f"**Quant** : {pct(current_holdings[current_holdings['pocket'] == 'Quant']['weight'].sum())}  \n"
                f"**Positions** : {len(current_holdings)}"
            )
        st.plotly_chart(
            positions_chart(validation, shown_versions),
            use_container_width=True,
        )

    with tabs[3]:
        st.markdown("#### Mega-caps dynamiques")
        st.caption(
            "Une societe est mega-cap des que son poids agrege dans le benchmark depasse 3 %. "
            "GOOG et GOOGL sont regroupes sous Alphabet."
        )
        st.plotly_chart(mega_caps_chart(mega_caps, shown_versions), use_container_width=True)
        st.plotly_chart(mega_weight_chart(mega_caps, shown_versions), use_container_width=True)

        current_mega = mega_caps[
            (mega_caps["version"] == detail_version)
            & (pd.to_datetime(mega_caps["date"]) == pd.Timestamp(selected_date))
        ].copy()
        current_mega = (
            current_mega.groupby(["company", "sector"], as_index=False)
            .agg(
                tickers=("ticker", lambda values: ", ".join(sorted(set(values)))),
                benchmark_weight=("benchmark_weight", "sum"),
                weight=("weight", "sum"),
                active_weight=("active_weight", "sum"),
                held=("held", "max"),
            )
            .sort_values("benchmark_weight", ascending=False)
        )
        display_table(
            current_mega.assign(
                benchmark_weight=lambda frame: frame["benchmark_weight"].map(pct),
                weight=lambda frame: frame["weight"].map(pct),
                active_weight=lambda frame: frame["active_weight"].map(
                    lambda value: pct(value, signed=True)
                ),
                held=lambda frame: frame["held"].map({True: "Oui", False: "Non"}),
            )
        )

    with tabs[4]:
        st.markdown(f"#### Budgets sectoriels - {detail_version}")
        st.caption(
            "Neutralite sectorielle stricte : le poids de chaque secteur est celui "
            "du benchmark, reparti entre mega-caps et poche Quant 40."
        )
        current_budget = budget[
            (budget["version"] == detail_version)
            & (pd.to_datetime(budget["date"]) == pd.Timestamp(selected_date))
        ].copy()
        budget_columns = [
            "sector",
            "benchmark_weight",
            "mega_weight",
            "quant_budget",
            "quant_weight",
            "portfolio_weight",
            "active_weight",
            "n_mega",
            "n_quant",
            "n_total",
        ]
        if "benchmark_weight_official" in current_budget.columns:
            budget_columns.insert(2, "benchmark_weight_official")
        budget_display = current_budget[budget_columns].assign(
            benchmark_weight=lambda frame: frame["benchmark_weight"].map(pct),
            mega_weight=lambda frame: frame["mega_weight"].map(pct),
            quant_budget=lambda frame: frame["quant_budget"].map(pct),
            quant_weight=lambda frame: frame["quant_weight"].map(pct),
            portfolio_weight=lambda frame: frame["portfolio_weight"].map(pct),
            active_weight=lambda frame: frame["active_weight"].map(
                lambda value: pct(value, signed=True)
            ),
        )
        if "benchmark_weight_official" in budget_display.columns:
            budget_display["benchmark_weight_official"] = current_budget[
                "benchmark_weight_official"
            ].map(pct)
        display_table(budget_display)
        st.plotly_chart(sector_chart(current_budget), use_container_width=True)

        st.markdown("#### Neutralite sectorielle dans le temps")
        pivot = (
            sector_exposure[sector_exposure["version"] == detail_version]
            .pivot(index="date", columns="sector", values="active_weight")
            .sort_index()
        )
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values * 100.0,
                x=list(pivot.columns),
                y=[pd.Timestamp(date).strftime("%Y-%m-%d") for date in pivot.index],
                colorscale="RdBu",
                zmid=0,
                colorbar=dict(title="pp"),
            )
        )
        st.plotly_chart(
            layout(fig, "Ecart de poids sectoriel (portefeuille - benchmark)", 420),
            use_container_width=True,
        )
        st.plotly_chart(active_share_chart(validation, shown_versions), use_container_width=True)

    with tabs[5]:
        st.markdown(f"#### Contributions cumulees - {detail_version}")
        company_map = (
            panel.sort_values("date")
            .drop_duplicates("ticker", keep="last")
            .set_index("ticker")["company"]
        )
        current_contribution = contributions[
            contributions["version"] == detail_version
        ].copy()
        current_contribution["company"] = current_contribution["ticker"].map(
            company_map
        ).fillna(current_contribution["ticker"])
        current_contribution = (
            current_contribution.groupby(
                ["version", "company", "sector"], as_index=False
            )["contribution"].sum()
        )
        left, right = st.columns(2)
        with left:
            st.plotly_chart(
                contribution_bar(
                    current_contribution,
                    "contribution",
                    "company",
                    "Top contributeurs",
                    positive=True,
                ),
                use_container_width=True,
            )
        with right:
            st.plotly_chart(
                contribution_bar(
                    current_contribution,
                    "contribution",
                    "company",
                    "Top detracteurs",
                    positive=False,
                ),
                use_container_width=True,
            )

        current_sector = sector_contributions[
            (sector_contributions["version"] == detail_version)
            & (sector_contributions["year"].isna())
        ].copy()
        st.plotly_chart(
            contribution_bar(
                current_sector,
                "contribution",
                "sector",
                "Contribution par secteur",
                positive=True,
                height=420,
            ),
            use_container_width=True,
        )

        annual_contribution = contributions_annual[
            contributions_annual["version"] == detail_version
        ].copy()
        annual_contribution["company"] = annual_contribution["ticker"].map(
            company_map
        ).fillna(annual_contribution["ticker"])
        annual_contribution = annual_contribution.groupby(
            ["year", "company"], as_index=False
        )["contribution"].sum()
        top_companies = (
            annual_contribution.groupby("company")["contribution"]
            .apply(lambda series: series.abs().sum())
            .nlargest(15)
            .index
        )
        pivot_contribution = (
            annual_contribution[annual_contribution["company"].isin(top_companies)]
            .pivot_table(
                index="year",
                columns="company",
                values="contribution",
                aggfunc="sum",
            )
            .fillna(0.0)
        )
        fig = go.Figure(
            go.Heatmap(
                z=pivot_contribution.values * 100.0,
                x=list(pivot_contribution.columns),
                y=[str(year) for year in pivot_contribution.index],
                colorscale="RdYlGn",
                zmid=0,
                colorbar=dict(title="%"),
            )
        )
        st.plotly_chart(
            layout(fig, "Contributions annuelles des 15 principaux titres", 420),
            use_container_width=True,
        )
        st.plotly_chart(turnover_chart(turnover, shown_versions), use_container_width=True)

    with tabs[6]:
        st.markdown(
            r"""
#### Methode

**Poche 1 - Mega-caps dynamiques.** Une societe est mega-cap si son poids agrege
dans le benchmark depasse **3 %** a la date consideree. `GOOG` et `GOOGL` sont
regroupes sous **Alphabet** pour la detection et l'affichage. Les mega-caps
conservent leur poids indiciel dans les deux versions.

**Poche 2 - Stock-picking quantitatif (Quant 40).** Pour chaque secteur, le score
de qualite est standardise en Z-score intra-sectoriel. Le nombre de titres quant
est reparti entre secteurs proportionnellement au budget residuel
$B_S^{quant} = b_S - \sum_{i \in Mega \cap S} w_i$, avec un minimum de 1 titre
par secteur actif. Les poids bruts sont ajustes par le signal puis normalises pour
saturer le budget sectoriel : la neutralite sectorielle est **stricte**.

**Deux variantes de score.**

- **Quality pure** : `55 % ROE + 45 % FCF yield`. Le score mesure la rentabilite
  du capital et la generation de cash, sans composante de valorisation.
- **Quality pure 3x** : meme score, avec un plafond de poids par ligne
  $w_i \le 3\,b_i$. Si ce plafond empeche de saturer le budget du secteur,
  des titres supplementaires sont selectionnes dans ce meme secteur. Lorsque
  l'univers note ne permet pas de saturer le budget sous 3x, le plafond est
  relaxe au minimum necessaire et le secteur est signale dans la validation.

Les deux variantes partent de la construction **Quant 40** : aucun optimiseur,
aucune contrainte de tracking error. La version 3x peut depasser 40 lignes
lorsque le plafond impose des titres supplementaires pour saturer les budgets
sectoriels. La TE affichee est une mesure realisee ex-post, pas une contrainte.

**Metriques relatives.** Les Up/Down Capture sont calculees sur les rendements
mensuels composes, puis rapportees aux mois ou le benchmark monte ou baisse.
L'alpha affiche est l'alpha CAPM/Jensen annualise ; le rendement actif moyen est
reporte separement.
            """
        )
        st.markdown("#### Validation")
        summary = validation.groupby("version").agg(
            sum_weight_ecart_max=("sum_weight", lambda series: (series - 1.0).abs().max()),
            secteur_ecart_max=("max_sector_gap", "max"),
            budget_residuel_max=("budget_residual", "max"),
            positions_min=("n_positions", "min"),
            positions_max=("n_positions", "max"),
            active_share_moyen=("active_share", "mean"),
            ratio_max=("max_weight_ratio_quant", "max"),
            cap_relaxed=("cap_relaxed", "any"),
            checks_ok=("ok", "all"),
        )
        display_table(
            summary.reset_index().assign(
                sum_weight_ecart_max=lambda frame: frame["sum_weight_ecart_max"].map(lambda value: f"{value:.2e}"),
                secteur_ecart_max=lambda frame: frame["secteur_ecart_max"].map(lambda value: f"{value:.2e}"),
                budget_residuel_max=lambda frame: frame["budget_residuel_max"].map(lambda value: f"{value:.2e}"),
                active_share_moyen=lambda frame: frame["active_share_moyen"].map(pct),
                ratio_max=lambda frame: frame["ratio_max"].map(num),
                cap_relaxed=lambda frame: frame["cap_relaxed"].map(
                    {True: "Oui", False: "Non"}
                ),
            )
        )
        st.markdown("#### Detail des controles par date")
        display_table(
            validation.sort_values(["version", "date"]).assign(
                date=lambda frame: pd.to_datetime(frame["date"]).dt.strftime("%Y-%m-%d"),
                sum_weight=lambda frame: frame["sum_weight"].map(lambda value: f"{value:.10f}"),
                max_sector_gap=lambda frame: frame["max_sector_gap"].map(lambda value: f"{value:.2e}"),
                budget_residual=lambda frame: frame["budget_residual"].map(lambda value: f"{value:.2e}"),
                active_share=lambda frame: frame["active_share"].map(pct),
            ),
            height=420,
        )
        st.markdown(
            f"""
#### Hypotheses

- Benchmark : **{metadata['benchmark']}**, niveau officiel.
- Fenetre : **{metadata['start']} -> {metadata['end']}**.
- Rebalancement : **trimestriel**, 21 dates.
- Couts de transaction : **{tx_cost:.0f} bps** (reglables dans la sidebar).
- Prix : composants non ajustes pour la NAV.
- Scores : qualite fondamentale PIT, uniquement avec des donnees publiees avant
  la date de rebalancement.
- Les titres sans prix ou sans score restent dans le benchmark mais ne sont pas
  eligibles a la poche quant.
- Les deux versions sont construites avec **Quant 40** ; la TE realisee est
  reportee a titre informatif.
            """
        )


if __name__ == "__main__":
    main()
