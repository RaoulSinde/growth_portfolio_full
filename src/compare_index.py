import numpy as np
import pandas as pd

from src.config import (
    COMPARISON_FILE,
    PERIOD_RETURNS_FILE,
    RECONSTRUCTED_FILE,
    TRADING_DAYS,
)

SERIES = {
    "official": "index_level",
    "recon_main": "recon_level",
    "recon_cash": "recon_cash_level",
    "recon_adj": "recon_level_adj",
}


def max_drawdown(level):
    running_max = level.cummax()
    drawdown = level / running_max - 1.0
    trough = drawdown.idxmin()
    peak = level.loc[:trough].idxmax()
    return float(drawdown.min()), peak, trough


def level_metrics(level):
    level = level.dropna()
    daily = level.pct_change().dropna()
    cumulative = float(level.iloc[-1] / level.iloc[0] - 1.0)
    days = (level.index[-1] - level.index[0]).days
    cagr = float((level.iloc[-1] / level.iloc[0]) ** (365.25 / days) - 1.0) if days > 0 else np.nan
    vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ann_mean = float(daily.mean() * TRADING_DAYS)
    sharpe = float(ann_mean / vol) if vol > 0 else np.nan
    mdd, peak, trough = max_drawdown(level)
    return {
        "cumulative_return": cumulative,
        "cagr": cagr,
        "ann_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": mdd,
        "mdd_peak": peak,
        "mdd_trough": trough,
        "final_level": float(level.iloc[-1]),
        "ann_mean_return": ann_mean,
    }


def relative_metrics(recon, official):
    joined = pd.concat([recon.rename("recon"), official.rename("official")], axis=1).dropna()
    r = joined["recon"].pct_change().dropna()
    o = joined["official"].pct_change().dropna()
    diff = r - o
    var = float(o.var(ddof=1))
    beta = float(r.cov(o) / var) if var > 0 else np.nan
    corr = float(r.corr(o))
    te = float(diff.std(ddof=1) * np.sqrt(TRADING_DAYS))
    ann_active = float(diff.mean() * TRADING_DAYS)
    ir = float(ann_active / te) if te > 0 else np.nan
    active_cum = float((1.0 + diff).prod() - 1.0)
    return {
        "beta": beta,
        "correlation": corr,
        "tracking_error": te,
        "information_ratio": ir,
        "mean_annual_excess": ann_active,
        "active_cum_return": active_cum,
    }


def period_returns_from_level(level, periods):
    series = level.iloc[:, 0] if isinstance(level, pd.DataFrame) else level
    out = []
    for row in periods.itertuples():
        out.append(float(series.loc[row.end] / series.loc[row.start] - 1.0))
    return np.asarray(out)


def build_comparison(reconstructed, periods):
    data = reconstructed.set_index("date")
    records = []

    metrics_by_series = {}
    for name, column in SERIES.items():
        metrics = level_metrics(data[column])
        metrics_by_series[name] = metrics
        for metric in (
            "cumulative_return",
            "cagr",
            "ann_vol",
            "sharpe",
            "max_drawdown",
            "final_level",
        ):
            records.append(
                {"metric": metric, "series": name, "value": metrics[metric], "detail": ""}
            )
        records.append(
            {
                "metric": "mdd_peak",
                "series": name,
                "value": np.nan,
                "detail": str(metrics["mdd_peak"].date()),
            }
        )
        records.append(
            {
                "metric": "mdd_trough",
                "series": name,
                "value": np.nan,
                "detail": str(metrics["mdd_trough"].date()),
            }
        )

    official_cum = metrics_by_series["official"]["cumulative_return"]
    for name in ("recon_main", "recon_cash", "recon_adj"):
        cum = metrics_by_series[name]["cumulative_return"]
        records.append(
            {
                "metric": "gap_vs_official_bps",
                "series": name,
                "value": (cum - official_cum) * 10000.0,
                "detail": "",
            }
        )
        records.append(
            {
                "metric": "gap_vs_official_pct",
                "series": name,
                "value": (cum - official_cum) * 100.0,
                "detail": "",
            }
        )
        rel = relative_metrics(data[SERIES[name]], data["index_level"])
        for metric, value in rel.items():
            records.append({"metric": metric, "series": name, "value": value, "detail": ""})
        series_period = period_returns_from_level(data[[SERIES[name]]], periods)
        excess = series_period - periods["official_return"].to_numpy()
        records.append(
            {
                "metric": "pct_periods_outperform",
                "series": name,
                "value": float((excess > 0).mean()),
                "detail": "",
            }
        )
        records.append(
            {
                "metric": "worst_period_excess",
                "series": name,
                "value": float(np.nanmin(excess)),
                "detail": str(periods.loc[int(np.nanargmin(excess)), "start"].date()),
            }
        )
        records.append(
            {
                "metric": "best_period_excess",
                "series": name,
                "value": float(np.nanmax(excess)),
                "detail": str(periods.loc[int(np.nanargmax(excess)), "start"].date()),
            }
        )

    main_cum = metrics_by_series["recon_main"]["cumulative_return"]
    cash_cum = metrics_by_series["recon_cash"]["cumulative_return"]
    adj_cum = metrics_by_series["recon_adj"]["cumulative_return"]

    decomposition = {
        "gap_official_cash_bps": (cash_cum - official_cum) * 10000.0,
        "gap_official_main_bps": (main_cum - official_cum) * 10000.0,
        "coverage_effect_bps": (main_cum - cash_cum) * 10000.0,
        "dividend_effect_bps": (adj_cum - main_cum) * 10000.0,
        "residual_method_bps": (official_cum - main_cum) * -10000.0,
    }
    for metric, value in decomposition.items():
        records.append({"metric": metric, "series": "decomposition", "value": value, "detail": ""})

    comparison = pd.DataFrame(records)
    return comparison, metrics_by_series, decomposition


def main():
    reconstructed = pd.read_parquet(RECONSTRUCTED_FILE)
    reconstructed["date"] = pd.to_datetime(reconstructed["date"])
    periods = pd.read_parquet(PERIOD_RETURNS_FILE)
    for column in ("start", "end"):
        periods[column] = pd.to_datetime(periods[column])

    comparison, metrics, decomposition = build_comparison(reconstructed, periods)
    comparison.to_parquet(COMPARISON_FILE, index=False)

    print("INDEX COMPARISON")
    header = f"{'metric':<26}{'official':>14}{'MAIN':>14}{'CASH':>14}{'ADJ':>14}"
    print(header)
    print("-" * len(header))
    for metric in (
        "cumulative_return",
        "cagr",
        "ann_vol",
        "sharpe",
        "max_drawdown",
        "gap_vs_official_bps",
        "tracking_error",
        "beta",
        "correlation",
        "information_ratio",
        "pct_periods_outperform",
    ):
        row = []
        for series in ("official", "recon_main", "recon_cash", "recon_adj"):
            subset = comparison[(comparison["metric"] == metric) & (comparison["series"] == series)]
            row.append("" if subset.empty else f"{subset['value'].iloc[0]:.6f}")
        print(f"{metric:<26}{row[0]:>14}{row[1]:>14}{row[2]:>14}{row[3]:>14}")

    print("\nDECOMPOSITION (bps)")
    for key, value in decomposition.items():
        print(f"{key:<28}{value:>12.1f}")

    print(f"\nComparison written: {COMPARISON_FILE} ({len(comparison)} rows)")


if __name__ == "__main__":
    main()
