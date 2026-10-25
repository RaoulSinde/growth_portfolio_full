import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import (
    COMPARISON_FILE,
    PERIOD_RETURNS_FILE,
    PROCESSED_DIR,
    RECONSTRUCTED_FILE,
    REPORT_CSV,
    REPORT_PNG,
    TRADING_DAYS,
)

EXCLUSIONS_FILE = PROCESSED_DIR / "index_exclusions_by_snapshot.csv"
SERIES = ("official", "recon_main", "recon_cash", "recon_adj")


def load_data():
    reconstructed = pd.read_parquet(RECONSTRUCTED_FILE)
    reconstructed["date"] = pd.to_datetime(reconstructed["date"])
    periods = pd.read_parquet(PERIOD_RETURNS_FILE)
    for column in ("start", "end"):
        periods[column] = pd.to_datetime(periods[column])
    comparison = pd.read_parquet(COMPARISON_FILE)
    exclusions = pd.read_csv(EXCLUSIONS_FILE, parse_dates=["date", "next_date"])
    return reconstructed, periods, comparison, exclusions


def metrics_section(comparison):
    rows = []
    metrics = [m for m in comparison["metric"].unique() if m not in ("decomposition",)]
    for metric in metrics:
        if metric in ("mdd_peak", "mdd_trough"):
            details = []
            for series in SERIES:
                subset = comparison[
                    (comparison["metric"] == metric) & (comparison["series"] == series)
                ]
                if not subset.empty:
                    details.append(f"{series}={subset['detail'].iloc[0]}")
            rows.append(
                {"section": "METRIQUES", "metric": metric, "detail": "; ".join(details)}
            )
            continue
        row = {"section": "METRIQUES", "metric": metric}
        has_value = False
        for series in SERIES:
            subset = comparison[(comparison["metric"] == metric) & (comparison["series"] == series)]
            if subset.empty:
                continue
            row[series] = subset["value"].iloc[0]
            has_value = True
        if has_value:
            rows.append(row)
    return pd.DataFrame(rows)


def periods_section(periods, exclusions):
    data = periods.merge(
        exclusions[["date", "n_excluded", "excluded_tickers"]],
        left_on="start",
        right_on="date",
        how="left",
    ).drop(columns=["date"])
    output = pd.DataFrame(
        {
            "section": "PERIODES",
            "metric": "period",
            "start": data["start"],
            "end": data["end"],
            "official": data["official_return"],
            "recon_main": data["recon_return"],
            "recon_cash": data["recon_cash_return"],
            "recon_adj": data["recon_adj_return"],
            "excess": data["excess"],
            "coverage": data["coverage"],
            "n_tickers": data["n_tickers"],
            "n_excluded": data["n_excluded"],
            "excluded_tickers": data["excluded_tickers"],
        }
    )
    uncovered = 1.0 - output["coverage"]
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = (output["official"] - output["coverage"] * output["recon_main"]) / uncovered
    output["implied_missing_return"] = np.where(uncovered > 1e-8, implied, np.nan)
    output["diagnostic"] = np.where(
        output["coverage"] < 0.985,
        "couverture faible / titres sans prix",
        "couverture elevee : methode ou incoherence composite/indice",
    )
    return output


def worst_section(period_rows, n=5):
    worst = period_rows.sort_values("excess", ascending=True).head(n).copy()
    worst["section"] = "PIRE_PERIODES"
    worst["metric"] = "worst_excess"
    return worst


def decomposition_section(comparison):
    subset = comparison[comparison["series"] == "decomposition"].copy()
    subset["section"] = "DECOMPOSITION"
    subset["metric"] = subset["metric"]
    return subset[["section", "metric", "value", "detail"]]


def build_csv(reconstructed, periods, comparison, exclusions):
    metrics = metrics_section(comparison)
    period_rows = periods_section(periods, exclusions)
    worst = worst_section(period_rows, 5)
    decomposition = decomposition_section(comparison)
    combined = pd.concat(
        [metrics, period_rows, worst, decomposition], ignore_index=True, sort=False
    )
    columns = [
        "section",
        "metric",
        "start",
        "end",
        "official",
        "recon_main",
        "recon_cash",
        "recon_adj",
        "value",
        "excess",
        "coverage",
        "implied_missing_return",
        "n_tickers",
        "n_excluded",
        "excluded_tickers",
        "detail",
        "diagnostic",
    ]
    combined = combined.reindex(columns=columns)
    numeric = [
        "official",
        "recon_main",
        "recon_cash",
        "recon_adj",
        "value",
        "excess",
        "coverage",
        "implied_missing_return",
    ]
    for column in numeric:
        combined[column] = pd.to_numeric(combined[column], errors="coerce").round(8)
    combined.to_csv(REPORT_CSV, index=False)
    return combined, period_rows, worst


def rolling_tracking_error(recon, official, window=63):
    diff = recon.pct_change() - official.pct_change()
    return diff.rolling(window).std(ddof=1) * np.sqrt(TRADING_DAYS)


def build_chart(reconstructed, periods):
    data = reconstructed.set_index("date").sort_index()
    fig, axes = plt.subplots(3, 2, figsize=(15, 13))
    fig.suptitle("MSCI USA Growth - indice officiel vs reconstruction composants", fontsize=15)

    ax = axes[0, 0]
    base = data["index_level"].iloc[0]
    for column, label in (
        ("index_level", "Officiel"),
        ("recon_level", "Reconstruit MAIN"),
        ("recon_cash_level", "Reconstruit CASH"),
        ("recon_level_adj", "Reconstruit ADJ (dividendes)"),
    ):
        ax.plot(data.index, data[column] / base * 100.0, label=label, linewidth=1.4)
    ax.set_title("Niveaux rebases a 100")
    ax.set_ylabel("Indice base 100")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    for column, label in (
        ("recon_level", "MAIN"),
        ("recon_cash_level", "CASH"),
        ("recon_level_adj", "ADJ"),
    ):
        active = (data[column] / data["index_level"] - 1.0) * 100.0
        ax.plot(data.index, active, label=f"Actif cumule {label}", linewidth=1.4)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Rendement actif cumule vs officiel")
    ax.set_ylabel("%")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for column, label in (
        ("recon_level", "MAIN"),
        ("recon_cash_level", "CASH"),
        ("recon_level_adj", "ADJ"),
    ):
        ratio = data[column] / data["index_level"]
        ax.plot(data.index, ratio, label=label, linewidth=1.4)
    ax.axhline(1.0, color="black", linewidth=0.8)
    ax.set_title("Ratio reconstruit / officiel")
    ax.set_ylabel("Ratio")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    for column, label in (
        ("recon_level", "MAIN"),
        ("recon_cash_level", "CASH"),
        ("recon_level_adj", "ADJ"),
    ):
        te = rolling_tracking_error(data[column], data["index_level"]) * 100.0
        ax.plot(data.index, te, label=label, linewidth=1.3)
    ax.set_title("Tracking error glissante 63 jours (annualisee)")
    ax.set_ylabel("%")
    ax.legend(loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 0]
    ax.step(data.index, data["coverage"] * 100.0, where="post", color="tab:green", linewidth=1.3)
    ax.set_title("Couverture ponderee des composants")
    ax.set_ylabel("% du poids")
    ax.set_ylim(min(95.0, data["coverage"].min() * 100.0 - 0.5), 100.3)
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    colors = np.where(periods["excess"] >= 0, "tab:green", "tab:red")
    ax.bar(periods["start"], periods["excess"] * 100.0, width=20, color=colors)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("Ecart de rendement par periode (MAIN - officiel)")
    ax.set_ylabel("Ecart (%)")
    ax.grid(True, alpha=0.3)

    for ax in axes.flat:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        for label in ax.get_xticklabels():
            label.set_rotation(30)
            label.set_horizontalalignment("right")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(REPORT_PNG, dpi=150)
    plt.close(fig)


def main():
    reconstructed, periods, comparison, exclusions = load_data()
    combined, period_rows, worst = build_csv(reconstructed, periods, comparison, exclusions)
    build_chart(reconstructed, periods)

    print("REPORT")
    print(f"CSV: {REPORT_CSV} ({len(combined)} rows)")
    print(f"PNG: {REPORT_PNG}")
    print("\n5 PIRE PERIODES (MAIN - officiel)")
    for row in worst.itertuples():
        print(
            f"{row.start.date()} -> {row.end.date()} | "
            f"MAIN {row.recon_main * 100:+.2f}% | officiel {row.official * 100:+.2f}% | "
            f"ecart {row.excess * 100:+.2f}% | couverture {row.coverage * 100:.2f}% | "
            f"rendement implicite des exclus {row.implied_missing_return * 100:+.2f}% | "
            f"{row.n_excluded} exclus"
        )
        print(f"    exclus: {row.excluded_tickers}")


if __name__ == "__main__":
    main()
