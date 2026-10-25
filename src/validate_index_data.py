import pandas as pd

from src.config import (
    CACHE_DIR,
    COMPONENT_PRICES_FILE,
    INDEX_PRICES_FILE_OUT,
    TICKER_MAP_FILE,
    WEIGHTS_FILE,
)

UNRESOLVED_FILE = CACHE_DIR / "mapping_unresolved.csv"
COVERAGE_FILE = CACHE_DIR / "coverage_by_snapshot.csv"
NO_PRICE_FILE = CACHE_DIR / "tickers_without_prices.csv"


def check_weights(weights):
    totals = weights.groupby("date")["weight"].sum()
    max_deviation = (totals - 1.0).abs().max()
    if max_deviation > 1e-9:
        raise ValueError(f"Weights do not sum to 1: max deviation {max_deviation}")
    if not weights["date"].is_monotonic_increasing:
        raise ValueError("Weight dates are not sorted")
    duplicates = weights.duplicated(["date", "cusip"]).sum()
    if duplicates:
        raise ValueError(f"Duplicate date/cusip rows: {duplicates}")
    return totals, max_deviation


def build_coverage(weights, prices):
    dates = sorted(weights["date"].unique())
    available_tickers = set(prices.columns)
    rows = []
    for index, date in enumerate(dates):
        next_date = dates[index + 1] if index + 1 < len(dates) else pd.NaT
        bound_dates = [date] if pd.isna(next_date) else [date, next_date]
        snapshot = weights[weights["date"] == date]
        window = prices.reindex(bound_dates)
        priced = set(window.columns[window.notna().all(axis=0)])
        covered = snapshot[snapshot["ticker"].isin(priced)]
        missing = snapshot[~snapshot["ticker"].isin(priced)]
        covered_weight = float(covered["weight"].sum())
        rows.append(
            {
                "date": date,
                "next_date": next_date,
                "n_constituents": int((snapshot["weight"] > 0).sum()),
                "n_covered": int((covered["weight"] > 0).sum()),
                "covered_weight": covered_weight,
                "uncovered_weight": float(1.0 - covered_weight),
                "coverage_ratio": covered_weight,
                "missing_tickers": ";".join(
                    sorted(missing.loc[missing["weight"] > 0, "ticker"].dropna().astype(str))
                ),
            }
        )
    coverage = pd.DataFrame(rows)
    coverage["missing_from_matrix"] = coverage["missing_tickers"].apply(
        lambda value: sum(
            1 for ticker in str(value).split(";") if ticker and ticker not in available_tickers
        )
    )
    return coverage


def main():
    weights = pd.read_parquet(WEIGHTS_FILE)
    prices = pd.read_parquet(COMPONENT_PRICES_FILE)
    mapping = pd.read_parquet(TICKER_MAP_FILE)
    index_prices = pd.read_parquet(INDEX_PRICES_FILE_OUT)

    totals, max_deviation = check_weights(weights)
    coverage = build_coverage(weights, prices)
    coverage.to_csv(COVERAGE_FILE, index=False)

    unresolved = mapping[mapping["ticker"].isna()].copy()
    unresolved.to_csv(UNRESOLVED_FILE, index=False)
    no_price = mapping[~mapping["ticker"].isin(prices.columns)].copy()
    no_price.to_csv(NO_PRICE_FILE, index=False)

    print("MAPPING")
    print(f"Resolved: {mapping['ticker'].notna().sum()} / {len(mapping)}")
    print(mapping.groupby("source").size().to_string())
    print(f"Unresolved: {len(unresolved)}")
    if len(unresolved):
        print(unresolved[["cusip", "name"]].to_string(index=False))

    print("\nPRICES")
    print(f"Component price matrix: {prices.shape[0]} dates x {prices.shape[1]} tickers")
    print(f"Mapped tickers without price column: {len(no_price)}")
    if len(no_price):
        print(no_price[["cusip", "name", "ticker"]].to_string(index=False))

    print("\nWEIGHTS")
    print(f"Snapshots: {len(totals)}")
    print(f"Sum weight min/max: {totals.min():.9f} / {totals.max():.9f}")
    print(f"Max deviation from 1: {max_deviation:.2e}")

    print("\nCOVERAGE")
    print(f"Covered weight min/max: {coverage['covered_weight'].min():.4f} / {coverage['covered_weight'].max():.4f}")
    print(f"Mean covered weight: {coverage['covered_weight'].mean():.4f}")
    worst = coverage.sort_values("covered_weight").head(10)
    print("Worst snapshots:")
    print(
        worst[["date", "next_date", "covered_weight", "n_covered", "n_constituents"]]
        .to_string(index=False)
    )

    print("\nCHECKS")
    if list(prices.index) != list(index_prices["date"]):
        raise ValueError("Component price index is not aligned with the official index calendar")
    if prices.index.has_duplicates:
        raise ValueError("Duplicate dates in component price matrix")
    if not weights["date"].is_monotonic_increasing:
        raise ValueError("Weight dates are not monotonic")
    if weights["date"].max() > index_prices["date"].max():
        raise ValueError("Weight snapshot after index price range")
    if coverage["coverage_ratio"].max() > 1.0 + 1e-9:
        raise ValueError("Coverage ratio above 1")
    print("Weights sum to 1: OK")
    print("Component calendar aligned to index: OK")
    print("No future-dated snapshot: OK")
    print("Coverage bounded by [0, 1]: OK")


if __name__ == "__main__":
    main()
