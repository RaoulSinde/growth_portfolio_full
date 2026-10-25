import re

import pandas as pd

from src.config import (
    CONSTITUENTS_FILE,
    INDEX_PRICES_FILE,
    INDEX_PRICES_FILE_OUT,
    INDEX_SECURITY_ID,
    TICKER_MAP_FILE,
    WEIGHTS_FILE,
)


def _parse_snapshot_date(label):
    return pd.to_datetime(label.replace(" Weight", ""), format="%d-%b-%Y")


def load_constituents():
    raw = pd.read_excel(CONSTITUENTS_FILE, header=5)
    weight_columns = [c for c in raw.columns if str(c).endswith(" Weight")]
    if not weight_columns:
        raise ValueError("No weight columns found in constituents file")

    dates = [_parse_snapshot_date(c) for c in weight_columns]
    rename = dict(zip(weight_columns, dates))

    frame = raw.rename(columns=rename)
    frame = frame.rename(columns={"Name": "name", "Symbol": "cusip"})
    frame["cusip"] = frame["cusip"].astype("string").str.strip()
    frame = frame[frame["cusip"].notna()]
    frame = frame[frame["cusip"] != INDEX_SECURITY_ID]
    frame = frame[frame["name"].notna()]

    long = frame.melt(
        id_vars=["name", "cusip"],
        value_vars=dates,
        var_name="date",
        value_name="weight",
    )
    long["weight"] = pd.to_numeric(
        long["weight"].replace("-", pd.NA).astype("string"), errors="coerce"
    ).fillna(0.0)
    long["date"] = pd.to_datetime(long["date"])

    long = (
        long.groupby(["date", "cusip", "name"], as_index=False)["weight"].sum()
        .sort_values(["date", "cusip"])
        .reset_index(drop=True)
    )

    totals = long.groupby("date")["weight"].sum()
    if not totals.between(99.999999, 100.000001).all():
        raise ValueError(f"Weights do not sum to 100 per snapshot:\n{totals}")

    long["weight"] = long["weight"] / 100.0
    totals = long.groupby("date")["weight"].sum()
    if not totals.between(0.999999, 1.000001).all():
        raise ValueError(f"Normalized weights do not sum to 1:\n{totals}")

    return long


def load_index_prices():
    raw = pd.read_excel(INDEX_PRICES_FILE, header=15)
    prices = raw[["Date", "Price"]].copy()
    prices.columns = ["date", "index_level"]
    prices["date"] = pd.to_datetime(prices["date"])
    prices["index_level"] = pd.to_numeric(prices["index_level"], errors="coerce")
    prices = prices.dropna().sort_values("date").drop_duplicates("date")
    return prices.reset_index(drop=True)


def attach_tickers(weights):
    if TICKER_MAP_FILE.exists():
        mapping = pd.read_parquet(TICKER_MAP_FILE)[["cusip", "ticker"]]
        weights = weights.merge(mapping, on="cusip", how="left")
    else:
        weights["ticker"] = pd.NA
    return weights[["date", "cusip", "name", "weight", "ticker"]]


def main():
    weights = load_constituents()
    prices = load_index_prices()
    weights = attach_tickers(weights)
    weights.to_parquet(WEIGHTS_FILE, index=False)
    prices.to_parquet(INDEX_PRICES_FILE_OUT, index=False)
    print(f"Snapshots: {weights['date'].nunique()}")
    print(f"Securities: {weights['cusip'].nunique()}")
    print(f"Weight rows: {len(weights)}")
    print(f"Index prices: {len(prices)} ({prices['date'].min().date()} -> {prices['date'].max().date()})")
    print(f"Start level: {prices['index_level'].iloc[0]:.6f}")


if __name__ == "__main__":
    main()
