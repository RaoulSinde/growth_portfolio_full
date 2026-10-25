import numpy as np
import pandas as pd

from src.config import (
    COMPONENT_PRICES_ADJ_FILE,
    COMPONENT_PRICES_FILE,
    COMPARISON_FILE,
    INDEX_PRICES_FILE_OUT,
    PERIOD_RETURNS_FILE,
    PROCESSED_DIR,
    RECONSTRUCTED_FILE,
    START,
    END,
    TRADING_DAYS,
    WEIGHTS_FILE,
)

EXCLUSIONS_FILE = PROCESSED_DIR / "index_exclusions_by_snapshot.csv"


def load_inputs():
    weights = pd.read_parquet(WEIGHTS_FILE)
    weights["date"] = pd.to_datetime(weights["date"]).dt.normalize()
    weights["weight"] = weights["weight"].astype(float)
    weights = weights[weights["ticker"].notna()].copy()
    weights["ticker"] = weights["ticker"].astype(str)

    official = pd.read_parquet(INDEX_PRICES_FILE_OUT)
    official["date"] = pd.to_datetime(official["date"]).dt.normalize()
    official = official.sort_values("date").reset_index(drop=True)

    close = pd.read_parquet(COMPONENT_PRICES_FILE)
    close.index = pd.to_datetime(close.index).normalize()
    close = close.sort_index()

    adj = pd.read_parquet(COMPONENT_PRICES_ADJ_FILE)
    adj.index = pd.to_datetime(adj.index).normalize()
    adj = adj.sort_index().reindex(columns=close.columns)

    return weights, official, close, adj


def validate_inputs(weights, official, close, adj):
    totals = weights.groupby("date")["weight"].sum()
    if not totals.between(1.0 - 1e-9, 1.0 + 1e-9).all():
        raise ValueError("Weight snapshots do not sum to 1")
    snapshots = pd.DatetimeIndex(sorted(weights["date"].unique()))
    if not snapshots.is_monotonic_increasing:
        raise ValueError("Snapshot dates are not strictly increasing")
    if not official["date"].is_monotonic_increasing or official["date"].duplicated().any():
        raise ValueError("Official index calendar is not strictly increasing")
    if snapshots.min() < official["date"].min() or snapshots.max() > official["date"].max():
        raise ValueError("Snapshot dates fall outside the official index window")
    if not list(close.index) == list(official["date"]):
        raise ValueError("Component calendar is not aligned on the official calendar")
    if not list(adj.index) == list(official["date"]):
        raise ValueError("Adjusted component calendar is not aligned on the official calendar")
    if weights["date"].max() > official["date"].max():
        raise ValueError("Lookahead: snapshot after the last official index date")
    return snapshots


def build_period_levels(weights, official, close, adj, snapshots):
    calendar = pd.DatetimeIndex(official["date"])
    n_dates = len(calendar)
    positions = {date: i for i, date in enumerate(calendar)}

    pivot = weights.pivot_table(
        index="date", columns="ticker", values="weight", aggfunc="sum", fill_value=0.0
    ).astype(float)
    pivot = pivot.sort_index()

    level_main = np.full(n_dates, np.nan)
    level_cash = np.full(n_dates, np.nan)
    level_adj = np.full(n_dates, np.nan)
    coverage_series = np.full(n_dates, np.nan)
    ntickers_series = np.full(n_dates, np.nan)

    level_main[0] = float(official["index_level"].iloc[0])
    level_cash[0] = level_main[0]
    level_adj[0] = level_main[0]

    periods = []
    exclusions = []
    consistency = []

    n_snapshots = len(snapshots)
    for k in range(n_snapshots - 1):
        start = snapshots[k]
        end = snapshots[k + 1]
        i0 = positions[start]
        i1 = positions[end]
        window_dates = calendar[i0 : i1 + 1]
        snap_weights = pivot.loc[start]
        active = snap_weights[snap_weights > 0]

        close_start = close.loc[start]
        close_end = close.loc[end]
        adj_start = adj.loc[start]
        adj_end = adj.loc[end]

        valid_main = active.index[
            active.index.isin(close.columns)
            & close_start.reindex(active.index).notna()
            & close_end.reindex(active.index).notna()
        ]
        valid_adj = active.index[
            active.index.isin(adj.columns)
            & adj_start.reindex(active.index).notna()
            & adj_end.reindex(active.index).notna()
        ]

        w_active = active.astype(float)
        w_main = w_active.reindex(valid_main).astype(float)
        covered_weight = float(w_main.sum())
        if covered_weight > 0:
            w_main = w_main / covered_weight

        w_adj = w_active.reindex(valid_adj).astype(float)
        covered_adj_weight = float(w_adj.sum())
        if covered_adj_weight > 0:
            w_adj = w_adj / covered_adj_weight

        if len(valid_main) and level_main[i0] > 0:
            p_start = close_start.reindex(valid_main).astype(float)
            p_end = close_end.reindex(valid_main).astype(float)
            shares = (w_main * level_main[i0]) / p_start
            p_window = close.loc[window_dates[1:], valid_main].ffill()
            p_window = p_window.fillna(p_start)
            values = p_window.mul(shares, axis=1).sum(axis=1)
            level_main[i0 + 1 : i1 + 1] = values.to_numpy()
            r_main = float((w_main * (p_end / p_start - 1.0)).sum())
            implied = level_main[i0] * (1.0 + r_main)
            consistency.append(abs(level_main[i1] - implied))
        else:
            r_main = 0.0
            level_main[i0 + 1 : i1 + 1] = level_main[i0]
            consistency.append(0.0)

        if len(valid_main) and level_cash[i0] > 0:
            p_start = close_start.reindex(valid_main).astype(float)
            shares = (w_active.reindex(valid_main).astype(float) * level_cash[i0]) / p_start
            cash = (1.0 - covered_weight) * level_cash[i0]
            p_window = close.loc[window_dates[1:], valid_main].ffill()
            p_window = p_window.fillna(p_start)
            values = p_window.mul(shares, axis=1).sum(axis=1) + cash
            level_cash[i0 + 1 : i1 + 1] = values.to_numpy()
            r_cash = float(
                (w_active.reindex(valid_main).astype(float) * (p_end / p_start - 1.0)).sum()
            )
            implied_cash = level_cash[i0] * (1.0 + r_cash)
        else:
            r_cash = 0.0
            level_cash[i0 + 1 : i1 + 1] = level_cash[i0]
            implied_cash = level_cash[i0]

        if len(valid_adj) and level_adj[i0] > 0:
            a_start = adj_start.reindex(valid_adj).astype(float)
            a_end = adj_end.reindex(valid_adj).astype(float)
            shares = (w_adj * level_adj[i0]) / a_start
            a_window = adj.loc[window_dates[1:], valid_adj].ffill()
            a_window = a_window.fillna(a_start)
            values = a_window.mul(shares, axis=1).sum(axis=1)
            level_adj[i0 + 1 : i1 + 1] = values.to_numpy()
            r_adj = float((w_adj * (a_end / a_start - 1.0)).sum())
        else:
            r_adj = 0.0
            level_adj[i0 + 1 : i1 + 1] = level_adj[i0]

        coverage_series[i0:i1] = covered_weight
        ntickers_series[i0:i1] = len(valid_main)

        excluded = active.index.difference(valid_main)
        exclusions.append(
            {
                "date": start,
                "next_date": end,
                "covered_weight": covered_weight,
                "excluded_weight": float(1.0 - covered_weight),
                "n_eligible": len(valid_main),
                "n_excluded": len(excluded),
                "excluded_tickers": ";".join(sorted(excluded.tolist())),
            }
        )

        official_return = float(official["index_level"].iloc[i1] / official["index_level"].iloc[i0] - 1.0)
        periods.append(
            {
                "start": start,
                "end": end,
                "recon_return": r_main,
                "official_return": official_return,
                "excess": r_main - official_return,
                "coverage": covered_weight,
                "n_tickers": len(valid_main),
                "recon_cash_return": r_cash,
                "recon_adj_return": r_adj,
                "uncovered_weight": 1.0 - covered_weight,
            }
        )

    last = snapshots[-1]
    i_last = positions[last]
    last_weights = pivot.loc[last]
    last_active = last_weights[last_weights > 0]
    last_valid = last_active.index[
        last_active.index.isin(close.columns) & close.loc[last].reindex(last_active.index).notna()
    ]
    coverage_series[i_last] = float(last_active.reindex(last_valid).sum())
    ntickers_series[i_last] = len(last_valid)
    exclusions.append(
        {
            "date": last,
            "next_date": pd.NaT,
            "covered_weight": coverage_series[i_last],
            "excluded_weight": float(1.0 - coverage_series[i_last]),
            "n_eligible": len(last_valid),
            "n_excluded": int(len(last_active) - len(last_valid)),
            "excluded_tickers": ";".join(sorted(last_active.index.difference(last_valid).tolist())),
        }
    )

    max_consistency = float(np.nanmax(consistency)) if consistency else 0.0
    if max_consistency > 1e-8:
        raise ValueError(f"Daily series and period returns diverge: {max_consistency:.2e}")

    reconstructed = pd.DataFrame(
        {
            "date": calendar,
            "index_level": official["index_level"].to_numpy(),
            "recon_level": level_main,
            "recon_cash_level": level_cash,
            "recon_level_adj": level_adj,
            "coverage": coverage_series,
            "n_tickers": ntickers_series,
        }
    )

    period_frame = pd.DataFrame(periods)
    exclusion_frame = pd.DataFrame(exclusions)
    return reconstructed, period_frame, exclusion_frame, max_consistency


def main():
    weights, official, close, adj = load_inputs()
    snapshots = validate_inputs(weights, official, close, adj)

    reconstructed, periods, exclusions, max_consistency = build_period_levels(
        weights, official, close, adj, snapshots
    )

    reconstructed.to_parquet(RECONSTRUCTED_FILE, index=False)
    periods.to_parquet(PERIOD_RETURNS_FILE, index=False)
    exclusions.to_csv(EXCLUSIONS_FILE, index=False)

    official_cum = reconstructed["index_level"].iloc[-1] / reconstructed["index_level"].iloc[0] - 1.0
    main_cum = reconstructed["recon_level"].iloc[-1] / reconstructed["recon_level"].iloc[0] - 1.0
    cash_cum = reconstructed["recon_cash_level"].iloc[-1] / reconstructed["recon_cash_level"].iloc[0] - 1.0
    adj_cum = reconstructed["recon_level_adj"].iloc[-1] / reconstructed["recon_level_adj"].iloc[0] - 1.0

    print("RECONSTRUCTION")
    print(f"Snapshots: {len(snapshots)} | periods: {len(periods)}")
    print(f"Dates: {len(reconstructed)} | {reconstructed['date'].min().date()} -> {reconstructed['date'].max().date()}")
    print(f"Coverage mean/min: {reconstructed['coverage'].mean():.4f} / {reconstructed['coverage'].min():.4f}")
    print(f"Held tickers mean/min/max: {reconstructed['n_tickers'].mean():.1f} / {reconstructed['n_tickers'].min():.0f} / {reconstructed['n_tickers'].max():.0f}")
    print(f"Max daily vs period return divergence: {max_consistency:.2e}")
    print(f"Cumulative official : {official_cum:.6f}")
    print(f"Cumulative MAIN     : {main_cum:.6f}")
    print(f"Cumulative CASH     : {cash_cum:.6f}")
    print(f"Cumulative ADJ      : {adj_cum:.6f}")
    print(f"Final gap MAIN      : {(main_cum - official_cum) * 10000:.1f} bps")
    print(f"Exclusions log      : {EXCLUSIONS_FILE} ({len(exclusions)} snapshots)")
    print(f"Reconstructed       : {RECONSTRUCTED_FILE}")
    print(f"Period returns      : {PERIOD_RETURNS_FILE}")


if __name__ == "__main__":
    main()
