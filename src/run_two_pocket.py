"""CLI de generation du portefeuille two-pocket MSCI USA Growth."""
from __future__ import annotations

import argparse

import pandas as pd

from .two_pocket_backtest import DEFAULT_MOMENTUM_WEIGHT, OUTPUT_DIR, run_two_pocket_backtest


def _print_summary(results: dict) -> None:
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)

    print("\n=== Portefeuille two-pocket MSCI USA Growth ===")
    meta = results["metadata"]
    print(f"Fenetre            : {meta['start']} -> {meta['end']}")
    print(f"Rebalancements     : {meta['n_rebalances']} dates trimestrielles")
    print(f"Couts de transaction: {meta['tx_cost_bps']:.1f} bps")
    print(
        "Score composite    : "
        f"Quality {meta.get('quality_weight', 1.0 - DEFAULT_MOMENTUM_WEIGHT):.0%} / "
        f"Momentum {meta.get('momentum_weight', DEFAULT_MOMENTUM_WEIGHT):.0%}"
    )
    print(
        "Mega-caps          : dynamiques, poids societe > "
        f"{meta['mega_weight_threshold']:.0%} "
        f"(Alphabet = {' + '.join(meta['alphabet_tickers'])})"
    )
    print(
        "Scores             : "
        + " | ".join(
            f"{label} (Quality {data.get('quality_weight', 1.0 - DEFAULT_MOMENTUM_WEIGHT):.0%}"
            f" / Momentum {data.get('momentum_weight', DEFAULT_MOMENTUM_WEIGHT):.0%}"
            + (
                f", cap x{data['max_weight_multiple']:.1f}"
                if data.get("max_weight_multiple") is not None
                else ""
            )
            + ")"
            for label, data in meta["versions"].items()
        )
    )
    print(f"Derniere liste     : {', '.join(meta['mega_caps_latest'])}")

    stats = results["stats"]
    display = stats[
        [
            "cum_return",
            "ann_return",
            "vol_ann",
            "sharpe",
            "max_dd",
            "alpha_ann",
            "te_ann",
            "info_ratio",
            "active_share",
            "turnover_avg",
            "n_positions_latest",
        ]
    ].copy()
    print("\n--- Performance et structure ---")
    print(display.to_string(float_format=lambda x: f"{x: .4f}"))

    validation = results["validation"]
    print("\n--- Validation (toutes dates) ---")
    summary = validation.groupby("version").agg(
        sum_weight_max=("sum_weight", lambda s: (s - 1.0).abs().max()),
        sector_gap_max=("max_sector_gap", "max"),
        budget_residual_max=("budget_residual", "max"),
        positions_min=("n_positions", "min"),
        positions_max=("n_positions", "max"),
        active_share_mean=("active_share", "mean"),
        cap_relaxed=("cap_relaxed", "any"),
        checks_ok=("ok", "all"),
    )
    print(summary.to_string(float_format=lambda x: f"{x:.10f}"))

    print("\n--- Dernier rebalancement ---")
    latest = results["holdings"]
    latest_date = pd.to_datetime(validation["date"]).max()
    for version, group in latest[pd.to_datetime(latest["date"]) == latest_date].groupby(
        "version"
    ):
        mega = group[group["pocket"] == "Mega-cap"]
        quant = group[group["pocket"] == "Quant"]
        print(
            f"{version}: {len(group)} positions | "
            f"{len(mega)} mega-caps ({mega['weight'].sum():.2%}) | "
            f"{len(quant)} quant ({quant['weight'].sum():.2%})"
        )
    print(f"\nExports ecrits dans: {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest Quant 40 MSCI USA Growth (Quality-Momentum / Quality-Momentum 3x)."
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Couts de transaction en points de base (defaut: 10).",
    )
    parser.add_argument(
        "--momentum-weight",
        type=float,
        default=DEFAULT_MOMENTUM_WEIGHT,
        help="Part du momentum dans le score composite, entre 0 et 1 (defaut: 0.25).",
    )
    parser.add_argument(
        "--force-panel",
        action="store_true",
        help="Recalcule le panel de scores PIT au lieu d'utiliser le cache.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="N'ecrit pas les exports dans reports/two_pocket.",
    )
    args = parser.parse_args()

    results = run_two_pocket_backtest(
        tx_cost_bps=float(args.cost_bps),
        force_panel=bool(args.force_panel),
        save=not bool(args.no_save),
        momentum_weight=float(args.momentum_weight),
    )
    _print_summary(results)


if __name__ == "__main__":
    main()
