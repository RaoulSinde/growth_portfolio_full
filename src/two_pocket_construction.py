"""Construction deterministe d'un portefeuille MSCI USA Growth en deux poches.

Poche 1 : mega-caps ancrees a leur poids indiciel.
Poche 2 : selection quantitative intra-sectorielle, neutre sectoriellement.

Le coeur du module est pur : il ne depend que du DataFrame d'entree et ne fait
aucun acces disque. Les validations sont executees a chaque construction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np
import pandas as pd

__all__ = [
    "ALPHABET_TICKERS",
    "MEGA_WEIGHT_THRESHOLD",
    "PortfolioResult",
    "build_two_pocket_portfolio",
    "validate_two_pocket",
    "allocate_sector_quotas",
    "company_name",
]

ALPHABET_TICKERS = ("GOOG", "GOOGL")
MEGA_WEIGHT_THRESHOLD = 0.03

_REQUIRED_COLUMNS = (
    "ticker",
    "sector",
    "benchmark_weight",
)

_EPS = 1e-12


def company_name(ticker: str) -> str:
    """Nom societe utilise pour agreger les classes d'actions."""
    ticker = str(ticker).strip().upper()
    return "Alphabet" if ticker in ALPHABET_TICKERS else ticker


@dataclass
class PortfolioResult:
    """Resultat complet d'une construction two-pocket."""

    holdings: pd.DataFrame
    sector_budget: pd.DataFrame
    diagnostics: dict


def _clean_input(df: pd.DataFrame, mega_threshold: float) -> pd.DataFrame:
    missing = [c for c in _REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes: {missing}")

    out = df.copy()
    out["ticker"] = out["ticker"].astype(str).str.strip()
    out["sector"] = out["sector"].astype(str).str.strip()
    out["benchmark_weight"] = pd.to_numeric(out["benchmark_weight"], errors="coerce")
    out["company"] = out["ticker"].map(company_name)

    if out["ticker"].eq("").any():
        raise ValueError("Ticker vide dans le DataFrame d'entree")
    if out["ticker"].duplicated().any():
        dup = out.loc[out["ticker"].duplicated(), "ticker"].tolist()
        raise ValueError(f"Tickers dupliques: {sorted(set(dup))[:10]}")
    if out["sector"].eq("").any() or out["sector"].eq("nan").any():
        raise ValueError("Secteur manquant dans le DataFrame d'entree")
    if out["benchmark_weight"].isna().any():
        raise ValueError("benchmark_weight contient des valeurs non numeriques")

    out = out[out["benchmark_weight"] > 0.0].copy()
    if out.empty:
        raise ValueError("Aucun titre avec un poids benchmark strictement positif")

    total = float(out["benchmark_weight"].sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("Somme des poids benchmark invalide")
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"Les poids benchmark doivent sommer a 1.0 (somme={total:.12f})"
        )
    out["benchmark_weight"] = out["benchmark_weight"] / total

    company_weight = out.groupby("company")["benchmark_weight"].transform("sum")
    out["is_megacap"] = company_weight > float(mega_threshold)
    return out.reset_index(drop=True)


def allocate_sector_quotas(
    budgets: pd.Series,
    capacities: pd.Series,
    n_total: int,
) -> pd.Series:
    """Repartit ``n_total`` sieges proportionnellement aux budgets sectoriels.

    Un minimum de 1 siege est attribue a chaque secteur avec budget > 0 et au
    moins un candidat. Le reste est reparti au plus fort reste, en respectant la
    capacite de chaque secteur. Le total est exact si la capacite globale suffit.
    """
    n_total = int(n_total)
    sectors = [s for s in budgets.index if float(budgets[s]) > _EPS]
    counts = {s: 0 for s in sectors}
    if n_total <= 0 or not sectors:
        return pd.Series(0, index=budgets.index, dtype=int)

    cap = {s: int(capacities.get(s, 0)) for s in sectors}
    feasible = [s for s in sectors if cap[s] >= 1]
    if len(feasible) > n_total:
        raise ValueError(
            f"{len(feasible)} secteurs actifs pour seulement {n_total} titres quant"
        )
    if not feasible:
        raise ValueError("Aucun secteur ne dispose de candidat quant")

    for s in feasible:
        counts[s] = 1
    remaining = n_total - len(feasible)

    while remaining > 0:
        available = [s for s in feasible if counts[s] < cap[s]]
        if not available:
            break
        weights = budgets.reindex(available).astype(float).clip(lower=0.0)
        if float(weights.sum()) <= 0:
            weights = pd.Series(1.0, index=available)
        desired = remaining * weights / float(weights.sum())
        room = pd.Series({s: cap[s] - counts[s] for s in available}, dtype=int)
        add = np.floor(desired).astype(int)
        add = pd.Series(np.minimum(add, room), index=available, dtype=int)

        if int(add.sum()) > remaining:
            for s in add.sort_values(ascending=False).index:
                if int(add.sum()) <= remaining:
                    break
                excess = int(add.sum()) - remaining
                dec = min(int(add[s]), excess)
                add.loc[s] -= dec

        if int(add.sum()) == 0:
            ranked = sorted(
                available,
                key=lambda s: (
                    -(float(desired[s]) - np.floor(float(desired[s]))),
                    -float(budgets[s]),
                    s,
                ),
            )
            add.loc[ranked[0]] = 1

        for s in available:
            value = int(add.loc[s])
            if value:
                counts[s] += value
                remaining -= value

    total = int(sum(counts.values()))
    if remaining > 0 or total != n_total:
        raise ValueError(
            f"Capacite insuffisante pour allouer {n_total} titres "
            f"(alloues={total}, reste={remaining})"
        )
    return pd.Series(counts).reindex(budgets.index).fillna(0).astype(int)


def _allocate_sector_weights(
    raw: pd.Series,
    budget: float,
    cap: Optional[pd.Series] = None,
) -> pd.Series:
    """Alloue un budget sectoriel en respectant un plafond par ligne si fourni."""
    raw = raw.astype(float).clip(lower=0.0)
    if float(raw.sum()) <= _EPS:
        raw = pd.Series(1.0, index=raw.index)
    weights = float(budget) * raw / float(raw.sum())
    if cap is None:
        return weights

    cap = cap.astype(float)
    free = pd.Series(True, index=weights.index)
    for _ in range(len(weights) + 2):
        over = free & (weights > cap + 1e-15)
        if not bool(over.any()):
            break
        weights.loc[over] = cap.loc[over]
        free.loc[over] = False
        residual = float(budget) - float(weights.sum())
        if residual <= 1e-15:
            break
        if not bool(free.any()):
            break
        base = raw.loc[free]
        if float(base.sum()) <= _EPS:
            base = pd.Series(1.0, index=base.index)
        weights.loc[free] += residual * base / float(base.sum())

    residual = float(budget) - float(weights.sum())
    if residual > 1e-12 and float(weights.sum()) > _EPS:
        weights *= float(budget) / float(weights.sum())
    return weights


def build_two_pocket_portfolio(
    df: pd.DataFrame,
    n_quant: int = 40,
    eligible_tickers: Optional[Iterable[str]] = None,
    mega_threshold: float = MEGA_WEIGHT_THRESHOLD,
    score_column: str = "score_quality_pure",
    max_weight_multiple: Optional[float] = None,
    min_multiplier: float = 0.5,
    signal_tilt: float = 0.5,
) -> PortfolioResult:
    """Construit le portefeuille two-pocket a une date de rebalancement.

    ``eligible_tickers`` restreint la poche quant aux titres investissables
    (score et prix disponibles). Les titres non eligibles restent dans le
    benchmark et leur poids sectoriel est realloue aux titres selectionnes.
    """
    data = _clean_input(df, mega_threshold)
    if score_column not in data.columns:
        raise ValueError(f"Colonne de score absente: {score_column}")
    data["score"] = data[score_column].astype(float)
    n_quant = int(n_quant)
    if n_quant <= 0:
        raise ValueError("n_quant doit etre strictement positif")

    benchmark = data.set_index("ticker")["benchmark_weight"].astype(float)
    sectors = data.set_index("ticker")["sector"]
    b_sector = benchmark.groupby(sectors).sum().sort_index()

    mega_mask = data["is_megacap"].astype(bool)
    mega = data.loc[mega_mask].copy()
    mega_sector = (
        mega.groupby("sector")["benchmark_weight"].sum().reindex(b_sector.index).fillna(0.0)
    )

    quant_budget = (b_sector - mega_sector).clip(lower=0.0)
    if float((b_sector - mega_sector).min()) < -1e-9:
        raise ValueError("Poids mega-caps superieur au poids sectoriel benchmark")

    candidates = data.loc[~mega_mask].copy()
    candidates = candidates[candidates["benchmark_weight"] > 0.0]
    candidates = candidates[candidates["score"].notna()]
    if eligible_tickers is not None:
        eligible = {str(t).strip() for t in eligible_tickers}
        candidates = candidates[candidates["ticker"].isin(eligible)]
    candidates = candidates[candidates["sector"].isin(quant_budget.index)]

    active_sectors = quant_budget[quant_budget > _EPS].index
    capacities = candidates.groupby("sector").size().reindex(active_sectors).fillna(0)
    missing_sectors = [
        str(s) for s in active_sectors if int(capacities.get(s, 0)) == 0
    ]
    if missing_sectors:
        raise ValueError(
            "Secteurs avec budget quant mais aucun candidat eligible: "
            + ", ".join(missing_sectors)
        )

    candidates["z"] = 0.0
    for sector, idx in candidates.groupby("sector").groups.items():
        sub = candidates.loc[idx, "score"].astype(float)
        if len(sub) >= 2:
            sigma = float(sub.std(ddof=0))
            if np.isfinite(sigma) and sigma > 0:
                candidates.loc[idx, "z"] = (sub - float(sub.mean())) / sigma

    multiple = None if max_weight_multiple is None else float(max_weight_multiple)
    if multiple is not None and multiple <= 0:
        raise ValueError("max_weight_multiple doit etre strictement positif")

    ranked_by_sector = {
        sector: candidates[candidates["sector"] == sector]
        .sort_values(
            ["z", "benchmark_weight", "ticker"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
        for sector in quant_budget.index
        if float(quant_budget[sector]) > _EPS
    }
    cap_relaxed_sectors: list[str] = []

    if multiple is None:
        counts = allocate_sector_quotas(quant_budget, capacities, n_quant)
    else:
        counts = {}
        for sector, part in ranked_by_sector.items():
            budget = float(quant_budget[sector])
            need = budget / multiple
            cumulative = part["benchmark_weight"].cumsum()
            feasible = cumulative + 1e-12 >= need
            if not bool(feasible.any()):
                n_select = len(part)
                cap_relaxed_sectors.append(str(sector))
            else:
                n_select = int(feasible.idxmax()) + 1
            counts[sector] = n_select

        extra = max(0, n_quant - int(sum(counts.values())))
        while extra > 0:
            options = []
            for sector, part in ranked_by_sector.items():
                n_select = int(counts[sector])
                if n_select < len(part):
                    options.append(
                        (
                            float(part.iloc[n_select]["z"]),
                            float(part.iloc[n_select]["benchmark_weight"]),
                            str(sector),
                        )
                    )
            if not options:
                break
            _, _, sector = max(options)
            counts[sector] = int(counts[sector]) + 1
            extra -= 1
        counts = pd.Series(counts).reindex(quant_budget.index).fillna(0).astype(int)

    selected_parts = []
    for sector, count in counts.items():
        if int(count) <= 0:
            continue
        selected_parts.append(ranked_by_sector[sector].head(int(count)))
    if not selected_parts:
        raise ValueError("Aucun titre quant selectionne")
    selected = pd.concat(selected_parts, axis=0).reset_index(drop=True)

    selected["raw_weight"] = selected["benchmark_weight"] * np.maximum(
        float(min_multiplier), 1.0 + float(signal_tilt) * selected["z"]
    )
    selected["weight"] = 0.0
    for sector, idx in selected.groupby("sector").groups.items():
        raw_sum = float(selected.loc[idx, "raw_weight"].sum())
        if raw_sum <= 0:
            raise ValueError(f"Somme des poids bruts nulle dans le secteur {sector}")
        cap = (
            multiple * selected.loc[idx, "benchmark_weight"]
            if multiple is not None
            else None
        )
        selected.loc[idx, "weight"] = _allocate_sector_weights(
            selected.loc[idx, "raw_weight"],
            float(quant_budget[sector]),
            cap=cap,
        )

    quant_holdings = selected[
        [
            "ticker",
            "company",
            "sector",
            "benchmark_weight",
            "score",
            "z",
            "weight",
        ]
    ].copy()
    quant_holdings["pocket"] = "Quant"

    mega_holdings = mega[
        ["ticker", "company", "sector", "benchmark_weight", "score"]
    ].copy()
    mega_holdings["z"] = np.nan
    mega_holdings["weight"] = mega_holdings["benchmark_weight"]
    mega_holdings["pocket"] = "Mega-cap"

    holdings = pd.concat([mega_holdings, quant_holdings], ignore_index=True)
    holdings = holdings[
        [
            "ticker",
            "company",
            "sector",
            "pocket",
            "score",
            "z",
            "benchmark_weight",
            "weight",
        ]
    ]
    holdings["active_weight"] = holdings["weight"] - holdings["benchmark_weight"]
    holdings = holdings.sort_values(
        ["weight", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)

    mega_by = (
        holdings[holdings["pocket"] == "Mega-cap"]
        .groupby("sector")
        .agg(
            mega_weight=("weight", "sum"),
            n_mega=("ticker", "size"),
            n_mega_companies=("company", "nunique"),
        )
    )
    quant_by = (
        holdings[holdings["pocket"] == "Quant"]
        .groupby("sector")
        .agg(quant_weight=("weight", "sum"), n_quant=("ticker", "size"))
    )
    budget = pd.DataFrame(index=b_sector.index)
    budget.index.name = "sector"
    budget["benchmark_weight"] = b_sector
    budget["mega_weight"] = mega_by.reindex(budget.index)["mega_weight"].fillna(0.0)
    budget["quant_budget"] = quant_budget
    budget["quant_weight"] = quant_by.reindex(budget.index)["quant_weight"].fillna(0.0)
    budget["portfolio_weight"] = budget["mega_weight"] + budget["quant_weight"]
    budget["active_weight"] = budget["portfolio_weight"] - budget["benchmark_weight"]
    budget["n_mega"] = mega_by.reindex(budget.index)["n_mega"].fillna(0).astype(int)
    budget["n_mega_companies"] = (
        mega_by.reindex(budget.index)["n_mega_companies"].fillna(0).astype(int)
    )
    budget["n_quant"] = quant_by.reindex(budget.index)["n_quant"].fillna(0).astype(int)
    budget["n_total"] = budget["n_mega"] + budget["n_quant"]
    budget = budget.reset_index()

    portfolio_sector = (
        holdings.groupby("sector")["weight"].sum().reindex(b_sector.index).fillna(0.0)
    )
    sector_gap = portfolio_sector - b_sector
    total_weight = float(holdings["weight"].sum())
    active_share_frame = pd.concat(
        [
            holdings.set_index("ticker")["weight"].rename("portfolio"),
            benchmark.rename("benchmark"),
        ],
        axis=1,
    ).fillna(0.0)
    active_share = float(
        0.5 * (active_share_frame["portfolio"] - active_share_frame["benchmark"]).abs().sum()
    )
    budget_residual = float(
        (budget["quant_budget"] - budget["quant_weight"]).abs().max()
    )

    max_weight_ratio_quant = np.nan
    if len(quant_holdings):
        ratios = (
            quant_holdings["weight"] / quant_holdings["benchmark_weight"]
        ).replace([np.inf, -np.inf], np.nan)
        max_weight_ratio_quant = float(ratios.max())
    diagnostics = {
        "n_quant_target": n_quant,
        "n_quant_min": n_quant,
        "n_quant_max": int(capacities.sum()),
        "n_quant_selected": int(len(quant_holdings)),
        "max_weight_multiple": multiple if multiple is not None else np.nan,
        "max_weight_ratio_quant": max_weight_ratio_quant,
        "cap_relaxed": bool(cap_relaxed_sectors),
        "cap_relaxed_sectors": cap_relaxed_sectors,
        "n_mega": int(len(mega_holdings)),
        "n_mega_companies": int(mega_holdings["company"].nunique()),
        "n_positions": int(len(holdings)),
        "mega_weight": float(mega_holdings["weight"].sum()),
        "quant_weight": float(quant_holdings["weight"].sum()),
        "total_weight": total_weight,
        "active_share": active_share,
        "max_sector_gap": float(sector_gap.abs().max()),
        "budget_residual": budget_residual,
        "sector_gap": sector_gap.rename("gap").reset_index(),
        "n_active_sectors": int(len(active_sectors)),
    }

    result = PortfolioResult(
        holdings=holdings,
        sector_budget=budget,
        diagnostics=diagnostics,
    )
    checks = validate_two_pocket(result, raise_on_error=True)
    result.diagnostics["checks"] = checks
    return result


def validate_two_pocket(
    result: PortfolioResult,
    tol_sum: float = 1e-6,
    tol_sector: float = 1e-4,
    tol_budget: float = 1e-8,
    raise_on_error: bool = True,
) -> pd.DataFrame:
    """Verifie les contraintes du portefeuille et retourne le detail des checks."""
    diag = result.diagnostics
    broadcast = [
        (
            "sum_weight",
            float(diag["total_weight"]),
            1.0,
            tol_sum,
            abs(float(diag["total_weight"]) - 1.0) <= tol_sum,
        ),
        (
            "max_sector_gap",
            float(diag["max_sector_gap"]),
            0.0,
            tol_sector,
            float(diag["max_sector_gap"]) <= tol_sector,
        ),
        (
            "budget_residual",
            float(diag["budget_residual"]),
            0.0,
            tol_budget,
            float(diag["budget_residual"]) <= tol_budget,
        ),
        (
            "n_positions",
            float(diag["n_positions"]),
            np.nan,
            np.nan,
            int(diag["n_positions"]) > 0,
        ),
        (
            "n_quant_selected",
            float(diag["n_quant_selected"]),
            float(diag["n_quant_target"]),
            0.0,
            int(diag["n_quant_min"])
            <= int(diag["n_quant_selected"])
            <= int(diag["n_quant_max"]),
        ),
        (
            "active_share",
            float(diag["active_share"]),
            np.nan,
            np.nan,
            0.0 <= float(diag["active_share"]) <= 1.0 + 1e-9,
        ),
    ]
    checks = pd.DataFrame(
        broadcast,
        columns=["check", "value", "target", "tolerance", "ok"],
    )
    if raise_on_error and not bool(checks["ok"].all()):
        failed = checks.loc[~checks["ok"]].to_string(index=False)
        raise AssertionError(f"Validation two-pocket echouee:\n{failed}")
    return checks
