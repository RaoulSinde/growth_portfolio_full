"""Score d'acceleration fondamentale et construction du portefeuille 50 titres."""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from . import data_loader as dl
from . import factors
from . import msci_config as cfg
from . import msci_universe as universe


def compute_acceleration_scores(
    fund: pd.DataFrame,
    date: pd.Timestamp,
    tickers: Iterable[str],
    sector: pd.Series,
) -> pd.DataFrame:
    """Z_Acceleration PIT a une date, sans purge ni autre pilier."""
    date = pd.Timestamp(date)
    tickers = pd.Index(sorted(set(str(t) for t in tickers)))
    if fund is None or fund.empty:
        return pd.DataFrame(index=tickers, columns=["Z_Acceleration"])

    f = fund.copy()
    f["pub_date"] = pd.to_datetime(f["pub_date"])
    f["period_end"] = pd.to_datetime(f["period_end"])
    f = f[(f["pub_date"] <= date) & (f["ticker"].isin(tickers))]
    f = f.drop_duplicates(subset=["ticker", "period_end"], keep="last")

    rev = dl.annual_flow_points(f, "revenue_q", tickers)
    oi = dl.annual_flow_points(f, "operating_income_q", tickers)
    ni = dl.annual_flow_points(f, "net_income_q", tickers)
    ta = dl.annual_stock_points(f, "total_assets", tickers)

    sec = pd.Series(sector).reindex(tickers)
    is_fin = (sec == "Financials").fillna(False)

    sales_accel = (rev["N"] / rev["N1"] - 1.0) - (rev["N1"] / rev["N2"] - 1.0)
    margin_exp = oi["N"] / rev["N"] - oi["N1"] / rev["N1"]
    asset_eff = rev["N"] / ta["N"] - rev["N1"] / ta["N1"]
    roa_delta = ni["N"] / ta["N"] - ni["N1"] / ta["N1"]
    asset_eff = asset_eff.where(~is_fin, roa_delta)

    sales_accel = sales_accel.replace([np.inf, -np.inf], np.nan)
    margin_exp = margin_exp.replace([np.inf, -np.inf], np.nan)
    asset_eff = asset_eff.replace([np.inf, -np.inf], np.nan)

    z_sales = factors.zscore(sales_accel, by_sector=True, sector=sec)
    z_margin = factors.zscore(margin_exp, by_sector=True, sector=sec)
    z_asset = factors.zscore(asset_eff, by_sector=True, sector=sec)
    blended = pd.concat([z_sales, z_margin, z_asset], axis=1).mean(axis=1, skipna=True)
    z_accel = factors.zscore(blended, by_sector=True, sector=sec)

    out = pd.DataFrame(
        {
            "Z_Acceleration": z_accel,
            "S_SalesAccel": z_sales,
            "S_MarginExp": z_margin,
            "S_AssetEff": z_asset,
            "sales_accel": sales_accel,
            "margin_exp": margin_exp,
            "asset_eff": asset_eff,
        },
        index=tickers,
    )
    return out


def select_and_weight(
    scores: pd.DataFrame,
    benchmark_weights: pd.Series,
    n: int = cfg.N_HOLDINGS,
    lam: float = cfg.LAMBDA,
    z_clip: float = cfg.Z_CLIP,
) -> pd.DataFrame:
    """Selectionne les n meilleurs Z et applique le multiplicateur exponentiel."""
    w = pd.Series(benchmark_weights).astype(float)
    w.index = w.index.astype(str)
    z = scores["Z_Acceleration"].astype(float)
    z.index = z.index.astype(str)
    candidates = pd.DataFrame({"z": z, "benchmark_weight": w}).dropna(subset=["z"])
    candidates = candidates[candidates["benchmark_weight"] > 0].copy()
    if len(candidates) < n:
        raise ValueError(f"{len(candidates)} titres scores < {n} demandes")
    candidates = candidates.sort_values(
        ["z", "benchmark_weight"], ascending=[False, False]
    )
    selected = candidates.head(n).copy()
    selected["z_clipped"] = selected["z"].clip(-z_clip, z_clip)
    selected["raw_weight"] = selected["benchmark_weight"] * np.exp(lam * selected["z_clipped"])
    total = float(selected["raw_weight"].sum())
    if total <= 0:
        raise ValueError("Somme des poids bruts nulle")
    selected["target_weight"] = selected["raw_weight"] / total
    selected["selected"] = True
    selected = selected.reset_index().rename(columns={"index": "ticker"})
    return selected[
        [
            "ticker",
            "z",
            "z_clipped",
            "benchmark_weight",
            "raw_weight",
            "target_weight",
            "selected",
        ]
    ]


def build_acceleration_portfolios(
    uni: dict,
    fund: pd.DataFrame,
    lam: float = cfg.LAMBDA,
    n: int = cfg.N_HOLDINGS,
) -> dict:
    """Construit scores et poids a chaque date de rebalancement."""
    score_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for date in uni["dates"]:
        elig = universe.eligible_universe(uni, date)
        if len(elig) < n:
            raise ValueError(f"{date.date()}: seulement {len(elig)} titres eligibles")
        sector = elig.set_index("ticker")["sector"]
        scores = compute_acceleration_scores(fund, date, elig["ticker"], sector)
        scored = scores.join(elig.set_index("ticker")[["weight", "sector"]], how="inner")
        scored = scored.dropna(subset=["Z_Acceleration"])
        if len(scored) < n:
            raise ValueError(f"{date.date()}: seulement {len(scored)} titres scores")

        selected = select_and_weight(
            scored[["Z_Acceleration"]], elig.set_index("ticker")["weight"], n=n, lam=lam
        )
        selected = selected.merge(
            scored[["sector", "S_SalesAccel", "S_MarginExp", "S_AssetEff"]],
            left_on="ticker",
            right_index=True,
            how="left",
        )
        selected.insert(0, "date", date)
        score_frames.append(selected)

        weights = selected.set_index("ticker")["target_weight"].rename(date)
        weight_frames.append(weights)

        summary_rows.append(
            {
                "date": date,
                "n_eligible": int(len(elig)),
                "n_scored": int(len(scored)),
                "n_selected": int(len(selected)),
                "benchmark_weight_selected": float(selected["benchmark_weight"].sum()),
                "max_weight": float(selected["target_weight"].max()),
                "top10_weight": float(selected["target_weight"].nlargest(10).sum()),
                "mean_z": float(selected["z"].mean()),
                "min_z": float(selected["z"].min()),
                "max_z": float(selected["z"].max()),
            }
        )

    scores_out = pd.concat(score_frames, ignore_index=True)
    weights_out = pd.concat(weight_frames, axis=1).fillna(0.0).sort_index()
    summary = pd.DataFrame(summary_rows)
    return {"scores": scores_out, "weights": weights_out, "summary": summary}


def save_acceleration_outputs(result: dict) -> None:
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result["scores"].to_parquet(cfg.SCORES_FILE, index=False)
    result["weights"].to_parquet(cfg.WEIGHTS_FILE)

def project_bounded_simplex(
    weights: pd.Series,
    lower: pd.Series,
    upper: pd.Series,
    max_iter: int = 10_000,
    tol: float = 1e-12,
) -> pd.Series:
    """Projette des poids sur le simplexe avec bornes individuelles."""
    index = pd.Index(weights.index)
    w = np.asarray(pd.Series(weights).reindex(index), dtype=float)
    lo = np.asarray(pd.Series(lower).reindex(index), dtype=float)
    hi = np.asarray(pd.Series(upper).reindex(index), dtype=float)
    if not np.isfinite(w).all() or not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("Poids ou bornes non finis")
    if np.any(lo > hi + 1e-12):
        raise ValueError("Borne inferieure superieure a la borne superieure")
    if lo.sum() > 1.0 + 1e-9 or hi.sum() < 1.0 - 1e-9:
        raise ValueError("Bornes incompatibles avec une somme de poids egale a 1")

    x = np.clip(w, lo, hi)
    for _ in range(max_iter):
        gap = 1.0 - x.sum()
        if abs(gap) <= tol:
            break
        capacity = (hi - x) if gap > 0 else (x - lo)
        capacity = np.clip(capacity, 0.0, None)
        total = capacity.sum()
        if total <= tol:
            break
        x = np.clip(x + gap * capacity / total, lo, hi)
    else:
        raise RuntimeError("Projection sur le simplexe non convergente")

    residual = 1.0 - x.sum()
    if abs(residual) > 1e-10:
        room = (hi - x) if residual > 0 else (x - lo)
        j = int(np.argmax(room))
        if room[j] + 1e-12 < abs(residual):
            raise RuntimeError("Projection sur le simplexe impossible")
        x[j] += residual
        x = np.clip(x, lo, hi)
    if abs(x.sum() - 1.0) > 1e-8:
        raise RuntimeError("Projection sur le simplexe non normalisee")
    return pd.Series(x, index=index, name=weights.name)


def build_anchor_score_weights(
    scores: pd.DataFrame,
    benchmark_weights: pd.Series,
    n_anchors: int = cfg.N_ANCHORS,
    n_total: int = cfg.N_TOTAL_HOLDINGS,
    lam: float = cfg.LAMBDA,
    z_clip: float = cfg.Z_CLIP,
    band_rel: float = cfg.BAND_REL,
    band_abs_cap: float = cfg.BAND_ABS_CAP,
    max_weight: float | None = None,
    exclude_tickers: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Anchors par poids d'indice, completes par les meilleurs scores."""
    if n_anchors <= 0 or n_anchors >= n_total:
        raise ValueError("n_anchors doit etre strictement entre 0 et n_total")
    n_score_picks = n_total - n_anchors

    z = pd.Series(scores["Z_Acceleration"]).astype(float)
    z.index = z.index.astype(str)
    w = pd.Series(benchmark_weights).astype(float)
    w.index = w.index.astype(str)
    candidates = pd.DataFrame({"z": z, "benchmark_weight": w})
    candidates = candidates[candidates["benchmark_weight"] > 0].copy()
    excluded = {str(ticker) for ticker in (exclude_tickers or ())}
    excluded_present = candidates.index[candidates.index.isin(excluded)]
    if excluded_present.size:
        candidates = candidates.drop(index=excluded_present)
    if len(candidates) < n_total:
        raise ValueError(f"{len(candidates)} titres eligibles < {n_total} demandes")

    anchors = candidates.sort_values(
        ["benchmark_weight", "z"], ascending=[False, False], na_position="last"
    ).head(n_anchors)
    remaining = candidates.drop(index=anchors.index)
    scored = remaining.dropna(subset=["z"]).sort_values(
        ["z", "benchmark_weight"], ascending=[False, False]
    )
    if len(scored) < n_score_picks:
        raise ValueError(f"{len(scored)} titres scores < {n_score_picks} demandes")
    score_picks = scored.head(n_score_picks)
    selected = pd.concat([anchors, score_picks], axis=0).copy()

    w0 = selected["benchmark_weight"] / selected["benchmark_weight"].sum()
    z_filled = selected["z"].fillna(0.0)
    z_clipped = z_filled.clip(-z_clip, z_clip)
    raw = w0 * np.exp(lam * z_clipped)
    if not np.isfinite(raw).all() or raw.sum() <= 0:
        raise ValueError("Poids bruts invalides")
    tilt = raw / raw.sum()

    if max_weight is not None and not (0.0 < max_weight <= 1.0):
        raise ValueError("max_weight doit etre dans (0, 1]")

    band = np.minimum(band_rel * w0, band_abs_cap)
    lower = (w0 - band).clip(lower=0.0)
    upper = w0 + band
    if max_weight is not None:
        upper = upper.clip(upper=max_weight)
        lower = np.minimum(lower, upper)
    target = project_bounded_simplex(tilt, lower, upper)

    anchor_order = {ticker: rank + 1 for rank, ticker in enumerate(anchors.index)}
    score_order = {ticker: rank + 1 for rank, ticker in enumerate(score_picks.index)}
    selected["is_anchor"] = selected.index.isin(anchors.index)
    selected["anchor_rank"] = [anchor_order.get(t, np.nan) for t in selected.index]
    selected["score_rank"] = [score_order.get(t, np.nan) for t in selected.index]
    selected["rebased_weight"] = w0
    selected["band"] = band
    selected["lower_weight"] = lower
    selected["upper_weight"] = upper
    selected["z_filled"] = z_filled
    selected["z_clipped"] = z_clipped
    selected["tilt_weight"] = tilt
    selected["target_weight"] = target
    selected["active_weight"] = target - w0
    selected["projection_adjustment"] = (target - tilt).abs()
    selected["band_binding_low"] = np.isclose(target, lower, atol=1e-10)
    selected["band_binding_high"] = np.isclose(target, upper, atol=1e-10)
    selected["cap_binding"] = (
        np.isclose(target, max_weight, atol=1e-10)
        if max_weight is not None
        else False
    )
    selected["cap_forced"] = (
        np.isclose(lower, upper) & np.isclose(upper, max_weight)
        if max_weight is not None
        else False
    )
    selected["max_weight_cap"] = max_weight if max_weight is not None else np.nan
    selected.index.name = "ticker"
    return selected


def build_anchor_portfolios(
    uni: dict,
    fund: pd.DataFrame,
    n_anchors: int = cfg.N_ANCHORS,
    n_total: int = cfg.N_TOTAL_HOLDINGS,
    lam: float = cfg.LAMBDA,
    max_weight: float | None = None,
    exclude_tickers: Iterable[str] | None = None,
) -> dict:
    """Construit les portefeuilles anchors + score picks a chaque date."""
    score_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    summary_rows: list[dict] = []

    for date in uni["dates"]:
        elig = universe.eligible_universe(uni, date)
        if len(elig) < n_total:
            raise ValueError(f"{date.date()}: seulement {len(elig)} titres eligibles")
        sector = elig.set_index("ticker")["sector"]
        scores = compute_acceleration_scores(fund, date, elig["ticker"], sector)
        scored = scores.join(elig.set_index("ticker")[["weight", "sector"]], how="inner")

        selected = build_anchor_score_weights(
            scored[["Z_Acceleration"]],
            elig.set_index("ticker")["weight"],
            n_anchors=n_anchors,
            n_total=n_total,
            lam=lam,
            max_weight=max_weight,
            exclude_tickers=exclude_tickers,
        )
        selected = selected.join(
            scored[
                [
                    "sector",
                    "S_SalesAccel",
                    "S_MarginExp",
                    "S_AssetEff",
                    "sales_accel",
                    "margin_exp",
                    "asset_eff",
                ]
            ],
            how="left",
        )
        selected.insert(0, "date", date)
        score_frames.append(selected.reset_index())

        weights = selected["target_weight"].rename(date)
        weight_frames.append(weights)

        anchors_mask = selected["is_anchor"]
        excluded = {str(ticker) for ticker in (exclude_tickers or ())}
        summary_rows.append(
            {
                "date": date,
                "n_eligible": int(len(elig)),
                "n_excluded_present": int(elig["ticker"].isin(excluded).sum()),
                "n_anchors": int(anchors_mask.sum()),
                "n_score_picks": int((~anchors_mask).sum()),
                "n_selected": int(len(selected)),
                "benchmark_weight_selected": float(selected["benchmark_weight"].sum()),
                "anchor_benchmark_weight": float(
                    selected.loc[anchors_mask, "benchmark_weight"].sum()
                ),
                "score_benchmark_weight": float(
                    selected.loc[~anchors_mask, "benchmark_weight"].sum()
                ),
                "anchor_target_weight": float(
                    selected.loc[anchors_mask, "target_weight"].sum()
                ),
                "score_target_weight": float(
                    selected.loc[~anchors_mask, "target_weight"].sum()
                ),
                "max_weight": float(selected["target_weight"].max()),
                "top10_weight": float(selected["target_weight"].nlargest(10).sum()),
                "mean_score_z": float(selected.loc[~anchors_mask, "z"].mean()),
                "min_score_z": float(selected.loc[~anchors_mask, "z"].min()),
                "max_score_z": float(selected.loc[~anchors_mask, "z"].max()),
                "n_missing_anchor_score": int(
                    selected.loc[anchors_mask, "z"].isna().sum()
                ),
                "band_low_bindings": int(selected["band_binding_low"].sum()),
                "band_high_bindings": int(selected["band_binding_high"].sum()),
                "cap_bindings": int(selected["cap_binding"].sum()),
                "cap_forced": int(selected["cap_forced"].sum()),
                "max_weight_cap": (
                    float(max_weight) if max_weight is not None else np.nan
                ),
                "max_active_weight": float(selected["active_weight"].abs().max()),
                "projection_adjustment": float(
                    selected["projection_adjustment"].sum()
                ),
            }
        )

    scores_out = pd.concat(score_frames, ignore_index=True)
    weights_out = pd.concat(weight_frames, axis=1).fillna(0.0).sort_index()
    summary = pd.DataFrame(summary_rows)
    return {"scores": scores_out, "weights": weights_out, "summary": summary}


def anchor_variant_specs() -> dict[str, dict]:
    """Specifications des variantes de base, capees et hors NVDA."""
    specs = {
        name: {"n_anchors": n_anchors, "max_weight": None, "exclude_tickers": ()}
        for name, n_anchors in cfg.ANCHOR_VARIANTS.items()
    }
    specs.update(
        {
            name: {
                "n_anchors": n_anchors,
                "max_weight": cfg.CAP_VARIANT_CAPS.get(name, cfg.CAP_MAX_WEIGHT),
                "exclude_tickers": (),
            }
            for name, n_anchors in cfg.CAP_VARIANTS.items()
        }
    )
    specs.update(
        {
            name: {**spec, "exclude_tickers": cfg.EXCLUDED_TICKERS}
            for name, spec in cfg.EX_NVDA_VARIANTS.items()
        }
    )
    return specs


def build_anchor_variants(
    uni: dict,
    fund: pd.DataFrame,
    variants: dict[str, dict | int] | None = None,
    n_total: int = cfg.N_TOTAL_HOLDINGS,
    lam: float = cfg.LAMBDA,
) -> dict[str, dict]:
    """Construit les variantes anchors + score picks."""
    if variants is None:
        variants = anchor_variant_specs()
    output = {}
    for name, spec in variants.items():
        if isinstance(spec, int):
            spec = {"n_anchors": spec, "max_weight": None, "exclude_tickers": ()}
        output[name] = build_anchor_portfolios(
            uni,
            fund,
            n_anchors=spec["n_anchors"],
            n_total=n_total,
            lam=lam,
            max_weight=spec.get("max_weight"),
            exclude_tickers=spec.get("exclude_tickers"),
        )
    return output


def save_anchor_outputs(name: str, result: dict) -> None:
    """Sauvegarde les scores et poids d'une variante anchors + score."""
    cfg.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    result["scores"].to_parquet(
        cfg.PROCESSED_DIR / f"anchor_{name}_scores.parquet", index=False
    )
    result["weights"].to_parquet(cfg.PROCESSED_DIR / f"anchor_{name}_weights.parquet")
