"""Construction du portefeuille "Core + Satellites" (45 a 60 titres).

La construction suit la methodologie du CONTRACT.md :

1. **Selection** (:func:`select_titles`) : les Mega-Caps (poids indiciel
   ``>= core_threshold``) forment le noyau *Core* toujours detenu ; les
   *Satellites* sont repartis par secteur proportionnellement au poids sectoriel
   de l'indice, puis remplis par les meilleurs ``Z_Composite`` eligibles. Une
   *buffer rule* conserve un satellite deja detenu tant qu'il reste dans le
   premier tiers de son secteur.
2. **Tilt factoriel** (:func:`tilt_weights`) : poids brut
   ``w_i ~ w_idx_i * exp(tilt * clip(z_i, -3, 3))``, avec un multiplicateur de
   conviction deux fois plus sensible sur les Satellites que sur le Core.
3. **Projection** (:func:`project_weights`) : projection sur le simplexe borne
   (poids actifs ``[dw_min, dw_max]``, cap ``w_max``, cibles sectorielles) par
   clamp + redistribution iterative par secteur puis au global.

L'orchestration est assuree par :func:`build_portfolio` (une date) et
:func:`build_all_portfolios` (toutes les dates de rebalancement).

SECURITE : aucune donnee issue de l'attribut ``info`` d'un Ticker.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from . import config
from . import data_loader as dl
from . import factors

__all__ = [
    "select_titles",
    "tilt_weights",
    "project_weights",
    "build_portfolio",
    "build_all_portfolios",
    "DEFAULT_PARAMS",
    "clear_sector_cache",
]

# Seuil de detection du noyau Mega-Cap (poids indiciel en fraction).
CORE_THRESHOLD = 0.025

# Plafond de concentration historique (somme des 10 plus gros poids). CONSERVE
# uniquement pour la retro-compatibilite des imports ; la contrainte Top10 n'est
# PLUS appliquee par defaut (``max_top10=None``). Le Top10 resulte librement de
# la selection Core + Satellites et du tilt factoriel.
MAX_TOP10 = 0.53

# Parametres par defaut (identiques a src.config mais regroupes ici pour l'UI).
DEFAULT_PARAMS: dict = {
    "n_titles": config.DEFAULT_N,
    "sector_band": config.DEFAULT_SECTOR_BAND,
    "sector_pilot": True,
    "dw_min": config.DEFAULT_DW_MIN,
    "dw_max": config.DEFAULT_DW_MAX,
    "w_max": config.DEFAULT_W_MAX,
    "buffer_rule": True,
    # Aucune borne sur le Top10 cumule (decision utilisateur) : None desactive
    # toute redistribution de concentration.
    "max_top10": None,
    "weights": {
        "acceleration": config.DEFAULT_W_ACCEL,
        "momentum": config.DEFAULT_W_MOMENTUM,
        "quality": config.DEFAULT_W_QUALITY,
    },
}

# Cache du mapping ticker -> secteur GICS.
_SECTOR_MAP_CACHE: Optional[pd.Series] = None


# ===========================================================================
# Helpers
# ===========================================================================
def clear_sector_cache() -> None:
    """Vide le cache interne du mapping secteur (utile en test)."""
    global _SECTOR_MAP_CACHE
    _SECTOR_MAP_CACHE = None


def _get_sector_map() -> pd.Series:
    """Mapping ``ticker -> gics_sector`` (mis en cache, univers PIT complet).

    On utilise :func:`data_loader.get_sector_map` (``data/sector_map.csv``,
    couverture 100% des ~213 tickers de l'univers trimestriel) et non l'ancien
    ``universe.csv`` statique (90 titres), qui laisserait des secteurs NaN sur la
    majorite de l'univers PIT multi-dates.
    """
    global _SECTOR_MAP_CACHE
    if _SECTOR_MAP_CACHE is None:
        _SECTOR_MAP_CACHE = dl.get_sector_map()
    return _SECTOR_MAP_CACHE


def _to_ts(date) -> pd.Timestamp:
    """Normalise une date en Timestamp timezone-naive a minuit."""
    ts = pd.Timestamp(date)
    if ts.tzinfo is not None:
        ts = ts.tz_localize(None)
    return ts.normalize()


def _index_weight_row(index_weights, date) -> pd.Series:
    """Extrait le vecteur de poids indiciels d'une date (``Series`` index=ticker).

    Accepte les trois formats produits par :func:`data_loader.get_index_weights` :
    ``DataFrame`` MultiIndex ``(date, ticker)`` (cas multi-dates), ``Series``
    indexee par ticker (cas date unique, retro-compatibilite) ou ``DataFrame``
    index=dates / colonnes=tickers (ancien format).
    """
    if index_weights is None:
        return pd.Series(dtype=float)
    d = _to_ts(date)

    if isinstance(index_weights, pd.Series):
        out = index_weights.astype(float)
        out.name = "weight"
        return out

    if isinstance(index_weights.index, pd.MultiIndex):
        for level in ("date", 0):
            try:
                sub = index_weights.xs(d, level=level)
            except (KeyError, TypeError):
                continue
            if isinstance(sub, pd.DataFrame):
                sub = sub["weight"] if "weight" in sub.columns else sub.iloc[:, 0]
            out = sub.astype(float)
            out.name = "weight"
            return out
        return pd.Series(dtype=float)

    idx = pd.DatetimeIndex(pd.to_datetime(index_weights.index)).normalize()
    if d in idx:
        out = index_weights.loc[d].astype(float)
        if isinstance(out, pd.DataFrame):
            out = out["weight"] if "weight" in out.columns else out.iloc[:, 0]
        out.name = "weight"
        return out
    return pd.Series(dtype=float)


def _allocate_counts(total: int, weights: pd.Series, min_each: int = 1) -> Dict:
    """Repartit ``total`` unites sur des cles proportionnellement a ``weights``.

    Methode du plus fort reste, avec un minimum ``min_each`` par cle. Le total
    est respecte exactement des lors que ``total >= min_each * n_cles``.
    """
    keys = list(weights.index)
    n = len(keys)
    total = int(total)
    if n == 0 or total <= 0:
        return {k: 0 for k in keys}

    w = weights.astype(float).clip(lower=0.0)
    if w.sum() <= 0:
        w = pd.Series(1.0, index=keys)
    quota = w / w.sum() * total
    base = np.floor(quota.to_numpy()).astype(int)
    base = np.maximum(base, min_each)

    # Si le minimum force un depassement, on retire au plus gros poste.
    if base.sum() > total:
        order = np.argsort(-w.to_numpy())
        i = 0
        while base.sum() > total:
            j = order[i % n]
            if base[j] > min_each:
                base[j] -= 1
            i += 1
            if i > 10 * total + 10:  # garde-fou
                break

    # Repartition du reste au plus fort reste.
    remainder = total - int(base.sum())
    if remainder > 0:
        frac = (quota - np.floor(quota)).to_numpy()
        order = np.argsort(-frac)
        i = 0
        while remainder > 0:
            base[order[i % n]] += 1
            remainder -= 1
            i += 1
    return {k: int(v) for k, v in zip(keys, base)}


# ===========================================================================
# 1) Selection Core + Satellites
# ===========================================================================
def select_titles(panel_date: pd.DataFrame,
                  w_idx_date: pd.Series,
                  prev_holdings: Optional[Iterable[str]],
                  n: int,
                  buffer_rule: bool,
                  core_threshold: float = CORE_THRESHOLD,
                  sector_target: Optional[pd.Series] = None,
                  dw_max: float = config.DEFAULT_DW_MAX,
                  w_max: float = config.DEFAULT_W_MAX) -> List[str]:
    """Selectionne ``n`` titres : noyau Mega-Cap + satellites par secteur.

    - **Core** : titres eligibles dont le poids indiciel ``>= core_threshold``
      (toujours detenus pour ancrer le beta).
    - **Satellites** : les ``n - len(core)`` places sont reparties par secteur
      proportionnellement au poids sectoriel de l'indice (ou, si fourni, au poids
      sectoriel *cible* de maniere a garantir la faisabilite des bornes), minimum
      1 par secteur disposant de candidats non-Core, puis remplies par les
       meilleurs ``Z_Composite``.
    - **Buffer rule** : un satellite detenu en ``t-1`` reste selectionne si son
      ``Z_Composite`` depasse le 60e centile de son secteur ce mois-ci.

    Retourne la liste des tickers (au plus ``n`` si l'univers est trop petit).
    """
    n = int(n)
    panel_date = panel_date.copy()
    w_idx_date = w_idx_date.astype(float)
    sector_map = _get_sector_map()

    eligible = panel_date.index[panel_date["eligible"].astype(bool)]
    # Restreindre aux titres presents dans l'indice (poids strictement positif).
    eligible = pd.Index([t for t in eligible if float(w_idx_date.get(t, 0.0)) > 0.0])

    w_elig = w_idx_date.reindex(eligible).fillna(0.0)
    z_elig = panel_date["Z_Composite"].reindex(eligible).fillna(0.0)
    sec_elig = sector_map.reindex(eligible)

    core = [t for t in eligible if float(w_elig[t]) >= core_threshold]
    sat_pool = eligible.difference(pd.Index(core))
    n_sat = max(0, n - len(core))

    prev = set(prev_holdings) if prev_holdings is not None else set()

    if n_sat == 0 or len(sat_pool) == 0:
        return list(core[:n])

    # Poids sectoriel indiciel (agregation du poids de TOUS les titres).
    sec_all = sector_map.reindex(w_idx_date.index).fillna("__NA__")
    sec_weight = w_idx_date.groupby(sec_all).sum()

    sec_sat = sector_map.reindex(sat_pool)
    # Ne conserver que les secteurs disposant de candidats non-Core.
    present = [s for s in sec_weight.index if (sec_sat == s).any()]
    # Repartition : on utilise la cible sectorielle si fournie (faisabilite des
    # bornes actives) ; sinon le poids sectoriel indiciel.
    alloc_w = sec_weight.reindex(present)
    if sector_target is not None:
        tgt = pd.Series(sector_target).reindex(present)
        if tgt.notna().any() and float(tgt.fillna(0.0).sum()) > 0:
            alloc_w = tgt.fillna(0.0)
    if float(alloc_w.sum()) <= 0:
        alloc_w = sec_weight.reindex(present)
    # Ecarter les secteurs sans allocation (cible nulle) pour eviter des lignes
    # a poids nul : les sieges sont redonnes aux secteurs porteurs.
    present = [s for s in present if float(alloc_w.get(s, 0.0)) > 1e-12]
    alloc_w = alloc_w.reindex(present).fillna(0.0)
    counts = _allocate_counts(n_sat, alloc_w, min_each=1)

    # Candidats tries par Z decroissant, par secteur.
    cand_sorted: dict = {}
    for s in present:
        cand = sat_pool[sec_sat.reindex(sat_pool) == s]
        cand_sorted[s] = z_elig.reindex(cand).sort_values(ascending=False)

    # NOTE : l'ancien ajustement de faisabilite (comparant la cible sectorielle a
    # la capacite des *satellites seuls*, sans les Core deja presents dans le
    # secteur) retirait des sieges aux secteurs en deficit au profit de l'IT et
    # oscillait. On conserve l'allocation proportionnelle aux cibles issue de
    # ``_allocate_counts`` : les ecarts sectoriels restent dans la bande +/-
    # (garantie par la projection bornee de :func:`project_weights`).

    selected: List[str] = list(core)
    for s, cnt in counts.items():
        if cnt <= 0:
            continue
        cand = sat_pool[sec_sat.reindex(sat_pool) == s]
        if len(cand) == 0:
            continue
        z_s = z_elig.reindex(cand)

        picks: List[str] = []
        if buffer_rule and prev:
            p60 = float(z_s.quantile(0.60))
            held = [t for t in cand if t in prev and float(z_s[t]) >= p60]
            held_sorted = sorted(held, key=lambda t: float(z_s[t]), reverse=True)
            picks = held_sorted[:cnt]

        if len(picks) < cnt:
            rest = z_s.drop(index=[t for t in picks if t in z_s.index])
            rest = rest.sort_values(ascending=False)
            picks.extend(list(rest.index[:cnt - len(picks)]))

        selected.extend(picks)

    # Exactitude du compte : completer par le meilleur Z global restant.
    if len(selected) < n:
        have = set(selected)
        rest = z_elig.drop(index=[t for t in have if t in z_elig.index])
        rest = rest.sort_values(ascending=False)
        selected.extend([t for t in rest.index if t not in have][:n - len(selected)])

    # Deduplication en preservant l'ordre (core d'abord).
    seen: set = set()
    out: List[str] = []
    for t in selected:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:n]


# ===========================================================================
# 2) Tilt factoriel
# ===========================================================================
def tilt_weights(selected: Sequence[str],
                 z_date: pd.Series,
                 w_idx_date: pd.Series,
                 tilt_core: float = 0.35,
                 tilt_sat: float = 0.70,
                 core_threshold: float = CORE_THRESHOLD) -> pd.Series:
    """Poids brut ``w_idx_i * exp(tilt * clip(z_i, -3, 3))`` par titre.

    Le multiplicateur ``tilt`` vaut ``tilt_core`` pour le noyau Mega-Cap
    (``w_idx >= core_threshold``) et ``tilt_sat`` (deux fois plus sensible) pour
    les satellites. Le resultat n'est pas normalise (la projection s'en charge).
    """
    sel = list(dict.fromkeys(selected))
    z = z_date.reindex(sel).astype(float).fillna(0.0).clip(-3.0, 3.0)
    w = w_idx_date.reindex(sel).astype(float).fillna(0.0)

    is_core = w >= core_threshold
    tilt = np.where(is_core.to_numpy(), float(tilt_core), float(tilt_sat))
    factor = np.exp(tilt * z.to_numpy())
    raw = pd.Series(w.to_numpy() * factor, index=sel, dtype=float).clip(lower=0.0)

    if raw.sum() <= 0:
        raw = w.clip(lower=0.0)
    if raw.sum() <= 0:
        raw = pd.Series(1.0 / len(sel) if sel else 0.0, index=sel, dtype=float)
    return raw


# ===========================================================================
# Helpers de projection
# ===========================================================================
def _project_box_sum(raw: pd.Series, lo: pd.Series, hi: pd.Series,
                     target: float, tol: float = 1e-10,
                     max_iter: int = 1000) -> pd.Series:
    """Projette ``raw`` sur ``{lo <= x <= hi, sum(x) = target}``.

    Clamp initial puis redistribution iterative du deficit/exces proportionnelle
    au slack restant ``(hi - x)`` ou ``(x - lo)`` jusqu'a convergence.
    """
    x = raw.astype(float).clip(lower=lo, upper=hi)
    if not np.isfinite(target):
        return x
    for _ in range(max_iter):
        diff = float(target) - float(x.sum())
        if abs(diff) <= tol:
            break
        free = (hi - x) if diff > 0 else (x - lo)
        free_sum = float(free.sum())
        if free_sum <= tol:
            break
        x = (x + diff * free / free_sum).clip(lower=lo, upper=hi)
    return x


def _project_global(raw: pd.Series, lo: pd.Series, hi: pd.Series,
                    target: float = 1.0, tol: float = 1e-10,
                    max_iter: int = 1000) -> pd.Series:
    """Projection globale sur le simplexe borne (sans contrainte sectorielle)."""
    return _project_box_sum(raw, lo, hi, target, tol, max_iter)


def _enforce_top10(x: pd.Series, lo: pd.Series, hi: pd.Series, sec: pd.Series,
                   max_top10: Optional[float], iters: int = 2000,
                   tol: float = 1e-13) -> pd.Series:
    """Contraint la somme des 10 plus gros poids en redistribuant *intra-secteur*.

    On abaisse iterativement le poids des plus grosses lignes (jamais sous ``lo``)
    et on redistribue le surplus aux autres titres du **meme secteur** disposant
    de slack (sous ``hi``). Cette operation preserve exactement la somme de chaque
    secteur, la somme globale et toutes les bornes actives ``[lo, hi]``.

    La redistribution est proportionnelle au slack restant des receveurs : elle
    arrose en priorite les lignes sous-ponderees, ce qui abaisse reellement la
    concentration. Une ponderation uniforme sur un secteur domine par quelques
    mega-caps (ici l'IT) converge vers une queue de poids plus faible et un
    Top10 nettement inferieur a l'indice.
    """
    if max_top10 is None or not np.isfinite(max_top10):
        return x
    x = x.astype(float).clip(lower=lo, upper=hi).copy()

    # Pre-groupage des titres par secteur.
    groups: dict = {}
    for t in x.index:
        groups.setdefault(sec.get(t), []).append(t)

    room_all = (x - lo).clip(lower=0.0)
    slack_all = (hi - x).clip(lower=0.0)

    for _ in range(iters):
        nlarg = x.nlargest(10)
        excess = float(nlarg.sum()) - float(max_top10)
        if excess <= tol:
            break
        top_set = set(nlarg.index)

        # Capacite de transfert intra-secteur : min(room des donneurs du Top10,
        # slack des receveurs hors Top10).
        cap: dict = {}
        for s, members in groups.items():
            don = [t for t in members if t in top_set]
            recv = [t for t in members if t not in top_set]
            d_room = float(room_all.reindex(don).sum()) if don else 0.0
            r_slack = float(slack_all.reindex(recv).sum()) if recv else 0.0
            cap[s] = min(d_room, r_slack)
        total_cap = float(sum(cap.values()))
        if total_cap <= tol:
            break

        # Transfert agrege par secteur : evite les oscillations du transfert
        # titre par titre et converge vers la concentration minimale realisable.
        take = min(excess, total_cap)
        for s, c in cap.items():
            if c <= tol:
                continue
            amt = take * (c / total_cap)
            members = groups[s]
            don = [t for t in members if t in top_set]
            recv = [t for t in members if t not in top_set]
            d_room = room_all.reindex(don)
            r_slack = slack_all.reindex(recv)
            dsum = float(d_room.sum())
            rsum = float(r_slack.sum())
            if dsum <= tol or rsum <= tol:
                continue
            amt = min(amt, dsum, rsum)
            if amt <= tol:
                continue
            x.loc[don] = x.reindex(don).to_numpy() - amt * d_room.to_numpy() / dsum
            x.loc[recv] = x.reindex(recv).to_numpy() + amt * r_slack.to_numpy() / rsum
            # Mise a jour incrementale des rooms/slacks pour ce secteur.
            room_all.loc[don] = (x.reindex(don) - lo.reindex(don)).clip(lower=0.0)
            slack_all.loc[recv] = (hi.reindex(recv) - x.reindex(recv)).clip(lower=0.0)

    return x.clip(lower=lo, upper=hi)


# ===========================================================================
# 3) Projection sur le simplexe borne (par secteur)
# ===========================================================================
def project_weights(raw: pd.Series,
                    w_idx_date: pd.Series,
                    sector_target: Optional[pd.Series],
                    sector_map: Optional[pd.Series],
                    dw_min: float = config.DEFAULT_DW_MIN,
                    dw_max: float = config.DEFAULT_DW_MAX,
                    w_max: float = config.DEFAULT_W_MAX,
                    max_top10: Optional[float] = None,
                    sector_band: Optional[float] = None,
                    tol: float = 1e-10,
                    max_iter: int = 1000) -> pd.Series:
    """Projette les poids bruts sur les contraintes de risque.

    Bornes par titre : ``lo_i = max(0, w_idx_i + dw_min)`` et
    ``hi_i = min(w_max, w_idx_i + dw_max)``. Si ``sector_target`` est fourni (les
    poids GICS de l'indice EGUS du trimestre), la somme des poids de chaque
    secteur est projetee vers la cible indicielle : dans l'intervalle
    ``[cible - sector_band, cible + sector_band]`` si ``sector_band`` est fourni
    (la cible indicielle exacte si ``sector_band=None``), au plus pres des bornes
    agregees du secteur si la bande est infaisable. Sinon on normalise au global a
    1. La contrainte de concentration Top10 est desactivee par defaut
    (``max_top10=None``) : elle n'est appliquee que si l'appelant fournit une
    valeur finie. Retourne une Series ``sum = 1`` et ``w >= 0``.
    """
    raw = pd.Series(raw).astype(float)
    idx = raw.index
    w = w_idx_date.reindex(idx).astype(float).fillna(0.0)

    lo = pd.Series(np.maximum(0.0, w.to_numpy() + float(dw_min)), index=idx)
    hi = pd.Series(np.minimum(float(w_max), w.to_numpy() + float(dw_max)), index=idx)
    hi = pd.Series(np.maximum(hi.to_numpy(), lo.to_numpy()), index=idx)

    raw_pos = raw.clip(lower=0.0)
    if raw_pos.sum() <= 0:
        raw_pos = lo.copy()
    if raw_pos.sum() <= 0:
        raw_pos = pd.Series(1.0, index=idx)

    # Sans pilotage sectoriel : simplexe borne global.
    if sector_target is None or sector_map is None:
        out = _project_global(raw_pos, lo, hi, target=1.0, tol=tol, max_iter=max_iter)
        out = _finalize(out, lo, hi, tol, max_iter)
    else:
        sec = pd.Series(sector_map).reindex(idx)
        band = float(sector_band) if sector_band is not None else None
        targets = {s: float(sector_target.loc[s]) for s in sector_target.index}
        out = pd.Series(0.0, index=idx, dtype=float)
        for s, sub in sec.groupby(sec).groups.items():
            rows = pd.Index(sub)
            r = raw_pos.reindex(rows)
            l = lo.reindex(rows)
            h = hi.reindex(rows)
            lo_s, hi_s = float(l.sum()), float(h.sum())
            if pd.isna(s) or s not in targets:
                target_s = lo_s
            elif band is None:
                # Cible indicielle exacte, au plus pres des bornes du secteur.
                target_s = float(np.clip(targets[s], lo_s, hi_s))
            else:
                # Bande autorisee autour de la cible : on laisse le tilt
                # s'exprimer tant qu'il reste dans [cible-band, cible+band].
                feas_lo = max(lo_s, targets[s] - band)
                feas_hi = min(hi_s, targets[s] + band)
                if feas_lo > feas_hi:
                    target_s = float(np.clip(targets[s], lo_s, hi_s))
                else:
                    target_s = float(np.clip(float(r.sum()), feas_lo, feas_hi))
            out.loc[rows] = _project_box_sum(r, l, h, target_s, tol, max_iter).to_numpy()
        out = _finalize(out, lo, hi, tol, max_iter)
        if band is not None:
            # IPF : apres la normalisation globale, on ramene chaque secteur hors
            # bande dans l'intervalle autorise puis on renormalise.
            out = _enforce_sector_band(out, lo, hi, sec, targets, band, tol, max_iter)

    # Contrainte de concentration (Top 10), redistribution intra-secteur.
    if sector_map is not None and max_top10 is not None and np.isfinite(max_top10):
        sec_all = pd.Series(sector_map).reindex(idx)
        out = _enforce_top10(out, lo, hi, sec_all, max_top10, tol=tol)
        # La redistribution preserve la somme ; on re-projette malgre tout sur le
        # simplexe borne pour garantir sum=1 ET bornes exactes (jamais de simple
        # division qui ferait sortir les poids de [lo, hi]).
        out = _finalize(out, lo, hi, tol, max_iter)
    return out.rename(None)


def _finalize(x: pd.Series, lo: pd.Series, hi: pd.Series,
              tol: float, max_iter: int) -> pd.Series:
    """Restaure la somme a 1 tout en respectant les bornes (au plus pres)."""
    x = x.clip(lower=lo, upper=hi)
    total = float(x.sum())
    if abs(total - 1.0) > tol:
        x = _project_box_sum(x, lo, hi, 1.0, tol, max_iter)
    x = x.clip(lower=0.0)
    # Garde-fou numerique : renormalisation infime sans casser les bornes.
    total = float(x.sum())
    if total > 0 and abs(total - 1.0) > tol:
        x = x / total
        x = x.clip(lower=lo, upper=hi)
    return x.rename(None)


def _enforce_sector_band(x: pd.Series, lo: pd.Series, hi: pd.Series,
                         sec: pd.Series, targets: dict, band: float,
                         tol: float, max_iter: int, iters: int = 50) -> pd.Series:
    """Ramene chaque secteur dans ``[cible - band, cible + band]`` (IPF).

    Apres la projection sectorielle et la normalisation globale, un secteur peut
    deriver hors bande. On le re-projette alors sur la borne de bande la plus
    proche (bornee par les poids actifs agreges) puis on renormalise a 1 ;
    l'operation preserve les bornes par titre. Quelques iterations suffisent.
    """
    x = x.clip(lower=lo, upper=hi)
    for _ in range(iters):
        sec_sum = x.groupby(sec).sum()
        changed = False
        for s, base in targets.items():
            if s not in sec_sum.index:
                continue
            cur = float(sec_sum.loc[s])
            if cur > base + band + tol:
                tgt = base + band
            elif cur < base - band - tol:
                tgt = base - band
            else:
                continue
            rows = pd.Index(sec.index[sec == s])
            if len(rows) == 0:
                continue
            l = lo.reindex(rows)
            h = hi.reindex(rows)
            tgt = float(np.clip(tgt, float(l.sum()), float(h.sum())))
            x.loc[rows] = _project_box_sum(
                x.reindex(rows), l, h, tgt, tol, max_iter).to_numpy()
            changed = True
        x = _finalize(x, lo, hi, tol, max_iter)
        if not changed:
            break
    return x


# ===========================================================================
# Orchestration
# ===========================================================================
def build_portfolio(date,
                    panel_date: pd.DataFrame,
                    w_idx_date: pd.Series,
                    sector_target: Optional[pd.Series] = None,
                    sector_map: Optional[pd.Series] = None,
                    n: int = config.DEFAULT_N,
                    buffer_rule: bool = True,
                    prev_holdings: Optional[Iterable[str]] = None,
                    dw_min: float = config.DEFAULT_DW_MIN,
                    dw_max: float = config.DEFAULT_DW_MAX,
                    w_max: float = config.DEFAULT_W_MAX,
                    core_threshold: float = CORE_THRESHOLD,
                    tilt_core: float = 0.35,
                    tilt_sat: float = 0.70,
                    max_top10: Optional[float] = None,
                    sector_band: Optional[float] = None) -> pd.Series:
    """Construit le portefeuille d'une date : selection + tilt + projection."""
    if sector_map is None:
        sector_map = _get_sector_map()

    sel = select_titles(panel_date, w_idx_date, prev_holdings, n, buffer_rule,
                        core_threshold=core_threshold, sector_target=sector_target,
                        dw_max=dw_max, w_max=w_max)
    raw = tilt_weights(sel, panel_date["Z_Composite"], w_idx_date,
                       tilt_core=tilt_core, tilt_sat=tilt_sat,
                       core_threshold=core_threshold)
    weights = project_weights(raw, w_idx_date, sector_target, sector_map,
                              dw_min=dw_min, dw_max=dw_max, w_max=w_max,
                              max_top10=max_top10, sector_band=sector_band)
    return weights


def build_all_portfolios(params: Optional[dict] = None,
                         panel: Optional[pd.DataFrame] = None,
                         index_weights: Optional[pd.DataFrame] = None,
                         index_sector_w: Optional[pd.DataFrame] = None,
                         score_series: Optional[pd.Series] = None
                         ) -> Dict[pd.Timestamp, pd.Series]:
    """Construit les portefeuilles sur toutes les dates de rebalancement.

    ``params`` (tous optionnels, defauts = :data:`DEFAULT_PARAMS`) :
    ``n_titles``, ``sector_band``, ``sector_pilot``, ``dw_min``, ``dw_max``,
    ``w_max``, ``buffer_rule``, ``max_top10`` (``None`` = aucune borne Top10),
    ``weights`` (dict acceleration/momentum/quality).

    ``n_titles`` doit appartenir a ``[config.N_MIN, config.N_MAX]`` = ``[40, 50]`` ;
    une valeur hors plage leve une ``ValueError`` explicite.

    Retourne ``{date: Series(poids, index=ticker)}`` (``sum = 1``, ``w >= 0``).
    """
    p = dict(DEFAULT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if k in DEFAULT_PARAMS or v is not None})
    # Le bloc "weights" est fusionne partiellement.
    weights = dict(DEFAULT_PARAMS["weights"])
    if params and isinstance(params.get("weights"), dict):
        weights.update(params["weights"])
    p["weights"] = weights

    n_int = int(p["n_titles"])
    if not (config.N_MIN <= n_int <= config.N_MAX):
        raise ValueError(
            f"n_titles={n_int} hors plage autorisee "
            f"[{config.N_MIN}, {config.N_MAX}]."
        )
    n = n_int
    sector_band = float(np.clip(float(p["sector_band"]), 0.0, 0.05))
    sector_pilot = bool(p["sector_pilot"])
    dw_min = float(p["dw_min"])
    dw_max = float(p["dw_max"])
    w_max = float(p["w_max"])
    buffer_rule = bool(p["buffer_rule"])
    max_top10 = p.get("max_top10", None)
    if max_top10 is not None:
        max_top10 = float(max_top10)

    if panel is None:
        panel = factors.build_factor_panel()
    panel = panel.copy()

    # Recalcul du Z_Composite selon les 3 poids piliers choisis dans l'UI.
    wa = float(weights.get("acceleration", config.DEFAULT_W_ACCEL))
    wm = float(weights.get("momentum", config.DEFAULT_W_MOMENTUM))
    wq = float(weights.get("quality", config.DEFAULT_W_QUALITY))
    if score_series is not None:
        # Colonne de score de substitution (ex. Z_GScore de l'onglet de
        # recherche) : on court-circuite la combinaison des 3 piliers tout en
        # reutilisant toute la construction Core + Satellites.
        ser = pd.Series(score_series)
        if not isinstance(ser.index, pd.MultiIndex):
            ser.index = pd.MultiIndex.from_arrays(
                [ser.index.get_level_values(0), ser.index.get_level_values(1)],
                names=["date", "ticker"])
        panel["Z_Composite"] = ser.reindex(panel.index)
    else:
        panel["Z_Composite"] = factors.composite_weights_apply(panel, wa, wm, wq)

    dates = sorted(pd.Index(panel.index.get_level_values(0)).unique())
    if index_weights is None:
        index_weights = dl.get_index_weights(dates)

    if sector_pilot:
        if index_sector_w is None:
            index_sector_w = dl.get_index_sector_weights(dates)
        index_sector_w = index_sector_w.copy()
        index_sector_w.index = pd.to_datetime(index_sector_w.index).normalize()
        # Cibles sectorielles = poids GICS de l'indice EGUS du trimestre
        # (portefeuille sector-neutre : plus aucune dependance macro). Le
        # `sector_band` borne la deviation autorisee autour de ces cibles dans la
        # projection IPF (cf. `build_portfolio` / `project_weights`).
        sector_targets = index_sector_w.reindex(dates).fillna(0.0)
    else:
        sector_targets = None

    sector_map = _get_sector_map()

    out: Dict[pd.Timestamp, pd.Series] = {}
    prev: Optional[List[str]] = None
    for d in dates:
        d = _to_ts(d)
        panel_date = panel.loc[d]
        w_idx = _index_weight_row(index_weights, d).reindex(panel_date.index).fillna(0.0)
        st = (sector_targets.loc[d] if (sector_targets is not None and d in sector_targets.index)
              else None)
        w = build_portfolio(d, panel_date, w_idx, sector_target=st,
                            sector_map=sector_map, n=n, buffer_rule=buffer_rule,
                            prev_holdings=prev, dw_min=dw_min, dw_max=dw_max,
                            w_max=w_max, max_top10=max_top10,
                            sector_band=sector_band)
        out[d] = w
        prev = list(w[w > 0].index)
    return out
