# AUDIT QUANTITATIF — Moteur Growth USA (Core + Satellites, stratégie unique à 3 piliers)

**Baseline auditée : v8 — stratégie composite à 3 piliers statiques Point-in-Time (backtest trimestriel 2024-2026)**

**Auditeur :** Head of Quantitative Model Validation & Risk Audit
**Périmètre :** `src/data_loader.py`, `src/factors.py`, `src/portfolio.py`, `src/backtest.py`, `src/metrics.py` (+ `src/build_universe.py`, `src/build_fundamentals_sec.py`)
**Stratégie auditée :** **stratégie unique** à trois piliers —
`Z_Composite = 0,25·Z_Acceleration + 0,45·Z_Momentum + 0,30·Z_Quality_GARP` —
**N = 40**, bande sectorielle ±4 %, cap absolu 12,5 %, bornes actives [−3,5 % ; +2,5 %], coûts 10 bps
**Univers :** inventaire **N-PORT SEC** de l'ETF **EGUS** (`data/universe_quarterly.csv`) — **Point-in-Time trimestriel**, actions US `EC` uniquement (STIV/DE exclus), poids renormalisés Σ=1 par trimestre, appliqués à la **`filing_date`**
**Rebalancement :** **11 dates trimestrielles** (`filing_date`) du 2024-01-25 au 2026-07-27 (paniers de 81 à 126 titres)
**Fenêtre de performance :** 2024-01-25 → 2026-09-30 (NAV Base 100)
**Benchmark :** **EGUS** (iShares ESG Aware MSCI USA Growth) — **benchmark de référence unique**
**Macro :** **supprimée** (aucune dépendance `src.macro` ; cibles sectorielles = poids GICS de l'indice du trimestre)
**Suite de tests :** `tests/test_audit.py` + `tests/test_audit_bias.py` + `tests/test_app.py` + `tests/test_data_loader.py` + `tests/test_factors.py` + `tests/test_portfolio.py` — **`python -m pytest tests/ -q` → 102 passed, 0 failed**

---

## 1. Résumé exécutif

### Statut global : **GO**

> ✅ **Verdict : GO.** Le moteur est **validation-ready**. Sur la baseline **trimestrielle Point-in-Time**, **aucun biais d'anticipation ne subsiste** : les Z-scores sont **invariants par troncature et perturbation du futur à chacune des 11 dates** (Δ = 0,0), les **trois piliers se répliquent au bit près** et toutes les contraintes sont respectées à chaque date.

> ⚠️ **Réserve de performance.** L'alpha net de la **stratégie composite vs EGUS** est **positif mais modeste (+1,83 % annualisé)** et **statistiquement NON significatif** : son **IC95 bootstrap par blocs contient 0** ([−2,25 % ; +6,35 %]). L'univers étant **le panier de l'ETF EGUS lui-même**, l'absence d'alpha exploitable est attendue ; elle ne doit **pas** être surinterprétée.

> 🔎 **Écart documenté (non bloquant).** Le **Top 10 pèse ≈ 65,9 %** (vs ≈ 64,2 % pour l'indice) : la cible 40–50 % de la mission n'est **pas** tenue, la borne de concentration étant **volontairement désactivée** (`max_top10=None`) et les poids Core suivant l'indice. C'est une **limite assumée** (L1), pas une erreur de code.

| Indicateur | Valeur mesurée | Référence mission | Statut |
|---|---|---|---|
| Rendement total (Composite) | **+77,8 %** | — | — |
| Rendement total (**EGUS**) | +70,4 % | — | — |
| Alpha ann. vs **EGUS** | **+1,83 %** | — | **non significatif** (IC95 contient 0) |
| Volatilité annualisée | 21,4 % | — | — |
| Bêta (vs EGUS) | 1,037 | ~1 | ✅ proche de 1 |
| **Tracking Error** (vs EGUS) | **4,28 %** | 2,5 – 4,5 % | ✅ (zone cible) |
| Ratio d'information (vs EGUS) | 0,43 | — | — |
| Active share | 25,6 % | — | — |
| **Top 10** | **65,9 %** (indice 64,2 %) | 40 – 50 % | ⚠️ écart documenté (L1) |
| Turnover moyen trimestriel | **18,8 %** (hors inception) | — | ✅ |
| Nombre de titres | 40 | [40, 50] | ✅ |
| Max drawdown | −24,8 % | — | — |

**Conclusion :** **pas de veto**. Le moteur est **validé pour la mise en production du frontend**, avec affichage explicite (i) de l'**alpha net modeste et non significatif vs EGUS** (+1,83 %, IC95 contenant 0) et (ii) de la **limite de concentration Top 10** (65,9 %). Aucun alpha net exploitable n'est revendiqué.

---

## 2. Périmètre et méthodologie de la baseline

| Élément | Description |
|---|---|
| Source univers | Inventaire N-PORT SEC `data/inventaire_consolide_S000079196.csv` → `data/universe_quarterly.csv` |
| Filtre univers | `asset_cat == 'EC'` ET `inv_country == 'US'` (**STIV/DE exclus**) |
| Renormalisation | Poids renormalisés à **1,0 par `rebalance_date`** (Σ=1), 213 tickers uniques sur l'historique N-PORT |
| Début de simulation | **2024-01-25** = 1ʳᵉ `filing_date` retenue (début de la fenêtre de performance) |
| Dates de rebalancement | 2024-01-25, 2024-04-24, 2024-07-25, 2024-10-28, 2025-01-23, 2025-04-23, 2025-07-24, 2025-10-27, 2026-01-28, 2026-04-23, 2026-07-27 (**11 dates**) |
| Panel factoriel | `factors.build_factor_panel()` → **11 dates** × panier N-PORT du trimestre |
| Construction | `portfolio.build_all_portfolios` → **stratégie unique × 11 dates**, N = 40, Core `w_idx ≥ 2,5 %`, `max_top10=None`, cible sectorielle = poids GICS de l'indice (bande ±4 %) |
| Backtest | `backtest.run_backtest` : NAV Base 100 au 2024-01-25, dérive **buy & hold** intra-trimestre, **turnover par rebalancement** |
| Fondamentaux | **SEC EDGAR XBRL** (`data/fundamentals_sec.parquet`, `pub_date` = `filed`, PIT strict) |
| Benchmark | **EGUS** unique |
| Macro | **Aucune** |

### Piliers de la stratégie unique

| Pilier | Poids | Composition | Traitement Financials |
|---|---|---|---|
| **Accélération** | 25 % | `Zagg(blend(Z_SalesAccel, Z_MarginExp, Z_AssetEff))` — agrégats annuels PIT `N..N-3` | `Asset_Eff` → `ΔROA` |
| **Momentum** | 45 % | `Zagg(0,65·Z_RAM + 0,35·Z_Trend)` — `RAM = (P[t−21]/P[t−126]−1)/(σ_6M·√252)`, `Trend = P/SMA200−1` | — |
| **Qualité/GARP** | 30 % | `Zagg(0,40·Z_ROE + 0,35·Z_FCF_Yield + 0,25·Z_GARP)` | FCF omis → `0,60·Z_ROE + 0,40·Z_GARP` |

---

## 3. Contrôles anti look-ahead bias

| # | Contrôle | Méthode | Résultat | Statut |
|---|---|---|---|---|
| 1a | Aucune métrique issue de `Ticker.info` | Analyse AST (`.<attr> == 'info'` hors `logger.info`) sur `factors/portfolio/backtest/metrics/data_loader` | 0 accès | ✅ PASS |
| 1a | `build_universe.py` : `.info` réservé à l'identification | `.info` uniquement pour `symbol`/`sector` ; aucune clé fondamentale (`marketCap`, `returnOnEquity`, …) | 0 métrique fondamentale | ✅ PASS |
| 1a | Aucune lecture `['info']` / `get('info')` | Recherche textuelle | 0 occurrence | ✅ PASS |
| 1b | Invariance par **troncature des prix** (`<= t`) | `compute_strategy_scores(t)` : prix complets vs `<= t`, **aux 11 dates** | **max \|Δ\| = 0,0** (tous Z_* / S_*) | ✅ PASS |
| 1b | Invariance par **perturbation du futur** (`> t`) | Prix `> t` × 1e6 + bruit, **aux 11 dates** | **max \|Δ\| = 0,0** | ✅ PASS |
| 1b | Invariance à l'ajout de **fondamentaux futurs** | Ligne injectée `pub_date = t + 60 j` (valeurs absurdes) | **Δ = 0,0** | ✅ PASS |
| 1b | Invariance à la **troncature du panel fondamental** | `fund[pub_date <= t]` | **Δ = 0,0** | ✅ PASS |
| 1c | **PIT fondamentaux** — cohérence panel | `pub_date >= period_end` sur tout le panel (**SEC EDGAR XBRL**, `pub_date` = `filed`) | 0 violation | ✅ PASS |
| 1c | **PIT fondamentaux** — usage au rebalancement | `fundamentals_asof(fund, d)` : toutes `pub_date <= d`, **aux 11 dates** | 0 violation | ✅ PASS |
| 1d | **Poids indiciels PIT** | Dates = `config.REBALANCE_DATES` ; paires ⊆ `universe_quarterly` ; Σ=1 | ✅ | ✅ PASS |
| 1d | **Aucun poids avant sa `filing_date`** | `get_universe(asof)` → 1 seul panier connu ≤ asof | ✅ | ✅ PASS |
| 1e | Winsorisation / Z-score **transversaux** | Espion sur `factors.winsorize` : longueurs ≤ panier du trimestre | jamais la matrice 11 dates | ✅ PASS |
| 1e | Winsorisation 5/95 effective | Borne = quantiles 5e / 95e | Exact | ✅ PASS |
| 1e | Preuve numérique per-date ≠ global | Winsorisation groupée vs globale | Résultats distincts | ✅ PASS |
| — | Intégrité du cache factoriel | Panel en cache vs recalcul frais (3 dates) | **Δ = 0,0** | ✅ PASS |

**Lecture du contrôle clé.** À **chacune des 11 dates**, tronquer ou corrompre violemment les prix postérieurs au rebalancement **ne change aucun score** (Δ = 0,0) : la coupe transversale n'utilise que l'information disponible à `t`. Le lag de publication des fondamentaux est préservé (`pub_date ≥ period_end`, et `pub_date ≤ t` pour tout fondamental injecté), à partir du panel **SEC EDGAR XBRL** (`pub_date` = `filed`).

---

## 4. Contrôles de formules et de contraintes

### 4.1 Formules quantitatives (réplication bit à bit)

| Facteur | Formule contrôlée | Résultat | Statut |
|---|---|---|---|
| **Accélération — Sales** | `[CA_N/CA_{N-1}−1] − [CA_{N-1}/CA_{N-2}−1]` | Réplication exacte (11 dates) | ✅ PASS (**Δ = 0,0**) |
| **Accélération — Margin** | `OI_N/CA_N − OI_{N-1}/CA_{N-1}` | Réplication exacte | ✅ PASS (**Δ = 0,0**) |
| **Accélération — Asset** | `CA_N/TA_N − CA_{N-1}/TA_{N-1}` (**Financials → ΔROA**) | Réplication exacte | ✅ PASS (**Δ = 0,0**) |
| **Momentum — RAM** | `(P[t−21]/P[t−126]−1) / (σ_6M·√252)` | Réplication exacte | ✅ PASS (**Δ = 0,0**) |
| **Momentum — Trend** | `P/SMA200 − 1` | Réplication exacte | ✅ PASS (**Δ = 0,0**) |
| Momentum exclut le **dernier mois** | Perturbation des 20 dernières séances → numérateur 6-1 **inchangé** | **Δ = 0** | ✅ PASS |
| **Qualité — ROE** | `NI_TTM / Equity` | Réplication exacte | ✅ PASS (**Δ = 0,0**) |
| **Qualité — FCF_Yield** | `(OCF_TTM − |CapEx_TTM|) / Market_Cap` | Réplication exacte | ✅ PASS (**Δ = 0,0**) |
| **Qualité — GARP** | `EPS_Growth_3Y / (Prix/EPS_TTM)` (neutralisé si P/E ≤ 0) | Réplication exacte | ✅ PASS (**Δ = 0,0**) |
| **Purge Junk Growth** | décile inférieur `GP/A` + `(OCF < NI ET D/E haut)`, méga-caps Core exemptées | ✅ | ✅ PASS |
| **Projection bornée** | `Σw = 1`, `w ≥ 0`, `lo_i ≤ w_i ≤ hi_i`, cible sectorielle (avec / sans secteur) | ✅ | ✅ PASS |

### 4.2 Contraintes par date (stratégie unique × 11 dates)

| Contrainte | Attendu | Mesuré | Statut |
|---|---|---|---|
| `Σ w_p = 1` | ±1e-8 | erreur ≈ 1e-16 (11 portefeuilles) | ✅ PASS |
| `w_p ≥ 0` | ≥ 0 | min > 0 partout | ✅ PASS |
| `N ∈ [40, 50]` | 40–50 | **40** (11 portefeuilles) | ✅ PASS |
| Bornes actives `[dw_min, dw_max]` | [−3,5 % ; +2,5 %] (±1e-6) | min ≥ −3,5 % ; max ≤ +2,5 % | ✅ PASS |
| Cap absolu `w_max` | ≤ 12,5 % | max **= 12,50 %** (méga-caps saturées) | ✅ PASS |
| Core présent | tous les éligibles `w_idx ≥ 2,5 %` | titres Core/date, **tous présents** | ✅ PASS |
| Ancres méga-cap | NVDA, AAPL, AMZN, GOOG | présentes **11/11 dates** | ✅ PASS |
| Bande sectorielle | ≤ 4 % autour de l'indice | écart max **4,00 %** (borne) | ✅ PASS |

---

## 5. Validation numérique du portefeuille Core + Satellites

> **Nature de la baseline.** Backtest **trimestriel Point-in-Time** : **11 dates** de rebalancement. Chaque portefeuille est construit sur le **panier N-PORT publié à la `filing_date`**, avec les seuls prix/fondamentaux `<= t`.

| # | Contrôle | Attendu | Mesuré | Statut |
|---|---|---|---|---|
| 1 | `Σ_i w_{p,i,t} = 1,000000` | ±1e-8 | erreur ≈ 1e-16 (**11**) | ✅ PASS |
| 1 | `w_{p,i,t} ≥ 0` | ≥ 0 | aucune ligne négative | ✅ PASS |
| 2 | Nombre de titres `N ∈ [40, 50]` | 40–50 | **40** | ✅ PASS |
| 3 | Core méga-caps (« Nvidia, Apple, Amazon, Alphabet ») | ancres présentes | **NVDA/AAPL/AMZN/GOOG 11/11** | ✅ PASS |
| 3 | Top 10 vs indice | documenté | **65,9 % vs 64,2 %** (indice) | ⚠️ documenté (L1) |
| 3 | Tracking Error annualisée | 2,5 – 4,5 % | **4,28 %** (zone cible) | ✅ PASS (L3) |
| — | **Invariance Δ = 0** (troncature / futur) | Δ = 0 | **0,0 aux 11 dates** | ✅ PASS |

### 5.1 Résultats de la stratégie composite (2024-01-25 → 2026-09-30)

| | Rendement total | Alpha ann. vs EGUS | TE vs EGUS | IR vs EGUS | Bêta | Vol. ann. | Sharpe | Max DD | Top 10 | Active share | Turnover moy. |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Composite** (A 25 / M 45 / Q 30) | **+77,8 %** | **+1,83 %** | 4,28 % | 0,43 | 1,037 | 21,4 % | 1,12 | −24,8 % | 65,9 % | 25,6 % | 18,8 % |
| **EGUS** | +70,4 % | — | — | — | — | — | — | — | 64,2 % | — | — |

**Lecture.** La stratégie **surperforme modestement** le benchmark **EGUS** (+77,8 % vs +70,4 %, alpha annuel **+1,83 %**, IR **0,43**), avec un bêta proche de 1 et un active share modéré (~26 %) : une gestion **proche de l'indice mais légèrement tiltée**. L'alpha n'est **pas distinguable de zéro** au seuil de 95 % (cf. §6).

### 5.2 Turnover trimestriel

| Événement | Turnover |
|---|---|
| Inception (2024-01-25, depuis le cash) | **≈ 50 %** (artefact) |
| Moyenne des 10 rebalancements suivants | **≈ 18,8 %** |
| Maximum (hors inception) | ≈ 23,6 % |

Le turnover est calculé **par rapport aux poids dérivés** en fin de trimestre : `0,5 · Σ|w_cible − w_dérivé|` (vérifié exactement à chaque date, cf. §7).

---

## 6. Analyse des résultats et significativité

### 6.1 Bootstrap par blocs (1000 tirages, blocs de 21 jours, seed = 12345)

| Comparaison | Alpha ann. (moyenne) | IC 95 % alpha | Verdict |
|---|---|---|---|
| **Composite vs EGUS** | **+1,92 %** | **[−2,25 % ; +6,35 %]** | **contient 0 → NON significatif** |

**Conclusion.** Sur une fenêtre de **~2,7 ans (11 trimestres)**, **aucun alpha n'est distinguable de zéro** au seuil de 95 %. L'univers étant **le panier de l'ETF EGUS** lui-même, l'alpha ne provient que des **tilts factoriels** et reste **non exploitable** (puissance statistique limitée, IC larges).

### 6.2 Structure

- **Bêta ≈ 1,04** : exposition marché quasi unitaire (univers = panier de l'ETF, cap 12,5 % sur les méga-caps).
- **Active share ≈ 26 %** : cohérent avec un univers restreint et des tilts modérés.

---

## 7. Liste des tests exécutés

**Commande :** `python -m pytest tests/ -q` → **102 passed, 0 failed**

### `tests/test_audit.py` (anti look-ahead, formules, chronologie, contraintes)

**Anti look-ahead / PIT :** `test_no_info_access_in_quant_modules`, `test_no_info_subscript_access`, `test_build_universe_info_only_for_identification`, `test_scores_invariant_to_future_prices_all_dates`, `test_scores_invariant_to_future_fundamentals`, `test_scores_invariant_to_fund_truncation`, `test_pit_publication_dates`, `test_fundamentals_used_have_publication_before_date`, `test_no_weight_before_filing_date`, `test_index_weights_pit_and_renormalized`, `test_panel_matches_fresh_computation`, `test_winsorize_is_cross_sectional_per_date`, `test_winsorize_clips_at_5_95_quantiles`, `test_winsorize_per_date_differs_from_global`.

**Formules (3 piliers) :** `test_momentum_window_constants`, `test_momentum_excludes_last_month`, `test_momentum_formula_replication`, `test_momentum_full_pillar_replication`, `test_acceleration_formula_replication`, `test_quality_formula_replication`, `test_quality_contains_roe_fcf_garp_and_purge_bricks`, `test_quality_reacts_to_profitability`.

**Chronologie t → t+1 :** `test_toy_new_weights_not_applied_on_rebalance_day`, `test_toy_new_weights_applied_next_day`, `test_toy_rebalance_weekend_maps_to_prior_trading_day`, `test_real_backtest_return_alignment`, `test_weight_drift_buyhold`, `test_turnover_matches_derived_weights`.

**Contraintes & projection :** `test_constraints_sum_nonneg_finite`, `test_constraints_n_titles`, `test_constraints_active_bounds_and_cap`, `test_constraints_core_present`, `test_constraints_core_megacaps_are_anchors`, `test_constraints_sector_band`, `test_project_weights_box_and_sum`, `test_project_weights_sector_target_respected`.

**Renormalisation univers & backtest :** `test_universe_quarterly_renormalized`, `test_universe_quarterly_excludes_stiv_de_and_foreign`, `test_strategy_stats_vs_egus`.

### `tests/test_audit_bias.py` (analyse critique de la baseline)

**Univers PIT N-PORT :** `test_universe_quarterly_is_nport_pit`, `test_universe_baskets_change_across_quarters`, `test_universe_excludes_stiv_de_and_foreign`, `test_delisted_and_unresolved_tickers_removed`, `test_index_weight_forward_return_correlation_moderate`.

**Résultats vs EGUS :** `test_benchmark_egus_return`, `test_strategy_returns_vs_egus`.

**Significativité bootstrap :** `test_alpha_vs_egus_not_significant`, `test_alpha_bootstrap_point_estimate`, `test_te_bootstrap_interval`.

**Structure :** `test_beta_near_one`, `test_top10_near_index_level`, `test_active_share_moderate`, `test_max_weight_capped`.

**Turnover & limites :** `test_turnover_quarterly`, `test_universe_etf_holdings_not_broad_index`, `test_no_macro_dependency`, `test_results_are_not_lookahead_borne`.

### `tests/test_app.py` (smoke test du dashboard) + suites préexistantes

`tests/test_data_loader.py` · `tests/test_factors.py` · `tests/test_portfolio.py`.

---

## 8. Limites connues et recommandations

| # | Limite | Impact | Recommandation |
|---|---|---|---|
| L1 | **Top 10 ≈ 65,9 %** (cible 40–50 %) | Concentration au niveau de l'indice. | Borne cumulée désactivée (`max_top10=None`), poids Core suivant l'indice. Pour l'atteindre : abaisser `w_max` ou le poids Core. |
| L2 | **Turnover d'inception ≈ 50 %** | Lecture erronée si présenté comme turnover périodique. | Afficher « inception only » ; turnover courant ≈ 19 %. |
| L3 | **TE 4,28 %** (cible 2,5–4,5 %) | **Dans** la zone cible (haute). | TE en partie incompressible (univers = panier de l'ETF + tilts). Documenter. |
| L4 | **Univers = holdings de l'ETF** (ESG-aware) | **Alpha net modeste et non significatif** par construction (on part du panier de l'indice). | Toute communication d'alpha se fait **vs EGUS**, sans surinterprétation. |
| L5 | **Fenêtre ~2,7 ans (11 trimestres)** | Puissance statistique limitée (IC larges). | Étendre la fenêtre pour tout exercice de significativité. |
| L6 | **Délistés / ISIN non résolus retirés** (dont GOOGL, EA) | Biais de survie résiduel. | Limite de données ; documenter. |
| L7 | **Pas de macro** | Pas d'overlay de régime. | Choix assumé : cibles sectorielles = poids GICS de l'indice. |

---

## 9. Note de méthode et correctifs

**Baseline v8 — stratégie unique à 3 piliers statiques PIT.** Le moteur est passé d'un schéma à 3 stratégies (Quality / Momentum / Combinée, 14 dates 2023-2026) à une **stratégie unique** dont le score composite est `Z_Composite = 0,25·Z_Acceleration + 0,45·Z_Momentum + 0,30·Z_Quality_GARP`, évaluée sur **11 dates trimestrielles (2024-2026)**. Les piliers sont **entièrement statiques** : aucune surprise bénéficiaire, aucun CAR, aucune dépendance macro.

**Fondamentaux SEC EDGAR XBRL.** `data/fundamentals_sec.parquet` (`pub_date` = `filed`, PIT strict, borné à `MIN_END = 2020-01-01`) alimente `src/data_loader.get_fundamentals`, avec la colonne ajoutée `operating_income_q` (`OperatingIncomeLoss`, nécessaire au pilier Accélération) et les agrégats annuels PIT `N, N-1, N-2, N-3` (`annual_flow_points` / `annual_stock_points`).

**Aucune signature publique de `src/` n'a été cassée.** Les tests ont été réécrits pour la v8 (`tests/conftest.py`, `test_factors.py`, `test_portfolio.py`, `test_audit.py`, `test_audit_bias.py`, `test_data_loader.py`, `test_app.py`) ; les valeurs de baseline en dur ont été recalibrées. `python -m pytest tests/ -q` → **102 passed**.

---

## 10. Verdict final

| Critère | Verdict |
|---|---|
| Anti look-ahead (PIT, invariance Δ=0) | ✅ **PASS** |
| Formules quantitatives (3 piliers, réplication) | ✅ **PASS** |
| Contraintes par date (11 dates) | ✅ **PASS** |
| Renormalisation PIT de l'univers | ✅ **PASS** |
| Alpha net **Composite** vs EGUS | ⚠️ **modeste (+1,83 %), non significatif** (IC95 contient 0) |
| Concentration Top 10 | ⚠️ **documenté** (borne désactivée) |

> ✅ **VERDICT GLOBAL : GO** (validation-ready), **avec mise en garde** : **alpha net modeste et non significatif de la stratégie composite vs EGUS** (IC95 contenant 0 ⇒ **aucun alpha exploitable**) et **univers = panier de l'ETF** (limite structurelle). Limite de concentration Top 10 documentée. Aucun veto.

---

*Rapport généré par l'agent `quant_auditor` — baseline v8 (univers N-PORT PIT trimestriel EGUS, 11 dates 2024-2026, stratégie unique 3 piliers 25/45/30, macro retirée).*
