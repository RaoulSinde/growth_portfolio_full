# CONTRAT D'INTERFACE — Moteur Quant Growth USA

Toutes les dates sont des `pandas.Timestamp` (timezone-naive, normalisées à minuit).
Interpréteur : `python` (3.13) — packages déjà installés globalement (pandas, numpy, scipy, yfinance, openpyxl, pyarrow, plotly, streamlit, pytest).

## CONSTANTES GLOBALES (src/config.py)
```
UNIVERSE_CSV           = "data/universe.csv"            # DEPRECATED (univers statique)
INVENTORY_CSV          = "data/inventaire_consolide_S000079196.csv"  # source N-PORT SEC
UNIVERSE_QUARTERLY_CSV = "data/universe_quarterly.csv"  # source de verite PIT trimestrielle
SECTOR_MAP_CSV         = "data/sector_map.csv"          # ticker -> secteur GICS
CACHE_DIR              = "data/cache"
BENCHMARK_TICKER = "EGUS"                 # benchmark de reference unique
PRICE_START    = "2023-01-01"
PRICE_END      = "2026-09-30"
PERF_START     = "2024-01-25"             # = 1er filing_date retenu (debut de simulation)
PERF_END       = "2026-09-30"
SIM_END        = "2026-09-30"
REBALANCE_DATES = les 11 `filing_date` retenus (fenetre de test 2024-2026) :
    [2024-01-25, 2024-04-24, 2024-07-25, 2024-10-28, 2025-01-23,
     2025-04-23, 2025-07-24, 2025-10-27, 2026-01-28, 2026-04-23, 2026-07-27]
SECTORS        = 11 secteurs GICS
# Poids par defaut de la strategie unique (3 piliers statiques PIT) :
DEFAULT_W_ACCEL = 0.25 ; DEFAULT_W_MOMENTUM = 0.45 ; DEFAULT_W_QUALITY = 0.30
DEFAULT_N = 40 ; N_MIN = 40 ; N_MAX = 50
DEFAULT_SECTOR_BAND = 0.04 ; DEFAULT_DW_MIN = -0.035 ; DEFAULT_DW_MAX = 0.025
DEFAULT_W_MAX = 0.125 ; DEFAULT_TX_COST_BPS = 10.0
```

### Univers PIT trimestriel (data/universe_quarterly.csv)
Une ligne par `(rebalance_date, ticker)` ; `rebalance_date` = `filing_date` (date de
connaissance publique des poids N-PORT). Poids reels du trimestre (pas de derive),
renormalises a 1.0 par `rebalance_date`. Genere par `python -m src.build_universe`.
- Filtre : `asset_cat == 'EC'` ET `inv_country == 'US'` (exclut STIV/DE).
- Resolution ISIN->ticker via yfinance, cache `data/cache/isin_ticker.parquet`.
- Le CSV contient tout l'historique N-PORT disponible ; `config.REBALANCE_DATES`
  n'en retient que les **11 derniers trimestres** (fenetre de test 2024-2026).
- ISIN non resolus retires (GOOGL, EA, ...) : biais de survie residuel documente.

## src/data_loader.py  (agent: data_engineer)
Pipeline multi-dates PIT trimestriel. Aucune dependance macro (`get_macro` supprime,
`macro.parquet` retire). Source de verite = `data/universe_quarterly.csv`.
```python
get_rebalance_dates() -> list[pd.Timestamp]
    # les 11 `filing_date` triés (depuis config.REBALANCE_DATES).

get_universe(asof=None) -> pd.DataFrame
    # colonnes: rebalance_date, report_period, ticker, name, isin, cusip, weight.
    # asof=None -> tout l'historique ; sinon panier du dernier trimestre connu
    # (rebalance_date <= asof). Remplace `load_universe` en interne.

get_index_weights(dates=None) -> pd.DataFrame
    # Poids PIT REELS du trimestre (pas de derive), source universe_quarterly.
    # MultiIndex (date, ticker), colonne `weight`, Σ=1 par date. Si `dates` ne
    # contient qu'une date -> pd.Series (SingleIndex) indexee par ticker.

get_index_sector_weights(dates=None) -> pd.DataFrame
    # index=dates, colonnes=11 secteurs GICS, Σ=1 par date (agregation via
    # get_sector_map, renormalisation sur les secteurs connus).

get_sector_map() -> pd.Series
    # ticker -> secteur GICS (data/sector_map.csv). 100% de couverture (213/213).

get_prices(tickers=None, start=PRICE_START, end=PRICE_END) -> pd.DataFrame
    # Adj Close, index=DatetimeIndex trié, colonnes=tickers. Par defaut : union de
    # TOUS les tickers de universe_quarterly (2023-01 -> 2026-09-30). Tickers sans
    # donnees (ex. WBD) exclus. Cache data/cache/prices.parquet.

get_benchmark(start=PRICE_START, end=PRICE_END) -> pd.Series
    # série de prix du benchmark de référence (EGUS), nom "Benchmark" ; fallback
    # synthétique si vide/incomplet (<90% des jours ouvrés). Rétro-compatible.

get_benchmarks(start=PRICE_START, end=PRICE_END) -> dict[str, pd.Series]
    # {"EGUS": Series} — benchmark de référence unique. Cache par ticker dans
    # data/cache/benchmark_*.parquet.

load_universe(eligible=True) -> pd.DataFrame
    # DEPRECATED : ancien univers statique data/universe.csv (benchmark synthétique).

get_fundamentals(tickers=None) -> pd.DataFrame
    # PANEL POINT-IN-TIME (source SEC EDGAR XBRL, data/fundamentals_sec.parquet).
    # Une ligne par (ticker, period_end). Colonnes:
    #   ticker, period_end, pub_date (date de depot « filed », >= period_end),
    #   total_assets, stockholders_equity, total_debt, shares,
    #   net_income_q, gross_profit_q, operating_income_q, ocf_q, revenue_q,
    #   research_dev_q, capex_q, buyback_q, dividends_q
    # Les valeurs *_q sont trimestrielles (pas TTM) ; l'agregation annuelle PIT
    # (N, N-1, N-2, N-3) est faite dans factors.py via annual_flow_points /
    # annual_stock_points. AUCUNE donnee issue de ticker.info.

annual_flow_points(fund, col, tickers) -> pd.DataFrame
    # Agregat annuel PIT d'un FLUX : somme des 4 derniers trimestres (N), des 4
    # precedents (N1), etc. -> colonnes ["N","N1","N2","N3"]. `fund` deja filtre
    # (pub_date <= t) par l'appelant.

annual_stock_points(fund, col, tickers) -> pd.DataFrame
    # Agregat annuel PIT d'un STOCK : valeur ponctuelle au dernier period_end
    # (N), puis 4 trimestres plus tot (N1), etc. -> ["N","N1","N2","N3"].

fundamentals_asof(fund, asof) -> pd.DataFrame
    # pour chaque ticker, derniere ligne avec pub_date <= asof. index=ticker.

get_earnings_dates(tickers=None) -> pd.DataFrame
    # colonnes: ticker, earnings_date, surprise_pct (legacy ; NON utilise par le
    # modele v8, qui est purement statique). Conservé pour compatibilite.
```

## src/factors.py  (agent: quant_factor_macro)
Stratégie **unique** à trois piliers statiques Point-in-Time.
```python
# Z_Composite = w_accel*Z_Acceleration + w_mom*Z_Momentum + w_qual*Z_Quality_GARP

FACTOR_COLUMNS = ["Z_Acceleration", "Z_Momentum", "Z_Quality_GARP", "Z_Composite",
                  "S_SalesAccel", "S_MarginExp", "S_AssetEff",
                  "S_RAM", "S_Trend",
                  "S_ROE", "S_FCFYield", "S_GARP", "eligible"]

build_factor_panel(rebalance_dates=None) -> pd.DataFrame
    # index MultiIndex (date, ticker) sur les 11 dates ; Z_Composite avec les poids
    # par défaut (accel 0.25 / mom 0.45 / qual 0.30). Cache data/cache/factor_panel.parquet.

winsorize(s, lo=0.05, hi=0.95) -> pd.Series
zscore(s, by_sector=True, cap=None, sector=None) -> pd.Series
    # winsorisation 5/95 PUIS z-score (intra-sectoriel GICS si >= 5 titres, sinon
    # univers). cap optionnel = pondération par capitalisation.
_blend(pairs) -> pd.Series      # moyenne pondérée robuste aux NaN (renormalise les poids)
ttm(x) -> float                 # annualisation robuste : mean(>=1, <=4 derniers) * 4
purge_junk_growth(fund_asof, sector, mega_cap=None, decile=0.10) -> pd.Series(bool)
    # False si GP/Assets dans le décile inférieur intra-sectoriel,
    # OU (OCF < NetIncome ET Debt/Equity dans le décile supérieur intra-sectoriel).
    # Les méga-caps Core (mega_cap=True) restent éligibles.

compute_strategy_scores(date, prices, fund_asof, bench_returns, universe) -> pd.DataFrame
    # coupe transversale d'une date : toutes les colonnes de FACTOR_COLUMNS.
    # bench_returns conservé pour la signature (non utilisé : modèle statique).
composite_weights_apply(panel, w_accel, w_mom, w_qual) -> pd.Series   # renvoie Z_Composite
```

Constantes : `MOM_1M=21, MOM_6M=126, MOM_12M=252, VOL_WINDOW_6M=126, SMA_TREND=200,
MIN_VOL_OBS=60, SECTOR_MIN_N=5, PURGE_MIN_N=5, MEGA_CAP_THRESHOLD=0.025`.
Pilier 1 = `Zagg(blend(Z_SalesAccel, Z_MarginExp, Z_AssetEff))` (Financials → ΔROA) ;
Pilier 2 = `Zagg(0.65·Z_RAM + 0.35·Z_Trend)` ; Pilier 3 = `Zagg(blend(0.40·Z_ROE +
0.35·Z_FCFYield + 0.25·Z_GARP))` (Financials → 0.60·Z_ROE + 0.40·Z_GARP).

## src/portfolio.py  (agent: portfolio_constructor)
Aucune dépendance macro (`src.macro` supprimé) : les cibles sectorielles sont les
**poids GICS de l'indice EGUS du trimestre** (portefeuille sector-neutre, bande
`±sector_band`).
```python
select_titles(panel_date, w_idx_date, prev_holdings, n, buffer_rule,
              core_threshold=0.025, sector_target=None, ...) -> list[str]
    # Core = titres éligibles w_idx >= core_threshold (toujours inclus).
    # Satellites = répartis par secteur selon la cible puis top Z_Composite,
    # complétés jusqu'à N total. `prev_holdings` transmis entre trimestres.
    # buffer_rule: conserver un satellite déjà détenu si Z >= 60e centile de son secteur.
tilt_weights(sel, z, w_idx, tilt_core=0.35, tilt_sat=0.70) -> pd.Series
project_weights(raw, w_idx, sector_target, sector_map, dw_min, dw_max, w_max,
                max_top10=None, sector_band=None) -> pd.Series
    # projection sur le simplexe borné : bornes actives [dw_min,dw_max],
    # cap w_max, cibles sectorielles dans [cible-band, cible+band] (IPF).
build_portfolio(date, panel_date, w_idx_date, sector_target=None, ..., sector_band=None) -> pd.Series
build_all_portfolios(params, panel=None, index_weights=None, index_sector_w=None)
    -> dict[Timestamp, pd.Series]   # les 11 dates de rebalancement
    # n_titles ∈ [40, 50] (défaut 40), max_top10=None, buffer_rule inter-trimestres.
    # params["weights"] = {"acceleration", "momentum", "quality"} ; recalcule
    # panel["Z_Composite"] via factors.composite_weights_apply.
```

## src/backtest.py  (agent: portfolio_constructor)
```python
run_backtest(weights_by_date, prices, bench, tx_cost_bps=10, rebalance_dates=None) -> dict
    # rebalance aux 11 dates ; NAV Base 100 au PERF_START = 2024-01-25 ;
    # returns quotidiens, turnover par rebalancement (11 événements trimestriels),
    # weights_daily (dérive buy&hold entre rebalancements).
    # Poids fixés en t s'appliquent à partir du jour ouvré suivant t (rendement du
    # jour de rebalancement calculé avec les anciens poids dérivés).
```

## src/metrics.py  (agent: portfolio_constructor)
```python
performance_stats(nav, bench_nav, weights_daily=None, index_weights=None,
                  bench_returns=None, rf=0.0, turnover=None) -> dict
    # cum_return, ann_return, vol_ann, sharpe, sortino, alpha_ann, te_ann, info_ratio,
    # beta, max_dd, active_share, top10_weight, turnover_avg, n_effective,
    # up_capture, down_capture, tracking_error_ex_post.
    # index_weights = MultiIndex (date, ticker) ; active_share / top10 / n_effective
    # moyennés sur les snapshots aux dates de rebalancement (_asof_row).
    # turnover_avg = moyenne des turnovers par rebalancement (trimestriel).
drawdown_series(nav) -> pd.Series
```

## tests/  (agent: quant_auditor) -> AUDIT_REPORT.md
`tests/conftest.py` : fixtures partagées (scope session) `prices`, `benchmarks`,
`bench`, `bench_ret`, `fund`, `panel` (11 dates), `index_weights`,
`index_sector_w`, `sector_map`, `composite_portfolios` (stratégie unique
« Growth Composite ») et `backtest_result`.

`tests/test_audit.py` (baseline v8 : trimestriel PIT 2024-2026, EGUS unique, 3 piliers) :
- aucune utilisation de `ticker.info` pour les fondamentaux (analyse AST) ;
  `build_universe.py` n'utilise `.info` que pour l'identification symbol/sector
- PIT: `pub_date <= date` de rebalancement à chaque date ; aucun poids avant
  `filing_date` ; invariance par troncature/perturbation du futur (Δ=0) aux 11 dates
- somme poids == 1, w >= 0, N ∈ [40,50], bornes actives <±1e-6, cap `w_max`,
  secteur dans la bande, Core présent — sur les 11 dates
- dérive buy&hold intra-trimestre, poids appliqués à `t+1`, turnover vs poids dérivés
- réplication bit-à-bit des 3 piliers (Accélération / Momentum / Qualité-GARP) ;
  winsorisation 5/95 et z-scores en coupe transversale uniquement (par date)
- renormalisation PIT de `data/universe_quarterly.csv` (EC/US, sans STIV/DE)

`tests/test_audit_bias.py` (analyse critique de la baseline) :
- univers = holdings N-PORT de l'ETF (alpha attendu faible) ; bootstrap par blocs
  (21 j, 1000 tirages) → IC95 de l'alpha Composite vs EGUS (contient 0) ;
  bêta, TE, turnover trimestriel, Top10 vs indice ; limites (univers ESG ≠
  MSCI USA Growth pur, délistés retirés, pas de macro)

`tests/test_app.py` (smoke test du dashboard) · `tests/test_data_loader.py` ·
`tests/test_factors.py` · `tests/test_portfolio.py`.
Commande : `python -m pytest tests/ -q` → **102 passed**.

## app.py  (agent: streamlit_frontend)
- sidebar (3 sliders de pondération des piliers 25/45/30, N, bande sectorielle,
  bornes actives, plafond Mega-Cap, buffer, coûts, rf)
- 4 onglets Plotly (Stratégies & Paramètres, Performance, Secteurs,
  Core/Satellites & Factoriel/Audit). Pas d'onglet Régime (macro supprimée).
  Utiliser `src` modules. `@st.cache_data`.
```
