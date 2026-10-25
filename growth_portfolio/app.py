"""Dashboard Streamlit (Plotly) — Moteur Quant Growth USA.

Interface de reporting de niveau institutionnel pour le moteur << Core +
Satellites >> (``CONTRACT.md``). Aucune logique quantitative n'est reimplementee
ici : le dashboard consomme exclusivement les modules de ``src/`` (data_loader,
factors, portfolio, backtest, metrics).

Baseline **v8** : **backtest trimestriel Point-in-Time** sur l'univers historique
EGUS (11 dates de rebalancement, 25/01/2024 -> 27/07/2026). **Strategie unique**
a trois piliers statiques :

    Z_Composite = 0.25*Z_Acceleration + 0.45*Z_Momentum + 0.30*Z_Quality_GARP

NAV Base 100 au ``PERF_START`` (25/01/2024). Benchmark de reference **unique** :
``EGUS`` (iShares ESG Aware MSCI USA Growth). Aucune dependance macro : les
cibles sectorielles sont les poids GICS de l'indice EGUS du trimestre.

Trois onglets :
  1. Performance & Risque
  2. Secteurs (GICS, par trimestre)
  3. Core/Satellites & Factoriel/Audit

Lancement :  streamlit run app.py
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from src import backtest as bt
from src import config
from src import data_loader as dl
from src import factors
from src import gscore
from src import metrics as M
from src import mohanram as mo
from src import portfolio as pf

# ===========================================================================
# Configuration generale
# ===========================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Growth USA - Dashboard Quant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PLOTLY_TEMPLATE = "plotly_dark"
REFERENCE = "EGUS"                      # benchmark de reference unique
STRATEGY_NAME = "Growth Composite"      # strategie unique a trois piliers
COLORS = {STRATEGY_NAME: "#B388FF"}
BENCH_COLOR = "#58A6FF"
DTYPE = "rgba(0,0,0,0)"

# Palette discrete par secteur GICS (scatter factoriel / area chart).
SECTOR_PALETTE = [
    "#4C9AFF", "#F0B429", "#2ECC71", "#E74C3C", "#B388FF", "#22D3EE",
    "#F97316", "#A3E635", "#F472B6", "#94A3B8", "#FACC15",
]
SECTOR_COLORS = {s: SECTOR_PALETTE[i % len(SECTOR_PALETTE)]
                 for i, s in enumerate(config.GICS_SECTORS)}

STAT_LABELS = {
    "cum_return": "Rendement cumulé",
    "ann_return": "Rendement annualisé",
    "vol_ann": "Volatilité annualisée",
    "sharpe": "Ratio de Sharpe",
    "sortino": "Ratio de Sortino",
    "alpha_ann": "Alpha annualisé",
    "te_ann": "Tracking Error",
    "info_ratio": "Ratio d'information",
    "beta": "Bêta",
    "max_dd": "Max Drawdown",
    "active_share": "Active Share",
    "top10_weight": "Poids Top 10",
    "turnover_avg": "Turnover moyen / rebal. (trim.)",
    "n_effective": "Nombre effectif de titres",
    "up_capture": "Up Capture",
    "down_capture": "Down Capture",
}
PCT_KEYS = {
    "cum_return", "ann_return", "vol_ann", "alpha_ann", "te_ann", "max_dd",
    "active_share", "top10_weight", "turnover_avg", "up_capture",
    "down_capture",
}
RATIO_KEYS = {"sharpe", "sortino", "info_ratio", "beta", "n_effective"}
STAT_ORDER = [
    "cum_return", "ann_return", "vol_ann", "sharpe", "sortino", "alpha_ann",
    "te_ann", "info_ratio", "beta", "max_dd", "active_share", "top10_weight",
    "turnover_avg", "n_effective", "up_capture", "down_capture",
]


def fmt_stat(key: str, value) -> str:
    """Formate une statistique pour l'affichage (pct / ratio / nombre)."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    if key in PCT_KEYS:
        return f"{v * 100:+.2f} %" if key in {
            "cum_return", "alpha_ann", "max_dd"} else f"{v * 100:.2f} %"
    if key in RATIO_KEYS:
        return f"{v:.2f}"
    return f"{v:.2f}"


def base_layout(fig: go.Figure, height: int = 420,
                title: str | None = None) -> go.Figure:
    """Applique le style sombre commun (titre en haut, legende sous le titre)."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        title=dict(text=title, y=0.97, yanchor="top", x=0, xanchor="left")
        if title else None,
        margin=dict(l=10, r=10, t=(84 if title else 44), b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.0,
                    xanchor="left", x=0, bgcolor=DTYPE),
        hovermode="x unified",
    )
    return fig


def empty_fig(message: str, height: int = 320) -> go.Figure:
    """Figure vide avec message (gestion des cas sans donnees)."""
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=14))
    fig.update_layout(template=PLOTLY_TEMPLATE, height=height,
                      xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def rel_base(nav: pd.Series, ref_nav: pd.Series) -> pd.Series:
    """Performance relative Base 100 d'une NAV contre une NAV de reference."""
    common = nav.index.intersection(ref_nav.index)
    return (nav.reindex(common) / ref_nav.reindex(common)) * 100.0


def _idx_weight_row(index_weights, date) -> pd.Series:
    """Poids indiciels (Series index=ticker) pour une date de rebalancement."""
    d = pd.Timestamp(date).normalize()
    try:
        sub = index_weights.xs(d, level=0)
    except (KeyError, TypeError):
        return pd.Series(dtype=float)
    if isinstance(sub, pd.DataFrame):
        sub = sub["weight"] if "weight" in sub.columns else sub.iloc[:, 0]
    return sub.astype(float)


def _weights_at(weights_daily: pd.DataFrame, date) -> pd.Series:
    """Ligne de poids quotidiens a la derniere seance <= ``date``."""
    idx = weights_daily.index
    pos = int(idx.searchsorted(pd.Timestamp(date), side="right")) - 1
    pos = min(max(pos, 0), len(idx) - 1)
    return weights_daily.iloc[pos].astype(float)


# ===========================================================================
# Chargements & calculs caches
# ===========================================================================
@st.cache_data(show_spinner="Chargement des donnees de marche…")
def load_base() -> dict:
    """Charge prix, benchmark EGUS, poids d'indice PIT et calendrier (cache)."""
    prices = dl.get_prices()
    benchmarks = dl.get_benchmarks()                  # {"EGUS": Series}
    index_weights = dl.get_index_weights()            # MultiIndex (date, ticker)
    index_sector_w = dl.get_index_sector_weights()    # index=dates, cols=secteurs
    rebalance_dates = dl.get_rebalance_dates()        # 11 dates PIT
    bench_blocks = bt.run_backtest({}, prices, benchmarks)
    return {
        "prices": prices,
        "benchmarks": benchmarks,
        "index_weights": index_weights,
        "index_sector_w": index_sector_w,
        "rebalance_dates": rebalance_dates,
        "bench_blocks": bench_blocks,
    }


@st.cache_data(show_spinner="Calcul du panel factoriel…")
def load_panel() -> pd.DataFrame:
    """Panel factoriel PIT (MultiIndex date x ticker), mis en cache disque."""
    return factors.build_factor_panel()


@st.cache_data(show_spinner=False)
def sector_map_series() -> pd.Series:
    """Mapping ticker -> secteur GICS (univers PIT, 100 % de couverture)."""
    return dl.get_sector_map()


def _drop_inception(turnover: pd.Series) -> pd.Series:
    """Retire l'evenement d'inception (construction depuis le cash).

    Le backtest demarre a ``PERF_START`` = 1re date de rebalancement : le premier
    turnover (0,5 * somme |w| = 1) est un **artefact de construction**, pas un
    rebalancement periodique. On conserve donc les rebalancements suivants.
    """
    if turnover is None or len(turnover) <= 1:
        return turnover
    return turnover.iloc[1:]


@st.cache_data(show_spinner="Construction de la stratégie et backtest…")
def compute_strategy(name: str, wa: float, wm: float, wq: float,
                     n_titles: int, sector_band: float, sector_pilot: bool,
                     dw_min: float, dw_max: float, w_max: float,
                     buffer_rule: bool, tx_bps: float, rf: float) -> dict:
    """Construit la strategie unique (backtest trimestriel PIT), la back-teste et
    calcule ses statistiques **contre le benchmark EGUS**.

    Tous les arguments sont primitifs (hashables) afin que ``st.cache_data``
    ne recalcule que si les parametres ont change.
    """
    base = load_base()
    prices = base["prices"]
    benchmarks = base["benchmarks"]
    index_weights = base["index_weights"]
    index_sector_w = base["index_sector_w"]
    panel = load_panel()

    params = {
        "n_titles": int(n_titles),
        "sector_band": float(sector_band),
        "sector_pilot": bool(sector_pilot),
        "dw_min": float(dw_min),
        "dw_max": float(dw_max),
        "w_max": float(w_max),
        "buffer_rule": bool(buffer_rule),
        "max_top10": None,                        # borne Top10 desactivee
        "weights": {"acceleration": float(wa), "momentum": float(wm),
                    "quality": float(wq)},
    }
    portfolios = pf.build_all_portfolios(
        params, panel=panel, index_weights=index_weights,
        index_sector_w=index_sector_w,
    )
    res = bt.run_backtest({name: portfolios}, prices, benchmarks,
                          tx_cost_bps=tx_bps)
    blk = res[name]
    bench_nav = base["bench_blocks"][REFERENCE]["nav"]

    turnover_full = blk["turnover"]                        # 11 evenements PIT
    turnover_real = _drop_inception(turnover_full)         # hors inception

    stats = M.performance_stats(
        blk["nav"], bench_nav,
        turnover=turnover_real,
        weights_daily=blk["weights_daily"],
        index_weights=index_weights,
        bench_returns=base["bench_blocks"][REFERENCE]["returns"],
        rf=float(rf),
    )

    # Expositions sectorielles cibles agregees par date de construction.
    sec_map = sector_map_series()
    sec_rows = {}
    for d, w in portfolios.items():
        sec = sec_map.reindex(w.index).fillna("__NA__")
        sec_rows[d] = w.groupby(sec).sum().reindex(config.GICS_SECTORS).fillna(0.0)
    sector_w = pd.DataFrame(sec_rows).T
    sector_w.index = pd.to_datetime(sector_w.index).normalize()

    return {
        "portfolios": portfolios,
        "nav": blk["nav"],
        "returns": blk["returns"],
        "turnover": turnover_full,
        "turnover_real": turnover_real,
        "weights_daily": blk["weights_daily"],
        "stats": stats,
        "sector_w": sector_w,
    }


@st.cache_data(show_spinner=False)
def compute_bench_stats(rf: float) -> dict:
    """Statistiques du benchmark EGUS (relatif a lui-meme : alpha/TE nuls)."""
    base = load_base()
    bblk = base["bench_blocks"][REFERENCE]
    return M.performance_stats(bblk["nav"], bblk["nav"],
                               bench_returns=bblk["returns"], rf=float(rf))


# ===========================================================================
# Barre laterale
# ===========================================================================
def render_sidebar() -> dict:
    """Construit la barre laterale et retourne les parametres d'execution."""
    st.sidebar.title("⚙️ Paramétrage")
    st.sidebar.caption("Stratégie unique à trois piliers PIT — backtest "
                       "trimestriel — univers EGUS")
    st.sidebar.caption(
        f"Benchmark de référence unique : **{REFERENCE}** (iShares ESG Aware "
        "MSCI USA Growth). Les métriques relatives (alpha, TE, IR, bêta, active "
        "share) sont calculées contre cet indice."
    )

    st.sidebar.markdown("#### 🧬 Pondération des piliers")
    wa = st.sidebar.slider("ω Accélération (%)", 0, 100, 25, 1)
    wm = st.sidebar.slider("ω Momentum (%)", 0, 100, 45, 1)
    wq = st.sidebar.slider("ω Qualité/GARP (%)", 0, 100, 30, 1)
    tot = wa + wm + wq
    if tot <= 0:
        tot = 1
    st.sidebar.caption(
        f"Normalisation → Accélération **{wa/tot:.0%}** · Momentum "
        f"**{wm/tot:.0%}** · Qualité/GARP **{wq/tot:.0%}**"
    )

    st.sidebar.markdown("#### 🏗️ Construction Core + Satellites")
    n_titles = st.sidebar.slider("Nombre de titres (N)", config.N_MIN,
                                 config.N_MAX, config.DEFAULT_N, 1)
    sector_pilot = st.sidebar.toggle("Pilotage sectoriel (neutre vs EGUS)",
                                     value=True)
    sector_band = st.sidebar.slider(
        "Bande active sectorielle (± %)", 0.0, 5.0, 4.0, 0.5,
        disabled=not sector_pilot,
    )
    w_max = st.sidebar.slider("Plafond Mega-Cap (%)", 8.0, 15.0, 12.5, 0.5)
    dw_min = st.sidebar.slider("Borne active Δw min (%)", -5.0, 0.0, -3.5, 0.5)
    dw_max = st.sidebar.slider("Borne active Δw max (%)", 0.0, 5.0, 2.5, 0.5)
    buffer_rule = st.sidebar.toggle(
        "Buffer Rule", value=True,
        help="Conserver un satellite détenu tant qu'il reste dans le premier "
             "tiers de son secteur.")

    st.sidebar.markdown("#### 💸 Coûts & taux")
    tx_bps = st.sidebar.slider("Coûts de transaction (bps)", 0.0, 50.0, 10.0, 1.0)
    rf_pct = st.sidebar.slider("Taux sans risque annualisé (%)", 0.0, 5.0, 0.0, 0.1)

    return {
        "weights": (wa / 100.0, wm / 100.0, wq / 100.0),
        "n_titles": int(n_titles),
        "sector_pilot": bool(sector_pilot),
        "sector_band": float(sector_band / 100.0),
        "w_max": float(w_max / 100.0),
        "dw_min": float(dw_min / 100.0),
        "dw_max": float(dw_max / 100.0),
        "buffer_rule": bool(buffer_rule),
        "tx_bps": float(tx_bps),
        "rf": float(rf_pct / 100.0),
    }


# ===========================================================================
# Onglet 0 - Strategies & Parametres (documentation)
# ===========================================================================
def render_tab_guide(params: dict, base: dict) -> None:
    """Page de presentation : strategies, formules et parametres du moteur."""
    st.subheader("🧭 Stratégies, formules & paramètres")
    st.caption(
        "Documentation du moteur : stratégies factorielles, construction "
        "Core + Satellites et l'ensemble des paramètres ajustables depuis la "
        "barre latérale."
    )

    # --- Strategie unique a trois piliers ----------------------------------
    st.markdown("### 🧬 La stratégie — trois piliers PIT")
    st.markdown(
        "Chaque pilier est un **Z-score** calculé en coupe transversale "
        "(winsorisation 5/95 puis standardisation intra-sectorielle ; repli sur "
        "l'univers du trimestre si un secteur compte moins de 5 titres). Les "
        "agrégats fondamentaux sont **annuels Point-in-Time** (somme de 4 "
        "trimestres, `pub_date ≤ t`). La sélection ne retient pas directement les "
        "plus forts Z : elle s'appuie sur la construction **Core + Satellites** "
        "(section suivante)."
    )

    st.markdown(
        "| Pilier | Ce qu'il récompense | Poids défaut |\n"
        "| --- | --- | --- |\n"
        "| **1 · Accélération** | Le **fondamental prospectif** : entreprises qui "
        "*accélèrent* croissance, marges et efficacité, pas seulement celles déjà "
        "rentables. | 25 % |\n"
        "| **2 · Momentum** | La **confirmation par le marché** : prix qui "
        "poursuivent leur tendance, ajustés du risque et purgés du reversal court "
        "terme. | 45 % |\n"
        "| **3 · Qualité & GARP** | La **solidité et la valorisation** : "
        "rentabilité, génération de cash et prix raisonnable (mariage "
        "qualité/prix). | 30 % |\n\n"
        "L'idée directrice : un actif n'est retenu que s'il **surperforme sur le "
        "terrain fondamental**, **est validé par le marché** et **offre une "
        "qualité défendable à un prix raisonnable**. Les trois signaux sont "
        "imparfaitement corrélés, ce qui diversifie les sources d'alpha."
    )

    with st.expander("**Pilier 1 — Accélération fondamentale** (défaut 25 %)",
                     expanded=True):
        st.markdown(
            "`Z_Acceleration = mean(Z_SalesAccel, Z_MarginExp, Z_AssetEff)`\n\n"
            "| Métrique | Formule |\n"
            "| --- | --- |\n"
            "| `Sales_Accel` | `[CA_N/CA_N1 − 1] − [CA_N1/CA_N2 − 1]` |\n"
            "| `Margin_Exp` | `OI_N/CA_N − OI_N1/CA_N1` |\n"
            "| `Asset_Eff` | `CA_N/TA_N − CA_N1/TA_N1` "
            "(**Financials** → `ROA_N − ROA_N1`) |\n\n"
            "**Intuition économique.** Ce pilier est le moteur **fondamental "
            "prospectif** du modèle : il ne récompense pas la taille ou le niveau "
            "absolu de la croissance, mais son **accélération** (le *delta*, pas "
            "le stock). Une entreprise dont le chiffre d'affaires progresse plus "
            "vite qu'au trimestre précédent, dont la marge opérationnelle "
            "s'étend et dont le capital tourne mieux est en train d'**améliorer "
            "son économie sous-jacente** — souvent avant que le consensus ne "
            "l'ait pleinement intégré. Le risque assumé est celui du **retournement "
            "ou de la déception prospective** (révisions à la baisse) si "
            "l'accélération ne se matérialise pas."
        )

    with st.expander("**Pilier 2 — Momentum ajusté du risque** (défaut 45 %)"):
        st.markdown(
            "`Z_Momentum = 0.65·Z_RAM + 0.35·Z_Trend`\n\n"
            "| Métrique | Formule |\n"
            "| --- | --- |\n"
            "| `RAM` | `(P[t−21]/P[t−126] − 1) / (σ_6M·√252)` "
            "(momentum 6M−1M normalisé par la volatilité 6M) |\n"
            "| `Trend` | `Prix / SMA200 − 1` |\n\n"
            "**Intuition économique.** Ce pilier apporte la **confirmation par "
            "le marché** : les investisseurs ont historiquement tendance à "
            "sous-réagir puis à poursuivre un mouvement (*momentum*). Le RAM "
            "mesure la performance sur 6 mois **hors dernier mois** (fenêtre "
            "`t−126 → t−21`), excluant le *reversal* court terme pour ne capter "
            "que la persistance ; le ratio `Trend` (prix vs SMA200) recoupe la "
            "force de tendance. Normaliser par la volatilité 6M pénalise les "
            "trajectoires erratiques et récompense une tendance **régulière**. "
            "Le risque assumé est celui du **retournement de tendance** "
            "(*momentum crash*), atténué ici par l'ajustement au risque et la "
            "diversification sectorielle."
        )

    with st.expander("**Pilier 3 — Qualité & GARP** (défaut 30 %)"):
        st.markdown(
            "`Z_Quality_GARP = 0.40·Z_ROE + 0.35·Z_FCF_Yield + 0.25·Z_GARP`\n\n"
            "| Métrique | Formule |\n"
            "| --- | --- |\n"
            "| `ROE` | `Net_Income_TTM / Equity` |\n"
            "| `FCF_Yield` | `(OCF_TTM − \\|CapEx_TTM\\|) / Market_Cap` |\n"
            "| `GARP` | `EPS_Growth_3Y / (Prix / EPS_TTM)` "
            "(neutralisé si P/E ≤ 0 ou croissance ≤ 0) |\n\n"
            "**Financials** : FCF omis → `Z_Quality_GARP = 0.60·Z_ROE + "
            "0.40·Z_GARP`.\n\n"
            "**Intuition économique.** Ce pilier récompense la **robustesse "
            "bilancielle et un prix raisonnable** (*Growth at a Reasonable "
            "Price*). Le ROE atteste d'une rentabilité durable des fonds propres, "
            "le FCF Yield d'une capacité à **générer du cash réel** (qualité du "
            "bénéfice comptable), et le GARP marie la croissance des résultats au "
            "multiple payé — un P/E élevé n'est justifié que si la croissance "
            "l'est aussi. C'est le contrepoids **défensif** du modèle : il limite "
            "l'exposition aux valeurs de pure momentum chères ou fragiles. Le "
            "risque assumé est celui de la **value trap** (valorisation basse mais "
            "détérioration structurelle), atténué par le filtre de rentabilité."
        )

    with st.expander("**Pourquoi ces trois piliers sont complémentaires**"):
        st.markdown(
            "- **Accélération** : *signal prospectif* — anticipe l'amélioration des "
            "fondamentaux, avec le risque d'un pari contrariant.\n"
            "- **Momentum** : *signal de confirmation* — valide que le marché "
            "intègre déjà la dynamique, au prix d'un risque de retournement.\n"
            "- **Qualité/GARP** : *signal de solidité* — ancre le portefeuille sur "
            "des bilans sains et des valorisations raisonnables.\n\n"
            "Combinés, ils couvrent **trois horizons et deux familles de risque** : "
            "le fondamental prospectif (Accélération), le comportement de marché "
            "(Momentum) et le profil de qualité/prix (Qualité/GARP). Le réglage par "
            "défaut **25 / 45 / 30** donne au **Momentum** le poids dominant (45 %) "
            "car c'est historiquement le facteur le plus robuste et diversifiant "
            "dans ce type d'univers de grandes valeurs de croissance, tout en "
            "gardant une assise fondamentale (25 %) et un amortisseur de qualité "
            "(30 %). Régler ces poids depuis la barre latérale modifie directement "
            "la pondération des trois piliers dans `Z_Composite`."
        )

    wa, wm, wq = params["weights"]
    tot = (wa + wm + wq) or 1.0
    a_pct, m_pct, q_pct = 100.0 * wa / tot, 100.0 * wm / tot, 100.0 * wq / tot
    with st.expander("**Composite** — agrégation des piliers", expanded=True):
        st.markdown(
            "`Z_Composite = ωA·Z_Acceleration + ωM·Z_Momentum + ωQ·Z_Quality_GARP`"
            "  (ω normalisés à 100 %)\n\n"
            f"Configuration courante : **Accélération {a_pct:.0f} % · "
            f"Momentum {m_pct:.0f} % · Qualité/GARP {q_pct:.0f} %** "
            "(réglage par défaut : 25 % / 45 % / 30 %)."
        )

    # --- Construction Core + Satellites ------------------------------------
    st.markdown("### 🏗️ Construction Core + Satellites")
    st.markdown(
        "Un portefeuille **intégralement actif** dériverait trop loin de "
        "l'indice et coûterait cher en turnover. La construction **Core + "
        "Satellites** sépare donc deux intentions : le **Core** fournit "
        "l'**ancrage** (bêta et empreinte sectorielle proches de l'indice EGUS), "
        "tandis que les **Satellites** portent les **convictions actives** issues "
        "de `Z_Composite`, sous contraintes strictes de poids et de déviation.\n"
    )
    st.markdown(
        "- **Core** : titres dont le poids indiciel EGUS est ≥ **2,5 %** — "
        "toujours détenus.\n"
        "- **Satellites** : meilleurs `Z_Composite` de chaque secteur, complétés "
        "jusqu'à **N** titres au total.\n"
        "- **Buffer Rule** : un satellite déjà détenu au trimestre précédent "
        "reste sélectionné tant que son `Z_Composite` demeure au-dessus du "
        "**60ᵉ centile** de son secteur.\n"
        "- **Pondération** : multiplicateur de tilt sur `Z` (0,35 pour le Core, "
        "0,70 pour les Satellites), puis projection sous contraintes "
        "(bornes actives, plafond Mega-Cap, cibles sectorielles)."
    )

    # --- Tableau des parametres -------------------------------------------
    st.markdown("### 📋 Paramètres")
    st.caption(
        "Réglables depuis la barre latérale : ω Accélération / ω Momentum / "
        "ω Qualité de la stratégie, Nombre de titres (N), Pilotage sectoriel et "
        "Bande active sectorielle, Plafond Mega-Cap, Bornes actives Δw, "
        "Buffer Rule, Coûts de transaction et Taux sans risque."
    )
    n_dates = len(base.get("rebalance_dates", config.REBALANCE_DATES))
    iw = base.get("index_weights")
    if iw is not None and len(iw):
        sizes = iw.groupby(level=0).size()
        uni_txt = (f"{int(sizes.min())}–{int(sizes.max())} titres / trimestre")
    else:
        uni_txt = "n.a."

    def _onoff(flag: bool) -> str:
        return "Activé" if flag else "Désactivé"

    rows = [
        {"Paramètre": "ω Accélération",
         "Valeur actuelle": f"{a_pct:.0f} %",
         "Défaut": "25 %",
         "Description": "Poids de Z_Acceleration dans Z_Composite (normalisé à 100 %)."},
        {"Paramètre": "ω Momentum",
         "Valeur actuelle": f"{m_pct:.0f} %",
         "Défaut": "45 %",
         "Description": "Poids de Z_Momentum dans Z_Composite (normalisé à 100 %)."},
        {"Paramètre": "ω Qualité/GARP",
         "Valeur actuelle": f"{q_pct:.0f} %",
         "Défaut": "30 %",
         "Description": "Poids de Z_Quality_GARP dans Z_Composite (normalisé à 100 %)."},
        {"Paramètre": "Nombre de titres (N)",
         "Valeur actuelle": str(params["n_titles"]),
         "Défaut": f"{config.DEFAULT_N}",
         "Description": f"Taille du portefeuille (plage {config.N_MIN}–{config.N_MAX})."},
        {"Paramètre": "Pilotage sectoriel",
         "Valeur actuelle": _onoff(params["sector_pilot"]),
         "Défaut": "Activé",
         "Description": "Cibles de secteurs = poids GICS de l'indice EGUS du trimestre."},
        {"Paramètre": "Bande active sectorielle (±)",
         "Valeur actuelle": f"{params['sector_band'] * 100:.1f} %",
         "Défaut": "4,0 %",
         "Description": "Déviation maximale autorisée par secteur autour de la cible."},
        {"Paramètre": "Plafond Mega-Cap",
         "Valeur actuelle": f"{params['w_max'] * 100:.1f} %",
         "Défaut": "12,5 %",
         "Description": "Poids maximal d'un titre dans le portefeuille."},
        {"Paramètre": "Borne active Δw min",
         "Valeur actuelle": f"{params['dw_min'] * 100:.1f} %",
         "Défaut": "−3,5 %",
         "Description": "Sous-pondération active minimale vs poids indiciel."},
        {"Paramètre": "Borne active Δw max",
         "Valeur actuelle": f"{params['dw_max'] * 100:+.1f} %",
         "Défaut": "+2,5 %",
         "Description": "Surpondération active maximale vs poids indiciel."},
        {"Paramètre": "Buffer Rule",
         "Valeur actuelle": _onoff(params["buffer_rule"]),
         "Défaut": "Activé",
         "Description": "Maintien des satellites détenus au-dessus du 60ᵉ centile sectoriel."},
        {"Paramètre": "Coûts de transaction",
         "Valeur actuelle": f"{params['tx_bps']:.0f} bps",
         "Défaut": f"{config.DEFAULT_TX_COST_BPS:.0f} bps",
         "Description": "Frais appliqués au turnover à chaque rebalancement."},
        {"Paramètre": "Taux sans risque",
         "Valeur actuelle": f"{params['rf'] * 100:.1f} %",
         "Défaut": "0,0 %",
         "Description": "Taux annualisé utilisé par le ratio de Sharpe."},
        {"Paramètre": "Borne Top 10",
         "Valeur actuelle": "Désactivée",
         "Défaut": "Désactivée",
         "Description": "Aucune contrainte de concentration Top 10 (choix utilisateur)."},
    ]
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # --- Methodologie ------------------------------------------------------
    st.markdown("### 🔄 Méthodologie du backtest")
    st.info(
        f"**{n_dates} rebalancements trimestriels PIT** (dates de connaissance "
        "publique des poids N-PORT). Univers = **actions US de l'ETF EGUS** "
        "(poches cash et dérivés exclues), renormalisé à 100 % chaque trimestre "
        f"({uni_txt}).\n\n"
        "Les poids fixés à la date `t` s'appliquent à partir de `t+1` puis "
        "**dérivent en buy & hold** jusqu'au rebalancement suivant. NAV "
        "**Base 100** au premier rebalancement ; benchmark de référence "
        "**EGUS** (iShares ESG Aware MSCI USA Growth).\n\n"
        "Aucune donnée postérieure n'entre dans les scores (**Point-in-Time**) : "
        "fondamentaux datés par leur publication, aucune lecture `ticker.info` "
        "pour les fondamentaux."
    )


# ===========================================================================
# Onglet 1 - Performance & Risque
# ===========================================================================
def render_tab_performance(results: dict, base: dict, primary: str,
                           rf: float) -> None:
    """KPIs, courbes de performance, alpha cumule, drawdowns, turnover, tableau."""
    navs = {name: r["nav"] for name, r in results.items()}
    bench_nav = base["bench_blocks"][REFERENCE]["nav"]

    start = bench_nav.index[0]
    end = bench_nav.index[-1]
    st.subheader(
        f"Indicateurs clés — {primary} vs **{REFERENCE}** — "
        f"fenêtre {start:%d/%m/%Y} → {end:%d/%m/%Y}"
    )
    s_p = results[primary]["stats"]
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Rendement cumulé", fmt_stat("cum_return", s_p["cum_return"]))
    c2.metric(f"Alpha vs {REFERENCE}", fmt_stat("alpha_ann", s_p["alpha_ann"]))
    c3.metric("Tracking Error", fmt_stat("te_ann", s_p["te_ann"]))
    c4.metric("Ratio d'information", fmt_stat("info_ratio", s_p["info_ratio"]))
    c5.metric("Ratio de Sharpe", fmt_stat("sharpe", s_p["sharpe"]))
    c6.metric("Max Drawdown", fmt_stat("max_dd", s_p["max_dd"]))
    st.caption(
        f"KPIs relatifs au benchmark **{REFERENCE}**. Alpha annualisé "
        "arithmétique (moyenne des rendements actifs × 252) ; l'excès géométrique "
        "sur la fenêtre peut différer."
    )

    # --- Performances cumulees Base 100 ------------------------------------
    fig_nav = go.Figure()
    fig_nav.add_trace(go.Scatter(
        x=bench_nav.index, y=bench_nav.values, name=f"Benchmark {REFERENCE}",
        line=dict(color=BENCH_COLOR, width=2.4, dash="dash"),
    ))
    for name, nav in navs.items():
        fig_nav.add_trace(go.Scatter(
            x=nav.index, y=nav.values, name=name,
            line=dict(color=COLORS.get(name, "#FFFFFF"), width=2.2),
        ))
    fig_nav.update_yaxes(title_text="Base 100")
    base_layout(fig_nav, height=440,
                title="Performances cumulées (Base 100, depuis 25/01/2024)")

    # --- Alpha cumule -------------------------------------------------------
    fig_alpha = go.Figure()
    for name, nav in navs.items():
        rel = rel_base(nav, bench_nav)
        fig_alpha.add_trace(go.Scatter(
            x=rel.index, y=rel.values, name=name,
            line=dict(color=COLORS.get(name, "#FFFFFF"), width=2.0),
        ))
    fig_alpha.add_hline(y=100.0, line=dict(color="#8B949E", width=1.2, dash="dot"))
    fig_alpha.update_yaxes(title_text="Perf. relative (Base 100)")
    base_layout(fig_alpha, height=440,
                title=f"Alpha cumulé (performance relative vs {REFERENCE})")

    left, right = st.columns([1.35, 1])
    with left:
        st.plotly_chart(fig_nav, use_container_width=True)
    with right:
        st.plotly_chart(fig_alpha, use_container_width=True)

    # --- Underwater --------------------------------------------------------
    fig_dd = go.Figure()
    dd_b = M.drawdown_series(bench_nav) * 100.0
    fig_dd.add_trace(go.Scatter(
        x=dd_b.index, y=dd_b.values, name=f"Benchmark {REFERENCE}",
        line=dict(color=BENCH_COLOR, width=1.6, dash="dash"),
    ))
    for name, nav in navs.items():
        dd = M.drawdown_series(nav) * 100.0
        fig_dd.add_trace(go.Scatter(
            x=dd.index, y=dd.values, name=name,
            line=dict(color=COLORS.get(name, "#FFFFFF"), width=1.6),
        ))
    fig_dd.update_yaxes(title_text="Drawdown (%)")
    base_layout(fig_dd, height=340,
                title="Underwater — drawdowns quotidiens comparés")
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Turnover par rebalancement (trimestriel) --------------------------
    to = results[primary]["turnover"]
    if to is not None and len(to) > 0:
        to_pct = to * 100.0
        colors = [BENCH_COLOR if i == 0 else COLORS.get(primary, "#B388FF")
                  for i in range(len(to_pct))]
        fig_to = go.Figure(go.Bar(
            x=[pd.Timestamp(d) for d in to_pct.index], y=to_pct.values,
            marker_color=colors,
            text=[f"{v:.1f}%" for v in to_pct.values], textposition="outside",
        ))
        fig_to.update_yaxes(title_text="Turnover aller simple (%)")
        base_layout(fig_to, height=340,
                    title="Turnover par trimestre (1re barre = inception depuis le cash)")
        st.plotly_chart(fig_to, use_container_width=True)
        to_real = results[primary]["turnover_real"]
        st.caption(
            f"**Turnover moyen par rebalancement trimestriel** (hors inception, "
            f"{len(to_real)} événements) : **{to_real.mean() * 100:.2f} %** "
            "aller simple. La 1re barre (clair) est la construction initiale "
            "depuis le cash — un artefact, non un rebalancement périodique."
        )

    # --- Tableau recapitulatif --------------------------------------------
    st.subheader("Tableau récapitulatif comparatif")
    table: dict = {}
    for name, r in results.items():
        s = r["stats"]
        table[name] = {STAT_LABELS[k]: fmt_stat(k, s[k]) for k in STAT_ORDER}
    bstats = compute_bench_stats(rf)
    table[f"Benchmark {REFERENCE}"] = {STAT_LABELS[k]: fmt_stat(k, bstats[k])
                                       for k in STAT_ORDER}

    df_table = pd.DataFrame(table, index=[STAT_LABELS[k] for k in STAT_ORDER])
    df_table.index.name = "Statistique"
    st.dataframe(df_table, use_container_width=True)
    st.caption(
        f"Métriques relatives calculées contre **{REFERENCE}**. Le turnover est "
        "la **moyenne des rebalancements trimestriels** (hors construction "
        "initiale)."
    )
    st.download_button(
        "⬇️ Exporter le tableau (CSV)",
        data=df_table.to_csv().encode("utf-8"),
        file_name="growth_usa_stats.csv",
        mime="text/csv",
    )


# ===========================================================================
# Onglet 2 - Allocation sectorielle GICS par trimestre
# ===========================================================================
def render_tab_sectors(results: dict, base: dict, primary: str) -> None:
    """Allocation sectorielle GICS du portefeuille vs EGUS, par trimestre."""
    index_sector_w = base["index_sector_w"]
    rebalance_dates = base["rebalance_dates"]
    strat_names = list(results.keys())
    strat_sel = st.selectbox("Stratégie", strat_names,
                             index=strat_names.index(primary), key="sec_strat")

    # Date de rebalancement (defaut = derniere).
    idx_dates = [pd.Timestamp(d) for d in index_sector_w.index]
    date_options = [d for d in rebalance_dates if d in idx_dates] or idx_dates
    date_sel = st.selectbox(
        "Trimestre (date de rebalancement)", date_options,
        index=len(date_options) - 1,
        format_func=lambda d: f"{pd.Timestamp(d):%d/%m/%Y}", key="sec_date")

    port_sector = results[strat_sel]["sector_w"]
    port_row = port_sector.loc[pd.Timestamp(date_sel)].reindex(
        config.GICS_SECTORS).fillna(0.0)
    idx_row = index_sector_w.loc[pd.Timestamp(date_sel)].reindex(
        config.GICS_SECTORS).fillna(0.0)

    st.subheader(
        f"Poids sectoriels — {strat_sel} vs EGUS "
        f"({pd.Timestamp(date_sel):%d/%m/%Y})")
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=config.GICS_SECTORS, y=idx_row.values * 100.0,
        name="EGUS (indice)", marker_color=BENCH_COLOR,
    ))
    fig_bar.add_trace(go.Bar(
        x=config.GICS_SECTORS, y=port_row.values * 100.0,
        name=strat_sel, marker_color=COLORS.get(strat_sel, "#4C9AFF"),
    ))
    fig_bar.update_layout(barmode="group")
    fig_bar.update_yaxes(title_text="Poids (%)")
    fig_bar.update_xaxes(tickangle=-35)
    base_layout(fig_bar, height=430,
                title=f"Barres groupées — {strat_sel} vs EGUS")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- Ecarts actifs ------------------------------------------------------
    active = (port_row - idx_row) * 100.0
    active = active.sort_values()
    colors = ["#E74C3C" if v < 0 else "#2ECC71" for v in active.values]
    fig_act = go.Figure(go.Bar(
        x=active.values, y=list(active.index), orientation="h",
        marker_color=colors,
        text=[f"{v:+.2f}%" for v in active.values], textposition="outside",
    ))
    fig_act.update_xaxes(title_text="Écart actif Δw (%)")
    base_layout(fig_act, height=430,
                title=f"Écarts actifs sectoriels Δw — {strat_sel} vs EGUS")
    st.plotly_chart(fig_act, use_container_width=True)

    # --- Evolution empilee de la composition sectorielle --------------------
    st.subheader(f"Évolution de la composition sectorielle — {strat_sel}")
    fig_area = go.Figure()
    for sector in config.GICS_SECTORS:
        if sector not in port_sector.columns:
            continue
        fig_area.add_trace(go.Scatter(
            x=[pd.Timestamp(d) for d in port_sector.index],
            y=port_sector[sector].values * 100.0,
            name=sector, stackgroup="one", mode="lines",
            line=dict(width=0.5, color=SECTOR_COLORS.get(sector)),
        ))
    fig_area.update_yaxes(title_text="Poids (%)")
    base_layout(fig_area, height=430,
                title=f"Area chart empilé — composition GICS par trimestre ({strat_sel})")
    st.plotly_chart(fig_area, use_container_width=True)
    st.caption(
        "Chaque trimestre correspond à une date de rebalancement PIT "
        "(filing N-PORT). Les cibles sectorielles sont les poids GICS de l'indice "
        "EGUS du trimestre (bande active bornée par le pilotage sectoriel)."
    )


# ===========================================================================
# Onglet 3 - Core/Satellites & Factoriel/Audit
# ===========================================================================
def render_tab_anatomy(results: dict, base: dict, primary: str) -> None:
    """Core/Satellites, derive des poids, Top 15 actifs, nuage factoriel, audit."""
    index_weights = base["index_weights"]
    rebalance_dates = base["rebalance_dates"]
    panel = load_panel()
    sec_map = sector_map_series()

    strat_names = list(results.keys())
    strat_sel = st.selectbox("Stratégie", strat_names,
                             index=strat_names.index(primary), key="anat_strat")
    results[strat_sel]["portfolios"]

    # Date de rebalancement (defaut = derniere).
    date_options = list(rebalance_dates)
    date_sel = st.selectbox(
        "Trimestre (date de rebalancement)", date_options,
        index=len(date_options) - 1,
        format_func=lambda d: f"{pd.Timestamp(d):%d/%m/%Y}", key="anat_date")
    date_sel = pd.Timestamp(date_sel)

    portfolios = results[strat_sel]["portfolios"]
    weights_daily = results[strat_sel]["weights_daily"]

    # --- Core / Satellites par trimestre -----------------------------------
    st.subheader("Anatomie Core + Satellites (par trimestre)")
    core_w, sat_w, n_core_l, n_sat_l = [], [], [], []
    for d in rebalance_dates:
        d = pd.Timestamp(d)
        if d not in portfolios:
            continue
        w = portfolios[d]
        w = w[w > 0]
        iw = _idx_weight_row(index_weights, d).reindex(w.index).fillna(0.0)
        is_core = iw >= pf.CORE_THRESHOLD
        core_w.append(float(w[is_core].sum()))
        sat_w.append(float(w[~is_core].sum()))
        n_core_l.append(int(is_core.sum()))
        n_sat_l.append(int((~is_core).sum()))

    x_dates = [f"{pd.Timestamp(d):%d/%m/%Y}" for d in rebalance_dates
               if pd.Timestamp(d) in portfolios]
    fig_cs = go.Figure()
    fig_cs.add_trace(go.Bar(x=x_dates, y=np.array(core_w) * 100.0,
                            name="Core (w_idx ≥ 2,5 %)", marker_color="#4C9AFF"))
    fig_cs.add_trace(go.Bar(x=x_dates, y=np.array(sat_w) * 100.0,
                            name="Satellites (convictions)", marker_color="#B388FF"))
    fig_cs.update_layout(barmode="stack")
    fig_cs.update_yaxes(title_text="Poids agrégé (%)")
    fig_cs.update_xaxes(tickangle=-45)
    base_layout(fig_cs, height=430,
                title=f"Répartition Core / Satellites par trimestre — {strat_sel}")
    st.plotly_chart(fig_cs, use_container_width=True)

    # Detail du trimestre selectionne.
    d_pos = list(portfolios.keys()).index(date_sel) \
        if date_sel in portfolios else len(portfolios) - 1
    sel_date = list(portfolios.keys())[d_pos]
    port_w = portfolios[sel_date]
    port_w = port_w[port_w > 0].sort_values(ascending=False)
    idx_w_sel = _idx_weight_row(index_weights, sel_date).reindex(port_w.index).fillna(0.0)
    is_core = idx_w_sel >= pf.CORE_THRESHOLD

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Titres total", f"{len(port_w)}")
    k2.metric("Noyau Core (w_idx ≥ 2,5 %)", f"{int(is_core.sum())}")
    k3.metric("Satellites", f"{int((~is_core).sum())}")
    top10 = float(port_w.head(10).sum())
    top10_idx = float(idx_w_sel.sort_values(ascending=False).head(10).sum())
    k4.metric("Top 10 portefeuille", f"{top10*100:.1f} %",
              f"{(top10 - top10_idx)*100:+.1f} pts vs indice")

    # --- Poids initiaux vs derivees entre deux rebalancements ---------------
    st.subheader("Poids initiaux vs poids dérivés (buy & hold inter-trimestre)")
    order = [pd.Timestamp(d) for d in portfolios.keys()]
    pos = order.index(sel_date)
    init = _weights_at(weights_daily, sel_date)
    if pos + 1 < len(order):
        nxt = order[pos + 1]
        idx_wd = weights_daily.index
        nxt_pos = int(idx_wd.searchsorted(pd.Timestamp(nxt), side="right")) - 1
        derived = weights_daily.iloc[max(nxt_pos - 1, 0)].astype(float)
        derived_label = f"{pd.Timestamp(nxt):%d/%m/%Y}"
    else:
        derived = weights_daily.iloc[-1].astype(float)
        derived_label = f"{weights_daily.index[-1]:%d/%m/%Y}"

    ranked = init[init > 0].sort_values(ascending=False)
    top20 = ranked.head(20).index
    w0 = init.reindex(top20).fillna(0.0) * 100.0
    w1 = derived.reindex(top20).fillna(0.0) * 100.0
    fig_drift = go.Figure()
    fig_drift.add_trace(go.Bar(
        x=list(top20), y=w0.values,
        name=f"Poids initiaux ({sel_date:%d/%m/%Y})", marker_color="#8B949E"))
    fig_drift.add_trace(go.Bar(
        x=list(top20), y=w1.values,
        name=f"Poids dérivés ({derived_label})", marker_color="#2ECC71"))
    fig_drift.update_layout(barmode="group")
    fig_drift.update_xaxes(tickangle=-65)
    fig_drift.update_yaxes(title_text="Poids (%)")
    base_layout(fig_drift, height=470,
                title="Poids initiaux vs dérivés (Top 20, dérive buy & hold)")
    st.plotly_chart(fig_drift, use_container_width=True)
    drift = w1 - w0
    if len(drift):
        st.caption(
            "Entre deux rebalancements, les poids **dérivent** avec les "
            "rendements relatifs (buy & hold). Amplitude de dérive sur le Top 20 : "
            f"{drift.max():+.2f} pt à {drift.min():+.2f} pt."
        )

    # --- Top 15 sur / sous-ponderations actives vs EGUS --------------------
    st.subheader("Top 15 surpondérations / sous-pondérations actives vs EGUS")
    idx_w = _idx_weight_row(index_weights, sel_date)
    cols = pd.Index(port_w.index).union(idx_w.index)
    active = (port_w.reindex(cols).fillna(0.0)
              - idx_w.reindex(cols).fillna(0.0)) * 100.0
    active = active[active.abs() > 1e-9]
    over = active.sort_values(ascending=False).head(15)
    under = active.sort_values().head(15)
    fig_act = go.Figure()
    fig_act.add_trace(go.Bar(
        x=over.values, y=list(over.index), orientation="h",
        name="Surpondérations actives", marker_color="#2ECC71"))
    fig_act.add_trace(go.Bar(
        x=under.values, y=list(under.index), orientation="h",
        name="Sous-pondérations actives", marker_color="#E74C3C"))
    fig_act.update_xaxes(title_text="Poids actif Δw (%)")
    base_layout(fig_act, height=560,
                title=f"Top 15 actifs vs EGUS — {strat_sel} ({sel_date:%d/%m/%Y})")
    st.plotly_chart(fig_act, use_container_width=True)

    # --- Explorateur factoriel ---------------------------------------------
    st.subheader("Explorateur factoriel — Z-scores de l'univers EGUS")
    panel_dates = sorted(pd.Index(panel.index.get_level_values(0)).unique())
    p_date = sel_date if pd.Timestamp(sel_date) in [pd.Timestamp(d) for d in panel_dates] \
        else pd.Timestamp(panel_dates[-1])
    sectors_sel = st.multiselect("Secteurs GICS", list(config.GICS_SECTORS),
                                 default=list(config.GICS_SECTORS), key="fac_sec")
    z_axes = ["Z_Composite", "Z_Acceleration", "Z_Momentum", "Z_Quality_GARP"]
    ac1, ac2 = st.columns(2)
    x_axis = ac1.selectbox("Axe X", z_axes, index=2, key="fac_x")
    y_axis = ac2.selectbox("Axe Y", z_axes, index=0, key="fac_y")

    panel_date = panel.xs(p_date, level=0).copy()
    portfolio = results[strat_sel]["portfolios"].get(pd.Timestamp(p_date))
    weight = (portfolio.reindex(panel_date.index).fillna(0.0)
              if portfolio is not None
              else pd.Series(0.0, index=panel_date.index))
    panel_date["weight"] = weight
    panel_date["sector"] = sec_map.reindex(panel_date.index)
    plot_df = panel_date[panel_date["sector"].isin(sectors_sel)]
    plot_df = plot_df[plot_df[[x_axis, y_axis]].notna().any(axis=1)]

    if plot_df.empty:
        st.plotly_chart(empty_fig("Aucun titre pour les filtres sélectionnés."),
                        use_container_width=True)
    else:
        wmax = max(float(plot_df["weight"].max()), 1e-9)
        sizes = 8 + 55 * (plot_df["weight"] / wmax)
        fig = go.Figure()
        for sector in config.GICS_SECTORS:
            if sector not in sectors_sel:
                continue
            sub = plot_df[plot_df["sector"] == sector]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub[x_axis], y=sub[y_axis], mode="markers",
                name=sector,
                marker=dict(size=sizes.reindex(sub.index),
                            color=SECTOR_COLORS.get(sector, "#FFFFFF"),
                            opacity=0.85, line=dict(width=0)),
                text=[f"{t}<br>poids {w*100:.2f} %" for t, w in
                      zip(sub.index, sub["weight"])],
                hoverinfo="text+name",
            ))
        fig.add_hline(y=0, line=dict(color="#4A5568", width=1, dash="dot"))
        fig.add_vline(x=0, line=dict(color="#4A5568", width=1, dash="dot"))
        fig.update_xaxes(title_text=x_axis)
        fig.update_yaxes(title_text=y_axis)
        base_layout(fig, height=620,
                    title=f"Nuage factoriel — {x_axis} vs {y_axis} "
                          f"({p_date:%d/%m/%Y}) (couleur = secteur, taille = poids)")
        st.plotly_chart(fig, use_container_width=True)

    # --- Controles de validation & audit -----------------------------------
    st.subheader("Contrôles de validation & audit quantitatif")
    st.info(
        "**Absence de Look-Ahead Bias (quant_auditor).** Univers **Point-in-Time** "
        "reconstruit à chaque date de rebalancing (panier N-PORT du trimestre). "
        "Les Z-scores sont **invariants par troncature** des prix et des "
        "fondamentaux postérieurs (Δ = 0). Poids indiciels ancrés sur les "
        "`filing_date` (aucune donnée future). Winsorisation / Z-score en coupe "
        "transversale uniquement. Aucune utilisation de `ticker.info`."
    )
    st.warning(
        "**Limites résiduelles.** (1) `max_top10=None` : la concentration Top 10 "
        "n'est pas bornée (proche de celle de l'indice). (2) L'univers EGUS "
        "diffère d'un indice Growth USA non-ESG ; une part de la Tracking Error "
        "est incompressible. (3) Le turnover du 1er rebalancement est un artefact "
        "de construction depuis le cash. Voir `AUDIT_REPORT.md`."
    )

    audit_path = os.path.join(BASE_DIR, "AUDIT_REPORT.md")
    with st.expander("📋 Rapport d'audit quantitatif (AUDIT_REPORT.md)",
                     expanded=False):
        try:
            with open(audit_path, encoding="utf-8") as fh:
                st.markdown(fh.read())
        except FileNotFoundError:
            st.warning("AUDIT_REPORT.md introuvable à la racine du projet.")
        except Exception as exc:  # pragma: no cover
            st.warning(f"Impossible de lire AUDIT_REPORT.md : {exc}")


# ===========================================================================
# Onglet 4 — Stratégie de recherche « Pure Accélération »
# ===========================================================================
ACCEL_NAME = "Pure Accélération"
ACCEL_COLOR = "#2ECC71"
ACCEL_W_MAX_DEFAULT = 0.15
ACCEL_PILLARS = {"acceleration": 1.0, "momentum": 0.0, "quality": 0.0}


@st.cache_data(show_spinner="Backtest « Pure Accélération »…")
def compute_accel_strategy(w_max: float, n_titles: int, sector_band: float,
                           max_top10: float | None, tx_bps: float,
                           rf: float) -> dict:
    """Backtest de la stratégie de recherche 100 % Z_Acceleration (PIT)."""
    base = load_base()
    panel = load_panel()
    params = {
        "n_titles": int(n_titles),
        "sector_band": float(sector_band),
        "sector_pilot": True,
        "dw_min": float(config.DEFAULT_DW_MIN),
        "dw_max": float(config.DEFAULT_DW_MAX),
        "w_max": float(w_max),
        "buffer_rule": True,
        "max_top10": (None if max_top10 is None else float(max_top10)),
        "weights": dict(ACCEL_PILLARS),
    }
    portfolios = pf.build_all_portfolios(
        params, panel=panel, index_weights=base["index_weights"],
        index_sector_w=base["index_sector_w"])
    blk = bt.run_backtest({ACCEL_NAME: portfolios}, base["prices"],
                          base["benchmarks"], tx_cost_bps=float(tx_bps))[ACCEL_NAME]
    bench_nav = base["bench_blocks"][REFERENCE]["nav"]
    turnover_real = _drop_inception(blk["turnover"])
    stats_ = M.performance_stats(
        blk["nav"], bench_nav, turnover=turnover_real,
        weights_daily=blk["weights_daily"], index_weights=base["index_weights"],
        bench_returns=base["bench_blocks"][REFERENCE]["returns"], rf=float(rf))
    return {"portfolios": portfolios, "nav": blk["nav"], "returns": blk["returns"],
            "turnover": blk["turnover"], "turnover_real": turnover_real,
            "weights_daily": blk["weights_daily"], "stats": stats_,
            "bench_nav": bench_nav}


def _accel_rb_day(date, trading_days):
    pos = int(trading_days.searchsorted(pd.Timestamp(date), side="right")) - 1
    return trading_days[pos] if pos >= 0 else None


def _accel_quarter_returns(ser: pd.Series) -> pd.Series:
    q = ser.resample("QE").last().dropna()
    base = pd.concat([pd.Series([float(ser.iloc[0])], index=[ser.index[0]]), q])
    r = base.pct_change().dropna()
    r.index = pd.PeriodIndex(r.index, freq="Q")
    return r


def _accel_regression(rp: pd.Series, rb: pd.Series, rf: float = 0.0) -> dict:
    """OLS des rendements excédentaires quotidiens r_p = α + β·r_b (+ rf).

    Renvoie l'alpha de Jensen annualisé (intercept × 252) et des grandeurs
    additives en base **arithmétique** : ``port_excess_ann = β·bench_excess_ann
    + alpha_ann``.
    """
    rf_d = float(rf) / M.TRADING_DAYS
    al = pd.concat([(rp - rf_d).rename("p"), (rb - rf_d).rename("b")],
                   axis=1).dropna()
    p_, b_ = al["p"].to_numpy(), al["b"].to_numpy()
    res = stats.linregress(b_, p_)
    beta, alpha_d = float(res.slope), float(res.intercept)
    se_alpha = float(getattr(res, "intercept_stderr", np.nan))
    n = len(al)
    t_alpha = alpha_d / se_alpha if se_alpha and np.isfinite(se_alpha) else np.nan
    p_alpha = (float(2 * stats.t.sf(abs(t_alpha), df=max(n - 2, 1)))
               if np.isfinite(t_alpha) else np.nan)
    up, dn = b_ > 0, b_ < 0

    def _cond(mask):
        if mask.sum() < 3:
            return np.nan
        var_b = float(np.var(b_[mask], ddof=1))
        if var_b == 0:
            return np.nan
        return float(np.cov(b_[mask], p_[mask], ddof=1)[0, 1] / var_b)

    return {"beta": beta, "alpha_daily": alpha_d, "alpha_ann": alpha_d * M.TRADING_DAYS,
            "jensen_t": t_alpha, "jensen_p": p_alpha, "n": n,
            "r2": float(res.rvalue) ** 2,
            "resid_vol": float(np.std(p_ - beta * b_, ddof=1) * np.sqrt(M.TRADING_DAYS)),
            "bench_excess_ann": float(b_.mean() * M.TRADING_DAYS),
            "port_excess_ann": float(p_.mean() * M.TRADING_DAYS),
            "beta_up": _cond(up), "beta_down": _cond(dn), "corr": float(res.rvalue)}


def _accel_bootstrap(rp: pd.Series, rb: pd.Series, n_boot: int = 10000,
                     block: int = 21, seed: int = 7) -> dict:
    """Bootstrap par blocs (21 séances) des rendements actifs quotidiens.

    Préserve l'autocorrélation de court terme (cohérent avec l'audit PIT) ; IC
    95 % et P(alpha ≤ 0) sur l'alpha annualisé.
    """
    ar = (rp - rb).dropna().to_numpy()
    n = ar.size
    if n < block + 1:
        return {"ci_lo": np.nan, "ci_hi": np.nan, "p_neg": np.nan,
                "mean": np.nan, "draws": np.array([])}
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    draws = ar[idx].mean(axis=1) * M.TRADING_DAYS
    return {"ci_lo": float(np.percentile(draws, 2.5)),
            "ci_hi": float(np.percentile(draws, 97.5)),
            "p_neg": float(np.mean(draws <= 0.0)),
            "mean": float(ar.mean() * M.TRADING_DAYS), "draws": draws}


def _accel_active_tstat(rp: pd.Series, rb: pd.Series) -> dict:
    ar = (rp - rb).dropna()
    n = len(ar)
    if n < 3:
        return {"t": np.nan, "p": np.nan, "n": n, "mean_ann": np.nan}
    se = ar.std(ddof=1) / np.sqrt(n)
    t = float(ar.mean() / se) if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=n - 1)) if np.isfinite(t) else np.nan
    return {"t": t, "p": p, "n": n, "mean_ann": float(ar.mean() * M.TRADING_DAYS)}


def _accel_attribution(base: dict, portfolios: dict, sector_map: pd.Series) -> dict:
    """Brinson-Fachler par secteur + contribution par titre (moyennes trimestrielles)."""
    prices = base["prices"]
    iw = base["index_weights"]
    bench_nav = base["bench_blocks"][REFERENCE]["nav"]
    tdays = bench_nav.index
    order = sorted(portfolios)
    days = {d: _accel_rb_day(d, tdays) for d in order}
    sectors = list(config.GICS_SECTORS)
    sec_acc = {s: {k: [] for k in ("Wp", "Wb", "rp", "rb", "alloc", "select",
                                   "inter", "cp", "cb")} for s in sectors}
    tick = {}
    per = []
    recon = {"r_p_hold": [], "r_b_hold": [], "r_b_nav": [], "effect": []}
    for i, d0 in enumerate(order):
        day0 = days[d0]
        day1 = days[order[i + 1]] if i + 1 < len(order) else tdays[-1]
        if day0 is None or day1 is None or day1 <= day0:
            continue
        ret = (prices.loc[day1] / prices.loc[day0] - 1.0)
        w_p = portfolios[d0].astype(float); w_p = w_p[w_p > 0]
        w_b = _idx_weight_row(iw, d0).astype(float); w_b = w_b[w_b > 0]
        if w_p.empty or w_b.empty:
            continue
        r_b_nav = float(bench_nav.loc[day1] / bench_nav.loc[day0] - 1.0)
        sec_p = sector_map.reindex(w_p.index).fillna("__NA__")
        sec_b = sector_map.reindex(w_b.index).fillna("__NA__")
        Wp_map, Wb_map, rp_map, rb_map = {}, {}, {}, {}
        for s in sectors:
            mp, mb = sec_p == s, sec_b == s
            Wp = float(w_p[mp].sum()); Wb = float(w_b[mb].sum())
            Wp_map[s] = Wp; Wb_map[s] = Wb
            rp_map[s] = (float((w_p[mp] * ret.reindex(w_p.index[mp]).fillna(0.0)).sum()
                               / Wp) if Wp > 0 else np.nan)
            rb_map[s] = (float((w_b[mb] * ret.reindex(w_b.index[mb]).fillna(0.0)).sum()
                               / Wb) if Wb > 0 else np.nan)
        # Jambe benchmark « panier détenu » : rendement reconstruit sur les
        # titres N-PORT pondérés (et non la NAV de l'ETF, qui dérive).
        r_b_hold = float(sum(Wb_map[s] * rb_map[s] for s in sectors
                             if np.isfinite(rb_map[s])))
        r_p_hold = float(sum(Wp_map[s] * rp_map[s] for s in sectors
                             if np.isfinite(rp_map[s])))
        tot_a = tot_s = tot_i = 0.0
        for s in sectors:
            Wp, Wb = Wp_map[s], Wb_map[s]
            rb_s = rb_map[s] if np.isfinite(rb_map[s]) else r_b_hold
            rp_s = rp_map[s] if np.isfinite(rp_map[s]) else rb_s
            a = (Wp - Wb) * (rb_s - r_b_hold)
            se = Wb * (rp_s - rb_s)
            it = (Wp - Wb) * (rp_s - rb_s)
            sec_acc[s]["Wp"].append(Wp); sec_acc[s]["Wb"].append(Wb)
            sec_acc[s]["rp"].append(rp_s); sec_acc[s]["rb"].append(rb_s)
            sec_acc[s]["alloc"].append(a); sec_acc[s]["select"].append(se)
            sec_acc[s]["inter"].append(it)
            sec_acc[s]["cp"].append(Wp * rp_s); sec_acc[s]["cb"].append(Wb * rb_s)
            tot_a += a; tot_s += se; tot_i += it
        for t in w_b.index:
            tick[t] = tick.get(t, 0.0) - float(w_b[t]) * float(ret.get(t, 0.0))
        for t in w_p.index:
            tick[t] = tick.get(t, 0.0) + float(w_p[t]) * float(ret.get(t, 0.0))
        per.append({"date": d0, "Allocation": tot_a, "Sélection": tot_s,
                    "Interaction": tot_i})
        recon["r_p_hold"].append(r_p_hold); recon["r_b_hold"].append(r_b_hold)
        recon["r_b_nav"].append(r_b_nav)
        recon["effect"].append(tot_a + tot_s + tot_i)

    def _m(x):
        return float(np.mean(x)) if x else np.nan

    rows = {}
    for s in sectors:
        a = sec_acc[s]
        if not a["Wp"]:
            continue
        rows[s] = {"Wp": _m(a["Wp"]), "Wb": _m(a["Wb"]), "R_p": _m(a["rp"]),
                   "R_b": _m(a["rb"]), "Allocation": _m(a["alloc"]),
                   "Sélection": _m(a["select"]), "Interaction": _m(a["inter"]),
                   "Contrib_p": _m(a["cp"]), "Contrib_b": _m(a["cb"])}
    sec_df = pd.DataFrame(rows).T
    if not sec_df.empty:
        sec_df["Contrib_active"] = sec_df["Contrib_p"] - sec_df["Contrib_b"]
        sec_df = sec_df[sec_df[["Wp", "Wb"]].abs().sum(axis=1) > 1e-9]
    per_df = pd.DataFrame(per).set_index("date") if per else pd.DataFrame()
    tick_ser = pd.Series(tick, dtype=float).sort_values()
    recon_df = (pd.DataFrame(recon).mean() if per else pd.DataFrame())
    return {"sector": sec_df, "ticker": tick_ser, "periods": per_df,
            "recon": recon_df}


def _accel_sector_weights(res: dict, sector_map: pd.Series) -> pd.DataFrame:
    rows = {}
    for d, w in res["portfolios"].items():
        w = w[w > 0]
        sec = sector_map.reindex(w.index).fillna("__NA__")
        rows[d] = w.groupby(sec).sum().reindex(config.GICS_SECTORS).fillna(0.0)
    df = pd.DataFrame(rows).T
    df.index = pd.to_datetime(df.index).normalize()
    return df


def accel_controls(key_prefix: str = "acc_") -> dict:
    """Contrôles de la stratégie de recherche (dans l'onglet, pas la barre latérale)."""
    with st.expander("Paramètres de la stratégie de recherche", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        w_max = c1.slider("Poids max par titre (%)", 8.0, 20.0,
                          ACCEL_W_MAX_DEFAULT * 100, 0.5, key=f"{key_prefix}wmax")
        n_titles = c2.slider("Nombre de titres (N)", config.N_MIN, config.N_MAX,
                             config.DEFAULT_N, 1, key=f"{key_prefix}n")
        sector_band = c3.slider("Bande sectorielle (± %)", 0.0, 5.0, 4.0, 0.5,
                                key=f"{key_prefix}band")
        tx_bps = c4.slider("Coûts de transaction (bps)", 0.0, 50.0, 10.0, 1.0,
                           key=f"{key_prefix}tx")
        cap_on = st.toggle("Plafonner la concentration Top 10", value=False,
                           key=f"{key_prefix}cap",
                           help="Contrainte absente par défaut : la concentration "
                                "est une source d'alpha sur cette stratégie.")
        max_top10 = None
        if cap_on:
            max_top10 = st.slider("Plafond Top 10 (%)", 40.0, 70.0, 50.0, 1.0,
                                  key=f"{key_prefix}cap_val") / 100.0
    return {"w_max": float(w_max / 100.0), "n_titles": int(n_titles),
            "sector_band": float(sector_band / 100.0), "max_top10": max_top10,
            "tx_bps": float(tx_bps)}


def _accel_header(rf: float) -> None:
    st.markdown("### La stratégie — 100 % Accélération fondamentale")
    st.markdown(
        "Le composite est réduit à son **seul pilier Accélération** (moteur "
        "fondamental prospectif) : on ne récompense pas le *niveau* de croissance "
        "mais son **accélération** (le delta).\n\n"
        "`Z_Composite = Z_Acceleration = mean(Z_SalesAccel, Z_MarginExp, Z_AssetEff)`"
    )
    st.markdown(
        "| Métrique | Formule | Lecture |\n| --- | --- | --- |\n"
        "| `Sales_Accel` | `[CA_N/CA_N1 − 1] − [CA_N1/CA_N2 − 1]` | "
        "l'accélération du chiffre d'affaires |\n"
        "| `Margin_Exp` | `OI_N/CA_N − OI_N1/CA_N1` | l'expansion de la marge "
        "opérationnelle |\n"
        "| `Asset_Eff` | `CA_N/TA_N − CA_N1/TA_N1` (Financials → `ROA_N − ROA_N1`) "
        "| la rotation des actifs |\n\n"
        "**Intuition** : une entreprise qui accélère ses ventes, étend sa marge et "
        "fait mieux tourner son capital **améliore son économie sous-jacente** — "
        "souvent avant que le consensus ne l'intègre. Le risque assumé est le "
        "**retournement** (déception prospective)."
    )


def _accel_kpis(res: dict, params: dict, rf: float) -> None:
    s = res["stats"]
    st.subheader(f"Indicateurs clés vs {REFERENCE}")
    c = st.columns(6)
    c[0].metric("Rendement cumulé", fmt_stat("cum_return", s["cum_return"]))
    c[1].metric(f"Alpha vs {REFERENCE}", fmt_stat("alpha_ann", s["alpha_ann"]))
    c[2].metric("Tracking Error", fmt_stat("te_ann", s["te_ann"]))
    c[3].metric("Ratio d'information", fmt_stat("info_ratio", s["info_ratio"]))
    c[4].metric("Ratio de Sharpe", fmt_stat("sharpe", s["sharpe"]))
    c[5].metric("Max Drawdown", fmt_stat("max_dd", s["max_dd"]))

    ref = compute_accel_strategy(ACCEL_W_MAX_DEFAULT, config.DEFAULT_N, 0.04,
                                 None, params["tx_bps"], rf)
    keys = ["cum_return", "ann_return", "vol_ann", "sharpe", "alpha_ann", "te_ann",
            "info_ratio", "beta", "max_dd", "active_share", "top10_weight",
            "turnover_avg"]
    ref_lbl = f"Référence (w_max {ACCEL_W_MAX_DEFAULT * 100:.0f}% · N {config.DEFAULT_N})"
    cur_lbl = f"Courante (w_max {params['w_max'] * 100:.0f}% · N {params['n_titles']})"
    table = {ref_lbl: {STAT_LABELS[k]: fmt_stat(k, ref["stats"][k]) for k in keys},
             cur_lbl: {STAT_LABELS[k]: fmt_stat(k, res["stats"][k]) for k in keys}}
    df = pd.DataFrame(table, index=[STAT_LABELS[k] for k in keys])
    df.index.name = "Statistique"
    st.markdown("#### Comparaison à la configuration de référence")
    st.dataframe(df, use_container_width=True)
    st.caption(
        "La **référence** = stratégie concluante du balayage (pure accélération, "
        "`w_max` 15 %, N=40, bande 0,04, Top 10 libre). La concentration est une "
        "source d'alpha ici : plafonner le Top 10 ou augmenter N tend à réduire "
        "l'alpha."
    )


def render_accel_performance(res: dict, name: str = ACCEL_NAME,
                             color: str = ACCEL_COLOR) -> None:
    nav, bnav = res["nav"], res["bench_nav"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bnav.index, y=bnav.values, name=f"Benchmark {REFERENCE}",
                             line=dict(color=BENCH_COLOR, width=2.4, dash="dash")))
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name=name,
                             line=dict(color=color, width=2.2)))
    fig.update_yaxes(title_text="Base 100")
    base_layout(fig, height=430, title="Performances cumulées (Base 100)")

    act = (nav.reindex(bnav.index) / bnav - 1.0) * 100.0
    fig_rel = go.Figure(go.Scatter(
        x=act.index, y=act.values, name="Surperformance cumulée",
        line=dict(color=color, width=2.0),
        fill="tozeroy", fillcolor="rgba(46,204,113,0.12)"))
    fig_rel.add_hline(y=0.0, line=dict(color="#8B949E", width=1.2, dash="dot"))
    lo, hi = float(act.min()), float(act.max())
    pad = max(0.5, 0.12 * (hi - lo))
    fig_rel.update_yaxes(title_text="Surperformance cumulée (%)", ticksuffix=" %",
                         range=[min(lo, 0.0) - pad, hi + pad])
    base_layout(fig_rel, height=430,
                title=f"Alpha cumulé vs {REFERENCE} (%)")
    a, b = st.columns([1.3, 1])
    with a:
        st.plotly_chart(fig, use_container_width=True)
    with b:
        st.plotly_chart(fig_rel, use_container_width=True)

    dd = M.drawdown_series(nav) * 100.0
    ddb = M.drawdown_series(bnav) * 100.0
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=ddb.index, y=ddb.values, name=f"Benchmark {REFERENCE}",
                                line=dict(color=BENCH_COLOR, width=1.6, dash="dash")))
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name,
                                line=dict(color=color, width=1.6)))
    fig_dd.update_yaxes(title_text="Drawdown (%)")
    base_layout(fig_dd, height=320, title="Underwater — drawdowns quotidiens")
    st.plotly_chart(fig_dd, use_container_width=True)


def render_accel_stats(res: dict, rf: float, name: str = ACCEL_NAME,
                       file_name: str = "pure_acceleration_stats.csv") -> None:
    st.subheader("Statistiques usuelles")
    bblk = load_base()["bench_blocks"][REFERENCE]
    bstats = M.performance_stats(bblk["nav"], bblk["nav"],
                                 bench_returns=bblk["returns"], rf=float(rf))
    table = {name: {STAT_LABELS[k]: fmt_stat(k, res["stats"][k]) for k in STAT_ORDER},
             f"Benchmark {REFERENCE}": {STAT_LABELS[k]: fmt_stat(k, bstats[k])
                                        for k in STAT_ORDER}}
    df = pd.DataFrame(table, index=[STAT_LABELS[k] for k in STAT_ORDER])
    df.index.name = "Statistique"
    st.dataframe(df, use_container_width=True)
    st.download_button("Exporter les statistiques (CSV)",
                       data=df.to_csv().encode("utf-8"),
                       file_name=file_name, mime="text/csv")


def render_accel_quarterly(res: dict, name: str = ACCEL_NAME,
                           color: str = ACCEL_COLOR) -> None:
    st.subheader("Performances trimestrielles")
    nav, bnav = res["nav"], res["bench_nav"]
    qp = _accel_quarter_returns(nav)
    qb = _accel_quarter_returns(bnav)
    df = pd.DataFrame({"Stratégie": qp, "EGUS": qb})
    df["Actif"] = df["Stratégie"] - df["EGUS"]
    df["Relatif cumulé"] = (1 + qp).cumprod() / (1 + qb).cumprod() - 1
    df.index = [str(p) for p in df.index]
    df.loc["Cumulé"] = [(1 + qp).prod() - 1, (1 + qb).prod() - 1,
                        ((1 + qp).prod() - 1) - ((1 + qb).prod() - 1),
                        (1 + qp).prod() / (1 + qb).prod() - 1]
    disp = df.copy()
    for c in disp.columns:
        disp[c] = disp[c].map(lambda v: f"{v * 100:+.2f} %")
    st.dataframe(disp, use_container_width=True)
    st.caption(
        "« Relatif cumulé » compose les rendements géométriquement "
        "(`(1+qp)/(1+qb) − 1`). La 1re ligne (Q1) est un rendement **partiel** "
        "(du 25/01 à la clôture du trimestre)."
    )

    x = list(df.index[:-1])
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=(df["Stratégie"].iloc[:-1] * 100).values,
                         name=name, marker_color=color))
    fig.add_trace(go.Bar(x=x, y=(df["EGUS"].iloc[:-1] * 100).values,
                         name=f"Benchmark {REFERENCE}", marker_color=BENCH_COLOR))
    fig.add_trace(go.Scatter(x=x, y=(df["Actif"].iloc[:-1] * 100).values,
                             name="Actif", mode="lines+markers",
                             line=dict(color="#F0B429", width=2)))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Rendement trimestriel (%)")
    base_layout(fig, height=360,
                title="Rendements trimestriels — Stratégie vs EGUS (et actif)")
    st.plotly_chart(fig, use_container_width=True)


def render_accel_decomposition(res: dict, base: dict, rf: float) -> None:
    st.subheader("Décomposition de la performance")
    rp = res["returns"].reindex(res["nav"].index)
    rb = res["bench_nav"].pct_change(fill_method=None)
    reg = _accel_regression(rp, rb, rf=rf)
    bench_part = reg["beta"] * reg["bench_excess_ann"]
    jensen = reg["alpha_ann"]
    port_excess = reg["port_excess_ann"]

    c = st.columns(4)
    c[0].metric("Beta", f"{reg['beta']:.3f}")
    c[1].metric("R² (vs EGUS)", f"{reg['r2']:.3f}")
    c[2].metric("Alpha de Jensen (ann.)", f"{jensen * 100:+.2f} %")
    c[3].metric("Vol. résiduelle (ann.)", f"{reg['resid_vol'] * 100:.2f} %")
    dec = pd.DataFrame({"Contribution": [port_excess, bench_part, jensen]},
                       index=["Rendement excédentaire stratégie (ann., arith.)",
                              f"dont bêta × benchmark (β·R_{REFERENCE}, exc.)",
                              "dont alpha de Jensen (résiduel, ann.)"])
    disp = dec.copy()
    disp["Contribution"] = disp["Contribution"].map(lambda v: f"{v * 100:+.2f} %")
    st.dataframe(disp, use_container_width=True)
    st.caption(
        f"`R_exc_strat = β·R_exc_{REFERENCE} + α_Jensen` en base arithmétique "
        f"(annualisation ×252, taux sans risque rf = {rf * 100:.1f} %). "
        f"α_Jensen = intercept de la régression des excédents quotidiens ×252 ; "
        "volatilité résiduelle affichée à part (unité différente)."
    )

    sec_map = sector_map_series()
    frames = _accel_attribution(base, res["portfolios"], sec_map)
    sec = frames["sector"]
    st.markdown("#### Contribution à la performance par secteur")
    if sec.empty:
        st.info("Contribution sectorielle indisponible.")
    else:
        cc = sec[["Wp", "Wb", "Contrib_p", "Contrib_b", "Contrib_active"]].copy()
        cc = cc.sort_values("Contrib_active", ascending=False)
        disp2 = cc.copy()
        for col in ("Wp", "Wb", "Contrib_p", "Contrib_b", "Contrib_active"):
            disp2[col] = disp2[col].map(lambda v: f"{v * 100:+.2f} %")
        st.dataframe(disp2, use_container_width=True)
        fig = go.Figure(go.Bar(x=list(cc.index),
                               y=cc["Contrib_active"].values * 100,
                               marker_color=["#2ECC71" if v >= 0 else "#E74C3C"
                                             for v in cc["Contrib_active"].values]))
        fig.update_yaxes(title_text="Contribution active moyenne (pts %)")
        fig.update_xaxes(tickangle=-35)
        base_layout(fig, height=380,
                    title="Contribution active par secteur (moyenne / trimestre)")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### Principaux contributeurs et détracteurs (titres)")
    tick = frames["ticker"]
    if tick.empty:
        st.info("Contributions par titre indisponibles.")
    else:
        top = tick.sort_values(ascending=False).head(8)
        bot = tick.sort_values().head(8)
        comb = pd.concat([top, bot])
        colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in comb.values]
        fig = go.Figure(go.Bar(x=comb.values * 100, y=list(comb.index),
                               orientation="h", marker_color=colors,
                               text=[f"{v * 100:+.2f}" for v in comb.values],
                               textposition="outside"))
        fig.update_xaxes(title_text="Contribution active cumulée (pts %)")
        base_layout(fig, height=430,
                    title="Top / flop contributeurs au rendement actif")
        st.plotly_chart(fig, use_container_width=True)


def render_accel_attribution(res: dict, base: dict) -> None:
    st.subheader("Attribution — allocation & sélection (Brinson-Fachler)")
    frames = _accel_attribution(base, res["portfolios"], sector_map_series())
    sec = frames["sector"]
    if sec.empty:
        st.info("Attribution indisponible (données insuffisantes).")
        return
    disp = sec[["Allocation", "Sélection", "Interaction"]].copy()
    disp["Total"] = disp.sum(axis=1)
    disp = disp.sort_values("Allocation", ascending=False)
    disp_show = disp.copy()
    for c in disp_show.columns:
        disp_show[c] = disp_show[c].map(lambda v: f"{v * 1e4:+.1f} bps")
    st.dataframe(disp_show, use_container_width=True)
    x = list(disp.index)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=x, y=disp["Allocation"].values * 1e4, name="Allocation",
                         marker_color="#4C9AFF"))
    fig.add_trace(go.Bar(x=x, y=disp["Sélection"].values * 1e4, name="Sélection",
                         marker_color="#2ECC71"))
    fig.add_trace(go.Bar(x=x, y=disp["Interaction"].values * 1e4, name="Interaction",
                         marker_color="#F0B429"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Effet moyen par trimestre (bps)")
    fig.update_xaxes(tickangle=-35)
    base_layout(fig, height=420, title="Attribution sectorielle moyenne (par trimestre)")
    st.plotly_chart(fig, use_container_width=True)

    per = frames["periods"]
    if not per.empty:
        fig2 = go.Figure()
        for col, col_color in (("Allocation", "#4C9AFF"),
                               ("Sélection", "#2ECC71"),
                               ("Interaction", "#F0B429")):
            fig2.add_trace(go.Bar(x=[str(p) for p in per.index],
                                  y=per[col].values * 1e4, name=col,
                                  marker_color=col_color))
        fig2.update_layout(barmode="stack")
        fig2.update_yaxes(title_text="Effet par trimestre (bps)")
        fig2.update_xaxes(tickangle=-45)
        base_layout(fig2, height=360, title="Attribution par trimestre (décomposition)")
        st.plotly_chart(fig2, use_container_width=True)
    recon = frames.get("recon")
    if recon is not None and not recon.empty:
        st.markdown("#### Réconciliation")
        r = recon
        rec = pd.DataFrame({
            "Rendement": [r["r_p_hold"], r["r_b_hold"], r["effect"],
                          r["r_b_nav"], r["r_p_hold"] - r["r_b_nav"]]},
            index=["Stratégie (panier, /trim.)",
                   "Benchmark N-PORT reconstruit (panier, /trim.)",
                   "Σ Allocation + Sélection + Interaction",
                   "EGUS NAV (ETF, /trim.)",
                   "Actif stratégie vs EGUS NAV (référence affichée)"])
        rec_show = rec.copy()
        rec_show["Rendement"] = rec_show["Rendement"].map(lambda v: f"{v * 100:+.2f} %")
        st.dataframe(rec_show, use_container_width=True)
        st.caption(
            "L'attribution réconcilie la stratégie contre son **panier benchmark "
            "reconstruit** (titres N-PORT), base homogène à la jambe portefeuille. "
            "La dérive du panier N-PORT vs la **NAV de l'ETF EGUS** (dernière ligne) "
            "explique l'écart résiduel avec l'actif affiché ailleurs."
        )
    st.caption("Allocation = timing sectoriel ; Sélection = choix de titres ; "
               "Interaction = croisement.")


def render_accel_significance(res: dict, rf: float) -> None:
    st.subheader("Significativité de l'alpha")
    rp = res["returns"].reindex(res["nav"].index)
    rb = res["bench_nav"].pct_change(fill_method=None)
    tt = _accel_active_tstat(rp, rb)
    reg = _accel_regression(rp, rb, rf=rf)
    boot = _accel_bootstrap(rp, rb)
    c = st.columns(4)
    c[0].metric("Alpha brut (ann.)", f"{tt['mean_ann'] * 100:+.2f} %")
    c[1].metric("t-stat (moy. actif quotidien)", f"{tt['t']:.2f}")
    c[2].metric("p-value (bilatéral)", f"{tt['p']:.3f}")
    c[3].metric("Jensen alpha t-stat", f"{reg['jensen_t']:.2f}")
    verdict = ("rejetée à 5 % — alpha significatif"
               if (np.isfinite(tt["p"]) and tt["p"] < 0.05)
               else "non rejetée à 5 % — alpha non significatif")
    st.markdown(
        f"**H₀ : alpha brut = 0** (moyenne des rendements actifs, non ajustée du "
        f"bêta). Sur {tt['n']} séances, p-value bilatérale = "
        f"**{tt['p']:.3f}** → H₀ **{verdict}**. Alpha de Jensen (ajusté du bêta) "
        f"t = {reg['jensen_t']:.2f} (p = {reg['jensen_p']:.3f})."
    )
    st.markdown(
        f"**Bootstrap par blocs** (10 000 rééchantillonnages de 21 séances) — IC "
        f"95 % de l'alpha brut annualisé : "
        f"**[{boot['ci_lo'] * 100:+.2f} % ; {boot['ci_hi'] * 100:+.2f} %]**, "
        f"P(alpha ≤ 0) = **{boot['p_neg'] * 100:.0f} %**. Bloc de 21 j pour "
        "préserver l'autocorrélation quotidienne (cf. audit PIT)."
    )
    if np.isfinite(tt["p"]) and tt["p"] >= 0.05:
        st.warning("Alpha **non significatif** au seuil de 5 % : la surperformance "
                   "n'est pas distinguable de zéro sur cette fenêtre (11 "
                   "rebalancements). Résultat **directionnel**.")
    else:
        st.success("Alpha significatif au seuil de 5 %.")

    draws = boot.get("draws", np.array([]))
    if draws.size:
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=draws * 100, nbinsx=60,
                                   marker_color="#4C9AFF", opacity=0.85,
                                   name="Alpha bootstrap"))
        fig.add_vline(x=0.0, line=dict(color="#E74C3C", width=1.6, dash="dash"))
        if np.isfinite(boot["ci_lo"]):
            fig.add_vline(x=boot["ci_lo"] * 100, line=dict(color="#8B949E", width=1.2))
            fig.add_vline(x=boot["ci_hi"] * 100, line=dict(color="#8B949E", width=1.2))
        fig.update_xaxes(title_text="Alpha annualisé (%)")
        fig.update_yaxes(title_text="Fréquence")
        base_layout(fig, height=340,
                    title="Distribution bootstrap de l'alpha (IC 95 %)")
        st.plotly_chart(fig, use_container_width=True)


def render_accel_beta(res: dict, rf: float) -> None:
    st.subheader("Bêta global, bêta haussier & baissier")
    rp = res["returns"].reindex(res["nav"].index)
    rb = res["bench_nav"].pct_change(fill_method=None)
    reg = _accel_regression(rp, rb, rf=rf)
    s = res["stats"]
    c = st.columns(6)
    c[0].metric("Bêta global", f"{reg['beta']:.3f}")
    c[1].metric("Bêta haussier", f"{reg['beta_up']:.3f}")
    c[2].metric("Bêta baissier", f"{reg['beta_down']:.3f}")
    c[3].metric("Up Capture", fmt_stat("up_capture", s["up_capture"]))
    c[4].metric("Down Capture", fmt_stat("down_capture", s["down_capture"]))
    c[5].metric("Corrélation", f"{reg['corr']:.3f}")
    asym = reg["beta_up"] - reg["beta_down"]
    st.caption(f"Asymétrie de bêta (haussier − baissier) = **{asym:+.3f}**. "
               + ("Capture d'upside favorable." if asym > 0
                  else "Sensibilité défavorable aux replis."))

    al = pd.concat([rp.rename("p"), rb.rename("b")], axis=1).dropna()
    up = al[al["b"] > 0]
    dn = al[al["b"] < 0]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=up["b"] * 100, y=up["p"] * 100, mode="markers",
                             name="Jours haussiers", marker=dict(color="#2ECC71", size=5,
                             opacity=0.6)))
    fig.add_trace(go.Scatter(x=dn["b"] * 100, y=dn["p"] * 100, mode="markers",
                             name="Jours baissiers", marker=dict(color="#E74C3C", size=5,
                             opacity=0.6)))
    xs = np.linspace(float(al["b"].min()) * 100, float(al["b"].max()) * 100, 50)
    fig.add_trace(go.Scatter(x=xs, y=(reg["alpha_ann"] / M.TRADING_DAYS + reg["beta"] * xs / 100) * 100,
                             name="OLS (α + β·x)",
                             line=dict(color="#FFFFFF", width=1.4, dash="dash")))
    fig.update_xaxes(title_text=f"Rendement {REFERENCE} (%)")
    fig.update_yaxes(title_text="Rendement stratégie (%)")
    base_layout(fig, height=380, title="Rendements quotidiens — stratégie vs EGUS")
    st.plotly_chart(fig, use_container_width=True)

    roll = al.groupby(al.index.to_period("Q")).apply(
        lambda g: g["p"].cov(g["b"]) / g["b"].var() if g["b"].var() > 0 else np.nan)
    fig_r = go.Figure(go.Bar(x=[str(p) for p in roll.index], y=roll.values,
                             marker_color="#4C9AFF"))
    fig_r.add_hline(y=1.0, line=dict(color="#8B949E", width=1.2, dash="dot"))
    fig_r.update_yaxes(title_text="Bêta")
    base_layout(fig_r, height=320, title="Bêta glissant par trimestre")
    st.plotly_chart(fig_r, use_container_width=True)


def render_accel_sectors(res: dict, base: dict, name: str = ACCEL_NAME,
                         color: str = ACCEL_COLOR) -> None:
    st.subheader(f"Allocation sectorielle vs {REFERENCE} (dernière date)")
    isw = base["index_sector_w"]
    sec_map = sector_map_series()
    last = max(res["portfolios"])
    w = res["portfolios"][last]; w = w[w > 0]
    sec = sec_map.reindex(w.index).fillna("__NA__")
    port_s = w.groupby(sec).sum().reindex(config.GICS_SECTORS).fillna(0.0)
    idx_s = isw.loc[pd.Timestamp(last)].reindex(config.GICS_SECTORS).fillna(0.0)

    fig = go.Figure()
    fig.add_trace(go.Bar(x=config.GICS_SECTORS, y=idx_s.values * 100,
                         name=f"Benchmark {REFERENCE}", marker_color=BENCH_COLOR))
    fig.add_trace(go.Bar(x=config.GICS_SECTORS, y=port_s.values * 100,
                         name=name, marker_color=color))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Poids (%)")
    fig.update_xaxes(tickangle=-35)
    base_layout(fig, height=420,
                title=f"Poids sectoriels — {name} vs EGUS ({last:%d/%m/%Y})")
    st.plotly_chart(fig, use_container_width=True)

    active = (port_s - idx_s) * 100
    active = active[active.abs() > 1e-9].sort_values()
    colors = ["#E74C3C" if v < 0 else "#2ECC71" for v in active.values]
    fig_a = go.Figure(go.Bar(x=active.values, y=list(active.index), orientation="h",
                             marker_color=colors,
                             text=[f"{v:+.2f}%" for v in active.values],
                             textposition="outside"))
    fig_a.update_xaxes(title_text="Écart actif Δw (%)")
    base_layout(fig_a, height=420, title="Écarts actifs sectoriels vs EGUS")
    st.plotly_chart(fig_a, use_container_width=True)

    sw = _accel_sector_weights(res, sec_map)
    idx_sw = base["index_sector_w"].reindex(sw.index).fillna(0.0)
    idx_sw = idx_sw.reindex(columns=sw.columns).fillna(0.0)
    act_ts = (sw - idx_sw) * 100
    fig_h = go.Figure(go.Heatmap(
        z=act_ts.T.values, x=[f"{pd.Timestamp(d):%d/%m/%y}" for d in act_ts.index],
        y=list(act_ts.columns), colorscale="RdYlGn", zmid=0.0,
        colorbar=dict(title="Δw (%)")))
    base_layout(fig_h, height=420,
                title="Écarts actifs sectoriels dans le temps (Δw %)")
    st.plotly_chart(fig_h, use_container_width=True)


def render_accel_top10(res: dict, base: dict, panel: pd.DataFrame,
                       name: str = ACCEL_NAME, color: str = ACCEL_COLOR,
                       score_col: str = "Z_Acceleration",
                       score_label: str = "Z_Accélération") -> None:
    st.subheader(f"Top 10 titres vs {REFERENCE} (dernière date)")
    sec_map = sector_map_series()
    last = max(res["portfolios"])
    w = res["portfolios"][last]; w = w[w > 0].sort_values(ascending=False)
    idx_w = _idx_weight_row(base["index_weights"], last)
    top10 = w.head(10)
    rows = []
    for t, wp in top10.items():
        wb = float(idx_w.get(t, 0.0))
        rows.append({"Titre": t, "Secteur": sec_map.get(t, "n/a"),
                     f"Poids {name}": f"{wp * 100:.2f} %",
                     f"Poids {REFERENCE}": f"{wb * 100:.2f} %",
                     "Actif Δw": f"{(wp - wb) * 100:+.2f} pts"})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    top10_w = float(top10.sum())
    top10_idx = float(idx_w.sort_values(ascending=False).head(10).sum())
    m = st.columns(4)
    m[0].metric("Top 10 stratégie", f"{top10_w * 100:.1f} %")
    m[1].metric("Top 10 EGUS", f"{top10_idx * 100:.1f} %")
    m[2].metric("Concentration relative", f"{(top10_w - top10_idx) * 100:+.1f} pts")
    m[3].metric("Nb titres portefeuille", f"{len(w)}")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(top10.index), y=top10.values * 100,
                         name=name, marker_color=color))
    fig.add_trace(go.Bar(x=list(top10.index),
                         y=idx_w.reindex(top10.index).fillna(0.0).values * 100,
                         name=f"Benchmark {REFERENCE}", marker_color=BENCH_COLOR))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Poids (%)")
    fig.update_xaxes(tickangle=-45)
    base_layout(fig, height=380, title="Top 10 — poids stratégie vs EGUS")
    st.plotly_chart(fig, use_container_width=True)

    cols = pd.Index(w.index).union(idx_w.index)
    active = (w.reindex(cols).fillna(0.0) - idx_w.reindex(cols).fillna(0.0)) * 100
    active = active[active.abs() > 1e-9]
    over = active.sort_values(ascending=False).head(12)
    under = active.sort_values().head(12)
    fig_a = go.Figure()
    fig_a.add_trace(go.Bar(x=over.values, y=list(over.index), orientation="h",
                           name="Surpondérations", marker_color="#2ECC71"))
    fig_a.add_trace(go.Bar(x=under.values, y=list(under.index), orientation="h",
                           name="Sous-pondérations", marker_color="#E74C3C"))
    fig_a.update_xaxes(title_text="Poids actif Δw (%)")
    base_layout(fig_a, height=560,
                title=f"Top 12 sur/sous-pondérations actives vs EGUS ({last:%d/%m/%Y})")
    st.plotly_chart(fig_a, use_container_width=True)

    try:
        panel_d = panel.xs(pd.Timestamp(last).normalize(), level=0)
        z = panel_d.reindex(top10.index)[[score_col]].copy()
        z["Poids"] = top10
        z = z.rename(columns={score_col: score_label})
        z["Poids"] = z["Poids"].map(lambda v: f"{v * 100:.2f} %")
        z[score_label] = z[score_label].map(lambda v: f"{v:+.2f}")
        st.caption("Scores du Top 10 à la dernière date de rebalancement.")
        st.dataframe(z, use_container_width=True)
    except (KeyError, ValueError):
        pass


def render_tab_accel(base: dict, rf: float) -> None:
    """Onglet de recherche : stratégie Pure Accélération (w_max configurable)."""
    st.subheader("Stratégie de recherche — Pure Accélération")
    st.caption(
        "Composite réduit au pilier **Accélération** (100 %), plafond de "
        f"concentration par défaut **w_max = {ACCEL_W_MAX_DEFAULT * 100:.0f} %**, "
        f"benchmark **{REFERENCE}**. Backtest trimestriel Point-in-Time (11 "
        "rebalancements)."
    )
    params = accel_controls()
    res = compute_accel_strategy(params["w_max"], params["n_titles"],
                                 params["sector_band"], params["max_top10"],
                                 params["tx_bps"], rf)
    with st.expander("Méthodologie du backtest (PIT strict)", expanded=False):
        st.markdown(
            "- **Dates de rebalancement** : 11 `filing_date` trimestriels PIT "
            f"({pd.Timestamp(config.PERF_START):%d/%m/%Y} → {config.SIM_END:%d/%m/%Y}).\n"
            "- **Scores** : Z-scores cross-sectionnels par date, calculés "
            "**uniquement** sur les fondamentaux publiés (`pub_date ≤ t`) et les "
            "prix ≤ t — aucun look-ahead.\n"
            "- **Poids** : appliqués à partir de **t+1**, puis dérive **buy & hold** "
            "jusqu'au rebalancement suivant.\n"
            "- **Coûts** : `tx_bps` × turnover (`½·Σ|w_cible − w_dérivé|`) par "
            "rebalancement.\n"
            "- **Contraintes** : `w_max` (plafond par titre), bande sectorielle "
            f"±{config.DEFAULT_SECTOR_BAND:.0%} vs GICS de l'indice, "
            "`dw_min`/`dw_max`, buffer rule.\n"
            "- **Benchmark** : EGUS (iShares ESG Aware MSCI USA Growth), NAV "
            "Base 100 au 25/01/2024."
        )
    _accel_header(rf)
    _accel_kpis(res, params, rf)

    s1, s2, s3, s4 = st.tabs([
        "Performance & Stats",
        "Décomposition & Attribution",
        "Alpha & Bêta",
        "Secteurs & Top 10",
    ])
    with s1:
        render_accel_performance(res)
        render_accel_stats(res, rf)
        render_accel_quarterly(res)
    with s2:
        render_accel_decomposition(res, base, rf)
        render_accel_attribution(res, base)
    with s3:
        render_accel_significance(res, rf)
        render_accel_beta(res, rf)
    with s4:
        render_accel_sectors(res, base)
        render_accel_top10(res, base, load_panel())


# ===========================================================================
# Onglet 5 — Stratégie de recherche « G-Score de Mohanram »
# ===========================================================================
GSCORE_NAME = "G-Score (Mohanram)"
GSCORE_COLOR = "#F0B429"
GSCORE_W_MAX_DEFAULT = 0.15
GSCORE_CRITERIA_LABELS = {
    "G_Roa": "G1 · ROA > médiane",
    "G_Cfo": "G2 · CFO/TA > médiane",
    "G_Accruals": "G3 · CFO > résultat net",
    "G_RoaVar": "G4 · Stabilité ROA (σ faible)",
    "G_SalesVar": "G5 · Stabilité ΔCA (σ faible)",
    "G_Rd": "G6 · R&D/TA > médiane",
    "G_Capex": "G7 · CAPEX/TA > médiane",
}


@st.cache_data(show_spinner="Chargement du panel G-Score…")
def load_gscore_panel() -> pd.DataFrame:
    """Panel G-Score PIT (MultiIndex date x ticker), cache disque via src.gscore."""
    return gscore.build_gscore_panel()


@st.cache_data(show_spinner="Backtest « G-Score »…")
def compute_gscore_strategy(w_max: float, n_titles: int, sector_band: float,
                            max_top10: float | None, tx_bps: float,
                            rf: float, score_field: str = "Z_GScore") -> dict:
    """Backtest de la stratégie de recherche sélectionnant sur ``Z_GScore`` (PIT).

    Le G-Score se substitue au composite des trois piliers via le paramètre
    ``score_series`` de ``portfolio.build_all_portfolios`` ; toute la construction
    Core + Satellites reste inchangée.
    """
    base = load_base()
    panel = load_gscore_panel()
    params = {
        "n_titles": int(n_titles),
        "sector_band": float(sector_band),
        "sector_pilot": True,
        "dw_min": float(config.DEFAULT_DW_MIN),
        "dw_max": float(config.DEFAULT_DW_MAX),
        "w_max": float(w_max),
        "buffer_rule": True,
        "max_top10": (None if max_top10 is None else float(max_top10)),
    }
    portfolios = pf.build_all_portfolios(
        params, panel=panel, index_weights=base["index_weights"],
        index_sector_w=base["index_sector_w"],
        score_series=panel[score_field])
    blk = bt.run_backtest({GSCORE_NAME: portfolios}, base["prices"],
                          base["benchmarks"], tx_cost_bps=float(tx_bps))[GSCORE_NAME]
    bench_nav = base["bench_blocks"][REFERENCE]["nav"]
    turnover_real = _drop_inception(blk["turnover"])
    stats_ = M.performance_stats(
        blk["nav"], bench_nav, turnover=turnover_real,
        weights_daily=blk["weights_daily"], index_weights=base["index_weights"],
        bench_returns=base["bench_blocks"][REFERENCE]["returns"], rf=float(rf))
    return {"portfolios": portfolios, "nav": blk["nav"], "returns": blk["returns"],
            "turnover": blk["turnover"], "turnover_real": turnover_real,
            "weights_daily": blk["weights_daily"], "stats": stats_,
            "bench_nav": bench_nav}


@st.cache_data(show_spinner="Diagnostics G-Score (G_Core vs G_Ext, Qualité)…")
def compute_gscore_diagnostics(rf: float) -> dict:
    """Diagnostics de risque du G-Score : corrélation à Qualité/GARP, dispersion,
    variantes G_Core (G1-G5) / G_Ext (G1-G7), asymétrie de couverture."""
    gpanel = load_gscore_panel()
    fpanel = load_panel()
    corr = gscore.gscore_quality_correlation(gpanel, fpanel)
    disp = gscore.gscore_dispersion(gpanel)
    base = load_base()

    def _active_share(score_field):
        params = {"n_titles": config.DEFAULT_N, "sector_band": 0.04,
                  "sector_pilot": True, "dw_min": config.DEFAULT_DW_MIN,
                  "dw_max": config.DEFAULT_DW_MAX, "w_max": GSCORE_W_MAX_DEFAULT,
                  "buffer_rule": True, "max_top10": None}
        ports = pf.build_all_portfolios(
            params, panel=gpanel, index_weights=base["index_weights"],
            index_sector_w=base["index_sector_w"], score_series=gpanel[score_field])
        blk = bt.run_backtest({"s": ports}, base["prices"], base["benchmarks"],
                              tx_cost_bps=10.0)["s"]
        st_ = M.performance_stats(
            blk["nav"], base["bench_blocks"][REFERENCE]["nav"],
            weights_daily=blk["weights_daily"], index_weights=base["index_weights"],
            turnover=_drop_inception(blk["turnover"]), rf=float(rf))
        return st_

    core = _active_share("Z_GScore_Core")
    ext = _active_share("Z_GScore_Ext")
    both = gpanel[["Z_GScore_Core", "Z_GScore_Ext"]].dropna()
    corr_core_ext = (float(both.corr().iloc[0, 1]) if len(both) > 2 else np.nan)
    return {"corr": corr, "disp": disp, "core_stats": core, "ext_stats": ext,
            "corr_core_ext": corr_core_ext}


def render_gscore_diagnostics(res: dict, rf: float) -> None:
    st.subheader("Diagnostics de risque & chevauchement")
    diag = compute_gscore_diagnostics(float(rf))
    corr = diag["corr"]
    if not corr.empty:
        avg_p = float(corr["Pearson"].mean())
        avg_s = float(corr["Spearman"].mean())
        c = st.columns(4)
        c[0].metric("Corr. moyenne (G-Score, Qualité)",
                    f"{avg_p:+.2f}", help="Pearson, moyenne des dates")
        c[1].metric("Corr. rang (Spearman)", f"{avg_s:+.2f}")
        c[2].metric("Corr. G_Core / G_Ext", f"{diag['corr_core_ext']:+.2f}",
                    help="Entre le score cœur (G1-G5) et l'extension (G1-G7)")
        c[3].metric("Active Share G_Core vs G_Ext",
                    f"{abs(diag['core_stats']['active_share'] - diag['ext_stats']['active_share']):.1%}"
                    if np.isfinite(diag["core_stats"]["active_share"])
                    and np.isfinite(diag["ext_stats"]["active_share"]) else "n/a")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=[f"{pd.Timestamp(d):%d/%m/%y}" for d in corr.index],
                             y=corr["Pearson"].values, name="Pearson",
                             marker_color=GSCORE_COLOR))
        fig.add_trace(go.Scatter(x=[f"{pd.Timestamp(d):%d/%m/%y}" for d in corr.index],
                                 y=corr["Spearman"].values, name="Spearman",
                                 mode="lines+markers",
                                 line=dict(color=BENCH_COLOR, width=2)))
        fig.update_yaxes(title_text="Corrélation", range=[-0.2, 1.0])
        fig.update_xaxes(tickangle=-45)
        base_layout(fig, height=340,
                    title="Corrélation G-Score vs pilier Qualité/GARP, par date")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Un G-Score largement colinéaire à Qualité/GARP n'ajouterait qu'un "
            "**proxy** : la corrélation mesure ici l'apport d'information *distinct*."
        )

    st.markdown("#### Variantes de couverture — G_Core (G1-G5) vs G_Ext (G1-G7)")
    cs, es = diag["core_stats"], diag["ext_stats"]
    keys = ["cum_return", "ann_return", "sharpe", "alpha_ann", "te_ann",
            "info_ratio", "active_share", "top10_weight"]
    tab = {"G_Core (G1-G5)": {STAT_LABELS[k]: fmt_stat(k, cs[k]) for k in keys},
           "G_Ext (G1-G7)": {STAT_LABELS[k]: fmt_stat(k, es[k]) for k in keys}}
    df = pd.DataFrame(tab, index=[STAT_LABELS[k] for k in keys])
    df.index.name = "Statistique"
    st.dataframe(df, use_container_width=True)
    st.caption(
        "G_Core = G1-G5 (jamais manquants) ; G_Ext = G1-G7 (ajoute R&D/CAPEX, "
        "convention « manquant = 0 »). L'écart isole le **poids marginal** des "
        "critères d'investissement."
    )

    disp = diag["disp"]
    if not disp.empty:
        st.markdown("#### Dispersion transversale du score entier (0-7)")
        share_cols = [f"p_{k}" for k in range(8)]
        fig = go.Figure()
        for k, col in enumerate(share_cols):
            fig.add_trace(go.Bar(x=[f"{pd.Timestamp(d):%d/%m/%y}" for d in disp.index],
                                 y=disp[col].values * 100, name=f"G={k}"))
        fig.update_layout(barmode="stack")
        fig.update_yaxes(title_text="Part des titres (%)")
        fig.update_xaxes(tickangle=-45)
        base_layout(fig, height=380,
                    title="Répartition des titres par score entier (par date)")
        st.plotly_chart(fig, use_container_width=True)


def render_gscore_scores(res: dict, base: dict, panel: pd.DataFrame) -> None:
    st.subheader("Décomposition du score — Top 10 titres")
    last = max(res["portfolios"])
    w = res["portfolios"][last]; w = w[w > 0].sort_values(ascending=False)
    top10 = w.head(10)
    try:
        pd_date = panel.xs(pd.Timestamp(last).normalize(), level=0)
        sub = pd_date.reindex(top10.index)
        rows = []
        for t in top10.index:
            rec = {"Titre": t, "Poids": f"{top10[t] * 100:.2f} %",
                   "Score G": (int(sub.loc[t, "G_Count"])
                               if pd.notna(sub.loc[t, "G_Count"]) else "n/a"),
                   "G_N (évaluables)": (int(sub.loc[t, "G_N"])
                                        if pd.notna(sub.loc[t, "G_N"]) else "n/a")}
            for c, lab in GSCORE_CRITERIA_LABELS.items():
                v = sub.loc[t, c]
                rec[lab] = ("✓" if v == 1 else ("✗" if v == 0 else "—"))
            rows.append(rec)
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(
            "✓/✗ = critère validé / non validé ; « — » = critère non évaluable "
            "(R&D/CAPEX non rapportés ou historique insuffisant pour G4/G5)."
        )
    except (KeyError, ValueError):
        st.info("Décomposition indisponible pour la dernière date.")

    n_eff = st.columns(3)
    if not panel.empty:
        cnt = panel.groupby(level="date")["G_Count"].mean()
        n_eff[0].metric("Score G moyen (univers)", f"{cnt.mean():.2f} / 7")
        hi = panel[["G_Count"]].dropna()
        n_eff[1].metric("Titres à G ≥ 5 (moy.)", f"{int((hi['G_Count'] >= 5).sum() / max(len(panel.index.levels[0]), 1))}")
        n_eff[2].metric("Titres à G = 7 (total)", f"{int((hi['G_Count'] == 7).sum())}")


def _gscore_kpis(res: dict, params: dict) -> None:
    s = res["stats"]
    st.subheader(f"Indicateurs clés vs {REFERENCE}")
    c = st.columns(6)
    c[0].metric("Rendement cumulé", fmt_stat("cum_return", s["cum_return"]))
    c[1].metric(f"Alpha vs {REFERENCE}", fmt_stat("alpha_ann", s["alpha_ann"]))
    c[2].metric("Tracking Error", fmt_stat("te_ann", s["te_ann"]))
    c[3].metric("Ratio d'information", fmt_stat("info_ratio", s["info_ratio"]))
    c[4].metric("Active Share", fmt_stat("active_share", s["active_share"]))
    c[5].metric("Max Drawdown", fmt_stat("max_dd", s["max_dd"]))


def render_tab_gscore(base: dict, rf: float) -> None:
    """Onglet de recherche : sélection par G-Score de Mohanram (score entier 0-7)."""
    st.subheader("Stratégie de recherche — G-Score de Mohanram")
    st.caption(
        "Score **entier 0-7** (G1-G7) mesurant la **qualité fondamentale** "
        "persistante (Mohanram, 2005). Sélection directe sur `Z_GScore` "
        "(Z-score intra-sectoriel du score), plafond par défaut "
        f"**w_max = {GSCORE_W_MAX_DEFAULT * 100:.0f} %**, benchmark **{REFERENCE}**. "
        "Backtest trimestriel Point-in-Time (11 rebalancements). Score de "
        "**recherche** : la baseline à trois piliers reste inchangée. "
        "L'onglet affiche l'**Active Share** vs EGUS et les diagnostics de "
        "chevauchement avec Qualité/GARP."
    )
    params = accel_controls("gsc_")
    res = compute_gscore_strategy(params["w_max"], params["n_titles"],
                                  params["sector_band"], params["max_top10"],
                                  params["tx_bps"], rf)
    with st.expander("Méthodologie du G-Score (PIT strict)", expanded=False):
        st.markdown(
            "- **G1** ROA > médiane sectorielle ; **G2** CFO/Actifs > médiane ; "
            "**G3** CFO > résultat net (qualité des accruals) ; **G4** volatilité "
            "du ROA sur 20 trimestres < médiane ; **G5** volatilité de la "
            "croissance du CA < médiane ; **G6** R&D/Actifs > médiane ; "
            "**G7** CAPEX/Actifs > médiane (convention : R&D/CAPEX manquants = 0).\n"
            "- **G8 (publicité) exclu** : donnée quasi indisponible en XBRL.\n"
            "- Score **renormalisé sur 7** selon le nombre de critères "
            "évaluables (`G_Count / G_N × 7`) pour rester comparable entre titres.\n"
            "- **Médianes sectorielles** en coupe transversale à `t` (3 niveaux : "
            "secteur GICS → super-secteur → univers), titres non publiés exclus.\n"
            "- **PIT strict** : seuls `pub_date ≤ t` et `prix ≤ t` sont utilisés — "
            "aucun look-ahead."
        )
    _gscore_kpis(res, params)
    s1, s2 = st.tabs(["Performance & Diagnostics", "Secteurs & Titres"])
    with s1:
        render_accel_performance(res, name=GSCORE_NAME, color=GSCORE_COLOR)
        render_accel_stats(res, rf, name=GSCORE_NAME,
                           file_name="gscore_stats.csv")
        render_accel_quarterly(res, name=GSCORE_NAME, color=GSCORE_COLOR)
        render_gscore_diagnostics(res, rf)
    with s2:
        render_accel_sectors(res, base, name=GSCORE_NAME, color=GSCORE_COLOR)
        render_accel_top10(res, base, load_gscore_panel(), name=GSCORE_NAME,
                           color=GSCORE_COLOR, score_col="Z_GScore",
                           score_label="Z_GScore")
        render_gscore_scores(res, base, load_gscore_panel())


MOHANRAM_NAME = "Mohanram Growth Tilt"
MOHANRAM_COLOR = "#FF6B9D"


def _mohanram_multipliers() -> dict:
    """Controles des multiplicateurs G_SCORE."""
    with st.expander("⚙️ Paramètres de la simulation (multiplicateurs & plafonds)",
                     expanded=False):
        st.caption(
            "Poids final = **poids EGUS × multiplicateur G_SCORE**, puis "
            "neutralisation sectorielle et plafonnement itératif. Les scores "
            "**0-3 sont exclus** (multiplicateur nul, « torpedo »)."
        )
        cols = st.columns(5)
        mult = dict(mo.MULTIPLIERS)
        for i, g in enumerate((4, 5, 6, 7, 8)):
            mult[g] = float(cols[i].number_input(
                f"G{g}", min_value=0.0, max_value=5.0,
                value=float(mo.MULTIPLIERS[g]), step=0.25, key=f"mr_m{g}"))
        return mult


def _mohanram_nav_fig(res: dict) -> go.Figure:
    nav, bench = res["nav"], res["bench_nav"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bench.index, y=bench.values, name="EGUS",
                             line=dict(color=BENCH_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=nav.index, y=nav.values, name=MOHANRAM_NAME,
                             line=dict(color=MOHANRAM_COLOR, width=2.4)))
    fig.update_yaxes(title="Base 100")
    return base_layout(fig, title="NAV Base 100 — Mohanram Growth Tilt vs EGUS")


def _mohanram_dd_fig(res: dict) -> go.Figure:
    fig = go.Figure()
    for ser, name, color in ((res["nav"], MOHANRAM_NAME, MOHANRAM_COLOR),
                             (res["bench_nav"], "EGUS", BENCH_COLOR)):
        dd = (ser / ser.cummax() - 1.0) * 100.0
        fig.add_trace(go.Scatter(x=dd.index, y=dd.values, name=name,
                                 fill="tozeroy", line=dict(color=color)))
    fig.update_yaxes(title="Drawdown (%)")
    return base_layout(fig, title="Drawdown sur l'horizon 6 mois")


def _mohanram_brinson_fig(res: dict) -> go.Figure:
    b = res["brinson"]
    fig = go.Figure()
    for col, color in (("Allocation", "#4C9AFF"),
                       ("Sélection", "#2ECC71"),
                       ("Interaction", "#F0B429")):
        if col in b.columns:
            fig.add_trace(go.Bar(x=b.index, y=b[col].values * 100.0, name=col,
                                 marker_color=color))
    if "Total" in b.columns:
        fig.add_trace(go.Scatter(x=b.index, y=b["Total"].values * 100.0,
                                 name="Total", mode="markers",
                                 marker=dict(color="#E74C3C", size=9)))
    fig.update_layout(barmode="relative")
    fig.update_yaxes(title="Contribution (%)")
    return base_layout(fig, title="Attribution Brinson-Fachler (secteurs, 6 mois)")


def _mohanram_sector_fig(res: dict) -> go.Figure:
    sb = res["sector_bench"] * 100.0
    sp = res["sector_t0"] * 100.0
    fig = go.Figure()
    fig.add_trace(go.Bar(x=sb.index, y=sb.values, name="EGUS",
                         marker_color=BENCH_COLOR))
    fig.add_trace(go.Bar(x=sp.index, y=sp.values, name=MOHANRAM_NAME,
                         marker_color=MOHANRAM_COLOR))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="Poids (%)")
    return base_layout(fig, title="Poids sectoriels (t0) — neutralité vs EGUS")


def _mohanram_score_fig(res: dict) -> go.Figure:
    scores = res["scores"]["G_SCORE"].dropna()
    counts = scores.value_counts().reindex(range(9)).fillna(0)
    colors = ["#E74C3C"] * 4 + ["#F0B429"] * 2 + ["#2ECC71"] * 3
    fig = go.Figure(go.Bar(x=[str(i) for i in range(9)], y=counts.values,
                           marker_color=colors))
    fig.update_xaxes(title="G_SCORE")
    fig.update_yaxes(title="Nombre de titres")
    return base_layout(fig, title="Distribution du G_SCORE (univers EGUS)")


def _mohanram_active_fig(res: dict) -> go.Figure:
    p = res["portfolio"]
    p = p[p["w_final"] > 0].copy()
    top = p.reindex(p["w_actif"].abs().sort_values(ascending=False).index).head(15)
    top = top.sort_values("w_actif")
    colors = ["#2ECC71" if v >= 0 else "#E74C3C" for v in top["w_actif"]]
    fig = go.Figure(go.Bar(x=top["w_actif"].values * 100.0, y=top["ticker"],
                           orientation="h", marker_color=colors))
    fig.update_xaxes(title="Poids actif vs EGUS (%)")
    return base_layout(fig, title="Top 15 positions actives (Δw vs EGUS)")


def _mohanram_audit(res: dict) -> None:
    level2 = res["level2"]
    with st.expander("🔍 Rapport d'Audit Data (contrôle de cohérence)",
                     expanded=False):
        st.caption(
            "**Level-1 (fatal)** : absence de variables comptables de base → "
            "arrêt immédiat. **Level-2 (avertissement)** : historiques courts, "
            "R&D/CAPEX/ADV non publiés (traités à 0), secteurs ramenés à la "
            "neutralité. Backtest **Point-In-Time strict** "
            "(`pub_date ≤ t0`, `period_end ≤ t0 − 4 mois`)."
        )
        if not level2:
            st.success("Aucun avertissement — données complètes pour ce t0.")
            return
        groups: dict[str, list] = {}
        for it in level2:
            groups.setdefault(it.get("categorie", "autre"), []).append(it)
        for cat, items in sorted(groups.items()):
            tk = ", ".join(sorted({i.get("ticker", "—") for i in items}))
            st.warning(f"**{cat}** ({len(items)}) — {tk}")


def _mohanram_kpis(res: dict) -> None:
    stats = res["stats"]
    r1 = st.columns(6)
    r1[0].metric("Rendement 6 mois", fmt_stat("cum_return", stats["cum_return"]))
    r1[1].metric("Rendement annualisé", fmt_stat("ann_return", stats["ann_return"]))
    r1[2].metric("Volatilité annualisée", fmt_stat("vol_ann", stats["vol_ann"]))
    r1[3].metric("Ratio de Sharpe", fmt_stat("sharpe", stats["sharpe"]))
    r1[4].metric("Bêta vs EGUS", fmt_stat("beta", stats["beta"]))
    r1[5].metric("Max Drawdown", fmt_stat("max_dd", stats["max_dd"]))
    r2 = st.columns(6)
    r2[0].metric("Titres détenus", f"{res['n_held']} / {res['n_bench']}")
    r2[1].metric("Active Share", f"{res['active_share'] * 100:.2f} %")
    r2[2].metric("Poids exclus (G≤3)",
                 f"{res['excluded_weight'] * 100:.2f} %")
    r2[3].metric("Tracking Error", fmt_stat("te_ann", stats["te_ann"]))
    r2[4].metric("Ratio d'information", fmt_stat("info_ratio", stats["info_ratio"]))
    r2[5].metric("Infos détenues",
                 f"{int((res['portfolio']['w_final'] > 0).sum())}")


def render_tab_mohanram(base: dict, rf: float) -> None:
    """Onglet de recherche : portefeuille Mohanram Growth Tilt (G-Score 0-8)."""
    st.subheader("🚀 Stratégie Growth Tilted (Mohanram G-Score 0-8)")
    st.caption(
        "Recherche : tilt déterministe des poids **EGUS** par le **G_SCORE "
        "0-8** de Mohanram (2005) enrichi du critère **G8 = Publicité/Actifs**, "
        "à **horizon 6 mois**. Poids final = poids EGUS × multiplicateur, "
        "neutralité sectorielle stricte (financiers = neutres), plafond "
        f"**min({mo.DEFAULT_CAP_ABS:.0%}, w_EGUS + {mo.DEFAULT_CAP_REL:.0%})**. "
        "Point-In-Time strict ; radiations traitées selon **Shumway (1997)**."
    )
    candidates = mo.candidate_rebalance_dates()
    if not candidates:
        st.error("Aucune date de rebalancement disposant d'une fenêtre 6 mois.")
        st.stop()
        return

    labels = [d.strftime("%d/%m/%Y") for d in candidates]
    default_idx = labels.index("23/01/2025") if "23/01/2025" in labels \
        else len(labels) - 1
    cc = st.columns([2, 1, 1, 1])
    sel = cc[0].selectbox("Date de construction t0 (PIT)", labels,
                          index=default_idx, key="mr_t0")
    cap_abs = cc[1].slider("Plafond absolu (%)", 4.0, 12.0,
                           mo.DEFAULT_CAP_ABS * 100, 0.5, key="mr_capa")
    cap_rel = cc[2].slider("Plafond relatif EGUS (+%)", 0.0, 6.0,
                           mo.DEFAULT_CAP_REL * 100, 0.5, key="mr_capr")
    tx_bps = cc[3].slider("Coûts de transaction (bps)", 0.0, 50.0,
                          mo.DEFAULT_TX_BPS, 1.0, key="mr_tx")
    mult = _mohanram_multipliers()

    t0 = candidates[labels.index(sel)]
    params = {"multipliers": mult, "cap_abs": cap_abs / 100.0,
              "cap_rel": cap_rel / 100.0, "tx_bps": float(tx_bps),
              "rf": float(rf)}
    try:
        res = mo.run_mohanram(t0, params)
    except mo.MohanramDataError as exc:
        st.error("Données insuffisantes (Level-1) : exécution interrompue.")
        st.dataframe(pd.DataFrame(exc.issues), use_container_width=True)
        st.stop()
        return

    _mohanram_audit(res)
    _mohanram_kpis(res)

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(_mohanram_nav_fig(res), use_container_width=True)
    with c2:
        st.plotly_chart(_mohanram_dd_fig(res), use_container_width=True)
    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(_mohanram_brinson_fig(res), use_container_width=True)
    with c4:
        st.plotly_chart(_mohanram_sector_fig(res), use_container_width=True)
    c5, c6 = st.columns(2)
    with c5:
        st.plotly_chart(_mohanram_score_fig(res), use_container_width=True)
    with c6:
        st.plotly_chart(_mohanram_active_fig(res), use_container_width=True)

    st.markdown("#### Portefeuille détaillé")
    port = res["portfolio"].copy()
    port = port[port["w_final"] > 0].sort_values("w_final", ascending=False)
    show = port.rename(columns={
        "ticker": "Ticker", "name": "Nom", "sector": "Secteur",
        "w_bench": "Poids EGUS", "multiplier": "Multiplicateur",
        "w_final": "Poids final", "w_actif": "Actif Δw",
        "rendement_6m": "Rendement 6m", "G_SCORE": "G_SCORE",
        "statut": "Statut"})
    cols = ["Ticker", "Nom", "Secteur", "Poids EGUS", "Multiplicateur",
            "G_SCORE", "Poids final", "Actif Δw", "Rendement 6m", "Statut"]
    cols += [f"G{k}" for k in range(1, 9)]
    cols = [c for c in cols if c in show.columns]
    st.dataframe(show[cols], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Télécharger le portefeuille (CSV)",
                       show[cols].to_csv(index=False).encode("utf-8"),
                       file_name=f"mohanram_{t0:%Y%m%d}.csv", mime="text/csv")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    """Point d'entrée du dashboard."""
    st.title("📈 Growth USA — Dashboard Quant (trois piliers PIT)")
    st.caption(
        "Baseline v8 — **backtest trimestriel Point-in-Time** sur l'univers "
        "historique EGUS (11 rebalancements, 25/01/2024 → 27/07/2026). NAV "
        "Base 100 au 25/01/2024. Stratégie unique (Accélération 25 % · Momentum "
        "45 % · Qualité/GARP 30 %). Benchmark de référence unique **EGUS** "
        "(iShares ESG Aware MSCI USA Growth). Aucune dépendance macro."
    )

    params = render_sidebar()

    try:
        base = load_base()
    except Exception as exc:  # pragma: no cover
        st.error(f"Chargement des données impossible : {exc}")
        st.stop()
        return

    wa, wm, wq = params["weights"]
    results: dict = {}
    try:
        results[STRATEGY_NAME] = compute_strategy(
            STRATEGY_NAME, wa, wm, wq,
            params["n_titles"], params["sector_band"], params["sector_pilot"],
            params["dw_min"], params["dw_max"], params["w_max"],
            params["buffer_rule"], params["tx_bps"], params["rf"],
        )
    except Exception as exc:  # pragma: no cover
        st.warning(f"Stratégie « {STRATEGY_NAME} » non calculée : {exc}")

    if not results:
        st.error("La stratégie n'a pu être calculée.")
        st.stop()
        return

    primary = STRATEGY_NAME

    tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🧭 Stratégies & Paramètres",
        "① Performance & Risque",
        "② Secteurs",
        "③ Core/Satellites & Factoriel/Audit",
        "④ Pure Accélération",
        "⑤ G-Score (recherche)",
        "🚀 Stratégie Growth (Mohanram)",
    ])
    with tab0:
        try:
            render_tab_guide(params, base)
        except Exception as exc:  # pragma: no cover
            st.warning(f"Erreur onglet Stratégies & Paramètres : {exc}")
    with tab1:
        try:
            render_tab_performance(results, base, primary, params["rf"])
        except Exception as exc:  # pragma: no cover
            st.warning(f"Erreur onglet Performance : {exc}")
    with tab2:
        try:
            render_tab_sectors(results, base, primary)
        except Exception as exc:  # pragma: no cover
            st.warning(f"Erreur onglet Secteurs : {exc}")
    with tab3:
        try:
            render_tab_anatomy(results, base, primary)
        except Exception as exc:  # pragma: no cover
            st.warning(f"Erreur onglet Core/Satellites & Factoriel : {exc}")
    with tab4:
        try:
            render_tab_accel(base, params["rf"])
        except Exception as exc:  # pragma: no cover
            st.warning(f"Erreur onglet Pure Accélération : {exc}")
    with tab5:
        try:
            render_tab_gscore(base, params["rf"])
        except Exception as exc:  # pragma: no cover
            st.warning(f"Erreur onglet G-Score : {exc}")
    with tab6:
        try:
            render_tab_mohanram(base, params["rf"])
        except Exception as exc:  # pragma: no cover
            st.warning(f"Erreur onglet Stratégie Growth (Mohanram) : {exc}")


main()
