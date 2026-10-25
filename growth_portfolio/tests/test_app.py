"""Smoke test du dashboard Streamlit via ``streamlit.testing.v1.AppTest``.

Vérifie que ``app.py`` s'exécute sans exception, expose **7 onglets**
(Stratégies & Paramètres, Performance & Risque, Secteurs, Core/Satellites &
Factoriel/Audit, Pure Accélération, G-Score de recherche, Stratégie Growth
Mohanram), affiche le **benchmark EGUS** (unique), documente les
stratégies/paramètres et **ne contient plus l'onglet / motif « Régime »**
(dépendance macro supprimée).

Exécution :  python -m pytest tests/test_app.py -q
"""
from __future__ import annotations

import os

from streamlit.testing.v1 import AppTest

APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py"
)
REPORT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "AUDIT_REPORT.md"
)

BENCH_NAMES = ("EGUS",)
FORBIDDEN_TOKENS = ("Régime", "Regime")


def _run() -> AppTest:
    at = AppTest.from_file(APP_PATH, default_timeout=600)
    at.run()
    return at


def _all_text(at: AppTest) -> str:
    """Concatène le texte de tous les éléments textuels accessibles."""
    chunks: list[str] = []
    for kind in ("markdown", "caption", "info", "warning", "title", "subheader",
                 "header", "text"):
        try:
            elements = at.get(kind)
        except Exception:  # pragma: no cover - type indisponible
            continue
        for el in elements:
            val = getattr(el, "value", None)
            if val:
                chunks.append(str(val))
    return "\n".join(chunks)


def test_app_no_exception():
    """L'application se charge sans lever d'exception."""
    at = _run()
    assert not at.exception, [getattr(e, "value", e) for e in at.exception]


def test_app_renders_elements():
    """Au moins un élément markdown / metric / graphique est rendu."""
    at = _run()
    n_markdown = len(at.markdown)
    n_metric = len(at.metric)
    try:
        n_plotly = len(at.get("plotly_chart"))
    except Exception:  # pragma: no cover - type d'élément indisponible
        n_plotly = 0
    assert (n_markdown + n_metric + n_plotly) > 0, (
        "Aucun élément rendu par l'application."
    )


def test_app_has_kpis():
    """Les KPIs (st.metric) de l'onglet 1 sont présents."""
    at = _run()
    assert len(at.metric) >= 6, f"seulement {len(at.metric)} métriques rendues"


def test_app_has_five_tabs():
    """Le dashboard expose 7 onglets principaux (+ 4 sous-onglets Accélération,
    + 2 sous-onglets G-Score)."""
    at = _run()
    labels = [getattr(t, "label", "") for t in at.tabs]
    joined = " ".join(labels)
    assert len(at.tabs) == 13, f"{len(at.tabs)} onglets rendus (7 + 6 sous-onglets)"
    assert "Stratégies" in joined, f"onglet Stratégies & Paramètres absent : {labels}"
    assert "Performance" in joined, f"onglet Performance absent : {labels}"
    assert "Secteurs" in joined, f"onglet Secteurs absent : {labels}"
    assert "Audit" in joined, f"onglet Audit absent : {labels}"
    assert "Accélération" in joined, f"onglet Pure Accélération absent : {labels}"
    assert "G-Score" in joined, f"onglet G-Score absent : {labels}"
    assert "Mohanram" in joined, f"onglet Mohanram absent : {labels}"
    for token in ("Performance & Stats", "Décomposition", "Alpha & Bêta",
                  "Secteurs & Top 10", "Performance & Diagnostics",
                  "Secteurs & Titres"):
        assert token in joined, f"sous-onglet '{token}' absent : {labels}"


def test_app_mohanram_tab_content():
    """L'onglet Mohanram Growth rend KPIs, paramètres et documentation."""
    at = _run()
    text = _all_text(at)
    for token in ("Growth Tilted", "Mohanram", "G8", "Shumway",
                  "Active Share", "PIT"):
        assert token in text, f"token '{token}' absent de l'onglet Mohanram"
    labels = " ".join(getattr(s, "label", "") for s in at.slider)
    for token in ("Plafond absolu", "Plafond relatif", "Coûts de transaction"):
        assert token in labels, f"contrôle '{token}' absent : {labels}"


def test_app_gscore_tab_content():
    """L'onglet G-Score rend KPIs, contrôles, critères et diagnostics."""
    at = _run()
    text = _all_text(at)
    for token in ("G-Score", "Mohanram", "G1", "R&D", "accruals",
                  "Active Share"):
        assert token in text, f"token '{token}' absent de l'onglet G-Score"


def test_app_accel_tab_content():
    """L'onglet Pure Accélération rend KPIs, contrôles et documentation."""
    at = _run()
    assert len(at.metric) >= 20, f"seulement {len(at.metric)} métriques"
    assert len(at.dataframe) >= 4, f"seulement {len(at.dataframe)} tableaux"
    labels = " ".join(getattr(s, "label", "") for s in at.slider)
    assert "w_max" in labels or "Poids max" in labels, labels
    text = _all_text(at)
    for token in ("Accélération", "w_max", "Brinson", "Jensen", "référence"):
        assert token in text, f"token '{token}' absent de l'onglet Accélération"


def test_app_guide_lists_strategies_and_params():
    """L'onglet documentation présente les 3 piliers et les paramètres clés."""
    at = _run()
    text = _all_text(at)
    for token in ("Accélération", "Momentum", "Qualité"):
        assert token in text, f"pilier '{token}' absent de la documentation"
    for token in ("Paramètres", "Buffer Rule", "Coûts de transaction"):
        assert token in text, f"paramètre '{token}' absent de la documentation"


def test_app_has_no_regime():
    """Aucun onglet ni texte applicatif ne doit mentionner « Régime » (macro supprimée).

    Le contenu du fichier ``AUDIT_REPORT.md`` (chargé tel quel dans un expander)
    est exclu du contrôle : on vérifie le texte **produit par l'application**.
    """
    at = _run()
    labels = " ".join(getattr(t, "label", "") for t in at.tabs)
    text = labels + "\n" + _all_text(at)
    # Retire le rapport d'audit rendu (contenu brut du fichier).
    if os.path.exists(REPORT_PATH):
        with open(REPORT_PATH, encoding="utf-8") as fh:
            text = text.replace(fh.read(), "")
    for token in FORBIDDEN_TOKENS:
        assert token not in text, f"motif interdit '{token}' encore présent"


def test_app_has_benchmark():
    """Le benchmark de référence EGUS est affiché (benchmark unique)."""
    at = _run()
    text = _all_text(at)
    for name in BENCH_NAMES:
        assert name in text, f"benchmark '{name}' absent de l'interface"
    # Plus aucun sélecteur de benchmark de référence (référence figée à EGUS).
    for radio in at.radio:
        assert "férence" not in str(getattr(radio, "label", "")), \
            "sélecteur de benchmark de référence encore présent"


def test_app_pit_and_turnover_mentions():
    """Le backtest PIT et le turnover trimestriel sont explicités."""
    at = _run()
    text = _all_text(at)
    assert "Point-in-Time" in text or "PIT" in text, "mention PIT absente"
    assert "rebalancement" in text.lower(), "mention rebalancement absente"


def test_audit_report_is_present_and_current():
    """``AUDIT_REPORT.md`` existe et décrit la baseline (GO, EGUS, PIT)."""
    assert os.path.exists(REPORT_PATH), REPORT_PATH
    with open(REPORT_PATH, encoding="utf-8") as fh:
        txt = fh.read()
    for token in ("GO", "EGUS"):
        assert token in txt, f"'{token}' absent de AUDIT_REPORT.md"


if __name__ == "__main__":
    import sys

    import pytest

    sys.exit(pytest.main([__file__, "-q"]))
