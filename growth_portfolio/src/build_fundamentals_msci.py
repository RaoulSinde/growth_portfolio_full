"""Construction du panel fondamental SEC pour l'univers MSCI USA Growth."""
from __future__ import annotations

import logging

import pandas as pd

from . import msci_config as cfg
from . import msci_universe as universe
from .build_fundamentals_sec import build_fundamentals_sec

logger = logging.getLogger(__name__)


def build_fundamentals_msci(save: bool = True) -> pd.DataFrame:
    uni = universe.build_universe()
    tickers = uni["tickers"]
    logger.info("Univers MSCI eligible : %d tickers", len(tickers))
    return build_fundamentals_sec(
        tickers=tickers,
        save=save,
        min_end=cfg.MIN_END,
        output_path=str(cfg.FUNDAMENTALS_MSCI),
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = build_fundamentals_msci(save=True)
    if not out.empty:
        print(f"tickers={out['ticker'].nunique()} rows={len(out)}")
        print(f"period_end={out['period_end'].min().date()} -> {out['period_end'].max().date()}")
        print(f"pub_date={out['pub_date'].min().date()} -> {out['pub_date'].max().date()}")


if __name__ == "__main__":
    main()
