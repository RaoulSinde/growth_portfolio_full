import difflib
import json
import re
import time

import pandas as pd
import requests
import yfinance as yf

from src.config import CACHE_DIR, TICKER_MAP_FILE, WEIGHTS_FILE

US_EXCHANGES = {
    "NYQ",
    "NMS",
    "NCM",
    "NGM",
    "ASE",
    "PCX",
    "BATS",
    "US",
    "UN",
    "UW",
    "UQ",
    "UR",
    "UA",
    "UC",
    "UP",
    "UB",
    "UT",
    "UM",
    "UX",
    "UD",
}
FOREIGN_TICKER_SUFFIXES = ("EUR", "USD", "CHF", "GBP", "JPY", "CAD", "AUD")
OPENFIGI_URL = "https://api.openfigi.com/v3/mapping"
OPENFIGI_CACHE = CACHE_DIR / "openfigi_raw.json"
BATCH_SIZE = 10
THROTTLE_SECONDS = 2.6

CORE_SECURITY_TYPES = {"Common Stock", "REIT", "Depositary Receipt", "Tracking Stk"}
EXCLUDED_SECURITY_TYPES = {
    "ETF",
    "Mutual Fund",
    "Closed End Fund",
    "Preferred Stock",
    "Warrant",
    "Right",
    "Unit",
    "Note",
    "Bond",
    "Index",
    "Structured Product",
    "Trust",
}

MANUAL_OVERRIDES = {
    "G8711010": "FTI",
    "G0750C10": "AXTA",
    "G5495010": "LIN",
    "G3730V10": "FTAI",
    "G0176J10": "ALLE",
    "G8994E10": "TT",
    "G2918310": "ETN",
    "G6700G10": "NVT",
    "G4756710": "INFO",
    "G3265R10": "APTV",
    "G8068L10": "SN",
    "G2004J10": "CCL",
    "G3643J10": "FLUT",
    "V7780T10": "RCL",
    "G6674U10": "NVCR",
    "G8473T10": "STE",
    "G4618810": "HZNP",
    "G7627910": "ROIV",
    "G0403H10": "AON",
    "G0450A10": "ACGL",
    "G3223R10": "EG",
    "G7496G10": "RNR",
    "G1151C10": "ACN",
    "Q4982L10": "IREN",
    "G7997R10": "STX",
    "G3323L10": "FN",
    "G2545710": "CRDO",
    "N6596X10": "NXPI",
    "G6118810": "LBTYA",
    "53030710": "LBRDA",
    "53030730": "LBRDK",
    "53122975": "FWONK",
    "53122978": "LSXMK",
    "53122981": "LSXMA",
    "00165C20": "APE",
    "00365410": "ABMD",
    "00507V10": "ATVI",
    "01717510": "Y",
    "03662Q10": "ANSS",
    "05338G10": "AVLR",
    "09215C10": "BKI",
    "14880610": "CTLT",
    "15677J10": "DAY",
    "15678210": "CERN",
    "17737610": "CTXS",
    "20717M10": "CFLT",
    "22266L10": "COUP",
    "28551210": "EA",
    "29109X10": "AZPN",
    "30063P10": "EXAS",
    "43644010": "HOLX",
    "56584910": "MRO",
    "57479510": "MASI",
    "67020Y10": "NUAN",
    "67181A10": "OSH",
    "69355F10": "PPD",
    "72378710": "PXD",
    "74736K10": "QRVO",
    "78486Q10": "SIVB",
    "81181C10": "SGEN",
    "84863710": "SPLK",
    "90184L10": "TWTR",
    "92220P10": "VAR",
    "92856340": "VMW",
    "93442310": "WBD",
    "98391910": "XLNX",
    "98936J10": "ZEN",
}


def cusip_check_digit(cusip8):
    def char_value(ch):
        if ch.isdigit():
            return int(ch)
        if ch.isalpha():
            return ord(ch) - 55
        if ch == "*":
            return 36
        if ch == "@":
            return 37
        return 38

    total = 0
    for index, ch in enumerate(cusip8):
        value = char_value(ch)
        if index % 2 == 1:
            value *= 2
        total += value // 10 + value % 10
    return str((10 - total % 10) % 10)


def normalize_name(value):
    text = str(value).upper().replace("&", " AND ")
    text = re.sub(r"[^A-Z0-9 ]", " ", text)
    tokens = [
        token
        for token in text.split()
        if token
        not in {
            "INC",
            "INCORPORATED",
            "CORP",
            "CORPORATION",
            "CO",
            "COMPANY",
            "LTD",
            "LIMITED",
            "PLC",
            "NV",
            "SA",
            "AG",
            "HOLDINGS",
            "HOLDING",
            "GROUP",
            "THE",
            "TRUST",
            "REIT",
            "LP",
            "LLC",
            "CLASS",
            "CL",
            "SERIES",
            "NEW",
            "DEL",
        }
    ]
    return " ".join(tokens)


def name_score(left, right):
    return difflib.SequenceMatcher(None, normalize_name(left), normalize_name(right)).ratio()


def class_tag(value):
    text = str(value).upper()
    match = re.search(r"\bCLASS\s+([A-Z])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\bSERIES\s+([A-Z])\b", text)
    if match:
        return match.group(1)
    match = re.search(r"\bCL\s*[- ]?\s*([A-Z])\b", text)
    if match:
        return match.group(1)
    return None


def load_universe():
    weights = pd.read_parquet(WEIGHTS_FILE)
    return (
        weights[["cusip", "name"]]
        .drop_duplicates("cusip")
        .sort_values("cusip")
        .reset_index(drop=True)
    )


def load_openfigi_cache():
    if OPENFIGI_CACHE.exists():
        return json.loads(OPENFIGI_CACHE.read_text(encoding="utf-8"))
    return {}


def save_openfigi_cache(cache):
    OPENFIGI_CACHE.write_text(json.dumps(cache), encoding="utf-8")


def openfigi_lookup(cusip9_values):
    cache = load_openfigi_cache()
    pending = [value for value in cusip9_values if value not in cache]
    total = len(pending)
    for start in range(0, total, BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        payload = [{"idType": "ID_CUSIP", "idValue": value} for value in batch]
        for attempt in range(6):
            try:
                response = requests.post(OPENFIGI_URL, json=payload, timeout=45)
                if response.status_code == 429:
                    time.sleep(10 * (attempt + 1))
                    continue
                if response.status_code == 413:
                    raise RuntimeError("OpenFIGI payload too large")
                response.raise_for_status()
                for value, item in zip(batch, response.json()):
                    cache[value] = item
                save_openfigi_cache(cache)
                break
            except requests.RequestException as exc:
                print(f"OpenFIGI retry {attempt + 1}: {exc}")
                time.sleep(3 * (attempt + 1))
        else:
            print(f"OpenFIGI failed batch starting {batch[0]}")
        print(f"OpenFIGI {min(start + BATCH_SIZE, total)}/{total}")
        if start + BATCH_SIZE < total:
            time.sleep(THROTTLE_SECONDS)
    return cache


def _entry_score(target_name, entry):
    ticker = str(entry.get("ticker") or "").strip().strip("*").replace("/", "-")
    if not ticker or any(ch in ticker for ch in "./^,"):
        return None
    if ticker.endswith(FOREIGN_TICKER_SUFFIXES):
        return None
    exch = entry.get("exchCode")
    if exch not in US_EXCHANGES:
        return None
    security_type2 = entry.get("securityType2") or ""
    if security_type2 in EXCLUDED_SECURITY_TYPES:
        return None
    if entry.get("marketSector") != "Equity":
        return None

    score = name_score(target_name, entry.get("name", ""))
    if exch == "US":
        score += 0.30
    else:
        score += 0.15
    if security_type2 in CORE_SECURITY_TYPES:
        score += 0.15

    target_class = class_tag(target_name)
    candidate_class = class_tag(entry.get("name", ""))
    if target_class and candidate_class:
        score += 0.20 if target_class == candidate_class else -0.35
    elif target_class and not candidate_class:
        score -= 0.10
    elif not target_class and candidate_class:
        score -= 0.15
    return score


def pick_openfigi(name, item):
    scored = []
    for entry in item.get("data", []) or []:
        score = _entry_score(name, entry)
        if score is not None:
            scored.append((score, entry))
    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    best_score, best = scored[0]
    if best_score < 0.40:
        return None
    return best


def yfinance_search(name):
    try:
        quotes = yf.Search(name, max_results=10).quotes
    except Exception:
        return None
    candidates = [
        quote
        for quote in quotes
        if quote.get("exchange") in US_EXCHANGES
        and "." not in str(quote.get("symbol", "."))
        and quote.get("quoteType") == "EQUITY"
    ]
    if not candidates:
        return None
    ranked = sorted(
        candidates,
        key=lambda quote: name_score(
            name, quote.get("shortname") or quote.get("longname") or ""
        ),
        reverse=True,
    )
    best = ranked[0]
    score = name_score(name, best.get("shortname") or best.get("longname") or "")
    if score < 0.50:
        return None
    return best.get("symbol")


def build_map(universe, openfigi_results):
    now = pd.Timestamp.utcnow()
    rows = []
    for row in universe.itertuples(index=False):
        cusip9 = row.cusip + cusip_check_digit(row.cusip)
        item = openfigi_results.get(cusip9, {})
        match = pick_openfigi(row.name, item)
        override = MANUAL_OVERRIDES.get(row.cusip)

        if match:
            ticker = match["ticker"].replace("/", "-").strip()
            source = "openfigi"
            if override and override != ticker:
                ticker = override
                source = "manual"
        elif override:
            ticker = override
            source = "manual"
        else:
            ticker = yfinance_search(row.name)
            source = "yfinance_search" if ticker else "unresolved"

        rows.append(
            {
                "cusip": row.cusip,
                "name": row.name,
                "ticker": ticker,
                "source": source,
                "validated_at": now,
            }
        )
    return pd.DataFrame(rows)


def main():
    universe = load_universe()
    lookup = {row.cusip: row.cusip + cusip_check_digit(row.cusip) for row in universe.itertuples()}
    results = openfigi_lookup(list(lookup.values()))
    mapping = build_map(universe, results)
    mapping = mapping.sort_values("cusip").reset_index(drop=True)
    mapping.to_parquet(TICKER_MAP_FILE, index=False)

    print(mapping.groupby("source").size().to_string())
    unresolved = mapping[mapping["ticker"].isna()]
    print(f"Resolved: {mapping['ticker'].notna().sum()} / {len(mapping)}")
    print(f"Unresolved: {len(unresolved)}")
    if len(unresolved):
        print(unresolved[["cusip", "name"]].to_string(index=False))


if __name__ == "__main__":
    main()
