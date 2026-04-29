"""
Refresh dynamique des listes de tickers (CAC40, SBF120, NASDAQ_GROWTH).

Sources :
  - CAC40, SBF120 : Wikipedia (FR) — colonne "Mnémo" + suffixe .PA
  - NASDAQ_GROWTH : iShares IWO (Russell 2000 Growth) holdings CSV

Stratégie :
  - Fetch puis cache JSON local (./ml_models/ticker_lists.json)
  - Diff avec version précédente : ajoute les nouveaux, soft-delete les disparus
  - Email de notification si changements détectés
  - Fallback sur la liste hardcodée si la source échoue
"""
from __future__ import annotations
import io
import json
import logging
import os
import pytz
from datetime import datetime
UTC = pytz.utc

import pandas as pd
import requests

logger = logging.getLogger(__name__)
CACHE_PATH = "./ml_models/ticker_lists.json"

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; StockAnalyzer/1.0)"}

# ── Surcharges pour cas particuliers (cotation hors Paris) ────────────────────
TICKER_OVERRIDES: dict[str, str] = {
    "MT":  "MT.AS",      # ArcelorMittal — cotation principale Amsterdam
    "STM": "STMPA.PA",   # STMicroelectronics — Paris
}


def _wiki_to_yfinance(mnemo: str) -> str:
    """Convertit un mnémo Wikipedia en ticker yfinance (ajoute .PA par défaut)."""
    mnemo = (mnemo or "").strip().upper()
    if not mnemo:
        return ""
    return TICKER_OVERRIDES.get(mnemo, f"{mnemo}.PA")


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_cac40_wiki() -> list[str] | None:
    """Scrape CAC40 depuis fr.wikipedia.org/wiki/CAC_40."""
    try:
        resp = requests.get(
            "https://fr.wikipedia.org/wiki/CAC_40",
            headers=_HEADERS, timeout=20,
        )
        tables = pd.read_html(io.StringIO(resp.text))
        for t in tables:
            cols = [str(c) for c in t.columns]
            mnemo_col = next((c for c in cols if "mném" in c.lower()), None)
            if mnemo_col and len(t) >= 30:
                tickers = [_wiki_to_yfinance(m) for m in t[mnemo_col].dropna().tolist()]
                tickers = [t for t in tickers if t]
                if 35 <= len(tickers) <= 45:
                    return tickers
    except Exception as e:
        logger.warning(f"[refresh_cac40] {e}")
    return None


def fetch_sbf120_wiki() -> list[str] | None:
    """Scrape SBF120 depuis fr.wikipedia.org/wiki/SBF_120."""
    try:
        resp = requests.get(
            "https://fr.wikipedia.org/wiki/SBF_120",
            headers=_HEADERS, timeout=20,
        )
        tables = pd.read_html(io.StringIO(resp.text))
        for t in tables:
            cols = [str(c) for c in t.columns]
            mnemo_col = next((c for c in cols if "mném" in c.lower()), None)
            if mnemo_col and len(t) >= 80:
                tickers = [_wiki_to_yfinance(m) for m in t[mnemo_col].dropna().tolist()]
                tickers = [t for t in tickers if t]
                if 100 <= len(tickers) <= 130:
                    return tickers
    except Exception as e:
        logger.warning(f"[refresh_sbf120] {e}")
    return None


def fetch_iwo_holdings() -> list[str] | None:
    """
    iShares Russell 2000 Growth ETF (IWO) — small caps US growth.
    Source : page produit iShares (CSV holdings).
    """
    # URL CSV publique d'iShares (peut changer, on essaie plusieurs variantes)
    urls = [
        "https://www.ishares.com/us/products/239710/ishares-russell-2000-growth-etf/1467271812596.ajax?fileType=csv&fileName=IWO_holdings&dataType=fund",
        "https://www.ishares.com/us/products/239710/ishares-russell-2000-growth-etf",
    ]
    for url in urls:
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=30)
            text = resp.text

            # Le CSV iShares contient un en-tête avec metadata, puis les holdings
            # On cherche la ligne qui commence par "Ticker," ou "Symbol,"
            lines = text.splitlines()
            header_idx = next(
                (i for i, l in enumerate(lines)
                 if l.lower().startswith("ticker,") or l.lower().startswith("symbol,")),
                None,
            )
            if header_idx is None:
                continue
            csv_body = "\n".join(lines[header_idx:])
            df = pd.read_csv(io.StringIO(csv_body))
            ticker_col = next(
                (c for c in df.columns if c.lower() in ("ticker", "symbol")), None
            )
            if not ticker_col:
                continue
            tickers = (
                df[ticker_col]
                .dropna()
                .astype(str)
                .str.strip()
                .str.upper()
                .tolist()
            )
            # Garde uniquement les tickers alphanumériques courts (exclut cash/futures)
            clean = [
                t for t in tickers
                if t.replace("-", "").isalpha() and 1 <= len(t) <= 5
                and t not in ("USD", "CASH", "MMK", "XTSL")
            ]
            if len(clean) > 200:
                return clean
        except Exception as e:
            logger.warning(f"[refresh_iwo] {url[:60]}… : {e}")
            continue
    return None


# ── Cache ─────────────────────────────────────────────────────────────────────

def _load_cache() -> dict:
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(data: dict) -> None:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(data, f, indent=2)
    except Exception as e:
        logger.warning(f"[ticker_cache] save failed: {e}")


def get_cached_list(market: str) -> list[str] | None:
    cache = _load_cache()
    entry = cache.get(market)
    if entry and isinstance(entry, dict):
        return entry.get("tickers")
    return None


# ── Refresh principal ─────────────────────────────────────────────────────────

def refresh_market(market: str) -> dict:
    """
    Refresh une liste de tickers et retourne un diff.
    Returns: {market, fetched, added, removed, total, source_ok}
    """
    fetcher = {
        "CAC40":          fetch_cac40_wiki,
        "SBF120":         fetch_sbf120_wiki,
        "NASDAQ_GROWTH":  fetch_iwo_holdings,
    }.get(market)

    if not fetcher:
        return {"market": market, "source_ok": False, "error": "no fetcher"}

    cache    = _load_cache()
    old      = set(cache.get(market, {}).get("tickers") or [])
    fetched  = fetcher()

    if not fetched:
        logger.warning(f"[refresh] {market} : fetch échoué, fallback sur cache")
        return {"market": market, "source_ok": False, "added": [], "removed": [], "total": len(old)}

    new_set  = set(fetched)
    added    = sorted(new_set - old) if old else []
    removed  = sorted(old - new_set) if old else []

    cache[market] = {
        "tickers":      sorted(new_set),
        "last_refresh": datetime.now(UTC).replace(tzinfo=None).isoformat(),
        "count":        len(new_set),
    }
    _save_cache(cache)

    return {
        "market":    market,
        "source_ok": True,
        "added":     added,
        "removed":   removed,
        "total":     len(new_set),
    }


def refresh_all_dynamic() -> dict[str, dict]:
    """Refresh tous les marchés dynamiques. Retourne un dict de diffs."""
    results = {}
    for market in ("CAC40", "SBF120", "NASDAQ_GROWTH"):
        results[market] = refresh_market(market)
    return results


# ── Soft-delete : sync DB avec les listes refresh ─────────────────────────────

def apply_diffs_to_db(db, diffs: dict[str, dict]) -> dict:
    """
    Applique les diffs à la table Stock :
      - Réactive les tickers présents dans la nouvelle liste
      - Soft-delete (is_active=False, delisted_at=now) ceux qui ont disparu
    Les nouveaux tickers seront créés au prochain sync prix (data_engine).
    """
    from .models import Stock
    now = datetime.now(UTC).replace(tzinfo=None)
    reactivated = deactivated = 0

    for market, diff in diffs.items():
        if not diff.get("source_ok"):
            continue

        # Soft-delete les disparus
        for ticker in diff.get("removed", []):
            stock = db.query(Stock).filter(
                Stock.ticker == ticker, Stock.market == market
            ).first()
            if stock and stock.is_active:
                stock.is_active   = False
                stock.delisted_at = now
                deactivated += 1

        # Réactive les revenants (ticker re-rentré dans l'index)
        for ticker in diff.get("added", []):
            stock = db.query(Stock).filter(
                Stock.ticker == ticker, Stock.market == market
            ).first()
            if stock and not stock.is_active:
                stock.is_active   = True
                stock.delisted_at = None
                reactivated += 1

    db.commit()
    return {"reactivated": reactivated, "deactivated": deactivated}
