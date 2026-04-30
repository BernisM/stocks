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
import re
import pytz
from datetime import datetime
UTC = pytz.utc

import pandas as pd
import requests

logger = logging.getLogger(__name__)
CACHE_PATH = "./ml_models/ticker_lists.json"

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

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

def _wiki_api_html(page: str) -> str | None:
    """Récupère le HTML d'une page Wikipedia via l'API MediaWiki (pas bloquée par cloud IPs)."""
    try:
        resp = requests.get(
            "https://fr.wikipedia.org/w/api.php",
            params={"action": "parse", "page": page, "prop": "text", "format": "json", "formatversion": "2"},
            headers=_HEADERS,
            timeout=20,
        )
        return resp.json()["parse"]["text"]
    except Exception:
        return None


def fetch_cac40_wiki() -> list[str] | None:
    """Scrape CAC40 via l'API MediaWiki (contourne le blocage cloud)."""
    try:
        html = _wiki_api_html("CAC_40")
        if not html:
            return None
        tables = pd.read_html(io.StringIO(html))
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
    """Scrape SBF120 via l'API MediaWiki (contourne le blocage cloud)."""
    try:
        html = _wiki_api_html("SBF_120")
        if not html:
            return None
        tables = pd.read_html(io.StringIO(html))
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


def fetch_euronext_growth_live() -> list[str] | None:
    """
    Fetch Euronext Growth Paris (MIC: ALXP) tickers depuis l'API live.euronext.com.
    Retourne les tickers avec suffixe .PA pour yfinance.
    Fallback : None (tickers.py utilisera la liste hardcodée).
    """
    base_url = "https://live.euronext.com/en/pd/data/quote"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "X-Requested-With": "XMLHttpRequest",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Referer": "https://live.euronext.com/en/markets/euronext-growth/list",
    }
    tickers: list[str] = []
    start   = 0
    length  = 100
    total   = None

    try:
        while True:
            params = {
                "mics":             "ALXP",
                "display_datapoints": "dp_values",
                "display_filters":  "df_screener_quote",
                "iDisplayStart":    start,
                "iDisplayLength":   length,
            }
            resp = requests.get(base_url, params=params, headers=headers, timeout=25)
            if resp.status_code != 200:
                logger.warning(f"[euronext_growth] HTTP {resp.status_code}")
                break

            try:
                data = resp.json()
            except Exception:
                logger.warning("[euronext_growth] réponse non-JSON")
                break

            if total is None:
                total = int(data.get("iTotalRecords") or data.get("iTotalDisplayRecords") or 0)

            rows = data.get("aaData", [])
            if not rows:
                break

            for row in rows:
                # L'API retourne des cellules HTML — extraire les mnémoniques
                for cell in row[:6]:
                    text = re.sub(r"<[^>]+>", "", str(cell)).strip()
                    # Ticker Euronext Growth : 2-6 lettres majuscules, parfois avec chiffres
                    if re.match(r"^[A-Z]{2,7}$", text) or re.match(r"^[A-Z]{2,5}[0-9]$", text):
                        tickers.append(text + ".PA")
                        break

            start += length
            if total and start >= total:
                break
            if start > 2000:  # garde-fou
                break

    except Exception as e:
        logger.warning(f"[euronext_growth] fetch error: {e}")
        return None

    # Déduplique + filtre
    seen: set[str] = set()
    result: list[str] = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            result.append(t)

    if len(result) < 10:
        logger.warning(f"[euronext_growth] trop peu de résultats ({len(result)}), fallback")
        return None

    logger.info(f"[euronext_growth] {len(result)} tickers récupérés depuis Euronext live")
    return result


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
        "CAC40":           fetch_cac40_wiki,
        "SBF120":          fetch_sbf120_wiki,
        "NASDAQ_GROWTH":   fetch_iwo_holdings,
        "EURONEXT_GROWTH": fetch_euronext_growth_live,
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
    for market in ("CAC40", "SBF120", "NASDAQ_GROWTH", "EURONEXT_GROWTH"):
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
