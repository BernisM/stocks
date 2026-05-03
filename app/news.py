from __future__ import annotations
import logging
import time
from datetime import datetime, UTC

import yfinance as yf

log = logging.getLogger(__name__)

_cache: dict[str, tuple[list, float]] = {}
_NEWS_TTL = 900  # 15 min

# ── Sentiment keywords ────────────────────────────────────────────────────────

_BULLISH = {
    "partnership", "acquisition", "beat", "record", "growth", "upgrade",
    "agreement", "expansion", "profit", "revenue", "strong", "outperform",
    "raise", "dividend", "surge", "rally", "positive", "innovation",
    "contract", "deal", "invest", "increase", "gain", "approval", "approved",
    "launch", "milestone", "exceed", "guidance", "raised",
    "partenariat", "croissance", "hausse", "accord", "bénéfice",
    "augmentation", "progression", "succès", "record", "approbation",
}

_BEARISH = {
    "lawsuit", "miss", "recall", "investigation", "downgrade", "tariff",
    "sanction", "decline", "loss", "warning", "weak", "underperform",
    "cut", "default", "debt", "probe", "fine", "penalty", "fall",
    "plunge", "crash", "fear", "risk", "recession", "inflation", "hike",
    "war", "conflict", "tension", "layoff", "shortfall", "missed",
    "procès", "baisse", "perte", "avertissement", "sanction", "taxe",
    "récession", "guerre", "conflit", "inflation", "amende", "licenciement",
}

_EVENT_TYPES: dict[str, set[str]] = {
    "earnings":    {"earnings", "revenue", "profit", "results", "quarterly", "eps", "bénéfice", "résultats"},
    "m&a":         {"acquisition", "merger", "takeover", "bid", "buyout", "acquire", "fusion"},
    "partnership": {"partnership", "agreement", "contract", "collaboration", "alliance", "partenariat", "accord"},
    "regulatory":  {"investigation", "probe", "fine", "lawsuit", "regulation", "sec", "ftc", "amende", "procès"},
    "macro":       {"fed", "ecb", "rate", "inflation", "gdp", "tariff", "geopolitical", "war", "recession", "taux", "bce"},
    "analyst":     {"upgrade", "downgrade", "target", "analyst", "outperform", "underperform", "price target"},
}

_EVENT_LABELS = {
    "earnings": "📊 Résultats",
    "m&a": "🤝 M&A",
    "partnership": "🔗 Partenariat",
    "regulatory": "⚖️ Réglementaire",
    "macro": "🌍 Macro",
    "analyst": "🔍 Analyste",
    "general": "📰 Général",
}


def _detect_event_type(text: str) -> str:
    words = set(text.lower().replace(",", " ").replace(".", " ").split())
    for etype, keywords in _EVENT_TYPES.items():
        if keywords & words:
            return etype
    return "general"


def _sentiment_score(title: str, summary: str = "") -> tuple[int, str]:
    text = (title + " " + summary).lower().replace(",", " ").replace(".", " ")
    words = set(text.split())
    bullish = len(_BULLISH & words)
    bearish = len(_BEARISH & words)
    total = bullish + bearish
    if total == 0:
        score = 0
    else:
        score = round((bullish - bearish) / total * 100)
    if score >= 40:   label = "Haussier"
    elif score >= 15: label = "Légèrement haussier"
    elif score <= -40: label = "Baissier"
    elif score <= -15: label = "Légèrement baissier"
    else:             label = "Neutre"
    return score, label


def _format_item(item: dict, source_market: str = "") -> dict:
    # yfinance v0.2.x+ nests data under "content"; older versions use flat keys
    c = item.get("content") or item
    title   = c.get("title", "")
    summary = c.get("summary", "") or c.get("description", "") or ""

    # Publisher
    provider = c.get("provider") or {}
    publisher = provider.get("displayName", "") or item.get("publisher", "")

    # URL
    click_url = c.get("clickThroughUrl") or c.get("canonicalUrl") or {}
    url = click_url.get("url", "") or item.get("link", "") or item.get("url", "")

    # Timestamp — new API uses ISO string; old used Unix int
    pub_date = c.get("pubDate") or c.get("displayTime", "")
    pub_ts_raw = item.get("providerPublishTime") or item.get("pubTime") or 0
    try:
        if pub_date:
            dt_obj = datetime.fromisoformat(pub_date.replace("Z", "+00:00"))
            pub_ts = int(dt_obj.timestamp())
            dt = dt_obj.strftime("%d/%m %H:%M")
        elif pub_ts_raw:
            pub_ts = int(pub_ts_raw)
            dt = datetime.fromtimestamp(pub_ts, tz=UTC).strftime("%d/%m %H:%M")
        else:
            pub_ts, dt = 0, ""
    except Exception:
        pub_ts, dt = 0, ""

    score, label = _sentiment_score(title, summary)
    event = _detect_event_type(title + " " + summary)
    return {
        "title":           title,
        "publisher":       publisher,
        "url":             url,
        "published_at":    dt,
        "pub_ts":          pub_ts,
        "sentiment_score": score,
        "sentiment_label": label,
        "event_type":      event,
        "event_label":     _EVENT_LABELS.get(event, "📰 Général"),
        "source_market":   source_market,
    }


def fetch_ticker_news(ticker: str, limit: int = 10) -> list[dict]:
    key = f"t:{ticker}"
    now = time.time()
    if key in _cache and now - _cache[key][1] < _NEWS_TTL:
        return _cache[key][0]
    try:
        raw  = yf.Ticker(ticker).news or []
        news = [_format_item(n) for n in raw[:limit]]
        _cache[key] = (news, now)
        return news
    except Exception as e:
        log.warning(f"[news] {ticker}: {e}")
        return []


_GLOBAL_SOURCES = [
    ("^GSPC", "S&P 500"),
    ("^FCHI", "CAC 40"),
    ("GC=F",  "Or / Macro"),
    ("^VIX",  "VIX"),
]


def compute_ticker_sentiment(ticker: str) -> int | None:
    """Calcule un score de sentiment agrégé (−100..+100) pour un ticker, None si pas de news."""
    items = fetch_ticker_news(ticker, limit=10)
    if not items:
        return None
    scores = [n["sentiment_score"] for n in items if n["sentiment_score"] != 0]
    if not scores:
        return None
    return round(sum(scores) / len(scores))


def update_news_sentiment_for_signals(db) -> dict:
    """
    Met à jour news_sentiment dans la dernière AnalysisResult des stocks Buy/Strong Buy.
    Appelle yfinance.news pour chaque ticker éligible, rate-limited à 0.3s/ticker.
    Retourne un résumé {updated, skipped, errors}.
    """
    from sqlalchemy import text
    from .config import DATABASE_URL

    # Latest AnalysisResult per stock with Buy/Strong Buy — compatible SQLite + PostgreSQL
    if "sqlite" in DATABASE_URL:
        sql = text("""
            SELECT s.id, s.ticker, ar.id AS ar_id
            FROM stocks s
            JOIN analysis_results ar ON ar.stock_id = s.id
            WHERE s.is_active = 1
              AND ar.ranking IN ('Buy', 'Strong Buy')
              AND s.market NOT IN ('COMMODITIES', 'CRYPTO')
              AND ar.date = (
                  SELECT MAX(ar2.date) FROM analysis_results ar2
                  WHERE ar2.stock_id = s.id
              )
        """)
    else:
        sql = text("""
            SELECT DISTINCT ON (s.id) s.id, s.ticker, ar.id AS ar_id
            FROM stocks s
            JOIN analysis_results ar ON ar.stock_id = s.id
            WHERE s.is_active = TRUE
              AND ar.ranking IN ('Buy', 'Strong Buy')
              AND s.market NOT IN ('COMMODITIES', 'CRYPTO')
            ORDER BY s.id, ar.date DESC
        """)

    rows = db.execute(sql).fetchall()

    updated = skipped = errors = 0
    for stock_id, ticker, ar_id in rows:
        try:
            sentiment = compute_ticker_sentiment(ticker)
            if sentiment is None:
                skipped += 1
                continue
            db.execute(
                text("UPDATE analysis_results SET news_sentiment = :s WHERE id = :id"),
                {"s": sentiment, "id": ar_id},
            )
            updated += 1
        except Exception as e:
            errors += 1
            log.warning(f"[news_sentiment] {ticker}: {e}")
        time.sleep(0.3)

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        log.warning(f"[news_sentiment] commit failed: {e}")

    log.info(f"[news_sentiment] {updated} mis à jour, {skipped} ignorés, {errors} erreurs")
    return {"updated": updated, "skipped": skipped, "errors": errors}


def fetch_global_news(limit: int = 20) -> list[dict]:
    key = "global"
    now = time.time()
    if key in _cache and now - _cache[key][1] < _NEWS_TTL:
        return _cache[key][0]
    seen: set[str] = set()
    all_news: list[dict] = []
    for sym, label in _GLOBAL_SOURCES:
        try:
            raw = yf.Ticker(sym).news or []
            for item in raw[:8]:
                title = item.get("title", "")
                if not title or title in seen:
                    continue
                seen.add(title)
                all_news.append(_format_item(item, source_market=label))
        except Exception as e:
            log.warning(f"[news] global {sym}: {e}")
    all_news.sort(key=lambda x: x["pub_ts"], reverse=True)
    result = all_news[:limit]
    _cache[key] = (result, now)
    return result
