from __future__ import annotations
"""
Data engine: télécharge les données yfinance et maintient
une fenêtre glissante de ROLLING_WINDOW jours par action.

Stratégie : batch de 20 tickers, puis fallback individuel si erreur.
"""
import gc
import json
import logging
import os
import time
import warnings
from datetime import UTC, datetime, timezone as dt_timezone

# Supprime les FutureWarning internes de yfinance (pandas CoW) — non actionnable de notre côté
warnings.filterwarnings("ignore", category=FutureWarning, module="yfinance")

import pandas as pd
import pytz
import yfinance as yf
from sqlalchemy.orm import Session


from .config import ROLLING_WINDOW
from .models import DailyData, Stock
from .tickers import COMMODITY_NAMES, CRYPTO_NAMES, get_all_tickers

logger = logging.getLogger(__name__)

BATCH_SIZE   = 20   # conservative pour éviter le rate limiting
UPDATE_PERIOD = "2d"  # fenêtre de mise à jour quotidienne (2j couvre weekends + jours fériés)

MARKET_STATUS_PATH = "./ml_models/market_status.json"
BLACKLIST_PATH     = "./ml_models/blacklisted_tickers.json"
_BLACKLIST_THRESHOLD = 4   # échecs consécutifs avant exclusion (4 = résiste aux erreurs réseau transitoires)

# Un ticker représentatif par marché pour récupérer l'état du marché
_MARKET_REPS = {
    "CAC40":       "MC.PA",
    "SBF120":      "OR.PA",
    "SP500":       "AAPL",
    "COMMODITIES": "GC=F",
    "CRYPTO":      "BTC-USD",
}

_STATE_LABELS = {
    "REGULAR":    "Market Open",
    "CLOSED":     "At close",
    "PRE":        "Pre-market",
    "POST":       "After hours",
    "POSTPOST":   "After hours",
    "PREPRE":     "Pre-market",
}


def fetch_market_status() -> dict:
    """Récupère l'horodatage et l'état du marché pour chaque marché via yfinance."""
    status: dict = {}
    for market, ticker in _MARKET_REPS.items():
        try:
            info     = yf.Ticker(ticker).info
            ts_raw   = info.get("regularMarketTime")
            state    = info.get("marketState", "UNKNOWN")
            tz_name  = info.get("exchangeTimezoneName") or info.get("timeZoneShortName") or "UTC"
            if ts_raw:
                dt_utc  = datetime.fromtimestamp(int(ts_raw), tz=dt_timezone.utc)
                local_tz = pytz.timezone(tz_name)
                dt_local = dt_utc.astimezone(local_tz)
                label    = _STATE_LABELS.get(state, state)
                status[market] = {
                    "timestamp_iso": dt_utc.isoformat(),
                    "display":       f"{label}: {dt_local.strftime('%H:%M:%S')} {dt_local.strftime('%Z')}",
                    "market_state":  state,
                    "timezone":      tz_name,
                }
        except Exception as e:
            logger.warning(f"[market_status/{ticker}] {e}")
    return status


def save_market_status(status: dict) -> None:
    try:
        os.makedirs(os.path.dirname(MARKET_STATUS_PATH), exist_ok=True)
        with open(MARKET_STATUS_PATH, "w") as f:
            json.dump(status, f)
    except Exception as e:
        logger.warning(f"[market_status] save failed: {e}")


def load_market_status() -> dict:
    try:
        if os.path.exists(MARKET_STATUS_PATH):
            with open(MARKET_STATUS_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


# ── Blacklist auto-détection tickers délistés ────────────────────────────────

def _load_blacklist() -> dict:
    try:
        if os.path.exists(BLACKLIST_PATH):
            with open(BLACKLIST_PATH) as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def _save_blacklist(bl: dict) -> None:
    try:
        os.makedirs(os.path.dirname(BLACKLIST_PATH), exist_ok=True)
        with open(BLACKLIST_PATH, "w") as f:
            json.dump(bl, f, indent=2, default=str)
    except Exception as e:
        logger.warning(f"[blacklist] save failed: {e}")


def _record_failure(ticker: str, market: str, bl: dict) -> None:
    entry = bl.get(ticker, {"failures": 0, "market": market})
    entry["failures"] = entry.get("failures", 0) + 1
    entry["last_failure"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
    entry["market"] = market
    bl[ticker] = entry
    _save_blacklist(bl)
    if entry["failures"] == _BLACKLIST_THRESHOLD:
        logger.warning(
            f"[blacklist] {ticker} auto-exclu après {entry['failures']} échecs consécutifs"
        )


def _record_success(ticker: str, bl: dict) -> None:
    if ticker in bl and bl[ticker].get("failures", 0) > 0:
        bl[ticker]["failures"] = 0
        _save_blacklist(bl)


def get_blacklisted() -> dict:
    """Retourne les tickers blacklistés (failures >= seuil)."""
    return {t: d for t, d in _load_blacklist().items()
            if d.get("failures", 0) >= _BLACKLIST_THRESHOLD}


def unblacklist(ticker: str) -> None:
    bl = _load_blacklist()
    if ticker in bl:
        del bl[ticker]
        _save_blacklist(bl)


# ── Helpers DB ────────────────────────────────────────────────────────────────

def _get_or_create_stock(db: Session, ticker: str, market: str, name: str | None = None) -> Stock:
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if not stock:
        try:
            stock = Stock(ticker=ticker, market=market, name=name)
            db.add(stock)
            db.flush()
        except Exception:
            db.rollback()
            stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if stock and name and not stock.name:
        stock.name = name
    return stock


def _upsert_row(db: Session, stock: Stock, row: pd.Series, date: datetime) -> None:
    existing = (
        db.query(DailyData)
        .filter(DailyData.stock_id == stock.id, DailyData.date == date)
        .first()
    )
    vals = dict(
        open=float(row["Open"]),
        high=float(row["High"]),
        low=float(row["Low"]),
        close=float(row["Close"]),
        volume=float(row["Volume"]),
    )
    if existing:
        for k, v in vals.items():
            setattr(existing, k, v)
    else:
        db.add(DailyData(stock_id=stock.id, date=date, **vals))


def _trim_rolling_window(db: Session, stock: Stock) -> None:
    rows = (
        db.query(DailyData)
        .filter(DailyData.stock_id == stock.id)
        .order_by(DailyData.date.asc())
        .all()
    )
    for row in rows[:max(0, len(rows) - ROLLING_WINDOW)]:
        db.delete(row)


# ── Download helpers ──────────────────────────────────────────────────────────

def _extract_ticker_df(raw: pd.DataFrame, ticker: str, n_tickers: int) -> pd.DataFrame:
    """Extrait le DataFrame d'un ticker depuis un résultat multi ou mono."""
    if n_tickers == 1:
        df = raw.copy()
    else:
        lvl0 = raw.columns.get_level_values(0)
        if ticker not in lvl0:
            return pd.DataFrame()
        df = raw[ticker].copy()

    # Aplatit le MultiIndex si nécessaire
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(-1)

    # Normalise : "close" → "Close" etc.
    rename = {c: c.capitalize() for c in df.columns if isinstance(c, str)}
    df = df.rename(columns=rename)

    if "Close" not in df.columns:
        return pd.DataFrame()
    return df.dropna(subset=["Close"])


def _download_batch(tickers: list[str], period: str) -> pd.DataFrame | None:
    try:
        return yf.download(
            " ".join(tickers),
            period=period,
            interval="1d",
            progress=False,
            auto_adjust=True,
            group_by="ticker",
        )
    except Exception as e:
        logger.warning(f"Batch download error: {e}")
        return None


def _download_single(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, period=period, interval="1d",
            progress=False, auto_adjust=True,
        )
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna(subset=["Close"])
    except Exception as e:
        logger.warning(f"[{ticker}] single download error: {e}")
        return pd.DataFrame()


# ── Sauvegarde d'un DataFrame pour un ticker ──────────────────────────────────

def _save_df(db: Session, ticker: str, market: str, df: pd.DataFrame, is_initial: bool, name: str | None = None) -> None:
    if df.empty:
        return
    stock = _get_or_create_stock(db, ticker, market, name=name)
    rows = df.tail(ROLLING_WINDOW) if is_initial else df
    for date, row in rows.iterrows():
        _upsert_row(db, stock, row, date.to_pydatetime())
    if not is_initial:
        _trim_rolling_window(db, stock)
    stock.last_updated = datetime.now(UTC).replace(tzinfo=None)
    db.commit()


# ── Point d'entrée principal ───────────────────────────────────────────────────

def update_all_markets(db: Session) -> None:
    bl = _load_blacklist()
    blacklisted = {t for t, d in bl.items() if d.get("failures", 0) >= _BLACKLIST_THRESHOLD}
    if blacklisted:
        logger.info(f"[blacklist] {len(blacklisted)} ticker(s) exclus : {sorted(blacklisted)}")

    markets = get_all_tickers()

    for market, tickers in markets.items():
        tickers = [t for t in tickers if t not in blacklisted]
        logger.info(f"=== {market} : {len(tickers)} tickers ===")
        batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]

        for b_idx, batch in enumerate(batches):
            # Identifie quels tickers ont besoin d'un chargement initial
            new_tickers, update_tickers = [], []
            for t in batch:
                stock = db.query(Stock).filter(Stock.ticker == t).first()
                count = db.query(DailyData).filter(DailyData.stock_id == stock.id).count() if stock else 0
                (new_tickers if count < 50 else update_tickers).append(t)

            # Chargement initial (1 an) — individuel pour fiabilité
            for ticker in new_tickers:
                df = _download_single(ticker, "1y")
                cname = COMMODITY_NAMES.get(ticker) or CRYPTO_NAMES.get(ticker) if market in ("COMMODITIES", "CRYPTO") else None
                if df.empty:
                    _record_failure(ticker, market, bl)
                else:
                    _record_success(ticker, bl)
                    try:
                        _save_df(db, ticker, market, df, is_initial=True, name=cname)
                    except Exception as e:
                        db.rollback()
                        logger.warning(f"[{ticker}] save failed: {e}")
                time.sleep(0.3)

            # Mise à jour quotidienne (5j) — batch d'abord, fallback individuel
            raw = None
            if update_tickers:
                raw = _download_batch(update_tickers, UPDATE_PERIOD)
                for ticker in update_tickers:
                    df = pd.DataFrame()
                    if raw is not None and not raw.empty:
                        df = _extract_ticker_df(raw, ticker, len(update_tickers))
                    if df.empty:
                        df = _download_single(ticker, UPDATE_PERIOD)
                    if df.empty:
                        _record_failure(ticker, market, bl)
                    else:
                        _record_success(ticker, bl)
                        cname = COMMODITY_NAMES.get(ticker) or CRYPTO_NAMES.get(ticker) if market in ("COMMODITIES", "CRYPTO") else None
                        try:
                            _save_df(db, ticker, market, df, is_initial=False, name=cname)
                        except Exception as e:
                            db.rollback()
                            logger.warning(f"[{ticker}] save failed: {e}")

            done = (b_idx + 1) * BATCH_SIZE
            logger.info(f"  {market}: {min(done, len(tickers))}/{len(tickers)} traités")
            if raw is not None:
                del raw
            time.sleep(1)  # pause entre batches

        gc.collect()
        logger.info(f"=== {market} terminé ===")


# ── Lecture des données pour analyse ──────────────────────────────────────────

def get_dataframe(db: Session, stock: Stock) -> pd.DataFrame:
    rows = (
        db.query(DailyData)
        .filter(DailyData.stock_id == stock.id)
        .order_by(DailyData.date.asc())
        .all()
    )
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame([{
        "Date": r.date, "Open": r.open, "High": r.high,
        "Low": r.low, "Close": r.close, "Volume": r.volume,
    } for r in rows])
    df.set_index("Date", inplace=True)
    return df


def sync_prices_fast(db: Session, on_progress=None) -> dict:
    """
    Sync rapide : télécharge les 5 derniers jours pour tous les stocks déjà en DB,
    upsert DailyData, puis recalcule les scores du jour.
    Pas de chargement initial (1y) — uniquement les stocks existants.
    Durée estimée : ~5-8 min pour 667 stocks.
    """
    from .indicators import compute_indicators, get_last_row
    from .ml_model import predict
    from .models import AnalysisResult
    from .scoring import compute_score

    bl = _load_blacklist()
    blacklisted = {t for t, d in bl.items() if d.get("failures", 0) >= _BLACKLIST_THRESHOLD}
    if blacklisted:
        logger.info(f"[blacklist] {len(blacklisted)} ticker(s) exclus du sync : {sorted(blacklisted)}")

    markets = get_all_tickers()
    all_tickers = [(t, m) for m, ts in markets.items() for t in ts if t not in blacklisted]
    total   = len(all_tickers)
    synced  = 0
    scored  = 0
    today   = datetime.now(UTC).replace(tzinfo=None).replace(hour=0, minute=0, second=0, microsecond=0)

    # ── Phase 1 : prix ────────────────────────────────────────────────────────
    flat = [t for t, _ in all_tickers]
    market_map = {t: m for t, m in all_tickers}
    batches = [flat[i:i + BATCH_SIZE] for i in range(0, len(flat), BATCH_SIZE)]

    for batch in batches:
        raw = _download_batch(batch, UPDATE_PERIOD)
        for ticker in batch:
            market = market_map[ticker]
            stock = db.query(Stock).filter(Stock.ticker == ticker).first()
            if not stock:
                # Auto-initialize new tickers (commodities / crypto) with 1y history
                cname = COMMODITY_NAMES.get(ticker) or CRYPTO_NAMES.get(ticker)
                df_init = _download_single(ticker, "1y")
                if not df_init.empty:
                    try:
                        _save_df(db, ticker, market, df_init, is_initial=True, name=cname)
                        stock = db.query(Stock).filter(Stock.ticker == ticker).first()
                    except Exception as e:
                        db.rollback()
                        logger.warning(f"[{ticker}] auto-init failed: {e}")
                else:
                    _record_failure(ticker, market, bl)
                if not stock:
                    synced += 1
                    if on_progress:
                        on_progress(synced, total, "prices")
                    continue
            df = pd.DataFrame()
            if raw is not None and not raw.empty:
                df = _extract_ticker_df(raw, ticker, len(batch))
            if df.empty:
                df = _download_single(ticker, UPDATE_PERIOD)
            if not df.empty:
                _record_success(ticker, bl)
                try:
                    for date, row in df.iterrows():
                        _upsert_row(db, stock, row, date.to_pydatetime())
                    stock.last_updated = datetime.now(UTC).replace(tzinfo=None)
                    db.commit()
                except Exception as e:
                    db.rollback()
                    logger.warning(f"[{ticker}] sync save error: {e}")
            else:
                _record_failure(ticker, market, bl)
            synced += 1
            if on_progress:
                on_progress(synced, total, "prices")
        del raw
        gc.collect()
        time.sleep(0.8)

    # ── Phase 2 : recalcul des scores ─────────────────────────────────────────
    stocks = db.query(Stock).all()
    total2 = len(stocks)
    for stock in stocks:
        try:
            df = get_dataframe(db, stock)
            if df.empty or len(df) < 30:
                scored += 1
                continue
            df   = compute_indicators(df)
            ind  = get_last_row(df)
            ml_prob = predict(df)
            score_base, ml_boost, score_final, ranking = compute_score(ind, ml_prob)

            existing = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.stock_id == stock.id, AnalysisResult.date == today)
                .first()
            )
            if not existing:
                existing = AnalysisResult(stock_id=stock.id, date=today)
                db.add(existing)

            existing.close           = ind.get("Close")
            existing.atr             = ind.get("ATR")
            existing.stop_loss_price = ind.get("Stop_Loss")
            existing.volatility      = ind.get("Volatility")
            existing.rsi             = ind.get("RSI")
            existing.macd            = ind.get("MACD")
            existing.macd_signal     = ind.get("MACD_signal")
            existing.macd_hist       = ind.get("MACD_hist")
            existing.bollinger_b     = ind.get("BB_pct")
            existing.ema50           = ind.get("EMA50")
            existing.sma200          = ind.get("SMA200")
            existing.volume_ratio    = ind.get("Vol_ratio")
            existing.score_base      = score_base
            existing.ml_probability  = ml_prob
            existing.ml_boost        = ml_boost
            existing.score_final     = score_final
            existing.ranking         = ranking
            db.commit()
        except Exception as e:
            db.rollback()
            logger.warning(f"[{stock.ticker}] score error: {e}")
        finally:
            del df
        scored += 1
        if scored % 50 == 0:
            gc.collect()
        if on_progress:
            on_progress(scored, total2, "scores")

    logger.info(f"[sync] {synced} prix + {scored} scores mis à jour")

    # Récupère et sauvegarde les timestamps des marchés
    try:
        status = fetch_market_status()
        save_market_status(status)
        logger.info(f"[sync] market status mis à jour: {list(status.keys())}")
    except Exception as e:
        logger.warning(f"[sync] market status failed: {e}")

    return {"synced": synced, "scored": scored}


def sync_selected_tickers(db: Session, tickers: list[str]) -> dict:
    """Sync rapide pour une liste de tickers spécifiques (prix 5j + rescore)."""
    from .indicators import compute_indicators, get_last_row
    from .ml_model import predict
    from .models import AnalysisResult
    from .scoring import compute_score

    markets  = get_all_tickers()
    mmap     = {t: m for m, ts in markets.items() for t in ts}
    today    = datetime.now(UTC).replace(tzinfo=None).replace(hour=0, minute=0, second=0, microsecond=0)
    synced   = 0
    scored   = 0

    # Phase 1 : prix
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    for batch in batches:
        raw = _download_batch(batch, UPDATE_PERIOD)
        for ticker in batch:
            stock = db.query(Stock).filter(Stock.ticker == ticker).first()
            if not stock:
                market = mmap.get(ticker, "SP500")
                cname  = COMMODITY_NAMES.get(ticker) or CRYPTO_NAMES.get(ticker)
                df_init = _download_single(ticker, "1y")
                if not df_init.empty:
                    try:
                        _save_df(db, ticker, market, df_init, is_initial=True, name=cname)
                        stock = db.query(Stock).filter(Stock.ticker == ticker).first()
                    except Exception as e:
                        db.rollback()
                        logger.warning(f"[{ticker}] auto-init failed: {e}")
            if not stock:
                continue
            df = pd.DataFrame()
            if raw is not None and not raw.empty:
                df = _extract_ticker_df(raw, ticker, len(batch))
            if df.empty:
                df = _download_single(ticker, UPDATE_PERIOD)
            if not df.empty:
                try:
                    for date, row in df.iterrows():
                        _upsert_row(db, stock, row, date.to_pydatetime())
                    stock.last_updated = datetime.now(UTC).replace(tzinfo=None)
                    db.commit()
                    synced += 1
                except Exception as e:
                    db.rollback()
                    logger.warning(f"[{ticker}] sync save error: {e}")
        time.sleep(0.5)

    # Phase 2 : rescore uniquement les tickers demandés
    for ticker in tickers:
        stock = db.query(Stock).filter(Stock.ticker == ticker).first()
        if not stock:
            continue
        try:
            df = get_dataframe(db, stock)
            if df.empty or len(df) < 30:
                continue
            df   = compute_indicators(df)
            ind  = get_last_row(df)
            ml_prob = predict(df)
            score_base, ml_boost, score_final, ranking = compute_score(ind, ml_prob)

            existing = (
                db.query(AnalysisResult)
                .filter(AnalysisResult.stock_id == stock.id, AnalysisResult.date == today)
                .first()
            )
            if not existing:
                existing = AnalysisResult(stock_id=stock.id, date=today)
                db.add(existing)

            existing.close           = ind.get("Close")
            existing.atr             = ind.get("ATR")
            existing.stop_loss_price = ind.get("Stop_Loss")
            existing.volatility      = ind.get("Volatility")
            existing.rsi             = ind.get("RSI")
            existing.macd            = ind.get("MACD")
            existing.macd_signal     = ind.get("MACD_signal")
            existing.macd_hist       = ind.get("MACD_hist")
            existing.bollinger_b     = ind.get("BB_pct")
            existing.ema50           = ind.get("EMA50")
            existing.sma200          = ind.get("SMA200")
            existing.volume_ratio    = ind.get("Vol_ratio")
            existing.score_base      = score_base
            existing.ml_probability  = ml_prob
            existing.ml_boost        = ml_boost
            existing.score_final     = score_final
            existing.ranking         = ranking
            db.commit()
            scored += 1
        except Exception as e:
            db.rollback()
            logger.warning(f"[{ticker}] score error: {e}")

    logger.info(f"[sync_selected] {synced} prix + {scored} scores — {tickers}")
    return {"synced": synced, "scored": scored, "tickers": tickers}


def get_current_price(ticker: str) -> float | None:
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if data.empty:
            return None
        close = data["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        val = close.iloc[-1]
        return float(val.item() if hasattr(val, "item") else val)
    except Exception:
        return None
