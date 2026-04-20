from __future__ import annotations
"""
Data engine: télécharge les données yfinance et maintient
une fenêtre glissante de ROLLING_WINDOW jours par action.

Stratégie : batch de 20 tickers, puis fallback individuel si erreur.
"""
import logging
import time
from datetime import datetime

import pandas as pd
import yfinance as yf
from sqlalchemy.orm import Session

try:
    from curl_cffi import requests as cf_requests
    _CF_SESSION = cf_requests.Session(impersonate="chrome120")
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.info("curl_cffi disponible — impersonation Chrome activée")
except ImportError:
    _CF_SESSION = None


def _yf_session():
    """Retourne une session curl_cffi (bypass Yahoo IP block) ou None."""
    return _CF_SESSION

from .config import ROLLING_WINDOW
from .models import DailyData, Stock
from .tickers import get_all_tickers

logger = logging.getLogger(__name__)

BATCH_SIZE = 20   # conservative pour éviter le rate limiting


# ── Helpers DB ────────────────────────────────────────────────────────────────

def _get_or_create_stock(db: Session, ticker: str, market: str) -> Stock:
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if not stock:
        stock = Stock(ticker=ticker, market=market)
        db.add(stock)
        db.flush()
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
            session=_yf_session(),
        )
    except Exception as e:
        logger.warning(f"Batch download error: {e}")
        return None


def _download_single(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(
            ticker, period=period, interval="1d",
            progress=False, auto_adjust=True,
            session=_yf_session(),
        )
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]
        return df.dropna(subset=["Close"])
    except Exception as e:
        logger.warning(f"[{ticker}] single download error: {e}")
        return pd.DataFrame()


# ── Sauvegarde d'un DataFrame pour un ticker ──────────────────────────────────

def _save_df(db: Session, ticker: str, market: str, df: pd.DataFrame, is_initial: bool) -> None:
    if df.empty:
        return
    stock = _get_or_create_stock(db, ticker, market)
    rows = df.tail(ROLLING_WINDOW) if is_initial else df
    for date, row in rows.iterrows():
        _upsert_row(db, stock, row, date.to_pydatetime())
    if not is_initial:
        _trim_rolling_window(db, stock)
    stock.last_updated = datetime.utcnow()
    db.commit()


# ── Point d'entrée principal ───────────────────────────────────────────────────

def update_all_markets(db: Session) -> None:
    markets = get_all_tickers()

    for market, tickers in markets.items():
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
                try:
                    _save_df(db, ticker, market, df, is_initial=True)
                except Exception as e:
                    db.rollback()
                    logger.warning(f"[{ticker}] save failed: {e}")
                time.sleep(0.3)

            # Mise à jour quotidienne (5j) — batch d'abord, fallback individuel
            if update_tickers:
                raw = _download_batch(update_tickers, "5d")
                for ticker in update_tickers:
                    df = pd.DataFrame()
                    if raw is not None and not raw.empty:
                        df = _extract_ticker_df(raw, ticker, len(update_tickers))
                    if df.empty:
                        df = _download_single(ticker, "5d")
                    try:
                        _save_df(db, ticker, market, df, is_initial=False)
                    except Exception as e:
                        db.rollback()
                        logger.warning(f"[{ticker}] save failed: {e}")

            done = (b_idx + 1) * BATCH_SIZE
            logger.info(f"  {market}: {min(done, len(tickers))}/{len(tickers)} traités")
            time.sleep(1)  # pause entre batches

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

    markets = get_all_tickers()
    all_tickers = [(t, m) for m, ts in markets.items() for t in ts]
    total   = len(all_tickers)
    synced  = 0
    scored  = 0
    today   = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # ── Phase 1 : prix ────────────────────────────────────────────────────────
    flat = [t for t, _ in all_tickers]
    market_map = {t: m for t, m in all_tickers}
    batches = [flat[i:i + BATCH_SIZE] for i in range(0, len(flat), BATCH_SIZE)]

    for batch in batches:
        raw = _download_batch(batch, "5d")
        for ticker in batch:
            stock = db.query(Stock).filter(Stock.ticker == ticker).first()
            if not stock:
                synced += 1
                continue
            df = pd.DataFrame()
            if raw is not None and not raw.empty:
                df = _extract_ticker_df(raw, ticker, len(batch))
            if df.empty:
                df = _download_single(ticker, "5d")
            if not df.empty:
                try:
                    for date, row in df.iterrows():
                        _upsert_row(db, stock, row, date.to_pydatetime())
                    stock.last_updated = datetime.utcnow()
                    db.commit()
                except Exception as e:
                    db.rollback()
                    logger.warning(f"[{ticker}] sync save error: {e}")
            synced += 1
            if on_progress:
                on_progress(synced, total, "prices")
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
        scored += 1
        if on_progress:
            on_progress(scored, total2, "scores")

    logger.info(f"[sync] {synced} prix + {scored} scores mis à jour")
    return {"synced": synced, "scored": scored}


def get_current_price(ticker: str) -> float | None:
    try:
        data = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
        if data.empty:
            return None
        close = data["Close"]
        if hasattr(close, "iloc"):
            return float(close.iloc[-1])
        return None
    except Exception:
        return None
