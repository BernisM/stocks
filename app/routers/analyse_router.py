from __future__ import annotations
import logging

import numpy as np
import pandas as pd
import yfinance as yf
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from ..auth import get_current_user
from ..indicators import compute_indicators, get_last_row
from ..ml_model import predict
from ..models import User
from ..scoring import compute_score, RANKING_EMOJI

router    = APIRouter()
templates = Jinja2Templates(directory="templates")
logger    = logging.getLogger(__name__)


@router.get("/analyse", response_class=HTMLResponse)
def analyse_page(request: Request, user: User = Depends(get_current_user)):
    return templates.TemplateResponse(request, "analyse.html", {"user": user})


@router.get("/api/analyse/search")
def analyse_search(
    q: str = Query(""),
    user: User = Depends(get_current_user),
):
    """Autocomplete : cherche tickers/ISINs via yfinance Search."""
    if len(q.strip()) < 2:
        return JSONResponse([])
    try:
        results = yf.Search(q.strip(), max_results=8, enable_fuzzy_query=True)
        quotes  = getattr(results, "quotes", []) or []
        return JSONResponse([{
            "symbol":   r.get("symbol", ""),
            "name":     r.get("shortname") or r.get("longname") or "",
            "exchange": r.get("exchange", ""),
            "type":     r.get("quoteType", ""),
        } for r in quotes if r.get("symbol")])
    except Exception as e:
        logger.warning(f"yf.Search error for '{q}': {e}")
        return JSONResponse([])


@router.post("/api/analyse/run")
async def analyse_run(request: Request, user: User = Depends(get_current_user)):
    """Analyse complète d'un ticker/ISIN quelconque (hors liste)."""
    body  = await request.json()
    query = body.get("q", "").strip()
    if not query:
        return JSONResponse({"error": "Requête vide"}, status_code=400)

    ticker_sym = query.upper()

    # Si ça ressemble à un ISIN (2 lettres + 10 alphanumériques), cherche via yf.Search
    is_isin = len(query) == 12 and query[:2].isalpha() and query[2:].isalnum()
    if is_isin:
        try:
            results = yf.Search(query, max_results=3)
            quotes  = getattr(results, "quotes", []) or []
            if quotes:
                ticker_sym = quotes[0].get("symbol", ticker_sym)
        except Exception:
            pass

    # Téléchargement 1 an via yfinance
    try:
        ticker_obj = yf.Ticker(ticker_sym)
        hist = ticker_obj.history(period="1y", auto_adjust=True)
    except Exception as e:
        return JSONResponse({"error": f"Erreur téléchargement : {e}"}, status_code=500)

    if hist is None or hist.empty:
        return JSONResponse(
            {"error": f"Aucune donnée trouvée pour « {query} ». Vérifiez le ticker ou l'ISIN."},
            status_code=404,
        )

    # Normalisation du DataFrame pour compute_indicators
    hist.index = pd.to_datetime(hist.index).tz_localize(None)
    df = hist[["Open", "High", "Low", "Close", "Volume"]].copy()
    df = df.dropna(subset=["Close"])

    if len(df) < 30:
        return JSONResponse(
            {"error": f"Données insuffisantes ({len(df)} jours). Minimum 30 jours requis."},
            status_code=422,
        )

    # Calcul des indicateurs + score
    df   = compute_indicators(df)
    ind  = get_last_row(df)
    ml_prob = predict(df)
    score_base, ml_boost, score_final, ranking = compute_score(ind, ml_prob)

    # Infos fondamentales (best-effort — peut être vide pour ETF/fonds)
    info = {}
    try:
        info = ticker_obj.info or {}
    except Exception:
        pass

    close = ind.get("Close") or 0
    atr   = ind.get("ATR")

    return JSONResponse({
        "ticker":       ticker_sym,
        "query":        query,
        "is_isin":      is_isin,
        "name":         info.get("longName") or info.get("shortName") or ticker_sym,
        "currency":     info.get("currency", ""),
        "exchange":     info.get("exchange", "") or info.get("fullExchangeName", ""),
        "sector":       info.get("sector", "") or info.get("category", ""),
        "asset_type":   info.get("quoteType", ""),
        "n_days":       len(df),
        # Prix & risque
        "close":        round(close, 4),
        "stop_loss":    round(ind.get("Stop_Loss") or 0, 4),
        "atr_pct":      round((atr / close * 100) if atr and close else 0, 2),
        # Score
        "score_final":  score_final,
        "score_base":   score_base,
        "ml_boost":     ml_boost,
        "ml_prob":      round((ml_prob or 0) * 100, 1),
        "ranking":      ranking,
        "emoji":        RANKING_EMOJI.get(ranking, "⚪"),
        # Indicateurs techniques
        "rsi":          round(ind.get("RSI") or 0, 1),
        "macd_hist":    round(ind.get("MACD_hist") or 0, 4),
        "macd_bull":    (ind.get("MACD_hist") or 0) > 0,
        "volatility":   round(ind.get("Volatility") or 0, 1),
        "bollinger_b":  round(ind.get("BB_pct") or 0, 2),
        "adx":          round(ind.get("ADX") or 0, 1),
        "sma200_slope": round(ind.get("SMA200_slope") or 0, 2),
        "atr_pct_rank": round(ind.get("ATR_pct_rank") or 0, 0),
        "bb_zscore":    round(ind.get("BB_zscore") or 0, 2),
        "obv_slope":    round(ind.get("OBV_slope") or 0, 0),
        "vol_ratio":    round(ind.get("Vol_ratio") or 0, 2),
        # Régimes
        "regime_trend":    int(ind.get("regime_trend") or 0),
        "regime_bull":     int(ind.get("regime_bull") or 0),
        "regime_vol_high": int(ind.get("regime_vol_high") or 0),
        # Fondamentaux (si disponibles)
        "pe_ratio":     info.get("forwardPE") or info.get("trailingPE"),
        "pb_ratio":     info.get("priceToBook"),
        "roe":          round(info.get("returnOnEquity", 0) * 100, 1) if info.get("returnOnEquity") else None,
        "debt_equity":  round(info.get("debtToEquity", 0) / 100, 2) if info.get("debtToEquity") else None,
        "rev_growth":   round(info.get("revenueGrowth", 0) * 100, 1) if info.get("revenueGrowth") else None,
        "market_cap":   info.get("marketCap"),
        "dividend_yield": round(info.get("dividendYield", 0) * 100, 2) if info.get("dividendYield") else None,
    })
