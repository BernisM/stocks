from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import exc as sa_exc

from ..auth import get_current_user
from ..database import get_db
from ..models import AnalysisResult, Stock, User, WatchlistItem

router    = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.post("/watchlist/toggle/{ticker}")
def toggle_watchlist(ticker: str, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    ticker = ticker.upper()
    existing = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id, WatchlistItem.ticker == ticker)
        .first()
    )
    if existing:
        db.delete(existing)
        db.commit()
        return JSONResponse({"watching": False})
    try:
        db.add(WatchlistItem(user_id=user.id, ticker=ticker))
        db.commit()
    except sa_exc.IntegrityError:
        db.rollback()
    return JSONResponse({"watching": True})


@router.get("/watchlist", response_class=HTMLResponse)
def watchlist_page(request: Request, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    items = (
        db.query(WatchlistItem)
        .filter(WatchlistItem.user_id == user.id)
        .order_by(WatchlistItem.added_at.desc())
        .all()
    )
    tickers = [w.ticker for w in items]

    # Dernière date d'analyse dispo
    last_date = (
        db.query(AnalysisResult.date)
        .order_by(AnalysisResult.date.desc())
        .limit(1)
        .scalar()
    )

    results = []
    if tickers and last_date:
        rows = (
            db.query(AnalysisResult, Stock)
            .join(Stock, AnalysisResult.stock_id == Stock.id)
            .filter(
                AnalysisResult.date == last_date,
                Stock.ticker.in_(tickers),
            )
            .order_by(AnalysisResult.score_final.desc())
            .all()
        )
        for ar, stock in rows:
            results.append({
                "ticker":            stock.ticker,
                "name":              stock.name or "",
                "market":            stock.market,
                "close":             round(ar.close or 0, 2),
                "score_final":       ar.score_final or 0,
                "fundamental_score": ar.fundamental_score,
                "ranking":           ar.ranking or "Neutral",
                "rsi":               round(ar.rsi or 0, 1),
                "macd_bull":         (ar.macd_hist or 0) > 0,
                "macd_hist":         round(ar.macd_hist or 0, 4),
                "volatility":        round(ar.volatility or 0, 1),
                "stop_loss":         round(ar.stop_loss_price or 0, 2),
                "ml_prob":           round(ar.ml_probability * 100, 1) if ar.ml_probability else None,
                "hyper_growth_score": ar.hyper_growth_score,
            })

    # Tickers en watchlist sans analyse (pas encore scorés)
    scored_tickers = {r["ticker"] for r in results}
    unscored = [t for t in tickers if t not in scored_tickers]

    return templates.TemplateResponse(request, "watchlist.html", {
        "user":     user,
        "results":  results,
        "unscored": unscored,
        "count":    len(tickers),
        "last_update": last_date.strftime("%d/%m/%Y") if last_date else "—",
    })
