from __future__ import annotations
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..ml_model import load_metrics
from ..models import AnalysisResult, Stock, User
from ..scoring import RANKING_EMOJI

router    = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    market: str = Query("CAC40"),
    ranking: str = Query(""),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    today = datetime.now(UTC).replace(tzinfo=None).replace(hour=0, minute=0, second=0, microsecond=0)
    # Si pas de données aujourd'hui, prendre la dernière date disponible
    last_date = (
        db.query(AnalysisResult.date)
        .order_by(AnalysisResult.date.desc())
        .limit(1)
        .scalar()
    )
    if last_date is None:
        results = []
    else:
        query = (
            db.query(AnalysisResult, Stock)
            .join(Stock, AnalysisResult.stock_id == Stock.id)
            .filter(AnalysisResult.date == last_date)
        )
        if market:
            query = query.filter(Stock.market == market)
        if ranking:
            query = query.filter(AnalysisResult.ranking == ranking)

        rows = query.order_by(AnalysisResult.score_final.desc()).limit(600).all()

        results = [{
            "ticker":             stock.ticker,
            "name":               stock.name or "",
            "market":             stock.market,
            "close":              round(ar.close or 0, 2),
            "score_final":        ar.score_final or 0,
            "score_composite":    ar.score_composite,
            "fundamental_score":  ar.fundamental_score,
            "ranking":            ar.ranking or "Neutral",
            "emoji":              RANKING_EMOJI.get(ar.ranking, "⚪"),
            "rsi":                round(ar.rsi or 0, 1),
            "stop_loss":          round(ar.stop_loss_price or 0, 2),
            "volatility":         round(ar.volatility or 0, 1),
            "macd_bull":          (ar.macd_hist or 0) > 0,
            "ml_prob":            round((ar.ml_probability or 0) * 100, 1),
            "bollinger_b":        round(ar.bollinger_b or 0, 2),
            "atr_pct":            round((ar.atr / ar.close * 100) if ar.atr and ar.close else 0, 2),
            "pe_ratio":           ar.pe_ratio,
            "pb_ratio":           ar.pb_ratio,
            "roe":                ar.roe,
            "debt_equity":        round(ar.debt_equity / 100, 2) if ar.debt_equity else None,
            "rev_growth":         ar.rev_growth,
        } for ar, stock in rows]

    ml_metrics  = load_metrics()
    markets     = ["CAC40", "SBF120", "SP500", "COMMODITIES"]
    rankings    = ["Strong Buy", "Buy", "Neutral", "Avoid"]
    last_update = last_date.strftime("%d/%m/%Y") if last_date else "Aucune donnée"

    return templates.TemplateResponse(request, "dashboard.html", {
        "user":        user,
        "results":     results,
        "markets":     markets,
        "rankings":    rankings,
        "sel_market":  market,
        "sel_ranking": ranking,
        "last_update": last_update,
        "ml_metrics":  ml_metrics,
    })
