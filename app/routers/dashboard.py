from __future__ import annotations
import io
from datetime import UTC, datetime, timedelta

import openpyxl
from openpyxl.styles import Alignment, Font, PatternFill
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..data_engine import load_market_status
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
            "macd_hist":          round(ar.macd_hist or 0, 4),
            "macd_bull":          (ar.macd_hist or 0) > 0,
            "stop_loss":          round(ar.stop_loss_price or 0, 2),
            "volatility":         round(ar.volatility or 0, 1),
            "ml_prob":            round(ar.ml_probability * 100, 1) if ar.ml_probability else None,
            "bollinger_b":        round(ar.bollinger_b or 0, 2),
            "atr_pct":            round((ar.atr / ar.close * 100) if ar.atr and ar.close else 0, 2),
            "adx":                round(ar.adx or 0, 1) if ar.adx is not None else None,
            "sma200_slope":       round(ar.sma200_slope or 0, 2) if ar.sma200_slope is not None else None,
            "atr_pct_rank":       round(ar.atr_pct_rank or 0, 0) if ar.atr_pct_rank is not None else None,
            "bb_zscore":          round(ar.bb_zscore or 0, 2) if ar.bb_zscore is not None else None,
            "pe_ratio":           ar.pe_ratio,
            "pb_ratio":           ar.pb_ratio,
            "roe":                ar.roe,
            "debt_equity":        round(ar.debt_equity / 100, 2) if ar.debt_equity else None,
            "rev_growth":         ar.rev_growth,
        } for ar, stock in rows]

    ml_metrics     = load_metrics()
    market_status  = load_market_status()
    markets        = ["CAC40", "SBF120", "SP500", "COMMODITIES", "CRYPTO"]
    rankings       = ["Strong Buy", "Buy", "Neutral", "Avoid"]
    last_update    = last_date.strftime("%d/%m/%Y") if last_date else "Aucune donnée"
    quote_status   = market_status.get(market, {}).get("display")  # ex: "At close: 17:37:12 CET"
    market_state   = market_status.get(market, {}).get("market_state", "")
    is_open        = market_state == "REGULAR"

    return templates.TemplateResponse(request, "dashboard.html", {
        "user":          user,
        "results":       results,
        "markets":       markets,
        "rankings":      rankings,
        "sel_market":    market,
        "sel_ranking":   ranking,
        "last_update":   last_update,
        "ml_metrics":    ml_metrics,
        "quote_status":  quote_status,
        "is_open":       is_open,
    })


@router.get("/dashboard/export")
def export_excel(
    market:  str   = Query("CAC40"),
    tickers: str   = Query(""),        # comma-separated, empty = all
    db: Session    = Depends(get_db),
    user: User     = Depends(get_current_user),
):
    last_date = (
        db.query(AnalysisResult.date)
        .order_by(AnalysisResult.date.desc())
        .limit(1)
        .scalar()
    )
    if not last_date:
        return HTMLResponse("Aucune donnée", status_code=404)

    ticker_filter = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    query = (
        db.query(AnalysisResult, Stock)
        .join(Stock, AnalysisResult.stock_id == Stock.id)
        .filter(AnalysisResult.date == last_date)
    )
    if ticker_filter:
        # Sélection cross-marché : pas de filtre sur le marché
        query = query.filter(Stock.ticker.in_(ticker_filter))
    else:
        query = query.filter(Stock.market == market)

    rows = query.order_by(AnalysisResult.score_final.desc()).all()

    # ── Build xlsx ────────────────────────────────────────────────────────────
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = market

    header_fill = PatternFill("solid", fgColor="1E293B")
    header_font = Font(bold=True, color="E2E8F0")
    green_font  = Font(color="4ADE80", bold=True)
    red_font    = Font(color="F87171", bold=True)

    headers = [
        "Ticker", "Nom", "Marché", "Prix", "Score Final", "Score Composite",
        "Score Fond.", "Signal", "RSI", "MACD ▲▼", "Volatilité %",
        "Stop-Loss", "ML Prob %", "ATR %", "Bollinger %B",
        "P/E", "P/B", "ROE %", "D/E", "Croiss. Rev %",
    ]
    ws.append(headers)
    for cell in ws[1]:
        cell.fill = header_fill
        cell.font = header_font
        cell.alignment = Alignment(horizontal="center")

    for ar, stock in rows:
        de  = round(ar.debt_equity / 100, 2) if ar.debt_equity else None
        row = [
            stock.ticker,
            stock.name or "",
            stock.market,
            round(ar.close or 0, 2),
            ar.score_final or 0,
            ar.score_composite,
            ar.fundamental_score,
            ar.ranking or "Neutral",
            round(ar.rsi or 0, 1),
            "▲" if (ar.macd_hist or 0) > 0 else "▼",
            round(ar.volatility or 0, 1),
            round(ar.stop_loss_price or 0, 2),
            round((ar.ml_probability or 0) * 100, 1),
            round((ar.atr / ar.close * 100) if ar.atr and ar.close else 0, 2),
            round(ar.bollinger_b or 0, 2),
            ar.pe_ratio,
            ar.pb_ratio,
            round(ar.roe * 100, 1) if ar.roe else None,
            de,
            round(ar.rev_growth * 100, 1) if ar.rev_growth else None,
        ]
        ws.append(row)
        r = ws.max_row
        # Colorier le signal
        signal_cell = ws.cell(r, 8)
        if ar.ranking in ("Strong Buy", "Buy"):
            signal_cell.font = green_font
        elif ar.ranking == "Avoid":
            signal_cell.font = red_font
        # Colorier score final
        score_cell = ws.cell(r, 5)
        if (ar.score_final or 0) >= 75:
            score_cell.font = green_font
        elif (ar.score_final or 0) < 42:
            score_cell.font = red_font

    # Largeurs de colonnes
    col_widths = [10, 28, 12, 10, 12, 14, 12, 12, 8, 10, 12,
                  11, 10, 8, 12, 8, 8, 8, 8, 12]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    date_str  = last_date.strftime("%Y-%m-%d")
    filename  = f"StockAnalyzer_{market}_{date_str}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
