from __future__ import annotations
import io
from datetime import UTC, datetime

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


def _compute_weighted(score_final: float, fundamental_score, wt: int, wf: int) -> tuple[int, str]:
    tech = score_final or 0
    if wf == 0 or fundamental_score is None:
        wscore = tech
    elif wt == 0:
        wscore = fundamental_score
    else:
        wscore = round(tech * wt / 100 + fundamental_score * wf / 100)
    wscore = int(wscore)
    if wscore >= 75:
        return wscore, "Strong Buy"
    if wscore >= 58:
        return wscore, "Buy"
    if wscore >= 42:
        return wscore, "Neutral"
    return wscore, "Avoid"


@router.get("/dashboard", response_class=HTMLResponse)
def dashboard(
    request: Request,
    market: str  = Query("CAC40"),
    ranking: str = Query(""),
    sector: str  = Query(""),
    weight: str  = Query("65-35"),
    db: Session  = Depends(get_db),
    user: User   = Depends(get_current_user),
):
    try:
        wt, wf = [int(x) for x in weight.split("-")]
        if wt + wf != 100:
            wt, wf = 65, 35
    except Exception:
        wt, wf = 65, 35
    weight = f"{wt}-{wf}"

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
        if sector:
            query = query.filter(Stock.sector == sector)

        rows = query.limit(600).all()

        results = []
        for ar, stock in rows:
            wscore, wranking = _compute_weighted(ar.score_final, ar.fundamental_score, wt, wf)
            results.append({
                "ticker":             stock.ticker,
                "name":               stock.name or "",
                "market":             stock.market,
                "sector":             stock.sector or "",
                "close":              round(ar.close or 0, 2),
                "score_final":        ar.score_final or 0,
                "score_composite":    ar.score_composite,
                "fundamental_score":  ar.fundamental_score,
                "ranking":            ar.ranking or "Neutral",
                "weighted_score":     wscore,
                "weighted_ranking":   wranking,
                "weighted_emoji":     RANKING_EMOJI.get(wranking, "⚪"),
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
                "peg_ratio":          round(ar.peg_ratio, 2)  if ar.peg_ratio  else None,
                "ev_ebit":            round(ar.ev_ebit, 1)    if ar.ev_ebit    else None,
                "ev_ebitda":          round(ar.ev_ebitda, 1)  if ar.ev_ebitda  else None,
                "fcf":                ar.fcf,
            })

        results.sort(key=lambda x: x["weighted_score"], reverse=True)

    # Secteurs disponibles pour le marché sélectionné (ou tous si pas de marché)
    sector_q = db.query(Stock.sector).filter(Stock.sector.isnot(None), Stock.sector != "")
    if market:
        sector_q = sector_q.filter(Stock.market == market)
    sectors = sorted({r[0] for r in sector_q.distinct().all()})

    ml_metrics    = load_metrics()
    market_status = load_market_status()
    markets       = ["CAC40", "SBF120", "SP500", "COMMODITIES", "CRYPTO"]
    rankings      = ["Strong Buy", "Buy", "Neutral", "Avoid"]
    last_update   = last_date.strftime("%d/%m/%Y") if last_date else "Aucune donnée"
    quote_status  = market_status.get(market, {}).get("display")
    market_state  = market_status.get(market, {}).get("market_state", "")
    is_open       = market_state == "REGULAR"

    return templates.TemplateResponse(request, "dashboard.html", {
        "user":         user,
        "results":      results,
        "markets":      markets,
        "rankings":     rankings,
        "sectors":      sectors,
        "sel_market":   market,
        "sel_ranking":  ranking,
        "sel_sector":   sector,
        "sel_weight":   weight,
        "last_update":  last_update,
        "ml_metrics":   ml_metrics,
        "quote_status": quote_status,
        "is_open":      is_open,
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
        "Ticker", "Nom", "Marché", "Secteur", "Prix", "Score Final", "Score Composite",
        "Score Fond.", "Signal", "RSI", "MACD ▲▼", "Volatilité %",
        "Stop-Loss", "ML Prob %", "ATR %", "Bollinger %B",
        "P/E", "P/B", "ROE %", "D/E", "Croiss. Rev %",
        "PEG", "EV/EBIT", "EV/EBITDA", "FCF",
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
            stock.sector or "",
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
            round(ar.peg_ratio, 2)  if ar.peg_ratio  else None,
            round(ar.ev_ebit, 1)    if ar.ev_ebit    else None,
            round(ar.ev_ebitda, 1)  if ar.ev_ebitda  else None,
            ar.fcf,
        ]
        ws.append(row)
        r = ws.max_row
        signal_cell = ws.cell(r, 9)
        if ar.ranking in ("Strong Buy", "Buy"):
            signal_cell.font = green_font
        elif ar.ranking == "Avoid":
            signal_cell.font = red_font
        score_cell = ws.cell(r, 6)
        if (ar.score_final or 0) >= 75:
            score_cell.font = green_font
        elif (ar.score_final or 0) < 42:
            score_cell.font = red_font

    col_widths = [10, 28, 12, 20, 10, 12, 14, 12, 12, 8, 10, 12,
                  11, 10, 8, 12, 8, 8, 8, 8, 12]
    for i, w in enumerate(col_widths, 1):
        ws.column_dimensions[openpyxl.utils.get_column_letter(i)].width = w

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    date_str = last_date.strftime("%Y-%m-%d")
    filename = f"StockAnalyzer_{market}_{date_str}.xlsx"
    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
