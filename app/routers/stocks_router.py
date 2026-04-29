from __future__ import annotations
import csv
import io

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import func, or_
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import Stock, User

router = APIRouter()


@router.get("/stocks/search")
def search_stocks(
    q: str = Query(""),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    if len(q.strip()) < 2:
        return JSONResponse([])
    pattern = f"%{q.strip().lower()}%"
    stocks = (
        db.query(Stock)
        .filter(Stock.is_active.is_(True))
        .filter(or_(
            func.lower(Stock.ticker).like(pattern),
            func.lower(Stock.name).like(pattern),
            func.lower(Stock.isin).like(pattern),
        ))
        .order_by(Stock.market, Stock.ticker)
        .limit(30)
        .all()
    )
    return JSONResponse([{
        "ticker": s.ticker,
        "name":   s.name or "",
        "market": s.market,
        "isin":   s.isin or "",
    } for s in stocks])


@router.get("/stocks/template-csv")
def stocks_template_csv(
    tickers: str = Query(""),
    db: Session = Depends(get_db),
    user: User  = Depends(get_current_user),
):
    ticker_list = [t.strip().upper() for t in tickers.split(",") if t.strip()]
    stocks = (
        db.query(Stock).filter(Stock.ticker.in_(ticker_list)).all()
        if ticker_list else []
    )
    # Preserve the requested order
    stock_map = {s.ticker: s for s in stocks}
    ordered   = [stock_map[t] for t in ticker_list if t in stock_map]

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ticker", "name", "isin", "shares", "buy_price", "buy_date", "fees", "notes"])
    for s in ordered:
        writer.writerow([s.ticker, s.name or "", s.isin or "", "", "", "", "", ""])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio_selection.csv"},
    )
