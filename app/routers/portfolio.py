from __future__ import annotations
import csv
import io
from datetime import datetime

from fastapi import APIRouter, Depends, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..data_engine import get_current_price
from ..models import AnalysisResult, PortfolioPosition, Stock, User

router    = APIRouter()
templates = Jinja2Templates(directory="templates")


def _compute_pnl(pos: PortfolioPosition, current_price: float | None) -> dict:
    cp = current_price or pos.buy_price
    pnl_abs = (cp - pos.buy_price) * pos.shares
    pnl_pct = (cp - pos.buy_price) / pos.buy_price * 100 if pos.buy_price else 0
    stop_alert = current_price is not None and pos.stop_loss_price and current_price <= pos.stop_loss_price
    return {
        "id":            pos.id,
        "ticker":        pos.ticker,
        "name":          pos.name,
        "shares":        pos.shares,
        "buy_price":     round(pos.buy_price, 2),
        "buy_date":      pos.buy_date.strftime("%d/%m/%Y"),
        "current_price": round(cp, 2),
        "stop_loss":     round(pos.stop_loss_price, 2) if pos.stop_loss_price else None,
        "pnl_abs":       round(pnl_abs, 2),
        "pnl_pct":       round(pnl_pct, 2),
        "stop_alert":    stop_alert,
        "notes":         pos.notes,
    }


@router.get("/portfolio", response_class=HTMLResponse)
def portfolio_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    positions = (
        db.query(PortfolioPosition)
        .filter(PortfolioPosition.user_id == user.id, PortfolioPosition.is_active == True)
        .order_by(PortfolioPosition.buy_date.desc())
        .all()
    )
    rows = []
    total_value = 0
    total_cost  = 0

    for pos in positions:
        price = get_current_price(pos.ticker)
        row   = _compute_pnl(pos, price)
        rows.append(row)
        total_value += row["current_price"] * pos.shares
        total_cost  += pos.buy_price       * pos.shares

    total_pnl_abs = round(total_value - total_cost, 2)
    total_pnl_pct = round((total_value / total_cost - 1) * 100, 2) if total_cost else 0

    return templates.TemplateResponse("portfolio.html", {
        "request":       request,
        "user":          user,
        "positions":     rows,
        "total_value":   round(total_value, 2),
        "total_pnl_abs": total_pnl_abs,
        "total_pnl_pct": total_pnl_pct,
        "error":         None,
    })


@router.post("/portfolio/add")
def add_position(
    ticker:     str  = Form(...),
    name:       str  = Form(""),
    shares:     float = Form(...),
    buy_price:  float = Form(...),
    buy_date:   str  = Form(...),
    notes:      str  = Form(""),
    db: Session = Depends(get_db),
    user: User  = Depends(get_current_user),
):
    try:
        date = datetime.strptime(buy_date, "%Y-%m-%d")
    except ValueError:
        date = datetime.utcnow()

    # Calcul du stop-loss ATR depuis la dernière analyse
    stop_loss = None
    stock = db.query(Stock).filter(Stock.ticker == ticker.upper()).first()
    if stock:
        ar = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.stock_id == stock.id)
            .order_by(AnalysisResult.date.desc())
            .first()
        )
        if ar and ar.stop_loss_price:
            stop_loss = ar.stop_loss_price

    pos = PortfolioPosition(
        user_id         = user.id,
        ticker          = ticker.upper().strip(),
        name            = name.strip(),
        shares          = shares,
        buy_price       = buy_price,
        buy_date        = date,
        stop_loss_price = stop_loss,
        notes           = notes.strip(),
    )
    db.add(pos)
    db.commit()
    return RedirectResponse("/portfolio", status_code=302)


@router.post("/portfolio/delete/{position_id}")
def delete_position(
    position_id: int,
    db: Session = Depends(get_db),
    user: User  = Depends(get_current_user),
):
    pos = (
        db.query(PortfolioPosition)
        .filter(PortfolioPosition.id == position_id, PortfolioPosition.user_id == user.id)
        .first()
    )
    if pos:
        pos.is_active = False
        db.commit()
    return RedirectResponse("/portfolio", status_code=302)


@router.get("/portfolio/template")
def download_template():
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["ticker", "name", "shares", "buy_price", "buy_date", "notes"])
    writer.writerow(["AAPL", "Apple Inc.", "10", "150.00", "2024-01-15", "Exemple"])
    writer.writerow(["MC.PA", "LVMH", "2", "750.00", "2024-03-01", ""])
    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode()),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=portfolio_template.csv"},
    )


@router.post("/portfolio/import")
async def import_csv(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    user: User  = Depends(get_current_user),
):
    content = await file.read()
    reader  = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))

    required = {"ticker", "shares", "buy_price", "buy_date"}
    imported = 0

    for row in reader:
        if not required.issubset(set(row.keys())):
            continue
        try:
            ticker    = row["ticker"].upper().strip()
            shares    = float(row["shares"])
            buy_price = float(row["buy_price"])
            buy_date  = datetime.strptime(row["buy_date"].strip(), "%Y-%m-%d")
            name      = row.get("name", "").strip()
            notes     = row.get("notes", "").strip()

            stock = db.query(Stock).filter(Stock.ticker == ticker).first()
            stop_loss = None
            if stock:
                ar = (
                    db.query(AnalysisResult)
                    .filter(AnalysisResult.stock_id == stock.id)
                    .order_by(AnalysisResult.date.desc())
                    .first()
                )
                if ar:
                    stop_loss = ar.stop_loss_price

            db.add(PortfolioPosition(
                user_id=user.id, ticker=ticker, name=name,
                shares=shares, buy_price=buy_price, buy_date=buy_date,
                stop_loss_price=stop_loss, notes=notes,
            ))
            imported += 1
        except (ValueError, KeyError):
            continue

    db.commit()
    return RedirectResponse(f"/portfolio?imported={imported}", status_code=302)
