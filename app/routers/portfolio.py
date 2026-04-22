from __future__ import annotations
import csv
import io
import logging
from datetime import UTC, datetime
from typing import Optional

from fastapi import APIRouter, Depends, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..data_engine import get_current_price
from ..models import AnalysisResult, Dividend, PortfolioPosition, Stock, User

logger = logging.getLogger(__name__)

router    = APIRouter()
templates = Jinja2Templates(directory="templates")


def _compute_pnl(pos: PortfolioPosition, current_price: float | None) -> dict:
    cp   = current_price or pos.buy_price
    fees = pos.fees or 0.0
    pnl_abs = (cp - pos.buy_price) * pos.shares - fees
    pnl_pct = (pnl_abs / (pos.buy_price * pos.shares)) * 100 if pos.buy_price and pos.shares else 0
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
        "fees":          round(fees, 2),
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

    total_fees    = round(sum(r["fees"] for r in rows), 2)
    total_pnl_abs = round(total_value - total_cost - total_fees, 2)
    total_pnl_pct = round((total_pnl_abs / total_cost) * 100, 2) if total_cost else 0

    dividends = (
        db.query(Dividend)
        .filter(Dividend.user_id == user.id)
        .order_by(Dividend.date.desc())
        .all()
    )
    total_dividends = round(sum(d.amount for d in dividends), 2)

    today = datetime.now(UTC).replace(tzinfo=None).strftime("%Y-%m-%d")
    return templates.TemplateResponse(request, "portfolio.html", {
        "user":             user,
        "positions":        rows,
        "total_value":      round(total_value, 2),
        "total_fees":       total_fees,
        "total_pnl_abs":    total_pnl_abs,
        "total_pnl_pct":    total_pnl_pct,
        "dividends":        dividends,
        "total_dividends":  total_dividends,
        "today":            today,
        "error":            None,
    })


def _parse_float(val, default: float = 0.0) -> float:
    try:
        return float(str(val).strip()) if val not in (None, "") else default
    except (ValueError, TypeError):
        return default


def _parse_date(val: str) -> datetime:
    try:
        return datetime.strptime(val.strip(), "%Y-%m-%d")
    except (ValueError, AttributeError):
        return datetime.now(UTC).replace(tzinfo=None)


def _get_stop_loss(db: Session, ticker: str) -> float | None:
    stock = db.query(Stock).filter(Stock.ticker == ticker).first()
    if stock:
        ar = (
            db.query(AnalysisResult)
            .filter(AnalysisResult.stock_id == stock.id)
            .order_by(AnalysisResult.date.desc())
            .first()
        )
        if ar and ar.stop_loss_price:
            return ar.stop_loss_price
    return None


@router.post("/portfolio/add")
def add_position(
    ticker:    str            = Form(...),
    name:      str            = Form(""),
    shares:    float          = Form(...),
    buy_price: float          = Form(...),
    buy_date:  str            = Form(...),
    fees:      Optional[str]  = Form("0"),
    notes:     str            = Form(""),
    db: Session = Depends(get_db),
    user: User  = Depends(get_current_user),
):
    try:
        pos = PortfolioPosition(
            user_id         = user.id,
            ticker          = ticker.upper().strip(),
            name            = name.strip(),
            shares          = shares,
            buy_price       = buy_price,
            buy_date        = _parse_date(buy_date),
            fees            = _parse_float(fees),
            stop_loss_price = _get_stop_loss(db, ticker.upper().strip()),
            notes           = notes.strip(),
        )
        db.add(pos)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"[portfolio/add] {e}")
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
    writer.writerow(["ticker", "name", "shares", "buy_price", "buy_date", "fees", "notes"])
    writer.writerow(["AAPL", "Apple Inc.", "10", "150.00", "2024-01-15", "9.99", "Exemple"])
    writer.writerow(["MC.PA", "LVMH", "2", "750.00", "2024-03-01", "0", ""])
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

    headers = reader.fieldnames or []
    if not required.issubset(set(h.strip().lower() for h in headers)):
        logger.warning(f"[portfolio/import] colonnes manquantes: {headers}")
        return RedirectResponse("/portfolio?error=colonnes_manquantes", status_code=302)

    errors = 0
    for row in reader:
        try:
            ticker    = row["ticker"].upper().strip()
            shares    = _parse_float(row["shares"])
            buy_price = _parse_float(row["buy_price"])
            buy_date  = _parse_date(row.get("buy_date", ""))
            name      = row.get("name", "").strip()
            notes     = row.get("notes", "").strip()
            fees      = _parse_float(row.get("fees", "0"))

            if not ticker or shares <= 0 or buy_price <= 0:
                continue

            db.add(PortfolioPosition(
                user_id         = user.id,
                ticker          = ticker,
                name            = name,
                shares          = shares,
                buy_price       = buy_price,
                buy_date        = buy_date,
                fees            = fees,
                stop_loss_price = _get_stop_loss(db, ticker),
                notes           = notes,
            ))
            db.flush()   # catch DB errors per row
            imported += 1
        except Exception as e:
            db.rollback()
            errors += 1
            logger.warning(f"[portfolio/import] ligne ignorée: {e} — {dict(row)}")

    try:
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"[portfolio/import] commit failed: {e}")
    return RedirectResponse(f"/portfolio?imported={imported}&errors={errors}", status_code=302)


# ── Dividendes ────────────────────────────────────────────────────────────────

@router.post("/portfolio/dividends/add")
def add_dividend(
    ticker: str   = Form(...),
    name:   str   = Form(""),
    amount: float = Form(...),
    date:   str   = Form(...),
    notes:  str   = Form(""),
    db: Session = Depends(get_db),
    user: User  = Depends(get_current_user),
):
    try:
        d = datetime.strptime(date, "%Y-%m-%d")
    except ValueError:
        d = datetime.now(UTC).replace(tzinfo=None)
    db.add(Dividend(
        user_id=user.id, ticker=ticker.upper().strip(),
        name=name.strip(), amount=amount, date=d, notes=notes.strip(),
    ))
    db.commit()
    return RedirectResponse("/portfolio#dividends", status_code=302)


@router.post("/portfolio/dividends/delete/{div_id}")
def delete_dividend(
    div_id: int,
    db: Session = Depends(get_db),
    user: User  = Depends(get_current_user),
):
    div = db.query(Dividend).filter(Dividend.id == div_id, Dividend.user_id == user.id).first()
    if div:
        db.delete(div)
        db.commit()
    return RedirectResponse("/portfolio#dividends", status_code=302)
