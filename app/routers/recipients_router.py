from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import EMAIL_USER
from ..database import get_db
from ..models import ExtraRecipient, User


def _require_owner(user: User = Depends(get_current_user)):
    if user.email != EMAIL_USER:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Accès réservé à l'administrateur")
    return user

router    = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/recipients", response_class=HTMLResponse)
def recipients_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(_require_owner),
):
    recipients = db.query(ExtraRecipient).order_by(ExtraRecipient.created_at.desc()).all()
    return templates.TemplateResponse("recipients.html", {
        "request":    request,
        "user":       user,
        "recipients": recipients,
    })


@router.post("/recipients/add")
def add_recipient(
    email: str = Form(...),
    name:  str = Form(""),
    level: str = Form("beginner"),
    db: Session = Depends(get_db),
    user: User = Depends(_require_owner),
):
    existing = db.query(ExtraRecipient).filter(ExtraRecipient.email == email).first()
    if not existing:
        db.add(ExtraRecipient(email=email, name=name, level=level))
        db.commit()
    return RedirectResponse("/recipients", status_code=303)


@router.post("/recipients/delete/{rid}")
def delete_recipient(
    rid: int,
    db: Session = Depends(get_db),
    user: User = Depends(_require_owner),
):
    r = db.query(ExtraRecipient).filter(ExtraRecipient.id == rid).first()
    if r:
        db.delete(r)
        db.commit()
    return RedirectResponse("/recipients", status_code=303)
