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
    added:   int = 0,
    skipped: int = 0,
    db: Session = Depends(get_db),
    user: User = Depends(_require_owner),
):
    recipients = db.query(ExtraRecipient).order_by(ExtraRecipient.created_at.desc()).all()
    return templates.TemplateResponse(request, "recipients.html", {
        "user":       user,
        "recipients": recipients,
        "added":      added,
        "skipped":    skipped,
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


@router.post("/recipients/bulk-add")
async def bulk_add_recipients(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(_require_owner),
):
    form  = await request.form()
    lines = (form.get("lines") or "").strip().splitlines()
    added = skipped = 0
    VALID_LEVELS = {"beginner", "intermediate", "expert"}
    for raw in lines:
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        parts = [p.strip() for p in raw.split(",")]
        email = parts[0].lower()
        if not email or "@" not in email:
            continue
        name  = parts[1] if len(parts) > 1 else ""
        level = parts[2].lower() if len(parts) > 2 and parts[2].lower() in VALID_LEVELS else "intermediate"
        if db.query(ExtraRecipient).filter(ExtraRecipient.email == email).first():
            skipped += 1
        else:
            db.add(ExtraRecipient(email=email, name=name, level=level))
            added += 1
    if added:
        db.commit()
    return RedirectResponse(f"/recipients?added={added}&skipped={skipped}", status_code=303)


@router.post("/recipients/toggle/{rid}")
def toggle_recipient(
    rid: int,
    db: Session = Depends(get_db),
    user: User = Depends(_require_owner),
):
    r = db.query(ExtraRecipient).filter(ExtraRecipient.id == rid).first()
    if r:
        r.is_active = not r.is_active
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
