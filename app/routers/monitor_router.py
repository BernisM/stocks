from __future__ import annotations
import json
import os
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import EMAIL_USER
from ..database import get_db
from ..models import User, UserEvent

_JOB_TIMINGS_PATH = "./ml_models/job_timings.json"


def _load_job_timings() -> list[dict]:
    try:
        if os.path.exists(_JOB_TIMINGS_PATH):
            with open(_JOB_TIMINGS_PATH) as f:
                return list(reversed(json.load(f)))
    except Exception:
        pass
    return []


def _fmt_duration(s: int) -> str:
    if s < 60:
        return f"{s}s"
    m, sec = divmod(s, 60)
    return f"{m}m{sec:02d}s" if sec else f"{m}m"

router    = APIRouter()
templates = Jinja2Templates(directory="templates")

_EVENT_LABELS = {
    "login_ok":   ("✅ Connexion", "success"),
    "login_fail": ("❌ Échec login", "danger"),
    "register":   ("🆕 Inscription", "primary"),
    "page":       ("📄 Page", "secondary"),
    "action":     ("⚡ Action", "warning"),
}


def _require_owner(user: User = Depends(get_current_user)):
    if user.email != EMAIL_USER:
        from fastapi import HTTPException
        raise HTTPException(status_code=403, detail="Accès réservé à l'administrateur")
    return user


@router.get("/admin/monitor", response_class=HTMLResponse)
def monitor_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(_require_owner),
):
    # Cleanup events older than 30 days
    cutoff = datetime.utcnow() - timedelta(days=30)
    db.query(UserEvent).filter(UserEvent.created_at < cutoff).delete()
    db.commit()

    today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)

    # All users
    users = db.query(User).order_by(User.created_at.desc()).all()

    # Last login per user
    last_logins: dict[str, datetime | None] = {}
    for u in users:
        ev = (
            db.query(UserEvent)
            .filter(UserEvent.user_email == u.email, UserEvent.event_type == "login_ok")
            .order_by(UserEvent.created_at.desc())
            .limit(1)
            .first()
        )
        last_logins[u.email] = ev.created_at if ev else None

    # Login counts per user (all time)
    login_counts: dict[str, int] = {}
    rows = (
        db.query(UserEvent.user_email, func.count(UserEvent.id))
        .filter(UserEvent.event_type == "login_ok", UserEvent.user_email.isnot(None))
        .group_by(UserEvent.user_email)
        .all()
    )
    for email, cnt in rows:
        login_counts[email] = cnt

    # Stats today
    def _count_today(etype: str) -> int:
        return db.query(UserEvent).filter(
            UserEvent.event_type == etype,
            UserEvent.created_at >= today_start,
        ).count()

    # Recent events (last 300)
    events = (
        db.query(UserEvent)
        .order_by(UserEvent.created_at.desc())
        .limit(300)
        .all()
    )

    job_timings = _load_job_timings()

    return templates.TemplateResponse(request, "monitor.html", {
        "user":          user,
        "users":         users,
        "last_logins":   last_logins,
        "login_counts":  login_counts,
        "events":        events,
        "event_labels":  _EVENT_LABELS,
        "logins_today":  _count_today("login_ok"),
        "fails_today":   _count_today("login_fail"),
        "registers_today": _count_today("register"),
        "visits_today":  _count_today("page"),
        "total_users":   len(users),
        "total_events":  db.query(UserEvent).count(),
        "job_timings":   job_timings,
        "fmt_duration":  _fmt_duration,
    })
