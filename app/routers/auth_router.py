from __future__ import annotations
from fastapi import APIRouter, Depends, Form, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import create_access_token, get_current_user_optional, hash_password, verify_password
from ..database import get_db
from ..events import log_event_sync
from ..models import User

router    = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/login", response_class=HTMLResponse)
def login_page(request: Request, user=Depends(get_current_user_optional)):
    if user:
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@router.post("/login")
def login(
    request: Request,
    response: Response,
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    ip    = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or (request.client.host if request.client else "")
    email = email.lower().strip()
    user  = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.password_hash):
        log_event_sync(email or None, "login_fail", detail=email, ip=ip)
        return templates.TemplateResponse(
            request, "login.html", {"error": "Email ou mot de passe incorrect"},
            status_code=400,
        )
    log_event_sync(user.email, "login_ok", ip=ip)
    token = create_access_token(user.id)
    resp  = RedirectResponse("/dashboard", status_code=302)
    resp.set_cookie("access_token", token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
    return resp


@router.get("/register", response_class=HTMLResponse)
def register_page(request: Request, user=Depends(get_current_user_optional)):
    if user:
        return RedirectResponse("/dashboard", status_code=302)
    return templates.TemplateResponse(request, "register.html", {"error": None})


@router.post("/register")
def register(
    request: Request,
    email: str = Form(...),
    password: str = Form(...),
    level: str = Form("intermediate"),
    db: Session = Depends(get_db),
):
    email = email.lower().strip()
    if db.query(User).filter(User.email == email).first():
        return templates.TemplateResponse(
            request, "register.html",
            {"error": "Cet email est déjà utilisé"},
            status_code=400,
        )
    if len(password) < 8:
        return templates.TemplateResponse(
            request, "register.html",
            {"error": "Le mot de passe doit faire au moins 8 caractères"},
            status_code=400,
        )
    if level not in ("beginner", "intermediate", "expert"):
        level = "intermediate"

    ip   = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() or (request.client.host if request.client else "")
    user = User(email=email, password_hash=hash_password(password), level=level)
    db.add(user)
    db.commit()
    log_event_sync(email, "register", detail=level, ip=ip)

    token = create_access_token(user.id)
    resp  = RedirectResponse("/dashboard", status_code=302)
    resp.set_cookie("access_token", token, httponly=True, samesite="lax", max_age=60 * 60 * 24 * 7)
    return resp


@router.get("/logout")
def logout():
    resp = RedirectResponse("/login", status_code=302)
    resp.delete_cookie("access_token")
    return resp
