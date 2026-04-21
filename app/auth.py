"""
Authentification JWT stockée dans un cookie HTTP-only.
"""
from __future__ import annotations
from datetime import datetime, timedelta

import hashlib
import secrets

from fastapi import Cookie, Depends, HTTPException, Request, status
from fastapi.responses import RedirectResponse
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from .config import ACCESS_TOKEN_EXPIRE_MINUTES, ALGORITHM, SECRET_KEY
from .database import get_db
from .models import User

_ITERATIONS = 600_000


def hash_password(password: str) -> str:
    salt = secrets.token_hex(32)
    key  = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS)
    return f"{salt}:{key.hex()}"


def verify_password(plain: str, hashed: str) -> bool:
    try:
        salt, key = hashed.split(":")
        new_key = hashlib.pbkdf2_hmac("sha256", plain.encode(), salt.encode(), _ITERATIONS)
        return secrets.compare_digest(new_key.hex(), key)
    except Exception:
        return False


def create_access_token(user_id: int) -> str:
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode({"sub": str(user_id), "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def _decode_token(token: str) -> int | None:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    access_token: str | None = Cookie(default=None),
) -> User:
    token = access_token or request.cookies.get("access_token")
    if not token:
        return RedirectResponse("/login", status_code=302)
    user_id = _decode_token(token)
    if user_id is None:
        return RedirectResponse("/login", status_code=302)
    user = db.query(User).filter(User.id == user_id, User.is_active == True).first()
    if not user:
        return RedirectResponse("/login", status_code=302)
    return user


def get_current_user_optional(
    request: Request,
    db: Session = Depends(get_db),
) -> User | None:
    token = request.cookies.get("access_token")
    if not token:
        return None
    user_id = _decode_token(token)
    if user_id is None:
        return None
    return db.query(User).filter(User.id == user_id, User.is_active == True).first()
