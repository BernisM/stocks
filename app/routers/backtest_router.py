from __future__ import annotations
import json
import os

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import User

router    = APIRouter()
templates = Jinja2Templates(directory="templates")

CACHE_PATH = "./ml_models/backtest_cache.json"


def _load_cache() -> dict | None:
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return None


@router.get("/backtest", response_class=HTMLResponse)
def backtest_page(
    request: Request,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    cache = _load_cache()
    return templates.TemplateResponse(request, "backtest.html", {
        "user":    user,
        "results": cache,
        "ready":   cache is not None,
    })
