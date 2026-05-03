from __future__ import annotations
import logging

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse

from ..auth import get_current_user
from ..models import User
from ..news import fetch_global_news, fetch_ticker_news

router = APIRouter()
log    = logging.getLogger(__name__)


@router.get("/api/news/global")
def global_news(user: User = Depends(get_current_user)):
    return JSONResponse({"news": fetch_global_news()})


@router.get("/api/news/{ticker:path}")
def ticker_news(ticker: str, user: User = Depends(get_current_user)):
    news = fetch_ticker_news(ticker.upper())
    return JSONResponse({"ticker": ticker.upper(), "news": news})
