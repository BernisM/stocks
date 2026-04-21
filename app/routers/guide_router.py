from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from ..auth import get_current_user
from ..models import User

router    = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/guide", response_class=HTMLResponse)
def guide_page(
    request: Request,
    user: User = Depends(get_current_user),
):
    return templates.TemplateResponse(request, "guide.html", {
        "user": user,
    })
