import logging
import threading
from datetime import datetime

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .auth import get_current_user
from .database import init_db
from .models import User
from .routers import auth_router, backtest_router, dashboard, portfolio
from .scheduler import job_update_and_score, start_scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

app = FastAPI(title="StockAnalyzer", docs_url=None, redoc_url=None)
app.mount("/static", StaticFiles(directory="static"), name="static")

app.include_router(auth_router.router)
app.include_router(dashboard.router)
app.include_router(portfolio.router)
app.include_router(backtest_router.router)


@app.on_event("startup")
def on_startup():
    init_db()
    start_scheduler()


@app.get("/")
def root():
    return RedirectResponse("/dashboard")


# ── Sync prix en temps réel ────────────────────────────────────────────────────

_sync_state: dict = {
    "running":  False,
    "phase":    "",      # "prices" | "scores" | "done"
    "progress": 0,
    "total":    0,
    "started":  None,
    "finished": None,
    "error":    None,
}


@app.post("/sync-prices")
def sync_prices(user: User = Depends(get_current_user)):
    if _sync_state["running"]:
        return JSONResponse({"status": "already_running"})

    def _run():
        from .data_engine import sync_prices_fast
        from .database import SessionLocal
        db = SessionLocal()
        _sync_state.update({
            "running": True, "phase": "prices",
            "progress": 0, "total": 0,
            "started": datetime.utcnow().isoformat(), "error": None,
        })
        try:
            def on_progress(done, total, phase):
                _sync_state["phase"]    = phase
                _sync_state["progress"] = done
                _sync_state["total"]    = total

            sync_prices_fast(db, on_progress=on_progress)
            _sync_state["phase"]    = "done"
            _sync_state["finished"] = datetime.utcnow().isoformat()
        except Exception as e:
            _sync_state["error"] = str(e)
        finally:
            _sync_state["running"] = False
            db.close()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.get("/sync-status")
def sync_status(user: User = Depends(get_current_user)):
    pct = 0
    if _sync_state["total"] > 0:
        pct = round(_sync_state["progress"] / _sync_state["total"] * 100)
    return JSONResponse({**_sync_state, "pct": pct})


# ── Admin endpoints ────────────────────────────────────────────────────────────

@app.post("/admin/backtest-run")
def backtest_run():
    import json
    def _run():
        from .backtest import run_backtest, stats_to_dict
        from .database import SessionLocal
        db = SessionLocal()
        try:
            results = run_backtest(db)
            cache = {k: stats_to_dict(v) for k, v in results.items()}
            with open("./ml_models/backtest_cache.json", "w") as f:
                json.dump(cache, f)
        finally:
            db.close()
    threading.Thread(target=_run, daemon=True).start()
    return RedirectResponse("/backtest", status_code=302)


@app.get("/admin/fundamentals-now")
def fundamentals_now():
    def _run():
        from .fundamentals import update_fundamentals
        from .database import SessionLocal
        db = SessionLocal()
        try:
            update_fundamentals(db)
        finally:
            db.close()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Fondamentaux en cours (~6 min). Revenez dans quelques minutes."})


@app.get("/admin/train-ml")
def train_ml():
    def _run():
        from .database import SessionLocal
        from .ml_model import train
        db = SessionLocal()
        try:
            train(db)
        finally:
            db.close()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Entraînement ML lancé (~5 min)."})


@app.get("/admin/send-email")
def send_email_now():
    def _run():
        from .scheduler import job_email_daily
        job_email_daily()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Email en cours d'envoi."})


@app.get("/admin/run-now")
def run_now():
    thread = threading.Thread(target=job_update_and_score, daemon=True)
    thread.start()
    return JSONResponse({"status": "started", "message": "Téléchargement complet lancé (~30-60 min)."})
