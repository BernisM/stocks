import logging
import threading

from fastapi import FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .database import init_db
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


@app.post("/admin/backtest-run")
def backtest_run():
    import json, threading
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
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/backtest", status_code=302)


@app.get("/admin/fundamentals-now")
def fundamentals_now():
    """Lance manuellement la mise à jour des fondamentaux en arrière-plan."""
    def _run():
        from .fundamentals import update_fundamentals
        from .database import SessionLocal
        db = SessionLocal()
        try:
            update_fundamentals(db)
        finally:
            db.close()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Fondamentaux en cours (~6 min pour 667 stocks). Revenez dans quelques minutes."})


@app.get("/admin/run-now")
def run_now():
    """Lance manuellement le téléchargement et le scoring en arrière-plan."""
    thread = threading.Thread(target=job_update_and_score, daemon=True)
    thread.start()
    return JSONResponse({"status": "started", "message": "Téléchargement lancé en arrière-plan. Revenez dans 30-60 min selon le nombre d'actions."})
