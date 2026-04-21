import logging
import threading
from datetime import datetime

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .auth import get_current_user
from .database import init_db
from .models import User
from .routers import auth_router, backtest_router, dashboard, guide_router, portfolio, recipients_router
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
app.include_router(guide_router.router)
app.include_router(recipients_router.router)


@app.on_event("startup")
def on_startup():
    init_db()
    start_scheduler()


@app.get("/")
def root():
    return RedirectResponse("/dashboard")


@app.get("/ping")
def ping():
    return {"status": "ok"}


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


# ── Envoi email à l'utilisateur connecté ──────────────────────────────────────

@app.get("/send-email-me")
def send_email_me(user: User = Depends(get_current_user)):
    def _run():
        from .database import SessionLocal
        from .email_sender import send_combined_report
        from .ml_model import load_metrics
        from .models import AnalysisResult, ExtraRecipient, Stock
        db = SessionLocal()
        try:
            last_date = (
                db.query(AnalysisResult.date)
                .order_by(AnalysisResult.date.desc())
                .limit(1)
                .scalar()
            )
            if not last_date:
                return

            def get_top(market):
                rows = (
                    db.query(AnalysisResult, Stock)
                    .join(Stock, AnalysisResult.stock_id == Stock.id)
                    .filter(
                        AnalysisResult.date == last_date,
                        Stock.market == market,
                        AnalysisResult.ranking.in_(["Strong Buy", "Buy"]),
                    )
                    .order_by(AnalysisResult.score_final.desc())
                    .limit(10)
                    .all()
                )
                return [{
                    "ticker":         s.ticker,
                    "name":           s.name or "",
                    "close":          ar.close or 0,
                    "score_final":    ar.score_final or 0,
                    "ranking":        ar.ranking or "Neutral",
                    "stop_loss":      ar.stop_loss_price or 0,
                    "rsi":            ar.rsi or 0,
                    "macd_hist":      ar.macd_hist or 0,
                    "volatility":     ar.volatility or 0,
                    "ml_probability": ar.ml_probability,
                    "atr_pct":        (ar.atr / ar.close * 100) if ar.atr and ar.close else 0,
                    "bollinger_b":    ar.bollinger_b or 0,
                } for ar, s in rows]

            extras = db.query(ExtraRecipient).all()
            recipients = [(user.email, user.level)] + [(e.email, e.level) for e in extras]
            send_combined_report(
                recipients=recipients,
                top_cac40=get_top("CAC40"),
                top_sbf120=get_top("SBF120"),
                top_sp500=get_top("SP500"),
                analysis_date=last_date,
                ml_metrics=load_metrics(),
            )
        finally:
            db.close()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": f"Email envoyé à {user.email}"})


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
