import logging
import threading
from datetime import UTC, datetime

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .auth import _decode_token, get_current_user
from .database import init_db
from .events import log_event
from .models import User
from .routers import analyse_router, auth_router, backtest_router, dashboard, guide_router, monitor_router, portfolio, recipients_router, stocks_router
from .scheduler import job_update_and_score, start_scheduler

# ── Verrou global : une seule opération RAM-lourde à la fois ──────────────────
_HEAVY_LOCK = threading.Lock()

_JOB_TIMES: dict = {
    "sync_prices": None,
    "train_ml":    None,
    "fondamentaux": None,
}

_CHAIN_STATE: dict = {
    "running":  False,
    "session":  "",        # "morning" | "afternoon"
    "step":     "",        # "run_now" | "sync_fast" | "train_ml" | "send_email" | "done"
    "started":  None,
    "finished": None,
    "error":    None,
}

_SMART_STATE: dict = {
    "running":    False,
    "jobs_done":  [],
    "email_sent": False,
    "error":      None,
}

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
app.include_router(stocks_router.router)
app.include_router(analyse_router.router)
app.include_router(monitor_router.router)


_LOG_GET_PATHS  = frozenset({"/dashboard", "/portfolio", "/analyse", "/backtest", "/guide"})
_LOG_POST_PATHS = frozenset({"/api/analyse/run", "/sync-prices", "/sync-tickers"})


@app.middleware("http")
async def _request_logger(request: Request, call_next):
    response = await call_next(request)
    path   = request.url.path
    method = request.method
    if (method == "GET" and path in _LOG_GET_PATHS) or \
       (method == "POST" and path in _LOG_POST_PATHS):
        token = request.cookies.get("access_token")
        ip    = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
                or (request.client.host if request.client else "")
        def _write(tok=token, p=path, remote_ip=ip):
            from .database import SessionLocal
            from .models import User, UserEvent
            db = SessionLocal()
            try:
                uid   = _decode_token(tok) if tok else None
                email = None
                if uid:
                    u = db.query(User).filter(User.id == uid).first()
                    email = u.email if u else None
                db.add(UserEvent(user_email=email, event_type="page" if method == "GET" else "action",
                                 detail=p, ip=remote_ip))
                db.commit()
            except Exception:
                pass
            finally:
                db.close()
        threading.Thread(target=_write, daemon=True).start()
    return response


@app.on_event("startup")
def on_startup():
    init_db()
    start_scheduler()


@app.get("/")
def root():
    return RedirectResponse("/login")


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


_ticker_sync_state: dict = {
    "running": False,
    "done":    False,
    "count":   0,
    "error":   None,
}


@app.post("/sync-tickers")
async def sync_tickers(request: Request, user: User = Depends(get_current_user)):
    body    = await request.json()
    tickers = [t.strip().upper() for t in body.get("tickers", []) if t.strip()]
    if not tickers:
        return JSONResponse({"status": "error", "message": "Aucun ticker fourni"}, status_code=400)

    def _run():
        if not _HEAVY_LOCK.acquire(blocking=False):
            _ticker_sync_state.update({"running": False, "done": False, "error": "Une opération lourde est déjà en cours, réessayez dans quelques minutes."})
            return
        from .data_engine import sync_selected_tickers
        from .database import SessionLocal
        _ticker_sync_state.update({"running": True, "done": False, "count": len(tickers), "error": None})
        db = SessionLocal()
        try:
            sync_selected_tickers(db, tickers)
            _ticker_sync_state["done"] = True
        except Exception as e:
            _ticker_sync_state["error"] = str(e)
        finally:
            _ticker_sync_state["running"] = False
            db.close()
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "tickers": tickers, "count": len(tickers)})


@app.get("/ticker-sync-status")
def ticker_sync_status(user: User = Depends(get_current_user)):
    return JSONResponse(_ticker_sync_state)


@app.post("/sync-prices")
def sync_prices(user: User = Depends(get_current_user)):
    if _sync_state["running"]:
        return JSONResponse({"status": "already_running"})
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "already_running", "message": "Une opération lourde est déjà en cours."})

    def _run():
        from .data_engine import sync_prices_fast
        from .database import SessionLocal
        db = SessionLocal()
        _sync_state.update({
            "running": True, "phase": "prices",
            "progress": 0, "total": 0,
            "started": datetime.now(UTC).replace(tzinfo=None).isoformat(), "error": None,
        })
        try:
            def on_progress(done, total, phase):
                _sync_state["phase"]    = phase
                _sync_state["progress"] = done
                _sync_state["total"]    = total

            sync_prices_fast(db, on_progress=on_progress)
            _sync_state["phase"]    = "done"
            _sync_state["finished"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
            _JOB_TIMES["sync_prices"] = _sync_state["finished"]
        except Exception as e:
            _sync_state["error"] = str(e)
        finally:
            _sync_state["running"] = False
            db.close()
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.get("/job-status")
def job_status(user: User = Depends(get_current_user)):
    return JSONResponse(_JOB_TIMES)


@app.get("/smart-email-status")
def smart_email_status(user: User = Depends(get_current_user)):
    return JSONResponse(_SMART_STATE)


@app.post("/send-email-smart")
async def send_email_smart(request: Request, user: User = Depends(get_current_user)):
    body = await request.json()
    jobs = body.get("jobs", [])

    ETA = {"sync_prices": 8, "train_ml": 5, "fondamentaux": 6}
    eta_min = sum(ETA.get(j, 0) for j in jobs) + 1

    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})

    _SMART_STATE.update({"running": True, "jobs_done": [], "email_sent": False, "error": None})

    def _run():
        from .database import SessionLocal
        try:
            if "sync_prices" in jobs:
                from .data_engine import sync_prices_fast
                db = SessionLocal()
                try:
                    sync_prices_fast(db)
                    _JOB_TIMES["sync_prices"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
                    _SMART_STATE["jobs_done"].append("sync_prices")
                finally:
                    db.close()

            if "train_ml" in jobs:
                from .scheduler import job_retrain_ml
                job_retrain_ml()
                _JOB_TIMES["train_ml"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
                _SMART_STATE["jobs_done"].append("train_ml")

            if "fondamentaux" in jobs:
                from .fundamentals import update_fundamentals
                db = SessionLocal()
                try:
                    update_fundamentals(db)
                    _JOB_TIMES["fondamentaux"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
                    _SMART_STATE["jobs_done"].append("fondamentaux")
                finally:
                    db.close()

            # Send email
            from .data_engine import load_market_status
            from .email_sender import send_combined_report
            from .ml_model import load_metrics
            from .models import AnalysisResult, ExtraRecipient, Stock
            db = SessionLocal()
            try:
                last_date = db.query(AnalysisResult.date).order_by(AnalysisResult.date.desc()).limit(1).scalar()
                if last_date:
                    def get_top(market):
                        rows = (
                            db.query(AnalysisResult, Stock)
                            .join(Stock, AnalysisResult.stock_id == Stock.id)
                            .filter(AnalysisResult.date == last_date, Stock.market == market,
                                    AnalysisResult.ranking.in_(["Strong Buy", "Buy"]))
                            .order_by(AnalysisResult.score_final.desc()).limit(10).all()
                        )
                        return [{"ticker": s.ticker, "name": s.name or "", "close": ar.close or 0,
                                 "score_final": ar.score_final or 0, "ranking": ar.ranking or "Neutral",
                                 "stop_loss": ar.stop_loss_price or 0, "rsi": ar.rsi or 0,
                                 "macd_hist": ar.macd_hist or 0, "volatility": ar.volatility or 0,
                                 "ml_probability": ar.ml_probability,
                                 "atr_pct": (ar.atr / ar.close * 100) if ar.atr and ar.close else 0,
                                 "bollinger_b": ar.bollinger_b or 0} for ar, s in rows]
                    extras = db.query(ExtraRecipient).filter(ExtraRecipient.is_active.is_(True)).all()
                    recipients = [(user.email, user.level)] + [(e.email, e.level) for e in extras]
                    send_combined_report(recipients=recipients, top_cac40=get_top("CAC40"),
                                         top_sbf120=get_top("SBF120"), top_sp500=get_top("SP500"),
                                         analysis_date=last_date, ml_metrics=load_metrics(),
                                         top_commodities=get_top("COMMODITIES"),
                                         top_crypto=get_top("CRYPTO"),
                                         market_status=load_market_status())
                    _SMART_STATE["email_sent"] = True
            finally:
                db.close()
        except Exception as e:
            _SMART_STATE["error"] = str(e)
        finally:
            _SMART_STATE["running"] = False
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "eta_min": eta_min, "jobs": jobs})


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
        from .data_engine import load_market_status
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

            extras = db.query(ExtraRecipient).filter(ExtraRecipient.is_active.is_(True)).all()
            recipients = [(user.email, user.level)] + [(e.email, e.level) for e in extras]
            send_combined_report(
                recipients=recipients,
                top_cac40=get_top("CAC40"),
                top_sbf120=get_top("SBF120"),
                top_sp500=get_top("SP500"),
                analysis_date=last_date,
                ml_metrics=load_metrics(),
                top_commodities=get_top("COMMODITIES"),
                top_crypto=get_top("CRYPTO"),
                market_status=load_market_status(),
            )
        finally:
            db.close()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": f"Email envoyé à {user.email}"})


# ── Admin endpoints ────────────────────────────────────────────────────────────

_backtest_state: dict = {
    "running":  False,
    "started":  None,
    "finished": None,
    "error":    None,
}


@app.post("/admin/backtest-run")
def backtest_run():
    if _backtest_state["running"]:
        return JSONResponse({"status": "already_running"})
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})

    def _run():
        import json
        from .backtest import run_backtest, stats_to_dict
        from .database import SessionLocal
        _backtest_state.update({
            "running": True,
            "started": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "finished": None, "error": None,
        })
        db = SessionLocal()
        try:
            results = run_backtest(db)
            cache = {k: stats_to_dict(v) for k, v in results.items()}
            with open("./ml_models/backtest_cache.json", "w") as f:
                json.dump(cache, f)
            _backtest_state["finished"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
        except Exception as e:
            _backtest_state["error"] = str(e)
        finally:
            _backtest_state["running"] = False
            db.close()
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started"})


@app.get("/backtest-status")
def backtest_status(user: User = Depends(get_current_user)):
    return JSONResponse(_backtest_state)


@app.get("/admin/fundamentals-now")
def fundamentals_now():
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})
    def _run():
        from .fundamentals import update_fundamentals
        from .database import SessionLocal
        db = SessionLocal()
        try:
            update_fundamentals(db)
            _JOB_TIMES["fondamentaux"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
        finally:
            db.close()
            _HEAVY_LOCK.release()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Fondamentaux en cours (~6 min). Revenez dans quelques minutes."})


@app.get("/admin/train-ml")
def train_ml():
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})
    def _run():
        try:
            from .scheduler import job_retrain_ml
            job_retrain_ml()
            _JOB_TIMES["train_ml"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
        finally:
            _HEAVY_LOCK.release()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Entraînement ML lancé (~5 min)."})


@app.get("/admin/send-email")
def send_email_now():
    def _run():
        from .scheduler import job_email_daily
        job_email_daily()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Email matin en cours d'envoi."})


@app.get("/admin/send-email-afternoon")
def send_email_afternoon():
    def _run():
        from .scheduler import job_email_afternoon
        job_email_afternoon()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Email après-midi en cours d'envoi."})


@app.get("/admin/sync-fast")
def sync_fast():
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})
    def _run():
        from .data_engine import sync_prices_fast
        from .database import SessionLocal
        db = SessionLocal()
        try:
            sync_prices_fast(db)
        finally:
            db.close()
            _HEAVY_LOCK.release()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Sync rapide lancé (~8 min)."})


@app.get("/admin/run-now")
def run_now():
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})
    def _run():
        try:
            job_update_and_score()
        finally:
            _HEAVY_LOCK.release()
    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Téléchargement complet lancé (~30-60 min)."})


# ── Chaînes séquentielles (1 seul appel cron-job.org) ─────────────────────────

@app.get("/admin/morning-chain")
def morning_chain():
    """Enchaîne train-ml → run-now → send-email dans un seul thread.
    train-ml en premier : le scoring utilise le nouveau modèle → ML prob correctes."""
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})

    def _run():
        from .scheduler import job_email_daily, job_retrain_ml
        _CHAIN_STATE.update({
            "running": True, "session": "morning", "step": "train_ml",
            "started": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "finished": None, "error": None,
        })
        try:
            logging.getLogger(__name__).info("[morning-chain] Étape 1 : train-ml")
            job_retrain_ml()
            _JOB_TIMES["train_ml"] = datetime.now(UTC).replace(tzinfo=None).isoformat()

            _CHAIN_STATE["step"] = "run_now"
            logging.getLogger(__name__).info("[morning-chain] Étape 2 : run-now")
            job_update_and_score()
            _JOB_TIMES["sync_prices"] = datetime.now(UTC).replace(tzinfo=None).isoformat()

            _CHAIN_STATE["step"] = "send_email"
            logging.getLogger(__name__).info("[morning-chain] Étape 3 : send-email")
            job_email_daily()

            _CHAIN_STATE["step"] = "done"
            _CHAIN_STATE["finished"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
            logging.getLogger(__name__).info("[morning-chain] Terminé ✅")
        except Exception as e:
            _CHAIN_STATE["error"] = str(e)
            logging.getLogger(__name__).error(f"[morning-chain] Erreur : {e}")
        finally:
            _CHAIN_STATE["running"] = False
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Chaîne matin lancée : train-ml → run-now → send-email."})


@app.get("/admin/afternoon-chain")
def afternoon_chain():
    """Enchaîne train-ml → sync-fast → send-email-afternoon dans un seul thread."""
    if not _HEAVY_LOCK.acquire(blocking=False):
        return JSONResponse({"status": "busy", "message": "Une opération lourde est déjà en cours."})

    def _run():
        from .data_engine import sync_prices_fast
        from .database import SessionLocal
        from .scheduler import job_email_afternoon, job_retrain_ml
        _CHAIN_STATE.update({
            "running": True, "session": "afternoon", "step": "train_ml",
            "started": datetime.now(UTC).replace(tzinfo=None).isoformat(),
            "finished": None, "error": None,
        })
        try:
            logging.getLogger(__name__).info("[afternoon-chain] Étape 1 : train-ml")
            job_retrain_ml()
            _JOB_TIMES["train_ml"] = datetime.now(UTC).replace(tzinfo=None).isoformat()

            _CHAIN_STATE["step"] = "sync_fast"
            logging.getLogger(__name__).info("[afternoon-chain] Étape 2 : sync-fast")
            db = SessionLocal()
            try:
                sync_prices_fast(db)
            finally:
                db.close()
            _JOB_TIMES["sync_prices"] = datetime.now(UTC).replace(tzinfo=None).isoformat()

            _CHAIN_STATE["step"] = "send_email"
            logging.getLogger(__name__).info("[afternoon-chain] Étape 3 : send-email-afternoon")
            job_email_afternoon()

            _CHAIN_STATE["step"] = "done"
            _CHAIN_STATE["finished"] = datetime.now(UTC).replace(tzinfo=None).isoformat()
            logging.getLogger(__name__).info("[afternoon-chain] Terminé ✅")
        except Exception as e:
            _CHAIN_STATE["error"] = str(e)
            logging.getLogger(__name__).error(f"[afternoon-chain] Erreur : {e}")
        finally:
            _CHAIN_STATE["running"] = False
            _HEAVY_LOCK.release()

    threading.Thread(target=_run, daemon=True).start()
    return JSONResponse({"status": "started", "message": "Chaîne après-midi lancée : train-ml → sync-fast → send-email-afternoon."})


@app.get("/admin/chain-status")
def chain_status():
    return JSONResponse(_CHAIN_STATE)


# ── Blacklist tickers ──────────────────────────────────────────────────────────

@app.get("/admin/blacklisted-tickers")
def admin_blacklisted_tickers():
    from .data_engine import get_blacklisted
    return JSONResponse(get_blacklisted())


@app.post("/admin/unblacklist/{ticker:path}")
def admin_unblacklist(ticker: str):
    from .data_engine import unblacklist
    unblacklist(ticker)
    return JSONResponse({"status": "ok", "ticker": ticker})
